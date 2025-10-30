from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import faiss
import numpy as np
import json

# Load embedding model
model = SentenceTransformer('all-mpnet-base-v2')

# Load and validate FAQ data with error handling
try:
    with open('faq.json', 'r') as f:
        faq_data = json.load(f)
    
    # Debug: Print the type and structure of loaded data
    print(f"Type of loaded data: {type(faq_data)}")
    print(f"Categories found: {list(faq_data.keys()) if isinstance(faq_data, dict) else 'Not a dictionary'}")
    
    # Handle nested structure - flatten the categorized FAQs into a single list
    faqs = []
    if isinstance(faq_data, dict):
        for category, questions in faq_data.items():
            if isinstance(questions, list):
                for faq in questions:
                    # Add category to each FAQ item
                    faq_with_category = faq.copy()
                    faq_with_category['category'] = category
                    faqs.append(faq_with_category)
                    
        print(f"Successfully flattened {len(faqs)} FAQ items from {len(faq_data)} categories")
        
    elif isinstance(faq_data, list):
        # Handle flat list structure (original format)
        faqs = faq_data
        print(f"Successfully loaded {len(faqs)} FAQ items from flat list")
        
    else:
        print(f"ERROR: Unexpected data structure. Expected dict or list, got {type(faq_data)}")
        exit(1)
    
    if len(faqs) == 0:
        print("ERROR: No FAQ items found")
        exit(1)
    
    # Validate first item structure
    required_keys = ['question', 'answer']
    if not all(key in faqs[0] for key in required_keys):
        print(f"ERROR: FAQ items missing required keys. Expected: {required_keys}")
        print(f"Found keys in first item: {list(faqs[0].keys())}")
        exit(1)
    
    # print(f"Sample FAQ item: {faqs[0]}")
    
except FileNotFoundError:
    print("ERROR: faq.json file not found")
    exit(1)
except json.JSONDecodeError as e:
    print(f"ERROR: Invalid JSON format: {e}")
    exit(1)

# Generate embeddings
questions = [faq["question"] for faq in faqs]
embeddings = model.encode(questions, normalize_embeddings=True)

# Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
index.add(embeddings)

# Search function with category support
def search_faq(user_query, threshold=0.8, category_filter=None):
    query_embedding = model.encode([user_query], normalize_embeddings=True)
    
    # If category filter is specified, search only within that category
    if category_filter:
        category_indices = [i for i, faq in enumerate(faqs) if faq.get('category') == category_filter]
        if not category_indices:
            return None
        
        # Create temporary embeddings for the filtered category
        category_embeddings = embeddings[category_indices]
        temp_index = faiss.IndexFlatIP(dimension)
        temp_index.add(category_embeddings)
        
        distances, indices = temp_index.search(query_embedding, k=1)
        actual_index = category_indices[indices[0][0]]
    else:
        distances, indices = index.search(query_embedding, k=1)
        actual_index = indices[0][0]
    
    if distances[0][0] >= threshold:
        matched_faq = faqs[actual_index]
        return {
            "question": matched_faq["question"],
            "answer": matched_faq["answer"],
            "confidence": float(distances[0][0]),
            "category": matched_faq.get("category", "Unknown")
        }
    return None

# Enhanced search function to get top N results
def search_faq_multiple(user_query, k=3, threshold=0.7, category_filter=None):
    query_embedding = model.encode([user_query], normalize_embeddings=True)
    
    if category_filter:
        category_indices = [i for i, faq in enumerate(faqs) if faq.get('category') == category_filter]
        if not category_indices:
            return []
        
        category_embeddings = embeddings[category_indices]
        temp_index = faiss.IndexFlatIP(dimension)
        temp_index.add(category_embeddings)
        
        distances, indices = temp_index.search(query_embedding, k=min(k, len(category_indices)))
        actual_indices = [category_indices[i] for i in indices[0]]
    else:
        distances, indices = index.search(query_embedding, k=k)
        actual_indices = indices[0]
    
    results = []
    for i, (distance, idx) in enumerate(zip(distances[0], actual_indices)):
        if distance >= threshold:
            matched_faq = faqs[idx]
            results.append({
                "question": matched_faq["question"],
                "answer": matched_faq["answer"],
                "confidence": float(distance),
                "category": matched_faq.get("category", "Unknown"),
                "rank": i + 1
            })
    
    return results

# Example usage
print("=== Single Best Match ===")
result = search_faq("is coverage immediate?")
if result:
    print(f"Question: {result['question']}")
    print(f"Answer: {result['answer']}")
    print(f"Category: {result['category']}")
    print(f"Confidence: {result['confidence']:.3f}")
else:
    print("No match found above threshold")

print("\n=== Multiple Matches ===")
results = search_faq_multiple("How do I pay my premium?", k=3, threshold=0.6)
for result in results:
    print(f"#{result['rank']} - Category: {result['category']}")
    print(f"Question: {result['question']}")
    print(f"Answer: {result['answer']}")
    print(f"Confidence: {result['confidence']:.3f}")
    print("-" * 50)

print("\n=== Category-Specific Search ===")
claims_result = search_faq("How long does processing take?", category_filter="claims")
if claims_result:
    print(f"Claims-specific result:")
    print(f"Question: {claims_result['question']}")
    print(f"Answer: {claims_result['answer']}")
    print(f"Confidence: {claims_result['confidence']:.3f}")

print("\n=== Available Categories ===")
categories = set(faq['category'] for faq in faqs)
print(f"Categories: {sorted(categories)}")
print(f"Total FAQs: {len(faqs)}")
for category in sorted(categories):
    count = sum(1 for faq in faqs if faq['category'] == category)
    print(f"  {category}: {count} FAQs")