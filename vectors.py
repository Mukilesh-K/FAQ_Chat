import json
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from langdetect import detect, LangDetectException
import os
from sklearn.feature_extraction.text import TfidfVectorizer

# Load and preprocess the FAQ data
with open("faq.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Language and category mapping
categories = {
    'en': ['coverage', 'general', 'policymanagement', 'payments', 'claims'],
    'es': ['cobertura', 'general', 'gestión_de_pólizas', 'pagos', 'reclamaciones'],
    'hi': ['कवरेज', 'सामान्य', 'नीति_प्रबंधन', 'भुगतान', 'दावा']
}

# Store embeddings and texts per language
language_data = {}

# Load SentenceTransformer model
model = SentenceTransformer("all-mpnet-base-v2")

# Language-specific thresholds (higher for non-English)
LANGUAGE_THRESHOLDS = {
    'en': 0.70,  # 70%
    'es': 0.75,  # 75%
    'hi': 0.85   # 85%
}

for lang, lang_categories in categories.items():
    # Combine all categories for this language
    faq_entries = []
    for category in lang_categories:
        faq_entries.extend(data.get(lang, {}).get(category, []))
    
    # Store questions and answers separately
    questions = [item['question'] for item in faq_entries]
    answers = [item['answer'] for item in faq_entries]
    
    # Generate separate embeddings for questions only
    print(f"Encoding {lang} questions...")
    question_embeddings = model.encode(questions, normalize_embeddings=True)
    question_embeddings = np.array(question_embeddings)
    
    # Build FAISS index for questions
    dimension = question_embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(question_embeddings)
    
    # Initialize TF-IDF vectorizer for keyword matching
    vectorizer = TfidfVectorizer()
    vectorizer.fit(questions)
    
    # Store data for this language
    language_data[lang] = {
        'index': index,
        'questions': questions,
        'answers': answers,
        'vectorizer': vectorizer,
        'threshold': LANGUAGE_THRESHOLDS[lang]
    }

# Save FAISS indices and FAQ data
for lang, data in language_data.items():
    faiss.write_index(data['index'], f"faq_index_{lang}.faiss")
    with open(f"faq_data_{lang}.pkl", "wb") as f:
        pickle.dump({
            'questions': data['questions'],
            'answers': data['answers'],
            'vectorizer': data['vectorizer'],
            'threshold': data['threshold']
        }, f)

print("Indices and data saved successfully!")

def detect_query_language(query: str, default_lang: str = 'en') -> str:
    """Detect language with fallback to default"""
    try:
        lang = detect(query)
        return lang if lang in categories else default_lang
    except LangDetectException:
        return default_lang

def keyword_overlap(query: str, question: str, vectorizer) -> float:
    """Calculate keyword overlap score between query and question"""
    query_vec = vectorizer.transform([query])
    question_vec = vectorizer.transform([question])
    return (query_vec * question_vec.T).toarray()[0][0]

def search_faq(query: str, top_k: int = 5):
    try:
        # Auto-detect language
        lang = detect_query_language(query)
        print(f"Detected language: {lang} for query: {query}")
        
        # Load index and data
        index = faiss.read_index(f"faq_index_{lang}.faiss")
        with open(f"faq_data_{lang}.pkl", "rb") as f:
            data = pickle.load(f)
        
        questions = data['questions']
        answers = data['answers']
        vectorizer = data['vectorizer']
        threshold = data['threshold']
        
        # Encode query
        query_embedding = model.encode([query], normalize_embeddings=True)
        
        # Search using inner product (cosine similarity)
        distances, indices = index.search(query_embedding, top_k)
        
        # Process results with combined scoring
        results = []
        for idx, score in zip(indices[0], distances[0]):
            similarity = (1 + score) / 2  # Convert to [0,1] range
            
            # Calculate keyword overlap
            kw_score = keyword_overlap(query, questions[idx], vectorizer)
            
            # Combined score (weighted average)
            combined_score = (0.7 * similarity) + (0.3 * kw_score)
            
            if combined_score >= threshold:
                results.append({
                    "question": questions[idx],
                    "answer": answers[idx],
                    "score": float(combined_score),
                    "score_percent": float(combined_score * 100),
                    "similarity": float(similarity),
                    "keyword_score": float(kw_score)
                })
        
        # Sort by combined score and return top_k
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]
        
    except Exception as e:
        print(f"Search error: {str(e)}")
        return []

if __name__ == "__main__":
    user_query = input("Ask a question: ")
    results = search_faq(user_query)
    
    print("\nTop Matches:\n")
    for i, result in enumerate(results, 1):
        print(f"{i}. Question: {result['question']}")
        print(f"   Answer: {result['answer']}")
        print(f"   Combined Score: {result['score_percent']:.1f}%")
        print(f"   (Similarity: {result['similarity']:.2f}, Keywords: {result['keyword_score']:.2f})\n")