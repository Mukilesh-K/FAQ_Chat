from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional, Union
from vectors import search_faq, detect_query_language
from fastapi.middleware.cors import CORSMiddleware
from main import InsuranceChatbot
import os
from dotenv import load_dotenv
from langchain_community.callbacks import get_openai_callback

load_dotenv()

app = FastAPI(title="Insurance Chatbot API")

origins = [
    "http://localhost",
    "http://localhost:3000",  
    "http://localhost:5173",# React default
    "https://rms.healtheesystems.com:9443",
    "https://rms.healtheesystems.com",
    "https://azpresencedemo.healtheesystems.com:9443",
    "https://azpresencedemo.healtheesystems.com"
    # "https://your-frontend-domain.com"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,            # List of allowed origins
    allow_credentials=True,           # Allow cookies/auth headers
    allow_methods=["*"],              # Allow all HTTP methods
    allow_headers=["*"],              # Allow all headers
)

# Initialize the chatbot
chatbot = InsuranceChatbot(api_key=os.getenv("OPENAI_API_KEY"))

class FAQMatch(BaseModel):
    question: str
    answer: str
    confidence_score: float  # as percentage
    is_above_threshold: bool

class LLMResponse(BaseModel):
    content: str
    # token_usage: Optional[Dict[str, int]] = None

class UserQuery(BaseModel):
    question: str
    top_k: Optional[int] = 3
    threshold: Optional[float] = 70.0  # Allow dynamic threshold

class APIResponse(BaseModel):
    success: bool
    message: str
    error: Optional[str] = None
    source: Optional[str] = None  # "faq" or "llm"
    query: Optional[str] = None
    threshold: Optional[float] = None
    # processing_time_ms: Optional[float] = None
    faq_matches: Optional[List[FAQMatch]] = None
    llm_response: Optional[LLMResponse] = None
    metadata: Optional[Dict[str, Union[str, int, float]]] = None

@app.get("/chat")
def health_check():
    return "Application Health Status Good"

@app.post("/ai", response_model=APIResponse)
async def ask_question(query: UserQuery):
    import time
    start_time = time.time()
    
    try:
        # Auto-detect language from query
        lang = detect_query_language(query.question)
        print(f"Detected language: {lang}")
        
        # Search FAQ with auto-detected language
        results = search_faq(query.question, top_k=query.top_k)
        print("Results: ", results)
        
        # Process results
        processed_matches = []
        best_match = None
        best_score = 0.0
        
        for result in results:
            # Use the already extracted Q&A from the search_faq results
            question = result["question"]
            answer = result["answer"]
            score_percent = result["score_percent"]
            is_above = score_percent >= query.threshold
            
            match = FAQMatch(
                question=question,
                answer=answer,
                confidence_score=score_percent,
                is_above_threshold=is_above
            )
            
            processed_matches.append(match)
            
            # Track the best match
            if score_percent > best_score:
                best_score = score_percent
                best_match = match
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000  # in milliseconds
        print("Processing time: ", processing_time)
        
        # Check if we have a match above threshold
        if best_score >= query.threshold:
            return APIResponse(
                success=True,
                message=f"Found FAQ match with {best_score:.1f}% confidence (threshold: {query.threshold}%)",
                source="faq",
                query=query.question,
                threshold=query.threshold,
                faq_matches=processed_matches,
                metadata={
                    "match_strategy": "vector_similarity"
                }
            )
        else:
            # No good FAQ match, use LLM
            llm_response = chatbot.get_insurance_response(query.question)

            return APIResponse(
                success=True,
                message=f"No FAQ match above {query.threshold}% threshold (best match: {best_score:.1f}%) - So Utilising LLM",
                source="llm",
                query=query.question,
                threshold=query.threshold,
                llm_response=LLMResponse(content=llm_response),
                metadata={
                    "match_strategy": "llm_fallback"
                }
            )
            
    except Exception as e:
        return APIResponse(
            success=False,
            message="Failed to process question",
            error=str(e),
            query=query.question if 'query' in locals() else None
        )

# @app.get("/health")
# async def health_check():
#     return {"status": "healthy", "version": "1.0.0"}
