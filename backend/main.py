"""
FastAPI backend for MineLawHub.
Provides endpoints for semantic search, web search, and chat functionality.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import uvicorn
import os

from search_engine import SearchEngine
from custom_client import CustomClient


# Initialize FastAPI app
app = FastAPI(
    title="MineLawHub API",
    description="AI-powered Mining Law Chatbot System",
    version="1.0.0"
)

# Configure CORS — allows Render frontend domain or localhost for dev
cors_origins = os.environ.get("CORS_ORIGINS", "http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize clients (lazy loading)
search_engine = None
custom_client = None


def get_search_engine():
    """Get or initialize search engine."""
    global search_engine
    if search_engine is None:
        search_engine = SearchEngine()
    return search_engine


def get_custom_client():
    """Get or initialize Custom Local client."""
    global custom_client
    if custom_client is None:
        custom_client = CustomClient()
    return custom_client


# Request/Response models
class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5


class TavilySearchRequest(BaseModel):
    query: str
    max_results: Optional[int] = 5


class ChatRequest(BaseModel):
    query: str


class SearchResponse(BaseModel):
    results: List[Dict]
    total: int


class TavilySearchResponse(BaseModel):
    results: List[Dict]
    total: int


class ChatResponse(BaseModel):
    answer: str
    citations: List[Dict]
    used_chunks: Optional[List[Dict]] = None
    intent: Optional[str] = None


# Endpoints
@app.get("/")
async def root():
    """Root endpoint to verify server is running."""
    return {
        "status": "online",
        "message": "Welcome to MineLawHub API",
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "MineLawHub API",
        "version": "1.0.0"
    }


@app.post("/embed-search", response_model=SearchResponse)
async def embed_search(request: SearchRequest):
    """
    Semantic search using ChromaDB embeddings.
    
    Args:
        request: SearchRequest with query and optional top_k
        
    Returns:
        SearchResponse with results and metadata
    """
    try:
        engine = get_search_engine()
        results = engine.search(request.query, top_k=request.top_k)
        
        return SearchResponse(
            results=results,
            total=len(results)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")


# Removed Tavily search endpoints to maintain "No external AI" requirement


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint using CUSTOM LOCAL MODELS only.
    """
    try:
        client = get_custom_client()
        
        # Step 1: Classify intent using custom model
        intent = client.classify_intent(request.query)
        
        context_chunks = None
        engine = get_search_engine()
        
        # Step 2: Retrieve context based on intent (Local Only)
        if intent == 'static_law' or len(request.query.split()) >= 3:
            context_chunks = engine.search(request.query, top_k=5)
            
        # Step 3: Generate answer using custom Transformer models
        result = client.generate_answer(
            query=request.query,
            context_chunks=context_chunks,
            search_engine=engine
        )
        
        # Step 4: Return response
        return ChatResponse(
            answer=result['answer'],
            citations=result['citations'],
            used_chunks=context_chunks,
            intent=intent
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")


@app.get("/stats")
async def get_stats():
    """Get system statistics."""
    try:
        engine = get_search_engine()
        stats = engine.get_collection_stats()
        
        return {
            "database": stats,
            "status": "operational"
        }
        
    except Exception as e:
        return {
            "database": {"status": "error", "error": str(e)},
            "status": "degraded"
        }


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print("\n" + "="*60)
    print("MineLawHub API Server")
    print("="*60)
    print(f"\nStarting server on http://0.0.0.0:{port}")
    print(f"API documentation: http://localhost:{port}/docs")
    print("\nPress Ctrl+C to stop\n")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=(os.environ.get("RENDER") is None)
    )
