# api/routes/v1/health.py
"""
Health check endpoints for API v1
"""
from fastapi import APIRouter, HTTPException
from api.schemas.responses import HealthResponse
import uuid
import time

router = APIRouter()

# Simple in-memory session storage (replace with proper storage in production)
_sessions = {}

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Basic health check for v1 API"""
    return HealthResponse(
        status="healthy",
        service="AroBot",
        version="2.0.0",
        architecture="LLM-as-Agent"
    )

@router.post("/session/create")
async def create_session():
    """Create a new chat session"""
    session_id = str(uuid.uuid4())
    _sessions[session_id] = {
        "created_at": time.time(),
        "last_activity": time.time(),
        "message_count": 0
    }
    return {
        "session_id": session_id,
        "status": "created",
        "created_at": _sessions[session_id]["created_at"]
    }

@router.get("/session/{session_id}")
async def get_session(session_id: str):
    """Get session information"""
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {
        "session_id": session_id,
        "status": "active",
        **_sessions[session_id]
    }

@router.get("/vector/indexes")
async def get_vector_indexes():
    """Get available vector store indexes"""
    return {
        "indexes": [
            {
                "name": "medical_knowledge",
                "description": "Medical knowledge base",
                "status": "active",
                "document_count": 1000,  # Placeholder
                "last_updated": time.time()
            },
            {
                "name": "prescription_data", 
                "description": "Prescription analysis data",
                "status": "active",
                "document_count": 500,  # Placeholder
                "last_updated": time.time()
            }
        ],
        "total_indexes": 2,
        "status": "healthy"
    } 