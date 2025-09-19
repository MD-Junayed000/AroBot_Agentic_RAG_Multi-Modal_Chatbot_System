# api/routes/v1/health.py
"""
Health check endpoints for API v1
"""
from fastapi import APIRouter, HTTPException
from api.schemas.responses import HealthResponse
from api.core_routes import get_mcp
import uuid
import time

router = APIRouter()

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Basic health check for v1 API"""
    return HealthResponse(
        status="healthy",
        version="2.0.0",
        architecture="LLM-as-Agent"
    )

@router.post("/session/create")
async def create_session():
    """Create a new chat session using MCP Handler"""
    try:
        mcp = get_mcp()
        session_result = mcp.initialize_session()
        
        if session_result.get("error"):
            raise HTTPException(
                status_code=500,
                detail=f"Failed to create session: {session_result['error']}"
            )
        
        return {
            "session_id": session_result.get("session_id"),
            "status": "created",
            "created_at": time.time(),
            "message": session_result.get("message", "Session created successfully")
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Session creation failed: {str(e)}"
        )

@router.get("/session/{session_id}")
async def get_session(session_id: str):
    """Get session information"""
    try:
        mcp = get_mcp()
        session_status = mcp.get_session_status(session_id)
        
        if not session_status.get("exists"):
            raise HTTPException(status_code=404, detail="Session not found")
        
        return {
            "session_id": session_id,
            "status": session_status.get("status", "active"),
            "created_at": session_status.get("created_at"),
            "last_activity": session_status.get("last_activity"),
            "message_count": session_status.get("message_count", 0)
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get session: {str(e)}"
        )

@router.get("/vector/indexes")
async def get_vector_indexes():
    """Get available vector store indexes"""
    try:
        from core.vector_store import PineconeStore
        store = PineconeStore()
        return {
            "indexes": store.list_indexes() if hasattr(store, 'list_indexes') else [],
            "status": "available"
        }
    except Exception as e:
        return {
            "indexes": [],
            "status": "unavailable",
            "error": str(e)
        } 