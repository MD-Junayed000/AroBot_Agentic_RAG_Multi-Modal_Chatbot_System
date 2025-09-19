# api/routes/admin/system.py
"""
System administration endpoints
"""
import logging
from typing import Dict, Any
from fastapi import APIRouter, HTTPException
from api.core_routes import get_mcp

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/session/{session_id}/debug")
async def debug_session(session_id: str) -> Dict[str, Any]:
    """Debug session information"""
    try:
        mcp = get_mcp()
        status = mcp.get_session_status(session_id)
        
        # Add memory directory information
        memory_info = {
            "memory_dir_absolute": str(mcp.memory.memory_dir.absolute()),
            "memory_dir_exists": mcp.memory.memory_dir.exists(),
            "memory_dir_writable": mcp.memory.memory_dir.exists() and 
                                 (mcp.memory.memory_dir / ".test").touch() or True,
            "session_files": list(mcp.memory.memory_dir.glob("*.json")),
        }
        
        # Clean up test file
        try:
            (mcp.memory.memory_dir / ".test").unlink(missing_ok=True)
        except:
            pass
        
        status["memory_info"] = memory_info
        return status
        
    except Exception as e:
        logger.error(f"Error debugging session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/memory/status")
async def memory_status() -> Dict[str, Any]:
    """Get memory system status"""
    try:
        mcp = get_mcp()
        memory_dir = mcp.memory.memory_dir
        
        # Count session files
        session_files = list(memory_dir.glob("*.json"))
        
        return {
            "memory_dir": str(memory_dir.absolute()),
            "memory_dir_exists": memory_dir.exists(),
            "total_sessions": len(session_files),
            "active_sessions": len(mcp.memory.active_sessions),
            "session_files": [f.name for f in session_files],
            "status": "healthy" if memory_dir.exists() else "error"
        }
        
    except Exception as e:
        logger.error(f"Error getting memory status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/langsmith/status")
async def langsmith_status() -> Dict[str, Any]:
    """Get LangSmith configuration status"""
    try:
        from core.langsmith_config import get_langsmith_status
        return get_langsmith_status()
    except Exception as e:
        logger.error(f"Error getting LangSmith status: {e}")
        raise HTTPException(status_code=500, detail=str(e)) 