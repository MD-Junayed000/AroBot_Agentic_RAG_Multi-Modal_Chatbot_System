# api/routes/admin.py
"""
Admin-only routes for system management
Hidden from main API schema and protected by environment guards
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from starlette.responses import JSONResponse

from config.env_config import DEBUG
from core.llm_modular import ModularLLMHandler as LLMHandler
from core.vector_store import PineconeStore
from mcp_server.mcp_handler import MCPHandler
from utils.data_ingestion import ingest_pdf_to_knowledge_base

import logging
logger = logging.getLogger(__name__)

# Admin router - hidden from schema
router = APIRouter(
    include_in_schema=False,
    tags=["admin"],
    prefix="/api/v1/admin"
)

_mcp_singleton: Optional[MCPHandler] = None
def get_mcp() -> MCPHandler:
    global _mcp_singleton
    if _mcp_singleton is None:
        _mcp_singleton = MCPHandler()
    return _mcp_singleton

def require_admin():
    """Security dependency for admin routes"""
    if not DEBUG and os.getenv("ENABLE_ADMIN_ROUTES") != "true":
        raise HTTPException(
            status_code=403, 
            detail="Admin routes are disabled in production. Set ENABLE_ADMIN_ROUTES=true to enable."
        )
    return True

@router.post("/pdf/upload")
async def upload_pdf_to_knowledge_base(
    file: UploadFile = File(...),
    namespace: Optional[str] = None,
    _: bool = Depends(require_admin)
):
    """
    ADMIN: Upload PDF to knowledge base (indexes into vector store)
    """
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="File must be a PDF")

    try:
        content = await file.read()
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        temp_file.write(content)
        temp_file.close()

        # Use the ingestion utility
        result = ingest_pdf_to_knowledge_base(
            pdf_path=temp_file.name,
            namespace=namespace or "general",
            filename_override=file.filename
        )

        # Clean up
        Path(temp_file.name).unlink(missing_ok=True)

        return {
            "message": f"PDF '{file.filename}' uploaded to knowledge base",
            "namespace": namespace or "general",
            "chunks_processed": result.get("chunks", 0),
            "status": "success"
        }

    except Exception as e:
        logger.exception("Error uploading PDF to knowledge base")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.post("/vector/create-index")
async def create_vector_index(
    payload: dict,
    _: bool = Depends(require_admin)
):
    """
    ADMIN: Create a new Pinecone index
    """
    try:
        index_name = payload.get("index_name")
        dimension = payload.get("dimension", 768)
        metric = payload.get("metric", "cosine")

        if not index_name:
            raise HTTPException(status_code=400, detail="index_name is required")

        vs = PineconeStore(index_name=index_name, dimension=dimension, metric=metric)
        ok = vs.create_index(index_name, dimension=dimension, metric=metric)

        return {
            "message": f"Index '{index_name}' created successfully",
            "index_name": index_name,
            "dimension": dimension,
            "metric": metric,
            "status": "success"
        }

    except Exception as e:
        logger.exception("Error creating vector index")
        raise HTTPException(status_code=500, detail=f"Index creation failed: {str(e)}")


@router.get("/vector/indexes")
async def list_vector_indexes(_: bool = Depends(require_admin)):
    """
    ADMIN: List all Pinecone indexes and their stats
    """
    try:
        vs = PineconeStore(index_name="placeholder")
        indexes = vs.list_indexes()

        index_info = []
        for idx in indexes:
            try:
                stats = vs.get_index_stats(idx)
                if isinstance(stats, dict) and "error" not in stats:
                    namespaces = stats.get("namespaces") or {}
                    total_vectors = stats.get("total_vector_count")
                    if total_vectors is None:
                        total_vectors = sum(
                            (ns.get("vector_count") or 0)
                            for ns in namespaces.values()
                            if isinstance(ns, dict)
                        )
                    index_info.append({
                        "name": idx,
                        "total_vector_count": total_vectors or 0,
                        "dimension": stats.get("dimension"),
                        "index_fullness": stats.get("index_fullness"),
                        "namespaces": namespaces,
                        "status": "active",
                        "raw_stats": stats,
                    })
                else:
                    index_info.append({
                        "name": idx,
                        "status": "error",
                        "error": stats.get("error") if isinstance(stats, dict) else str(stats),
                    })
            except Exception as e:
                index_info.append({
                    "name": idx,
                    "status": "error",
                    "error": str(e)
                })

        return {
            "indexes": index_info,
            "total_indexes": len(indexes),
            "status": "success"
        }

    except Exception as e:
        logger.exception("Error listing vector indexes")
        raise HTTPException(status_code=500, detail=f"Failed to list indexes: {str(e)}")


@router.get("/system/pinecone")
async def get_pinecone_system_info(_: bool = Depends(require_admin)):
    """
    ADMIN: Get Pinecone system information and diagnostics
    """
    try:
        vs = PineconeStore(index_name="placeholder")
        
        # Get environment info
        import pinecone
        env_info = {
            "api_key_configured": bool(os.getenv("PINECONE_API_KEY")),
            "environment": os.getenv("PINECONE_ENVIRONMENT", "unknown"),
            "pinecone_version": getattr(pinecone, "__version__", "unknown")
        }

        # Get indexes
        try:
            indexes = vs.list_indexes()
            index_details = {}
            
            for idx_name in indexes:
                try:
                    stats = vs.get_index_stats(idx_name)
                    index_details[idx_name] = stats
                except Exception as e:
                    index_details[idx_name] = {"error": str(e)}
                    
        except Exception as e:
            indexes = []
            index_details = {"error": str(e)}

        return {
            "environment": env_info,
            "indexes": {
                "names": indexes,
                "details": index_details
            },
            "health": "connected" if indexes else "disconnected",
            "status": "success"
        }

    except Exception as e:
        logger.exception("Error getting Pinecone system info")
        return JSONResponse(
            status_code=500,
            content={
                "error": str(e),
                "health": "error",
                "status": "error"
            }
        )


@router.get("/system/health")
async def admin_health_check(_: bool = Depends(require_admin)):
    """
    ADMIN: Comprehensive system health check
    """
    health_status = {
        "timestamp": __import__("datetime").datetime.now().isoformat(),
        "services": {},
        "overall": "healthy"
    }

    # Check LLM Handler
    try:
        llm = LLMHandler()
        health_status["services"]["llm"] = {
            "status": "healthy",
            "text_model": getattr(llm, "text_llm", {}).get("model", "unknown"),
            "vision_model": getattr(llm, "vision_llm", {}).get("model", "unknown")
        }
    except Exception as e:
        health_status["services"]["llm"] = {"status": "error", "error": str(e)}
        health_status["overall"] = "degraded"

    # Check Vector Store
    try:
        vs = VectorStore()
        indexes = vs.list_indexes()
        health_status["services"]["vector_store"] = {
            "status": "healthy",
            "indexes_count": len(indexes),
            "indexes": indexes[:5]  # First 5 only
        }
    except Exception as e:
        health_status["services"]["vector_store"] = {"status": "error", "error": str(e)}
        health_status["overall"] = "degraded"

    # Check MCP Handler
    try:
        mcp = get_mcp()
        session_count = len(mcp.memory.active_sessions) if hasattr(mcp, "memory") else 0
        health_status["services"]["mcp"] = {
            "status": "healthy",
            "active_sessions": session_count
        }
    except Exception as e:
        health_status["services"]["mcp"] = {"status": "error", "error": str(e)}
        health_status["overall"] = "degraded"

    return health_status


@router.delete("/system/cache/clear")
async def clear_system_caches(_: bool = Depends(require_admin)):
    """
    ADMIN: Clear system caches (sessions, memory, tool cache, etc.)
    """
    try:
        cleared = []

        # Clear MCP sessions
        try:
            mcp = get_mcp()
            if hasattr(mcp, "memory") and hasattr(mcp.memory, "active_sessions"):
                session_count = len(mcp.memory.active_sessions)
                mcp.memory.active_sessions.clear()
                cleared.append(f"MCP sessions: {session_count} cleared")
        except Exception as e:
            cleared.append(f"MCP sessions: error - {str(e)}")

        # Clear conversation memory
        try:
            from api.core_routes import clear_conversation_memory
            clear_conversation_memory()
            cleared.append("Conversation memory: cleared")
        except Exception as e:
            cleared.append(f"Conversation memory: error - {str(e)}")

        # Clear agent tool selection cache
        try:
            from api.routes.agent import get_agent
            agent = get_agent()
            cache_stats = agent.get_cache_stats()
            agent.clear_tool_cache()
            cleared.append(f"Tool selection cache: {cache_stats['cache_size']} entries cleared")
        except Exception as e:
            cleared.append(f"Tool selection cache: error - {str(e)}")

        return {
            "message": "System caches cleared",
            "cleared_items": cleared,
            "status": "success"
        }

    except Exception as e:
        logger.exception("Error clearing system caches")
        raise HTTPException(status_code=500, detail=f"Cache clear failed: {str(e)}")


@router.get("/system/cache/stats")
async def get_cache_stats(_: bool = Depends(require_admin)):
    """
    ADMIN: Get cache statistics
    """
    try:
        stats = {}

        # MCP session stats
        try:
            mcp = get_mcp()
            if hasattr(mcp, "memory") and hasattr(mcp.memory, "active_sessions"):
                stats["mcp_sessions"] = {
                    "count": len(mcp.memory.active_sessions),
                    "sessions": list(mcp.memory.active_sessions.keys())[:5]  # First 5 only
                }
        except Exception as e:
            stats["mcp_sessions"] = {"error": str(e)}

        # Tool selection cache stats
        try:
            from api.routes.agent import get_agent
            agent = get_agent()
            stats["tool_selection_cache"] = agent.get_cache_stats()
        except Exception as e:
            stats["tool_selection_cache"] = {"error": str(e)}

        return {
            "cache_statistics": stats,
            "status": "success"
        }

    except Exception as e:
        logger.exception("Error getting cache stats")
        raise HTTPException(status_code=500, detail=f"Failed to get cache stats: {str(e)}")


@router.get("/debug/agent-tools")
async def debug_agent_tools(_: bool = Depends(require_admin)):
    """
    ADMIN: Debug information about agent tools
    """
    try:
        from core.agent_core import LLMAgent
        
        agent = LLMAgent()
        
        debug_info = {
            "total_tools": len(agent.tools),
            "categories": {},
            "tools_detail": []
        }

        # Category breakdown
        for category, tool_names in agent.registry.categories.items():
            tools = agent.get_tools_by_category(category)
            debug_info["categories"][category] = {
                "count": len(tool_names),
                "tools": tool_names,
                "priorities": [t.priority for t in tools]
            }

        # Tool details
        for tool in agent.registry.get_all_tools():
            debug_info["tools_detail"].append({
                "name": tool.name,
                "category": tool.category,
                "priority": tool.priority,
                "description": tool.description,
                "parameter_count": len(tool.parameters),
                "parameters": [{"name": p.name, "type": p.type, "required": p.required} for p in tool.parameters]
            })

        return debug_info

    except Exception as e:
        logger.exception("Error getting agent tools debug info")
        raise HTTPException(status_code=500, detail=f"Debug failed: {str(e)}")
