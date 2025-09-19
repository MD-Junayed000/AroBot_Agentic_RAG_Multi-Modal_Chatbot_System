"""
API routes for AroBot Medical Chatbot

"""
import logging
import re
from typing import Optional, Dict, Any, List

from fastapi import APIRouter, HTTPException, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from config.env_config import TEMPLATES_DIR
from mcp_server.mcp_handler import MCPHandler

logger = logging.getLogger(__name__)

router = APIRouter()
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Global MCP instance
_mcp_handler: Optional[MCPHandler] = None

def get_mcp() -> MCPHandler:
    global _mcp_handler
    if _mcp_handler is None:
        _mcp_handler = MCPHandler()
    return _mcp_handler

# Pydantic models
class ChatMessage(BaseModel):
    message: str
    session_id: Optional[str] = None
    conversation_id: Optional[str] = None
    use_web_search: bool = False

class PrescriptionQuery(BaseModel):
    query: str
    session_id: Optional[str] = None

class MedicineQuery(BaseModel):
    query: str
    want_price: bool = False
    session_id: Optional[str] = None

# Conversation memory for the ring buffer (keyed by session_id)
_conversation_memory: Dict[str, List[Dict[str, str]]] = {}

def add_to_conversation_memory(session_id: str, role: str, content: str):
    """Add message to conversation memory ring buffer"""
    if session_id not in _conversation_memory:
        _conversation_memory[session_id] = []
    
    _conversation_memory[session_id].append({"role": role, "content": content})
    
    # Keep only last 10 messages
    if len(_conversation_memory[session_id]) > 10:
        _conversation_memory[session_id] = _conversation_memory[session_id][-10:]

def get_conversation_memory_context(session_id: str) -> str:
    """Get conversation context as string"""
    if session_id not in _conversation_memory:
        return ""
    
    context_parts = []
    for msg in _conversation_memory[session_id][-6:]:  # Last 6 messages
        context_parts.append(f"{msg['role']}: {msg['content']}")
    
    return "\n".join(context_parts)

def clear_conversation_memory():
    """Clear all conversation memory (admin function)"""
    global _conversation_memory
    _conversation_memory.clear()

# Session binding logic
def _resolve_or_bind_session(client_id: str) -> str:
    """Resolve client session ID to internal MCP session ID"""
    try:
        mcp = get_mcp()
        
        # Check if we already have a binding
        for session_id, session_data in mcp.memory.active_sessions.items():
            if session_data.get("client_id") == client_id:
                return session_id
        
        # Create new binding
        new_session = mcp.initialize_session()
        mcp_id = new_session.get("session_id")
        if mcp_id:
            # Store the binding
            session_data = mcp.memory.active_sessions.get(mcp_id, {})
            session_data["client_id"] = client_id
            mcp.memory.active_sessions[mcp_id] = session_data
            mcp.memory._save_session(mcp_id)
        
        return mcp_id or client_id
    except Exception:
        return client_id

# Helper functions for weather detection
def _looks_like_weather(message: str) -> bool:
    """Check if message is asking about weather"""
    weather_terms = ["weather", "temperature", "rain", "sunny", "cloudy", "forecast"]
    return any(term in message.lower() for term in weather_terms)

def _get_weather_text(lat: float, lon: float) -> str:
    """Get weather information"""
    try:
        from utils.weather_utils import get_weather_data
        weather_data = get_weather_data(lat, lon)
        if weather_data:
            return f"Weather for {lat:.2f}, {lon:.2f}: {weather_data}"
        else:
            return f"Weather information for coordinates {lat:.2f}, {lon:.2f} is currently unavailable."
    except Exception as e:
        logger.exception("Error getting weather")
        return f"Weather service error: {str(e)}"

# --------------------------------------------------------------------------------------
# UI Routes
# --------------------------------------------------------------------------------------
@router.get("/chat", response_class=HTMLResponse)
async def chat_ui(request: Request):
    """Main chat interface"""
    return templates.TemplateResponse(
        "chat_enhanced.html",
        {"request": request, "title": "AroBot - Medical Assistant"}
    )

# --------------------------------------------------------------------------------------
# Session Management
# --------------------------------------------------------------------------------------
@router.post("/session/create")
async def create_session():
    """Create a new session"""
    session_data = get_mcp().initialize_session()
    return {
        "session_id": session_data.get("session_id"),
        "status": "success",
        "message": "Session created successfully"
    }

@router.get("/session/{session_id}/context")
async def get_session_context(session_id: str):
    """Get session conversation context"""
    try:
        context = get_mcp().get_conversation_context(session_id)
        return {
            "session_id": session_id,
            "context": context,
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/session/{session_id}/history")
async def get_session_history(session_id: str):
    """Get session medical history with proper error handling"""
    try:
        if not session_id:
            raise HTTPException(status_code=400, detail="Session ID is required")
        
        history = get_mcp().get_user_medical_history(session_id)
        return {
            "session_id": session_id,
            "history": history,
            "status": "success"
        }
    except Exception as e:
        logger.exception(f"Error retrieving session history for {session_id}")
        raise HTTPException(status_code=500, detail=str(e))

# --------------------------------------------------------------------------------------
# Chat - Unified endpoint that forwards to agent
# --------------------------------------------------------------------------------------

@router.post("/chat")
async def unified_chat_endpoint(req: ChatMessage):
    
    try:
        # Validate input
        if not req.message or not req.message.strip():
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        
        # Resolve session
        session_id = req.session_id or get_mcp().initialize_session().get("session_id")
        
        # Use LLMAgent directly for chat processing
        from core.agent_core import LLMAgent
        agent = LLMAgent()
        
        # Process through agent with enhanced error handling
        result = await agent.process_request(
            text_input=req.message.strip(),
            image_data=None,
            pdf_data=None,
            session_id=session_id
        )
        
        # Note: LLMAgent already saves messages to persistent MCP memory system
        # No need to duplicate with add_to_conversation_memory()
        
        return {
            "response": result.response,
            "session_id": result.session_id,
            "tools_used": result.tools_used,
            "sources": result.sources,
            "status": result.status,
            "agent_type": "llm_agent"
        }

    except Exception as e:
        logger.exception("Error in chat endpoint")
        raise HTTPException(status_code=500, detail=str(e))

# --------------------------------------------------------------------------------------
# Prescription image - REMOVED: Use /api/v1/agent instead
# --------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------
# Prescription (text)
# --------------------------------------------------------------------------------------
@router.post("/prescription/analyze")
async def analyze_prescription_text(req: PrescriptionQuery):
    
    if not req.query:
        raise HTTPException(status_code=400, detail="Query is required")
    session_id = req.session_id or get_mcp().initialize_session().get("session_id")
    get_mcp().add_user_message(session_id, f"Prescription query: {req.query}")

    # Import here to avoid circular imports
    from agents.medical_agent import MedicalAgent
    
    try:
        result = MedicalAgent().rag_agent.analyze_prescription_query(req.query)
        if result.get("status") == "success":
            text = result.get("response", "Analysis completed.")
            get_mcp().add_assistant_response(session_id, text)
            get_mcp().record_medical_query(session_id, req.query, text, "prescription")
            # Note: MCP already handles persistent memory storage
            return {"response": text, "session_id": session_id, "status": "success"}
        else:
            err = f"Error: {result.get('error', 'Unknown error')}"
            get_mcp().add_assistant_response(session_id, err)
            return {"response": err, "session_id": session_id, "status": "error"}
    except Exception as e:
        logger.exception("Error analyzing prescription text")
        raise HTTPException(status_code=500, detail=str(e))

# --------------------------------------------------------------------------------------
# Aliases for deprecated endpoints
# --------------------------------------------------------------------------------------

@router.get("/vector/indexes", include_in_schema=False)
async def alias_vector_indexes():
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/api/v1/admin/vector/indexes")


@router.post("/prescription/upload", include_in_schema=False)
async def alias_prescription_upload(
    file: UploadFile = File(...),
    message: Optional[str] = Form(None),
    session_id: Optional[str] = Form(None),
):
    """Deprecated alias: forwards to unified agent image analysis."""
    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Empty file")
        # Use LLMAgent directly to avoid circular imports
        from core.agent_core import LLMAgent
        agent = LLMAgent()
        result = await agent.process_request(
            text_input=message,
            image_data=content,
            pdf_data=None,
            session_id=session_id,
        )
        return {
            "response": result.response,
            "session_id": result.session_id,
            "tools_used": result.tools_used,
            "status": result.status,
            "agent_type": "llm_agent",
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error in deprecated prescription upload alias")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/pdf/analyze", include_in_schema=False)
async def alias_pdf_analyze(
    file: UploadFile = File(...),
    question: Optional[str] = Form(None),
    session_id: Optional[str] = Form(None),
):
    """Deprecated alias: forwards to unified agent PDF analysis."""
    try:
        if not file.content_type == "application/pdf":
            raise HTTPException(status_code=400, detail="File must be a PDF")
        
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Empty file")
            
        # Use LLMAgent directly
        from core.agent_core import LLMAgent
        agent = LLMAgent()
        result = await agent.process_request(
            text_input=question,
            image_data=None,
            pdf_data=content,
            session_id=session_id,
        )
        return {
            "response": result.response,
            "session_id": result.session_id,
            "tools_used": result.tools_used,
            "status": result.status,
            "agent_type": "llm_agent",
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error in PDF analysis alias")
        raise HTTPException(status_code=500, detail=str(e))

# --------------------------------------------------------------------------------------
# PDF: analyze - REMOVED: Use /api/v1/agent instead
# --------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------
# Medicine search
# --------------------------------------------------------------------------------------
@router.post("/medicine/search")
async def search_medicine(req: MedicineQuery):
    
    # Validate input
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="Medicine query cannot be empty")
    
    session_id = req.session_id or get_mcp().initialize_session().get("session_id")
    get_mcp().add_user_message(session_id, f"Medicine query: {req.query}")

    try:
        from core.llm_modular import ModularLLMHandler as LLMHandler
        
        llm = LLMHandler()
        response = llm.answer_medicine(req.query, want_price=req.want_price)
        
        get_mcp().add_assistant_response(session_id, response)
        get_mcp().record_medical_query(session_id, req.query, response, "medicine_search")
        # Note: MCP already handles persistent memory storage
        
        return {
            "response": response,
            "session_id": session_id,
            "query": req.query,
            "include_price": req.want_price,
            "status": "success"
        }
    except Exception as e:
        logger.exception("Error searching medicine")
        raise HTTPException(status_code=500, detail=str(e))

# --------------------------------------------------------------------------------------
# Weather
# --------------------------------------------------------------------------------------
@router.get("/weather")
async def get_weather(lat: float = 40.73061, lon: float = -73.935242):
    """Get weather information for coordinates"""
    try:
        weather_text = _get_weather_text(lat, lon)
        return {
            "weather": weather_text,
            "coordinates": {"lat": lat, "lon": lon},
            "status": "success"
        }
    except Exception as e:
        logger.exception("Error getting weather")
        raise HTTPException(status_code=500, detail=str(e))

# --------------------------------------------------------------------------------------
# Admin routes moved to api/routes/admin.py
# --------------------------------------------------------------------------------------
