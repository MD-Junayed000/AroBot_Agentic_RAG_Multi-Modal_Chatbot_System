# api/routes/simple.py
"""Simplified API routes for essential medical chatbot functionality"""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import Optional
import logging
from pydantic import BaseModel

from core.agent_core import LLMAgent

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize agent
agent = LLMAgent()

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    tools_used: list
    status: str

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Simple unified chat endpoint"""
    try:
        result = await agent.process_request(
            text_input=request.message,
            session_id=request.session_id
        )
        return ChatResponse(
            response=result.response,
            tools_used=result.tools_used,
            status=result.status
        )
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/agent")
async def unified_agent(
    message: str = Form(...),
    file: Optional[UploadFile] = File(None)
):
    """Unified agent endpoint for text and file uploads"""
    try:
        image_data = None
        pdf_data = None
        
        if file:
            # Read uploaded file
            file_content = await file.read()
            
            # Determine file type
            if file.content_type and file.content_type.startswith('image/'):
                image_data = file_content
            elif file.content_type == 'application/pdf' or file.filename.endswith('.pdf'):
                pdf_data = file_content
            else:
                # Default to image for other file types
                image_data = file_content
        
        result = await agent.process_request(
            text_input=message,
            image_data=image_data,
            pdf_data=pdf_data
        )
        
        return {
            "response": result.response,
            "tools_used": result.tools_used,
            "status": result.status,
            "session_id": result.session_id
        }
        
    except Exception as e:
        logger.error(f"Agent error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "AroBot-Simplified",
        "version": "3.1.0"
    }
