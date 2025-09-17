# api/routes/agent.py
"""
Unified Agent-Based Endpoint
Single entry point where LLM agent decides which tools to use
"""
from __future__ import annotations

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import Optional
from PIL import Image, UnidentifiedImageError
import io

from core.agent_core import LLMAgent

router = APIRouter()

# Single agent instance
_agent: Optional[LLMAgent] = None

def get_agent() -> LLMAgent:
    """Get or create the LLM agent instance"""
    global _agent
    if _agent is None:
        _agent = LLMAgent()
    return _agent


@router.post("/api/v1/agent")
async def unified_agent_endpoint(
    # Text input
    message: Optional[str] = Form(None),
    # File inputs
    file: Optional[UploadFile] = File(None),
    # Session management
    session_id: Optional[str] = Form(None),
    # Optional metadata
    file_type: Optional[str] = Form(None),  # "image", "pdf", or auto-detect
):
    """
    Unified Agent Endpoint: LLM decides which tools to use based on input
    
    Accepts:
    - Text message (chat, questions, queries)
    - Image files (prescriptions, diagrams, photos)
    - PDF files (documents, papers, manuals)
    - Any combination of the above
    
    The LLM agent will analyze the input and automatically select the appropriate tools.
    """
    try:
        # Validate input
        if not message and not file:
            raise HTTPException(status_code=400, detail="Either message or file must be provided")
        
        # Process file if provided
        image_data = None
        pdf_data = None
        
        if file:
            file_content = await file.read()
            
            # Auto-detect file type if not specified
            if not file_type:
                if file.content_type and file.content_type.startswith("image/"):
                    file_type = "image"
                elif file.content_type == "application/pdf":
                    file_type = "pdf"
                else:
                    # Try to detect by content
                    try:
                        Image.open(io.BytesIO(file_content))
                        file_type = "image"
                    except:
                        if file_content.startswith(b'%PDF'):
                            file_type = "pdf"
                        else:
                            raise HTTPException(status_code=400, detail="Unsupported file type")
            
            # Assign to appropriate variable
            if file_type == "image":
                # Validate image
                try:
                    Image.open(io.BytesIO(file_content))
                    image_data = file_content
                except UnidentifiedImageError:
                    raise HTTPException(status_code=400, detail="Invalid image file")
            elif file_type == "pdf":
                # Validate PDF
                if not file_content.startswith(b'%PDF'):
                    raise HTTPException(status_code=400, detail="Invalid PDF file")
                pdf_data = file_content
            else:
                raise HTTPException(status_code=400, detail="Unsupported file type")
        
        # Process with agent
        agent = get_agent()
        result = await agent.process_request(
            text_input=message,
            image_data=image_data,
            pdf_data=pdf_data,
            session_id=session_id
        )
        
        # Return standardized response
        return {
            "response": result.response,
            "session_id": result.session_id,
            "tools_used": result.tools_used,
            "sources": result.sources,
            "status": result.status,
            "agent_type": "llm_agent"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent processing failed: {str(e)}")


@router.get("/api/v1/agent/tools")
async def list_available_tools(category: Optional[str] = None):
    """
    List all available tools that the agent can use
    """
    try:
        agent = get_agent()
        
        if category:
            tools = agent.get_tools_by_category(category)
        else:
            tools = agent.registry.get_all_tools()
        
        tools_info = [tool.to_dict() for tool in tools]
        
        return {
            "tools": tools_info,
            "categories": list(agent.registry.categories.keys()),
            "total_tools": len(tools_info),
            "filtered_by_category": category,
            "status": "success"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list tools: {str(e)}")


@router.get("/api/v1/agent/tools/categories")
async def list_tool_categories():
    """
    List all available tool categories
    """
    try:
        agent = get_agent()
        
        category_info = {}
        for category, tool_names in agent.registry.categories.items():
            tools = agent.get_tools_by_category(category)
            category_info[category] = {
                "tools": tool_names,
                "count": len(tool_names),
                "priorities": [tool.priority for tool in tools]
            }
        
        return {
            "categories": category_info,
            "total_categories": len(category_info),
            "status": "success"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list categories: {str(e)}")


@router.post("/api/v1/agent/explain")
async def explain_tool_selection(
    message: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    session_id: Optional[str] = Form(None),
):
    """
    Explain which tools the agent would select for given input (without executing them)
    Useful for debugging and understanding agent behavior
    """
    try:
        if not message and not file:
            raise HTTPException(status_code=400, detail="Either message or file must be provided")
        
        agent = get_agent()
        
        # Prepare input description
        image_data = None
        pdf_data = None
        
        if file:
            file_content = await file.read()
            if file.content_type and file.content_type.startswith("image/"):
                image_data = file_content
            elif file.content_type == "application/pdf":
                pdf_data = file_content
        
        explanation = await agent.explain_request(
            text_input=message,
            image_data=image_data,
            pdf_data=pdf_data,
            session_id=session_id,
        )

        conversation_context = explanation.get("conversation_context", "")
        if conversation_context and len(conversation_context) > 200:
            explanation["conversation_context"] = conversation_context[:200] + "..."

        explanation["status"] = "success"
        return explanation
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to explain tool selection: {str(e)}")
