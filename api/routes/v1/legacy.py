# api/routes/v1/legacy.py
"""
Legacy endpoints for backward compatibility
"""
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends
from typing import Optional
from api.schemas.responses import ChatResponse, AgentResponse
from core.agent_core import LLMAgent

router = APIRouter()

# Agent dependency
_agent: Optional[LLMAgent] = None

def get_agent() -> LLMAgent:
    global _agent
    if _agent is None:
        _agent = LLMAgent()
    return _agent

# Commented out legacy chat endpoint to allow functional chat endpoint from __init__.py
# @router.post("/chat", response_model=ChatResponse, deprecated=True)
# async def legacy_chat_endpoint():
#     """Legacy chat endpoint - redirects to new agent endpoint"""
#     raise HTTPException(
#         status_code=301,
#         detail={
#             "message": "This endpoint is deprecated. Please use /api/v1/agent instead.",
#             "new_endpoint": "/api/v1/agent",
#             "migration_guide": "https://docs.arobot.com/migration"
#         }
#     )

@router.post("/prescription/upload", response_model=AgentResponse)
async def legacy_prescription_endpoint(
    file: UploadFile = File(...),
    message: Optional[str] = Form("Analyze this prescription and provide information about the medicine"),
    session_id: Optional[str] = Form(None),
    agent: LLMAgent = Depends(get_agent)
):
    """Legacy prescription endpoint - now functional for backward compatibility"""
    try:
        # Validate file
        if not file:
            raise HTTPException(status_code=400, detail="No file uploaded")
        
        # Read file content
        file_content = await file.read()
        
        # Validate file size (50MB limit)
        if len(file_content) > 50 * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail="File too large. Maximum 50MB allowed."
            )
        
        # Process with agent (removed options parameter)
        result = await agent.process_request(
            text_input=message,
            image_data=file_content,
            pdf_data=None,
            session_id=session_id
        )
        
        # Transform core AgentResponse to API AgentResponse
        from api.schemas.responses import SourceInfo, ToolInfo
        
        # Convert tools information
        tool_infos = []
        if result.sources and 'tools' in result.sources:
            for tool in result.sources['tools']:
                tool_infos.append(ToolInfo(
                    name=tool.get('name', ''),
                    description=tool.get('description', ''),
                    category=tool.get('category', 'general'),
                    priority=tool.get('priority', 1),
                    execution_time=tool.get('execution_time'),
                    confidence=tool.get('confidence')
                ))
        
        # Create SourceInfo
        source_info = SourceInfo(
            tools=tool_infos,
            llm_agent=result.sources.get('llm_agent', True) if result.sources else True,
            knowledge_base=result.sources.get('knowledge_base', 0) if result.sources else 0,
            web_search=result.sources.get('web_search', 0) if result.sources else 0,
            cached=result.sources.get('cached', False) if result.sources else False
        )
        
        # Return properly structured agent response
        return AgentResponse(
            response=result.response,
            session_id=result.session_id or session_id or "unknown",
            tools_used=result.tools_used,
            sources=source_info,
            status=result.status,
            confidence=getattr(result, 'confidence', None),
            response_time=getattr(result, 'response_time', None),
            token_count=getattr(result, 'token_count', None),
            agent_type="llm_agent",
            model_version="2.0.0"
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process prescription: {str(e)}"
        )

@router.post("/pdf/analyze", deprecated=True)
async def legacy_pdf_endpoint():
    """Legacy PDF endpoint - redirects to new agent endpoint"""
    raise HTTPException(
        status_code=301,
        detail={
            "message": "This endpoint is deprecated. Please use /api/v1/agent instead.",
            "new_endpoint": "/api/v1/agent", 
            "migration_guide": "https://docs.arobot.com/migration"
        }
    ) 