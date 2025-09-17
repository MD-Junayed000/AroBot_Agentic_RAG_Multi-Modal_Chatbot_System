# api/routes/v2/chat.py
"""
Enhanced chat endpoints for API v2
"""
from fastapi import APIRouter, HTTPException, Depends
from api.schemas.requests import ChatRequest
from api.schemas.responses import ChatResponse
from core.agent_core import LLMAgent

router = APIRouter()

_agent: LLMAgent = None

def get_agent() -> LLMAgent:
    global _agent
    if _agent is None:
        _agent = LLMAgent()
    return _agent

@router.post("/chat", response_model=ChatResponse)
async def enhanced_chat(
    request: ChatRequest,
    agent: LLMAgent = Depends(get_agent)
):
    """Enhanced chat with better context management"""
    try:
        result = await agent.process_request(
            text_input=request.message,
            session_id=request.session_id
        )
        
        return ChatResponse(
            response=result.response,
            session_id=result.session_id,
            status=result.status,
            response_time=getattr(result, 'response_time', None)
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Chat processing failed: {str(e)}"
        ) 