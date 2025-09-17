# api/routes/v1/__init__.py
"""
API v1 - Stable endpoints
"""
from fastapi import APIRouter
from . import agent, health, legacy

router = APIRouter()

# Include sub-routers
router.include_router(agent.router, tags=["Agent"])
router.include_router(health.router, tags=["Health"])
router.include_router(legacy.router, tags=["Legacy"])

# Add a simple chat endpoint for v1 compatibility
from fastapi import HTTPException
from api.schemas.requests import ChatRequest
from api.schemas.responses import ChatResponse
from core.agent_core import LLMAgent

_agent: LLMAgent = None

def get_agent() -> LLMAgent:
    global _agent
    if _agent is None:
        _agent = LLMAgent()
    return _agent

@router.post("/chat", response_model=ChatResponse)
async def chat_v1(request: ChatRequest):
    """V1 chat endpoint for backward compatibility"""
    try:
        agent = get_agent()
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