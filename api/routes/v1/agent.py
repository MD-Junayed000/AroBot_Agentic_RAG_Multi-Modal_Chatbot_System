# api/routes/v1/agent.py
"""
Enhanced Agent Router with validation and error handling
"""
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from typing import Optional
import time
import io
from PIL import Image, UnidentifiedImageError

from api.schemas.responses import AgentResponse, ToolListResponse, ToolInfo
from api.middleware.error_handler import handle_agent_errors, AgentProcessingError, ToolExecutionError
from core.agent_core import LLMAgent

router = APIRouter()

# Single agent instance with proper initialization
_agent: Optional[LLMAgent] = None

def get_agent() -> LLMAgent:
    """Get or create the LLM agent instance"""
    global _agent
    if _agent is None:
        _agent = LLMAgent()
    return _agent

@router.post("/agent", response_model=AgentResponse)
@handle_agent_errors
async def unified_agent_endpoint(
    # Text input
    message: Optional[str] = Form(None),
    # File inputs
    file: Optional[UploadFile] = File(None),
    # Session management
    session_id: Optional[str] = Form(None),
    # Optional parameters
    file_type: Optional[str] = Form(None),
    use_web_search: bool = Form(False),
    max_tokens: int = Form(500),
    temperature: float = Form(0.2),
    # Dependency injection
    agent: LLMAgent = Depends(get_agent)
):
    """
    Enhanced Unified Agent Endpoint with validation and error handling
    
    Accepts:
    - Text message (chat, questions, queries) 
    - Image files (prescriptions, diagrams, photos)
    - PDF files (documents, papers, manuals)
    - Any combination of the above
    
    The LLM agent analyzes input and automatically selects appropriate tools.
    """
    start_time = time.time()
    
    try:
        # Input validation
        if not message and not file:
            raise HTTPException(
                status_code=400, 
                detail="Either message or file must be provided"
            )
        
        # Validate message length
        if message and len(message.strip()) > 10000:
            raise HTTPException(
                status_code=400,
                detail="Message too long. Maximum 10,000 characters allowed."
            )
        
        # Validate parameters
        if max_tokens < 50 or max_tokens > 2000:
            raise HTTPException(
                status_code=400,
                detail="max_tokens must be between 50 and 2000"
            )
        
        if temperature < 0.0 or temperature > 1.0:
            raise HTTPException(
                status_code=400,
                detail="temperature must be between 0.0 and 1.0"
            )
        
        # Process file uploads
        image_data = None
        pdf_data = None
        
        if file:
            # Validate file size (50MB limit)
            file_content = await file.read()
            if len(file_content) > 50 * 1024 * 1024:
                raise HTTPException(
                    status_code=413,
                    detail="File too large. Maximum 50MB allowed."
                )
            
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
                            raise HTTPException(
                                status_code=400, 
                                detail="Unsupported file type. Only images and PDFs are supported."
                            )
            
            # Validate and assign file data
            if file_type == "image":
                # Validate image
                try:
                    Image.open(io.BytesIO(file_content))
                    image_data = file_content
                except UnidentifiedImageError:
                    raise HTTPException(
                        status_code=400, 
                        detail="Invalid image file. Supported formats: JPEG, PNG, WebP"
                    )
            elif file_type == "pdf":
                # Validate PDF
                if not file_content.startswith(b'%PDF'):
                    raise HTTPException(
                        status_code=400,
                        detail="Invalid PDF file"
                    )
                pdf_data = file_content
        
        # Process with enhanced agent
        result = await agent.process_request(
            text_input=message,
            image_data=image_data,
            pdf_data=pdf_data,
            session_id=session_id
        )
        
        # Calculate response time
        response_time = time.time() - start_time
        
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
        
        # Return enhanced response
        return AgentResponse(
            response=result.response,
            session_id=result.session_id or session_id or "unknown",
            tools_used=result.tools_used,
            sources=source_info,
            status=result.status,
            confidence=getattr(result, 'confidence', None),
            response_time=response_time,
            token_count=getattr(result, 'token_count', None),
            agent_type="llm_agent",
            model_version="2.0.0"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise AgentProcessingError(
            message="Failed to process agent request",
            details={"original_error": str(e)}
        )

@router.get("/agent/tools", response_model=ToolListResponse)
async def list_available_tools(
    category: Optional[str] = None,
    agent: LLMAgent = Depends(get_agent)
):
    """List all available tools with optional category filtering"""
    try:
        tools_info = []
        categories = set()
        
        for tool in agent.registry.get_all_tools():
            if category is None or tool.category == category:
                tools_info.append(ToolInfo(
                    name=tool.name,
                    description=tool.description,
                    category=tool.category,
                    priority=tool.priority
                ))
            categories.add(tool.category)
        
        return ToolListResponse(
            tools=tools_info,
            categories=sorted(list(categories)),
            total_count=len(tools_info)
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list tools: {str(e)}"
        )

@router.get("/agent/tools/categories")
async def list_tool_categories(agent: LLMAgent = Depends(get_agent)):
    """List all available tool categories"""
    try:
        categories = set()
        for tool in agent.registry.get_all_tools():
            categories.add(tool.category)
        
        return {
            "categories": sorted(list(categories)),
            "count": len(categories)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list categories: {str(e)}"
        )

@router.post("/agent/explain")
async def explain_tool_selection(
    message: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    agent: LLMAgent = Depends(get_agent)
):
    """Explain which tools would be selected without executing them"""
    try:
        if not message and not file:
            raise HTTPException(
                status_code=400,
                detail="Either message or file must be provided"
            )
        
        # Prepare input description
        input_description = agent._describe_input(
            message, 
            await file.read() if file else None,
            None
        )
        
        # Get tool selection explanation
        explanation = await agent.explain_request(
            text_input=message,
            image_data=await file.read() if file else None
        )
        
        return {
            "input_analysis": input_description,
            "selected_tools": explanation.get("selected_tools", []),
            "reasoning": explanation.get("reasoning", ""),
            "confidence": explanation.get("confidence", 0.0)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to explain tool selection: {str(e)}"
        ) 