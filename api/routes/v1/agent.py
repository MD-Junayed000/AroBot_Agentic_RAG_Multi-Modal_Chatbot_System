# api/routes/v1/agent.py
"""
Enhanced Agent Router with validation and error handling
"""
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from typing import Optional
import time
import io
from PIL import Image, UnidentifiedImageError
import json
import logging

logger = logging.getLogger(__name__)

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
async def unified_agent_endpoint(
    # Text input
    message: Optional[str] = Form(None),
    # File inputs
    file: Optional[UploadFile] = File(None),
    # Session management
    session_id: Optional[str] = Form(None),
    # Enhanced image classification parameters
    image_type: Optional[str] = Form(None),
    confidence_hint: Optional[float] = Form(None),
    context_info: Optional[str] = Form(None),
    # Optional parameters
    file_type: Optional[str] = Form(None),
    use_web_search: bool = Form(False),
    max_tokens: int = Form(500),
    temperature: float = Form(0.2),
    # Dependency injection
    agent: LLMAgent = Depends(get_agent)
):
    """
    Enhanced Unified Agent Endpoint with Advanced Image Classification
    
    Supports:
    - Intelligent image classification with confidence scoring
    - Context-aware processing based on user intent
    - Multiple image categories (prescription, medicine package, lab results, etc.)
    - Enhanced metadata for better classification accuracy
    """
    try:
        # Validate input
        if not message and not file:
            raise HTTPException(
                status_code=400, 
                detail="Either message or file must be provided"
            )
        
        # Initialize variables
        image_data = None
        pdf_data = None
        enhanced_context = {}
        
        # Parse context information if provided
        if context_info:
            try:
                enhanced_context = json.loads(context_info)
            except json.JSONDecodeError:
                logger.warning("Invalid context_info JSON provided")
                enhanced_context = {}
        
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
                    
                    # Add image classification hints to enhanced context
                    enhanced_context.update({
                        'image_type_hint': image_type,
                        'confidence_hint': confidence_hint,
                        'file_size': len(file_content),
                        'content_type': file.content_type,
                        'filename': file.filename
                    })
                    
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
        
        # Create enhanced message with context if available
        enhanced_message = message
        if enhanced_context and message:
            # Add context hints to help with classification
            context_hints = []
            if enhanced_context.get('image_type_hint'):
                context_hints.append(f"[Image type hint: {enhanced_context['image_type_hint']}]")
            if enhanced_context.get('confidence_hint'):
                context_hints.append(f"[Confidence: {enhanced_context['confidence_hint']:.1f}]")
            
            if context_hints:
                enhanced_message = f"{message} {' '.join(context_hints)}"
        
        # Process with enhanced agent
        result = await agent.process_request(
            text_input=enhanced_message,
            image_data=image_data,
            pdf_data=pdf_data,
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
        
        # Add classification information to metadata if available
        metadata = {}
        if hasattr(result, 'classification'):
            metadata['classification'] = result.classification
        if hasattr(result, 'confidence'):
            metadata['confidence'] = result.confidence
        if hasattr(result, 'reasoning'):
            metadata['reasoning'] = result.reasoning
        
        # Return properly structured agent response
        return AgentResponse(
            response=result.response,
            tools_used=result.tools_used,
            session_id=result.session_id,
            sources=source_info,
            status=result.status,
            metadata=metadata if metadata else None
        )
        
    except HTTPException:
        raise
    except AgentProcessingError as e:
        logger.error(f"Agent processing error: {e.message}")
        raise HTTPException(
            status_code=500,
            detail=f"Agent processing failed: {e.message}"
        )
    except ToolExecutionError as e:
        logger.error(f"Tool execution error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Tool execution failed: {str(e)}"
        )
    except Exception as e:
        logger.exception("Unexpected error in unified agent endpoint")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
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