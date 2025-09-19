# api/schemas/responses.py
"""
Response schemas for consistent API responses
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Union
from enum import Enum

class ResponseStatus(str, Enum):
    SUCCESS = "success"
    ERROR = "error"
    PARTIAL = "partial"

class ToolInfo(BaseModel):
    """Tool information schema"""
    name: str
    description: str
    category: str
    priority: int
    execution_time: Optional[float] = None
    confidence: Optional[float] = None

class SourceInfo(BaseModel):
    """Source attribution schema"""
    tools: List[ToolInfo] = Field(default_factory=list)
    llm_agent: bool = True
    knowledge_base: int = 0
    web_search: int = 0
    cached: bool = False

class AgentResponse(BaseModel):
    """Unified agent response schema"""
    
    response: str = Field(..., description="Generated response text")
    
    session_id: str = Field(..., description="Session identifier")
    
    tools_used: List[str] = Field(
        default_factory=list,
        description="Names of tools that were executed"
    )
    
    sources: SourceInfo = Field(
        default_factory=SourceInfo,
        description="Source attribution information"
    )
    
    status: ResponseStatus = Field(
        ResponseStatus.SUCCESS,
        description="Response status"
    )
    
    confidence: Optional[float] = Field(
        None,
        description="Overall confidence score (0-1)"
    )
    
    metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional metadata including classification info, reasoning, etc."
    )
    
    response_time: Optional[float] = Field(
        None,
        description="Processing time in seconds"
    )
    
    token_count: Optional[int] = Field(
        None,
        description="Number of tokens in response"
    )
    
    # Metadata
    agent_type: str = Field("llm_agent", description="Type of agent used")
    model_version: str = Field("2.0.0", description="Model version")
    
    class Config:
        schema_extra = {
            "example": {
                "response": "Paracetamol is typically dosed at 500-1000mg every 4-6 hours, with a maximum of 4000mg per day.",
                "session_id": "550e8400-e29b-41d4-a716-446655440000",
                "tools_used": ["analyze_text", "get_medicine_info"],
                "sources": {
                    "tools": [
                        {
                            "name": "analyze_text",
                            "description": "Medical text analysis",
                            "category": "medical",
                            "priority": 5,
                            "execution_time": 1.2,
                            "confidence": 0.95
                        }
                    ],
                    "llm_agent": True,
                    "knowledge_base": 2,
                    "web_search": 0,
                    "cached": False
                },
                "status": "success",
                "confidence": 0.92,
                "response_time": 2.1,
                "token_count": 45,
                "agent_type": "llm_agent",
                "model_version": "2.0.0"
            }
        }

class ChatResponse(BaseModel):
    """Simple chat response schema"""
    
    response: str = Field(..., description="Chat response")
    session_id: str = Field(..., description="Session identifier")
    status: ResponseStatus = Field(ResponseStatus.SUCCESS)
    response_time: Optional[float] = None
    
    class Config:
        schema_extra = {
            "example": {
                "response": "Hello! I'm here to help with your medical questions.",
                "session_id": "550e8400-e29b-41d4-a716-446655440000",
                "status": "success",
                "response_time": 0.5
            }
        }

class ToolResponse(BaseModel):
    """Tool execution response schema"""
    
    tool_name: str = Field(..., description="Name of executed tool")
    result: Dict[str, Any] = Field(..., description="Tool execution result")
    execution_time: float = Field(..., description="Execution time in seconds")
    status: ResponseStatus = Field(ResponseStatus.SUCCESS)
    error_message: Optional[str] = None
    
    class Config:
        schema_extra = {
            "example": {
                "tool_name": "analyze_text",
                "result": {
                    "response": "Analysis complete",
                    "confidence": 0.95
                },
                "execution_time": 1.2,
                "status": "success"
            }
        }

class HealthResponse(BaseModel):
    """Health check response schema"""
    
    status: str = Field("healthy", description="Service status")
    service: str = Field("AroBot", description="Service name")
    version: str = Field("2.0.0", description="Service version")
    architecture: str = Field("LLM-as-Agent", description="Architecture type")
    
    # System metrics
    uptime: Optional[float] = Field(None, description="Uptime in seconds")
    memory_usage: Optional[float] = Field(None, description="Memory usage percentage")
    active_sessions: Optional[int] = Field(None, description="Number of active sessions")
    
    # Component health
    components: Optional[Dict[str, str]] = Field(
        None,
        description="Health status of system components"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "service": "AroBot",
                "version": "2.0.0",
                "architecture": "LLM-as-Agent",
                "uptime": 3600.5,
                "memory_usage": 45.2,
                "active_sessions": 12,
                "components": {
                    "database": "healthy",
                    "llm_service": "healthy",
                    "vector_store": "healthy"
                }
            }
        }

class ToolListResponse(BaseModel):
    """Tool list response schema"""
    
    tools: List[ToolInfo] = Field(..., description="Available tools")
    categories: List[str] = Field(..., description="Tool categories")
    total_count: int = Field(..., description="Total number of tools")
    
    class Config:
        schema_extra = {
            "example": {
                "tools": [
                    {
                        "name": "analyze_text",
                        "description": "Analyze text queries and medical questions",
                        "category": "medical",
                        "priority": 5
                    }
                ],
                "categories": ["medical", "document", "utility"],
                "total_count": 7
            }
        } 