# api/schemas/requests.py
"""
Request validation schemas using Pydantic
"""
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
import re

class AgentRequest(BaseModel):
    """Unified agent request schema with strict validation"""
    
    message: Optional[str] = Field(
        None, 
        max_length=10000,
        description="Text message or query"
    )
    
    session_id: Optional[str] = Field(
        None,
        pattern=r'^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$',
        description="UUID session identifier"
    )
    
    file_type: Optional[str] = Field(
        None,
        pattern=r'^(image|pdf|document)$',
        description="Type of file being uploaded"
    )
    
    # File data will be handled separately in multipart forms
    image_data: Optional[bytes] = Field(None, description="Image file data")
    pdf_data: Optional[bytes] = Field(None, description="PDF file data")
    
    # Optional parameters
    use_web_search: bool = Field(False, description="Enable web search")
    max_tokens: int = Field(500, ge=50, le=2000, description="Maximum response tokens")
    temperature: float = Field(0.2, ge=0.0, le=1.0, description="Response creativity")
    
    @validator('message')
    def validate_message(cls, v):
        if v is not None:
            # Remove excessive whitespace
            v = ' '.join(v.split())
            
            # Check for minimum content
            if len(v.strip()) < 2:
                raise ValueError('Message must be at least 2 characters long')
                
            # Basic content filtering
            forbidden_patterns = [
                r'<script.*?>.*?</script>',  # Script tags
                r'javascript:',              # JavaScript URLs
                r'on\w+\s*=',               # Event handlers
            ]
            
            for pattern in forbidden_patterns:
                if re.search(pattern, v, re.IGNORECASE):
                    raise ValueError('Message contains forbidden content')
        
        return v
    
    @validator('session_id')
    def validate_session_id(cls, v):
        if v is not None and not re.match(
            r'^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$', 
            v
        ):
            raise ValueError('Invalid session ID format. Must be UUID.')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "message": "What is the dosage for paracetamol?",
                "session_id": "550e8400-e29b-41d4-a716-446655440000",
                "use_web_search": False,
                "max_tokens": 500,
                "temperature": 0.2
            }
        }

class ChatRequest(BaseModel):
    """Simple chat request schema"""
    
    message: str = Field(
        ..., 
        min_length=1,
        max_length=5000,
        description="Chat message"
    )
    
    session_id: Optional[str] = Field(
        None,
        description="Session identifier"
    )
    
    context_window: int = Field(
        10,
        ge=1,
        le=50,
        description="Number of previous messages to include"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "message": "Hello, I need help with a prescription",
                "session_id": "550e8400-e29b-41d4-a716-446655440000",
                "context_window": 10
            }
        }

class ToolRequest(BaseModel):
    """Tool execution request schema"""
    
    tool_name: str = Field(
        ...,
        pattern=r'^[a-z_]+$',
        description="Name of tool to execute"
    )
    
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Tool parameters"
    )
    
    session_id: Optional[str] = Field(
        None,
        description="Session identifier"
    )
    
    @validator('tool_name')
    def validate_tool_name(cls, v):
        allowed_tools = [
            'analyze_text',
            'analyze_image', 
            'analyze_pdf',
            'get_medicine_info',
            'web_search',
            'access_memory',
        ]
        
        if v not in allowed_tools:
            raise ValueError(f'Tool {v} not allowed. Allowed tools: {allowed_tools}')
        
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "tool_name": "analyze_text",
                "parameters": {
                    "query": "What is diabetes?",
                    "use_rag": True
                },
                "session_id": "550e8400-e29b-41d4-a716-446655440000"
            }
        }

class FileUploadRequest(BaseModel):
    """File upload validation"""
    
    file_type: str = Field(
        ...,
        pattern=r'^(image|pdf)$',
        description="File type"
    )
    
    file_size: int = Field(
        ...,
        ge=1,
        le=50 * 1024 * 1024,  # 50MB max
        description="File size in bytes"
    )
    
    content_type: str = Field(
        ...,
        description="MIME content type"
    )
    
    @validator('content_type')
    def validate_content_type(cls, v, values):
        file_type = values.get('file_type')
        
        valid_types = {
            'image': ['image/jpeg', 'image/png', 'image/jpg', 'image/webp'],
            'pdf': ['application/pdf']
        }
        
        if file_type and v not in valid_types.get(file_type, []):
            raise ValueError(f'Invalid content type {v} for file type {file_type}')
        
        return v 