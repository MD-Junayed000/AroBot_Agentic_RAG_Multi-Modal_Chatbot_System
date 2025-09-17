# api/schemas/errors.py
"""
Error response schemas for consistent error handling
"""
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List

class ErrorResponse(BaseModel):
    """Standard error response schema"""
    
    error: str = Field(..., description="Error code")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: float = Field(..., description="Error timestamp")
    request_id: Optional[str] = Field(None, description="Request correlation ID")
    
    class Config:
        schema_extra = {
            "example": {
                "error": "validation_error",
                "message": "Invalid input format",
                "details": {
                    "field": "message",
                    "reason": "Message too short"
                },
                "timestamp": 1640995200.0,
                "request_id": "550e8400-e29b-41d4-a716-446655440000"
            }
        }

class ValidationErrorResponse(BaseModel):
    """Validation error response schema"""
    
    error: str = Field("validation_error", description="Error code")
    message: str = Field("Invalid input format", description="Error message")
    details: List[Dict[str, Any]] = Field(..., description="Validation error details")
    timestamp: float = Field(..., description="Error timestamp")
    request_id: Optional[str] = Field(None, description="Request correlation ID")
    
    class Config:
        schema_extra = {
            "example": {
                "error": "validation_error",
                "message": "Invalid input format",
                "details": [
                    {
                        "loc": ["message"],
                        "msg": "field required",
                        "type": "value_error.missing"
                    }
                ],
                "timestamp": 1640995200.0,
                "request_id": "550e8400-e29b-41d4-a716-446655440000"
            }
        }

class ToolExecutionErrorResponse(BaseModel):
    """Tool execution error response schema"""
    
    error: str = Field("tool_execution_error", description="Error code")
    message: str = Field(..., description="Error message")
    tool_name: str = Field(..., description="Name of failed tool")
    details: Optional[Dict[str, Any]] = Field(None, description="Error details")
    timestamp: float = Field(..., description="Error timestamp")
    request_id: Optional[str] = Field(None, description="Request correlation ID")
    
    class Config:
        schema_extra = {
            "example": {
                "error": "tool_execution_error",
                "message": "Tool 'analyze_image' failed to execute",
                "tool_name": "analyze_image",
                "details": {
                    "reason": "Image format not supported",
                    "supported_formats": ["jpg", "png", "webp"]
                },
                "timestamp": 1640995200.0,
                "request_id": "550e8400-e29b-41d4-a716-446655440000"
            }
        }

class RateLimitErrorResponse(BaseModel):
    """Rate limit error response schema"""
    
    error: str = Field("rate_limit_exceeded", description="Error code")
    message: str = Field("Too many requests", description="Error message")
    retry_after: int = Field(..., description="Seconds to wait before retry")
    limit: int = Field(..., description="Rate limit")
    remaining: int = Field(0, description="Remaining requests")
    timestamp: float = Field(..., description="Error timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "error": "rate_limit_exceeded",
                "message": "Too many requests. Please try again later.",
                "retry_after": 60,
                "limit": 100,
                "remaining": 0,
                "timestamp": 1640995200.0
            }
        } 