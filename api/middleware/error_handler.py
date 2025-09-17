# api/middleware/error_handler.py
"""
Centralized error handling middleware for consistent error responses
"""
import logging
import traceback
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError as PydanticValidationError
from typing import Dict, Any, Optional
import time

logger = logging.getLogger(__name__)

class AgentProcessingError(Exception):
    """Custom exception for agent processing errors"""
    def __init__(self, message: str, error_code: str = "AGENT_ERROR", details: Optional[Dict] = None):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(message)

class ToolExecutionError(Exception):
    """Custom exception for tool execution errors"""
    def __init__(self, tool_name: str, message: str, details: Optional[Dict] = None):
        self.tool_name = tool_name
        self.message = message
        self.details = details or {}
        super().__init__(f"Tool '{tool_name}' failed: {message}")

class ErrorHandlerMiddleware:
    """Centralized error handling middleware"""
    
    def __init__(self):
        self.error_count = 0
        self.error_types = {}
    
    async def __call__(self, request: Request, call_next):
        start_time = time.time()
        
        try:
            response = await call_next(request)
            return response
            
        except HTTPException:
            # Let FastAPI handle HTTP exceptions
            raise
            
        except (PydanticValidationError, RequestValidationError) as e:
            logger.warning(f"Validation error for {request.url}: {str(e)}")
            return JSONResponse(
                status_code=422,
                content={
                    "error": "validation_error",
                    "message": "Invalid input format",
                    "details": e.errors() if hasattr(e, 'errors') else str(e),
                    "timestamp": time.time(),
                    "request_id": getattr(request.state, 'request_id', None)
                }
            )
            
        except AgentProcessingError as e:
            logger.error(f"Agent processing error: {e.message}")
            self._track_error("agent_error")
            return JSONResponse(
                status_code=500,
                content={
                    "error": e.error_code,
                    "message": e.message,
                    "details": e.details,
                    "timestamp": time.time(),
                    "request_id": getattr(request.state, 'request_id', None)
                }
            )
            
        except ToolExecutionError as e:
            logger.error(f"Tool execution error: {e.tool_name} - {e.message}")
            self._track_error("tool_error")
            return JSONResponse(
                status_code=500,
                content={
                    "error": "tool_execution_error",
                    "message": f"Tool '{e.tool_name}' failed to execute",
                    "details": e.details,
                    "timestamp": time.time(),
                    "request_id": getattr(request.state, 'request_id', None)
                }
            )
            
        except Exception as e:
            # Log full traceback for debugging
            logger.exception(f"Unexpected error processing {request.url}")
            self._track_error("unexpected_error")
            
            # Don't expose internal errors in production
            return JSONResponse(
                status_code=500,
                content={
                    "error": "internal_server_error",
                    "message": "An unexpected error occurred while processing your request",
                    "timestamp": time.time(),
                    "request_id": getattr(request.state, 'request_id', None),
                    # Include error details only in debug mode
                    "details": str(e) if logger.level <= logging.DEBUG else None
                }
            )
    
    def _track_error(self, error_type: str):
        """Track error statistics"""
        self.error_count += 1
        self.error_types[error_type] = self.error_types.get(error_type, 0) + 1
    
    def get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics"""
        return {
            "total_errors": self.error_count,
            "error_types": self.error_types
        }

def handle_agent_errors(func):
    """Decorator for handling agent-specific errors"""
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            logger.exception(f"Error in {func.__name__}")
            raise AgentProcessingError(
                message=f"Failed to process request in {func.__name__}",
                details={"original_error": str(e)}
            )
    return wrapper 