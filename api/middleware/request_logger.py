# api/middleware/request_logger.py
"""
Structured request logging middleware for observability
"""
import time
import uuid
import logging
import json
from fastapi import Request
from typing import Dict, Any

logger = logging.getLogger(__name__)

class RequestLoggerMiddleware:
    """Structured request logging with correlation IDs"""
    
    def __init__(self, log_body: bool = False):
        self.log_body = log_body
        self.request_count = 0
        self.total_response_time = 0.0
    
    async def __call__(self, request: Request, call_next):
        # Generate correlation ID
        correlation_id = str(uuid.uuid4())
        request.state.request_id = correlation_id
        
        start_time = time.time()
        self.request_count += 1
        
        # Log request
        request_log = self._create_request_log(request, correlation_id)
        logger.info("Request started", extra={"request_data": request_log})
        
        try:
            response = await call_next(request)
            
            # Calculate response time
            response_time = time.time() - start_time
            self.total_response_time += response_time
            
            # Log response
            response_log = self._create_response_log(
                request, response, correlation_id, response_time
            )
            logger.info("Request completed", extra={"response_data": response_log})
            
            # Add correlation ID to response headers
            response.headers["X-Correlation-ID"] = correlation_id
            response.headers["X-Response-Time"] = f"{response_time:.3f}s"
            
            return response
            
        except Exception as e:
            response_time = time.time() - start_time
            
            # Log error
            error_log = self._create_error_log(
                request, e, correlation_id, response_time
            )
            logger.error("Request failed", extra={"error_data": error_log})
            
            raise
    
    def _create_request_log(self, request: Request, correlation_id: str) -> Dict[str, Any]:
        """Create structured request log"""
        return {
            "correlation_id": correlation_id,
            "method": request.method,
            "url": str(request.url),
            "path": request.url.path,
            "query_params": dict(request.query_params),
            "headers": dict(request.headers),
            "client_ip": self._get_client_ip(request),
            "user_agent": request.headers.get("user-agent"),
            "timestamp": time.time(),
            "request_size": request.headers.get("content-length", "unknown")
        }
    
    def _create_response_log(self, request: Request, response, 
                           correlation_id: str, response_time: float) -> Dict[str, Any]:
        """Create structured response log"""
        return {
            "correlation_id": correlation_id,
            "status_code": response.status_code,
            "response_time": response_time,
            "response_size": response.headers.get("content-length", "unknown"),
            "path": request.url.path,
            "method": request.method,
            "success": 200 <= response.status_code < 400
        }
    
    def _create_error_log(self, request: Request, error: Exception,
                         correlation_id: str, response_time: float) -> Dict[str, Any]:
        """Create structured error log"""
        return {
            "correlation_id": correlation_id,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "path": request.url.path,
            "method": request.method,
            "response_time": response_time,
            "client_ip": self._get_client_ip(request)
        }
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address"""
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        return request.client.host if request.client else "unknown"
    
    def get_stats(self) -> Dict[str, Any]:
        """Get request statistics"""
        avg_response_time = (
            self.total_response_time / self.request_count 
            if self.request_count > 0 else 0
        )
        
        return {
            "total_requests": self.request_count,
            "average_response_time": avg_response_time,
            "total_response_time": self.total_response_time
        } 