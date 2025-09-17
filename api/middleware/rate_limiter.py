# api/middleware/rate_limiter.py
"""
Rate limiting middleware to prevent abuse and ensure fair usage
"""
import time
import logging
from fastapi import Request, HTTPException
from typing import Dict, Optional
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

class RateLimiterMiddleware:
    """Rate limiting middleware with sliding window algorithm"""
    
    def __init__(self, 
                 requests_per_minute: int = 60,
                 requests_per_hour: int = 1000,
                 burst_limit: int = 10):
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        self.burst_limit = burst_limit
        
        # Track requests by IP
        self.minute_windows: Dict[str, deque] = defaultdict(deque)
        self.hour_windows: Dict[str, deque] = defaultdict(deque)
        self.burst_windows: Dict[str, deque] = defaultdict(deque)
        
        # Track blocked IPs
        self.blocked_ips: Dict[str, float] = {}
    
    async def __call__(self, request: Request, call_next):
        client_ip = self._get_client_ip(request)
        current_time = time.time()
        
        # Check if IP is temporarily blocked
        if client_ip in self.blocked_ips:
            if current_time < self.blocked_ips[client_ip]:
                raise HTTPException(
                    status_code=429,
                    detail={
                        "error": "rate_limit_exceeded",
                        "message": "IP temporarily blocked due to excessive requests",
                        "retry_after": int(self.blocked_ips[client_ip] - current_time)
                    }
                )
            else:
                # Unblock IP
                del self.blocked_ips[client_ip]
        
        # Check rate limits
        if not self._check_rate_limits(client_ip, current_time):
            # Block IP for 5 minutes on repeated violations
            self.blocked_ips[client_ip] = current_time + 300
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "rate_limit_exceeded", 
                    "message": "Too many requests. Please try again later.",
                    "retry_after": 60
                }
            )
        
        # Record request
        self._record_request(client_ip, current_time)
        
        # Add rate limit headers
        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(self.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(
            max(0, self.requests_per_minute - len(self.minute_windows[client_ip]))
        )
        response.headers["X-RateLimit-Reset"] = str(int(current_time + 60))
        
        return response
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address from request"""
        # Check for forwarded IP first (common in production)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        return request.client.host if request.client else "unknown"
    
    def _check_rate_limits(self, client_ip: str, current_time: float) -> bool:
        """Check if request is within rate limits"""
        # Clean old entries
        self._cleanup_windows(client_ip, current_time)
        
        # Check burst limit (10 requests per 10 seconds)
        burst_window = self.burst_windows[client_ip]
        if len(burst_window) >= self.burst_limit:
            return False
        
        # Check per-minute limit
        minute_window = self.minute_windows[client_ip]
        if len(minute_window) >= self.requests_per_minute:
            return False
        
        # Check per-hour limit
        hour_window = self.hour_windows[client_ip]
        if len(hour_window) >= self.requests_per_hour:
            return False
        
        return True
    
    def _record_request(self, client_ip: str, current_time: float):
        """Record a request in all windows"""
        self.burst_windows[client_ip].append(current_time)
        self.minute_windows[client_ip].append(current_time)
        self.hour_windows[client_ip].append(current_time)
    
    def _cleanup_windows(self, client_ip: str, current_time: float):
        """Remove old entries from sliding windows"""
        # Clean burst window (10 seconds)
        burst_window = self.burst_windows[client_ip]
        while burst_window and current_time - burst_window[0] > 10:
            burst_window.popleft()
        
        # Clean minute window (60 seconds)
        minute_window = self.minute_windows[client_ip]
        while minute_window and current_time - minute_window[0] > 60:
            minute_window.popleft()
        
        # Clean hour window (3600 seconds)
        hour_window = self.hour_windows[client_ip]
        while hour_window and current_time - hour_window[0] > 3600:
            hour_window.popleft()
    
    def get_stats(self) -> Dict[str, int]:
        """Get rate limiting statistics"""
        return {
            "active_ips": len(self.minute_windows),
            "blocked_ips": len(self.blocked_ips),
            "total_windows": sum(len(w) for w in self.minute_windows.values())
        } 