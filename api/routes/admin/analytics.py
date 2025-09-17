# api/routes/admin/analytics.py
"""
Analytics endpoints
"""
from fastapi import APIRouter

router = APIRouter()

@router.get("/usage")
async def get_usage_analytics():
    """Get usage analytics"""
    return {
        "message": "Usage analytics endpoint", 
        "status": "available"
    } 