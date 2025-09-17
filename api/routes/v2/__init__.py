# api/routes/v2/__init__.py
"""
API v2 - Latest features and enhancements
"""
from fastapi import APIRouter
from . import chat, multimodal, tools

router = APIRouter()

# Include sub-routers
router.include_router(chat.router, tags=["Enhanced Chat"])
router.include_router(multimodal.router, tags=["Multi-Modal"])
router.include_router(tools.router, tags=["Tool Management"]) 