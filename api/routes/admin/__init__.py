# api/routes/admin/__init__.py
"""
Admin endpoints for system management
"""
from fastapi import APIRouter
from . import system, analytics

router = APIRouter()

# Include sub-routers
router.include_router(system.router, tags=["System Management"])
router.include_router(analytics.router, tags=["Analytics"]) 