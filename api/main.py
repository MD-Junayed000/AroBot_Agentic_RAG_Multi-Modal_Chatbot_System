# api/main.py
"""
FastAPI main application for AroBot Multi-Modal Medical Chatbot
Enhanced with middleware, proper routing, and error handling
"""
import logging
import time
import psutil
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from config.env_config import DEBUG, APP_HOST, APP_PORT, TEMPLATES_DIR, STATIC_DIR
from utils.ocr_pipeline import warmup_ocr
from core.langsmith_config import initialize_langsmith

# Import middleware
from .middleware import ErrorHandlerMiddleware, RateLimiterMiddleware, RequestLoggerMiddleware

# Import new router structure
from .routes import v1, v2, admin

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize middleware instances
error_handler = ErrorHandlerMiddleware()
rate_limiter = RateLimiterMiddleware(
    requests_per_minute=60,
    requests_per_hour=1000,
    burst_limit=10
)
request_logger = RequestLoggerMiddleware()

# Application startup time for uptime calculation
app_start_time = time.time()

app = FastAPI(
    title="AroBot - Enhanced LLM-as-Agent Medical Assistant",
    description="Intelligent agent with advanced middleware, validation, and error handling. Supports text, images, PDFs with unified endpoint.",
    version="2.0.0",
    debug=DEBUG,
)

# Add middleware in correct order (last added = first executed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add custom middleware
app.middleware("http")(error_handler)
app.middleware("http")(rate_limiter)
app.middleware("http")(request_logger)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Clean router architecture with proper versioning
app.include_router(v1.router, prefix="/api/v1", tags=["API v1 - Stable"])
app.include_router(v2.router, prefix="/api/v2", tags=["API v2 - Latest"]) 
app.include_router(admin.router, prefix="/admin", tags=["Admin"])

# Global error tracking
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception(f"Unhandled exception: {exc}")
    return error_handler._create_error_response(exc, getattr(request.state, 'request_id', None))

# Initialize services on startup
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    # Initialize LangSmith tracing
    langsmith_initialized = initialize_langsmith()
    if langsmith_initialized:
        logger.info("✅ LangSmith tracing initialized successfully")
    else:
        logger.warning("⚠️ LangSmith tracing not available")
    
    # Warm up OCR models
    warmup_ocr(lang="en")
    
    # Ensure a favicon exists to prevent 404s from browsers requesting /favicon.ico
    try:
        favicon_path = STATIC_DIR / "favicon.ico"
        if not favicon_path.exists():
            try:
                # Generate a minimal favicon using Pillow (already in requirements)
                from PIL import Image, ImageDraw

                img = Image.new("RGBA", (64, 64), (26, 26, 32, 255))  # dark background
                draw = ImageDraw.Draw(img)
                # simple green cross
                draw.rectangle([(30, 12), (34, 52)], fill=(16, 185, 129, 255))
                draw.rectangle([(12, 30), (52, 34)], fill=(16, 185, 129, 255))
                img.save(favicon_path, format="ICO", sizes=[(16, 16), (32, 32)])
            except Exception:
                # If Pillow is unavailable at runtime for any reason, skip silently
                pass
    except Exception:
        # Never block startup on favicon generation
        pass

@app.get("/")
async def root():
    return RedirectResponse(url="/chat")

@app.get("/chat", response_class=HTMLResponse)
async def root_chat_interface(request: Request):
    return templates.TemplateResponse("chat_enhanced.html", {"request": request})

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    fp = STATIC_DIR / "favicon.ico"
    if fp.exists():
        return FileResponse(str(fp), media_type="image/x-icon")
    # If somehow missing, respond with 204 to avoid noisy 404 in logs
    from fastapi import Response
    return Response(status_code=204)

@app.get("/api")
async def api_info():
    return {
        "name": "AroBot Multi-Modal Medical Chatbot API",
        "version": "2.0.0",
        "architecture": "LLM-as-Agent",
        "description": "LLM agent automatically selects tools based on input",
        "endpoints": {
            "health": "/health",
            "chat_ui": "/chat",
            # NEW AGENT-BASED ENDPOINTS
            "unified_agent": "/api/v1/agent",
            "agent_tools": "/api/v1/agent/tools",
            "agent_tools_by_category": "/api/v1/agent/tools?category={category}",
            "agent_categories": "/api/v1/agent/tools/categories", 
            "agent_explain": "/api/v1/agent/explain",
            # LEGACY ENDPOINTS (for backward compatibility)
            "legacy_chat": "/api/v1/chat",
            "legacy_prescription": "/api/v1/prescription/upload",
            "legacy_pdf_analyze": "/api/v1/pdf/analyze",
            "legacy_image_analyze": "/api/v1/image/analyze",
        },
        "recommended_usage": "Use /api/v1/agent for all new integrations"
    }

@app.get("/health")
async def health_check():
    """Enhanced health check with system metrics"""
    try:
        # System metrics
        uptime = time.time() - app_start_time
        memory = psutil.virtual_memory()
        
        # Component health checks
        components = {
            "middleware": "healthy",
            "database": "healthy",  # TODO: Add actual DB health check
            "llm_service": "healthy",  # TODO: Add Ollama health check
            "vector_store": "healthy"  # TODO: Add Pinecone health check
        }
        
        return {
            "status": "healthy",
            "service": "AroBot",
            "version": "2.0.0",
            "architecture": "LLM-as-Agent",
            "uptime": uptime,
            "memory_usage": memory.percent,
            "components": components,
            "middleware_stats": {
                "error_handler": error_handler.get_error_stats(),
                "rate_limiter": rate_limiter.get_stats(),
                "request_logger": request_logger.get_stats()
            }
        }
    except Exception as e:
        logger.exception("Health check failed")
        return {
            "status": "degraded",
            "service": "AroBot", 
            "version": "2.0.0",
            "error": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=APP_HOST, port=APP_PORT, reload=DEBUG)
