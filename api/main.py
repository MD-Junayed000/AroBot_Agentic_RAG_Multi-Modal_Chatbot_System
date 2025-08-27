"""
FastAPI main application for AroBot Multi-Modal Medical Chatbot
"""
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from .routes import router
from config.env_config import DEBUG, APP_HOST, APP_PORT
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="AroBot - Multi-Modal Medical Chatbot",
    description="AI-powered medical assistant with prescription OCR and RAG capabilities",
    version="1.0.0",
    debug=DEBUG
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

# Include routes  
app.include_router(router, prefix="/api/v1")

# Add a redirect from root to chat
@app.get("/")
async def redirect_to_chat():
    """Redirect root to chat interface"""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/chat")

# Add root-level chat route
@app.get("/chat", response_class=HTMLResponse)
async def root_chat_interface(request: Request):
    """Serve the enhanced chat interface at root level"""
    return templates.TemplateResponse("chat_enhanced.html", {"request": request})

# API info endpoint
@app.get("/api")
async def api_info():
    """API information endpoint"""
    return {
        "name": "AroBot Multi-Modal Medical Chatbot API",
        "version": "1.0.0",
        "description": "AI-powered medical assistant with prescription OCR and RAG capabilities",
        "endpoints": {
            "health": "/health",
            "chat": "/chat",
            "api_chat": "/api/v1/chat",
            "prescription": "/api/v1/prescription",
            "search": "/api/v1/search"
        }
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "AroBot Medical Chatbot",
        "version": "1.0.0"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=APP_HOST, port=APP_PORT, reload=DEBUG)
