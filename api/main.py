# api/main.py
"""
FastAPI main application for AroBot Multi-Modal Medical Chatbot
"""
import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from config.env_config import DEBUG, APP_HOST, APP_PORT, TEMPLATES_DIR
from utils.ocr_pipeline import warmup_ocr

# existing API routes
from .core_routes import router as core_router
# new image router
from .routes.image import router as image_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AroBot - Multi-Modal Medical Chatbot",
    description="AI-powered medical assistant with prescription OCR, RAG, and vision",
    version="1.1.0",
    debug=DEBUG,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Routers
app.include_router(core_router, prefix="/api/v1")
app.include_router(image_router)  # exposes /api/v1/image/analyze

# Warm up PaddleOCR models in the background
@app.on_event("startup")
async def _warmup():
    warmup_ocr(lang="en")

@app.get("/")
async def root():
    return RedirectResponse(url="/chat")

@app.get("/chat", response_class=HTMLResponse)
async def root_chat_interface(request: Request):
    return templates.TemplateResponse("chat_enhanced.html", {"request": request})

@app.get("/api")
async def api_info():
    return {
        "name": "AroBot Multi-Modal Medical Chatbot API",
        "version": "1.1.0",
        "endpoints": {
            "health": "/health",
            "chat_ui": "/chat",
            "api_chat": "/api/v1/chat",
            "prescription": "/api/v1/prescription/upload",
            "pdf_analyze": "/api/v1/pdf/analyze",
            "pdf_upload_to_kb": "/api/v1/pdf/upload",
            "weather": "/api/v1/weather",
            "image_analyze": "/api/v1/image/analyze",
        },
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "AroBot", "version": "1.1.0"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=APP_HOST, port=APP_PORT, reload=DEBUG)
