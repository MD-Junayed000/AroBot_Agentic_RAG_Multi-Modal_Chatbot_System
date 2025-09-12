# api/main.py
"""
FastAPI main application for AroBot Multi-Modal Medical Chatbot
"""
import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from config.env_config import DEBUG, APP_HOST, APP_PORT, TEMPLATES_DIR, STATIC_DIR
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
