"""
Simple environment configuration for AroBot Multi-Modal Chatbot System
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()  # load .env early

def _get(name: str, default=None, *, strip: bool = True):
    v = os.getenv(name, default)
    if strip and isinstance(v, str):
        v = v.strip()
    return v

def _get_bool(name: str, default: str = "false") -> bool:
    return _get(name, default).lower() in {"1", "true", "yes", "on"}

# ---------- API Keys ----------
PINECONE_API_KEY = _get("PINECONE_API_KEY")
LANGSMITH_API_KEY = _get("LANGSMITH_API_KEY")

if not PINECONE_API_KEY:
    print("⚠️  Warning: PINECONE_API_KEY not set in .env")
if not LANGSMITH_API_KEY:
    print("⚠️  Warning: LANGSMITH_API_KEY not set in .env")

# ---------- Pinecone ----------
PINECONE_CLOUD = _get("PINECONE_CLOUD", "aws")
PINECONE_REGION = _get("PINECONE_REGION", "us-east-1")
# Removed PINECONE_PDF_INDEX — PDFs use ad-hoc indexes created from UI
PINECONE_MEDICINE_INDEX = _get("PINECONE_MEDICINE_INDEX", "arobot-medicine-data")
PINECONE_BD_PHARMACY_INDEX = _get("PINECONE_BD_PHARMACY_INDEX", "arobot-bd-pharmacy")
PINECONE_IMAGE_INDEX = _get("PINECONE_IMAGE_INDEX", "arobot-clip")
EMBEDDING_DIMENSION = int(_get("EMBEDDING_DIMENSION", "384"))

# Pinecone hardening knobs
PINECONE_QUERY_TIMEOUT_S = float(os.getenv("PINECONE_QUERY_TIMEOUT_S", "3.5"))  # fail fast
PINECONE_ENABLE = os.getenv("PINECONE_ENABLE", "1").strip().lower() not in ("0", "false", "no", "")

# ---------- LangSmith ----------
LANGCHAIN_TRACING_V2 = _get_bool("LANGCHAIN_TRACING_V2", "true")
LANGCHAIN_ENDPOINT = _get("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
LANGCHAIN_PROJECT = _get("LANGCHAIN_PROJECT", "AroBot")

# ---------- Ollama ----------
OLLAMA_BASE_URL = _get("OLLAMA_BASE_URL")
OLLAMA_TEXT_MODEL = _get("OLLAMA_TEXT_MODEL")
OLLAMA_VISION_MODEL = _get("OLLAMA_VISION_MODEL")
OLLAMA_FAST_TEXT_MODEL = _get("OLLAMA_FAST_TEXT_MODEL")

# ---------- Region ----------
DEFAULT_REGION = _get("DEFAULT_REGION")

# ---------- OCR ----------
# Sensible defaults so the app doesn't crash without .env values
OCR_LANGUAGE = _get("OCR_LANGUAGE", "en")
try:
    OCR_CONFIDENCE_THRESHOLD = float(_get("OCR_CONFIDENCE_THRESHOLD", "0.5"))
except Exception:
    OCR_CONFIDENCE_THRESHOLD = 0.5

# ---------- App ----------
APP_HOST = _get("APP_HOST")
APP_PORT = int(_get("APP_PORT"))
DEBUG = _get_bool("DEBUG")

# ---------- Data paths ----------
DATA_DIR = Path("data")
PRESCRIPTION_DIR = Path("prescription_data")   # fixed typo
WEB_SCRAPE_DIR = Path("Web Scrape")
STATIC_DIR = Path("static")
TEMPLATES_DIR = Path("templates")
MEMORY_DIR = Path("memory")

for p in [DATA_DIR, PRESCRIPTION_DIR, WEB_SCRAPE_DIR, STATIC_DIR, TEMPLATES_DIR, MEMORY_DIR]:
    p.mkdir(parents=True, exist_ok=True)
