"""
Environment configuration for AroBot Multi-Modal Chatbot System
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys - Load from environment variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")

# Validate API keys
if not PINECONE_API_KEY or PINECONE_API_KEY == "your_pinecone_api_key_here":
    print("⚠️  Warning: PINECONE_API_KEY not set in .env file")
    
if not LANGSMITH_API_KEY or LANGSMITH_API_KEY == "your_langsmith_api_key_here":
    print("⚠️  Warning: LANGSMITH_API_KEY not set in .env file")

# Pinecone Configuration
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")
PINECONE_INDEX_PDF = os.getenv("PINECONE_INDEX_PDF", "arobot-medical-pdfs")
PINECONE_INDEX_MEDICINE = os.getenv("PINECONE_INDEX_MEDICINE", "arobot-medicine-data")
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "384"))

# LangSmith Configuration
LANGCHAIN_TRACING_V2 = True
LANGCHAIN_ENDPOINT = "https://api.smith.langchain.com"
LANGCHAIN_PROJECT = "arobot-multimodal-chatbot"

# Set environment variables for LangSmith
if LANGSMITH_API_KEY and LANGSMITH_API_KEY != "your_langsmith_api_key_here":
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"] = LANGCHAIN_ENDPOINT
    os.environ["LANGCHAIN_API_KEY"] = LANGSMITH_API_KEY
    os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT
    print("✅ LangSmith monitoring configured")
else:
    print("⚠️  LangSmith monitoring disabled - API key not configured")

# Ollama Configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_TEXT_MODEL = os.getenv("OLLAMA_TEXT_MODEL", "neural-chat:7b")
OLLAMA_VISION_MODEL = os.getenv("OLLAMA_VISION_MODEL", "llava:7b")

# OCR Configuration
OCR_LANGUAGE = os.getenv("OCR_LANGUAGE", "en")
OCR_CONFIDENCE_THRESHOLD = float(os.getenv("OCR_CONFIDENCE_THRESHOLD", "0.5"))

# Application Configuration
APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
APP_PORT = int(os.getenv("APP_PORT", "8000"))
DEBUG = os.getenv("DEBUG", "true").lower() == "true"

# Data Paths
DATA_DIR = Path("data")
PRESCRIPTION_DIR = Path("prescribtion data")
WEB_SCRAPE_DIR = Path("Web Scrape")
STATIC_DIR = Path("static")
TEMPLATES_DIR = Path("templates")

# Ensure directories exist
for dir_path in [DATA_DIR, PRESCRIPTION_DIR, WEB_SCRAPE_DIR, STATIC_DIR, TEMPLATES_DIR]:
    dir_path.mkdir(exist_ok=True)
