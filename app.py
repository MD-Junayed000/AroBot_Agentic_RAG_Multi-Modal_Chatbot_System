# app.py
"""
AroBot - Multi-Modal Medical Chatbot System
Main application entry point
"""
import os
import uvicorn
from config.env_config import APP_HOST, APP_PORT, DEBUG

if __name__ == "__main__":
    host = APP_HOST or "127.0.0.1"
    port = int(APP_PORT or 8000)
    print("🏥 Starting AroBot Multi-Modal Medical Chatbot System...")
    print(f"🌐 UI:       http://{host}:{port}/chat")
    print(f"📖 OpenAPI:  http://{host}:{port}/docs")
    print(f"🔍 Health:   http://{host}:{port}/health")
    print("\n⚡ Starting server...\n")
    # Use module:app target (prevents double import of app object)
    uvicorn.run(
        "api.main:app",
        host=host,
        port=port,
        reload=bool(DEBUG),
        log_level="info"
    )
