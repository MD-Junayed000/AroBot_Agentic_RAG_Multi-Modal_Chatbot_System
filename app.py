"""
AroBot - Multi-Modal Medical Chatbot System
Main application entry point
"""
import uvicorn
from api.main import app
from config.env_config import APP_HOST, APP_PORT, DEBUG

if __name__ == "__main__":
    print("🏥 Starting AroBot Multi-Modal Medical Chatbot System...")
    print(f"🌐 Main interface: http://localhost:{APP_PORT}")
    print(f"💬 Chat interface: http://localhost:{APP_PORT}/chat") 
    print(f"📖 API documentation: http://localhost:{APP_PORT}/docs")
    print(f"🔍 Health check: http://localhost:{APP_PORT}/health")
    print("\n🎯 Features available:")
    print("   • 💊 Medicine information with RAG")
    print("   • 🔍 Prescription image analysis") 
    print("   • 🧠 Multi-modal AI assistance")
    print("   • 📊 LangSmith monitoring")
    print("   • 💾 Conversation memory")
    print("\n⚡ Starting server...")
    
    uvicorn.run(
        "api.main:app",
        host=APP_HOST,
        port=APP_PORT,
        reload=DEBUG,
        log_level="info"
    )