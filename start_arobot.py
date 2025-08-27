#!/usr/bin/env python3
"""
AroBot Startup Script - Starts the server and runs tests
"""
import subprocess
import time
import requests
import sys
import os

def check_ollama():
    """Check if Ollama is running"""
    print("🔄 Checking Ollama...")
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ Ollama is running")
            models = result.stdout
            if "llama3.2:3b" in models:
                print("✅ Text model (llama3.2:3b) available")
            if "llava:7b" in models:
                print("✅ Vision model (llava:7b) available")
            return True
        else:
            print("❌ Ollama not responding")
            return False
    except Exception as e:
        print(f"❌ Ollama check failed: {e}")
        return False

def start_server():
    """Start the FastAPI server"""
    print("🚀 Starting AroBot server...")
    
    # Check if we're in virtual environment
    if not os.path.exists("venv"):
        print("❌ Virtual environment not found. Please run: python -m venv venv")
        return None
    
    # Activate venv and start server
    if sys.platform == "win32":
        python_path = "venv\\Scripts\\python.exe"
    else:
        python_path = "venv/bin/python"
    
    if not os.path.exists(python_path):
        print(f"❌ Python not found at {python_path}")
        return None
    
    # Start the server process
    process = subprocess.Popen(
        [python_path, "app.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True
    )
    
    print("⏳ Waiting for server to start...")
    
    # Wait for server to be ready
    max_wait = 30
    for i in range(max_wait):
        try:
            response = requests.get("http://localhost:8000/health", timeout=2)
            if response.status_code == 200:
                print("✅ Server is ready!")
                return process
        except:
            pass
        
        time.sleep(1)
        print(f"   Waiting... ({i+1}/{max_wait})")
    
    print("❌ Server failed to start within 30 seconds")
    process.terminate()
    return None

def run_tests():
    """Run basic functionality tests"""
    print("\n🧪 Running quick tests...")
    
    # Test health
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("✅ Health check passed")
        else:
            print(f"❌ Health check failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Health check failed: {e}")
    
    # Test basic chat
    try:
        payload = {"message": "Hello, what can you help me with?", "session_id": "test"}
        response = requests.post("http://localhost:8000/api/v1/chat", json=payload, timeout=15)
        if response.status_code == 200:
            data = response.json()
            print("✅ Chat functionality working")
            print(f"   Response preview: {data['response'][:100]}...")
        else:
            print(f"❌ Chat test failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Chat test failed: {e}")

def main():
    """Main startup routine"""
    print("🏥 AroBot Medical Chatbot System")
    print("=" * 50)
    
    # Check prerequisites
    if not check_ollama():
        print("\n⚠️  Ollama is required. Please:")
        print("   1. Install Ollama from https://ollama.ai")
        print("   2. Run: ollama pull llama3.2:3b")
        print("   3. Run: ollama pull llava:7b")
        return
    
    # Start server
    process = start_server()
    if not process:
        return
    
    # Run tests
    run_tests()
    
    # Show access information
    print("\n" + "=" * 50)
    print("🎉 AroBot is ready!")
    print("=" * 50)
    print("🌐 Access your chatbot:")
    print("   • Web Interface: http://localhost:8000/api/v1/chat")
    print("   • API Documentation: http://localhost:8000/docs")
    print("   • Health Check: http://localhost:8000/health")
    print("\n📊 Monitoring:")
    print("   • LangSmith: https://smith.langchain.com/")
    print("   • Pinecone: https://app.pinecone.io/")
    print("\n⚡ Features available:")
    print("   • 💬 Medical Q&A with RAG")
    print("   • 🔍 Prescription image analysis")
    print("   • 💊 Medicine database search")
    print("   • 🧠 Multi-modal AI (text + vision)")
    print("   • 📈 LangSmith monitoring")
    print("   • 💾 Conversation memory")
    print("\n🛑 Press Ctrl+C to stop the server")
    
    try:
        # Keep server running and show logs
        for line in process.stdout:
            if line.strip():
                print(f"[SERVER] {line.strip()}")
    except KeyboardInterrupt:
        print("\n🛑 Shutting down AroBot...")
        process.terminate()
        process.wait()
        print("✅ Server stopped")

if __name__ == "__main__":
    main()
