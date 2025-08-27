#!/usr/bin/env python3
"""
Setup script for AroBot Multi-Modal Medical Chatbot System
"""
import os
import sys
import subprocess
from pathlib import Path

def print_step(step, description):
    """Print setup step"""
    print(f"\n{'='*50}")
    print(f"STEP {step}: {description}")
    print('='*50)

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"Running: {description}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"Error: {e.stderr}")
        return False

def check_python_version():
    """Check Python version"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        sys.exit(1)
    print(f"✅ Python {sys.version.split()[0]} detected")

def setup_environment():
    """Setup the development environment"""
    print_step(1, "Environment Setup")
    
    # Check Python version
    check_python_version()
    
    # Install requirements
    if not run_command("pip install -r requirements.txt", "Installing Python dependencies"):
        print("⚠️  Some dependencies might have failed to install. Check the output above.")
        return False
    
    return True

def setup_ollama():
    """Setup Ollama models"""
    print_step(2, "Ollama Setup")
    
    print("📋 Ollama setup instructions:")
    print("1. Install Ollama from https://ollama.ai/")
    print("2. Run the following commands to download required models:")
    print("   ollama pull llama3.2:3b")
    print("   ollama pull llava:7b")
    print("\n⚠️  This step requires manual installation of Ollama")
    
    response = input("Have you installed Ollama and downloaded the models? (y/N): ")
    if response.lower() != 'y':
        print("⚠️  Please install Ollama and download models before proceeding")
        return False
    
    return True

def setup_knowledge_base():
    """Setup the knowledge base"""
    print_step(3, "Knowledge Base Setup")
    
    print("Setting up Pinecone knowledge base...")
    
    try:
        from utils.setup_knowledge_base import main as setup_kb
        if setup_kb():
            print("✅ Knowledge base setup completed")
            return True
        else:
            print("❌ Knowledge base setup failed")
            return False
    except Exception as e:
        print(f"❌ Error setting up knowledge base: {e}")
        print("⚠️  You can run this later with: python -m utils.setup_knowledge_base")
        return False

def create_run_script():
    """Create run script"""
    print_step(4, "Creating Run Scripts")
    
    # Windows batch script
    windows_script = """@echo off
echo Starting AroBot Multi-Modal Medical Chatbot System...
python app.py
pause
"""
    
    with open("run_arobot.bat", "w") as f:
        f.write(windows_script)
    
    # Unix shell script
    unix_script = """#!/bin/bash
echo "Starting AroBot Multi-Modal Medical Chatbot System..."
python app.py
"""
    
    with open("run_arobot.sh", "w") as f:
        f.write(unix_script)
    
    # Make shell script executable on Unix systems
    if os.name != 'nt':
        os.chmod("run_arobot.sh", 0o755)
    
    print("✅ Run scripts created: run_arobot.bat (Windows) and run_arobot.sh (Unix)")
    return True

def verify_setup():
    """Verify the setup"""
    print_step(5, "Verification")
    
    try:
        # Test imports
        print("Testing imports...")
        from core.llm_handler import LLMHandler
        from agents.medical_agent import MedicalAgent
        from api.main import app
        print("✅ All imports successful")
        
        # Test LLM connectivity
        print("Testing LLM connectivity...")
        llm = LLMHandler()
        status = llm.check_model_availability()
        
        if status.get('text_model_available') and status.get('vision_model_available'):
            print("✅ Both text and vision models are available")
        else:
            print("⚠️  Some models may not be available. Check Ollama setup.")
            print(f"Available models: {status.get('models_found', [])}")
        
        return True
        
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return False

def main():
    """Main setup function"""
    print("🏥 AroBot Multi-Modal Medical Chatbot System Setup")
    print("=" * 60)
    
    steps_success = []
    
    # Step 1: Environment setup
    steps_success.append(setup_environment())
    
    # Step 2: Ollama setup
    steps_success.append(setup_ollama())
    
    # Step 3: Knowledge base setup
    steps_success.append(setup_knowledge_base())
    
    # Step 4: Create run scripts
    steps_success.append(create_run_script())
    
    # Step 5: Verification
    steps_success.append(verify_setup())
    
    # Summary
    print("\n" + "="*60)
    print("SETUP SUMMARY")
    print("="*60)
    
    step_names = [
        "Environment Setup",
        "Ollama Setup", 
        "Knowledge Base Setup",
        "Run Scripts Creation",
        "System Verification"
    ]
    
    for i, (name, success) in enumerate(zip(step_names, steps_success), 1):
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"Step {i}: {name:<25} {status}")
    
    if all(steps_success):
        print("\n🎉 Setup completed successfully!")
        print("\nTo start the system:")
        print("  Windows: run_arobot.bat")
        print("  Unix/Linux/Mac: ./run_arobot.sh")
        print("  Or directly: python app.py")
        print(f"\nWeb interface will be available at: http://localhost:8000/api/v1/chat")
    else:
        print("\n⚠️  Setup completed with some issues.")
        print("Please review the failed steps above and resolve them before running the system.")
    
    return all(steps_success)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
