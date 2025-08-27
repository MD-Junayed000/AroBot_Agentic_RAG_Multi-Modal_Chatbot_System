#!/usr/bin/env python3
"""
System verification script for AroBot
"""
import sys
import importlib
from pathlib import Path

def check_file_structure():
    """Check if all required files and directories exist"""
    print("📁 Checking file structure...")
    
    required_dirs = [
        "agents", "api", "config", "core", "mcp_server", 
        "static", "templates", "utils", "data", "Web Scrape"
    ]
    
    required_files = [
        "app.py", "requirements.txt", "README.md",
        "setup_system.py", "run_arobot.bat", "run_arobot.sh"
    ]
    
    missing_dirs = [d for d in required_dirs if not Path(d).exists()]
    missing_files = [f for f in required_files if not Path(f).exists()]
    
    if missing_dirs:
        print(f"❌ Missing directories: {missing_dirs}")
        return False
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    
    print("✅ All required files and directories exist")
    return True

def check_imports():
    """Check if all modules can be imported"""
    print("\n📦 Checking imports...")
    
    modules_to_check = [
        "config.env_config",
        "core.embeddings", 
        "core.vector_store",
        "agents.rag_agent",
        "agents.ocr_agent", 
        "agents.medical_agent",
        "utils.ocr_pipeline",
        "utils.web_search",
        "mcp_server.conversation_memory",
        "mcp_server.mcp_handler",
        "api.main"
    ]
    
    failed_imports = []
    
    for module in modules_to_check:
        try:
            importlib.import_module(module)
            print(f"✅ {module}")
        except ImportError as e:
            print(f"❌ {module}: {e}")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"\n❌ Failed imports: {len(failed_imports)}")
        return False
    
    print(f"\n✅ All {len(modules_to_check)} modules imported successfully")
    return True

def check_configuration():
    """Check configuration"""
    print("\n⚙️ Checking configuration...")
    
    try:
        from config.env_config import (
            PINECONE_API_KEY, LANGSMITH_API_KEY,
            OLLAMA_BASE_URL, OLLAMA_TEXT_MODEL, OLLAMA_VISION_MODEL
        )
        
        if not PINECONE_API_KEY or PINECONE_API_KEY == "your_api_key":
            print("⚠️ Pinecone API key not configured")
        else:
            print("✅ Pinecone API key configured")
        
        if not LANGSMITH_API_KEY or LANGSMITH_API_KEY == "your_api_key":
            print("⚠️ LangSmith API key not configured")
        else:
            print("✅ LangSmith API key configured")
        
        print(f"✅ Ollama URL: {OLLAMA_BASE_URL}")
        print(f"✅ Text model: {OLLAMA_TEXT_MODEL}")
        print(f"✅ Vision model: {OLLAMA_VISION_MODEL}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Configuration import failed: {e}")
        return False

def check_data_files():
    """Check if data files exist"""
    print("\n📊 Checking data files...")
    
    pdf_files = list(Path("data").glob("*.pdf"))
    csv_files = list(Path("Web Scrape").glob("*.csv"))
    prescription_images = list(Path("prescribtion data").glob("*.jpg"))
    
    print(f"📚 PDF files: {len(pdf_files)}")
    print(f"📊 CSV files: {len(csv_files)}")
    print(f"🏥 Prescription images: {len(prescription_images)}")
    
    if len(pdf_files) == 0:
        print("⚠️ No PDF files found in data directory")
    
    if len(csv_files) == 0:
        print("⚠️ No CSV files found in Web Scrape directory")
    
    if len(prescription_images) == 0:
        print("⚠️ No prescription images found")
    
    return True

def main():
    """Main verification function"""
    print("🏥 AroBot System Verification")
    print("=" * 50)
    
    checks = [
        ("File Structure", check_file_structure),
        ("Module Imports", check_imports),
        ("Configuration", check_configuration),
        ("Data Files", check_data_files)
    ]
    
    results = []
    
    for name, check_func in checks:
        try:
            result = check_func()
            results.append(result)
        except Exception as e:
            print(f"❌ {name} check failed with error: {e}")
            results.append(False)
    
    print("\n" + "=" * 50)
    print("VERIFICATION SUMMARY")
    print("=" * 50)
    
    for (name, _), result in zip(checks, results):
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{name:<20} {status}")
    
    overall_success = all(results)
    
    if overall_success:
        print("\n🎉 System verification completed successfully!")
        print("\nYou can now run the system with:")
        print("  python app.py")
        print("  or")
        print("  python setup_system.py  (for full setup)")
    else:
        print("\n⚠️ System verification found issues.")
        print("Please resolve the failed checks before running the system.")
    
    return overall_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
