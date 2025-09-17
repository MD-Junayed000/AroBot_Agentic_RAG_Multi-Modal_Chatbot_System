#!/usr/bin/env python3
"""
Setup script to create .env file and initialize vector database
"""

import os
import sys
from pathlib import Path

def create_env_file():
    """Create .env file with required configuration"""
    env_content = """# Pinecone Configuration
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_REGION=us-east-1
PINECONE_ENABLE=1
PINECONE_MEDICINE_INDEX=arobot-medicine-data
PINECONE_BD_PHARMACY_INDEX=arobot-bd-pharmacy
PINECONE_IMAGE_INDEX=arobot-clip

# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_TEXT_MODEL=llama3.2:3b
OLLAMA_VISION_MODEL=llava:7b
OLLAMA_FAST_TEXT_MODEL=llama3.2:3b

# App Configuration
DEBUG=false
APP_HOST=0.0.0.0
APP_PORT=8000

# OCR Configuration
OCR_LANGUAGE=en
OCR_CONFIDENCE_THRESHOLD=0.5

# Optional: LangSmith for tracing
LANGSMITH_API_KEY=your_langsmith_key_here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_PROJECT=AroBot

# Optional: Web search
DEFAULT_REGION=Bangladesh
"""
    
    env_path = Path(".env")
    if env_path.exists():
        print("✅ .env file already exists")
        return True
    
    try:
        with open(env_path, "w") as f:
            f.write(env_content)
        print("✅ Created .env file")
        print("⚠️  Please update PINECONE_API_KEY with your actual key")
        return True
    except Exception as e:
        print(f"❌ Failed to create .env file: {e}")
        return False

def test_imports():
    """Test critical imports"""
    try:
        from core.vector_store import PineconeStore
        print("✅ Vector store import OK")
    except Exception as e:
        print(f"❌ Vector store import failed: {e}")
        return False
    
    try:
        from core.llm_modular import ModularLLMHandler
        print("✅ LLM handler import OK")
    except Exception as e:
        print(f"❌ LLM handler import failed: {e}")
        return False
    
    try:
        from utils.web_search import WebSearchTool
        print("✅ Web search import OK")
    except Exception as e:
        print(f"❌ Web search import failed: {e}")
        return False
    
    return True

def create_indexes():
    """Create Pinecone indexes if they don't exist"""
    try:
        from core.vector_store import PineconeStore
        from config.env_config import PINECONE_API_KEY, PINECONE_MEDICINE_INDEX, PINECONE_BD_PHARMACY_INDEX
        
        if not PINECONE_API_KEY or PINECONE_API_KEY == "your_pinecone_api_key_here":
            print("⚠️  PINECONE_API_KEY not set properly in .env file")
            return False
        
        # Create medicine index
        try:
            medicine_store = PineconeStore(index_name=PINECONE_MEDICINE_INDEX, dimension=384)
            indexes = medicine_store.list_indexes()
            
            if PINECONE_MEDICINE_INDEX not in indexes:
                if medicine_store.create_index(PINECONE_MEDICINE_INDEX, dimension=384):
                    print(f"✅ Created medicine index: {PINECONE_MEDICINE_INDEX}")
                else:
                    print(f"❌ Failed to create medicine index: {PINECONE_MEDICINE_INDEX}")
            else:
                print(f"✅ Medicine index already exists: {PINECONE_MEDICINE_INDEX}")
            
            # Create BD pharmacy index
            if PINECONE_BD_PHARMACY_INDEX not in indexes:
                bd_store = PineconeStore(index_name=PINECONE_BD_PHARMACY_INDEX, dimension=384)
                if bd_store.create_index(PINECONE_BD_PHARMACY_INDEX, dimension=384):
                    print(f"✅ Created BD pharmacy index: {PINECONE_BD_PHARMACY_INDEX}")
                else:
                    print(f"❌ Failed to create BD pharmacy index: {PINECONE_BD_PHARMACY_INDEX}")
            else:
                print(f"✅ BD pharmacy index already exists: {PINECONE_BD_PHARMACY_INDEX}")
            
            return True
            
        except Exception as e:
            print(f"❌ Index creation failed: {e}")
            return False
            
    except Exception as e:
        print(f"❌ Pinecone setup failed: {e}")
        return False

def populate_knowledge_base():
    """Populate knowledge base with medicine data"""
    try:
        from utils.setup_knowledge_base import setup_medicine_knowledge_base
        print("📝 Setting up medicine knowledge base...")
        
        if setup_medicine_knowledge_base():
            print("✅ Medicine knowledge base populated")
            return True
        else:
            print("⚠️  Medicine knowledge base setup had issues")
            return False
            
    except Exception as e:
        print(f"❌ Knowledge base population failed: {e}")
        return False

def main():
    """Main setup function"""
    print("🏥 AroBot Setup Script")
    print("=" * 40)
    
    # Step 1: Create .env file
    if not create_env_file():
        sys.exit(1)
    
    # Step 2: Test imports
    if not test_imports():
        print("⚠️  Some imports failed, but continuing...")
    
    # Step 3: Create indexes (if Pinecone key is set)
    if not create_indexes():
        print("⚠️  Index creation failed - please set PINECONE_API_KEY")
    
    # Step 4: Populate knowledge base (if indexes exist)
    # if not populate_knowledge_base():
    #     print("⚠️  Knowledge base population failed")
    
    print("\n🎉 Setup completed!")
    print("\n📋 Next steps:")
    print("1. Update PINECONE_API_KEY in .env file with your actual key")
    print("2. Restart your server: python app.py")
    print("3. Test image processing and RAG functionality")
    print("4. Visit: http://localhost:8000/admin/vector/indexes")

if __name__ == "__main__":
    main()
