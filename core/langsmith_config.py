"""
LangSmith Configuration and Initialization
"""
import os
import logging
from typing import Optional
from pathlib import Path

logger = logging.getLogger(__name__)

def load_env_file() -> bool:
    """Load environment variables from .env file if it exists"""
    try:
        env_file = Path(".env")
        if not env_file.exists():
            logger.warning("No .env file found. LangSmith configuration will rely on system environment variables.")
            return False
        
        # Simple .env file parser
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    if key and not os.getenv(key):  # Don't override existing env vars
                        os.environ[key] = value
        
        logger.info("Environment variables loaded from .env file")
        return True
    except Exception as e:
        logger.error(f"Error loading .env file: {e}")
        return False

def initialize_langsmith() -> bool:
    """Initialize LangSmith tracing if configured"""
    try:
        # First, try to load .env file
        load_env_file()
        
        # Check if LangSmith is enabled
        langsmith_enabled = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() in ("true", "1", "yes")
        
        if not langsmith_enabled:
            logger.info("LangSmith tracing is disabled (LANGCHAIN_TRACING_V2=false)")
            return False
            
        # Set required environment variables
        api_key = os.getenv("LANGSMITH_API_KEY")
        if not api_key or api_key == "your_langsmith_key_here":
            logger.warning("LANGSMITH_API_KEY not properly set, disabling tracing")
            logger.info("To enable LangSmith tracing:")
            logger.info("1. Get your API key from https://smith.langchain.com")
            logger.info("2. Set LANGSMITH_API_KEY in your .env file")
            logger.info("3. Set LANGCHAIN_TRACING_V2=true in your .env file")
            return False
        
        # Validate API key format (basic check)
        if len(api_key) < 10:
            logger.warning("LANGSMITH_API_KEY appears to be invalid (too short)")
            return False
            
        # Set LangChain environment variables
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = api_key
        
        # Set optional configuration
        endpoint = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
        project = os.getenv("LANGCHAIN_PROJECT", "AroBot")
        
        os.environ["LANGCHAIN_ENDPOINT"] = endpoint
        os.environ["LANGCHAIN_PROJECT"] = project
        
        # Test the connection (optional but helpful)
        try:
            from langsmith import Client
            client = Client(api_key=api_key, api_url=endpoint)
            # Simple test to verify connection
            client.list_runs(project_name=project, limit=1)
            logger.info(f"✅ LangSmith tracing initialized successfully")
            logger.info(f"   Project: {project}")
            logger.info(f"   Endpoint: {endpoint}")
            return True
        except ImportError:
            logger.info(f"LangSmith tracing configured (langsmith package not available for connection test)")
            logger.info(f"   Project: {project}")
            logger.info(f"   Endpoint: {endpoint}")
            return True
        except Exception as test_error:
            logger.warning(f"LangSmith configuration set but connection test failed: {test_error}")
            logger.info("Tracing may still work - check your API key and network connection")
            return True
        
    except Exception as e:
        logger.error(f"Failed to initialize LangSmith: {e}")
        return False

def is_langsmith_enabled() -> bool:
    """Check if LangSmith is enabled and properly configured"""
    api_key = os.getenv("LANGSMITH_API_KEY", "")
    return (
        os.getenv("LANGCHAIN_TRACING_V2", "false").lower() in ("true", "1", "yes") and
        bool(api_key) and
        api_key != "your_langsmith_key_here"
    )

def get_langsmith_status() -> dict:
    """Get detailed LangSmith configuration status"""
    return {
        "tracing_enabled": os.getenv("LANGCHAIN_TRACING_V2", "false").lower() in ("true", "1", "yes"),
        "api_key_set": bool(os.getenv("LANGSMITH_API_KEY")) and os.getenv("LANGSMITH_API_KEY") != "your_langsmith_key_here",
        "project": os.getenv("LANGCHAIN_PROJECT", "AroBot"),
        "endpoint": os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com"),
        "fully_configured": is_langsmith_enabled()
    } 