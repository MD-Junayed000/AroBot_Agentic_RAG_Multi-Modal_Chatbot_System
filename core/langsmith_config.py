"""
LangSmith Configuration and Initialization
"""
import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

def initialize_langsmith() -> bool:
    """Initialize LangSmith tracing if configured"""
    try:
        # Check if LangSmith is enabled
        langsmith_enabled = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() in ("true", "1", "yes")
        
        if not langsmith_enabled:
            logger.info("LangSmith tracing is disabled")
            return False
            
        # Set required environment variables
        api_key = os.getenv("LANGSMITH_API_KEY")
        if not api_key:
            logger.warning("LANGSMITH_API_KEY not set, disabling tracing")
            return False
            
        # Set LangChain environment variables
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = api_key
        
        # Set optional configuration
        if os.getenv("LANGCHAIN_ENDPOINT"):
            os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGCHAIN_ENDPOINT")
        else:
            os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
            
        if os.getenv("LANGCHAIN_PROJECT"):
            os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")
        else:
            os.environ["LANGCHAIN_PROJECT"] = "AroBot"
            
        logger.info(f"LangSmith tracing initialized for project: {os.getenv('LANGCHAIN_PROJECT')}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize LangSmith: {e}")
        return False

def is_langsmith_enabled() -> bool:
    """Check if LangSmith is enabled and properly configured"""
    return (
        os.getenv("LANGCHAIN_TRACING_V2", "false").lower() in ("true", "1", "yes") and
        bool(os.getenv("LANGSMITH_API_KEY"))
    ) 