# Lazy/optional imports to avoid hard failures at app import time
try:
    from .embeddings import Embedder
except Exception:
    print("Warning: Embedder import failed. Torch/SentenceTransformers may be missing.")
    Embedder = None  # type: ignore

from .vector_store import PineconeStore
try:
    from .llm_modular import ModularLLMHandler as LLMHandler
except ImportError:
    print("Warning: LLMHandler import failed. Ollama may not be available.")
    LLMHandler = None

try:
    from .multimodal_processor import MultiModalProcessor  
except ImportError:
    print("Warning: MultiModalProcessor import failed. CLIP may not be available.")
    MultiModalProcessor = None