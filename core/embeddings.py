# core/embeddings.py
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
import logging

logger = logging.getLogger(__name__)

# Module-level singleton to avoid repeated GPU loads
_EMBEDDER_SINGLETON = None


class Embedder:
    def __new__(cls, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        global _EMBEDDER_SINGLETON
        if _EMBEDDER_SINGLETON is None or getattr(_EMBEDDER_SINGLETON, "_model_name", None) != model_name:
            _EMBEDDER_SINGLETON = super().__new__(cls)
            _EMBEDDER_SINGLETON._initialized = False  # type: ignore[attr-defined]
        return _EMBEDDER_SINGLETON

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        if getattr(self, "_initialized", False):
            return
        self._model_name = model_name
        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.model = SentenceTransformer(model_name, device=device)
            msg_device = torch.cuda.get_device_name(0) if device == "cuda" else "cpu"
            logger.info(f"✅ Loaded embeddings model '{model_name}' on {msg_device}")
        except Exception:
            logger.warning(f"Failed to load {model_name}, falling back to default")
            self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
        self._initialized = True  # type: ignore[attr-defined]

    def embed(self, texts):
        if len(texts) > 100:
            logger.info(f"Embedding {len(texts)} texts...")
        embs = self.model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=32,  # good default for RTX 3060
            show_progress_bar=False,
        )
        return np.array(embs).tolist()
