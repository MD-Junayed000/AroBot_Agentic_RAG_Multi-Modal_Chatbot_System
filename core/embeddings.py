# core/embeddings.py
import os
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
import logging

logger = logging.getLogger(__name__)

# Module-level singleton to avoid repeated GPU loads
_EMBEDDER_SINGLETON = None
_DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _pick_device() -> str:
    # Env overrides
    force = (os.getenv("EMBEDDINGS_DEVICE", "") or "").strip().lower()  # 'cpu' | 'cuda' | ''
    force_cpu = (os.getenv("AROBOT_FORCE_CPU", "") or "").strip().lower() in ("1", "true", "yes")

    if force_cpu:
        return "cpu"
    if force in ("cpu", "cuda"):
        return force
    # auto
    return "cuda" if torch.cuda.is_available() else "cpu"


class Embedder:
    def __new__(cls, model_name: str = _DEFAULT_MODEL):
        global _EMBEDDER_SINGLETON
        if _EMBEDDER_SINGLETON is None or getattr(_EMBEDDER_SINGLETON, "_model_name", None) != model_name:
            _EMBEDDER_SINGLETON = super().__new__(cls)
            _EMBEDDER_SINGLETON._initialized = False  # type: ignore[attr-defined]
        return _EMBEDDER_SINGLETON

    def __init__(self, model_name: str = _DEFAULT_MODEL):
        if getattr(self, "_initialized", False):
            return
        self._model_name = model_name
        device = _pick_device()

        # Try requested/auto device first, then fall back to CPU on any error
        try:
            self.model = SentenceTransformer(model_name, device=device)
            msg_device = torch.cuda.get_device_name(0) if (device == "cuda" and torch.cuda.is_available()) else "cpu"
            logger.info(f"✅ Loaded embeddings model '{model_name}' on {msg_device}")
        except Exception as e:
            logger.warning(f"Embeddings init failed on '{device}' ({e}); falling back to CPU.")
            self.model = SentenceTransformer(model_name, device="cpu")

        self._initialized = True  # type: ignore[attr-defined]

    def embed(self, texts):
        if len(texts) > 100:
            logger.info(f"Embedding {len(texts)} texts...")
        embs = self.model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=32,  # override with PINECONE_BATCH in vector_store if needed
            show_progress_bar=False,
        )
        return np.array(embs).tolist()

    @staticmethod
    def reset(model_name: str = _DEFAULT_MODEL) -> "Embedder":
        """Drop the singleton and re-create (e.g., to switch CPU after a CUDA error)."""
        global _EMBEDDER_SINGLETON
        _EMBEDDER_SINGLETON = None
        return Embedder(model_name)
