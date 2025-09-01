# core/image_index.py
from __future__ import annotations
import os
from typing import List, Dict, Any, Tuple, Optional
from PIL import Image
import numpy as np

# Torch/CLIP are optional; degrade gracefully if absent
try:
    import torch
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False
    torch = None  # type: ignore

try:
    import clip  # OpenAI CLIP
except Exception:
    clip = None  # tolerate missing CLIP

# Pinecone optional as well
try:
    from pinecone import Pinecone, ServerlessSpec
    HAS_PINECONE = True
except Exception:
    Pinecone = None  # type: ignore
    ServerlessSpec = None  # type: ignore
    HAS_PINECONE = False

from config.env_config import PINECONE_API_KEY, PINECONE_IMAGE_INDEX, PINECONE_REGION


class CLIPImageIndex:
    """
    Lazy Pinecone + CLIP wrapper (no circular imports):
      - No imports from LLMHandler.
      - Creates Pinecone index on first use.
      - Graceful fallback when CLIP/Pinecone unavailable or disabled
        (set AROBOT_DISABLE_CLIP_INDEX=true to force-disable).
    """

    def __init__(self, index_name: str = PINECONE_IMAGE_INDEX):
        self.index_name = index_name
        self.device = "cuda" if (HAS_TORCH and torch.cuda.is_available()) else "cpu"
        self._pc: Optional["Pinecone"] = None
        self._index = None
        self._model = None
        self._preprocess = None
        self._disabled = (os.getenv("AROBOT_DISABLE_CLIP_INDEX", "").strip().lower() in ("1", "true", "yes"))

    # ----------------- Internal initializers ----------------- #
    def _ensure_clip(self):
        if self._model is not None:
            return
        if clip is None or not HAS_TORCH:
            raise RuntimeError("CLIP not available (install torch and openai-clip).")
        self._model, self._preprocess = clip.load("ViT-B/32", device=self.device)
        self._model.eval()

    def _ensure_pinecone(self):
        if self._pc is not None and self._index is not None:
            return
        if self._disabled:
            raise RuntimeError("CLIP image index disabled by AROBOT_DISABLE_CLIP_INDEX")
        if not HAS_PINECONE:
            raise RuntimeError("Pinecone client not installed.")
        if not PINECONE_API_KEY:
            raise RuntimeError("PINECONE_API_KEY not set")

        self._pc = Pinecone(api_key=PINECONE_API_KEY)

        # Create serverless index if missing
        names = [i.name for i in self._pc.list_indexes()]
        if self.index_name not in names:
            self._pc.create_index(
                name=self.index_name,
                dimension=512,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region=PINECONE_REGION),
            )
        self._index = self._pc.Index(self.index_name)

    # ----------------- Embeddings ----------------- #
    def _embed_image(self, img: Image.Image) -> np.ndarray:
        self._ensure_clip()
        im = self._preprocess(img).unsqueeze(0)
        if HAS_TORCH:
            im = im.to(self.device)
            with torch.no_grad():
                feats = self._model.encode_image(im)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                return feats.squeeze(0).detach().cpu().numpy().astype("float32")
        raise RuntimeError("Torch not available for CLIP.")

    def _embed_text(self, text: str) -> np.ndarray:
        self._ensure_clip()
        if not HAS_TORCH:
            raise RuntimeError("Torch not available for CLIP.")
        toks = clip.tokenize([text]).to(self.device)
        with torch.no_grad():
            feats = self._model.encode_text(toks)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            return feats.squeeze(0).detach().cpu().numpy().astype("float32")

    # ----------------- Public API ----------------- #
    def upsert_images(self, items: List[Tuple[str, Image.Image, Dict[str, Any]]], namespace: str = "anatomy"):
        try:
            self._ensure_pinecone()
        except Exception:
            return  # disabled/unavailable
        vectors = []
        for _id, pil, meta in items:
            try:
                vec = self._embed_image(pil)
            except Exception:
                continue
            vectors.append({"id": _id, "values": vec.tolist(), "metadata": meta})
        try:
            self._index.upsert(vectors=vectors, namespace=namespace)
        except Exception:
            pass  # ignore transient failure

    def query_by_image(self, img: Image.Image, top_k: int = 8, namespace: str = "anatomy"):
        try:
            self._ensure_pinecone()
        except Exception:
            return {"matches": []}
        try:
            qv = self._embed_image(img).tolist()
            return self._index.query(vector=qv, top_k=top_k, include_metadata=True, namespace=namespace)
        except Exception:
            return {"matches": []}

    def query_by_text(self, text: str, top_k: int = 8, namespace: str = "anatomy"):
        try:
            self._ensure_pinecone()
        except Exception:
            return {"matches": []}
        try:
            qv = self._embed_text(text).tolist()
            return self._index.query(vector=qv, top_k=top_k, include_metadata=True, namespace=namespace)
        except Exception:
            return {"matches": []}
