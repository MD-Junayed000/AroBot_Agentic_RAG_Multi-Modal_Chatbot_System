# core/image_index.py
from __future__ import annotations
import torch
from typing import List, Dict, Any, Tuple
from PIL import Image
import clip
import numpy as np
from pinecone import Pinecone, ServerlessSpec
from config.env_config import PINECONE_API_KEY, PINECONE_IMAGE_INDEX

class CLIPImageIndex:
    def __init__(self, index_name: str = PINECONE_IMAGE_INDEX):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)  # 512-d
        self.pc = Pinecone(api_key=PINECONE_API_KEY)

        names = [i.name for i in self.pc.list_indexes()]
        if index_name not in names:
            self.pc.create_index(
                name=index_name,
                dimension=512,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
        self.index = self.pc.Index(index_name)
        # sanity check (best-effort; ignore errors quietly)
        try:
            stats = self.index.describe_index_stats()
            # Can't always read dim, but if available, validate
        except Exception:
            pass


    @torch.no_grad()
    def embed_image(self, img: Image.Image) -> np.ndarray:
        im = self.preprocess(img).unsqueeze(0).to(self.device)
        feats = self.model.encode_image(im)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.squeeze(0).detach().cpu().numpy().astype("float32")

    @torch.no_grad()
    def embed_text(self, text: str) -> np.ndarray:
        toks = clip.tokenize([text]).to(self.device)
        feats = self.model.encode_text(toks)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.squeeze(0).detach().cpu().numpy().astype("float32")

    def upsert_images(self, items: List[Tuple[str, Image.Image, Dict[str,Any]]], namespace: str="anatomy"):
        vectors = []
        for _id, pil, meta in items:
            vec = self.embed_image(pil)
            vectors.append({"id": _id, "values": vec.tolist(), "metadata": meta})
        self.index.upsert(vectors=vectors, namespace=namespace)

    def query_by_image(self, img: Image.Image, top_k: int=8, namespace: str="anatomy"):
        qv = self.embed_image(img).tolist()
        return self.index.query(vector=qv, top_k=top_k, include_metadata=True, namespace=namespace)

    def query_by_text(self, text: str, top_k: int=8, namespace: str="anatomy"):
        qv = self.embed_text(text).tolist()
        return self.index.query(vector=qv, top_k=top_k, include_metadata=True, namespace=namespace)
