# utils/pdf_image_extractor.py
"""
Extract figures from a PDF, embed them with CLIP, and index into Pinecone
safely (trimming metadata and splitting requests under the 4 MB limit).
"""

import os
import io
import uuid
import logging
from typing import List, Dict, Any, Iterable

import fitz  # PyMuPDF
from PIL import Image

import torch
import clip  # from openai/CLIP repo

from core.vector_store import PineconeStore

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

# -----------------------------
# CLIP (lazy) model + preprocess
# -----------------------------
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_MODEL = None
_PREPROCESS = None


def _load_clip():
    global _MODEL, _PREPROCESS
    if _MODEL is None or _PREPROCESS is None:
        _MODEL, _PREPROCESS = clip.load("ViT-B/32", device=_DEVICE)
        _MODEL.eval()
    return _MODEL, _PREPROCESS


def _embed_pil(pil: Image.Image) -> List[float]:
    """Return L2-normalized CLIP image embedding as a Python list[float]."""
    model, preprocess = _load_clip()
    with torch.no_grad():
        t = preprocess(pil).unsqueeze(0).to(_DEVICE)  # [1,3,224,224]
        feats = model.encode_image(t)                 # [1,512]
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.squeeze(0).cpu().tolist()           # length 512


# -----------------------------
# PDF image extraction helpers
# -----------------------------
def _iter_pdf_images(pdf_path: str) -> Iterable[Dict[str, Any]]:
    """
    Yield dicts: { 'pil', 'page', 'caption' } for each raster image found.
    """
    doc = fitz.open(pdf_path)
    try:
        for page_idx in range(len(doc)):
            page = doc[page_idx]
            # crude "nearest caption": first text line on the page
            first_line = (page.get_text().strip().split("\n") or [""])[0]
            caption = first_line[:500]  # keep short; will trim again later
            for img in page.get_images(full=True):
                xref = img[0]
                base = doc.extract_image(xref)
                img_bytes = base["image"]
                pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                yield {
                    "pil": pil,
                    "page": page_idx + 1,
                    "caption": caption,
                }
    finally:
        doc.close()


# -----------------------------
# Metadata slimming
# -----------------------------
def _slim_meta(md: dict) -> dict:
    """
    Drop heavy fields and trim long strings to keep request bytes small.
    """
    for k in ("image_bytes", "image", "pixels", "thumbnail_b64"):
        md.pop(k, None)
    for k, v in list(md.items()):
        if isinstance(v, str) and len(v) > 900:
            md[k] = v[:900]
    return md


# -----------------------------
# Public API
# -----------------------------
def extract_images_from_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Kept for backward compatibility with older imports. Returns a list of:
      { 'pil': PIL.Image, 'page': int, 'caption': str }
    """
    return list(_iter_pdf_images(pdf_path))


def index_anatomy_pdf(
    pdf_path: str,
    title: str,
    namespace: str = "anatomy",
    index_name: str | None = None,
):
    """
    Extract images from `pdf_path`, embed with CLIP, and index into Pinecone.
    Splits requests safely via PineconeStore._safe_upsert and trims metadata.
    """
    pdf_path = os.fspath(pdf_path)
    if index_name is None:
        # Use a separate CLIP index so its dimension (512) won't clash with text (384)
        index_name = os.getenv("PINECONE_IMAGE_INDEX")

    # CLIP embeddings are 512-dim for ViT-B/32
    store = PineconeStore(index_name=index_name, dimension=512)

    batch: List[Dict[str, Any]] = []
    batch_limit = int(os.getenv("PINECONE_BATCH", "32"))

    count = 0
    for i, rec in enumerate(_iter_pdf_images(pdf_path)):
        pil: Image.Image = rec["pil"]
        page_no: int = rec["page"]
        caption: str = (rec.get("caption") or "").strip()

        # 1) embed
        embedding = _embed_pil(pil)

        # 2) vector payload
        vec_id = f"{title}-{uuid.uuid5(uuid.NAMESPACE_URL, f'{pdf_path}-{i}')}"
        meta = _slim_meta(
            {
                "doc": title,
                "page": page_no,
                "path": f"{pdf_path}#page={page_no}",
                "caption": caption,
                "doc_type": "anatomy_figure",
                "source": os.path.basename(pdf_path),
            }
        )

        batch.append({"id": vec_id, "values": embedding, "metadata": meta})
        count += 1

        # upsert in small batches to avoid 4 MB limit
        if len(batch) >= batch_limit:
            store._safe_upsert(batch, namespace=namespace)
            batch = []

    # flush remaining
    if batch:
        store._safe_upsert(batch, namespace=namespace)

    print(f"Indexed {count} images from {pdf_path}")
