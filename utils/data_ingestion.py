# utils/data_ingestion.py
import io
import fitz
from pathlib import Path
from typing import List, Dict
from PIL import Image

try:
    import pytesseract
except Exception:
    pytesseract = None

from core.vector_store import PineconeStore

def _page_text_or_ocr(page) -> str:
    """Get vectorizable text for one page; OCR if needed."""
    text = (page.get_text("text") or "").strip()
    if text:
        return text
    if not pytesseract:
        return ""
    # OCR scanned page
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x improves OCR
    img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("L")
    return pytesseract.image_to_string(img) or ""

def load_pdfs(folder: str) -> List[Dict]:
    docs = []
    for p in Path(folder).glob("*.pdf"):
        doc = fitz.open(str(p))
        for i in range(len(doc)):
            t = _page_text_or_ocr(doc[i])
            if t and t.strip():
                docs.append({"text": t.strip(), "meta": {"id": f"{p.stem}_p{i}", "source": str(p), "page": i}})
    return docs

if __name__ == "__main__":
    store = PineconeStore(dimension=384)
    folder = "knowledge/pdfs"
    docs = load_pdfs(folder)
    print("Loaded pages:", len(docs))
    store.upsert_texts([d["text"] for d in docs], [d["meta"] for d in docs])
    print("Indexed to Pinecone.")
