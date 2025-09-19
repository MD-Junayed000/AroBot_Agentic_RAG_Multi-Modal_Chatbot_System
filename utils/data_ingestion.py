# utils/data_ingestion.py
import io
import fitz
from pathlib import Path
from typing import List, Dict
from PIL import Image
import logging

try:
    import pytesseract
except Exception:
    pytesseract = None

from core.vector_store import PineconeStore

logger = logging.getLogger(__name__)


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


def discover_pdf_directories(base_dir: str) -> List[Path]:
    """Discover all directories that might contain PDFs."""
    base_path = Path(base_dir)
    if not base_path.exists():
        logger.warning(f"Base directory not found: {base_path}")
        return []
    
    pdf_dirs = []
    
    # Check base directory for PDFs
    if list(base_path.glob("*.pdf")):
        pdf_dirs.append(base_path)
    
    # Recursively check subdirectories
    for subdir in base_path.rglob("*"):
        if subdir.is_dir() and list(subdir.glob("*.pdf")):
            pdf_dirs.append(subdir)
    
    logger.info(f"Found {len(pdf_dirs)} directories containing PDFs")
    for pdf_dir in pdf_dirs:
        pdf_count = len(list(pdf_dir.glob("*.pdf")))
        logger.info(f"  - {pdf_dir}: {pdf_count} PDFs")
    
    return pdf_dirs


def load_pdfs(folder: str) -> List[Dict]:
    """Load PDFs from folder and all subdirectories."""
    docs = []
    base_path = Path(folder)
    
    if not base_path.exists():
        logger.warning(f"PDF folder not found: {folder}")
        return docs
    
    # Find all PDF files recursively
    pdf_files = list(base_path.rglob("*.pdf"))
    logger.info(f"Found {len(pdf_files)} PDF files in {folder}")
    
    for pdf_path in pdf_files:
        try:
            logger.info(f"Processing PDF: {pdf_path}")
            doc = fitz.open(str(pdf_path))
            
            for i in range(len(doc)):
                try:
                    t = _page_text_or_ocr(doc[i])
                    if t and t.strip():
                        docs.append({
                            "text": t.strip(), 
                            "meta": {
                                "id": f"{pdf_path.stem}_p{i}", 
                                "source": str(pdf_path), 
                                "page": i,
                                "filename": pdf_path.name,
                                "directory": str(pdf_path.parent.relative_to(base_path)),
                            }
                        })
                except Exception as e:
                    logger.warning(f"Error processing page {i} of {pdf_path}: {e}")
                    continue
            
            doc.close()
            logger.info(f"Successfully processed {pdf_path.name}")
            
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {e}")
            continue
    
    logger.info(f"Total pages extracted: {len(docs)}")
    return docs


def ingest_pdf_to_knowledge_base(pdf_path: str, namespace: str = "general", filename_override: str | None = None) -> Dict:
    """Ingest a single PDF into Pinecone under the given namespace/index.

    - Extracts text per page (OCR fallback for scanned pages)
    - Upserts to Pinecone with minimal metadata
    - Creates the index on-the-fly if missing
    """
    p = Path(pdf_path)
    if not p.exists():
        return {"status": "error", "error": f"PDF not found: {pdf_path}"}

    try:
        doc = fitz.open(str(p))
        texts: list[str] = []
        metas: list[Dict] = []
        for i in range(len(doc)):
            t = _page_text_or_ocr(doc[i])
            if t and t.strip():
                texts.append(t.strip())
                metas.append({
                    "id": f"{p.stem}_p{i}",
                    "source": str(p),
                    "filename": filename_override or p.name,
                    "page": i,
                    "text": t.strip(),
                })

        store = PineconeStore(index_name=namespace, dimension=384)
        # Ensure index exists
        store.create_index(namespace, dimension=384, metric="cosine")
        upserted = store.upsert_texts(texts, metas, namespace=namespace)

        return {"status": "success", "chunks": upserted, "pages": len(texts)}
    except Exception as e:
        return {"status": "error", "error": str(e)}


if __name__ == "__main__":
    import sys
    
    # Allow command line argument for folder
    folder = sys.argv[1] if len(sys.argv) > 1 else "data"
    
    store = PineconeStore(dimension=384)
    docs = load_pdfs(folder)
    print("Loaded pages:", len(docs))
    
    if docs:
        store.upsert_texts([d["text"] for d in docs], [d["meta"] for d in docs])
        print("Indexed to Pinecone.")
    else:
        print("No documents to index.")
