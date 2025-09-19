# scripts/ingest_pdfs_bd.py
import os, re, uuid, io, shutil, logging, argparse, hashlib
from pathlib import Path
from typing import List, Tuple, Dict
import pdfplumber
from PIL import Image
import pytesseract
from langchain.text_splitter import RecursiveCharacterTextSplitter
from core.vector_store import PineconeStore
from core.embeddings import Embedder
from config.env_config import PINECONE_BD_PHARMACY_INDEX, DATA_DIR  # default target index

# Silence pdfminer noise
logging.getLogger("pdfminer").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

INDEX_NAME = PINECONE_BD_PHARMACY_INDEX

HDR_FTR_RE = re.compile(r"^\s*(page\s*\d+|[\-–—]\s*\d+\s*[\-–—]|Copyright.*|©.*)$", re.I)

# Detect Tesseract once
TESSERACT_EXE = shutil.which("tesseract") or os.getenv("TESSERACT_PATH")
OCR_AVAILABLE = bool(TESSERACT_EXE)
if OCR_AVAILABLE:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_EXE
else:
    logger.info("[INFO] Tesseract not found; proceeding without OCR fallback.")


def discover_pdfs() -> List[Tuple[str, str, str]]:
    """
    Dynamically discover PDFs in data/pdfs/ and data/pdfs/anatomy/ directories.
    Returns list of (path, namespace, title) tuples.
    """
    pdf_configs = []
    
    # Define search directories and their default namespaces
    search_dirs = [
        (DATA_DIR / "pdfs", "general"),
        (DATA_DIR / "pdfs" / "anatomy", "anatomy"),
    ]
    
    for search_dir, default_namespace in search_dirs:
        if not search_dir.exists():
            logger.info(f"Directory not found, skipping: {search_dir}")
            continue
            
        pdf_files = list(search_dir.glob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDFs in {search_dir}")
        
        for pdf_path in pdf_files:
            # Generate title from filename
            title = pdf_path.stem.replace("_", " ").replace("-", " ").title()
            
            # Determine namespace based on content or directory
            namespace = _determine_namespace(pdf_path, default_namespace)
            
            pdf_configs.append((str(pdf_path), namespace, title))
            logger.info(f"  - {pdf_path.name} -> namespace: {namespace}, title: {title}")
    
    if not pdf_configs:
        logger.warning("No PDFs found in any search directories")
        
    return pdf_configs


def _determine_namespace(pdf_path: Path, default_namespace: str) -> str:
    """
    Determine appropriate namespace based on file name and location.
    """
    filename_lower = pdf_path.name.lower()
    
    # Namespace mapping based on keywords in filename
    namespace_keywords = {
        "prescribing": ["prescribing", "prescription", "who guide"],
        "otc": ["otc", "over-the-counter", "over the counter"],
        "guidelines": ["guideline", "protocol", "management", "hypertension"],
        "policy": ["policy", "law", "drug policy", "cosmetics law"],
        "textbook": ["anatomy", "textbook", "atlas", "physiology"],
        "anatomy": ["anatomy", "atlas", "physiology", "human body"],
    }
    
    # Check filename against keywords
    for namespace, keywords in namespace_keywords.items():
        if any(keyword in filename_lower for keyword in keywords):
            return namespace
    
    return default_namespace


def _sentinel_for(pdf_path: Path) -> Path:
    h = hashlib.md5(str(pdf_path.resolve()).encode("utf-8")).hexdigest()
    return Path(f".ingested_{INDEX_NAME}_{h}.done")


def ocr_image(img_or_pil):
    if not OCR_AVAILABLE:
        return ""
    pil = img_or_pil if isinstance(img_or_pil, Image.Image) else Image.open(io.BytesIO(img_or_pil))
    pil = pil.convert("RGB")
    return pytesseract.image_to_string(pil, lang="eng")


def page_to_text(pl_page):
    # try structured text first
    txt = (pl_page.extract_text() or "").strip()
    if txt:
        return txt
    # OCR fallback only if available
    if OCR_AVAILABLE:
        img = pl_page.to_image(resolution=300).original  # np array
        return ocr_image(img)
    return ""


def clean_text(t: str) -> str:
    lines = []
    for ln in t.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        if HDR_FTR_RE.match(ln):
            continue
        ln = re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", ln)
        lines.append(ln)
    out = " ".join(lines)
    out = re.sub(r"\s+", " ", out)
    out = out.replace("µg", "mcg").replace("μg", "mcg")
    out = out.replace("millilitre", "mL").replace("Millilitre", "mL")
    return out.strip()


splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200, chunk_overlap=200,
    separators=["\n## ", "\n# ", "\n\n", "\n- ", ". "]
)


def ingest_pdf(pdf_path: str, namespace: str, title: str, store: PineconeStore, skip_existing: bool = False):
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        logger.warning(f"PDF not found: {pdf_path}")
        return

    sentinel = _sentinel_for(pdf_path)
    if skip_existing and sentinel.exists():
        logger.info(f"[SKIP] {pdf_path.name} (sentinel present)")
        return

    doc_id = f"{title}-{uuid.uuid5(uuid.NAMESPACE_URL, str(pdf_path))}"

    texts, metas = [], []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for pi, p in enumerate(pdf.pages, start=1):
                raw = page_to_text(p)
                cleaned = clean_text(raw)
                if not cleaned:
                    continue
                page_chunks = splitter.split_text(cleaned)
                for ch in page_chunks:
                    meta = {
                        "doc_id": doc_id,
                        "namespace": namespace,
                        "title": title,
                        "section": "General",
                        "page_start": pi,
                        "page_end": pi,
                        "country": "Bangladesh",
                        "safety": "authoritative" if namespace in ("guidelines", "policy", "prescribing") else "reference",
                        "source": str(pdf_path.name),
                    }
                    texts.append(ch)
                    metas.append(meta)
    except Exception as e:
        logger.error(f"Error processing PDF {pdf_path}: {e}")
        return

    if not texts:
        logger.warning(f"[WARN] No text extracted from {pdf_path.name}")
        return

    # Try upsert; CUDA errors => switch to CPU and retry once
    try:
        store.upsert_texts(texts, metas, namespace=namespace)
    except RuntimeError as e:
        msg = str(e).lower()
        if "cuda" in msg or "cublas" in msg:
            logger.warning("[WARN] CUDA error during upsert; retrying on CPU...")
            os.environ["EMBEDDINGS_DEVICE"] = "cpu"
            # rebuild singleton on CPU
            store.embedder = Embedder.reset()
            store.upsert_texts(texts, metas, namespace=namespace)
        else:
            raise

    logger.info(f"Indexed: {title} ({namespace}) — {pdf_path.name}  chunks={len(texts)}")
    try:
        sentinel.write_text("ok")
    except Exception:
        pass


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Ingest BD PDFs into Pinecone")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto",
                    help="Device for SentenceTransformer during ingestion (default: auto)")
    ap.add_argument("--batch", type=int, default=None,
                    help="Override embedding batch size (env PINECONE_BATCH)")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip PDFs that already have a local sentinel")
    ap.add_argument("--pdf-dir", type=str, default=None,
                    help="Override default PDF search directories")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.device != "auto":
        os.environ["EMBEDDINGS_DEVICE"] = args.device
    if args.batch:
        os.environ["PINECONE_BATCH"] = str(args.batch)

    # Discover PDFs dynamically
    pdf_configs = discover_pdfs()
    
    if not pdf_configs:
        logger.error("No PDFs found to process")
        exit(1)

    store = PineconeStore(index_name=INDEX_NAME, dimension=384)
    
    for path, ns, title in pdf_configs:
        try:
            ingest_pdf(path, ns, title, store, skip_existing=args.skip_existing)
        except Exception as e:
            logger.error(f"[ERROR] {path}: {e}")
