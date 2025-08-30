# scripts/ingest_pdfs_bd.py
import os, re, uuid, io, shutil, logging
from pathlib import Path
import pdfplumber
from PIL import Image
import pytesseract
from langchain.text_splitter import RecursiveCharacterTextSplitter
from core.vector_store import PineconeStore
from config.env_config import PINECONE_BD_PHARMACY_INDEX  # default target index

# Silence pdfminer noise
logging.getLogger("pdfminer").setLevel(logging.ERROR)

# --------- CONFIG: map your PDFs to namespaces/titles ----------
DOCS = [
    # path,                                                                namespace,       title
    ("data/pdfs/WHO Guide to Good Prescribing.pdf",                        "prescribing",   "WHO Guide to Good Prescribing"),
    ("data/pdfs/Over-the-Counter (OTC) Medicines List (official list).pdf","otc",           "Bangladesh OTC Medicines List"),
    ("data/pdfs/National Protocol for Management of.pdf",                  "guidelines",    "National Protocols for Management"),
    ("data/pdfs/National Guideline on Hypertension Bangladesh.pdf",        "guidelines",    "National Guideline on Hypertension (Bangladesh)"),
    ("data/pdfs/National Drug Policy, Bangladesh (2016).pdf",              "policy",        "National Drug Policy (2016)"),
    ("data/pdfs/Drugs & Cosmetics Law.pdf",                                "policy",        "Drugs & Cosmetics Law"),
    ("data/pdfs/A_Study_on_the_National_Drug_Policies_of_Banglades.pdf",   "policy",        "Study on National Drug Policies of Bangladesh"),
    # <- this file lives under the anatomy subfolder in your repo
    ("data/pdfs/anatomy/Human Anatomy.pdf",                                "textbook",      "Textbook of Human Anatomy"),
]

INDEX_NAME = PINECONE_BD_PHARMACY_INDEX

HDR_FTR_RE = re.compile(r"^\s*(page\s*\d+|[\-–—]\s*\d+\s*[\-–—]|Copyright.*|©.*)$", re.I)

# Detect Tesseract once
TESSERACT_EXE = shutil.which("tesseract") or os.getenv("TESSERACT_PATH")
OCR_AVAILABLE = bool(TESSERACT_EXE)
if OCR_AVAILABLE:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_EXE
else:
    print("[INFO] Tesseract not found; proceeding without OCR fallback.")

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

def ingest_pdf(pdf_path: str, namespace: str, title: str, store: PineconeStore):
    pdf_path = Path(pdf_path)
    assert pdf_path.exists(), f"Missing: {pdf_path}"
    doc_id = f"{title}-{uuid.uuid5(uuid.NAMESPACE_URL, str(pdf_path))}"

    texts, metas = [], []
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

    if not texts:
        print(f"[WARN] No text in {pdf_path.name}")
        return

    store.upsert_texts(texts, metas, namespace=namespace)
    print(f"Indexed: {title} ({namespace}) — {pdf_path.name}  chunks={len(texts)}")

if __name__ == "__main__":
    store = PineconeStore(index_name=INDEX_NAME, dimension=384)
    for path, ns, title in DOCS:
        try:
            ingest_pdf(path, ns, title, store)
        except Exception as e:
            print(f"[ERROR] {path}: {e}")
