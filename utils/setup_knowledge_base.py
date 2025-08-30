# utils/setup_knowledge_base.py
"""
Setup script for knowledge base ingestion
- PDF text  -> PINECONE_PDF_INDEX
- Medicine  -> PINECONE_MEDICINE_INDEX (generic.csv + medicine.csv)
- BD PDFs   -> via scripts/ingest_pdfs_bd.py (namespaced into arobot-bd-pharmacy)
- Anatomy   -> via scripts/ingest_anatomy_images.py (CLIP -> arobot-clip)
"""

import os
import sys
import re
import logging
import subprocess
from pathlib import Path
from typing import List, Dict

import pandas as pd

from core.vector_store import PineconeStore
from core.embeddings import Embedder  # ensure embedder is importable
from .data_ingestion import load_pdfs
from config.env_config import (
    PINECONE_API_KEY,
    PINECONE_PDF_INDEX,
    PINECONE_MEDICINE_INDEX,
    DATA_DIR,
    WEB_SCRAPE_DIR,
    EMBEDDING_DIMENSION,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------
# Helpers
# ---------------------------

def _clean_html(text: str) -> str:
    if pd.isna(text):
        return ""
    txt = re.sub(r"<.*?>", " ", str(text))
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt


def _read_csv_safe(path: Path) -> pd.DataFrame:
    """Read CSV with UTF-8 / UTF-8-SIG fallback and normalized columns."""
    if not path.exists():
        raise FileNotFoundError(f"Missing CSV: {path}")
    try:
        df = pd.read_csv(path)
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [re.sub(r"\s+", " ", c.strip().lower()) for c in df.columns]
    return df


def _require_columns(df: pd.DataFrame, needed: List[str], label: str) -> None:
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise KeyError(f"{label}: missing columns {missing}. Present: {list(df.columns)}")


def _concat_parts(parts: List[str]) -> str:
    return " | ".join([p for p in parts if p and str(p).strip()])


# ---------------------------
# PDF KB
# ---------------------------

def setup_pdf_knowledge_base() -> bool:
    """Setup PDF knowledge base in Pinecone from DATA_DIR/pdfs."""
    try:
        logger.info("📚 Setting up PDF knowledge base...")
        if not DATA_DIR.exists():
            logger.error(f"DATA_DIR not found: {DATA_DIR}")
            return False

        dim = EMBEDDING_DIMENSION or 384
        pdf_store = PineconeStore(index_name=PINECONE_PDF_INDEX, dimension=dim)

        logger.info(f"Loading PDFs from: {DATA_DIR}")
        docs = load_pdfs(str(DATA_DIR))
        if not docs:
            logger.warning("No PDF documents were discovered.")
            return False

        texts = [d["text"] for d in docs]
        metadatas = [d["meta"] for d in docs]

        total_chars = sum(len(t) for t in texts)
        avg_chunk = total_chars // max(1, len(texts))
        logger.info("📊 PDF chunking:")
        logger.info(f"   • Total chunks: {len(texts)}")
        logger.info(f"   • Average chunk size: {avg_chunk} chars")
        logger.info(f"   • Total text: {total_chars:,} chars")
        logger.info(f"   • Estimated batches: {(len(texts) + 63) // 64}")

        logger.info("🚀 Upserting PDF chunks to Pinecone...")
        pdf_store.upsert_texts(texts, metadatas)
        logger.info(f"✅ PDF KB indexed to: {PINECONE_PDF_INDEX}")
        return True

    except Exception as e:
        logger.error(f"Error setting up PDF knowledge base: {e}")
        return False


# ---------------------------
# Medicine CSV KB
# ---------------------------

def setup_medicine_knowledge_base() -> bool:
    """
    Setup medicine CSV knowledge base in Pinecone from:
      WEB_SCRAPE_DIR/generic.csv
      WEB_SCRAPE_DIR/medicine.csv
    Produces a 'medicine_description' per row and upserts to PINECONE_MEDICINE_INDEX.
    """
    try:
        logger.info("💊 Setting up medicine knowledge base...")

        generic_path = WEB_SCRAPE_DIR / "generic.csv"
        medicine_path = WEB_SCRAPE_DIR / "medicine.csv"
        logger.info(f"Reading: {generic_path}")
        logger.info(f"Reading: {medicine_path}")

        generic_df = _read_csv_safe(generic_path)
        medicine_df = _read_csv_safe(medicine_path)

        _require_columns(generic_df, ["generic name"], "generic.csv")
        _require_columns(medicine_df, ["generic"], "medicine.csv")

        desc_cols = [
            "indication description",
            "therapeutic class description",
            "pharmacology description",
            "dosage description",
            "side effects description",
            "contraindications description",
        ]

        for c in desc_cols:
            if c in medicine_df.columns:
                medicine_df[c] = medicine_df[c].map(_clean_html)

        merged = pd.merge(
            generic_df, medicine_df,
            left_on="generic name", right_on="generic",
            how="inner",
        )

        if merged.empty:
            logger.warning("Merged medicine dataframe is empty after join.")
            return False

        logger.info(f"🔗 Merged rows: {len(merged)}")

        def build_description(row: pd.Series) -> str:
            parts = []
            if "generic name" in row and pd.notna(row["generic name"]):
                parts.append(f"Generic Name: {row['generic name']}")
            for col, label in [
                ("brand name", "Brand Name"),
                ("drug class", "Drug Class"),
                ("strength", "Strength"),
                ("dosage form", "Dosage Form"),
            ]:
                if col in row and pd.notna(row[col]) and str(row[col]).strip():
                    parts.append(f"{label}: {str(row[col]).strip()}")
            for c in desc_cols:
                if c in row and pd.notna(row[c]) and len(str(row[c]).strip()) > 10:
                    parts.append(f"{c.title()}: {str(row[c]).strip()}")
            return _concat_parts(parts)

        merged["medicine_description"] = merged.apply(build_description, axis=1)

        texts: List[str] = []
        metas: List[Dict] = []

        for i, row in merged.iterrows():
            desc = row.get("medicine_description", "")
            if not desc or len(desc) < 50:
                continue
            texts.append(desc)
            metas.append({
                "id": f"medicine_{i}",
                "generic_name": str(row.get("generic name", "")).strip(),
                "brand_name": str(row.get("brand name", "")).strip() if "brand name" in row else "",
                "drug_class": str(row.get("drug class", "")).strip() if "drug class" in row else "",
                "source": "medicine_database",
            })

        if not texts:
            logger.warning("No medicine records passed the quality threshold.")
            return False

        total_chars = sum(len(t) for t in texts)
        avg_len = total_chars // max(1, len(texts))
        logger.info("📊 Medicine data:")
        logger.info(f"   • Records prepared: {len(texts)}")
        logger.info(f"   • Avg record size: {avg_len} chars")
        logger.info(f"   • Total text: {total_chars:,} chars")
        logger.info(f"   • Estimated batches: {(len(texts) + 63) // 64}")

        dim = EMBEDDING_DIMENSION or 384
        store = PineconeStore(index_name=PINECONE_MEDICINE_INDEX, dimension=dim)
        logger.info("🚀 Upserting medicine records to Pinecone...")
        store.upsert_texts(texts, metas)

        logger.info(f"✅ Medicine KB indexed to: {PINECONE_MEDICINE_INDEX}")
        return True

    except Exception as e:
        logger.error(f"Error setting up medicine knowledge base: {e}")
        return False


# ---------------------------
# Verification
# ---------------------------

def verify_knowledge_bases() -> bool:
    try:
        logger.info("🔎 Verifying KBs with sample queries...")

        dim = EMBEDDING_DIMENSION or 384

        pdf_store = PineconeStore(index_name=PINECONE_PDF_INDEX, dimension=dim)
        pdf_hits = pdf_store.query("medical anatomy", top_k=3)
        logger.info(f"PDF KB returned {len(pdf_hits)} results")

        med_store = PineconeStore(index_name=PINECONE_MEDICINE_INDEX, dimension=dim)
        med_hits = med_store.query("diabetes medication", top_k=3)
        logger.info(f"Medicine KB returned {len(med_hits)} results")

        ok = (len(pdf_hits) > 0) and (len(med_hits) > 0)
        if not ok:
            logger.warning("Verification incomplete: one of the KBs returned 0 results.")
        return ok
    except Exception as e:
        logger.error(f"Verification error: {e}")
        return False


# ---------------------------
# Child-process helpers
# ---------------------------

def run(cmd: List[str]) -> None:
    """Run a child process from project root with PYTHONPATH set."""
    print(">>>", " ".join(cmd), flush=True)
    env = os.environ.copy()

    # Ensure project root is on PYTHONPATH for child processes
    root = Path(__file__).resolve().parents[1]
    env["PYTHONPATH"] = (env.get("PYTHONPATH", "") + os.pathsep + str(root)).strip(os.pathsep)

    # Run from project root so relative paths (data/, scripts/) resolve
    code = subprocess.call(cmd, env=env, cwd=str(root))
    if code != 0:
        raise SystemExit(code)


# ---------------------------
# Entrypoint
# ---------------------------

def main() -> bool:
    logger.info("🚦 Starting knowledge base setup")
    if not PINECONE_API_KEY:
        logger.error("PINECONE_API_KEY is not set.")
        return False

    ok = True
    if not setup_pdf_knowledge_base():
        ok = False
    if not setup_medicine_knowledge_base():
        ok = False

    # Run the two specialized ingesters as part of the setup
    try:
        run([sys.executable, "scripts/ingest_pdfs_bd.py"])        # BD PDFs -> arobot-bd-pharmacy (namespaces)
        run([sys.executable, "scripts/ingest_anatomy_images.py"]) # Anatomy figures -> arobot-clip (CLIP)
    except SystemExit as e:
        logger.error(f"Ingestion script failed with exit code {e.code if hasattr(e, 'code') else e}")
        ok = False

    # Verify
    if ok and verify_knowledge_bases():
        logger.info("✅ Knowledge base setup completed successfully!")
        return True

    logger.warning("⚠️ Setup completed with warnings or verification failed.")
    return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
