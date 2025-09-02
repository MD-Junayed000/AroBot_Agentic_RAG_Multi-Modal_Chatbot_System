# utils/setup_knowledge_base.py
"""
Setup script for knowledge base ingestion
- PDF text  -> Indexed under namespace 'pdfs' in PINECONE_BD_PHARMACY_INDEX
- Medicine  -> PINECONE_MEDICINE_INDEX (generic.csv + medicine.csv + dosage_form.csv + drug_class.csv + indication.csv + manufacturer.csv)
- BD PDFs   -> via scripts/ingest_pdfs_bd.py (namespaced into arobot-bd-pharmacy)
- Anatomy   -> via scripts/ingest_anatomy_images.py (CLIP -> arobot-clip)
"""

import os
import sys
import re
import logging
import subprocess
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import pandas as pd

from core.vector_store import PineconeStore
from core.embeddings import Embedder  # ensure embedder importable
from .data_ingestion import load_pdfs
from config.env_config import (
    PINECONE_API_KEY,
    PINECONE_MEDICINE_INDEX,
    PINECONE_BD_PHARMACY_INDEX,
    DATA_DIR,
    WEB_SCRAPE_DIR,
    EMBEDDING_DIMENSION,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------------------
# Generic helpers
# --------------------------------------------------------------------------------------

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
    # normalize header names
    df.columns = [re.sub(r"\s+", " ", c.strip().lower()) for c in df.columns]
    return df


def _require_columns(df: pd.DataFrame, needed: List[str], label: str) -> None:
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise KeyError(f"{label}: missing columns {missing}. Present: {list(df.columns)}")


def _concat_parts(parts: List[str]) -> str:
    return " | ".join([p for p in parts if p and str(p).strip()])


def _first_present(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _norm_str(s: Optional[str]) -> str:
    if pd.isna(s) or s is None:
        return ""
    return re.sub(r"\s+", " ", str(s).strip())


def _dedupe_nonempty(seq: List[str]) -> List[str]:
    seen, out = set(), []
    for x in seq:
        xs = _norm_str(x)
        if xs and xs not in seen:
            seen.add(xs)
            out.append(xs)
    return out


# --------------------------------------------------------------------------------------
# PDF KB (now indexes into BD Pharmacy index under namespace 'pdfs')
# --------------------------------------------------------------------------------------

def setup_pdf_knowledge_base() -> bool:
    """Setup PDF pages into the BD Pharmacy index under namespace 'pdfs'."""
    try:
        logger.info("📚 Setting up PDF knowledge base...")
        if not DATA_DIR.exists():
            logger.error(f"DATA_DIR not found: {DATA_DIR}")
            return False

        dim = EMBEDDING_DIMENSION or 384
        pdf_store = PineconeStore(index_name=PINECONE_BD_PHARMACY_INDEX, dimension=dim)

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

        logger.info("🚀 Upserting PDF chunks to Pinecone (namespace='pdfs')...")
        upserted = pdf_store.upsert_texts(texts, metadatas, namespace="pdfs")
        logger.info(f"✅ PDF KB indexed to: {PINECONE_BD_PHARMACY_INDEX} (upserted={upserted})")
        return True

    except Exception as e:
        logger.error(f"Error setting up PDF knowledge base: {e}")
        return False


# --------------------------------------------------------------------------------------
# Medicine KB (extended to use 6 CSVs)
# --------------------------------------------------------------------------------------

def _load_base_tables() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load the original generic + medicine base tables."""
    generic_path = WEB_SCRAPE_DIR / "generic.csv"
    medicine_path = WEB_SCRAPE_DIR / "medicine.csv"

    logger.info(f"Reading: {generic_path}")
    logger.info(f"Reading: {medicine_path}")

    generic_df = _read_csv_safe(generic_path)
    medicine_df = _read_csv_safe(medicine_path)

    # Required columns with flexible naming
    gen_key = _first_present(generic_df, ["generic name", "generic"])
    med_key = _first_present(medicine_df, ["generic", "generic name"])

    if not gen_key:
        raise KeyError("generic.csv needs a 'generic name' (or 'generic') column.")
    if not med_key:
        raise KeyError("medicine.csv needs a 'generic' (or 'generic name') column.")

    # normalize join keys as 'generic'
    generic_df = generic_df.rename(columns={gen_key: "generic"})
    medicine_df = medicine_df.rename(columns={med_key: "generic"})

    # Clean long description fields in medicine_df
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

    return generic_df, medicine_df


def _load_enrichment_tables() -> Dict[str, pd.DataFrame]:
    """
    Load dosage_form.csv, drug_class.csv, indication.csv, manufacturer.csv
    and normalize likely columns into a tidy shape.
    """
    files = {
        "dosage_form": WEB_SCRAPE_DIR / "dosage_form.csv",
        "drug_class": WEB_SCRAPE_DIR / "drug_class.csv",
        "indication": WEB_SCRAPE_DIR / "indication.csv",
        "manufacturer": WEB_SCRAPE_DIR / "manufacturer.csv",
    }
    dfs: Dict[str, pd.DataFrame] = {}
    for label, path in files.items():
        logger.info(f"Reading: {path}")
        dfs[label] = _read_csv_safe(path)
    return dfs


def _tidy_dosage_form(df: pd.DataFrame) -> pd.DataFrame:
    # try to find columns
    gen = _first_present(df, ["generic", "generic name"])
    brand = _first_present(df, ["brand", "brand name"])
    form = _first_present(df, ["dosage form", "form", "dosage_form"])
    keep = {}
    if gen: keep["generic"] = df[gen].map(_norm_str)
    if brand: keep["brand"] = df[brand].map(_norm_str)
    if form: keep["dosage_form"] = df[form].map(_norm_str)
    tidy = pd.DataFrame(keep)
    tidy = tidy.dropna(how="all")
    return tidy


def _tidy_drug_class(df: pd.DataFrame) -> pd.DataFrame:
    gen = _first_present(df, ["generic", "generic name"])
    brand = _first_present(df, ["brand", "brand name"])
    dclass = _first_present(df, ["drug class", "class", "therapeutic class"])
    keep = {}
    if gen: keep["generic"] = df[gen].map(_norm_str)
    if brand: keep["brand"] = df[brand].map(_norm_str)
    if dclass: keep["drug_class"] = df[dclass].map(_norm_str)
    tidy = pd.DataFrame(keep).dropna(how="all")
    return tidy


def _tidy_indication(df: pd.DataFrame) -> pd.DataFrame:
    gen = _first_present(df, ["generic", "generic name"])
    brand = _first_present(df, ["brand", "brand name"])
    indic = _first_present(df, ["indication", "indications"])
    keep = {}
    if gen: keep["generic"] = df[gen].map(_norm_str)
    if brand: keep["brand"] = df[brand].map(_norm_str)
    if indic: keep["indication"] = df[indic].map(_norm_str)
    tidy = pd.DataFrame(keep).dropna(how="all")
    return tidy


def _tidy_manufacturer(df: pd.DataFrame) -> pd.DataFrame:
    gen = _first_present(df, ["generic", "generic name"])
    brand = _first_present(df, ["brand", "brand name"])
    manuf = _first_present(df, ["manufacturer", "company", "maker", "manufacturer name"])
    keep = {}
    if gen: keep["generic"] = df[gen].map(_norm_str)
    if brand: keep["brand"] = df[brand].map(_norm_str)
    if manuf: keep["manufacturer"] = df[manuf].map(_norm_str)
    tidy = pd.DataFrame(keep).dropna(how="all")
    return tidy


def _aggregate_grouped(series: pd.Series) -> str:
    return "; ".join(_dedupe_nonempty(list(series.dropna().astype(str))))


def _merge_medicine_tables(
    generic_df: pd.DataFrame,
    medicine_df: pd.DataFrame,
    enrichment: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Merge generic+medicine with (dosage_form, drug_class, indication, manufacturer).
    Join primarily on 'generic'; where brand exists we create brand-level variants too.
    """
    # tidy enrichment tables
    df_form = _tidy_dosage_form(enrichment["dosage_form"])
    df_class = _tidy_drug_class(enrichment["drug_class"])
    df_ind = _tidy_indication(enrichment["indication"])
    df_manu = _tidy_manufacturer(enrichment["manufacturer"])

    # base left-join by generic
    merged = pd.merge(
        generic_df, medicine_df, on="generic", how="inner", suffixes=("", "_med")
    )

    # generic-level merges (collapse multiple rows per generic)
    def collapse_generic_levels(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
        if "generic" not in df.columns or value_col not in df.columns:
            return pd.DataFrame(columns=["generic", value_col])
        agg = df.groupby("generic", dropna=True)[value_col].apply(_aggregate_grouped).reset_index()
        return agg

    g_form = collapse_generic_levels(df_form, "dosage_form")
    g_class = collapse_generic_levels(df_class, "drug_class")
    g_ind = collapse_generic_levels(df_ind, "indication")
    g_manu = collapse_generic_levels(df_manu, "manufacturer")

    for extra in [g_form, g_class, g_ind, g_manu]:
        if not extra.empty:
            merged = pd.merge(merged, extra, on="generic", how="left")

    # brand-level enrichments (optional): if brand columns exist anywhere, enrich per brand
    brand_cols = []
    if "brand name" in merged.columns:
        brand_cols.append("brand name")
    if "brand" in merged.columns:
        brand_cols.append("brand")

    # prefer a single normalized 'brand' column for downstream
    if brand_cols:
        first_brand_col = brand_cols[0]
        merged["brand"] = merged[first_brand_col].map(_norm_str)

        # brand enrich
        def collapse_brand_levels(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
            if "brand" not in df.columns or value_col not in df.columns:
                return pd.DataFrame(columns=["brand", value_col])
            agg = df.groupby("brand", dropna=True)[value_col].apply(_aggregate_grouped).reset_index()
            return agg

        b_form = collapse_brand_levels(df_form, "dosage_form")
        b_class = collapse_brand_levels(df_class, "drug_class")
        b_ind = collapse_brand_levels(df_ind, "indication")
        b_manu = collapse_brand_levels(df_manu, "manufacturer")

        for b_extra in [b_form, b_class, b_ind, b_manu]:
            if not b_extra.empty:
                # brand-level columns suffixed to avoid clobbering generic-level
                col = [c for c in b_extra.columns if c != "brand"][0]
                merged = pd.merge(
                    merged, b_extra.rename(columns={col: f"{col}_by_brand"}), on="brand", how="left"
                )

    return merged


def _build_medicine_description(row: pd.Series) -> str:
    parts = []

    # Key identity
    if pd.notna(row.get("generic")) and _norm_str(row.get("generic")):
        parts.append(f"Generic Name: {row['generic']}")
    if "brand" in row and _norm_str(row.get("brand")):
        parts.append(f"Brand Name: {row['brand']}")
    elif "brand name" in row and _norm_str(row.get("brand name")):
        parts.append(f"Brand Name: {row['brand name']}")

    # Structure / class / forms
    for col, label in [
        ("drug class", "Drug Class"),
        ("drug_class", "Drug Class"),
        ("drug_class_by_brand", "Drug Class (Brand)"),
        ("dosage form", "Dosage Form"),
        ("dosage_form", "Dosage Form"),
        ("dosage_form_by_brand", "Dosage Form (Brand)"),
        ("strength", "Strength"),
        ("dosage form_med", "Dosage Form"),
    ]:
        if col in row and _norm_str(row.get(col)):
            parts.append(f"{label}: {row[col]}")

    # Manufacturer
    for col, label in [
        ("manufacturer", "Manufacturer"),
        ("manufacturer_by_brand", "Manufacturer (Brand)"),
        ("company", "Manufacturer"),
    ]:
        if col in row and _norm_str(row.get(col)):
            parts.append(f"{label}: {row[col]}")

    # Indications
    for col in [
        "indication",
        "indication_by_brand",
        "indication description",
    ]:
        if col in row and _norm_str(row.get(col)):
            parts.append(f"Indications: {_norm_str(row.get(col))}")

    # Pharmacology / Dosage / Side effects / Contraindications (from original medicine.csv)
    for col, label in [
        ("pharmacology description", "Pharmacology"),
        ("dosage description", "Dosage & Administration"),
        ("side effects description", "Common Side Effects"),
        ("contraindications description", "Contraindications"),
        ("therapeutic class description", "Therapeutic Class"),
    ]:
        if col in row and _norm_str(row.get(col)):
            parts.append(f"{label}: {row[col]}")

    return _concat_parts(parts)


def setup_medicine_knowledge_base() -> bool:
    """
    Setup medicine CSV knowledge base in Pinecone from:
      Web Scrape/generic.csv
      Web Scrape/medicine.csv
      Web Scrape/dosage_form.csv
      Web Scrape/drug_class.csv
      Web Scrape/indication.csv
      Web Scrape/manufacturer.csv
    """
    try:
        logger.info("💊 Setting up medicine knowledge base (extended)…")

        generic_df, medicine_df = _load_base_tables()
        enrichment = _load_enrichment_tables()

        merged = _merge_medicine_tables(generic_df, medicine_df, enrichment)
        if merged.empty:
            logger.warning("Merged medicine dataframe is empty after joining enrichment tables.")
            return False

        logger.info(f"🔗 Merged rows (generic × medicine × enrichment): {len(merged)}")

        merged["medicine_description"] = merged.apply(_build_medicine_description, axis=1)

        # Filter low-quality rows
        rows: List[Dict] = []
        for i, row in merged.iterrows():
            desc = _norm_str(row.get("medicine_description"))
            if len(desc) < 50:
                continue
            meta = {
                "id": f"medicine_ext_{i}",
                "generic_name": _norm_str(row.get("generic")),
                "brand_name": _norm_str(row.get("brand")) or _norm_str(row.get("brand name")),
                "drug_class": _norm_str(row.get("drug_class")) or _norm_str(row.get("drug class")),
                "dosage_form": _norm_str(row.get("dosage_form")) or _norm_str(row.get("dosage form")),
                "manufacturer": _norm_str(row.get("manufacturer")) or _norm_str(row.get("company")),
                "source": "medicine_database_extended",
            }
            rows.append({"text": desc, "meta": meta})

        if not rows:
            logger.warning("No medicine records passed the quality threshold.")
            return False

        texts = [r["text"] for r in rows]
        metas = [r["meta"] for r in rows]

        total_chars = sum(len(t) for t in texts)
        avg_len = total_chars // max(1, len(texts))
        logger.info("📊 Medicine data:")
        logger.info(f"   • Records prepared: {len(texts)}")
        logger.info(f"   • Avg record size: {avg_len} chars")
        logger.info(f"   • Total text: {total_chars:,} chars")
        logger.info(f"   • Estimated batches: {(len(texts) + 63) // 64}")

        dim = EMBEDDING_DIMENSION or 384
        store = PineconeStore(index_name=PINECONE_MEDICINE_INDEX, dimension=dim)
        logger.info("🚀 Upserting medicine records to Pinecone…")
        store.upsert_texts(texts, metas)

        logger.info(f"✅ Medicine KB indexed to: {PINECONE_MEDICINE_INDEX}")
        return True

    except Exception as e:
        logger.error(f"Error setting up medicine knowledge base: {e}")
        return False


# --------------------------------------------------------------------------------------
# Verification (unchanged logic, still useful)
# --------------------------------------------------------------------------------------

def verify_knowledge_bases() -> bool:
    try:
        logger.info("🔎 Verifying KBs with sample queries...")

        dim = EMBEDDING_DIMENSION or 384

        bd_store = PineconeStore(index_name=PINECONE_BD_PHARMACY_INDEX, dimension=dim)
        pdf_hits = bd_store.query("hypertension guideline Bangladesh", top_k=3)
        logger.info(f"BD/Guidelines KB returned {len(pdf_hits)} results")

        med_store = PineconeStore(index_name=PINECONE_MEDICINE_INDEX, dimension=dim)
        med_hits = med_store.query("diabetes medication dosage form manufacturer", top_k=3)
        logger.info(f"Medicine KB returned {len(med_hits)} results")

        ok = (len(pdf_hits) > 0) and (len(med_hits) > 0)
        if not ok:
            logger.warning("Verification incomplete: one of the KBs returned 0 results.")
        return ok
    except Exception as e:
        logger.error(f"Verification error: {e}")
        return False


# --------------------------------------------------------------------------------------
# Child-process helpers
# --------------------------------------------------------------------------------------

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


# --------------------------------------------------------------------------------------
# Entrypoint
# --------------------------------------------------------------------------------------
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
        # Force CPU for PDF ingestion to avoid CUDA/CUBLAS init errors,
        # use smaller batches, and skip PDFs that were already ingested.
        run([sys.executable, "scripts/ingest_pdfs_bd.py", "--device", "cpu", "--batch", "16", "--skip-existing"])

        # Anatomy figures (CLIP index). Leave as-is unless you also want to force CPU here.
        run([sys.executable, "scripts/ingest_anatomy_images.py"])
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
