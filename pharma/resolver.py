# pharma/resolver.py
from __future__ import annotations
import os
import sqlite3
import time
import re
from typing import Dict, Any, Optional, List, Tuple

from rapidfuzz import process, fuzz

# NOTE: folder is "provider" (singular) in your tree
from .provider.medex import search_brand_or_generic, parse_brand_page

# Optional alias map (Bangladesh brand -> generic) to reduce ambiguity (e.g., Napa != Naproxen)
try:
    from utils.brand_aliases_bd import BRAND_TO_GENERIC_BD  # {"napa": "paracetamol", ...}
except Exception:
    BRAND_TO_GENERIC_BD = {}

DB_PATH = os.environ.get("PHARMA_DB", "data/pharma.db")
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

# Cache TTLs
BRAND_TTL_SECS = 14 * 24 * 3600   # 14 days for brand metadata
PRICE_TTL_SECS = 3 * 24 * 3600    # refresh prices every ~3 days


# ----------------------------- SQLite helpers ----------------------------- #
def _conn():
    return sqlite3.connect(DB_PATH)


def _ensure_schema():
    with _conn() as c:
        c.execute(
            """CREATE TABLE IF NOT EXISTS brands(
                brand TEXT PRIMARY KEY,
                generic TEXT, form TEXT, strength TEXT, company TEXT, url TEXT,
                source TEXT, ts INTEGER
            )"""
        )
        c.execute(
            """CREATE TABLE IF NOT EXISTS prices(
                brand TEXT, pack TEXT, price_text TEXT,
                url TEXT, source TEXT, ts INTEGER,
                PRIMARY KEY (brand, pack)
            )"""
        )
        c.commit()


_ensure_schema()


# ----------------------------- Normalization ------------------------------ #
def _norm(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9+.\s-]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s


def _expand_terms(term: str) -> List[str]:
    """
    Expand a user term with helpful variants:
    - Alias map (BD brand -> generic)
    - Obvious case variants
    """
    q = _norm(term)
    out = [q]
    # alias: brand -> generic
    if q in BRAND_TO_GENERIC_BD:
        out.append(_norm(BRAND_TO_GENERIC_BD[q]))
    # Basic trick: if looks like "brand 500", also try without number
    base = re.sub(r"\b\d+(\.\d+)?\s*(mg|mcg|g|ml)\b", "", q).strip()
    if base and base not in out:
        out.append(base)
    # Deduplicate
    seen, uniq = set(), []
    for t in out:
        if t not in seen and t:
            seen.add(t)
            uniq.append(t)
    return uniq


# ------------------------------ DB helpers -------------------------------- #
def _get_brand_local(brand: str) -> Optional[Dict[str, Any]]:
    with _conn() as c:
        row = c.execute(
            "SELECT brand,generic,form,strength,company,url,source,ts FROM brands WHERE brand=?",
            (brand,),
        ).fetchone()
        if not row:
            return None
        keys = ["brand", "generic", "form", "strength", "company", "url", "source", "ts"]
        return dict(zip(keys, row))


def _upsert_brand(row: Dict[str, Any]):
    with _conn() as c:
        c.execute(
            """INSERT INTO brands(brand,generic,form,strength,company,url,source,ts)
               VALUES(?,?,?,?,?,?,?,?)
               ON CONFLICT(brand) DO UPDATE SET
                 generic=excluded.generic, form=excluded.form, strength=excluded.strength,
                 company=excluded.company, url=excluded.url, source=excluded.source, ts=excluded.ts""",
            (
                row["brand"],
                row.get("generic", ""),
                row.get("form", ""),
                row.get("strength", ""),
                row.get("company", ""),
                row.get("url", ""),
                row.get("source", ""),
                int(time.time()),
            ),
        )
        c.commit()


def _upsert_price(brand: str, price_text: str, url: str, source: str, pack: str = "default"):
    with _conn() as co:
        co.execute(
            """INSERT INTO prices(brand,pack,price_text,url,source,ts)
               VALUES(?,?,?,?,?,?)
               ON CONFLICT(brand,pack) DO UPDATE SET 
                 price_text=excluded.price_text, url=excluded.url, source=excluded.source, ts=excluded.ts""",
            (brand, pack, price_text, url, source, int(time.time())),
        )
        co.commit()


def _get_price_local(brand: str, pack: str = "default") -> Optional[Dict[str, Any]]:
    with _conn() as c:
        row = c.execute(
            "SELECT price_text,url,source,ts FROM prices WHERE brand=? AND pack=?",
            (brand, pack),
        ).fetchone()
        if not row:
            return None
        return {"price_text": row[0], "url": row[1], "source": row[2], "ts": row[3]}


# ------------------------------ Resolver ---------------------------------- #
def _select_best_hit(term: str, hits: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not hits:
        return None
    choices = [h["title"] for h in hits]
    best = process.extractOne(term, choices, scorer=fuzz.WRatio)
    if not best:
        return hits[0]
    # Match title back to hit
    for h in hits:
        if h["title"] == best[0]:
            return h
    return hits[0]


def resolve_bd_medicine(term: str) -> Optional[Dict[str, Any]]:
    """
    Resolve a Bangladesh medicine brand or generic, with local cache + online refresh.
    Returns: {brand, generic, form, strength, company, url, source}
    """
    _ensure_schema()

    # Try local brand cache first (Title case is our storage key)
    q_norm = _norm(term).title()
    local = _get_brand_local(q_norm) or _get_brand_local(q_norm.upper()) or _get_brand_local(q_norm.lower())
    if local and (time.time() - local["ts"] < BRAND_TTL_SECS):
        return local

    # Expand the query with aliases/generics for better recall
    candidates = _expand_terms(term)

    page_data: Optional[Dict[str, Any]] = None
    chosen_title: Optional[str] = None
    chosen_url: Optional[str] = None

    for q in candidates:
        hits = search_brand_or_generic(q, max_hits=6)
        if not hits:
            continue
        best = _select_best_hit(term, hits)
        if not best:
            continue
        chosen_title, chosen_url = best["title"], best["url"]
        page = parse_brand_page(chosen_url)
        if page.get("status") == "success" and page.get("brand"):
            page_data = page
            break

    if not page_data:
        # Return any (possibly stale) local record rather than nothing
        return local

    brand_name = _norm(page_data["brand"]).title() or (chosen_title or q_norm)
    brandrec = {
        "brand": brand_name,
        "generic": page_data.get("generic", ""),
        "form": page_data.get("form", ""),
        "strength": page_data.get("strength", ""),
        "company": page_data.get("company", ""),
        "url": page_data.get("url", chosen_url or ""),
        "source": page_data.get("source", "medex.com.bd"),
    }
    _upsert_brand(brandrec)

    # Persist price snapshot if present
    if page_data.get("price_text"):
        _upsert_price(brandrec["brand"], page_data["price_text"], brandrec["url"], brandrec["source"])

    return brandrec


# ------------------------------ Price API --------------------------------- #
def _price_fallback_drug_international(name: str) -> Optional[Tuple[str, str]]:
    """
    Tiny fallback scraper for https://drug-international.com.bd
    Returns (url, price_text) or None.
    """
    import requests
    from bs4 import BeautifulSoup

    try:
        r = requests.get(
            "https://drug-international.com.bd/search",
            params={"q": name},
            headers={"User-Agent": "arobot/1.0 (+https://example.local)"},
            timeout=12,
        )
        if r.status_code != 200:
            return None
        s = BeautifulSoup(r.text, "html.parser")
        a = s.select_one("a[href*='/product/']")
        if not a:
            return None
        url = "https://drug-international.com.bd" + (a.get("href") or "")
        p = requests.get(url, headers={"User-Agent": "arobot/1.0 (+https://example.local)"}, timeout=12)
        if p.status_code != 200:
            return None
        sp = BeautifulSoup(p.text, "html.parser")
        # Look for text with "Tk" or "৳"
        cand = sp.find(string=lambda t: isinstance(t, str) and ("Tk" in t or "৳" in t))
        price_text = str(cand).strip() if cand else None
        if not price_text:
            tag = sp.select_one(".price, .product-price, .current-price")
            price_text = tag.get_text(" ", strip=True) if tag else None
        if not price_text:
            return None
        return (url, price_text)
    except Exception:
        return None


def get_price_bd(brand: str) -> Optional[Dict[str, Any]]:
    """
    Return latest BD retail price; refresh if older than TTL or missing.
    Combines Medex (primary) with Drug International (fallback).
    """
    _ensure_schema()
    bname = _norm(brand).title()

    # Try cached price
    p = _get_price_local(bname)
    if p and (time.time() - p["ts"] < PRICE_TTL_SECS):
        return {"brand": bname, **p, "status": "success", "cached": True}

    # Ensure we have a brand record (and fresh URL)
    br = resolve_bd_medicine(brand)
    if not br:
        return None

    # Try Medex page for price
    page = parse_brand_page(br["url"])
    price_text = page.get("price_text")
    if price_text:
        _upsert_price(br["brand"], price_text, br["url"], br.get("source", "medex.com.bd"))
        return {
            "brand": br["brand"],
            "price_text": price_text,
            "url": br["url"],
            "source": br.get("source", "medex.com.bd"),
            "status": "success",
            "cached": False,
        }

    # Fallback: Drug International
    di = _price_fallback_drug_international(br["brand"])
    if di:
        url, price_text = di
        _upsert_price(br["brand"], price_text, url, "drug-international.com.bd")
        return {
            "brand": br["brand"],
            "price_text": price_text,
            "url": url,
            "source": "drug-international.com.bd",
            "status": "success",
            "cached": False,
        }

    # No luck; if we had an older cached one, surface it
    p2 = _get_price_local(bname)
    if p2:
        return {"brand": bname, **p2, "status": "success", "cached": True}

    return None


# ----------------------------- Utilities ---------------------------------- #
def warm_cache(brands: List[str]) -> int:
    """
    Convenience helper to pre-populate cache for a list of brand names.
    Returns count successfully resolved.
    """
    ok = 0
    for b in brands:
        try:
            r = resolve_bd_medicine(b)
            if r:
                # Pull price too
                get_price_bd(r["brand"])
                ok += 1
        except Exception:
            pass
    return ok
