# pharma/provider/medex.py
from typing import List, Dict, Any, Optional
import requests
from bs4 import BeautifulSoup

UA = {"User-Agent": "arobot/1.0 (+https://example.local)"}
BASE = "https://medex.com.bd"


def _fetch(url: str, **kw) -> Optional[BeautifulSoup]:
    """GET a page and return a BeautifulSoup parser (or None on non-200)."""
    kw.setdefault("timeout", 12)
    kw.setdefault("headers", UA)
    try:
       r = requests.get(url, **kw)
       if r.status_code != 200:
            return None
       return BeautifulSoup(r.text, "html.parser")
    except Exception:
        return None


def search_brand_or_generic(term: str, max_hits: int = 6) -> List[Dict[str, Any]]:
    """
    Search Medex for a brand or generic term.
    Returns a list of candidates: [{title, url, meta}]
    """
    s = _fetch(f"{BASE}/search", params={"term": term})
    if not s:
        return []

    out: List[Dict[str, Any]] = []

    # Typical result cards: ".search-item a"
    for a in s.select(".search-item a")[:max_hits]:
        href = a.get("href") or ""
        if not href.startswith("/"):
            continue
        url = BASE + href
        title = a.get_text(strip=True)
        meta = a.find_next("div", class_="text-muted")
        meta_txt = meta.get_text(" ", strip=True) if meta else ""
        out.append({"title": title, "url": url, "meta": meta_txt})

    # Defensive: also look for table style results if cards are absent
    if not out:
        for tr in s.select("table tr"):
            a = tr.select_one("a[href^='/']")
            if not a:
                continue
            title = a.get_text(strip=True)
            url = BASE + (a.get("href") or "")
            meta_txt = tr.get_text(" ", strip=True)
            out.append({"title": title, "url": url, "meta": meta_txt})
            if len(out) >= max_hits:
                break

    return out


def _extract_price_text(soup: BeautifulSoup) -> Optional[str]:
    """
    Medex uses a few different layouts for price. Try several strategies.
    Returns a short human-readable price string or None.
    """
    # 1) Obvious labeled nodes
    lab = soup.find(string=lambda t: isinstance(t, str) and ("Price" in t or "Tk" in t or "৳" in t))
    if lab:
        txt = str(lab).strip()
        # Often the numeric part is in a sibling
        sib = getattr(getattr(lab, "parent", None), "find_next", lambda *_: None)("span")
        if sib and (sib_text := sib.get_text(" ", strip=True)):
            return f"{txt} {sib_text}".strip()
        return txt

    # 2) Common classes
    for sel in [".brand-price", ".drug-price", ".price-tk", ".price", ".unit-price"]:
        el = soup.select_one(sel)
        if el:
            t = el.get_text(" ", strip=True)
            if t:
                return t

    # 3) Anything containing currency symbol
    for el in soup.find_all(["div", "span", "p"]):
        t = el.get_text(" ", strip=True)
        if ("৳" in t or " Tk" in t) and len(t) <= 120:
            return t

    return None


def parse_brand_page(url: str) -> Dict[str, Any]:
    """
    Parse a brand page to extract brand metadata and price text.
    Works for e.g. https://medex.com.bd/brands/<id> and similar.
    """
    s = _fetch(url)
    if not s:
        return {"status": "error", "url": url}

    # Brand name (h1)
    h1 = s.select_one("h1")
    brand = h1.get_text(strip=True) if h1 else ""

    # Generic (first link to /generic/)
    gl = s.find("a", href=lambda x: x and "/generic/" in x)
    generic = gl.get_text(strip=True) if gl else ""

    # Company
    comp = s.select_one("a[href*='/companies/']")
    company = comp.get_text(strip=True) if comp else ""

    # Dosage form / strength (header contains something like "Tablet 500 mg")
    form = ""
    strength = ""
    info = s.select_one(".drug-page-header .small") or s.select_one(".small.text-muted")
    if info:
        txt = info.get_text(" ", strip=True)
        parts = txt.split()
        if parts:
            form = parts[0]
            strength = " ".join(parts[1:]) if len(parts) > 1 else ""

    price_txt = _extract_price_text(s)

    return {
        "status": "success",
        "brand": brand,
        "generic": generic,
        "company": company,
        "form": form,
        "strength": strength,
        "price_text": price_txt,
        "url": url,
        "source": "medex.com.bd",
    }
