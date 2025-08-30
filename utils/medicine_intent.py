# utils/medicine_intent.py
from __future__ import annotations
import re
from typing import Dict

_PRICE_PAT = re.compile(
    r"(price|cost|mrp|retail|how much|কত|মুল্য|দাম)\b", re.I
)
# one–two token brand-ish queries (e.g., "Napa", "Losectil 20")
_BRANDISH = re.compile(r"^[A-Za-z][A-Za-z0-9+.\-]*(?:\s+\d+[A-Za-z%/]*)?$")

def detect_intent(text: str) -> Dict[str, bool]:
    tx = (text or "").strip()
    low = tx.lower()
    return {
        "is_price": bool(_PRICE_PAT.search(low)),
        "looks_brandish": bool(_BRANDISH.match(tx)) or len(tx.split()) <= 3,
    }

def extract_candidate_brand(text: str) -> str:
    """
    Very small heuristic:
    - take the raw text (e.g., 'Napa', 'Losectil 20mg')
    - strip trailing question punctuation
    """
    tx = (text or "").strip()
    tx = re.sub(r"[?!.]+$", "", tx).strip()
    # remove common prefixes
    tx = re.sub(r"^(tell me about|what is|info on|details of)\s+", "", tx, flags=re.I)
    return tx
