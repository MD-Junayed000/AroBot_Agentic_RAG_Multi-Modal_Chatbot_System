# utils/medicine_intent.py
from __future__ import annotations
import re
from typing import Dict, Optional

# ------------------------------- Lexicons -------------------------------- #

_FORMS = [
    "tablet","tab","tabs","cap","caps","capsule","capsules","suspension","syrup","elixir",
    "drops","drop","injection","inj","cream","ointment","gel","lotion","suppository","powder",
    "granules","spray","solution","dispersion","orodispersible","odt","effervescent","e/c","enteric",
    # Bangla / transliterations
    "ট্যাবলেট","ক্যাপসুল","সিরাপ","ইনজেকশন","ক্রিম","অয়েন্টমেন্ট","জেল","লোশন",
]

_BD_HINTS = [
    "bangladesh"," bd","dgda","medex","beximco","square","incepta","renata","acme","eskayef",
    "বাংলাদেশ","ডিজিডিএ"
]

# Bangla → ASCII digits
_BN_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")

def _normalize(text: str) -> str:
    """Trim, normalize digits, units, dashes and collapse whitespace."""
    tx = (text or "").strip()
    tx = tx.translate(_BN_DIGITS)
    tx = tx.replace("µg", "mcg").replace("μg", "mcg").replace("％", "%")
    tx = tx.replace("–", "-").replace("—", "-")
    tx = re.sub(r"\s{2,}", " ", tx)
    return tx

# ------------------------------- Patterns -------------------------------- #

_PRICE_PAT = re.compile(r"(?:^|\b)(price|cost|mrp|retail|how much|tk|bdt|taka|৳|টাকা|মূল্য|দাম|কত(?:\s*টাকা)?)\b", re.I)

_BRAND_PACK_PAT = re.compile(
    r"(brand(?:s)?|bd brand(?:s)?|bangladesh brand(?:s)?|manufacturer|company|made by|by\s+[A-Za-z]+"
    r"|pack(?:\s*size|s)?|blister|strip|bottle|sachet|vial|ampoule|tabs?|caps?|সাইজ|স্ট্রিপ|বোতল|বোতলে)",
    re.I,
)

# True “brandish” phrase — conservative
_BRANDISH = re.compile(r"^[A-Za-z][A-Za-z0-9+.\-/]*(?:\s+\d+[A-Za-z%/–-]*)?$")

# Sub-intents
_DOSE_PAT   = re.compile(r"\b(dose|dosing|dosage|posology|how (?:to )?take|how many|mg|ml|mL|ডোজ|মিগ্রা|মিলিগ্রাম)\b", re.I)
_INDIC_PAT  = re.compile(r"\b(use|indication|for what|why use|কিসের জন্য|ব্যবহার)\b", re.I)
_SIDE_PAT   = re.compile(r"\b(side ?effects?|adverse|a/e|পার্শ্বপ্রতিক্রিয়া|প্রতিকূল)\b", re.I)
_CONTRA_PAT = re.compile(r"\b(contraindication|avoid|not (?:for|in)|বারণ|নিষেধ)\b", re.I)
_PREG_PAT   = re.compile(r"\b(pregnan(?:t|cy)|breast ?feed(?:ing)?|গর্ভাবস্থা|স্তন্যপান)\b", re.I)
_INTRA_PAT  = re.compile(r"\b(interaction|drug[- ]drug|মিথস্ক্রিয়া)\b", re.I)

# Clinical case/HPI detector (keeps it broad but specific to HPI tokens)
_CASE_PAT = re.compile(
    r"\b(\d{1,3})(?:\s*(?:y|yr|yrs|yo|year|years))?\s*(?:[/-]?\s*)?(m|f)\b|"
    r"\bfever|cough|sore throat|dyspnea|shortness of breath|chest pain|wheeze|asthma|"
    r"vomit|diarrhea|abdominal pain|rash|pregnan(?:t|cy)\b",
    re.I,
)

# Strength & pack
_STRENGTH_PAT = re.compile(
    r"(?P<num1>\d+(?:\.\d+)?)\s*(?P<u1>mg|mcg|g|iu|ml|mL|%)"
    r"(?:\s*/\s*(?P<num2>\d+(?:\.\d+)?)\s*(?P<u2>mg|mcg|g|iu|ml|mL))?", re.I,
)
_PACK_HINT_PAT = re.compile(
    r"(?:(?P<count>\d{1,3})(?:\s*[x×]\s*(?P<count2>\d{1,3}))\s*)?"
    r"(?:(?P<num>\d{1,4})\s*(?P<u>ml|mL|tabs?|tablets?|caps?|capsules?|sachets?|vials?|ampoules?|strips?))",
    re.I,
)

# Company detector (Bangladesh pharma manufacturers)
_COMPANY_PAT = re.compile(
    r"\b(beximco|square|incepta|renata|aci|eskayef|skf|acme|aristopharma|drug international|orion pharma|aci limited)\b|"
    r"\b(pharma(?:ceuticals?)?|pharmaceuticals?)\b",
    re.I,
)

# Meta/about queries (route to about/capabilities)
_META_PAT = re.compile(
    r"(who are you|what are you|about you|about yourself|what can you do|capabilities|skills|ability|abilities|your experience|your experiences|experience|experiences|who made you|who created you|are you a doctor|expertise|what is your expertise)",
    re.I,
)

# Policy/regulation (Bangladesh or general drug laws/regulators)
_POLICY_PAT = re.compile(
    r"\b(policy|act|regulation|rules|law|dgda|directorate\s+general\s+of\s+drug\s+administration)\b",
    re.I,
)

# ------------------------------- Helpers --------------------------------- #

def _pick_form(text: str) -> Optional[str]:
    low = text.lower()
    for f in _FORMS:
        if f.lower() in low:
            if f.lower() in ("tab","tabs"): return "tablet"
            if f.lower() in ("cap","caps"): return "capsule"
            return f.lower()
    return None

def _pick_strength(text: str) -> Optional[str]:
    m = _STRENGTH_PAT.search(text)
    if not m: return None
    n1,u1,n2,u2 = m.group("num1"), m.group("u1"), m.group("num2"), m.group("u2")
    return f"{n1} {u1}/{n2} {u2}" if n2 and u2 else f"{n1} {u1}"

def _pick_pack_hint(text: str) -> Optional[str]:
    m = _PACK_HINT_PAT.search(text)
    if not m: return None
    c1,c2,num,u = m.group("count"), m.group("count2"), m.group("num"), (m.group("u") or "").lower()
    if c1 and c2: return f"{c1}x{c2} {u}"
    if num and u: return f"{num} {u}"
    return None

def _is_bd_context(text: str) -> bool:
    low = " " + text.lower() + " "
    return any(tok in low for tok in _BD_HINTS)

def extract_candidate_brand(text: str) -> str:
    tx = _normalize(text)
    tx = re.sub(r"[?!.।！？]+$", "", tx).strip()
    # remove boilerplate leaders
    tx = re.sub(r"^(tell me about|what is|what's|info on|details of|give|show|price of|price for|cost of|how much is)\s+",
                "", tx, flags=re.I)
    # remove trailing qualifiers (keeps candidate clean)
    tx = re.sub(r"\b(in\s+bd|in\s+bangladesh|price|mrp|cost|pack(?:\s*size|s)?)\b\.?$", "",
                tx, flags=re.I).strip()
    tx = re.sub(r"\s{2,}", " ", tx)
    return tx

# ------------------------------- Public API ------------------------------ #

def detect_intent(text: str) -> Dict[str, bool | str | None]:
    tx = _normalize(text)
    low = tx.lower()
    word_count = len(tx.split())

    # conservative "brandish"
    looks_brandish = bool(_BRANDISH.match(tx)) and word_count <= 3 and not re.search(
        r"\b(what|which|how|why|when|where|who|please|advise|explain|tell)\b", low
    )

    intents: Dict[str, bool | str | None] = {
        "is_price": bool(_PRICE_PAT.search(low)),
        "wants_brand_pack": bool(_BRAND_PACK_PAT.search(low)) or _is_bd_context(low),
        "looks_brandish": looks_brandish,
        "is_company": bool(_COMPANY_PAT.search(low)),
        "is_meta": bool(_META_PAT.search(low)),
        "is_policy": bool(_POLICY_PAT.search(low)),
        "wants_dose": bool(_DOSE_PAT.search(low)),
        "wants_indication": bool(_INDIC_PAT.search(low)),
        "wants_side_effects": bool(_SIDE_PAT.search(low)),
        "wants_contra": bool(_CONTRA_PAT.search(low)),
        "wants_pregnancy": bool(_PREG_PAT.search(low)),
        "wants_interactions": bool(_INTRA_PAT.search(low)),
        "bd_context": _is_bd_context(low),
        "is_clinical_case": bool(_CASE_PAT.search(low)),
        "candidate": None,
        "strength": None,
        "form": None,
        "pack_hint": None,
    }

    intents["candidate"] = extract_candidate_brand(tx) or None
    intents["strength"]  = _pick_strength(tx)
    intents["form"]      = _pick_form(tx)
    intents["pack_hint"] = _pick_pack_hint(tx)
    return intents


# ------------------------------- Quick tests ------------------------------ #
if __name__ == "__main__":
    samples = [
        "price of paracetamol 500 tablet",
        "paracetamol 500mg",
        "Give Bangladesh brand examples and typical retail pack sizes for paracetamol",
        "omeprazole 20 mg price in BD",
        "Omeprazole 20mg capsule pack size?",
        "Paracetamol syrup 120 mg/5 mL bottle 100 mL",
        "What are side effects of azithromycin?",
        "Contraindications of Ibuprofen in pregnancy",
        "ibuprofen 200mg",
        "paracetamol price in Bangladesh",
        "How to take ibuprofen for fever?",
        "Paracetamol 500 mg dose?",
        "28F with fever 38.6°C, sore throat, dry cough. No dyspnea.",
    ]
    for s in samples:
        print(s, "=>", detect_intent(s))
