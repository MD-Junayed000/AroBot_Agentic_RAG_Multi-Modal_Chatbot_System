# utils/clinical_facts.py
from __future__ import annotations
from typing import Dict, List

OTC_DOSING: Dict[str, dict] = {
    "paracetamol": {
        "synonyms": ["paracetamol", "acetaminophen", "pcm"],
        "adult_dose": "500–1000 mg every 4–6 hours PRN; MAX 4,000 mg/day (≤3,000 mg/day in chronic use or liver disease).",
        "key_cautions": [
            "Avoid exceeding daily max (risk of hepatotoxicity).",
            "Lower max (e.g., 3 g/day) if chronic use, low body weight, or hepatic disease.",
            "Watch combination products (cold/flu) to avoid double-dosing.",
        ],
        "pregnancy": "Generally considered safe at recommended doses.",
        "pack_sizes_bd": "Common: 10-tablet blisters; syrups 60–100 mL (120 mg/5 mL).",
    },
    "ibuprofen": {
        "synonyms": ["ibuprofen", "ibu"],
        "adult_dose": "200–400 mg every 6–8 hours PRN; OTC MAX 1,200 mg/day (higher doses only under medical supervision).",
        "key_cautions": [
            "Avoid in active peptic ulcer/ GI bleed; use with caution in gastritis/GERD.",
            "Caution/avoid in chronic kidney disease, volume depletion, or with ACEi/ARB+diuretic (AKI risk).",
            "Avoid in 3rd-trimester pregnancy; use caution earlier in pregnancy (seek clinician advice).",
            "NSAID-exacerbated respiratory disease/asthma sensitivity possible.",
        ],
        "pregnancy": "Avoid in 3rd trimester; seek clinician advice earlier in pregnancy.",
        "pack_sizes_bd": "Common: 10-tablet blisters (200 mg); suspensions ~100 mg/5 mL in 60–100 mL bottles.",
    },
}

def find_otc_targets_in_text(text: str) -> List[str]:
    q = (text or "").lower()
    hits: List[str] = []
    for key, entry in OTC_DOSING.items():
        if any(s in q for s in entry["synonyms"] + [key]):
            hits.append(key)
    return sorted(set(hits))
