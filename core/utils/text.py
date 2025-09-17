# core/utils/text.py
"""Text processing utilities"""

import re
from typing import List

def clean_conversation_context(text: str, limit: int = 800) -> str:
    """Clean and limit conversation context"""
    if not text:
        return ""
    
    # Remove system prompts and instructions
    text = re.sub(r"(?i)^\s*you are .*?$", "", text.strip())
    text = re.sub(r"(?i)\b(system|instruction|prompt)\b.*", "", text)
    
    return text[:limit]

def deduplicate_paragraphs(text: str) -> str:
    """Remove duplicate paragraphs from text"""
    if not text:
        return text
    
    seen = set()
    output = []
    
    for block in [b.strip() for b in text.split("\n\n")]:
        if block and block not in seen:
            seen.add(block)
            output.append(block)
    
    return "\n\n".join(output)

def short_sources(sources: List[str]) -> str:
    """Format sources list concisely"""
    if not sources:
        return ""
    
    unique_sources = list(dict.fromkeys([s for s in sources if s]))
    return "Sources: " + ", ".join(unique_sources)

def normalize_text(text: str) -> str:
    """Normalize text for comparison"""
    try:
        text = (text or "").lower()
        text = re.sub(r"[^a-z0-9\s.+/-]", " ", text)
        return " ".join(text.split())
    except Exception:
        return text or ""

def extract_medical_terms(text: str) -> List[str]:
    """Extract medical terms from text"""
    text_lower = text.lower()
    
    # Common medical term patterns
    patterns = [
        r'\b\d+\s*(?:mg|g|ml|mcg|units?|tablets?|capsules?)\b',  # Dosages
        r'\b(?:tablet|capsule|syrup|injection|cream|ointment)\b',  # Forms
        r'\b(?:dose|dosage|side effect|contraindication)\b',      # Medical concepts
    ]
    
    terms = []
    for pattern in patterns:
        terms.extend(re.findall(pattern, text_lower))
    
    return list(set(terms))  # Remove duplicates

def truncate_text(text: str, max_length: int = 500, suffix: str = "...") -> str:
    """Truncate text with suffix if needed"""
    if not text or len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix

def is_medical_query(text: str) -> bool:
    """Check if text appears to be a medical query"""
    medical_indicators = [
        "medicine", "drug", "medication", "tablet", "capsule", "syrup",
        "dose", "dosage", "side effect", "contraindication", "treatment",
        "symptoms", "diagnosis", "prescription", "mg", "ml", "doctor"
    ]
    
    text_lower = text.lower()
    return any(indicator in text_lower for indicator in medical_indicators)

def extract_dosage_info(text: str) -> List[str]:
    """Extract dosage information from text"""
    dosage_patterns = [
        r'\b\d+(?:\.\d+)?\s*(?:mg|g|ml|mcg|units?)\b',
        r'\b\d+\s*(?:tablet|capsule|tab|cap)s?\b',
        r'\b(?:once|twice|three times?)\s*(?:daily|a day|per day)\b'
    ]
    
    dosages = []
    for pattern in dosage_patterns:
        dosages.extend(re.findall(pattern, text, re.IGNORECASE))
    
    return list(set(dosages))
