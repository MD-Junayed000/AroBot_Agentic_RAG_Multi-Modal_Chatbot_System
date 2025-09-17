# utils/clinical_facts_simple.py
"""Simplified clinical facts - dynamic lookup instead of hardcoded data"""

from typing import List, Optional
import re

def find_common_medicines_in_text(text: str) -> List[str]:
    """Find common medicine names in text"""
    text_lower = text.lower()
    
    # Common medicine patterns (without hardcoded brand names)
    medicine_patterns = [
        r'\b(?:acetaminophen|paracetamol)\b',
        r'\bibuprofen\b',
        r'\baspirin\b',
        r'\bcetirizine\b',
        r'\bloratadine\b',
        r'\bomeprazole\b',
        r'\bamoxicillin\b'
    ]
    
    found = []
    for pattern in medicine_patterns:
        if re.search(pattern, text_lower):
            # Extract the medicine name from pattern
            match = re.search(pattern, text_lower)
            if match:
                found.append(match.group().replace('\\b', ''))
    
    return list(set(found))  # Remove duplicates

def get_safety_reminder() -> str:
    """Get generic safety reminder for medicine queries"""
    return (
        "**Safety:** Use medicines as directed. Consult healthcare providers for "
        "persistent symptoms, drug interactions, or if you have medical conditions. "
        "Always read medicine labels and follow dosing instructions."
    )
