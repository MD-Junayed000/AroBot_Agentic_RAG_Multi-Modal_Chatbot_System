# core/prescriptions/parser.py
"""Prescription parsing and structured data extraction"""

import json
import logging
import re
from typing import Dict, Any, Optional, List
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)

class PrescriptionParser:
    """Handles prescription parsing and structured data extraction"""
    
    @staticmethod
    def extract_json(text: str) -> Optional[Dict[str, Any]]:
        """Extract JSON from text with error handling"""
        if not text:
            return None
            
        # Try to find JSON block
        patterns = [
            r"```json\s*(\{.*?\})\s*```",  # Fenced JSON
            r"(\{[\s\S]*\})"              # Plain JSON
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.S | re.I)
            if match:
                try:
                    return json.loads(match.group(1))
                except Exception:
                    # Try to fix common JSON issues
                    try:
                        fixed = re.sub(r",\s*}", "}", match.group(1))
                        fixed = re.sub(r",\s*]", "]", fixed)
                        return json.loads(fixed)
                    except Exception:
                        continue
        return None
    
    @staticmethod
    def validate_structured_data(structured: Dict[str, Any], ocr_text: str) -> Dict[str, Any]:
        """Validate structured data against OCR evidence"""
        if not structured or not ocr_text:
            return structured
        
        lines = [l.strip() for l in ocr_text.splitlines() if l.strip()]
        
        # Validate text fields
        for field in ["patient_name", "diagnosis_or_cc", "date"]:
            value = structured.get(field)
            if value and PrescriptionParser._best_line_score(lines, str(value)) < 0.55:
                structured[field] = ""
        
        # Validate medications
        medications = structured.get("medications", [])
        valid_medications = []
        
        for med in medications:
            if not isinstance(med, dict):
                continue
                
            name = med.get("name", "")
            confidence = 0.0
            
            if name:
                confidence = PrescriptionParser._best_line_score(lines, name)
                
            # Also check dose string
            if confidence < 0.72:
                dose_str = " ".join([
                    str(med.get("dose", "")), 
                    str(med.get("dose_pattern", ""))
                ]).strip()
                if dose_str:
                    confidence = max(confidence, PrescriptionParser._best_line_score(lines, dose_str))
            
            if confidence >= 0.5:  # Lower threshold for inclusion
                med["confidence"] = max(float(med.get("confidence", 0.0)), round(confidence, 2))
                valid_medications.append(med)
        
        structured["medications"] = valid_medications
        return structured
    
    @staticmethod
    def _best_line_score(lines: List[str], phrase: str) -> float:
        """Calculate best similarity score between phrase and OCR lines"""
        phrase_norm = PrescriptionParser._normalize_text(phrase)
        if not phrase_norm:
            return 0.0
        
        best_score = 0.0
        phrase_tokens = [t for t in phrase_norm.split() if len(t) >= 3]
        
        for line in lines:
            line_norm = PrescriptionParser._normalize_text(line)
            if not line_norm:
                continue
            
            # Token-based scoring
            token_score = 0
            if phrase_tokens:
                token_hits = sum(1 for token in phrase_tokens if token in line_norm)
                token_score = token_hits / len(phrase_tokens)
            
            # Sequence similarity
            similarity = SequenceMatcher(None, line_norm, phrase_norm).ratio()
            
            # Take the better score
            score = max(token_score, similarity)
            best_score = max(best_score, score)
        
        return best_score
    
    @staticmethod
    def _normalize_text(text: str) -> str:
        """Normalize text for comparison"""
        try:
            text = (text or "").lower()
            text = re.sub(r"[^a-z0-9\s.+/-]", " ", text)
            return " ".join(text.split())
        except Exception:
            return text or ""
    
    @staticmethod
    def summarize_prescription(structured: Dict[str, Any]) -> str:
        """Create human-readable summary from structured data"""
        if not structured:
            return "Could not parse the prescription reliably."
        
        parts = []
        
        # Patient information
        if structured.get("patient_name") or structured.get("age_sex"):
            patient_info = " ".join([
                s for s in [structured.get("patient_name"), structured.get("age_sex")] 
                if s
            ]).strip()
            if patient_info:
                parts.append(f"Patient: {patient_info}")
        
        # Doctor and clinic
        if structured.get("doctor"):
            parts.append(f"Doctor: {structured['doctor']}")
        if structured.get("clinic"):
            parts.append(f"Clinic: {structured['clinic']}")
        if structured.get("date"):
            parts.append(f"Date: {structured['date']}")
        
        # Diagnosis
        if structured.get("diagnosis_or_cc"):
            parts.append(f"Diagnosis: {structured['diagnosis_or_cc']}")
        
        # Medications
        medications = structured.get("medications", [])
        if medications:
            med_lines = []
            for i, med in enumerate(medications, 1):
                name = med.get("name", "").strip()
                
                # Filter out obvious OCR noise
                if len(re.sub(r"[^A-Za-z]", "", name)) < 3:
                    continue
                
                dose = med.get("dose") or med.get("dose_pattern") or ""
                instructions = med.get("additional_instructions") or ""
                
                med_line = f"{i}) {name}"
                if dose:
                    med_line += f" - {dose}"
                if instructions:
                    med_line += f" ({instructions})"
                
                med_lines.append(med_line)
            
            if med_lines:
                parts.append("Medications:\n" + "\n".join(med_lines))
        
        # Notes
        if structured.get("notes"):
            parts.append(f"Notes: {structured['notes']}")
        
        return "\n".join(parts) if parts else "Prescription parsed successfully."
