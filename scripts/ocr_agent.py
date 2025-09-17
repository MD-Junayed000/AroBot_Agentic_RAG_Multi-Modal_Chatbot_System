"""
OCR Agent for processing prescription images
"""
from typing import Dict, Any, Union
from PIL import Image
from core.multimodal_processor import MultiModalProcessor
from core.llm_modular import ModularLLMHandler as LLMHandler
from langsmith import traceable
import logging

logger = logging.getLogger(__name__)

class OCRAgent:
    """Agent specialized in OCR processing and prescription analysis"""
    
    def __init__(self):
        self.multimodal_processor = MultiModalProcessor()
        self.llm = LLMHandler()
    
    @traceable(name="process_prescription_image")
    def process_prescription_image(self, image_input: Union[str, bytes, Image.Image]) -> Dict[str, Any]:
        """Process prescription image through complete OCR pipeline"""
        try:
            # Use multimodal processor for complete processing
            results = self.multimodal_processor.process_prescription_image(image_input)
            
            if results.get('status') == 'error':
                return results
            
            # Extract key information
            ocr_results = results.get('ocr_results', {})
            entities = results.get('entities', {})
            
            # Create summary
            summary = self._create_prescription_summary(ocr_results, entities)
            
            return {
                "ocr_results": ocr_results,
                "entities": entities,
                "summary": summary,
                "embeddings": results.get('embeddings', {}),
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Error processing prescription image: {e}")
            return {
                "error": str(e),
                "status": "error"
            }
    
    @traceable(name="analyze_prescription_with_llm")
    def analyze_prescription_with_llm(self, image_input: Union[str, bytes, Image.Image], 
                                    ocr_text: str = None) -> Dict[str, Any]:
        """Analyze prescription using both OCR and vision LLM"""
        try:
            # Get OCR results if not provided
            if not ocr_text:
                ocr_results = self.multimodal_processor.process_image(image_input)
                if ocr_results.get('status') == 'error':
                    return ocr_results
                ocr_text = ocr_results.get('raw_text', '')
            
            # Use LLM for analysis
            if isinstance(image_input, str):
                llm_analysis = self.llm.analyze_prescription(image_path=image_input, ocr_text=ocr_text)
            else:
                llm_analysis = self.llm.analyze_prescription(image_data=image_input, ocr_text=ocr_text)
            
            return {
                "ocr_text": ocr_text,
                "llm_analysis": llm_analysis,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Error in LLM prescription analysis: {e}")
            return {
                "error": str(e),
                "status": "error"
            }
    
    @traceable(name="extract_medication_list")
    def extract_medication_list(self, image_input: Union[str, bytes, Image.Image]) -> Dict[str, Any]:
        """Extract structured medication list from prescription"""
        try:
            # Process image through OCR
            results = self.process_prescription_image(image_input)
            
            if results.get('status') == 'error':
                return results
            
            # Extract medications from entities
            entities = results.get('entities', {})
            medications = entities.get('medications', [])
            
            # Clean and structure medication data
            structured_medications = []
            for med in medications:
                if med.get('text', '').strip():
                    structured_med = {
                        'medication_text': med.get('text', '').strip(),
                        'strength': med.get('strength'),
                        'unit': med.get('unit'),
                        'frequency': med.get('frequency'),
                        'duration': med.get('duration'),
                        'confidence': 'high' if len(med.get('text', '')) > 5 else 'medium'
                    }
                    structured_medications.append(structured_med)
            
            return {
                "medications": structured_medications,
                "medication_count": len(structured_medications),
                "patient_info": entities.get('patient_info', {}),
                "doctor_info": entities.get('doctor_info', {}),
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Error extracting medication list: {e}")
            return {
                "error": str(e),
                "status": "error"
            }
    
    def _create_prescription_summary(self, ocr_results: Dict, entities: Dict) -> Dict[str, Any]:
        """Create a summary of prescription analysis"""
        try:
            summary = {
                "text_lines_detected": len(ocr_results.get('lines', [])),
                "structured_items_found": len(ocr_results.get('structured_items', [])),
                "medications_identified": len(entities.get('medications', [])),
                "has_patient_info": bool(entities.get('patient_info', {})),
                "has_doctor_info": bool(entities.get('doctor_info', {})),
                "text_quality": "good" if len(ocr_results.get('raw_text', '')) > 50 else "poor"
            }
            
            # Add key findings
            medications = entities.get('medications', [])
            if medications:
                summary["key_medications"] = [
                    med.get('text', '')[:30] + "..." if len(med.get('text', '')) > 30 
                    else med.get('text', '') 
                    for med in medications[:3]
                ]
            
            return summary
            
        except Exception as e:
            logger.error(f"Error creating prescription summary: {e}")
            return {"error": str(e)}
    
    @traceable(name="validate_prescription_quality")
    def validate_prescription_quality(self, image_input: Union[str, bytes, Image.Image]) -> Dict[str, Any]:
        """Validate the quality and readability of prescription image"""
        try:
            # Process image
            ocr_results = self.multimodal_processor.process_image(image_input)
            
            if ocr_results.get('status') == 'error':
                return ocr_results
            
            # Quality metrics
            raw_text = ocr_results.get('raw_text', '')
            lines = ocr_results.get('lines', [])
            items = ocr_results.get('structured_items', [])
            
            quality_score = 0
            issues = []
            recommendations = []
            
            # Text length check
            if len(raw_text) < 20:
                issues.append("Very little text detected")
                recommendations.append("Ensure image is clear and well-lit")
            else:
                quality_score += 25
            
            # Lines count check
            if len(lines) < 3:
                issues.append("Few text lines detected")
                recommendations.append("Check if entire prescription is visible")
            else:
                quality_score += 25
            
            # Structured items check
            if len(items) < 2:
                issues.append("Limited structured information extracted")
                recommendations.append("Improve image focus and resolution")
            else:
                quality_score += 25
            
            # Text coherence check (simple)
            medical_keywords = ['mg', 'ml', 'tablet', 'cap', 'dr.', 'patient', 'dose']
            keyword_count = sum(1 for keyword in medical_keywords if keyword in raw_text.lower())
            
            if keyword_count >= 2:
                quality_score += 25
            else:
                issues.append("Limited medical terminology detected")
                recommendations.append("Ensure prescription contains clear medical information")
            
            # Determine overall quality
            if quality_score >= 75:
                overall_quality = "excellent"
            elif quality_score >= 50:
                overall_quality = "good"
            elif quality_score >= 25:
                overall_quality = "fair"
            else:
                overall_quality = "poor"
            
            return {
                "quality_score": quality_score,
                "overall_quality": overall_quality,
                "issues": issues,
                "recommendations": recommendations,
                "text_detected": len(raw_text),
                "lines_detected": len(lines),
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Error validating prescription quality: {e}")
            return {
                "error": str(e),
                "status": "error"
            }
