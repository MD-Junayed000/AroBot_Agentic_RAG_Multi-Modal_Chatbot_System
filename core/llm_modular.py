# core/llm_modular.py
"""Modular LLM Handler - Clean architecture using focused modules"""

import logging
from typing import List, Dict, Any, Optional

# Import our modular components
from .llm.text import TextGenerator
from .llm.vision import VisionProcessor
from .rag.context import RAGContextManager
from .prescriptions.parser import PrescriptionParser
from .medicines.bd import BangladeshMedicineInfo
from .prompts.system import get_system_prompt, format_medical_prompt
from .utils.text import (
    clean_conversation_context, 
    deduplicate_paragraphs, 
    is_medical_query,
    truncate_text
)
from utils.ocr_pipeline import OCRPipeline

logger = logging.getLogger(__name__)

class ModularLLMHandler:
    """Clean, modular LLM handler with focused responsibilities"""
    
    def __init__(self):
        # Initialize modular components
        self.text_gen = TextGenerator()
        self.vision = VisionProcessor()
        self.rag = RAGContextManager()
        self.prescription_parser = PrescriptionParser()
        self.bd_medicines = BangladeshMedicineInfo()
    
    # ==================== TEXT GENERATION ====================
    
    def generate_text_response(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate text response"""
        return self.text_gen.generate_response(prompt, system_prompt)
    
    def greeting_response(self, text: str = "") -> str:
        """Generate greeting response"""
        return self.text_gen.greeting_response(text)
    
    def about_response(self) -> str:
        """Generate about response"""
        return self.text_gen.about_response()
    
    # ==================== VISION PROCESSING ====================
    
    def generate_vision_response(self, prompt: str, image_path: Optional[str] = None, 
                                image_data: Optional[bytes] = None) -> str:
        """Analyze image with vision model"""
        if image_data:
            return self.vision.analyze_image(image_data, prompt)
        elif image_path:
            with open(image_path, "rb") as f:
                return self.vision.analyze_image(f.read(), prompt)
        else:
            return "No image provided."
    
    def ocr_only(self, image_data: bytes) -> Dict[str, Any]:
        """Extract text from image using OCR only"""
        try:
            ocr = OCRPipeline(lang="en")
            lines, items, header = ocr.run_on_bytes(image_data)
            raw_text = "\n".join(lines)
            
            return {
                "raw_text": raw_text,
                "lines": lines,
                "items": items,
                "header": header,
                "status": "success"
            }
        except Exception as e:
            logger.error(f"OCR error: {e}")
            return {
                "raw_text": "",
                "lines": [],
                "items": [],
                "header": "",
                "error": str(e),
                "status": "error"
            }
    
    def analyze_prescription(self, image_path: Optional[str] = None, 
                           image_data: Optional[bytes] = None) -> Dict[str, Any]:
        """Analyze prescription image"""
        try:
            # Get image data
            if image_data:
                target_image_data = image_data
            elif image_path:
                with open(image_path, "rb") as f:
                    target_image_data = f.read()
            else:
                return {"error": "No image provided", "status": "error"}
            
            # Extract OCR text
            ocr_result = self.ocr_only(target_image_data)
            ocr_text = ocr_result.get("raw_text", "")
            
            # Try vision model for structured extraction
            vision_prompt = get_system_prompt("prescription")
            vision_result = self.vision.analyze_image(target_image_data, vision_prompt)
            
            # Extract structured data
            structured = self.prescription_parser.extract_json(vision_result)
            
            if structured:
                # Validate against OCR
                structured = self.prescription_parser.validate_structured_data(structured, ocr_text)
                summary = self.prescription_parser.summarize_prescription(structured)
                
                return {
                    "mode": "vision_extraction",
                    "ocr_results": ocr_result,
                    "structured": structured,
                    "prescription_analysis": summary,
                    "summary": summary,
                    "status": "success"
                }
            else:
                # Fallback to simple vision description
                desc = self.vision.analyze_image(
                    target_image_data, 
                    "Describe this medical document. Focus on patient, doctor, and medicines."
                )
                return {
                    "mode": "vision_fallback",
                    "ocr_results": ocr_result,
                    "prescription_analysis": desc,
                    "summary": desc,
                    "status": "success"
                }
                
        except Exception as e:
            logger.error(f"Prescription analysis error: {e}")
            return {"error": str(e), "status": "error"}
    
    # ==================== MEDICAL Q&A ====================
    
    def answer_medical_query(self, query: str, context: Optional[List[str]] = None, 
                           conversation_context: str = "") -> str:
        """Answer medical questions with RAG context"""
        try:
            # Use provided context or gather from RAG
            if context and len(context) > 0:
                rag_context = context[:3]  # Limit context
            else:
                rag_context = self.rag.gather_context(query)
            
            # Format prompt with context
            if rag_context:
                context_text = "\n---\n".join(rag_context)
                prompt = format_medical_prompt(query, context_text)
            else:
                prompt = format_medical_prompt(query)
            
            # Generate response
            system = get_system_prompt("medical")
            response = self.text_gen.generate_response(prompt, system)
            
            # Add safety footer and clean up
            response = deduplicate_paragraphs(response)
            response = self.text_gen.safety_footer(response)
            
            return response
            
        except Exception as e:
            logger.error(f"Medical query error: {e}")
            return f"I'm unable to answer that medical question right now. Error: {str(e)}"

    def gather_rag_context(self, query: str, extra_ctx: Optional[List[str]] = None, limit: int = 3) -> List[str]:
        """Expose RAG context gathering for agents and legacy integrations."""
        collected: List[str] = []

        if extra_ctx:
            for chunk in extra_ctx:
                if isinstance(chunk, str) and chunk.strip():
                    collected.append(chunk.strip())

        try:
            hits = self.rag.gather_context(query)
            for chunk in hits:
                if isinstance(chunk, str) and chunk.strip():
                    collected.append(chunk.strip())
        except Exception as e:
            logger.warning(f"RAG context gathering failed: {e}")

        # Deduplicate while preserving order
        seen = set()
        unique: List[str] = []
        for chunk in collected:
            norm = chunk
            if norm not in seen:
                seen.add(norm)
                unique.append(norm)
            if len(unique) >= limit:
                break

        return unique

    # Backwards compatibility alias (older agents expect private method)
    _gather_rag_context = gather_rag_context
    
    # ==================== BANGLADESH MEDICINES ====================
    
    def answer_medicine(self, query: str, want_price: bool = False) -> str:
        """Get Bangladesh medicine information"""
        try:
            return self.bd_medicines.get_medicine_info(query, include_price=want_price)
        except Exception as e:
            logger.error(f"Medicine info error: {e}")
            return f"I'm having trouble finding information about that medicine. Error: {str(e)}"
    
    def answer_bd_formulary_card(self, query: str) -> str:
        """Get formulary-style medicine card"""
        try:
            return self.bd_medicines.get_formulary_card(query)
        except Exception as e:
            logger.error(f"Formulary card error: {e}")
            return f"Unable to create medicine card. Error: {str(e)}"
    
    # ==================== GENERAL KNOWLEDGE ====================
    
    def answer_general_knowledge(
        self,
        query: str,
        context: Optional[List[str]] = None,
        conversation_context: str = "",
    ) -> str:
        """Answer general knowledge questions"""
        try:
            if context:
                context_text = "\n---\n".join(context)
                prompt = (
                    f"Answer the question using the provided context:\n\n"
                    f"CONTEXT:\n{context_text}\n\n"
                    f"QUESTION: {query}\n\nAnswer:"
                )
            else:
                prompt = f"Answer this question clearly and concisely:\n\nQUESTION: {query}\n\nAnswer:"
            
            system = get_system_prompt("general")
            return self.text_gen.generate_response(prompt, system)
            
        except Exception as e:
            logger.error(f"General knowledge error: {e}")
            return f"I'm unable to answer that question right now. Error: {str(e)}"
    
    # ==================== SMART ROUTING ====================
    
    def smart_response(self, query: str, image_data: Optional[bytes] = None) -> str:
        """Smart routing based on query type"""
        try:
            # Handle image queries
            if image_data:
                if "prescription" in query.lower():
                    result = self.analyze_prescription(image_data=image_data)
                    return result.get("summary", "Unable to analyze prescription")
                else:
                    return self.generate_vision_response(query, image_data=image_data)
            
            # Handle text queries
            query_lower = query.lower().strip()
            
            # Greetings
            greeting_words = ["hello", "hi", "hey", "good morning", "good evening"]
            if any(word in query_lower for word in greeting_words):
                return self.greeting_response(query)
            
            # Medicine queries
            if is_medical_query(query):
                medicine_words = ["medicine", "drug", "brand", "price", "mg", "tablet", "capsule"]
                if any(word in query_lower for word in medicine_words):
                    return self.answer_medicine(query, want_price=("price" in query_lower))
                else:
                    return self.answer_medical_query(query)
            
            # General queries
            return self.answer_general_knowledge(query)
            
        except Exception as e:
            logger.error(f"Smart response error: {e}")
            return f"I'm having trouble processing that request. Error: {str(e)}"
    
    # ==================== ADDITIONAL METHODS ====================
    
    def answer_over_ocr_text(self, question: str, ocr_text: str) -> str:
        """Answer question based on OCR text"""
        try:
            if not ocr_text.strip():
                return "I couldn't find any text in the image to answer your question."
            
            prompt = f"""Based on the following text extracted from an image, answer this question:

QUESTION: {question}

TEXT FROM IMAGE:
{ocr_text[:2000]}{'...' if len(ocr_text) > 2000 else ''}

Answer the question based only on the information in the text above. If the answer isn't in the text, say so."""
            
            return self.text_gen.generate_response(prompt, get_system_prompt("medical"))
        except Exception as e:
            logger.error(f"OCR text analysis error: {e}")
            return f"I couldn't analyze the text properly. Error: {str(e)}"
    
    def answer_over_pdf_text(self, question: str, pdf_text: str) -> str:
        """Answer question based on PDF text"""
        try:
            if not pdf_text.strip():
                return "The PDF appears to be empty or unreadable."
            
            # Truncate very long PDFs
            text_to_analyze = pdf_text[:8000] if len(pdf_text) > 8000 else pdf_text
            
            prompt = f"""Based on the following document content, answer this question:

QUESTION: {question}

DOCUMENT CONTENT:
{text_to_analyze}

Provide a comprehensive answer based on the document content. If the answer isn't in the document, clearly state that."""
            
            return self.text_gen.generate_response(prompt, get_system_prompt("medical"))
        except Exception as e:
            logger.error(f"PDF text analysis error: {e}")
            return f"I couldn't analyze the document properly. Error: {str(e)}"
    
    def default_image_brief(self, image_data: bytes) -> str:
        """Generate a brief description of an image"""
        try:
            return self.vision.analyze_image(
                image_data, 
                "Briefly describe what you see in this image. Focus on any medical or health-related content."
            )
        except Exception as e:
            logger.error(f"Image brief error: {e}")
            return f"I can see the image but couldn't analyze it properly. Error: {str(e)}"
    
    def try_fast_otc_dose_answer(self, query: str) -> Optional[str]:
        """Try to provide fast OTC dosing information"""
        try:
            query_lower = query.lower()
            
            # Common OTC medicines with basic info
            if "paracetamol" in query_lower or "acetaminophen" in query_lower:
                return "Paracetamol 500mg: Adults - 1-2 tablets every 4-6 hours, maximum 8 tablets in 24 hours. Always consult a healthcare provider for proper guidance."
            elif "ibuprofen" in query_lower:
                return "Ibuprofen: Adults - 200-400mg every 4-6 hours with food, maximum 1200mg in 24 hours. Consult healthcare provider if symptoms persist."
            elif "aspirin" in query_lower:
                return "Aspirin: Adults - 300-600mg every 4 hours, maximum 4g in 24 hours. Not recommended for children under 16. Consult healthcare provider."
            
            return None
        except Exception as e:
            logger.error(f"Fast OTC dose answer error: {e}")
            return None
