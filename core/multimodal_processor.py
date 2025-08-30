"""
Multi-modal processor for handling images, text, and embeddings
"""
import io, base64
from typing import List, Dict, Any, Optional, Union
from PIL import Image
import numpy as np
from sentence_transformers import SentenceTransformer
import tempfile
import torch
try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    print("Warning: OpenAI CLIP not available. Install with: pip install git+https://github.com/openai/CLIP.git")
    CLIP_AVAILABLE = False
from utils.ocr_pipeline import OCRPipeline
from config.env_config import OCR_LANGUAGE, OCR_CONFIDENCE_THRESHOLD
import logging

logger = logging.getLogger(__name__)

def _looks_like_b64(s: str) -> bool:
    try:
        if "," in s and s.split(",")[0].startswith("data:"):
            return True
        base64.b64decode(s, validate=True)
        return True
    except Exception:
        return False

class MultiModalProcessor:
    def __init__(self):
        self.ocr_pipeline = OCRPipeline(lang=OCR_LANGUAGE)
        # CLIP setup unchanged...
        self.text_embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        
        self.clip_available = False
        if CLIP_AVAILABLE:
            try:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=device)
                self.clip_model.eval()

                self.clip_available = True
            except Exception:
                self.clip_available = False

    def process_image(self, image_input: Union[str, bytes, Image.Image]) -> Dict[str, Any]:
        try:
            if isinstance(image_input, Image.Image):
                image = image_input
                image_path = None
            elif isinstance(image_input, bytes):
                image = Image.open(io.BytesIO(image_input)).convert("RGB")
                image_path = None
            elif isinstance(image_input, str):
                if _looks_like_b64(image_input):
                    raw = image_input.split(",")[-1]
                    image = Image.open(io.BytesIO(base64.b64decode(raw))).convert("RGB")
                    image_path = None
                else:
                    image_path = image_input
                    image = Image.open(image_path).convert("RGB")
            else:
                raise ValueError("Unsupported image input type")

            # OCR
            if image_path:
                lines, items = self.ocr_pipeline.run_on_image(image_path)
            else:
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=True) as tf:
                    image.save(tf.name, format="JPEG", quality=85, optimize=True)
                    lines, items = self.ocr_pipeline.run_on_image(tf.name)

            raw_text = "\n".join(lines)
            # defend against weird shapes
            safe_items = []
            for it in items or []:
                if isinstance(it, dict):
                    if it.get("raw", "").strip():
                        safe_items.append(it)
            return {
                "raw_text": raw_text,
                "lines": lines,
                "structured_items": safe_items,
                "item_count": len(safe_items),
                "status": "success"
            }
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            return {"error": str(e), "status": "error"}

    def generate_multimodal_embeddings(self, text: str, image: Optional[Image.Image] = None) -> Dict[str, np.ndarray]:
        """Generate embeddings for text and optionally combine with image embeddings"""
        embeddings = {}
        
        try:
            # Generate text embeddings
            text_embedding = self.text_embedder.encode([text], normalize_embeddings=True)[0]
            embeddings['text'] = text_embedding
            
            # Generate image embeddings if CLIP is available and image is provided
            if self.clip_available and image:
                with torch.no_grad():
                    # Preprocess image for CLIP
                    image_tensor = self.clip_preprocess(image).unsqueeze(0)
                    
                    # Generate image embedding
                    image_features = self.clip_model.encode_image(image_tensor)
                    image_embedding = image_features.cpu().numpy()[0]
                    embeddings['image'] = image_embedding
                    
                    # Generate text embedding using CLIP
                    text_tokens = clip.tokenize([text])
                    text_features = self.clip_model.encode_text(text_tokens)
                    clip_text_embedding = text_features.cpu().numpy()[0]
                    embeddings['clip_text'] = clip_text_embedding
                    
                    # Combine embeddings (simple concatenation)
                    combined_embedding = np.concatenate([image_embedding, clip_text_embedding])
                    embeddings['multimodal'] = combined_embedding
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating multimodal embeddings: {e}")
            return {"error": str(e)}
    
    def extract_prescription_entities(self, ocr_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract medical entities from OCR results"""
        try:
            structured_items = ocr_results.get('structured_items', [])
            raw_text = ocr_results.get('raw_text', '')
            
            # Extract medication information
            medications = []
            patient_info = {}
            doctor_info = {}
            
            for item in structured_items:
                raw = item.get('raw', '').lower()
                
                # Look for medication patterns
                if any(keyword in raw for keyword in ['mg', 'ml', 'tablet', 'cap', 'syrup']):
                    medication = {
                        'text': item.get('raw', ''),
                        'strength': item.get('strength'),
                        'unit': item.get('unit'),
                        'frequency': item.get('frequency'),
                        'duration': item.get('duration')
                    }
                    medications.append(medication)
                
                # Look for patient information
                if any(keyword in raw for keyword in ['patient', 'name:', 'age:', 'gender:']):
                    if 'name' in raw:
                        patient_info['name_line'] = item.get('raw', '')
                    elif 'age' in raw:
                        patient_info['age_line'] = item.get('raw', '')
                
                # Look for doctor information
                if any(keyword in raw for keyword in ['dr.', 'doctor', 'clinic', 'hospital']):
                    if 'dr.' in raw or 'doctor' in raw:
                        doctor_info['doctor_line'] = item.get('raw', '')
                    elif 'clinic' in raw or 'hospital' in raw:
                        doctor_info['facility_line'] = item.get('raw', '')
            
            return {
                'medications': medications,
                'patient_info': patient_info,
                'doctor_info': doctor_info,
                'medication_count': len(medications),
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"Error extracting prescription entities: {e}")
            return {
                'error': str(e),
                'status': 'error'
            }
    
    def process_prescription_image(self, image_input: Union[str, bytes, Image.Image]) -> Dict[str, Any]:
        """Complete prescription processing pipeline"""
        try:
            # Step 1: OCR processing
            ocr_results = self.process_image(image_input)
            if ocr_results.get('status') == 'error':
                return ocr_results
            
            # Step 2: Entity extraction
            entities = self.extract_prescription_entities(ocr_results)
            
            # Step 3: Generate embeddings
            raw_text = ocr_results.get('raw_text', '')
            if raw_text:
                embeddings = self.generate_multimodal_embeddings(
                    raw_text, 
                    image_input if isinstance(image_input, Image.Image) else None
                )
            else:
                embeddings = {}
            
            return {
                'ocr_results': ocr_results,
                'entities': entities,
                'embeddings': {k: v.tolist() if isinstance(v, np.ndarray) else v 
                             for k, v in embeddings.items()},
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"Error in prescription processing pipeline: {e}")
            return {
                'error': str(e),
                'status': 'error'
            }