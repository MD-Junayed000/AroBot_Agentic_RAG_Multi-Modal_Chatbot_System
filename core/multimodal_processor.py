"""
Multi-modal processor for handling images, text, and embeddings
"""
import io
import base64
from typing import List, Dict, Any, Optional, Union
from PIL import Image
import numpy as np
from sentence_transformers import SentenceTransformer
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

class MultiModalProcessor:
    """Handles multi-modal processing including CLIP embeddings"""
    
    def __init__(self):
        self.ocr_pipeline = OCRPipeline(lang=OCR_LANGUAGE)
        
        # Initialize CLIP for multimodal embeddings
        if CLIP_AVAILABLE:
            try:
                self.clip_model, self.clip_preprocess = clip.load("ViT-B/32")
                self.clip_available = True
            except Exception as e:
                logger.warning(f"CLIP model not available: {e}")
                self.clip_available = False
        else:
            self.clip_available = False
        
        # Initialize text embedding model
        self.text_embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    
    def process_image(self, image_input: Union[str, bytes, Image.Image]) -> Dict[str, Any]:
        """Process image through OCR and return extracted information"""
        try:
            # Handle different input types
            if isinstance(image_input, str):
                # File path
                image_path = image_input
                image = Image.open(image_path)
            elif isinstance(image_input, bytes):
                # Bytes data
                image = Image.open(io.BytesIO(image_input))
                image_path = None
            elif isinstance(image_input, Image.Image):
                # PIL Image
                image = image_input
                image_path = None
            else:
                raise ValueError("Unsupported image input type")
            
            # Extract text using OCR
            if image_path:
                lines, items = self.ocr_pipeline.run_on_image(image_path)
            else:
                # Save temporary image for OCR processing
                temp_path = "temp_image.jpg"
                image.save(temp_path)
                lines, items = self.ocr_pipeline.run_on_image(temp_path)
            
            # Extract structured information
            raw_text = "\n".join(lines)
            
            # Filter items by confidence (simulated)
            high_confidence_items = [
                item for item in items 
                if item.get('raw', '').strip() and len(item.get('raw', '').strip()) > 2
            ]
            
            return {
                "raw_text": raw_text,
                "lines": lines,
                "structured_items": high_confidence_items,
                "item_count": len(high_confidence_items),
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            return {
                "error": str(e),
                "status": "error"
            }
    
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
