# core/llm/vision.py
"""Vision processing module - image analysis with OCR fallback and CLIP support"""

import base64
import logging
import time
from typing import Optional, Dict, Any, List
from PIL import Image
import io
import ollama
import torch
import numpy as np
from config.env_config import OLLAMA_BASE_URL, OLLAMA_VISION_MODEL
from utils.ocr_pipeline import OCRPipeline

# CLIP imports
try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("Warning: OpenAI CLIP not available. Install with: pip install git+https://github.com/openai/CLIP.git")

logger = logging.getLogger(__name__)

class VisionProcessor:
    """Handles vision model interactions with fallbacks and CLIP support"""
    
    def __init__(self):
        self.client = ollama.Client(host=OLLAMA_BASE_URL)
        self.vision_model = OLLAMA_VISION_MODEL
        self._last_request_time = 0
        self._min_interval = 2  # Minimum 2 seconds between requests
        
        # CLIP setup
        self.clip_available = False
        self.clip_model = None
        self.clip_preprocess = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if CLIP_AVAILABLE:
            try:
                self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
                self.clip_model.eval()
                self.clip_available = True
                logger.info("CLIP model loaded successfully")
            except Exception as e:
                logger.warning(f"CLIP initialization failed: {e}")
                self.clip_available = False
    
    def analyze_image(self, image_data: bytes, prompt: str = "Describe this image") -> str:
        """Analyze image with vision model, OCR fallback"""
        try:
            # Rate limiting to prevent resource exhaustion
            current_time = time.time()
            if current_time - self._last_request_time < self._min_interval:
                time.sleep(self._min_interval - (current_time - self._last_request_time))
            
            # Optimize image size first
            optimized_image = self.optimize_image(image_data, max_size=512)
            
            # Try vision model with reduced resource requirements
            b64 = base64.b64encode(optimized_image).decode()
            r = self.client.chat(
                model=self.vision_model,
                messages=[{"role": "user", "content": prompt, "images": [b64]}],
                stream=False,
                options={
                    "temperature": 0.1, 
                    "num_ctx": 1024,  # Reduced from 1024
                    "num_predict": 100,  # Reduced from 150
                    "num_gpu": 1,
                    "num_thread": 2
                }
            )
            self._last_request_time = time.time()
            content = r.get("message", {}).get("content", "").strip()
            if content and len(content) > 10:
                return content
        except Exception as e:
            logger.warning(f"Vision model failed: {e}")
        
        # OCR fallback (which is working based on your screenshots)
        return self._ocr_fallback(image_data, prompt)
    
    def generate_multimodal_embeddings(self, text: str, image: Optional[Image.Image] = None) -> Dict[str, np.ndarray]:
        """Generate multimodal embeddings using CLIP and sentence transformers"""
        try:
            from sentence_transformers import SentenceTransformer
            
            embeddings = {}
            
            # Generate text embeddings
            if text:
                try:
                    text_embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
                    text_embedding = text_embedder.encode([text], normalize_embeddings=True)[0]
                    embeddings['text'] = text_embedding
                except Exception as e:
                    logger.warning(f"Text embedding generation failed: {e}")
            
            # Generate image embeddings if CLIP is available and image is provided
            if self.clip_available and image:
                try:
                    with torch.no_grad():
                        # Preprocess image for CLIP
                        image_tensor = self.clip_preprocess(image).unsqueeze(0)
                        
                        # Generate image embedding
                        image_features = self.clip_model.encode_image(image_tensor)
                        image_embedding = image_features.cpu().numpy()[0]
                        embeddings['image'] = image_embedding
                        
                        # Generate text embedding using CLIP
                        if text:
                            text_tokens = clip.tokenize([text])
                            text_features = self.clip_model.encode_text(text_tokens)
                            clip_text_embedding = text_features.cpu().numpy()[0]
                            embeddings['clip_text'] = clip_text_embedding
                            
                            # Combine embeddings (simple concatenation)
                            combined_embedding = np.concatenate([image_embedding, clip_text_embedding])
                            embeddings['multimodal'] = combined_embedding
                        
                except Exception as e:
                    logger.warning(f"CLIP embedding generation failed: {e}")
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating multimodal embeddings: {e}")
            return {"error": str(e)}
    
    def analyze_image_with_clip(self, image: Image.Image, text_prompt: str = "") -> Dict[str, Any]:
        """Analyze image using CLIP for better understanding"""
        try:
            if not self.clip_available:
                return {"error": "CLIP not available"}
            
            # Generate CLIP embeddings
            embeddings = self.generate_multimodal_embeddings(text_prompt, image)
            
            # Use CLIP for image classification/description
            with torch.no_grad():
                # Preprocess image
                image_tensor = self.clip_preprocess(image).unsqueeze(0)
                
                # Get image features
                image_features = self.clip_model.encode_image(image_tensor)
                
                # Generate description using CLIP's text encoder
                if text_prompt:
                    text_tokens = clip.tokenize([text_prompt])
                    text_features = self.clip_model.encode_text(text_tokens)
                    
                    # Calculate similarity
                    similarity = torch.cosine_similarity(image_features, text_features)
                    similarity_score = similarity.item()
                else:
                    similarity_score = 0.0
                
                # Generate embeddings
                image_embedding = image_features.cpu().numpy()[0]
                
                return {
                    "image_embedding": image_embedding,
                    "similarity_score": similarity_score,
                    "clip_available": True,
                    "embeddings": embeddings
                }
                
        except Exception as e:
            logger.error(f"CLIP analysis failed: {e}")
            return {"error": str(e), "clip_available": False}
    
    def _ocr_fallback(self, image_data: bytes, prompt: str = "") -> str:
        """OCR-based fallback analysis"""
        try:
            ocr = OCRPipeline(lang="en")
            lines, _, _ = ocr.run_on_bytes(image_data)
            text = "\n".join(lines)
            
            if text.strip():
                if prompt and "prescription" in prompt.lower():
                    return f"**Prescription Text Detected:**\n{text[:600]}{'...' if len(text) > 600 else ''}"
                else:
                    return f"**Text extracted from image:**\n{text[:500]}{'...' if len(text) > 500 else ''}"
            else:
                return "No readable text found in the image."
                
        except Exception as e:
            logger.error(f"OCR fallback failed: {e}")
            return f"Unable to analyze the image. Error: {str(e)}"
    
    def optimize_image(self, image_data: bytes, max_size: int = 512) -> bytes:
        """Optimize image for vision processing"""
        try:
            img = Image.open(io.BytesIO(image_data))
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            width, height = img.size
            max_dimension = max(width, height)
            
            if max_dimension > max_size:
                ratio = max_size / max_dimension
                new_width = int(width * ratio)
                new_height = int(height * ratio)
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            output = io.BytesIO()
            img.save(output, format='JPEG', quality=75, optimize=True)  # Reduced quality
            return output.getvalue()
            
        except Exception as e:
            logger.warning(f"Image optimization failed: {e}")
            return image_data
    
    def short_answer(self, question: str, image_data: bytes) -> str:
        """Get short answer from image"""
        prompt = f"Answer briefly: {question}"
        return self.analyze_image(image_data, prompt)
    
    def get_clip_status(self) -> Dict[str, Any]:
        """Get CLIP availability and status"""
        return {
            "clip_available": self.clip_available,
            "device": self.device,
            "model_loaded": self.clip_model is not None
        }