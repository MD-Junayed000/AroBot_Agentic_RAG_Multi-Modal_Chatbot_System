# core/llm/vision.py
"""Vision processing module - image analysis with OCR fallback"""

import base64
import logging
import time
from typing import Optional
from PIL import Image
import io
import ollama
from config.env_config import OLLAMA_BASE_URL, OLLAMA_VISION_MODEL
from utils.ocr_pipeline import OCRPipeline

logger = logging.getLogger(__name__)

class VisionProcessor:
    """Handles vision model interactions with fallbacks"""
    
    def __init__(self):
        self.client = ollama.Client(host=OLLAMA_BASE_URL)
        self.vision_model = OLLAMA_VISION_MODEL
        self._last_request_time = 0
        self._min_interval = 2  # Minimum 2 seconds between requests
    
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