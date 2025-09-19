"""
Multi-modal processor for handling images, text, and embeddings
"""
import io
import re
import base64
import tempfile
import logging
import numpy as np
from PIL import Image
from typing import Dict, Any, Union, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum

try:
    import torch
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    print("Warning: OpenAI CLIP not available. Install with: pip install git+https://github.com/openai/CLIP.git")
    CLIP_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("Warning: SentenceTransformers not available. Install with: pip install sentence-transformers")
    SENTENCE_TRANSFORMERS_AVAILABLE = False

from utils.ocr_pipeline import OCRPipeline

try:
    from config.env_config import OCR_LANGUAGE
except ImportError:
    OCR_LANGUAGE = "en"

logger = logging.getLogger(__name__)

# Image classification enums and dataclasses
class ImageCategory(Enum):
    PRESCRIPTION = "prescription"
    MEDICINE_PACKAGE = "medicine_package" 
    MEDICINE_BOTTLE = "medicine_bottle"
    MEDICINE_STRIP = "medicine_strip"
    LAB_RESULTS = "lab_results"
    MEDICAL_REPORT = "medical_report"
    XRAY_SCAN = "xray_scan"
    ANATOMY_DIAGRAM = "anatomy_diagram"
    MEDICAL_CHART = "medical_chart"
    SYMPTOM_GUIDE = "symptom_guide"
    GENERAL_IMAGE = "general_image"
    DOCUMENT = "document"
    UNKNOWN = "unknown"

@dataclass
class ImageClassification:
    category: ImageCategory
    confidence: float
    visual_confidence: float
    text_confidence: float
    context_confidence: float
    features: Dict[str, Any]
    reasoning: str

@dataclass
class VisualFeatures:
    has_header_layout: bool
    has_table_layout: bool
    has_signature_area: bool
    has_medical_logo: bool
    is_scan_image: bool
    is_diagram: bool
    is_chart: bool
    has_text_regions: bool
    has_medical_symbols: bool
    layout_type: str
    aspect_ratio: float

@dataclass
class TextFeatures:
    raw_text: str
    entities: Dict[str, List[str]]
    document_type: str
    medicine_count: int
    has_prescription_format: bool
    has_lab_format: bool
    confidence: float

def _looks_like_b64(s: str) -> bool:
    try:
        if isinstance(s, str):
            sb_bytes = bytes(s, 'ascii')
        elif isinstance(s, bytes):
            sb_bytes = s
        else:
            raise ValueError("Argument must be string or bytes")
        return base64.b64encode(base64.b64decode(sb_bytes)) == sb_bytes
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
                lines, items, header_entities = self.ocr_pipeline.run_on_image(image_path)
            else:
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=True) as tf:
                    image.save(tf.name, format="JPEG", quality=85, optimize=True)
                    lines, items, header_entities = self.ocr_pipeline.run_on_image(tf.name)

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
                "header_entities": header_entities,
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

    def classify_image(self, image_input: Union[str, bytes, Image.Image], 
                      user_question: Optional[str] = None, 
                      session_context: Optional[str] = None) -> ImageClassification:
        """Advanced multi-stage image classification with confidence scoring"""
        try:
            # Convert to PIL Image
            if isinstance(image_input, Image.Image):
                image = image_input
            elif isinstance(image_input, bytes):
                image = Image.open(io.BytesIO(image_input)).convert("RGB")
            elif isinstance(image_input, str):
                if _looks_like_b64(image_input):
                    raw = image_input.split(",")[-1]
                    image = Image.open(io.BytesIO(base64.b64decode(raw))).convert("RGB")
                else:
                    image = Image.open(image_input).convert("RGB")
            else:
                raise ValueError("Unsupported image input type")

            # Stage 1: Visual Analysis
            visual_features = self._analyze_visual_features(image)
            
            # Stage 2: Text Analysis
            text_features = self._analyze_text_features(image_input)
            
            # Stage 3: Context Analysis
            context_confidence = self._analyze_context(user_question, session_context)
            
            # Stage 4: Combined Classification
            classification = self._combine_classifications(
                visual_features, text_features, context_confidence, user_question
            )
            
            return classification
            
        except Exception as e:
            logger.error(f"Error in image classification: {e}")
            return ImageClassification(
                category=ImageCategory.UNKNOWN,
                confidence=0.0,
                visual_confidence=0.0,
                text_confidence=0.0,
                context_confidence=0.0,
                features={},
                reasoning=f"Classification failed: {str(e)}"
            )

    def _analyze_visual_features(self, image: Image.Image) -> VisualFeatures:
        """Analyze visual characteristics of the image"""
        try:
            width, height = image.size
            aspect_ratio = width / height
            
            # Convert to numpy for analysis
            img_array = np.array(image)
            
            # Detect layout features
            has_header = self._detect_header_layout(img_array)
            has_table = self._detect_table_layout(img_array)
            has_signature = self._detect_signature_area(img_array)
            has_logo = self._detect_medical_logo(img_array)
            
            # Detect content type
            is_scan = self._is_medical_scan(img_array)
            is_diagram = self._is_medical_diagram(img_array)
            is_chart = self._is_medical_chart(img_array)
            
            # Detect text regions
            has_text = self._detect_text_regions(img_array)
            has_symbols = self._detect_medical_symbols(img_array)
            
            # Determine layout type
            layout_type = self._determine_layout_type(
                has_header, has_table, has_signature, aspect_ratio
            )
            
            return VisualFeatures(
                has_header_layout=has_header,
                has_table_layout=has_table,
                has_signature_area=has_signature,
                has_medical_logo=has_logo,
                is_scan_image=is_scan,
                is_diagram=is_diagram,
                is_chart=is_chart,
                has_text_regions=has_text,
                has_medical_symbols=has_symbols,
                layout_type=layout_type,
                aspect_ratio=aspect_ratio
            )
            
        except Exception as e:
            logger.warning(f"Visual analysis failed: {e}")
            return VisualFeatures(
                has_header_layout=False, has_table_layout=False, has_signature_area=False,
                has_medical_logo=False, is_scan_image=False, is_diagram=False,
                is_chart=False, has_text_regions=False, has_medical_symbols=False,
                layout_type="unknown", aspect_ratio=1.0
            )

    def _analyze_text_features(self, image_input: Union[str, bytes, Image.Image]) -> TextFeatures:
        """Analyze text content extracted from image"""
        try:
            # Get OCR results
            ocr_result = self.process_image(image_input)
            raw_text = ocr_result.get('raw_text', '')
            
            # Extract medical entities
            entities = self._extract_medical_entities(raw_text)
            
            # Classify document type based on text
            document_type = self._classify_document_type(raw_text, entities)
            
            # Count medicines
            medicine_count = len(entities.get('medicines', []))
            
            # Check format patterns
            has_prescription_format = self._has_prescription_format(raw_text)
            has_lab_format = self._has_lab_format(raw_text)
            
            # Calculate text confidence
            confidence = ocr_result.get('confidence', 0.5)
            
            return TextFeatures(
                raw_text=raw_text,
                entities=entities,
                document_type=document_type,
                medicine_count=medicine_count,
                has_prescription_format=has_prescription_format,
                has_lab_format=has_lab_format,
                confidence=confidence
            )
            
        except Exception as e:
            logger.warning(f"Text analysis failed: {e}")
            return TextFeatures(
                raw_text="", entities={}, document_type="unknown",
                medicine_count=0, has_prescription_format=False,
                has_lab_format=False, confidence=0.0
            )

    def _analyze_context(self, user_question: Optional[str], session_context: Optional[str]) -> float:
        """Analyze user context to improve classification"""
        context_confidence = 0.5  # Default neutral confidence
        
        if user_question:
            question_lower = user_question.lower()
            
            # High confidence indicators
            if any(phrase in question_lower for phrase in [
                'prescription', 'medicine name', 'drug information', 'dosage'
            ]):
                context_confidence = 0.9
            elif any(phrase in question_lower for phrase in [
                'lab results', 'blood test', 'report'
            ]):
                context_confidence = 0.8
            elif any(phrase in question_lower for phrase in [
                'anatomy', 'diagram', 'body part'
            ]):
                context_confidence = 0.7
            elif any(phrase in question_lower for phrase in [
                'identify', 'what is this', 'analyze'
            ]):
                context_confidence = 0.6
        
        return context_confidence

    def _combine_classifications(self, visual_features: VisualFeatures, 
                               text_features: TextFeatures, 
                               context_confidence: float,
                               user_question: Optional[str]) -> ImageClassification:
        """Combine all analysis results into final classification"""
        
        # Calculate individual confidences
        visual_confidence = self._calculate_visual_confidence(visual_features)
        text_confidence = text_features.confidence
        
        # Classification logic with scoring
        category_scores = {
            ImageCategory.PRESCRIPTION: 0.0,
            ImageCategory.MEDICINE_PACKAGE: 0.0,
            ImageCategory.LAB_RESULTS: 0.0,
            ImageCategory.ANATOMY_DIAGRAM: 0.0,
            ImageCategory.MEDICAL_CHART: 0.0,
            ImageCategory.GENERAL_IMAGE: 0.0
        }
        
        # Prescription scoring
        if text_features.has_prescription_format:
            category_scores[ImageCategory.PRESCRIPTION] += 0.4
        if visual_features.has_header_layout and visual_features.has_signature_area:
            category_scores[ImageCategory.PRESCRIPTION] += 0.3
        if text_features.medicine_count > 0:
            category_scores[ImageCategory.PRESCRIPTION] += 0.2
        if visual_features.has_table_layout:
            category_scores[ImageCategory.PRESCRIPTION] += 0.1
            
        # Medicine package scoring
        if any(term in text_features.raw_text.lower() for term in ['manufactured', 'batch', 'expiry', 'mfg']):
            category_scores[ImageCategory.MEDICINE_PACKAGE] += 0.4
        if visual_features.aspect_ratio > 1.2 and visual_features.has_text_regions:
            category_scores[ImageCategory.MEDICINE_PACKAGE] += 0.3
        if text_features.medicine_count == 1:  # Single medicine name
            category_scores[ImageCategory.MEDICINE_PACKAGE] += 0.2
            
        # Lab results scoring
        if text_features.has_lab_format:
            category_scores[ImageCategory.LAB_RESULTS] += 0.5
        if visual_features.has_table_layout and not visual_features.has_signature_area:
            category_scores[ImageCategory.LAB_RESULTS] += 0.3
            
        # Anatomy diagram scoring
        if visual_features.is_diagram:
            category_scores[ImageCategory.ANATOMY_DIAGRAM] += 0.4
        if any(term in text_features.raw_text.lower() for term in ['anatomy', 'organ', 'muscle', 'bone']):
            category_scores[ImageCategory.ANATOMY_DIAGRAM] += 0.3
        if visual_features.has_medical_symbols:
            category_scores[ImageCategory.ANATOMY_DIAGRAM] += 0.2
            
        # Medical chart scoring
        if visual_features.is_chart:
            category_scores[ImageCategory.MEDICAL_CHART] += 0.4
        if any(term in text_features.raw_text.lower() for term in ['chart', 'graph', 'scale', 'measurement']):
            category_scores[ImageCategory.MEDICAL_CHART] += 0.3
            
        # Context adjustments
        if user_question:
            question_lower = user_question.lower()
            if 'prescription' in question_lower:
                category_scores[ImageCategory.PRESCRIPTION] += 0.2
            elif 'package' in question_lower:
                category_scores[ImageCategory.MEDICINE_PACKAGE] += 0.2
            elif 'lab' in question_lower or 'test' in question_lower:
                category_scores[ImageCategory.LAB_RESULTS] += 0.2
                
        # Find best category
        best_category = max(category_scores.items(), key=lambda x: x[1])
        category = best_category[0]
        category_confidence = min(1.0, best_category[1])
        
        # If no category has good confidence, default to general
        if category_confidence < 0.3:
            category = ImageCategory.GENERAL_IMAGE
            category_confidence = 0.5
            
        # Overall confidence calculation
        weights = {'visual': 0.4, 'text': 0.4, 'context': 0.2}
        overall_confidence = (
            visual_confidence * weights['visual'] +
            text_confidence * weights['text'] +
            context_confidence * weights['context']
        ) * category_confidence
        
        # Generate reasoning
        reasoning = self._generate_reasoning(category, visual_features, text_features, category_confidence)
        
        return ImageClassification(
            category=category,
            confidence=min(1.0, max(0.0, overall_confidence)),
            visual_confidence=visual_confidence,
            text_confidence=text_confidence,
            context_confidence=context_confidence,
            features={
                'visual': visual_features.__dict__,
                'text': text_features.__dict__,
                'category_scores': {k.value: v for k, v in category_scores.items()}
            },
            reasoning=reasoning
        )

    def _detect_header_layout(self, img_array: np.ndarray) -> bool:
        """Detect if image has header layout typical of medical documents"""
        height, width = img_array.shape[:2]
        top_region = img_array[:int(height * 0.2), :]
        
        # Simple heuristic: check if top region has different characteristics
        top_mean = np.mean(top_region)
        full_mean = np.mean(img_array)
        
        return abs(top_mean - full_mean) > 10  # Threshold for difference

    def _detect_table_layout(self, img_array: np.ndarray) -> bool:
        """Detect table-like structures"""
        # Simple edge detection to find grid patterns
        try:
            import cv2
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            
            # Look for horizontal and vertical lines
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
            
            horizontal_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horizontal_kernel)
            vertical_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, vertical_kernel)
            
            h_score = np.sum(horizontal_lines) / (img_array.shape[0] * img_array.shape[1])
            v_score = np.sum(vertical_lines) / (img_array.shape[0] * img_array.shape[1])
            
            return (h_score > 0.01 and v_score > 0.01)  # Both horizontal and vertical lines
        except:
            return False

    def _detect_signature_area(self, img_array: np.ndarray) -> bool:
        """Detect signature areas in documents"""
        height, width = img_array.shape[:2]
        bottom_region = img_array[int(height * 0.7):, :]
        
        # Look for sparse content in bottom region (typical of signature areas)
        bottom_variance = np.var(bottom_region)
        full_variance = np.var(img_array)
        
        return bottom_variance < full_variance * 0.8

    def _detect_medical_logo(self, img_array: np.ndarray) -> bool:
        """Detect medical logos or symbols"""
        # Simple heuristic based on color distribution in top corners
        height, width = img_array.shape[:2]
        top_left = img_array[:int(height * 0.2), :int(width * 0.2)]
        top_right = img_array[:int(height * 0.2), int(width * 0.8):]
        
        # Check for distinct regions (logos often have different color schemes)
        tl_mean = np.mean(top_left)
        tr_mean = np.mean(top_right)
        
        return abs(tl_mean - tr_mean) > 15

    def _is_medical_scan(self, img_array: np.ndarray) -> bool:
        """Detect if image is a medical scan (X-ray, CT, etc.)"""
        # Medical scans often have high contrast and specific intensity distributions
        gray_levels = np.mean(img_array, axis=2) if len(img_array.shape) == 3 else img_array
        
        # Check for high contrast (medical scans often have black backgrounds)
        low_intensity = np.sum(gray_levels < 50) / gray_levels.size
        high_intensity = np.sum(gray_levels > 200) / gray_levels.size
        
        return low_intensity > 0.3 and high_intensity > 0.1

    def _is_medical_diagram(self, img_array: np.ndarray) -> bool:
        """Detect if image is a medical diagram"""
        # Diagrams often have clean lines and limited color palette
        if len(img_array.shape) == 3:
            # Check color diversity
            unique_colors = len(np.unique(img_array.reshape(-1, img_array.shape[2]), axis=0))
            total_pixels = img_array.shape[0] * img_array.shape[1]
            color_ratio = unique_colors / total_pixels
            
            return color_ratio < 0.1  # Limited color palette typical of diagrams
        return False

    def _is_medical_chart(self, img_array: np.ndarray) -> bool:
        """Detect if image is a medical chart or graph"""
        # Charts often have regular patterns and axes
        try:
            import cv2
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            # Look for straight lines (axes)
            lines = cv2.HoughLines(cv2.Canny(gray, 50, 150), 1, np.pi/180, threshold=100)
            
            return lines is not None and len(lines) > 5
        except:
            return False

    def _detect_text_regions(self, img_array: np.ndarray) -> bool:
        """Detect if image has significant text regions"""
        # Simple heuristic: text regions have medium intensity values
        gray_levels = np.mean(img_array, axis=2) if len(img_array.shape) == 3 else img_array
        medium_intensity = np.sum((gray_levels > 100) & (gray_levels < 200)) / gray_levels.size
        
        return medium_intensity > 0.3

    def _detect_medical_symbols(self, img_array: np.ndarray) -> bool:
        """Detect medical symbols in the image"""
        # Placeholder - could be enhanced with symbol detection
        return False

    def _determine_layout_type(self, has_header: bool, has_table: bool, 
                              has_signature: bool, aspect_ratio: float) -> str:
        """Determine the overall layout type"""
        if has_header and has_signature and has_table:
            return "formal_document"
        elif has_table:
            return "tabular"
        elif aspect_ratio > 1.5:
            return "landscape"
        elif aspect_ratio < 0.7:
            return "portrait"
        else:
            return "standard"

    def _calculate_visual_confidence(self, visual_features: VisualFeatures) -> float:
        """Calculate confidence based on visual features"""
        confidence = 0.5  # Base confidence
        
        if visual_features.has_text_regions:
            confidence += 0.2
        if visual_features.has_header_layout:
            confidence += 0.1
        if visual_features.has_table_layout:
            confidence += 0.1
        if visual_features.layout_type != "unknown":
            confidence += 0.1
            
        return min(1.0, confidence)

    def _extract_medical_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract medical entities from text"""
        text_lower = text.lower()
        entities = {
            'medicines': [],
            'doctors': [],
            'dosages': [],
            'symptoms': [],
            'tests': []
        }
        
        # Medicine patterns
        medicine_patterns = [
            r'tab\s+(\w+)',
            r'syp\s+(\w+)',
            r'cap\s+(\w+)',
            r'inj\s+(\w+)',
            r'(\w+)\s+\d+\s*mg',
            r'(\w+)\s+\d+\s*ml'
        ]
        
        for pattern in medicine_patterns:
            matches = re.findall(pattern, text_lower)
            entities['medicines'].extend(matches)
        
        # Doctor patterns
        doctor_matches = re.findall(r'dr\.?\s+([a-z\s]+)', text_lower)
        entities['doctors'].extend(doctor_matches)
        
        # Dosage patterns
        dosage_matches = re.findall(r'\d+\s*(mg|ml|g|mcg)', text_lower)
        entities['dosages'].extend(dosage_matches)
        
        # Remove duplicates
        for key in entities:
            entities[key] = list(set(entities[key]))
        
        return entities

    def _classify_document_type(self, text: str, entities: Dict[str, List[str]]) -> str:
        """Classify document type based on text patterns"""
        text_lower = text.lower()
        
        # Prescription indicators
        prescription_score = 0
        if any(pattern in text_lower for pattern in ['prescription', 'rx', 'dr.', 'doctor']):
            prescription_score += 2
        if len(entities['medicines']) > 0:
            prescription_score += 2
        if any(pattern in text_lower for pattern in ['tab', 'syp', 'cap', 'inj']):
            prescription_score += 1
        
        # Lab results indicators
        lab_score = 0
        if any(pattern in text_lower for pattern in ['lab report', 'test results', 'blood test']):
            lab_score += 3
        if any(pattern in text_lower for pattern in ['normal', 'abnormal', 'reference', 'range']):
            lab_score += 1
        
        # Medical report indicators
        report_score = 0
        if any(pattern in text_lower for pattern in ['report', 'findings', 'diagnosis']):
            report_score += 2
        if any(pattern in text_lower for pattern in ['patient', 'age', 'gender']):
            report_score += 1
        
        # Return highest scoring type
        scores = {
            'prescription': prescription_score,
            'lab_results': lab_score,
            'medical_report': report_score
        }
        
        best_type = max(scores.items(), key=lambda x: x[1])
        return best_type[0] if best_type[1] > 0 else 'general_document'

    def _has_prescription_format(self, text: str) -> bool:
        """Check if text has prescription format"""
        text_lower = text.lower()
        
        # Look for prescription-specific patterns
        prescription_indicators = [
            r'dr\.?\s+\w+',  # Doctor name
            r'rx\s*:',       # RX symbol
            r'tab\s+\w+',    # Tablet
            r'syp\s+\w+',    # Syrup
            r'\d+-\d+-\d+',  # Dosage pattern like 1-0-1
            r'for\s+\d+\s+days' # Duration
        ]
        
        matches = sum(1 for pattern in prescription_indicators 
                     if re.search(pattern, text_lower))
        
        return matches >= 2

    def _has_lab_format(self, text: str) -> bool:
        """Check if text has lab results format"""
        text_lower = text.lower()
        
        lab_indicators = [
            r'test\s+name',
            r'result',
            r'normal\s+range',
            r'reference\s+value',
            r'units?',
            r'\d+\.\d+\s*(mg/dl|mmol/l|g/dl)'
        ]
        
        matches = sum(1 for pattern in lab_indicators 
                     if re.search(pattern, text_lower))
        
        return matches >= 2

    def _generate_reasoning(self, category: ImageCategory, visual_features: VisualFeatures, 
                          text_features: TextFeatures, confidence: float) -> str:
        """Generate human-readable reasoning for the classification"""
        reasons = []
        
        if category == ImageCategory.PRESCRIPTION:
            if text_features.has_prescription_format:
                reasons.append("prescription format detected in text")
            if visual_features.has_header_layout:
                reasons.append("medical document layout")
            if text_features.medicine_count > 0:
                reasons.append(f"{text_features.medicine_count} medicine(s) identified")
        
        elif category == ImageCategory.MEDICINE_PACKAGE:
            if "manufactured" in text_features.raw_text.lower():
                reasons.append("manufacturing information found")
            if visual_features.aspect_ratio > 1.2:
                reasons.append("rectangular package layout")
        
        elif category == ImageCategory.LAB_RESULTS:
            if text_features.has_lab_format:
                reasons.append("lab results format detected")
            if visual_features.has_table_layout:
                reasons.append("tabular data structure")
        
        elif category == ImageCategory.ANATOMY_DIAGRAM:
            if visual_features.is_diagram:
                reasons.append("diagram-like visual structure")
            if "anatomy" in text_features.raw_text.lower():
                reasons.append("anatomy-related text")
        
        if not reasons:
            reasons.append("general image classification")
        
        confidence_desc = "high" if confidence > 0.7 else "medium" if confidence > 0.4 else "low"
        
        return f"Classified as {category.value} with {confidence_desc} confidence: {', '.join(reasons)}"