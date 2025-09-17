

from __future__ import annotations

import os
import io
import re
import base64
import logging
import json
from typing import List, Dict, Any, Optional, Tuple

from PIL import Image, UnidentifiedImageError
import ollama
try:
    from langsmith import traceable  # optional, used for tracing if available
except Exception:
    def traceable(*args, **kwargs):  # type: ignore
        def _decorator(f):
            return f
        return _decorator

from utils.web_search import WebSearchTool
from utils.ocr_pipeline import OCRPipeline
from config.env_config import (
    OCR_LANGUAGE,
    OLLAMA_BASE_URL,
    OLLAMA_TEXT_MODEL,
    OLLAMA_VISION_MODEL,
    PINECONE_MEDICINE_INDEX,
    PINECONE_IMAGE_INDEX,
    PINECONE_BD_PHARMACY_INDEX,
)
from .vector_store import PineconeStore
from core.image_index import CLIPImageIndex

from utils.clinical_facts import OTC_DOSING, find_otc_targets_in_text

logger = logging.getLogger(__name__)

# ------------------------------ System prompts --------------------------- #

DEFAULT_SYSTEM = (
    "You are a medical assistant providing helpful, evidence-based guidance.\n"
    "RESPONSE GUIDELINES:\n"
    "- Be conversational and empathetic for greetings and general questions\n"
    "- For medical queries: provide structured, clear information\n"
    "- Use local Bangladesh drug context when available in the knowledge base\n"
    "- Format medical responses with clear sections and bullet points\n"
    "- Always recommend clinical consultation for serious symptoms\n"
    "- Keep responses concise but comprehensive (200-300 words for medical topics)\n"
    "- For greetings: be friendly and mention your capabilities briefly"
)

SYSTEM_GENERAL = (
    "You are a helpful assistant. Respond in these formats:\n"
    "- Definitions: term, 1-sentence description, 2-3 key points\n"
    "- Processes: step-by-step numbered lists (3-5 steps)\n"
    "- Comparisons: concise tables or bullet lists\n"
    "Use provided context verbatim when available."
)

STRICT_MEDICAL_SYSTEM = (
    "You are a clinical expert providing evidence-based guidance.\n"
    "- Structure: 1) Assessment 2) Key considerations 3) Management 4) Red flags\n"
    "- Use generic names first, then Bangladesh brand examples when relevant\n"
    "- Avoid inventing drugs, doses, or brand names\n"
    "- Include brief safety caveats\n"
    "- Format with bullet points and clear headings"
)

# ------------------------------- helpers ---------------------------------- #

def _downscale_jpeg(data: bytes, max_side: int = 1024) -> bytes:
    """Downscale/normalize an image quickly to keep vision prompt light."""
    im = Image.open(io.BytesIO(data)).convert("RGB")
    w, h = im.size
    scale = max(w, h) / float(max_side)
    if scale > 1.0:
        im = im.resize((int(w / scale), int(h / scale)), Image.LANCZOS)
    out = io.BytesIO()
    im.save(out, format="JPEG", quality=85, optimize=True)
    return out.getvalue()

def _clean_conv_ctx(s: str, limit: int = 800) -> str:
    if not s: return ""
    s = re.sub(r"(?i)^\s*you are .*?$", "", s.strip())
    s = re.sub(r"(?i)\b(system|instruction|prompt)\b.*", "", s)
    return s[:limit]

def _dedupe_paragraphs(text: str) -> str:
    if not text: return text
    seen, out = set(), []
    for block in [b.strip() for b in text.split("\n\n")]:
        if block and block not in seen:
            seen.add(block)
            out.append(block)
    return "\n\n".join(out)

def _short_sources(sources: List[str]) -> str:
    if not sources: return ""
    uniq = list(dict.fromkeys([s for s in sources if s]))
    return "Sources: " + ", ".join(uniq)

# ------------------------------- LLMHandler ------------------------------- #

class LLMHandler:
    """Handles both text and vision LLMs using Ollama."""

    def __init__(self) -> None:
        # Model endpoints
        self.base_url = OLLAMA_BASE_URL
        self.text_model = OLLAMA_TEXT_MODEL
        self.vision_model = OLLAMA_VISION_MODEL

        self.bd_pharmacy_store = PineconeStore(index_name=PINECONE_BD_PHARMACY_INDEX, dimension=384)
        self.image_index = CLIPImageIndex(index_name=PINECONE_IMAGE_INDEX)

        # Tools
        self.web = WebSearchTool()

        # Single Ollama client for both text and vision
        self.client = ollama.Client(host=self.base_url)
        self.fast_text_model = os.getenv("OLLAMA_FAST_TEXT_MODEL", self.text_model)

        # Optional RAG stores (PDF store removed; dynamic per-index uploads handled elsewhere)
        self.pdf_store: Optional[PineconeStore] = None
        self.medicine_store: Optional[PineconeStore] = None
        try:
            self.medicine_store = PineconeStore(index_name=PINECONE_MEDICINE_INDEX, dimension=384)
        except Exception as e:
            logger.warning(f"Medicine RAG store init failed: {e}")

    # --------------------------- Core text/vision --------------------------- #
    @traceable(name="text_completion")
    def generate_text_response(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        try:
            messages = [
                {"role": "system", "content": system_prompt or DEFAULT_SYSTEM},
                {"role": "user", "content": prompt},
            ]
            r = self.client.chat(
                model=self.text_model,
                messages=messages,
                stream=False,
                options={"temperature": 0.2, "num_ctx": 4096, "num_predict": 250, "top_p": 0.9},
                keep_alive="30m",
            )
            return (r.get("message", {}) or {}).get("content", "")
        except Exception as e:
            logger.error(f"Error in text generation: {e}")
            return f"Error generating response: {str(e)}"

    @traceable(name="vision_completion")
    def generate_vision_response(
        self, prompt: str, image_path: Optional[str] = None, image_data: Optional[bytes] = None
    ) -> str:
        """Enhanced vision response with robust error handling and retry mechanisms"""
        
        # Prepare image data
        try:
            if image_data:
                target_image_data = image_data
            elif image_path:
                with open(image_path, "rb") as f:
                    target_image_data = f.read()
            else:
                return "No image provided."
        except Exception as e:
            logger.error(f"Error reading image: {e}")
            return "Unable to read the provided image file."
        
        # Multiple retry attempts with different strategies
        retry_strategies = [
            {"name": "optimized", "optimize": True, "num_ctx": 512, "num_predict": 100},
            {"name": "standard", "optimize": False, "num_ctx": 1024, "num_predict": 150},
            {"name": "minimal", "optimize": True, "num_ctx": 256, "num_predict": 80}
        ]
        
        last_error = None
        
        for i, strategy in enumerate(retry_strategies):
            try:
                logger.info(f"Vision attempt {i+1}/3 using {strategy['name']} strategy")
                
                # Prepare image with or without optimization
                if strategy["optimize"]:
                    try:
                        optimized_data = self._optimize_image_for_vision(target_image_data)
                        b64 = base64.b64encode(optimized_data).decode("utf-8")
                    except Exception:
                        b64 = base64.b64encode(target_image_data).decode("utf-8")
                else:
                    b64 = base64.b64encode(target_image_data).decode("utf-8")
                
                # Adjust prompt based on attempt
                adjusted_prompt = prompt
                if i > 0:
                    adjusted_prompt = f"[Simplified analysis] {prompt}"
                
                # Call vision model with strategy-specific parameters
                r = self.client.chat(
                    model=self.vision_model,
                    messages=[{"role": "user", "content": adjusted_prompt, "images": [b64]}],
                    stream=False,
                    options={
                        "temperature": 0.1,
                        "num_ctx": strategy["num_ctx"],
                        "num_predict": strategy["num_predict"],
                        "top_p": 0.9
                    },
                    keep_alive="5m" if i > 0 else "10m",  # Shorter keep-alive for retries
                )
                
                content = r.get("message", {}).get("content", "").strip()
                
                if content and len(content) > 10:  # Minimum viable response
                    logger.info(f"Vision model succeeded with {strategy['name']} strategy")
                    return content
                else:
                    logger.warning(f"Vision model returned empty/minimal response with {strategy['name']} strategy")
                    if i == len(retry_strategies) - 1:  # Last attempt
                        return self._enhanced_fallback_analysis(target_image_data, prompt)
                        
            except Exception as e:
                last_error = e
                logger.warning(f"Vision attempt {i+1} failed with {strategy['name']} strategy: {e}")
                
                # Check for specific error types
                if "status code: 500" in str(e):
                    logger.error("Vision model server error - may be overloaded")
                elif "timeout" in str(e).lower():
                    logger.error("Vision model timeout - reducing complexity for next attempt")
                elif "memory" in str(e).lower() or "resource" in str(e).lower():
                    logger.error("Vision model resource limitations detected")
                
                # Continue to next strategy unless it's the last attempt
                if i < len(retry_strategies) - 1:
                    continue
        
        # All vision attempts failed - use enhanced fallback
        logger.error(f"All vision attempts failed. Last error: {last_error}")
        return self._enhanced_fallback_analysis(target_image_data, prompt)

    def _optimize_image_for_vision(self, image_data: bytes, max_size: int = 800) -> bytes:
        """Optimize image for vision model processing"""
        try:
            from PIL import Image
            import io
            
            # Open image and convert to RGB
            img = Image.open(io.BytesIO(image_data))
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Calculate new size maintaining aspect ratio
            width, height = img.size
            max_dimension = max(width, height)
            
            if max_dimension > max_size:
                ratio = max_size / max_dimension
                new_width = int(width * ratio)
                new_height = int(height * ratio)
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Save optimized image
            output = io.BytesIO()
            img.save(output, format='JPEG', quality=85, optimize=True)
            return output.getvalue()
            
        except Exception as e:
            logger.warning(f"Image optimization failed: {e}")
            return image_data  # Return original if optimization fails

    def _enhanced_fallback_analysis(self, image_data: bytes, question: str = "") -> str:
        """Enhanced fallback analysis with multiple strategies"""
        strategies = [
            ("ocr_advanced", self._try_advanced_ocr),
            ("ocr_basic", self._try_basic_ocr),
            ("pattern_recognition", self._try_pattern_recognition)
        ]
        
        for strategy_name, strategy_func in strategies:
            try:
                logger.info(f"Trying fallback strategy: {strategy_name}")
                result = strategy_func(image_data, question)
                if result and len(result.strip()) > 20:  # Minimum useful response
                    logger.info(f"Fallback strategy {strategy_name} succeeded")
                    return result
            except Exception as e:
                logger.warning(f"Fallback strategy {strategy_name} failed: {e}")
                continue
        
        # Ultimate fallback
        return ("I'm having technical difficulties analyzing this image. "
                "The vision model is temporarily unavailable, and OCR extraction also failed. "
                "Please try again later or contact support if the issue persists.")

    def _try_advanced_ocr(self, image_data: bytes, question: str = "") -> str:
        """Advanced OCR with preprocessing"""
        try:
            # Try OCR with image preprocessing
            ocr_result = self.ocr_only(image_data)
            text = ocr_result.get("raw_text", "").strip()
            
            if text and len(text) > 10:
                if question and question.strip():
                    # Try to answer question based on OCR text
                    answer = self.answer_over_ocr_text(question, text)
                    if answer and "can't find it" not in answer.lower():
                        return f"**Based on text extraction:** {answer}"
                    else:
                        return f"**Text found in image:**\n{text[:500]}{'...' if len(text) > 500 else ''}\n\n*Note: I couldn't find a specific answer to your question in this text.*"
                else:
                    # Determine if it looks like a prescription
                    if any(term in text.lower() for term in ["rx", "doctor", "patient", "medicine", "tablet", "mg", "dose"]):
                        return f"**Medical document detected:**\n{text[:600]}{'...' if len(text) > 600 else ''}"
                    else:
                        return f"**Text extracted from image:**\n{text[:500]}{'...' if len(text) > 500 else ''}"
            else:
                return ""  # No useful text found
                
        except Exception as e:
            logger.error(f"Advanced OCR failed: {e}")
            return ""

    def _try_basic_ocr(self, image_data: bytes, question: str = "") -> str:
        """Basic OCR fallback"""
        try:
            ocr_result = self.ocr_only(image_data)
            text = ocr_result.get("raw_text", "").strip()
            
            if text:
                return f"**Text found in image:** {text[:300]}{'...' if len(text) > 300 else ''}"
            else:
                return ""
        except Exception:
            return ""

    def _try_pattern_recognition(self, image_data: bytes, question: str = "") -> str:
        """Basic pattern recognition for common image types"""
        try:
            from PIL import Image
            import io
            
            img = Image.open(io.BytesIO(image_data))
            width, height = img.size
            
            # Basic image analysis
            if width > height * 1.3:
                orientation = "landscape"
            elif height > width * 1.3:
                orientation = "portrait"
            else:
                orientation = "square"
            
            # Try to determine image type based on characteristics
            if question:
                if "prescription" in question.lower():
                    return f"This appears to be a {orientation} medical document. The vision model is unavailable, but I can see it's likely a prescription or medical form."
                elif "medicine" in question.lower() or "drug" in question.lower():
                    return f"This appears to be a {orientation} image that may contain medicine or drug information. Unfortunately, detailed analysis requires the vision model which is currently unavailable."
            
            return f"I can see this is a {orientation} image ({width}x{height} pixels), but detailed analysis requires the vision model which is currently unavailable."
            
        except Exception:
            return ""

    def _fallback_image_analysis(self, image_data: bytes, question: str = "") -> str:
        """Legacy fallback method - redirects to enhanced version"""
        return self._enhanced_fallback_analysis(image_data, question)

    # ------------------------------- OCR utils ------------------------------ #

    def _fast_format(self, prompt: str) -> str:
        try:
            messages = [
                {"role": "system", "content": DEFAULT_SYSTEM},
                {"role": "user", "content": prompt},
            ]
            r = self.client.chat(
                model=self.fast_text_model,
                messages=messages,
                stream=False,
                options={"temperature": 0.1, "num_ctx": 1024, "num_predict": 200, "top_p": 0.9},
                keep_alive="30m",
            )
            return (r.get("message", {}) or {}).get("content", "")
        except Exception:
            return self.generate_text_response(prompt, system_prompt=DEFAULT_SYSTEM)

    def ocr_only(self, image_bytes: bytes) -> Dict[str, Any]:
        """Very fast OCR-only parse used when no complex vision is needed."""
        try:
            # downscale attempts to open; catch invalid image payloads
            data = _downscale_jpeg(image_bytes)
        except UnidentifiedImageError:
            return {"raw_text": "", "lines": [], "structured_items": [], "item_count": 0, "header": {}, "error": "not_an_image"}

        pipe = OCRPipeline(lang=OCR_LANGUAGE or "en")
        lines, items, header = pipe.run_on_bytes(data)  # (lines, items, header)
        return {
            "raw_text": "\n".join(lines),
            "lines": lines,
            "structured_items": items,
            "item_count": len(items),
            "header": header or {},
        }

    def answer_over_ocr_text(self, question: str, ocr_text: str, conversation_context: str = "") -> str:
        """QA over OCR text only; short and honest if missing."""
        if "doctor" in question.lower() or re.search(r"\bdr\b", question.lower()):
            m = re.search(r"(Dr\.?\s*[A-Z][A-Za-z.\s'-]{1,40})", ocr_text)
            if m: return m.group(1).strip()

        prompt = (
            f"{_clean_conv_ctx(conversation_context)}\n\n"
            "Answer the question using ONLY the OCR text below. "
            "If the answer is not present, reply exactly: \"Can't find it in the OCR.\" "
            "Keep it brief.\n\nOCR TEXT:\n"
            f"{ocr_text[:8000]}\n\nQUESTION: {question}\nANSWER:"
        )
        return self.generate_text_response(prompt, system_prompt=SYSTEM_GENERAL).strip()

    # ------------------------------- RAG ----------------------------------- #

    def _select_namespaces(self, q: str) -> List[str]:
        """Select appropriate namespaces based on query content"""
        ql = q.lower().strip()
        
        # Policy/Legal content
        if any(k in ql for k in ["law", "policy", "act", "regulation", "dgda", "directorate general of drug administration", "drug law", "drug rules", "legal", "compliance"]):
            return ["policy"]
        
        # Anatomy/Educational content
        if any(k in ql for k in ["anatomy", "nerve", "artery", "vein", "muscle", "bone", "cell", "organelle", "nucleus",
                                 "mitochondria", "golgi", "ribosome", "histology", "cytology", "epithelium", "tissue", "organelles", "physiology"]):
            return ["textbook"]
        
        # OTC/Self-care content
        if any(k in ql for k in ["otc", "self care", "over the counter", "without prescription", "pharmacy", "drugstore"]):
            return ["otc"]
        
        # Clinical prescribing content - for drug queries
        if any(k in ql for k in ["dose", "dosing", "indication", "contraindication", "interaction", "pregnancy", 
                                 "side effect", "adverse", "clinical", "prescribe", "treatment", "therapy"]):
            return ["prescribing"]
        
        # Bangladesh-specific queries
        if any(k in ql for k in ["bangladesh", " bd ", "bd:", "bd,", "bangla"]):
            return ["prescribing"]
        
        # Medicine/Drug queries - use prescribing as primary
        if any(k in ql for k in ["medicine", "drug", "medication", "tablet", "capsule", "syrup", "injection", 
                                 "antibiotic", "painkiller", "mg", "dosage"]):
            return ["prescribing"]
        
        # Default: guidelines for general medical queries
        return ["guidelines"]

    def _gather_rag_context(self, query: str, extra_ctx: Optional[List[str]] = None) -> List[str]:
        """Enhanced RAG context gathering with better relevance scoring and deduplication"""
        ctx: List[str] = []
        if extra_ctx: 
            ctx.extend([c for c in extra_ctx if isinstance(c, str) and c.strip()])

        max_chunks = 3          # Reduced further for focus
        target_hits = 2         # Reduced target for better quality
        per_ns_top_k = 2        # Maintain 2 per namespace
        
        # Track query terms for relevance scoring
        query_terms = set(query.lower().split())

        if self.bd_pharmacy_store:
            # Get the most relevant namespace (single best match)
            namespaces = self._select_namespaces(query)[:1]  # Use only the most relevant namespace
            
            for ns in namespaces:
                try:
                    hits = self.bd_pharmacy_store.query(query, top_k=per_ns_top_k, namespace=ns)
                    if hits:
                        # Score and filter hits for relevance
                        scored_hits = []
                        for hit in hits:
                            if hit and len(hit.strip()) > 50:  # Minimum content threshold
                                score = self._calculate_relevance_score(hit, query_terms)
                                if score > 0.1:  # Relevance threshold
                                    scored_hits.append((score, hit))
                        
                        # Sort by relevance and take top results
                        scored_hits.sort(reverse=True, key=lambda x: x[0])
                        ctx.extend([hit for _, hit in scored_hits[:per_ns_top_k]])
                        
                    if len(ctx) >= target_hits:  # early stop once we have enough signal
                        break
                except Exception as e:
                    logger.warning(f"BD pharmacy query failed ({ns}): {e}")

        # Only query medicine store if we still need more context and query is medicine-related
        try:
            if (self.medicine_store and len(ctx) < target_hits and 
                any(term in query.lower() for term in ["medicine", "drug", "medication", "tablet", "capsule"])):
                hits = self.medicine_store.query(query, top_k=1)
                if hits:
                    for hit in hits:
                        if hit and len(hit.strip()) > 50:
                            score = self._calculate_relevance_score(hit, query_terms)
                            if score > 0.15:  # Higher threshold for medicine store
                                ctx.append(hit)
                                break
        except Exception as e:
            logger.warning(f"Medicine RAG query failed: {e}")

        # Enhanced deduplication with semantic similarity
        uniq = self._deduplicate_context(ctx, max_chunks)
        return uniq
    
    def _calculate_relevance_score(self, text: str, query_terms: set) -> float:
        """Calculate relevance score for a text chunk based on query terms"""
        if not text or not query_terms:
            return 0.0
        
        text_lower = text.lower()
        text_words = set(text_lower.split())
        
        # Term frequency score
        term_matches = len(query_terms.intersection(text_words))
        term_score = term_matches / len(query_terms) if query_terms else 0
        
        # Boost score for medical terms
        medical_terms = {"dose", "dosage", "indication", "side", "effect", "contraindication", 
                        "interaction", "pregnancy", "safety", "clinical", "treatment"}
        medical_matches = len(medical_terms.intersection(text_words))
        medical_boost = min(medical_matches * 0.1, 0.3)
        
        # Penalize very short or very long chunks
        length = len(text)
        if length < 100:
            length_penalty = 0.5
        elif length > 1500:
            length_penalty = 0.7
        else:
            length_penalty = 1.0
            
        return (term_score + medical_boost) * length_penalty
    
    def _deduplicate_context(self, contexts: List[str], max_chunks: int) -> List[str]:
        """Enhanced deduplication with semantic similarity detection"""
        if not contexts:
            return []
        
        uniq = []
        seen_phrases = set()
        
        for ctx in contexts:
            if not ctx or not ctx.strip():
                continue
                
            # Normalize context
            ctx_clean = ctx.strip()[:600]  # Reduced chunk size for focus
            
            # Extract key phrases for similarity detection
            key_phrases = self._extract_key_phrases(ctx_clean)
            
            # Check for significant overlap with existing contexts
            overlap_score = 0
            for existing_phrases in seen_phrases:
                overlap = len(key_phrases.intersection(existing_phrases))
                overlap_score = max(overlap_score, overlap / max(len(key_phrases), 1))
            
            # Only add if not too similar (< 70% overlap)
            if overlap_score < 0.7:
                uniq.append(ctx_clean)
                seen_phrases.add(key_phrases)
                
                if len(uniq) >= max_chunks:
                    break
        
        return uniq
    
    def _extract_key_phrases(self, text: str) -> set:
        """Extract key phrases from text for similarity comparison"""
        import re
        
        # Remove common stop words and extract meaningful phrases
        text_lower = text.lower()
        
        # Extract medical terms, drug names, and key phrases
        phrases = set()
        
        # Medical terms
        medical_pattern = r'\b(?:dose|dosage|indication|contraindication|side effect|interaction|pregnancy|safety|clinical|treatment|therapy|medication|drug|medicine)\b'
        phrases.update(re.findall(medical_pattern, text_lower))
        
        # Drug names and quantities
        drug_pattern = r'\b[a-z]+(?:ol|in|ine|ide|ate|azole|mycin|cillin|prazole)\b'
        phrases.update(re.findall(drug_pattern, text_lower))
        
        # Quantities and measurements
        quantity_pattern = r'\b\d+\s*(?:mg|g|ml|mcg|units?|tablets?|capsules?)\b'
        phrases.update(re.findall(quantity_pattern, text_lower))
        
        return phrases

    def _prompt_with_context(self, query: str, context: List[str], conversation_context: str = "") -> str:
        ctx_block = "\n\n---\n".join(context) if context else "No relevant context."
        conv = _clean_conv_ctx(conversation_context)
        parts = []
        if conv: parts.append(f"{conv}\n\n")
        parts.append("Use the CONTEXT to answer the user's medical question. "
                     "Prefer Bangladesh drug/brand specifics when relevant. "
                     "If context is insufficient, say so briefly and give safe general guidance.\n\n")
        parts.append(f"CONTEXT START\n---\n{ctx_block}\n---\nCONTEXT END\n\n")
        parts.append(f"QUESTION: {query}")
        return "".join(parts)

    # ----------------------- High-level convenience APIs ------------------- #

    @staticmethod
    def _safety_footer(text: str) -> str:
        red = ("chest pain" in text.lower() or "stroke" in text.lower() or "shortness of breath" in text.lower())
        if red:
            return text + "\n\n**If this is about a real person with urgent symptoms, seek emergency care immediately.**"
        return text

    # ----------------------- General knowledge answers -------------------- #

    def answer_general_knowledge(self, query: str, context: Optional[List[str]] = None, conversation_context: str = "") -> str:
        """
        Friendly encyclopedic answer for greetings/definitions/anatomy topics.
        Keeps responses concise, avoids clinical dosing unless explicitly asked.
        """
        ctx_block = "\n---\n".join(context or []) if context else ""
        prompt = (
            "Answer the user's question clearly and directly. "
            "If this is a definition or anatomy topic, include: what it is, key functions, and 2-3 high-yield facts. "
            "If the question is broad (e.g., 'human anatomy'), provide a brief overview and end with three focused options the user can pick to go deeper. "
            "Avoid drug dosing or brand names unless explicitly requested.\n\n"
        )
        if ctx_block:
            prompt += f"CONTEXT\n---\n{ctx_block}\n---\n"
        prompt += f"QUESTION: {query}\n\nAnswer:"
        return self.generate_text_response(prompt, system_prompt=SYSTEM_GENERAL)

    def greeting_response(self, text: str = "") -> str:
        # Simple, contextual greeting without overwhelming information
        greetings = text.lower().strip() if text else ""
        
        if "good morning" in greetings:
            start = "Good morning!"
        elif "good evening" in greetings:
            start = "Good evening!"
        elif "good afternoon" in greetings:
            start = "Good afternoon!"
        else:
            start = "Hello!"
            
        return (
            f"{start} I'm AroBot, your medical assistant. I can help with medical questions, "
            "medicine information, prescription analysis, and document review. "
            "What would you like to know about today?"
        )

    def about_response(self) -> str:
        """Short capabilities + safety statement for meta questions about the assistant."""
        return (
            "I'm AroBot—an AI assistant for concise, evidence‑minded medical guidance. "
            "I can: answer symptom/medicine questions, read prescription images (OCR), analyze PDFs, look up Bangladesh brands/prices, and do quick web-supported summaries. "
            "I’m not a doctor; for real medical decisions, consult a licensed clinician—especially for urgent symptoms."
        )

    @traceable(name="medical_query")
    def answer_medical_query(self, query: str, context: Optional[List[str]] = None, conversation_context: str = "") -> str:
        # CRITICAL FIX: Use provided context if available to avoid double RAG
        if context and len(context) > 0:
            # Context already provided by RAGAgent - use it directly
            rag_ctx = context[:6]  # Limit to prevent context explosion
        else:
            # Only do RAG if no context provided
            rag_ctx = self._gather_rag_context(query, extra_ctx=None)
        
        ctx_block = "\n---\n".join(rag_ctx) if rag_ctx else "No relevant context"

        # Unified scaffold with adaptive brevity based on query length
        is_short = len(query.split()) <= 4 and any(word in query.lower() for word in ["what", "does", "do", "use", "for", "how"]) 
        scaffold_prompt = (
            "Using ONLY the CONTEXT (and general medical knowledge if needed), answer the question.\n"
            + ("Return a brief direct answer (1-2 sentences).\n\n" if is_short else
               "Structure your response with these sections when appropriate:\n"
               "1) Assessment / Most likely\n2) Key considerations\n3) Management approach\n4) Red flags (urgent/ER)\n\n")
            + "Be specific, avoid unnecessary drugs, keep concise. If data is sparse, say so briefly.\n\n"
            + f"CONTEXT START\n---\n{ctx_block}\n---\nCONTEXT END\n\n"
            + f"QUESTION: {query}\n\nAnswer:"
        )
        out = self.generate_text_response(scaffold_prompt, system_prompt=STRICT_MEDICAL_SYSTEM)
        try: out = _dedupe_paragraphs(out)
        except Exception: pass
        return self._safety_footer(out)

    def answer_over_pdf_text(self, question_or_none: Optional[str], pdf_text: str, conversation_context: str = "") -> str:
        if not pdf_text or not pdf_text.strip():
            return "The PDF appears to contain no readable text."
        if not question_or_none:
            prompt = (
                "Summarize the following PDF text for a clinician-friendly audience. "
                "Include section highlights and 3–5 key takeaways.\n\n"
                f"{pdf_text[:15000]}"
            )
            return self.generate_text_response(prompt, system_prompt=DEFAULT_SYSTEM)
        return self.answer_medical_query(question_or_none, context=[pdf_text], conversation_context=conversation_context)

    # ------------------------ Prescription (OCR-first) --------------------- #

    @staticmethod
    def _ocr_quality_ok(t: str) -> bool:
        """Tiny heuristic: length + proportion of readable chars."""
        if not t:
            return False
        s = t.strip()
        if len(s) < 80:
            return False
        alpha = sum(1 for c in s if c.isalnum() or c.isspace() or c in ".-:,/()|+")
        return (alpha / max(len(s), 1)) >= 0.55

    @staticmethod
    def _extract_json(s: str) -> Optional[Dict[str, Any]]:
        """Extract first JSON block from text; tolerate ```json fences or plain JSON."""
        if not s:
            return None
        m = re.search(r"```json\s*(\{.*?\})\s*```", s, flags=re.S | re.I)
        if not m:
            m = re.search(r"(\{[\s\S]*\})", s)
        if not m:
            return None
        try:
            return json.loads(m.group(1))
        except Exception:
            try:
                fixed = re.sub(r",\s*}", "}", m.group(1))
                fixed = re.sub(r",\s*]", "]", fixed)
                return json.loads(fixed)
            except Exception:
                return None

    # --------------------- Post-processing safeguards -------------------- #
    @staticmethod
    def _norm_text(s: str) -> str:
        try:
            s = (s or "").lower()
            s = re.sub(r"[^a-z0-9\s.+/-]", " ", s)
            s = " ".join(s.split())
            return s
        except Exception:
            return s or ""

    @staticmethod
    def _best_line_score(lines: List[str], phrase: str) -> float:
        from difflib import SequenceMatcher
        p = LLMHandler._norm_text(phrase)
        if not p:
            return 0.0
        scores: List[float] = []
        for ln in lines:
            lnn = LLMHandler._norm_text(ln)
            if not lnn:
                continue
            # quick token presence boost
            tok_hit = 0
            tokens = [t for t in p.split() if len(t) >= 3]
            if tokens:
                tok_hit = sum(1 for t in tokens if t in lnn) / float(len(tokens))
            r = SequenceMatcher(None, lnn, p).ratio()
            scores.append(max(r, tok_hit))
        return max(scores) if scores else 0.0

    @staticmethod
    def _present_in_ocr(ocr_text: str, target: str, threshold: float = 0.72) -> bool:
        lines = [l for l in (ocr_text or "").splitlines() if l.strip()]
        return LLMHandler._best_line_score(lines, target) >= threshold

    @staticmethod
    def _build_meds_from_ocr_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        meds: List[Dict[str, Any]] = []
        for it in items or []:
            if not isinstance(it, dict):
                continue
            raw = (it.get("raw") or "").strip()
            if not raw:
                continue
            meds.append({
                "name": raw,
                "generic_suspected": "",
                "form": "",
                "route": "",
                "dose": (f"{it.get('strength')} {it.get('unit')}".strip() if it.get('strength') else ""),
                "dose_pattern": it.get("dose_pattern") or it.get("frequency") or "",
                "frequency": it.get("frequency") or (it.get("dose_pattern") or ""),
                "duration": it.get("duration") or "",
                "quantity_or_count_mark": "",
                "additional_instructions": "",
                "confidence": 0.6,
            })
        return meds

    @staticmethod
    def _anchor_structured_to_ocr(structured: Dict[str, Any], ocr_blob: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure final fields are supported by OCR evidence. Drops inventions.

        - Overwrites doctor/clinic with header hints if available.
        - Removes medications that don't fuzzy-match any OCR line.
        - If no meds survive and OCR items exist, build meds from OCR items.
        """
        if not structured:
            return structured

        ocr_text = (ocr_blob or {}).get("raw_text", "") or ""
        lines = [l for l in ocr_text.splitlines() if l.strip()]
        header = (ocr_blob or {}).get("header") or {}
        items = (ocr_blob or {}).get("structured_items") or []

        # Prefer header for doctor/clinic when present
        if header.get("doctor"):
            structured["doctor"] = header.get("doctor")
        if header.get("clinic"):
            structured["clinic"] = header.get("clinic")

        # Validate top-level text fields when OCR text exists
        for k in ["patient_name", "diagnosis_or_cc", "date"]:
            v = structured.get(k)
            if v and ocr_text:
                if LLMHandler._best_line_score(lines, str(v)) < 0.55:
                    structured[k] = ""

        # Medication anchoring
        meds_in = structured.get("medications") or []
        meds_out: List[Dict[str, Any]] = []
        for med in meds_in:
            if not isinstance(med, dict):
                continue
            name = med.get("name") or ""
            conf = 0.0
            if name:
                conf = LLMHandler._best_line_score(lines, name)
            # Also try dose string as evidence
            if conf < 0.72:
                dose_str = " ".join([str(med.get("dose") or ""), str(med.get("dose_pattern") or "")]).strip()
                if dose_str:
                    conf = max(conf, LLMHandler._best_line_score(lines, dose_str))
            if conf >= 0.72:
                m2 = dict(med)
                m2["confidence"] = max(float(med.get("confidence") or 0.0), round(conf, 2))
                meds_out.append(m2)

        if not meds_out and items:
            meds_out = LLMHandler._build_meds_from_ocr_items(items)

        structured["medications"] = meds_out
        # Validate medications have reasonable confidence
        valid_meds = [m for m in meds_out if m.get("confidence", 0) >= 0.5]
        structured["medications"] = valid_meds

        return structured

    @staticmethod
    def _summarize_structured(obj: Dict[str, Any]) -> str:
        if not obj:
            return "Couldn't confidently read the prescription."
        parts = []
        # Patient and header first for clarity
        if obj.get("patient_name") or obj.get("age_sex"):
            parts.append("Patient: " + " ".join([s for s in [obj.get("patient_name"), obj.get("age_sex")] if s]).strip())
        if obj.get("doctor"):
            parts.append(f"Doctor: {obj['doctor']}")
        if obj.get("clinic"):
            parts.append(f"Clinic: {obj['clinic']}")
        if obj.get("date"):
            parts.append(f"Date: {obj['date']}")
        if obj.get("diagnosis_or_cc"):
            parts.append(f"Dx/CC: {obj['diagnosis_or_cc']}")

        meds = obj.get("medications") or []
        if meds:
            lines: List[str] = []
            idx = 1
            for m in meds:
                name = (m.get("name") or "").strip()
                # Filter obvious OCR noise like single symbols or 1-2 letters
                import re as _re
                if len(_re.sub(r"[^A-Za-z]", "", name)) < 3:
                    continue
                dose = m.get("dose") or m.get("dose_pattern") or ""
                extra = m.get("additional_instructions") or ""
                frag = name
                if dose:
                    frag += f" - {dose}"
                if extra:
                    frag += f" ({extra})"
                lines.append(f"{idx}) {frag}")
                idx += 1
            if lines:
                parts.append("Meds:\n" + "\n".join(lines))
        if obj.get("notes"):
            parts.append(f"Notes: {obj['notes']}")
        return "\n".join(parts) if parts else "Parsed the prescription."

    # inside LLMHandler

    @traceable(name="prescription_analysis")
    def analyze_prescription(
        self,
        image_path: Optional[str] = None,
        image_data: Optional[bytes] = None,
        ocr_text: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Robust prescription extractor:
        - OCR-first; if weak, schema-locked vision pass.
        - If JSON fails, still return a fallback human-readable summary.
        """
        try:
            # ---- 1) OCR ----
            ocr_blob: Dict[str, Any] = {}
            if not ocr_text:
                if image_data:
                    ox = self.ocr_only(image_data)
                elif image_path:
                    with open(image_path, "rb") as f:
                        ox = self.ocr_only(f.read())
                else:
                    return {"error": "No image provided", "status": "error"}

                if isinstance(ox, dict) and ox.get("error") == "not_an_image":
                    return {"error": "Provided bytes are not an image", "status": "error"}

                ocr_text = (ox or {}).get("raw_text", "") if isinstance(ox, dict) else ""
                ocr_blob = ox or {}
            else:
                ocr_blob = {}

            header = (ocr_blob.get("header") or {}) if isinstance(ocr_blob, dict) else {}
            doc_hint = []
            if header.get("doctor"): doc_hint.append(f"doctor={header.get('doctor')}")
            if header.get("clinic"): doc_hint.append(f"clinic={header.get('clinic')}")
            hint_str = f"HEADER HINTS: {', '.join(doc_hint)}" if doc_hint else ""

            # ---- 2) Schema prompt ----
            schema_prompt_prefix = (
                "Extract prescription data as JSON with fields:\n"
                "- patient_name, age_sex, doctor, clinic, date, diagnosis_or_cc, notes\n"
                "- medications[]: name, generic_suspected, form, route, dose, dose_pattern, frequency, duration, quantity_or_count_mark, additional_instructions, confidence\n\n"
                "RULES:\n"
                "- Output JSON ONLY (no prose)\n"
                "- Extract ONLY clearly legible text; leave fields empty if unclear\n"
                "- NEVER infer default/common drugs or diagnoses\n"
                "- confidence must be 0..1 reflecting clarity\n"
            )

            # ---- 3) OCR → JSON ----
            structured = None
            used_mode = "ocr_first"
            if self._ocr_quality_ok(ocr_text or ""):
                prompt = (
                    schema_prompt_prefix
                    + (f"\n{hint_str}\n" if hint_str else "\n")
                    + "OCR TEXT START:\n"
                    + (ocr_text or "")[:6000]
                    + "\nOCR TEXT END"
                )
                normalized_text = self._fast_format(prompt)
                structured = self._extract_json(normalized_text)

            # ---- 4) Fallback vision → JSON ----
            if structured is None:
                used_mode = "vision_due_to_weak_ocr"
                vision_prompt = (
                    schema_prompt_prefix
                    + (f"\n{hint_str}\n" if hint_str else "\n")
                    + "Read directly from the image. Output JSON only."
                )
                vision_out = self.generate_vision_response(
                    prompt=vision_prompt, image_path=image_path, image_data=image_data
                )
                structured = self._extract_json(vision_out)

            # ---- 5) Rescue fallback ----
            if structured is None:
                # Vision description fallback
                desc = self.generate_vision_response(
                    prompt="Read this medical note. Summarize what you see: patient, doctor, diagnosis, medicines (names only).",
                    image_path=image_path,
                    image_data=image_data
                )
                msg = desc or "Could not reliably parse the prescription."
                return {
                    "mode": used_mode,
                    "ocr_results": {"raw_text": ocr_text or ""},
                    "prescription_analysis": msg,
                    "summary": msg,
                    "status": "success",
                }

            # ---- 6) Anchor to OCR & summarize ----
            try:
                structured = self._anchor_structured_to_ocr(structured, ocr_blob)
            except Exception as _:
                pass
            summary = self._summarize_structured(structured)
            return {
                "mode": used_mode,
                "ocr_results": {"raw_text": ocr_text or "", **({"header": (ocr_blob or {}).get("header")} if isinstance(ocr_blob, dict) else {})},
                "structured": structured,
                "prescription_analysis": json.dumps(structured, ensure_ascii=False, indent=2),
                "summary": summary,
                "status": "success",
            }

        except Exception as e:
            logger.error(f"Rx analyze error: {e}")
            return {"error": str(e), "status": "error"}

    # --------------------------- Medicines (BD) ---------------------------- #

    @traceable(name="answer_medicine_bd")
    def answer_medicine(self, query: str, want_price: bool = False) -> str:
        rec = self.web.resolve_bd_medicine(query)
        brand_line = ""
        if rec.get("status") == "success":
            brand_line = (
                f"**Brand (BD):** {rec.get('brand','')}"
                f"{'  |  Generic: ' + rec.get('generic','') if rec.get('generic') else ''}"
                f"{'  |  ' + rec.get('form','') if rec.get('form') else ''}"
                f"{'  ' + rec.get('strength','') if rec.get('strength') else ''}"
                f"{'  |  ' + rec.get('company','') if rec.get('company') else ''}"
                f"\nSource: {rec.get('source','medex.com.bd')}"
            )

        rag_terms = [query]
        if rec.get("status") == "success":
            if rec.get("generic"): rag_terms.append(rec["generic"])
            if rec.get("brand"):   rag_terms.append(rec["brand"])

        ctx = self._gather_rag_context(" ; ".join(rag_terms))
        prompt = self._prompt_with_context(
            "Summarize indication/uses, adult dosing ranges, key cautions (renal/hepatic, pregnancy), "
            "major interactions, and common side effects. Prefer Bangladesh context and plain English.",
            ctx,
        )
        summary = self._fast_format(prompt)
        summary = _dedupe_paragraphs(summary)

        price_line = ""
        if want_price and rec.get("status") == "success":
            p = self.web.get_bd_medicine_price(rec["brand"])
            if p.get("status") == "success" and p.get("price"):
                price_line = f"\n\n**Bangladesh retail price ({p.get('source','medex')}):** {p['price']}"

        card = self.answer_bd_formulary_card(query, want_price=False)

        joined = "\n\n".join([x for x in [brand_line, summary, card, price_line] if x])
        return _dedupe_paragraphs(joined)

    def answer_bd_formulary_card(self, query: str, want_price: bool = False) -> str:
        rec = self.web.resolve_bd_medicine(query)
        brand_hit = rec.get("brand") if rec.get("status") == "success" else None
        generic   = rec.get("generic") if rec.get("status") == "success" else None
        company   = rec.get("company") if rec.get("status") == "success" else None
        form      = rec.get("form") if rec.get("status") == "success" else None
        strength  = rec.get("strength") if rec.get("status") == "success" else None
        provenance = [rec.get("source") or "medex.com.bd"] if rec.get("status") == "success" else []

        rag_terms = [query]
        if generic:   rag_terms.append(generic)
        if brand_hit: rag_terms.append(brand_hit)
        ctx = self._gather_rag_context(" ; ".join(rag_terms))
        ctx_text = "\n---\n".join(ctx) if ctx else "No context available"

        extraction_prompt = (
            "From the CONTEXT, extract Bangladesh formulary facts for the target generic/brand. "
            "Return a concise card with these sections if possible:\n"
            "- Generic & Class\n- Common BD brands (Manufacturer)\n- Common forms/strengths & typical retail pack sizes\n"
            "- Key cautions (renal/hepatic, pregnancy), notable interactions\n"
            "Use bullet points. Keep to ~8 bullets total. If some fields are unknown, omit them. "
            "Don't invent companies or pack sizes.\n\n"
            f"TARGET: {generic or brand_hit or query}\n"
            f"CONTEXT START\n---\n{ctx_text}\n---\nCONTEXT END"
        )

        card = self._fast_format(extraction_prompt)
        card = _dedupe_paragraphs(card)

        price_line = ""
        if want_price and brand_hit:
            p = self.web.get_bd_medicine_price(brand_hit)
            if p.get("status") == "success" and p.get("price"):
                provenance.append(p.get("source") or "medex.com.bd")
                price_line = f"\n\n**Typical retail (BD, {p.get('source','medex')}):** {p['price']}"

        prov = _short_sources(list(dict.fromkeys([s for s in provenance if s])))

        id_line = ""
        if brand_hit or generic:
            id_line = "**Identity (BD):** "
            if brand_hit: id_line += f"{brand_hit}"
            if generic:   id_line += f"  |  Generic: {generic}"
            if form:      id_line += f"  |  {form}"
            if strength:  id_line += f"  {strength}"
            if company:   id_line += f"  |  {company}"
            id_line += f"\nSource: {rec.get('source','medex.com.bd')}\n\n"

        return id_line + card + price_line + (f"\n\n{prov}" if prov else "")

    # --------------------------- Image helpers ----------------------------- #

    def vision_short_answer(self, question: str, image_path: Optional[str] = None, image_data: Optional[bytes] = None) -> str:
        prompt = ("Answer the user's question using only what is visible in the image. "
                  "Return a single short sentence.\n"
                  f"QUESTION: {question}\nANSWER:")
        out = self.generate_vision_response(prompt, image_path=image_path, image_data=image_data)
        return (out or "").strip()

    def default_image_brief(self, image_bytes: bytes) -> str:
        """1) OCR hints 2) One-pass concise vision description."""
        ocr = self.ocr_only(image_bytes)
        raw = ocr.get("raw_text", "") or ""
        prompt = ("Describe the image briefly (1–2 sentences). "
                  "Prefer exact words from the OCR text when relevant.\n\n"
                  f"OCR Hints:\n{raw[:800]}")
        vis = self.generate_vision_response(prompt=prompt, image_data=image_bytes)
        return (vis or "I couldn't extract much from the image.").strip()

    # --------------------------- OTC fast path ----------------------------- #

    def _format_bd_brand_examples(self, generic_name: str) -> str:
        try:
            q = generic_name
            # Use generic name directly for web search
            got = self.web.resolve_bd_medicine(q)
            if got.get("status") != "success": return ""
            brand = got.get("brand") or ""
            generic = got.get("generic") or ""
            form = got.get("form") or ""
            strength = got.get("strength") or ""
            company = got.get("company") or ""
            line = f"- {brand} ({generic}) {strength} {form}".strip()
            if company: line += f" — {company}"
            return line
        except Exception:
            return ""

    def try_fast_otc_dose_answer(self, query: str, conversation_context: str = "") -> Optional[str]:
        targets = find_otc_targets_in_text(query + " " + conversation_context)
        if not targets: return None
        lines = []
        for drug in targets:
            info = OTC_DOSING[drug]
            bex = self._format_bd_brand_examples(drug)
            bex_block = f"\n  • BD brand example:\n    {bex}" if bex else ""
            lines.append(
                f"**{drug.capitalize()} (OTC)**\n"
                f"- Adult dosing: {info['adult_dose']}\n"
                f"- Key cautions: " + "; ".join(info["key_cautions"]) + "\n"
                f"- Pregnancy: {info['pregnancy']}\n"
                f"- Typical retail pack sizes in BD: {info['pack_sizes_bd']}{bex_block}"
            )
        footer = (
            "\n\n**Safety:** Use the lowest effective dose for the shortest time. "
            "Avoid duplicate analgesics, check combination products, adjust for renal/hepatic issues or pregnancy. "
            "Seek clinician advice if symptoms persist >48–72h, worsen, or red flags develop."
        )
        return "\n\n".join(lines) + footer
