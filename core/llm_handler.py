# core/llm_handler.py
"""
LLM Handler for text and vision models using Ollama
- ChatOllama for text (proper system/user messages)
- LLaVA via ollama.chat for vision
- OCR-first utilities for fast image Q&A
- Medicine helper that resolves BD brands + optional prices
- RAG helpers over Pinecone PDF & medicine indexes
"""

from __future__ import annotations

import os
import io
import re
import base64
import logging
from typing import List, Dict, Any, Optional

from PIL import Image
import ollama
from langchain_ollama import ChatOllama
from langchain.schema import HumanMessage, SystemMessage
from langsmith import traceable

from utils.web_search import WebSearchTool
from utils.ocr_pipeline import OCRPipeline
from config.env_config import (
    OCR_LANGUAGE,
    OLLAMA_BASE_URL,
    OLLAMA_TEXT_MODEL,
    OLLAMA_VISION_MODEL,
    PINECONE_PDF_INDEX,
    PINECONE_MEDICINE_INDEX,
    PINECONE_IMAGE_INDEX,
    PINECONE_BD_PHARMACY_INDEX,
)
from .vector_store import PineconeStore
from core.image_index import CLIPImageIndex

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# System prompts
# --------------------------------------------------------------------------- #
DEFAULT_SYSTEM = (
    "You are AroBot, a careful medical assistant. Be concise and clinically safe. "
    "Prefer Bangladesh-specific drug/brand context when ambiguous (e.g., 'Napa' → paracetamol brand). "
    "Cite uncertainty briefly and advise consulting a clinician for decisions."
)

SYSTEM_GENERAL = (
    "You are AroBot, a helpful, concise assistant. Answer clearly and directly. "
    "Use provided context verbatim when present."
)


# ------------------------------- helpers ---------------------------------- #
def _downscale_jpeg(data: bytes, max_side: int = 1024) -> bytes:  # type: ignore[misc]
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
    """Strip any 'You are ...' style system leftovers & trim length."""
    if not s:
        return ""
    s = re.sub(r"(?i)^\s*you are .*?$", "", s.strip())
    s = re.sub(r"(?i)\b(system|instruction|prompt)\b.*", "", s)
    return s[:limit]


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

        # Text LLM (chat interface so system prompts work)
        self.text_llm = ChatOllama(
            model=self.text_model,
            base_url=self.base_url,
            temperature=0.2,
            model_kwargs={"num_ctx": 2048, "num_predict": 320, "top_p": 0.9},
        )

        # Tiny/fast model for formatting/normalizing tasks (falls back to main model)
        fast_model = os.getenv("OLLAMA_FAST_TEXT_MODEL", self.text_model)
        self.fast_llm = ChatOllama(
            model=fast_model,
            base_url=self.base_url,
            temperature=0.1,
            model_kwargs={"num_ctx": 1024, "num_predict": 200, "top_p": 0.9},
        )

        # Vision client (LLaVA)
        self.client = ollama.Client(host=self.base_url)

        # Optional RAG stores
        self.pdf_store: Optional[PineconeStore] = None
        self.medicine_store: Optional[PineconeStore] = None
        try:
            self.pdf_store = PineconeStore(index_name=PINECONE_PDF_INDEX, dimension=384)
        except Exception as e:
            logger.warning(f"PDF RAG store init failed: {e}")
        try:
            self.medicine_store = PineconeStore(index_name=PINECONE_MEDICINE_INDEX, dimension=384)
        except Exception as e:
            logger.warning(f"Medicine RAG store init failed: {e}")

    # --------------------------- Core text/vision --------------------------- #
    @traceable(name="text_completion")
    def generate_text_response(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate a response using the text LLM with an optional system prompt."""
        try:
            messages = [
                SystemMessage(content=system_prompt or DEFAULT_SYSTEM),
                HumanMessage(content=prompt),
            ]
            resp = self.text_llm.invoke(messages)
            return getattr(resp, "content", str(resp))
        except Exception as e:
            logger.error(f"Error in text generation: {e}")
            return f"Error generating response: {str(e)}"

    @traceable(name="vision_completion")
    def generate_vision_response(
        self,
        prompt: str,
        image_path: Optional[str] = None,
        image_data: Optional[bytes] = None,
    ) -> str:
        """Generate a response using a vision LLM (LLaVA) with an image."""
        try:
            if image_data:
                b64 = base64.b64encode(image_data).decode("utf-8")
                images = [b64]
            elif image_path:
                with open(image_path, "rb") as f:
                    images = [base64.b64encode(f.read()).decode("utf-8")]
            else:
                return "No image provided."

            r = self.client.chat(
                model=self.vision_model,
                messages=[{"role": "user", "content": prompt, "images": images}],
                stream=False,
                options={"temperature": 0.1, "num_ctx": 2048, "top_p": 0.9},
                keep_alive="30m",
            )
            return r.get("message", {}).get("content", "")
        except Exception as e:
            logger.error(f"Error in vision generation: {e}")
            return f"Error analyzing image: {str(e)}"

    # ------------------------------- OCR utils ------------------------------ #
    def _fast_format(self, prompt: str) -> str:
        try:
            resp = self.fast_llm.invoke([SystemMessage(content=DEFAULT_SYSTEM), HumanMessage(content=prompt)])
            return getattr(resp, "content", str(resp))
        except Exception:
            return self.generate_text_response(prompt, system_prompt=DEFAULT_SYSTEM)

    def ocr_only(self, image_bytes: bytes) -> Dict[str, Any]:
        """Very fast OCR-only parse used when no complex vision is needed."""
        pipe = OCRPipeline(lang=OCR_LANGUAGE)
        # run_on_bytes returns (lines, items, header)
        lines, items, header = pipe.run_on_bytes(_downscale_jpeg(image_bytes))
        return {
            "raw_text": "\n".join(lines),
            "lines": lines,
            "structured_items": items,
            "item_count": len(items),
            "header": header or {},
        }

    def answer_over_ocr_text(
        self,
        question: str,
        ocr_text: str,
        conversation_context: str = "",
    ) -> str:
        # quick heuristic: try to pull a Doctor name
        if "doctor" in question.lower() or "dr" in question.lower():
            m = re.search(r"(Dr\.?\s*[A-Z][A-Za-z.\s'-]{1,40})", ocr_text)
            if m:
                return m.group(1).strip()

        prompt = (
            f"{_clean_conv_ctx(conversation_context)}\n\n"
            "Answer the user's question using ONLY the OCR text below. "
            "If the answer is not present, say briefly that you can't find it in the OCR. "
            "Prefer header lines for doctor/clinic names. Be concise.\n\n"
            f"OCR TEXT:\n{ocr_text[:8000]}\n\n"
            f"QUESTION: {question}\n"
            "ANSWER:"
        )
        return self.generate_text_response(prompt, system_prompt=SYSTEM_GENERAL)


    # ------------------------------- RAG ----------------------------------- #
    def _select_namespaces(self, q: str) -> list[str]:
        ql = q.lower()
        if any(k in ql for k in ["law", "policy", "act", "regulation", "dgda"]):
            return ["policy"]
        if any(k in ql for k in ["otc", "self care", "over the counter"]):
            return ["otc", "prescribing", "guidelines"]
        if any(k in ql for k in ["dose", "dosing", "indication", "contraindication", "interaction", "pregnancy"]):
            return ["prescribing", "guidelines", "otc"]
        if any(k in ql for k in [
            "anatomy", "nerve", "artery", "vein", "muscle", "bone",
            "cell", "organelle", "nucleus", "mitochondria", "golgi", "ribosome",
            "histology", "cytology", "epithelium", "tissue", "organelles"
        ]):
            return ["textbook"]
        return ["guidelines", "prescribing", "otc"]

    def _gather_rag_context(self, query: str, extra_ctx: Optional[List[str]] = None) -> List[str]:
        ctx: List[str] = []
        if extra_ctx:
            ctx.extend([c for c in extra_ctx if isinstance(c, str) and c.strip()])

        if self.bd_pharmacy_store:
            for ns in self._select_namespaces(query):
                try:
                    ctx.extend(self.bd_pharmacy_store.query(query, top_k=4, namespace=ns))
                except Exception as e:
                    logger.warning(f"BD pharmacy query failed ({ns}): {e}")

        try:
            if self.pdf_store:
                ctx.extend(self.pdf_store.query(query, top_k=2))
        except Exception as e:
            logger.warning(f"PDF RAG query failed: {e}")

        try:
            if self.medicine_store:
                ctx.extend(self.medicine_store.query(query, top_k=2))
        except Exception as e:
            logger.warning(f"Medicine RAG query failed: {e}")

        uniq, seen = [], set()
        for c in ctx:
            c = (c or "").strip()
            if c and c not in seen:
                seen.add(c)
                uniq.append(c[:1200])
        return uniq[:6]

    def _prompt_with_context(self, query: str, context: List[str], conversation_context: str = "") -> str:
        ctx_block = "\n\n---\n".join(context) if context else "No relevant context."
        conv = _clean_conv_ctx(conversation_context)
        return (
            (f"{conv}\n\n" if conv else "")
            + "Use the CONTEXT to answer the user's medical question. "
              "Prefer Bangladesh drug/brand specifics when relevant. "
              "If context is insufficient, say so briefly and give safe general guidance.\n\n"
              f"CONTEXT START\n---\n{ctx_block}\n---\nCONTEXT END\n\n"
              f"QUESTION: {query}"
        )

    # ----------------------- High-level convenience APIs ------------------- #
    @staticmethod
    def _safety_footer(text: str) -> str:
        red = ("chest pain" in text.lower() or "stroke" in text.lower() or "shortness of breath" in text.lower())
        if red:
            return text + "\n\n**If this is about a real person with urgent symptoms, seek emergency care immediately.**"
        return text

    @traceable(name="medical_query")
    def answer_medical_query(
        self,
        query: str,
        context: Optional[List[str]] = None,
        conversation_context: str = "",
    ) -> str:
        rag_ctx = self._gather_rag_context(query, extra_ctx=context)
        prompt = self._prompt_with_context(query, rag_ctx, conversation_context)
        out = self.generate_text_response(prompt, system_prompt=DEFAULT_SYSTEM)
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
    @traceable(name="prescription_analysis")
    def analyze_prescription(
        self,
        image_path: Optional[str] = None,
        image_data: Optional[bytes] = None,
        ocr_text: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        OCR-first parse:
          - robust OCR (PaddleOCR)
          - extract doctor/clinic from header
          - normalize meds via fast model
        If OCR yields nothing, fall back to a single vision pass.
        """
        try:
            ocr_blob = None
            if not ocr_text:
                if image_data:
                    ox = self.ocr_only(image_data)
                    ocr_text = ox.get("raw_text", "")
                    ocr_blob = ox
                elif image_path:
                    with open(image_path, "rb") as f:
                        ox = self.ocr_only(f.read())
                        ocr_text = ox.get("raw_text", "")
                        ocr_blob = ox
                else:
                    return {"error": "No image provided", "status": "error"}

            if not (ocr_text or "").strip():
                vision_prompt = (
                    "Read the prescription image and extract: patient name/age (if visible), "
                    "doctor/clinic, medicines (name, strength, dosage, frequency, duration), "
                    "diagnosis/notes. Return concise JSON-like text."
                )
                vision_analysis = self.generate_vision_response(vision_prompt, image_path, image_data)
                return {"vision_analysis": vision_analysis, "status": "success"}

            # Use header entities (doctor/clinic) if available from OCR pipeline
            doc_hint = ""
            if isinstance(ocr_blob, dict):
                hdr = ocr_blob.get("header") or {}
                if hdr.get("doctor") or hdr.get("clinic"):
                    doc_hint = f"\n\nHEADER HINTS: doctor={hdr.get('doctor','')}, clinic={hdr.get('clinic','')}"

            prompt = (
                "You are parsing a prescription. From the OCR text below, extract a compact JSON with keys: "
                "patient_name, doctor, clinic, diagnosis, medications[]. Each medication has name, strength, unit, "
                "dose, frequency, duration, and any note. Only include fields you can infer."
                f"{doc_hint}\n\nOCR:\n{ocr_text[:6000]}"
            )
            normalized = self._fast_format(prompt)
            return {
                "ocr_results": {"raw_text": ocr_text},
                "prescription_analysis": normalized,
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
                f"**Brand (BD):** {rec['brand']}"
                f"{'  |  Generic: ' + rec.get('generic','') if rec.get('generic') else ''}"
                f"{'  |  ' + rec.get('form','') if rec.get('form') else ''}"
                f"{'  ' + rec.get('strength','') if rec.get('strength') else ''}"
                f"{'  |  ' + rec.get('company','') if rec.get('company') else ''}"
                f"\nSource: {rec.get('source','')}"
            )

        rag_terms = [query]
        if rec.get("status") == "success":
            if rec.get("generic"):
                rag_terms.append(rec["generic"])
            if rec.get("brand"):
                rag_terms.append(rec["brand"])

        ctx = self._gather_rag_context(" ; ".join(rag_terms))
        prompt = self._prompt_with_context(
            "Summarize indication/uses, adult dosing ranges, key cautions (renal/hepatic, pregnancy), "
            "major interactions, and common side effects. Prefer Bangladesh context and plain English.",
            ctx,
        )
        summary = self._fast_format(prompt)

        price_line = ""
        if want_price and rec.get("status") == "success":
            p = self.web.get_bd_medicine_price(rec["brand"])
            if p.get("status") == "success":
                price_line = f"\n\n**Bangladesh retail price ({p['source']}):** {p['price']}"

        if not brand_line:
            return summary + (("\n\n" + price_line) if price_line else "")
        return brand_line + "\n\n" + summary + price_line

    # --------------------------- Model utilities --------------------------- #
    def check_model_availability(self) -> Dict[str, Any]:
        try:
            models = self.client.list()
            if isinstance(models, dict) and "models" in models:
                model_list = models["models"]
            elif hasattr(models, "models"):
                model_list = models.models  # type: ignore[attr-defined]
            else:
                model_list = models

            names = []
            for m in model_list:
                if isinstance(m, dict):
                    names.append(m.get("name", "") or m.get("model", ""))
                elif hasattr(m, "name"):
                    names.append(m.name)
                else:
                    names.append(str(m))

            return {
                "text_model_available": self.text_model in names,
                "vision_model_available": self.vision_model in names,
                "models_found": names,
            }
        except Exception as e:
            logger.error(f"Error checking model availability: {e}")
            return {
                "text_model_available": False,
                "vision_model_available": False,
                "error": str(e),
            }

    def retrieve_similar_anatomy(self, image_bytes: bytes, top_k: int = 6):
        pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        res = self.image_index.query_by_image(pil, top_k=top_k, namespace="anatomy")
        hits = []
        for m in res.get("matches", []):
            md = m.get("metadata", {})
            title = md.get("doc") or md.get("title", "")
            hits.append(f"[Figure match ~{int(m.get('score', 0) * 100)}%] {title}, p.{md.get('page')}: {md.get('caption', '')}")
        return hits
