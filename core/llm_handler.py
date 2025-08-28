"""
LLM Handler for text and vision models using Ollama
- Uses ChatOllama for proper system/user message handling
- Supports LLaVA vision via Ollama client.chat with base64 images
- Provides RAG helpers over Pinecone PDF & medicine indexes
- Keeps backward-compatible methods used elsewhere in the codebase
"""

from __future__ import annotations

import base64
import logging
from typing import List, Dict, Any, Optional

import ollama
from langchain_ollama import ChatOllama
from langchain.schema import HumanMessage, SystemMessage
from langsmith import traceable

from config.env_config import (
    OLLAMA_BASE_URL,
    OLLAMA_TEXT_MODEL,
    OLLAMA_VISION_MODEL,
    PINECONE_PDF_INDEX,
    PINECONE_MEDICINE_INDEX,
)

from .vector_store import PineconeStore

logger = logging.getLogger(__name__)

# Conservative, medicine-first system prompt
DEFAULT_SYSTEM = (
    "You are AroBot, a careful medical assistant with MANDATORY conversation memory utilization. "
    "Prioritize medical/clinical interpretations. When a token (e.g., 'Napa') is ambiguous, assume the "
    "medicine/brand meaning first. Use provided context verbatim. Cite uncertainties briefly and advise "
    "consulting a clinician for medical decisions."
)


class LLMHandler:
    """Handles both text and vision LLMs using Ollama."""

    def __init__(self) -> None:
        # Model endpoints
        self.base_url = OLLAMA_BASE_URL
        self.text_model = OLLAMA_TEXT_MODEL
        self.vision_model = OLLAMA_VISION_MODEL

        # Text LLM (chat interface so system prompts work)
        self.text_llm = ChatOllama(
            model=self.text_model,
            base_url=self.base_url,
            temperature=0.2,
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
            # ChatOllama returns an AIMessage object with .content
            return getattr(resp, "content", str(resp))
        except Exception as e:
            logger.error(f"Error in text generation: {e}")
            return f"Error generating response: {str(e)}"

    @traceable(name="vision_completion")
    def generate_vision_response(
        self, prompt: str, image_path: Optional[str] = None, image_data: Optional[bytes] = None
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

            # Use chat endpoint for images with LLaVA
            r = self.client.chat(
                model=self.vision_model,
                messages=[{"role": "user", "content": prompt, "images": images}],
                stream=False,
            )
            # Response shape: {"message": {"role": "assistant","content": "..."} , ...}
            return r.get("message", {}).get("content", "")
        except Exception as e:
            logger.error(f"Error in vision generation: {e}")
            return f"Error analyzing image: {str(e)}"

    # ------------------------------- RAG ----------------------------------- #

    def _gather_rag_context(self, query: str, extra_ctx: Optional[List[str]] = None) -> List[str]:
        """Collect context snippets from Pinecone indexes + any extra context provided."""
        ctx: List[str] = []
        if extra_ctx:
            ctx.extend([c for c in extra_ctx if isinstance(c, str) and c.strip()])

        # Query PDF KB
        try:
            if self.pdf_store:
                ctx.extend(self.pdf_store.query(query, top_k=4))
        except Exception as e:
            logger.warning(f"PDF RAG query failed: {e}")

        # Query Medicine KB
        try:
            if self.medicine_store:
                ctx.extend(self.medicine_store.query(query, top_k=4))
        except Exception as e:
            logger.warning(f"Medicine RAG query failed: {e}")

        # Deduplicate & trim
        uniq, seen = [], set()
        for c in ctx:
            c = (c or "").strip()
            if c and c not in seen:
                seen.add(c)
                uniq.append(c[:1500])  # keep prompt small
        return uniq[:8]

    def _prompt_with_context(self, query: str, context: List[str], conversation_context: str = "") -> str:
        ctx_block = "\n\n---\n".join(context) if context else "No relevant context."
        return (
            f"{conversation_context}\n\n"
            f"Use the following CONTEXT to answer the user's medical question. "
            f"When possible, ground the answer explicitly in the context. "
            f"If the context is insufficient, say so briefly and provide safe general information.\n\n"
            f"CONTEXT START\n---\n{ctx_block}\n---\nCONTEXT END\n\n"
            f"QUESTION: {query}"
        )

    # ----------------------- High-level convenience APIs ------------------- #

    @traceable(name="medical_query")
    def answer_medical_query(
        self, query: str, context: Optional[List[str]] = None, conversation_context: str = ""
    ) -> str:
        """Answer medical queries using RAG + conversation memory string."""
        rag_ctx = self._gather_rag_context(query, extra_ctx=context)
        prompt = self._prompt_with_context(query, rag_ctx, conversation_context)
        return self.generate_text_response(prompt, system_prompt=DEFAULT_SYSTEM)

    def answer_over_pdf_text(
        self, question_or_none: Optional[str], pdf_text: str, conversation_context: str = ""
    ) -> str:
        """
        If a question is provided, answer grounded on that PDF text; otherwise produce a concise summary.
        """
        if not pdf_text or not pdf_text.strip():
            return "The PDF appears to contain no readable text."

        if not question_or_none:
            prompt = (
                "Summarize the following PDF text for a clinician-friendly audience. "
                "Include title (if present), section highlights, and 3–5 key takeaways.\n\n"
                f"{pdf_text[:15000]}"
            )
            return self.generate_text_response(prompt, system_prompt=DEFAULT_SYSTEM)
        else:
            return self.answer_medical_query(
                question_or_none, context=[pdf_text], conversation_context=conversation_context
            )

    @traceable(name="prescription_analysis")
    def analyze_prescription(
        self, image_path: Optional[str] = None, image_data: Optional[bytes] = None, ocr_text: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze a prescription image with vision and optionally enrich with OCR text.
        Returns a structured dict with analyses.
        """
        vision_prompt = (
            "You are a medical assistant analyzing a prescription image. Extract:\n"
            "1) Patient info (name/age if visible)\n"
            "2) Doctor/clinic details\n"
            "3) Medications (name, strength, dosage, frequency, duration)\n"
            "4) Diagnoses/indications or symptoms\n"
            "5) Instructions/notes\n\n"
            "Return a concise JSON-like summary. Only include what you can clearly identify."
        )

        try:
            vision_analysis = self.generate_vision_response(vision_prompt, image_path, image_data)

            if ocr_text:
                text_prompt = (
                    "Combine the following OCR text with the vision analysis of a prescription to improve accuracy. "
                    "Return a single concise JSON-like object with medications and fields where confidently extracted.\n\n"
                    f"OCR TEXT:\n{ocr_text[:5000]}\n\nVISION ANALYSIS:\n{vision_analysis}"
                )
                enhanced = self.generate_text_response(text_prompt, system_prompt=DEFAULT_SYSTEM)
                return {
                    "vision_analysis": vision_analysis,
                    "ocr_text": ocr_text,
                    "enhanced_analysis": enhanced,
                    "status": "success",
                }

            return {"vision_analysis": vision_analysis, "status": "success"}

        except Exception as e:
            logger.error(f"Error in prescription analysis: {e}")
            return {"error": str(e), "status": "error"}

    # ------------------------ Conversation helpers ------------------------ #

    def process_conversation_context(self, conversation_context: str) -> Dict[str, Any]:
        """
        Keep this method for backward compatibility.
        Parses a conversation transcript for simple personal/medical cues.
        """
        if not conversation_context or not conversation_context.strip():
            return {"has_context": False, "extracted_info": {}}

        lines = conversation_context.split("\n")
        structured = {
            "user_messages": [],
            "assistant_messages": [],
            "personal_info": {},
            "medical_context": [],
            "conversation_flow": [],
        }

        current_speaker = None
        import re

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if line.startswith("User:"):
                current_speaker = "user"
                content = line.replace("User:", "").strip()
                structured["user_messages"].append(content)

                low = content.lower()
                if "i am dr" in low:
                    m = re.search(r"i am dr\.?\s+([a-z]+\s+[a-z]+)", low)
                    if m:
                        structured["personal_info"]["name"] = m.group(1).title()

                if "department" in low:
                    m = re.search(r"(\w+)\s+department", low)
                    if m:
                        structured["personal_info"]["department"] = m.group(1)

                if "hospital" in low or "medical center" in low:
                    m = re.search(r"(?:at|from|in)\s+([a-z\s]+(?:hospital|medical center))", low)
                    if m:
                        structured["personal_info"]["hospital"] = m.group(1).title()

                medical_keywords = ["patient", "medication", "prescription", "symptom", "diagnosis", "treatment"]
                if any(k in low for k in medical_keywords):
                    structured["medical_context"].append(content)

            elif line.startswith("Assistant:") or line.startswith("AroBot:"):
                current_speaker = "assistant"
                content = line.replace("Assistant:", "").replace("AroBot:", "").strip()
                structured["assistant_messages"].append(content)

            if current_speaker:
                structured["conversation_flow"].append(
                    {"speaker": current_speaker, "content": content if current_speaker == "user" else line}
                )

        structured["has_context"] = True
        return structured

    # --------------------------- Model utilities --------------------------- #

    def check_model_availability(self) -> Dict[str, Any]:
        """Check if text/vision models are listed by the local Ollama server."""
        try:
            models = self.client.list()
            # Extract name list robustly across Ollama versions
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
            return {"text_model_available": False, "vision_model_available": False, "error": str(e)}
