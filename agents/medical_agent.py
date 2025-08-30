# agents/medical_agent.py
"""
Medical Agent - Main orchestrator for medical queries and multi-modal interactions

- Fast OCR-first image handling (vision is a fallback)
- Solid prescription parsing via LLMHandler.analyze_prescription()
- Sharp follow-ups over last OCR text (e.g., “what is the name in the image?”)
- BD medicine brand/generic resolution and price answers via LLMHandler.answer_medicine()
- Keeps RAG + web search pipeline for general medical queries
"""

from __future__ import annotations
from typing import Dict, Any, List, Union, Optional
import io
import logging
from PIL import Image

from .rag_agent import RAGAgent
from .ocr_agent import OCRAgent

# always import from core package
from core.llm_handler import LLMHandler
from utils.web_search import WebSearchTool
from utils.medicine_intent import detect_intent, extract_candidate_brand

try:
    # optional MCP memory; keep it best-effort
    from mcp_server.mcp_handler import MCPHandler  # type: ignore
except Exception:  # pragma: no cover
    MCPHandler = None  # type: ignore

logger = logging.getLogger(__name__)


def _to_image_bytes(image_input: Union[str, bytes, Image.Image]) -> bytes:
    """Accept path / bytes / PIL Image and return JPEG bytes."""
    if isinstance(image_input, bytes):
        return image_input
    if isinstance(image_input, str):
        with open(image_input, "rb") as f:
            return f.read()
    if isinstance(image_input, Image.Image):
        buf = io.BytesIO()
        image_input.convert("RGB").save(buf, format="JPEG", quality=90, optimize=True)
        return buf.getvalue()
    raise ValueError("Unsupported image_input type")


class MedicalAgent:
    """Main medical agent that orchestrates different capabilities"""

    def __init__(self):
        self.rag_agent = RAGAgent()
        self.ocr_agent = OCRAgent()
        self.llm = LLMHandler()
        self.web_search = WebSearchTool()
        self.mcp = MCPHandler() if MCPHandler else None
        self._last_brand_by_session: dict[str, str] = {}

    # --------------------------------------------------------------------- #
    # IMAGE HANDLING
    # --------------------------------------------------------------------- #
    def handle_image_query(
        self,
        image_input: Union[str, bytes, Image.Image],
        user_query: Optional[str] = None,
        image_type: str = "prescription",
    ) -> Dict[str, Any]:
        """Handle any image analysis with optional user query."""
        try:
            if image_type == "prescription":
                return self._handle_prescription_image(image_input, user_query)
            return self._handle_general_image(image_input, user_query)
        except Exception as e:
            logger.exception("Error handling image query")
            return {"error": str(e), "status": "error"}

    def handle_prescription_query(
        self, image_input: Union[str, bytes, Image.Image], user_query: Optional[str] = None
    ) -> Dict[str, Any]:
        """Compatibility wrapper."""
        return self.handle_image_query(image_input, user_query, "prescription")

    def _handle_prescription_image(
        self, image_input: Union[str, bytes, Image.Image], user_query: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        OCR-first, then normalize via tiny model; vision is used only if OCR fails.
        Returns a dict that the API routes store + show.
        """
        try:
            img_bytes = _to_image_bytes(image_input)

            # Unified analyzer (fast OCR -> normalize; vision fallback handled inside)
            rx = self.llm.analyze_prescription(image_data=img_bytes)
            if rx.get("status") != "success":
                return {"error": rx.get("error", "Failed to analyze image"), "status": "error"}

            # Optional follow-up over OCR
            follow_up_answer = None
            if user_query:
                ocr_text = (rx.get("ocr_results") or {}).get("raw_text", "") or ""
                if ocr_text.strip():
                    follow_up_answer = self.llm.answer_over_ocr_text(user_query, ocr_text)

            # Compose a concise string for the chat bubble
            analysis_text = (
                rx.get("prescription_analysis")
                or rx.get("enhanced_analysis")
                or rx.get("vision_analysis")
                or "Prescription analyzed."
            )

            if follow_up_answer:
                analysis_text = f"{analysis_text}\n\n**Follow-up:** {follow_up_answer}"

            return {
                "prescription_analysis": analysis_text,
                "ocr_results": rx.get("ocr_results", {}),
                "llm_analysis": {
                    "enhanced_analysis": rx.get("enhanced_analysis"),
                    "vision_analysis": rx.get("vision_analysis"),
                },
                "status": "success",
            }

        except Exception as e:
            logger.exception("Error handling prescription image")
            return {"error": str(e), "status": "error"}

    def _handle_general_image(
        self, image_input: Union[str, bytes, Image.Image], user_query: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Fast general image analysis:
        - Run OCR (quick).
        - If query asks about text/names/numbers, answer directly from OCR.
        - Else call vision once and produce a concise description that *uses* OCR text as hints.
        """
        try:
            img_bytes = _to_image_bytes(image_input)

            # 1) Fast OCR
            ocr = self.llm.ocr_only(img_bytes)
            raw = ocr.get("raw_text", "") or ""

            # 2) If a direct question about text/identity exists, answer from OCR only
            if user_query:
                ql = user_query.lower()
                if any(k in ql for k in
                       ["what is written", "what is the name", "name in the image",
                        "doctor", "patient", "phone", "date", "clinic"]):
                    direct = self.llm.answer_over_ocr_text(user_query, raw)
                    return {
                        "analysis": direct,
                        "ocr_results": ocr,
                        "type": "general_image_analysis",
                        "status": "success",
                    }

            # 3) Otherwise, do a single concise vision pass using OCR as context
            prompt = (
                "Describe the image briefly (1–3 sentences). "
                "Prefer exact words from the OCR text when relevant.\n\n"
                f"OCR Hints:\n{raw[:1200]}"
            )
            vis = self.llm.generate_vision_response(prompt=prompt, image_data=img_bytes)

            return {"analysis": vis, "ocr_results": ocr, "type": "general_image_analysis", "status": "success"}

        except Exception as e:
            logger.exception("Error handling general image")
            return {"error": str(e), "status": "error"}

    # --------------------------------------------------------------------- #
    # TEXT HANDLING
    # --------------------------------------------------------------------- #
    def handle_text_query(
        self,
        query: str,
        session_id: Optional[str] = None,
        conversation_context: str = "",
        use_web_search: bool = False,
    ) -> Dict[str, Any]:
        """
        Main router for text questions.
        Fast path for BD brands/prices; else fall back to RAG + (optional) web.
        """
        try:
            q = query.strip()
            intent = detect_intent(q)

            # -------- FAST BRAND / PRICE PATH --------
            if intent.get("is_price") or intent.get("looks_brandish"):
                brand_candidate = extract_candidate_brand(q)
                answer = self.llm.answer_medicine(brand_candidate, want_price=bool(intent.get("is_price")))
                if session_id and self.mcp:
                    try:
                        self.mcp.add_assistant_response(session_id, answer, message_type="response")
                    except Exception:
                        pass
                return {
                    "response": answer,
                    "sources": {"resolver": "medex.com.bd", "kb": "pinecone (if used)"},
                    "status": "success",
                }

            # Medicines vs general medical (prefer med RAG for druggy queries)
            medish_tokens = [
                "paracetamol", "acetaminophen", "aspirin", "ibuprofen", "naproxen", "tablet",
                "capsule", "syrup", "dose", "dosing", "side effect", "contraindication",
                "drug", "medicine", "brand", "generic", "napa"
            ]
            is_medicine_query = any(k in q.lower() for k in medish_tokens)

            if is_medicine_query:
                rag_response = self.rag_agent.search_medicine_by_name(q)
                if not rag_response or rag_response.get("medicine_sources", 0) == 0:
                    rag_response = self.rag_agent.generate_medical_response(
                        q, conversation_context=conversation_context
                    )
            else:
                rag_response = self.rag_agent.generate_medical_response(
                    q, conversation_context=conversation_context
                )

            # Web only if asked or RAG empty
            web_results = None
            if use_web_search or (rag_response.get("medical_sources", 0) == 0):
                web_results = self.web_search.search_medical_info(q)

            context = rag_response.get("context_used", []) or []
            if web_results and web_results.get("status") == "success":
                context.extend([r.get("snippet", "") for r in web_results.get("results", [])[:3]])

            final_response = self.llm.answer_medical_query(
                q, context=context, conversation_context=conversation_context
            )

            if session_id and self.mcp:
                try:
                    self.mcp.add_assistant_response(session_id, final_response, message_type="response")
                except Exception:
                    pass

            return {
                "response": final_response,
                "rag_response": rag_response,
                "web_results": web_results,
                "sources": {
                    "knowledge_base": rag_response.get("medical_sources", 0) + rag_response.get("medicine_sources", 0),
                    "web_search": len(web_results.get("results", [])) if web_results else 0,
                },
                "status": "success",
            }

        except Exception as e:
            logger.error(f"Error handling text query: {e}")
            return {"error": str(e), "status": "error"}

    # --------------------------------------------------------------------- #
    # CONDITION → MEDICINE SEARCH
    # --------------------------------------------------------------------- #
    def search_medicine_by_condition(self, condition: str) -> Dict[str, Any]:
        """Search for medicines that treat a specific condition (RAG + web)."""
        try:
            rag_results = self.rag_agent.search_medicine_by_condition(condition)
            web_results = self.web_search.search_medical_info(f"{condition} treatment medication")

            if rag_results.get("status") != "success":
                return rag_results

            response = rag_results.get("response", "")
            if web_results and web_results.get("status") == "success":
                web_context = [r.get("snippet", "") for r in web_results.get("results", [])[:2]]
                if web_context:
                    enhanced = self.llm.generate_text_response(
                        f"Enhance the following answer about **{condition}** with these extra notes:\n"
                        f"ANSWER:\n{response}\n\nEXTRA:\n" + "\n".join(web_context)
                    )
                    response = response + "\n\nAdditional Information:\n" + enhanced

            return {
                "response": response,
                "condition": condition,
                "rag_results": rag_results,
                "web_results": web_results,
                "status": "success",
            }

        except Exception as e:
            logger.exception("Error searching medicine by condition")
            return {"error": str(e), "status": "error"}

    # --------------------------------------------------------------------- #
    # STATUS
    # --------------------------------------------------------------------- #
    def get_system_status(self) -> Dict[str, Any]:
        """Get status of all system components."""
        try:
            llm_status = self.llm.check_model_availability()
            kb_status = self.rag_agent.get_knowledge_base_stats()
            return {
                "llm_models": llm_status,
                "knowledge_bases": kb_status,
                "ocr_agent": "active",
                "web_search": "active",
                "overall_status": "operational",
                "status": "success",
            }
        except Exception as e:
            logger.exception("Error getting system status")
            return {"error": str(e), "status": "error"}

    # helpers
    def _remember_brand(self, session_id: str, brand: str):
        if brand:
            self._last_brand_by_session[session_id] = brand

    def _recall_brand(self, session_id: str) -> str:
        return self._last_brand_by_session.get(session_id, "")
