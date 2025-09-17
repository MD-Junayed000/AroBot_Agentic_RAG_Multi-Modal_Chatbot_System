# agents/medical_agent.py
from __future__ import annotations
from typing import Dict, Any, List, Union, Optional
import io
import logging
from PIL import Image

from core.rag.context import RAGContextManager
# OCRAgent removed - not used in hot paths, only for tooling/scripts

from core.llm_modular import ModularLLMHandler as LLMHandler
from utils.web_search import WebSearchTool
from utils.medicine_intent import detect_intent, extract_candidate_brand

try:
    from mcp_server.mcp_handler import MCPHandler  # type: ignore
except Exception:
    MCPHandler = None  # type: ignore

logger = logging.getLogger(__name__)

def _to_image_bytes(image_input: Union[str, bytes, Image.Image]) -> bytes:
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
    def __init__(self):
        self.rag = RAGContextManager()
        # self.ocr_agent removed - not used, LLMHandler handles OCR directly
        self.llm = LLMHandler()
        self.web_search = WebSearchTool()
        self.mcp = MCPHandler() if MCPHandler else None
        self._last_brand_by_session: dict[str, str] = {}

    # ------------------------------- IMAGE -------------------------------- #

    def handle_image_query(
        self,
        image_input: Union[str, bytes, Image.Image],
        user_query: Optional[str] = None,
        image_type: str = "prescription",
    ) -> Dict[str, Any]:
        try:
            if image_type == "prescription":
                return self._handle_prescription_image(image_input, user_query)
            return self._handle_general_image(image_input, user_query)
        except Exception as e:
            logger.exception("Error handling image query")
            return {"error": str(e), "status": "error"}

    def handle_prescription_query(self, image_input: Union[str, bytes, Image.Image], user_query: Optional[str] = None) -> Dict[str, Any]:
        return self.handle_image_query(image_input, user_query, "prescription")

    def _handle_prescription_image(self, image_input: Union[str, bytes, Image.Image], user_query: Optional[str] = None) -> Dict[str, Any]:
        try:
            img_bytes = _to_image_bytes(image_input)
            rx = self.llm.analyze_prescription(image_data=img_bytes)
            if rx.get("status") != "success":
                return {"error": rx.get("error", "Failed to analyze image"), "status": "error"}

            follow_up_answer = None
            if user_query:
                ocr_text = (rx.get("ocr_results") or {}).get("raw_text", "") or ""
                if ocr_text.strip():
                    follow_up_answer = self.llm.answer_over_ocr_text(user_query, ocr_text)

            analysis_text = (
                rx.get("prescription_analysis")
                or rx.get("enhanced_analysis")
                or rx.get("vision_analysis")
                or self.llm.default_image_brief(img_bytes)  # fallback: brief
            )

            if follow_up_answer and follow_up_answer.strip() != "Can't find it in the OCR.":
                analysis_text = f"{analysis_text}\n\n**Follow-up:** {follow_up_answer}"
            elif follow_up_answer:
                analysis_text = f"{analysis_text}\n\n(That specific detail was not found in the OCR text.)"

            return {
                "prescription_analysis": analysis_text,
                "ocr_results": rx.get("ocr_results", {}),
                "llm_analysis": {"enhanced_analysis": rx.get("enhanced_analysis"), "vision_analysis": rx.get("vision_analysis")},
                "status": "success",
            }
        except Exception as e:
            logger.exception("Error handling prescription image")
            return {"error": str(e), "status": "error"}

    def _handle_general_image(self, image_input: Union[str, bytes, Image.Image], user_query: Optional[str] = None) -> Dict[str, Any]:
        try:
            img_bytes = _to_image_bytes(image_input)
            ocr = self.llm.ocr_only(img_bytes)
            raw = ocr.get("raw_text", "") or ""

            # If user asks for explicit OCR fields, answer directly
            if user_query:
                ql = user_query.lower()
                if any(k in ql for k in ["what is written","what is the name","name in the image","doctor","patient","phone","date","clinic"]):
                    direct = self.llm.answer_over_ocr_text(user_query, raw)
                    return {"analysis": direct, "ocr_results": ocr, "type": "general_image_analysis", "status": "success"}

            # Try CLIP image retrieval from anatomy index
            try:
                from core.image_index import CLIPImageIndex
                idx = CLIPImageIndex()
                pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                ivals = idx.query_by_image(pil, top_k=6, namespace="anatomy")
                matches = ivals.get("matches", []) if isinstance(ivals, dict) else []
                captions = [m.get("metadata", {}).get("caption", "") for m in matches]
                ctx = [c for c in captions if c]
            except Exception:
                ctx = []

            # If user asked a conceptual question (mitochondria, lysosome, etc.),
            # use encyclopedic answer with any context we have. If CLIP provided
            # no hints, enrich with light web snippets so the answer doesn't stall.
            if user_query and any(t in user_query.lower() for t in [
                "mitochondria","mitochondrion","lysosome","nucleus","ribosome","golgi","endoplasmic","skeletal","anatomy","cell"
            ]):
                web_ctx: List[str] = []
                try:
                    if not ctx:
                        web = self.web_search.search_medical_info(user_query, max_results=3)
                        if web and web.get("status") == "success":
                            web_ctx = [r.get("snippet", "") for r in web.get("results", [])[:3]]
                except Exception:
                    web_ctx = []

                rich_ctx = (ctx or []) + (web_ctx or [])
                text = self.llm.answer_general_knowledge(user_query, context=rich_ctx)
                return {"analysis": text, "ocr_results": ocr, "image_context": (ctx or [])[:3], "type": "general_image_analysis", "status": "success"}

            # Otherwise produce a brief description, enriched by any hints from CLIP captions
            brief = self.llm.default_image_brief(img_bytes)
            if ctx:
                brief = f"{brief}\n\nContext matches: " + "; ".join(ctx[:3])
            return {"analysis": brief, "ocr_results": ocr, "image_context": ctx[:3], "status": "success"}

        except Exception as e:
            logger.exception("Error handling general image")
            return {"error": str(e), "status": "error"}

    # ------------------------------- TEXT --------------------------------- #

    def handle_text_query(
        self,
        query: str,
        session_id: Optional[str] = None,
        conversation_context: str = "",
        use_web_search: bool = False,
    ) -> Dict[str, Any]:
        """
        Routing:
        1) Clinical case? -> clinical chain (answer_medical_query)
        2) OTC fast dose if the user asks dose or mentions OTC terms
        3) Brand/price/formulary card if NOT clinical and user asks brandish/price/pack
        4) Otherwise general medical RAG + optional web → structured clinician answer
        """
        try:
            q = query.strip()
            intent = detect_intent(q)

            # ---- 0) Greetings/small talk ----
            greet_tokens = ["hi", "hello", "hey", "good morning", "good evening", "good afternoon"]
            if len(q) <= 32 and any(q.lower().startswith(g) for g in greet_tokens):
                text = self.llm.greeting_response(q)
                return {"response": text, "sources": {"mode": "greeting"}, "status": "success"}

            # About/capabilities/experience questions (avoid triggering clinical chain)
            ql = q.lower().strip()
            if intent.get("is_meta") or ql in {"about","help"}:
                text = self.llm.about_response()
                return {"response": text, "sources": {"mode": "about"}, "status": "success"}

            # Short acknowledgements like “ok”, “thanks” should not trigger medical essays
            small_talk = {
                "ok","okay","k","kk","thanks","thank you","thx","hmm","hmmm","h","huh",
                "yes","no","yep","yup","sure","fine","cool","nice","great","alright","got it","understood",
            }
            if ql in small_talk or (len(q.split()) <= 2 and not any(t in ql for t in [
                "what is","define","explain","meaning","dose","dosing","symptom","treatment","cause",
                "anatomy","drug","medicine","tablet","capsule","syrup","pain","fever","diarrhea","cough"
            ])):
                brief = (
                    "Got it. How can I help? You can ask about symptoms, medicines, upload a prescription image, or a PDF."
                )
                return {"response": brief, "sources": {"mode": "small_talk"}, "status": "success"}

            # ---- 0b) General anatomy / definition topics ----
            anatomy_terms = [
                "anatomy","cell","cells","organelles","organelle","nucleus","mitochondria","mitochondrion",
                "golgi","lysosome","lysosomes","endoplasmic reticulum","ribosome","skeletal","bone","muscle",
            ]
            is_definition = any(t in q.lower() for t in ["what is","define","explain","tell about","overview of"])  # explicit only
            is_anatomy = any(t in q.lower() for t in anatomy_terms)
            is_policy = bool(intent.get("is_policy"))
            if is_definition or is_anatomy or is_policy:
                # Pull a little RAG context first (BD pharmacy/textbook namespaces via llm handler)
                context = self.llm._gather_rag_context(q)
                web_results = None
                if not context:
                    try:
                        web_results = self.web_search.search_medical_info(q)
                    except Exception as exc:
                        logger.warning(f"Policy fallback web search failed: {exc}")
                    if web_results and web_results.get("status") == "success":
                        context = [r.get("snippet", "") for r in web_results.get("results", []) if r.get("snippet")]
                if not context:
                    return {
                        "response": (
                            "I couldn't find Bangladesh policy guidance in the local knowledge base. "
                            "Please upload a document or enable web search so I can access external sources."
                        ),
                        "sources": {"mode": "policy", "kb_hits": 0, "web_search": bool(web_results)},
                        "status": "success",
                    }
                text = self.llm.answer_general_knowledge(
                    q,
                    context=context,
                    conversation_context=conversation_context,
                )
                return {"response": text, "sources": {"mode": "encyclopedic", "kb_hits": len(context)}, "status": "success"}

            # ---- 1) Clinical case goes straight to clinical chain ----
            if intent.get("is_clinical_case"):
                final_response = self.llm.answer_medical_query(q, context=None, conversation_context=conversation_context)
                if session_id and self.mcp:
                    try: self.mcp.add_assistant_response(session_id, final_response, message_type="response")
                    except Exception: pass
                return {"response": final_response, "sources": {"mode": "clinical"}, "status": "success"}

            # ---- 2) OTC dosing information ----
            if intent.get("wants_dose"):
                fast_otc = self.llm.try_fast_otc_dose_answer(q, conversation_context)
                if fast_otc:
                    if session_id and self.mcp:
                        try: self.mcp.add_assistant_response(session_id, fast_otc, message_type="response")
                        except Exception: pass
                    return {"response": fast_otc, "sources": {"mode": "otc_facts"}, "status": "success"}

            # ---- 2b) Company overview (e.g., "tell me about beximco pharma") ----
            if intent.get("is_company") and not intent.get("wants_brand_pack"):
                # Use web search to pull a few snippets
                web = self.web_search.search_medical_info(q)
                snippets = []
                if web and web.get("status") == "success":
                    snippets = [r.get("snippet", "") for r in web.get("results", [])[:5]]
                ctx = "\n---\n".join(snippets)
                prompt = (
                    "Using the CONTEXT snippets, write a brief profile of the Bangladesh pharmaceutical company mentioned. "
                    "Include: overview, product domains, notable brands (if well-known), scale/market standing, and website/ticker if available. "
                    "Keep to 5–7 bullets; avoid speculation.\n\n"
                    f"CONTEXT\n---\n{ctx}\n---\nQUESTION: {q}"
                )
                text = self.llm.generate_text_response(prompt)
                if session_id and self.mcp:
                    try: self.mcp.add_assistant_response(session_id, text, message_type="response")
                    except Exception: pass
                return {"response": text, "sources": {"web_search": web.get("result_count", 0) if web else 0}, "status": "success"}

            # ---- 3) Brand/pack/price shortcuts (non-clinical only) ----
            if intent.get("is_price") or intent.get("wants_brand_pack") or intent.get("looks_brandish"):
                brand_candidate = extract_candidate_brand(q)
                if not brand_candidate:
                    # Try to recover a target from recent conversation (e.g., "Give BD brands?" after dosing talk)
                    from utils.clinical_facts import OTC_DOSING
                    cc = conversation_context.lower() if conversation_context else ""
                    for g, meta in OTC_DOSING.items():
                        if any(s in cc for s in meta.get("synonyms", []) + [g]):
                            brand_candidate = g
                            break
                answer = self.llm.answer_medicine(brand_candidate, want_price=bool(intent.get("is_price")))
                if session_id and self.mcp:
                    try: self.mcp.add_assistant_response(session_id, answer, message_type="response")
                    except Exception: pass
                return {
                    "response": answer,
                    "sources": {"resolver": "medex.com.bd", "kb": "pinecone (if used)"},
                    "status": "success",
                }

            # ---- 4) General medical pipeline (RAG + optional web) ----
            hint_terms = []
            # Extract medical terms from conversation context
            medical_terms = ["acetaminophen", "amoxicillin", "azithromycin", "dose", "dosage", "mg"]
            for t in medical_terms:
                if t in conversation_context.lower(): hint_terms.append(t)
            if hint_terms: q = q + " ; topic hints: " + ", ".join(sorted(set(hint_terms)))

            # Generic medical indicators
            medish_tokens = [
                "tablet", "capsule", "syrup", "dose", "dosing", "side effect", 
                "contraindication", "drug", "medicine", "brand", "generic", "mg", "ml"
            ]
            is_medicine_query = any(k in q.lower() for k in medish_tokens)

            if is_medicine_query:
                # Use modular RAG system
                rag_context = self.rag.gather_context(q)
                rag_response = self.llm.answer_medical_query(q, context=rag_context, conversation_context=conversation_context)
            else:
                rag_context = self.rag.gather_context(q)
                rag_response = self.llm.answer_medical_query(q, context=rag_context, conversation_context=conversation_context)

            web_results = None
            if use_web_search or not rag_response or len(rag_response) < 50:
                web_results = self.web_search.search_medical_info(q)

            context = rag_context or []
            if web_results and web_results.get("status") == "success":
                context.extend([r.get("snippet", "") for r in web_results.get("results", [])[:3]])

            # Hard grounding guard: if no KB hits and no web snippets, avoid free-form hallucination  
            kb_hits = len(rag_context) if rag_context else 0
            web_ok = bool(web_results and web_results.get("status") == "success" and (web_results.get("result_count", 0) or 0) > 0)
            if kb_hits == 0 and not web_ok:
                safe_msg = (
                    "I don’t have enough trusted sources locally to answer confidently. "
                    "You can: upload a PDF/image for me to analyze, or enable web search so I can fetch current guidance."
                )
                return {
                    "response": safe_msg,
                    "sources": {"knowledge_base": 0, "web_search": 0},
                    "status": "success",
                }

            # Use the already generated RAG response
            final_response = rag_response

            # Guard against unhelpful refusals for simple topics
            refusal_markers = ["i cannot", "i can't", "i am unable", "i'm unable", "i'm sorry"]
            if (any(t in q.lower() for t in ["mitochond", "lysos", "nucleus", "anatomy", "skeletal"]) or len(q.split()) <= 3) and \
               any(m in (final_response or "").lower() for m in refusal_markers):
                gctx = self.rag.gather_context(q)
                final_response = self.llm.answer_general_knowledge(q, context=gctx)

            if session_id and self.mcp:
                try: self.mcp.add_assistant_response(session_id, final_response, message_type="response")
                except Exception: pass

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

    # ------------------------------- STATUS -------------------------------- #

    def get_system_status(self) -> Dict[str, Any]:
        try:
            llm_status = {"text_model": "available", "vision_model": "available"}
            kb_status = {"status": "modular_rag_active"}
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
