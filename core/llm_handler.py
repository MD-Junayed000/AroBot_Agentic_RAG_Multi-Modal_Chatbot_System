# core/llm_handler.py
"""
LLM Handler for text and vision models using Ollama
- ChatOllama for text (system/user messages)
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
from typing import List, Dict, Any, Optional, Tuple

from PIL import Image, UnidentifiedImageError
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

from utils.clinical_facts import OTC_DOSING, find_otc_targets_in_text

logger = logging.getLogger(__name__)

# ------------------------------ System prompts --------------------------- #

DEFAULT_SYSTEM = (
    "You are a careful medical assistant. Be concise and clinically safe. "
    "Prefer Bangladesh-specific drug/brand context when ambiguous (e.g., 'Napa' → paracetamol brand). "
    "State uncertainty briefly and advise consulting a clinician for medical decisions."
)

SYSTEM_GENERAL = (
    "You are a helpful, concise assistant. Answer clearly and directly. "
    "Use provided context verbatim when present."
)

STRICT_MEDICAL_SYSTEM = (
    "You are  a clinical domain expert. Use evidence-based, conservative guidance. "
    "Do NOT invent drugs, tests, doses, or brand names. "
    "Prefer generics; mention Bangladesh brand examples only when directly requested or clearly relevant. "
    "If information is insufficient, request the minimum critical details. "
    "Always include brief safety caveats for real patients."
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

        # Text LLM (chat interface so system prompts work)
        self.text_llm = ChatOllama(
            model=self.text_model,
            base_url=self.base_url,
            temperature=0.2,
            model_kwargs={"num_ctx": 2048, "num_predict": 320, "top_p": 0.9},
        )

        # Tiny/fast model for formatting/normalizing tasks
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
        self, prompt: str, image_path: Optional[str] = None, image_data: Optional[bytes] = None
    ) -> str:
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
        try:
            # downscale attempts to open; catch invalid image payloads
            data = _downscale_jpeg(image_bytes)
        except UnidentifiedImageError:
            return {"raw_text": "", "lines": [], "structured_items": [], "item_count": 0, "header": {}, "error": "not_an_image"}

        pipe = OCRPipeline(lang=OCR_LANGUAGE)
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
        ql = q.lower()
        if any(k in ql for k in ["law", "policy", "act", "regulation", "dgda"]): return ["policy"]
        if any(k in ql for k in ["otc", "self care", "over the counter"]): return ["otc","prescribing","guidelines"]
        if any(k in ql for k in ["dose","dosing","indication","contraindication","interaction","pregnancy"]):
            return ["prescribing","guidelines","otc"]
        if any(k in ql for k in ["anatomy","nerve","artery","vein","muscle","bone","cell","organelle","nucleus",
                                 "mitochondria","golgi","ribosome","histology","cytology","epithelium","tissue","organelles"]):
            return ["textbook"]
        return ["guidelines","prescribing","otc"]

    def _gather_rag_context(self, query: str, extra_ctx: Optional[List[str]] = None) -> List[str]:
        ctx: List[str] = []
        if extra_ctx: ctx.extend([c for c in extra_ctx if isinstance(c, str) and c.strip()])

        if self.bd_pharmacy_store:
            for ns in self._select_namespaces(query):
                try:
                    ctx.extend(self.bd_pharmacy_store.query(query, top_k=4, namespace=ns))
                except Exception as e:
                    logger.warning(f"BD pharmacy query failed ({ns}): {e}")

        try:
            if self.pdf_store: ctx.extend(self.pdf_store.query(query, top_k=2))
        except Exception as e:
            logger.warning(f"PDF RAG query failed: {e}")

        try:
            if self.medicine_store: ctx.extend(self.medicine_store.query(query, top_k=2))
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

    @traceable(name="medical_query")
    def answer_medical_query(self, query: str, context: Optional[List[str]] = None, conversation_context: str = "") -> str:
        rag_ctx = self._gather_rag_context(query, extra_ctx=context)
        ctx_block = "\n---\n".join(rag_ctx) if rag_ctx else "No relevant context"

        # Expert-style structure
        scaffold_prompt = (
            "You are a careful clinician. Using ONLY the CONTEXT (and general medical knowledge if needed), "
            "answer concisely with sections when appropriate:\n"
            "1) Most likely\n2) Consider\n3) Key questions / tests that change management\n"
            "4) Supportive care now\n5) Red flags (urgent/ER)\n"
            "Be specific, avoid unnecessary drugs, and align with safe primary-care practice. "
            "If data is sparse, say so briefly.\n\n"
            f"CONTEXT START\n---\n{ctx_block}\n---\nCONTEXT END\n\n"
            f"QUESTION: {query}\n\nAnswer:"
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

    @staticmethod
    def _summarize_structured(obj: Dict[str, Any]) -> str:
        if not obj:
            return "Couldn't confidently read the prescription."
        parts = []
        if obj.get("doctor"):
            parts.append(f"Doctor: {obj['doctor']}")
        if obj.get("clinic"):
            parts.append(f"Clinic: {obj['clinic']}")
        if obj.get("date"):
            parts.append(f"Date: {obj['date']}")
        if obj.get("patient_name"):
            pn = obj["patient_name"]
            if obj.get("age_sex"):
                pn += f", {obj['age_sex']}"
            parts.append(f"Patient: {pn}")
        if obj.get("diagnosis_or_cc"):
            parts.append(f"Dx/CC: {obj['diagnosis_or_cc']}")
        meds = obj.get("medications") or []
        if meds:
            lines = []
            for i, m in enumerate(meds, 1):
                name = m.get("name") or "—"
                dose = m.get("dose") or m.get("dose_pattern") or ""
                extra = m.get("additional_instructions") or ""
                frag = name
                if dose:
                    frag += f" — {dose}"
                if extra:
                    frag += f" ({extra})"
                lines.append(f"{i}) {frag}")
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
                "You are extracting data from a handwritten medical prescription. "
                "RULES:\n"
                "- OUTPUT JSON ONLY in schema with keys: patient_name, age_sex, doctor, clinic, date, "
                "diagnosis_or_cc, duration, medications[], notes.\n"
                "- medications[] fields: name, generic_suspected, form, route, dose, dose_pattern, "
                "frequency, duration, quantity_or_count_mark, additional_instructions, confidence.\n"
                "- If unclear, leave fields empty. NEVER invent.\n"
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

            # ---- 6) Summarize ----
            summary = self._summarize_structured(structured)
            return {
                "mode": used_mode,
                "ocr_results": {"raw_text": ocr_text or ""},
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
            if generic_name == "paracetamol": q = "paracetamol 500"
            if generic_name == "ibuprofen":   q = "ibuprofen 200"
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
