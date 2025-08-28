"""
API routes for AroBot Medical Chatbot
"""
import io
import logging
import re
from pathlib import Path
from typing import Optional, Dict, Any, List

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from PIL import Image
from pydantic import BaseModel

from config.env_config import TEMPLATES_DIR, PINECONE_API_KEY
from mcp_server.mcp_handler import MCPHandler

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------------------
# Conversation memory (fast in-memory ring + MCP persistent memory)
# --------------------------------------------------------------------------------------
conversation_memory: Dict[str, List[Dict[str, str]]] = {}


def add_to_conversation_memory(session_id: str, role: str, message: str):
    if session_id not in conversation_memory:
        conversation_memory[session_id] = []
    conversation_memory[session_id].append({"role": role, "message": message})
    # keep a short, fresh window
    if len(conversation_memory[session_id]) > 12:
        conversation_memory[session_id] = conversation_memory[session_id][-12:]


def get_conversation_memory_context(session_id: str) -> str:
    msgs = conversation_memory.get(session_id, [])
    if not msgs:
        return ""
    lines = [f"{m['role']}: {m['message']}" for m in msgs[-8:]]
    return "Previous conversation:\n" + "\n".join(lines)


# --------------------------------------------------------------------------------------
# Perfect “pattern” memory (fast answers to “what’s my name/department/hospital?”)
# --------------------------------------------------------------------------------------
class PerfectMemoryProcessor:
    def __init__(self):
        self.session_facts: Dict[str, Dict[str, str]] = {}

    def extract_facts(self, transcript: str) -> Dict[str, List[str]]:
        patterns = {
            "name": [
                r"I am (?:Dr\.?\s+)?([A-Z][a-z]+\s+[A-Z][a-z]+)\b",
                r"my name is (?:Dr\.?\s+)?([A-Z][a-z]+\s+[A-Z][a-z]+)\b",
                r"I'?m (?:Dr\.?\s+)?([A-Z][a-z]+\s+[A-Z][a-z]+)\b",
                r"call me (?:Dr\.?\s+)?([A-Z][a-z]+\s+[A-Z][a-z]+)\b",
            ],
            "department": [
                r"(?:in|from|work in)\s+(?:the\s+)?([a-z]+)\s+department",
                r"special(?:ty|ize) (?:is|in)\s+([a-z]+)",
            ],
            "hospital": [
                r"at ([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+Hospital)",
                r"work at ([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
            ],
        }
        facts = {"name": [], "department": [], "hospital": []}
        for key, pats in patterns.items():
            for p in pats:
                for m in re.findall(p, transcript, flags=re.IGNORECASE):
                    facts[key].append(str(m).strip().title())
        for k in facts:
            # dedupe
            seen, clean = set(), []
            for v in facts[k]:
                if v not in seen:
                    seen.add(v)
                    clean.append(v)
            facts[k] = clean
        return facts

    def detect_query_type(self, query: str) -> Optional[str]:
        q = query.lower()
        if any(k in q for k in ["what is my name", "what's my name", "who am i", "my name"]):
            return "name"
        if any(k in q for k in ["what department", "which department", "department do i work"]):
            return "department"
        if any(k in q for k in ["which hospital", "what hospital", "hospital do i work"]):
            return "hospital"
        return None

    def answer_if_memory(self, query: str, session_id: str, transcript: str) -> Optional[str]:
        facts_now = self.extract_facts(transcript)
        if session_id not in self.session_facts:
            self.session_facts[session_id] = {}
        # keep most recent
        for k, vals in facts_now.items():
            if vals:
                self.session_facts[session_id][k] = vals[-1]

        t = self.detect_query_type(query)
        if not t:
            return None
        val = self.session_facts[session_id].get(t)
        if not val:
            # try immediate transcript
            if facts_now.get(t):
                val = facts_now[t][-1]
        if not val:
            return None

        if t == "name":
            return f"Your name is Dr. {val}."
        if t == "department":
            return f"You work in the {val} department."
        if t == "hospital":
            return f"You work at {val}."
        return None


perfect_memory = PerfectMemoryProcessor()

# --------------------------------------------------------------------------------------
# Lazy singletons
# --------------------------------------------------------------------------------------
router = APIRouter()
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
_mcp: Optional[MCPHandler] = None
_medical_agent = None  # imported lazily to avoid heavy startup


def get_mcp() -> MCPHandler:
    global _mcp
    if _mcp is None:
        _mcp = MCPHandler()
    return _mcp


def get_medical_agent():
    global _medical_agent
    if _medical_agent is None:
        from agents.medical_agent import MedicalAgent
        _medical_agent = MedicalAgent()
    return _medical_agent


# --------------------------------------------------------------------------------------
# Small helper: create temp file
# --------------------------------------------------------------------------------------
def _write_temp_bytes(data: bytes, name: str) -> Path:
    temp_dir = Path("temp_uploads")
    temp_dir.mkdir(exist_ok=True)
    p = temp_dir / f"temp_{name}"
    p.write_bytes(data)
    return p


# --------------------------------------------------------------------------------------
# Weather helpers (Open-Meteo)
# --------------------------------------------------------------------------------------
def _get_weather_text(lat: float, lon: float) -> str:
    """Return a short one-line weather text. Uses openmeteo_requests if installed, else plain requests."""
    try:
        import openmeteo_requests  # type: ignore
        import requests_cache  # type: ignore
        from retry_requests import retry  # type: ignore

        cache_session = requests_cache.CachedSession(".cache", expire_after=3600)
        retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
        client = openmeteo_requests.Client(session=retry_session)
        url = "https://api.open-meteo.com/v1/forecast"
        params = {"latitude": lat, "longitude": lon, "hourly": "temperature_2m"}
        responses = client.weather_api(url, params=params)
        r = responses[0]
        hourly = r.Hourly()
        temp_c = float(hourly.Variables(0).ValuesAsNumpy()[0])
        return f"Current model temperature near {lat:.4f},{lon:.4f}: {temp_c:.1f}°C"
    except Exception:
        # Lightweight fallback
        import requests

        url = "https://api.open-meteo.com/v1/forecast"
        resp = requests.get(url, params={"latitude": lat, "longitude": lon, "hourly": "temperature_2m"}, timeout=10)
        j = resp.json()
        try:
            temp_c = float(j["hourly"]["temperature_2m"][0])
            return f"Current model temperature near {lat:.4f},{lon:.4f}: {temp_c:.1f}°C"
        except Exception:
            return "Weather service is temporarily unavailable."


# --------------------------------------------------------------------------------------
# Pydantic models
# --------------------------------------------------------------------------------------
class ChatMessage(BaseModel):
    message: str
    session_id: Optional[str] = None
    use_web_search: Optional[bool] = False


class PrescriptionQuery(BaseModel):
    query: Optional[str] = None
    session_id: Optional[str] = None


class MedicineSearch(BaseModel):
    condition: str
    session_id: Optional[str] = None


class SessionRequest(BaseModel):
    user_id: Optional[str] = None


# --------------------------------------------------------------------------------------
# UI route
# --------------------------------------------------------------------------------------
@router.get("/chat", response_class=HTMLResponse)
async def chat_interface(request: Request):
    return templates.TemplateResponse("chat_enhanced.html", {"request": request})


# --------------------------------------------------------------------------------------
# Sessions
# --------------------------------------------------------------------------------------
@router.post("/session/create")
async def create_session(req: SessionRequest):
    res = get_mcp().initialize_session(req.user_id)
    return res


@router.get("/session/{session_id}/context")
async def get_session_context(session_id: str):
    return get_mcp().get_conversation_context(session_id)


@router.get("/session/{session_id}/history")
async def get_session_history(session_id: str):
    return get_mcp().get_user_medical_history(session_id)


# --------------------------------------------------------------------------------------
# Chat
# --------------------------------------------------------------------------------------
@router.post("/chat")
async def chat_with_bot(payload: ChatMessage):
    try:
        # session
        session_id = payload.session_id or get_mcp().initialize_session().get("session_id")

        # record user message (both memories)
        get_mcp().add_user_message(session_id, payload.message)
        add_to_conversation_memory(session_id, "User", payload.message)

        # quick weather hook (simple trigger)
        lower_msg = payload.message.lower()
        if "weather" in lower_msg:
            # try to grab "lat,lon" from text; else use a sensible default (NYC)
            m = re.search(r"(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)", payload.message)
            lat, lon = (40.73061, -73.935242)
            if m:
                try:
                    lat, lon = float(m.group(1)), float(m.group(2))
                except Exception:
                    pass
            wx = _get_weather_text(lat, lon)
            add_to_conversation_memory(session_id, "Assistant", wx)
            get_mcp().add_assistant_response(session_id, wx)
            return {"response": wx, "session_id": session_id, "sources": {"weather": "open-meteo"}, "status": "success"}

        # fast memory answer?
        transcript = get_conversation_memory_context(session_id)
        mem = perfect_memory.answer_if_memory(payload.message, session_id, transcript)
        if mem:
            add_to_conversation_memory(session_id, "Assistant", mem)
            get_mcp().add_assistant_response(session_id, mem)
            return {
                "response": mem,
                "session_id": session_id,
                "sources": {"memory": "pattern"},
                "status": "success",
            }

        # augment context with last uploaded PDF / Image OCR if the user refers to "this pdf/image"
        extra_blocks: List[str] = []
        try:
            sess_ctx = get_mcp().memory.active_sessions.get(session_id, {}).get("context", {})
            if ("this pdf" in lower_msg or "the pdf" in lower_msg or "that pdf" in lower_msg or "pdf" in lower_msg) and sess_ctx.get(
                "last_pdf"
            ):
                extra_blocks.append(f"ATTACHED_PDF_TEXT:\n{sess_ctx['last_pdf']}")
            if ("this image" in lower_msg or "the image" in lower_msg or "that image" in lower_msg or "image" in lower_msg) and sess_ctx.get(
                "last_image_ocr"
            ):
                extra_blocks.append(f"ATTACHED_IMAGE_OCR:\n{sess_ctx['last_image_ocr']}")
        except Exception:
            pass

        augmented_context = transcript + ("\n\n" + "\n\n".join(extra_blocks) if extra_blocks else "")

        # full LLM handling (RAG + conversation context)
        resp = get_medical_agent().handle_text_query(
            payload.message,
            session_id=session_id,
            conversation_context=augmented_context,
            use_web_search=payload.use_web_search,
        )

        if resp.get("status") == "success":
            text = resp.get("response", "")
            add_to_conversation_memory(session_id, "Assistant", text)
            get_mcp().add_assistant_response(session_id, text)
            get_mcp().record_medical_query(session_id, payload.message, text)
            return {
                "response": text,
                "session_id": session_id,
                "sources": resp.get("sources", {}),
                "status": "success",
            }

        err = f"Error: {resp.get('error', 'Unknown error')}"
        get_mcp().add_assistant_response(session_id, err)
        return {"response": err, "session_id": session_id, "status": "error"}

    except Exception as e:
        logger.exception("Chat error")
        raise HTTPException(status_code=500, detail=str(e))


# --------------------------------------------------------------------------------------
# Prescription image
# --------------------------------------------------------------------------------------
@router.post("/prescription/upload")
async def upload_prescription(
    file: UploadFile = File(...),
    query: Optional[str] = Form(None),
    image_type: Optional[str] = Form("prescription"),
    session_id: Optional[str] = Form(None),
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    session_id = session_id or get_mcp().initialize_session().get("session_id")

    img_bytes = await file.read()
    Image.open(io.BytesIO(img_bytes))  # validates image

    # store user action
    get_mcp().add_user_message(session_id, f"Uploaded image: {file.filename}", "image", {"filename": file.filename})

    # type
    image_type = image_type or ("prescription" if "prescription" in (query or "").lower() else "general")

    result = get_medical_agent().handle_image_query(img_bytes, query, image_type)

    if result.get("status") == "success":
        analysis = result.get("prescription_analysis") or result.get("analysis", "Image analyzed.")
        get_mcp().record_prescription_analysis(session_id, result)
        get_mcp().add_assistant_response(session_id, analysis, "image_analysis_response")
        add_to_conversation_memory(session_id, "Assistant", analysis)

        # Save last OCR text for follow-up ("this image") questions
        try:
            sess = get_mcp().memory.active_sessions.get(session_id, {})
            ocr_text = result.get("ocr_results", {}).get("raw_text", "")
            sess.setdefault("context", {})["last_image_ocr"] = (ocr_text or "")[:4000]
            get_mcp().memory._save_session(session_id)
        except Exception:
            pass

        return {"response": analysis, "session_id": session_id, "ocr_results": result.get("ocr_results", {}), "status": "success"}

    error_message = f"Error analyzing image: {result.get('error', 'Unknown error')}"
    get_mcp().add_assistant_response(session_id, error_message)
    raise HTTPException(status_code=500, detail=error_message)


# --- Compatibility endpoint so the UI can call /image/analyze or /prescription/upload ----
@router.post("/image/analyze")
async def analyze_image_compat(
    file: UploadFile = File(...),
    session_id: Optional[str] = Form(None),
    question: Optional[str] = Form(None),
):
    # Delegate to the prescription handler with image_type="general"
    return await upload_prescription(file=file, query=question, image_type="general", session_id=session_id)


# --------------------------------------------------------------------------------------
# Prescription (text)
# --------------------------------------------------------------------------------------
@router.post("/prescription/analyze")
async def analyze_prescription_text(req: PrescriptionQuery):
    if not req.query:
        raise HTTPException(status_code=400, detail="Query is required")
    session_id = req.session_id or get_mcp().initialize_session().get("session_id")
    get_mcp().add_user_message(session_id, f"Prescription query: {req.query}")

    result = get_medical_agent().rag_agent.analyze_prescription_query(req.query)
    if result.get("status") == "success":
        text = result.get("response", "Analysis completed.")
        get_mcp().add_assistant_response(session_id, text)
        get_mcp().record_medical_query(session_id, req.query, text, "prescription")
        add_to_conversation_memory(session_id, "Assistant", text)
        return {"response": text, "session_id": session_id, "status": "success"}

    err = f"Error: {result.get('error', 'Unknown error')}"
    get_mcp().add_assistant_response(session_id, err)
    return {"response": err, "session_id": session_id, "status": "error"}


# --------------------------------------------------------------------------------------
# PDF: analyze (NO indexing). If no query -> default summary
# --------------------------------------------------------------------------------------
@router.post("/pdf/analyze")
async def analyze_pdf(
    file: UploadFile = File(...),
    query: Optional[str] = Form(""),
    session_id: Optional[str] = Form(None),
):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="File must be a PDF")

    session_id = session_id or get_mcp().initialize_session().get("session_id")
    pdf_bytes = await file.read()
    tmp = _write_temp_bytes(pdf_bytes, f"analyze_{file.filename}")

    try:
        from langchain_community.document_loaders import PyPDFLoader

        loader = PyPDFLoader(str(tmp))
        docs = loader.load()
        full_text = "\n".join([d.page_content for d in docs])

        agent = get_medical_agent()
        if not query or not query.strip():
            # default summary
            prompt = (
                "Summarize the following PDF text for a clinician-friendly audience. "
                "Include title (if present), section highlights, and 3–5 key takeaways.\n\n"
                f"{full_text[:15000]}"
            )
        else:
            prompt = (
                f"Answer the question based ONLY on this PDF's content.\n\n"
                f"QUESTION: {query}\n\nPDF TEXT:\n{full_text[:15000]}"
            )

        resp = agent.llm.generate_text_response(
            prompt,
            "You are a helpful AI assistant that analyzes documents and answers clearly.",
        )

        # remember in conversation + MCP
        get_mcp().add_user_message(
            session_id,
            f"Analyzed PDF: {file.filename} ({'summary' if not query.strip() else 'Q&A'})",
            "pdf",
            {"filename": file.filename},
        )
        get_mcp().add_assistant_response(session_id, resp)
        add_to_conversation_memory(session_id, "Assistant", resp)

        # store last PDF text to enable “this pdf” follow-ups
        try:
            sess = get_mcp().memory.active_sessions.get(session_id, {})
            sess.setdefault("context", {})["last_pdf"] = (full_text or "")[:4000]
            get_mcp().memory._save_session(session_id)
        except Exception:
            pass

        return {
            "response": resp,
            "session_id": session_id,
            "filename": file.filename,
            "query_type": "default_summary" if not query.strip() else "question",
            "status": "success",
        }
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass


# --------------------------------------------------------------------------------------
# PDF: upload into KB (indexing in Pinecone)
# --------------------------------------------------------------------------------------
@router.post("/pdf/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    description: Optional[str] = Form(None),
    session_id: Optional[str] = Form(None),
):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="File must be a PDF")

    if not PINECONE_API_KEY:
        raise HTTPException(status_code=500, detail="PINECONE_API_KEY not configured")

    session_id = session_id or get_mcp().initialize_session().get("session_id")
    pdf_bytes = await file.read()
    tmp = _write_temp_bytes(pdf_bytes, file.filename)

    try:
        from langchain_community.document_loaders import PyPDFLoader
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from core.vector_store import PineconeStore
        from config.env_config import PINECONE_PDF_INDEX

        loader = PyPDFLoader(str(tmp))
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(documents)

        texts, metas = [], []
        for i, d in enumerate(chunks):
            texts.append(d.page_content)
            metas.append(
                {
                    "source": file.filename,
                    "description": description or f"PDF document: {file.filename}",
                    "chunk_id": i,
                    "page": d.metadata.get("page", 0),
                    "total_chunks": len(chunks),
                }
            )

        store = PineconeStore(index_name=PINECONE_PDF_INDEX, dimension=384)
        store.upsert_texts(texts, metas)

        msg = (
            f"Successfully processed '{file.filename}' and added "
            f"{len(texts)} chunks to the medical knowledge base."
        )

        get_mcp().add_user_message(
            session_id, f"Uploaded PDF to KB: {file.filename}", "pdf_upload", {"filename": file.filename}
        )
        get_mcp().add_assistant_response(session_id, msg, "pdf_processing_response")
        add_to_conversation_memory(session_id, "Assistant", msg)

        return {"message": msg, "session_id": session_id, "chunks_processed": len(texts), "status": "success"}
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass


# --------------------------------------------------------------------------------------
# Vector DB management
# --------------------------------------------------------------------------------------
@router.post("/vector/create-index")
async def create_vector_index(
    file: UploadFile = File(...),
    index_name: str = Form(...),
    description: Optional[str] = Form(None),
    session_id: Optional[str] = Form(None),
):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="File must be a PDF")

    from pinecone import Pinecone

    if not PINECONE_API_KEY:
        raise HTTPException(status_code=500, detail="PINECONE_API_KEY not configured")

    session_id = session_id or get_mcp().initialize_session().get("session_id")
    pdf_bytes = await file.read()
    tmp = _write_temp_bytes(pdf_bytes, file.filename)

    try:
        from langchain_community.document_loaders import PyPDFLoader
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from core.vector_store import PineconeStore

        loader = PyPDFLoader(str(tmp))
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(documents)

        texts, metas = [], []
        for i, d in enumerate(chunks):
            texts.append(d.page_content)
            metas.append(
                {
                    "source": file.filename,
                    "description": description or f"PDF document: {file.filename}",
                    "chunk_id": i,
                    "page": d.metadata.get("page", 0),
                    "total_chunks": len(chunks),
                    "index_name": index_name,
                }
            )

        formatted = f"arobot-medical-pdf-{index_name.lower().replace('_','-').replace(' ','-')}"
        store = PineconeStore(index_name=formatted, dimension=384)
        store.upsert_texts(texts, metas)

        msg = f"Created vector index '{formatted}' with {len(texts)} chunks."
        get_mcp().add_user_message(session_id, f"Created index {formatted}", "vector_index_creation")
        get_mcp().add_assistant_response(session_id, msg, "vector_index_response")
        add_to_conversation_memory(session_id, "Assistant", msg)

        return {"message": msg, "session_id": session_id, "index_name": formatted, "status": "success"}
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass


@router.get("/vector/indexes")
async def list_vector_indexes():
    try:
        from pinecone import Pinecone

        pc = Pinecone(api_key=PINECONE_API_KEY)
        out = []
        for idx in pc.list_indexes():
            try:
                conn = pc.Index(idx.name)
                stats = conn.describe_index_stats()
                total = (
                    getattr(stats, "total_vector_count", 0)
                    if hasattr(stats, "total_vector_count")
                    else (stats.get("total_vector_count", 0) if isinstance(stats, dict) else 0)
                )
                namespaces = getattr(stats, "namespaces", {}) if hasattr(stats, "namespaces") else stats.get("namespaces", {})
            except Exception:
                total, namespaces = 0, {}
            out.append(
                {
                    "name": idx.name,
                    "dimension": idx.dimension,
                    "metric": idx.metric,
                    "host": idx.host,
                    "status": "ready" if idx.status.ready else "not_ready",
                    "total_vector_count": total,
                    "namespaces": len(namespaces) if namespaces else 0,
                }
            )
        return {"indexes": out, "total_count": len(out), "status": "success"}
    except Exception as e:
        return {"error": str(e), "status": "error"}


# --------------------------------------------------------------------------------------
# Medicine search
# --------------------------------------------------------------------------------------
@router.post("/search/medicine")
async def search_medicine_by_condition(req: MedicineSearch):
    session_id = req.session_id or get_mcp().initialize_session().get("session_id")
    get_mcp().add_user_message(session_id, f"Search medicines for: {req.condition}")
    result = get_medical_agent().search_medicine_by_condition(req.condition)
    if result.get("status") == "success":
        text = result.get("response", "")
        get_mcp().add_assistant_response(session_id, text)
        get_mcp().record_medical_query(session_id, req.condition, text, "medicine_search")
        add_to_conversation_memory(session_id, "Assistant", text)
        return {"response": text, "condition": req.condition, "session_id": session_id, "status": "success"}
    err = f"Error: {result.get('error', 'Unknown error')}"
    get_mcp().add_assistant_response(session_id, err)
    return {"response": err, "session_id": session_id, "status": "error"}


# --------------------------------------------------------------------------------------
# Weather (Open-Meteo) – JSON endpoint
# --------------------------------------------------------------------------------------
@router.get("/weather")
async def weather(lat: float, lon: float):
    try:
        try:
            import openmeteo_requests  # type: ignore
            import requests_cache  # type: ignore
            from retry_requests import retry  # type: ignore

            cache_session = requests_cache.CachedSession(".cache", expire_after=3600)
            retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
            client = openmeteo_requests.Client(session=retry_session)
            url = "https://api.open-meteo.com/v1/forecast"
            params = {"latitude": lat, "longitude": lon, "hourly": "temperature_2m"}
            responses = client.weather_api(url, params=params)
            r = responses[0]
            hourly = r.Hourly()
            temps = list(hourly.Variables(0).ValuesAsNumpy())
            return {
                "coords": [r.Latitude(), r.Longitude()],
                "elevation": r.Elevation(),
                "utc_offset_seconds": r.UtcOffsetSeconds(),
                "hourly_temp_2m_c": temps[:24],
                "status": "success",
            }
        except Exception:
            # Fallback plain HTTP (no cache/retry libs required)
            import requests

            url = "https://api.open-meteo.com/v1/forecast"
            resp = requests.get(url, params={"latitude": lat, "longitude": lon, "hourly": "temperature_2m"}, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            return {"raw": data, "status": "success"}
    except Exception as e:
        return {"error": str(e), "status": "error"}


# --------------------------------------------------------------------------------------
# System status and KB stats passthroughs
# --------------------------------------------------------------------------------------
@router.get("/system/status")
async def get_system_status():
    return get_medical_agent().get_system_status()


@router.get("/knowledge/stats")
async def get_knowledge_base_stats():
    return get_medical_agent().rag_agent.get_knowledge_base_stats()
