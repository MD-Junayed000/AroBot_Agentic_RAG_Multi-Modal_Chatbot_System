from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import Optional
from core.llm_handler import LLMHandler

router = APIRouter(prefix="/api/v1/image", tags=["image"])
_handler = LLMHandler()

def _ocr_is_weak(t: str) -> bool:
    if not t:
        return True
    t = t.strip()
    if len(t) < 120:
        return True
    alpha = sum(1 for c in t if c.isalpha() or c.isspace() or c in ".-:,/()")
    return (alpha / max(len(t), 1)) < 0.6


@router.post("/analyze")
async def analyze_image(
    file: Optional[UploadFile] = File(None, description="Send as 'file'"),
    image: Optional[UploadFile] = File(None, description="Send as 'image'"),
    question: Optional[str] = Form(None),
):
    upload: Optional[UploadFile] = image or file
    if upload is None:
        raise HTTPException(status_code=400, detail="No file provided. Send as form field 'image' or 'file'.")

    data = await upload.read()

    # OCR-first
    ox = _handler.ocr_only(data)
    ocr_text = ox.get("raw_text", "") if isinstance(ox, dict) else ""

    # No question -> structured parse (vision if OCR weak)
    if not question:
        if _ocr_is_weak(ocr_text):
            vprompt = (
                "Read the prescription image and extract ONLY:\n"
                "- patient_name (if visible)\n- doctor\n- clinic\n- diagnosis (if visible)\n"
                "- medications: list with name, strength, dose, frequency, duration\n"
                "Return JSON only, no commentary."
            )
            v = _handler.generate_vision_response(vprompt, image_data=data)
            return {"mode": "vision_due_to_weak_ocr", "filename": upload.filename, "vision_analysis": v, "status": "success"}
        parsed = _handler.analyze_prescription(image_data=data, ocr_text=ocr_text)
        return {"mode": "ocr_first", "filename": upload.filename, **parsed}

    # With a question:
    ql = (question or "").lower()

    # Prefer OCR QA when OCR is decent
    if not _ocr_is_weak(ocr_text):
        ans = _handler.answer_over_ocr_text(question, ocr_text)
        if ans.strip().lower().startswith("can't find"):
            # fallback to vision if OCR QA couldn't locate it
            ans = _handler.vision_short_answer(question, image_data=data)
            return {"mode": "ocr_then_vision_fallback", "filename": upload.filename, "answer": ans, "ocr_text_len": len(ocr_text)}
        return {"mode": "ocr_qa", "filename": upload.filename, "answer": ans, "ocr_text_len": len(ocr_text)}

    # OCR is weak → use targeted vision QA prompts
    if "doctor" in ql or "dr" in ql:
        ans = _handler.extract_doctor_from_vision(image_data=data)
        return {"mode": "vision_doctor_name", "filename": upload.filename, "answer": ans}

    if "medicine" in ql or "drug" in ql or "tablet" in ql:
        ans = _handler.extract_medicines_from_vision(image_data=data)
        return {"mode": "vision_medicines", "filename": upload.filename, "answer": ans}

    # generic vision answer (still short)
    v = _handler.vision_short_answer(question, image_data=data)
    return {"mode": "vision_fallback", "filename": upload.filename, "answer": v}
