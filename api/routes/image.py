# api/routes/image.py
from __future__ import annotations

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import Optional
from PIL import Image, UnidentifiedImageError
import io

from core.llm_handler import LLMHandler

router = APIRouter()
_handler = LLMHandler()


def _load_pil_or_400(data: bytes) -> Image.Image:
    """Open bytes as PIL.Image or raise 400 if not an image."""
    try:
        im = Image.open(io.BytesIO(data))
        im.load()  # force decode
        return im
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid image.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read image: {e}")


@router.post("/api/v1/image/analyze")
async def analyze_image(
    file: UploadFile = File(...),
    question: Optional[str] = Form(None),
    image_type: Optional[str] = Form("prescription"),
    # Optional CLIP controls
    use_clip: Optional[bool] = Form(False),
    clip_query: Optional[str] = Form(None),
    clip_namespace: Optional[str] = Form("anatomy"),
    clip_top_k: Optional[int] = Form(6),
):
    """
    - prescription: OCR-first structured extraction with human-friendly summary in `response`.
      Full structured JSON is returned in `structured`.
    - other images: brief description (no question) or short vision answer (with question).
    - CLIP: set use_clip=true for image→image; or pass clip_query="text" for text→image.
    """
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty upload.")

    try:
        if (image_type or "").lower() == "prescription":
            res = _handler.analyze_prescription(image_data=data)

            chat_text = res.get("summary") or res.get("vision_analysis") or "Analyzed the prescription."

            if question:
                ocr_text = ((res.get("ocr_results") or {}).get("raw_text", "") or "").strip()
                if ocr_text:
                    follow = _handler.answer_over_ocr_text(question, ocr_text)
                    if follow and follow.strip() != "Can't find it in the OCR.":
                        chat_text = f"{chat_text}\n\n**Follow-up:** {follow}"
                    elif follow:
                        # Add a gentle, non-contradictory note when OCR lacks the answer
                        chat_text = f"{chat_text}\n\n(Unable to locate that detail reliably in the OCR text.)"

            res["response"] = chat_text

            # Optional CLIP search
            if use_clip or clip_query:
                clip_block = {}
                try:
                    if clip_query and clip_query.strip():
                        clip_block["by_text"] = _handler.image_index.query_by_text(
                            text=clip_query.strip(),
                            top_k=int(clip_top_k or 6),
                            namespace=clip_namespace or "anatomy",
                        )
                    if use_clip:
                        pil = _load_pil_or_400(data)
                        clip_block["by_image"] = _handler.image_index.query_by_image(
                            img=pil,
                            top_k=int(clip_top_k or 6),
                            namespace=clip_namespace or "anatomy",
                        )
                except Exception:
                    clip_block["error"] = "CLIP index unavailable or disabled."
                res["clip_matches"] = clip_block

            return res

        # Non-prescription images
        if question:
            answer = _handler.generate_vision_response(question, image_data=data)
            out = {"response": (answer or "I couldn't read enough to answer."), "status": "success", "filename": file.filename}
        else:
            brief = _handler.default_image_brief(data)
            out = {"response": brief, "status": "success", "filename": file.filename}

        if use_clip or clip_query:
            clip_block = {}
            try:
                if clip_query and clip_query.strip():
                    clip_block["by_text"] = _handler.image_index.query_by_text(
                        text=clip_query.strip(),
                        top_k=int(clip_top_k or 6),
                        namespace=clip_namespace or "anatomy",
                    )
                if use_clip:
                    pil = _load_pil_or_400(data)
                    clip_block["by_image"] = _handler.image_index.query_by_image(
                        img=pil,
                        top_k=int(clip_top_k or 6),
                        namespace=clip_namespace or "anatomy",
                    )
            except Exception:
                clip_block["error"] = "CLIP index unavailable or disabled."
            out["clip_matches"] = clip_block

        return out

    except HTTPException:
        raise
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image processing failed: {e}")
