# utils/ocr_pipeline.py
from __future__ import annotations

import os
import re
import cv2
import json
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import threading

try:
    from config.env_config import OCR_CONFIDENCE_THRESHOLD  # float 0..1
except Exception:
    OCR_CONFIDENCE_THRESHOLD = 0.5

try:
    # paddleocr versions differ on accepted kwargs; we will handle gracefully
    from paddleocr import PaddleOCR  # pip install "paddleocr>=2.7.0.3"
except Exception:
    PaddleOCR = None

from unidecode import unidecode

# ------------------------- Heuristics & regexes ---------------------------- #
ABBREV = {
    "OD": "once daily",
    "BD": "twice daily",
    "TDS": "thrice daily",
    "QID": "four times daily",
    "HS": "at bedtime",
    "SOS": "as needed",
    "STAT": "immediately",
    "MR": "modified release",
}

FREQ_PAT = r"(once daily|twice daily|thrice daily|four times daily|as needed|at bedtime|immediately)"
DUR_PAT  = r"(\d+\s*(day|days|week|weeks|month|months))"
STR_PAT  = r"(\d{1,4})\s*(mg|mcg|g|ml)"
# e.g., 1-0-1, 0-1-0, 1-1-1, 1 x 5 days
DOSE_PATTERN_RE = re.compile(r"\b\d(?:\s*[-xX]\s*\d){1,3}\b")

DOCTOR_RX_RE = re.compile(r"\b(Dr\.?\s*[A-Z][\w.\-' ]{1,40})\b", re.IGNORECASE)

# ------------------------------ Preprocessing ------------------------------ #
def _deskew(gray: np.ndarray) -> np.ndarray:
    thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    inv = 255 - thr
    coords = np.column_stack(np.where(inv > 0))
    if coords.size == 0:
        return gray
    angle = cv2.minAreaRect(coords)[-1]
    angle = -(90 + angle) if angle < -45 else -angle
    h, w = gray.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

def _preprocess_bgr(bgr: np.ndarray) -> np.ndarray:
    """
    Tiny header-friendly preprocessor:
      1) grayscale
      2) denoise (fastNlMeans)
      3) CLAHE contrast boost
      4) 2x upscale (INTER_CUBIC) to help tiny fonts
      5) deskew
      6) adaptive threshold (Gaussian)
    Returns a binarized image well-suited for OCR.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # mild denoise (helps JPEG fuzz on letterheads)
    gray = cv2.fastNlMeansDenoising(gray, None, 12, 7, 21)

    # local contrast boost
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # upscale 2x to make small caps readable
    h, w = gray.shape[:2]
    gray = cv2.resize(gray, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)

    # correct skew after upscaling
    gray = _deskew(gray)

    # robust binarization
    bw = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 10
    )
    return bw

def _bytes_to_bgr(image_bytes: bytes) -> Optional[np.ndarray]:
    arr = np.frombuffer(image_bytes, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

# ------------------------------ Normalizers -------------------------------- #
def normalize_line(s: str) -> str:
    s = unidecode(s)
    s = s.replace("|", " ").replace("_", " ").replace("~", " ")
    s = " ".join(s.split())
    for k, v in ABBREV.items():
        s = re.sub(rf"\b{k}\b", v, s, flags=re.IGNORECASE)
    return s

def parse_line(line: str) -> Dict[str, Any]:
    L = normalize_line(line.lower())
    strength, unit = None, None
    m = re.search(STR_PAT, L)
    if m:
        strength, unit = m.group(1), m.group(2)
    freq     = (re.search(FREQ_PAT, L).group(1) if re.search(FREQ_PAT, L) else None)
    duration = (re.search(DUR_PAT,  L).group(1) if re.search(DUR_PAT,  L) else None)
    dose_pat = DOSE_PATTERN_RE.search(line)
    dose_pattern = dose_pat.group(0) if dose_pat else None
    return {"raw": line.strip(), "strength": strength, "unit": unit, "frequency": freq, "duration": duration, "dose_pattern": dose_pattern}

def extract_header_entities(lines: List[str]) -> Dict[str, Any]:
    """Pull doctor/clinic hints from the top 10 lines (letterhead & headers)."""
    header = "  ".join(lines[:10])
    m = DOCTOR_RX_RE.search(header)
    doctor = m.group(1).strip() if m else None
    clinic = None
    for tag in ["clinic", "centre", "center", "hospital", "medical", "health", "diagnostic"]:
        m2 = re.search(rf"\b([\w .,&'-]{{3,}}{tag}[\w .,&'-]{{0,}})\b", header, re.IGNORECASE)
        if m2:
            clinic = m2.group(1).strip()
            break
    return {"doctor": doctor, "clinic": clinic}

# ------------------------------ OCR wrapper -------------------------------- #
PADDLE_OCR_SINGLETON = {"obj": None}  # simple in-proc singleton

class OCRPipeline:
    """
    Robust OCR wrapper around PaddleOCR:
      - Avoids unsupported args across versions,
      - Tolerates different result shapes,
      - Returns raw lines + parsed items + header entities.
    """
    def __init__(self, lang: str = "en"):
        self.lang = lang
        self.ocr  = None  # lazy init

    def _init_ocr(self):
        """Robust PaddleOCR init with fallbacks. Caches singleton to avoid re-downloads."""
        if self.ocr is not None:
            return
        if PADDLE_OCR_SINGLETON["obj"] is not None:
            self.ocr = PADDLE_OCR_SINGLETON["obj"]
            return

        if PaddleOCR is None:
            raise RuntimeError("PaddleOCR is not installed. Please `pip install paddleocr`.")

        attempts = (
            {"lang": self.lang, "use_angle_cls": False, "show_log": False},
            {"lang": self.lang, "use_angle_cls": False},
            {"lang": self.lang},                      # minimal
            {"lang": self.lang, "use_angle_cls": False, "use_gpu": False},  # Windows CPU fallback
        )
        errors = []
        for kw in attempts:
            try:
                self.ocr = PaddleOCR(**kw)
                PADDLE_OCR_SINGLETON["obj"] = self.ocr
                return
            except Exception as e:
                errors.append(f"{kw} -> {e!r}")

        raise RuntimeError("PaddleOCR init failed with all fallbacks: " + " | ".join(errors))

    def _extract_lines(self, ocr_result: Any) -> List[str]:
        lines: List[str] = []
        if not ocr_result:
            return lines
        pages = ocr_result if isinstance(ocr_result, list) else [ocr_result]
        for page in pages:
            if not page:
                continue
            for item in page:
                try:
                    _, content = item
                    if isinstance(content, (list, tuple)) and len(content) >= 2:
                        text, score = content[0], float(content[1])
                    else:
                        continue
                except Exception:
                    try:
                        text = item[1][0]
                        score = float(item[1][1])
                    except Exception:
                        continue
                if text and float(score) >= float(OCR_CONFIDENCE_THRESHOLD):
                    lines.append(str(text))
        return lines

    def _ocr_on_array(self, arr: np.ndarray) -> Tuple[List[str], List[Dict[str, Any]], Dict[str, Any]]:
        try:
            self._init_ocr()
            bw = _preprocess_bgr(arr)
            res = self.ocr.ocr(bw, cls=False)
            lines = self._extract_lines(res)
            if not lines:
                # fallback to original color if binarized failed
                res = self.ocr.ocr(arr, cls=False)
                lines = self._extract_lines(res)
        except Exception as e:
            print(f"❌ OCR Error: {e}")
            return [], [], {}

        # dedupe + normalize
        out: List[str] = []
        seen = set()
        for ln in (l for l in lines if str(l).strip()):
            ln = normalize_line(ln)
            if ln and ln not in seen:
                seen.add(ln)
                out.append(ln)

        items = [parse_line(x) for x in out]
        header_entities = extract_header_entities(out)
        return out, items, header_entities

    # Public APIs
    def run_on_bytes(self, image_bytes: bytes) -> Tuple[List[str], List[Dict[str, Any]], Dict[str, Any]]:
        try:
            bgr = _bytes_to_bgr(image_bytes)
            if bgr is None:
                return [], [], {}
            return self._ocr_on_array(bgr)
        except Exception as e:
            print(f"❌ OCR bytes error: {e}")
            return [], [], {}

    def run_on_image(self, path: str) -> Tuple[List[str], List[Dict[str, Any]], Dict[str, Any]]:
        try:
            bgr = cv2.imread(path)
            if bgr is None:
                raise ValueError(f"Cannot read image: {path}")
            return self._ocr_on_array(bgr)
        except Exception as e:
            print(f"❌ OCR path error: {e}")
            return [], [], {}

    def run_folder(self, folder: str, out_csv: str = "outputs/prescription_ocr.csv"):
        import pandas as pd
        rows = []
        for fn in sorted(os.listdir(folder)):
            if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                full = str(Path(folder) / fn)
                lines, items, hdr = self.run_on_image(full)
                rows.append({
                    "image": fn,
                    "raw_text": "\n".join(lines),
                    "items": json.dumps(items, ensure_ascii=False),
                    "doctor": hdr.get("doctor"),
                    "clinic": hdr.get("clinic"),
                })
        df = pd.DataFrame(rows)
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
        return df

# Convenience helpers used by app startup and endpoints
def run_ocr(image_bytes: bytes, lang: str = "en") -> Dict[str, Any]:
    pipe = OCRPipeline(lang=lang)
    lines, items, hdr = pipe.run_on_bytes(image_bytes)
    return {
        "raw_text": "\n".join(lines),
        "lines": lines,
        "items": items,
        "item_count": len(items),
        "header": hdr
    }

def warmup_ocr(lang: str = "en") -> None:
    """Kick off OCR model download/init in a background thread at app startup."""
    def _worker():
        try:
            OCRPipeline(lang=lang)._init_ocr()
            print("✅ PaddleOCR warmed up.")
        except Exception as e:
            print(f"⚠️  OCR warmup failed: {e}")
    threading.Thread(target=_worker, daemon=True).start()
