import os, re, cv2, json, numpy as np, pandas as pd
from pathlib import Path
from paddleocr import PaddleOCR
from rapidfuzz import process, fuzz
from unidecode import unidecode

ABBREV = {
    "OD":"once daily","BD":"twice daily","TDS":"thrice daily",
    "QID":"four times daily","HS":"at bedtime","SOS":"as needed","STAT":"immediately","MR":"modified release"
}
FREQ_PAT = r"(once daily|twice daily|thrice daily|four times daily|as needed|at bedtime|immediately)"
DUR_PAT  = r"(\d+\\s*(day|days|week|weeks|month|months))"
STR_PAT  = r"(\\d{2,4})\\s*(mg|mcg|g|ml)"

def deskew(gray):
    thr = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    inv = 255-thr
    coords = np.column_stack(np.where(inv>0))
    if len(coords)==0: return gray
    angle = cv2.minAreaRect(coords)[-1]
    angle = -(90+angle) if angle<-45 else -angle
    h,w = gray.shape[:2]
    M = cv2.getRotationMatrix2D((w//2,h//2), angle, 1.0)
    return cv2.warpAffine(gray, M, (w,h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

def preprocess(path:str):
    img = cv2.imread(path)
    if img is None: raise ValueError(f"Cannot read {path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = deskew(gray)
    gray = cv2.fastNlMeansDenoising(gray,None,15,7,21)
    clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
    gray = clahe.apply(gray)
    bw = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,25,12)
    return bw

def normalize_line(s:str):
    s = unidecode(s)
    s = s.replace("|"," ").replace("_"," ").replace("~"," ")
    s = " ".join(s.split())
    for k,v in ABBREV.items():
        s = re.sub(rf"\\b{k}\\b", v, s, flags=re.IGNORECASE)
    return s

def parse_line(line:str):
    L = normalize_line(line.lower())
    strength, unit = None, None
    m = re.search(STR_PAT, L)
    if m: strength, unit = m.group(1), m.group(2)
    freq = (re.search(FREQ_PAT,L).group(1) if re.search(FREQ_PAT,L) else None)
    duration = (re.search(DUR_PAT,L).group(1) if re.search(DUR_PAT,L) else None)
    return {"raw": line.strip(), "strength": strength, "unit": unit, "frequency": freq, "duration": duration}

class OCRPipeline:
    def __init__(self, lang='en'):
        self.lang = lang
        self.ocr = None  # Lazy initialization
    
    def _init_ocr(self):
        if self.ocr is None:
            print("🔄 Initializing PaddleOCR (optimized for speed)...")
            self.ocr = PaddleOCR(
                lang=self.lang, 
                use_angle_cls=False,  # Disable angle classification for speed
                use_space_char=True,
                drop_score=0.3,  # Lower threshold for faster processing
                use_gpu=True  # Enable GPU if available
            )
            print("✅ PaddleOCR initialized with speed optimizations")

    def run_on_image(self, path: str):
        """Process image with OCR and handle errors gracefully"""
        try:
            self._init_ocr()  # Initialize OCR only when needed
            print(f"🔍 Processing image: {path}")
            
            # Try preprocessed image first
            bw = preprocess(path)
            res = self.ocr.ocr(bw, cls=True)
            
            # If no results, try original image
            if not res or not any(res):
                print("⚠️  Trying original image...")
                res = self.ocr.ocr(path, cls=True)
            
            if not res or not any(res):
                print("❌ No text detected in image")
                return {"error": "No text detected in image", "lines": []}
                
        except Exception as e:
            print(f"❌ OCR Error: {e}")
            return {"error": f"OCR processing failed: {str(e)}", "lines": []}
        lines = []
        if res:
            for page in res:
                if page:  # Check if page has content
                    for (box, txt, conf) in page:
                        if txt and txt[0].strip() and conf > 0.5:  # Add confidence threshold
                            lines.append(txt[0])
        
        print(f"✅ OCR completed. Extracted {len(lines)} text lines")
        items = [parse_line(x) for x in lines if x.strip()]
        return lines, items

    def run_folder(self, folder:str, out_csv='outputs/prescription_ocr.csv'):
        rows=[]
        for fn in sorted(os.listdir(folder)):
            if fn.lower().endswith(('.jpg','.jpeg','.png')):
                full = str(Path(folder)/fn)
                lines, items = self.run_on_image(full)
                rows.append({"image": fn, "raw_text": "\\n".join(lines), "items": json.dumps(items, ensure_ascii=False)})
        df = pd.DataFrame(rows)
        df.to_csv(out_csv, index=False)
        return df
