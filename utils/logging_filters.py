import re
RE_ID = re.compile(r"\b(\d[\d\- ]{8,})\b")

def redact(s: str) -> str:
    return RE_ID.sub("[REDACTED]", s)
