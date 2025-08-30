"""
Web search utility using DuckDuckGo (ddgs) + BD medicine resolvers
"""
from __future__ import annotations
from typing import Dict, Any, List
import logging
from langsmith import traceable

try:
    from ddgs import DDGS  # preferred (new)
except Exception:          # pragma: no cover
    try:
        from duckduckgo_search import DDGS  # fallback (old)
    except Exception:
        DDGS = None  # last resort

from pharma.resolver import get_price_bd, resolve_bd_medicine

logger = logging.getLogger(__name__)

def _domain(url: str) -> str:
    try:
        return url.split("/")[2] if url.startswith("http") else "unknown"
    except Exception:
        return "unknown"


class WebSearchTool:
    """DuckDuckGo-based search + BD medex helpers."""

    def __init__(self) -> None:
        self.ddgs = None
        try:
            if DDGS is not None:
                self.ddgs = DDGS()
        except Exception as e:  # pragma: no cover
            logger.warning(f"DDGS init failed: {e}")
            self.ddgs = None

    @traceable(name="web_search")
    def search_medical_info(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """Search for medical info; returns normalized results."""
        if not self.ddgs:
            return {"status": "error", "error": "ddgs unavailable"}
        try:
            medical_query = f"medical {query} health information"
            raw = self.ddgs.text(medical_query, max_results=max_results, safesearch="moderate")
            results = []
            for r in raw:
                title = r.get("title", "")
                href  = r.get("href", r.get("url", ""))
                body  = r.get("body", r.get("snippet", ""))
                results.append({
                    "title": title,
                    "snippet": body,
                    "url": href,
                    "source": _domain(href),
                })
            return {"query": query, "results": results, "result_count": len(results), "status": "success"}
        except Exception as e:
            logger.error(f"web_search error: {e}")
            return {"status": "error", "error": str(e)}

    @traceable(name="search_medicine_info")
    def search_medicine_info(self, medicine_name: str) -> Dict[str, Any]:
        try:
            q = f"{medicine_name} drug information uses dosing side effects"
            return self.search_medical_info(q, max_results=3)
        except Exception as e:
            logger.error(f"search_medicine_info error: {e}")
            return {"status": "error", "error": str(e)}

    @traceable(name="search_disease_info")
    def search_disease_info(self, disease: str) -> Dict[str, Any]:
        try:
            q = f"{disease} symptoms diagnosis treatment"
            return self.search_medical_info(q, max_results=5)
        except Exception as e:
            logger.error(f"search_disease_info error: {e}")
            return {"status": "error", "error": str(e)}

    @traceable(name="search_medical_news")
    def search_medical_news(self, topic: str) -> Dict[str, Any]:
        if not self.ddgs:
            return {"status": "error", "error": "ddgs unavailable"}
        try:
            q = f"{topic} medical news recent research"
            try:
                raw = self.ddgs.news(q, max_results=5)
            except Exception:
                raw = self.ddgs.text(q, max_results=5, safesearch="moderate")
            results = []
            for r in raw:
                href = r.get("url", r.get("href", ""))
                results.append({
                    "title": r.get("title", ""),
                    "snippet": r.get("body", r.get("snippet", "")),
                    "url": href,
                    "date": r.get("date", ""),
                    "source": r.get("source", _domain(href))
                })
            return {"query": topic, "results": results, "result_count": len(results), "status": "success", "type": "news"}
        except Exception as e:
            logger.error(f"search_medical_news error: {e}")
            return {"status": "error", "error": str(e)}

    # ---- BD medex wrappers (used by LLMHandler.answer_medicine) ----
    def get_bd_medicine_price(self, name: str) -> Dict[str, Any]:
        try:
            price = get_price_bd(name)
            if price:
                return {
                    "status": "success",
                    "brand": price.get("brand"),
                    "price": price.get("price_text"),
                    "url": price.get("url"),
                    "source": price.get("source", "medex"),
                    "cached": price.get("cached", False),
                }
            return {"status": "error", "query": name}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def resolve_bd_medicine(self, name: str) -> Dict[str, Any]:
        try:
            rec = resolve_bd_medicine(name)
            if rec:
                return {"status": "success", **rec}
            return {"status": "error", "query": name}
        except Exception as e:
            return {"status": "error", "error": str(e)}
