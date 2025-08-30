"""
Conversation Memory for maintaining chat history and context
"""
from __future__ import annotations

import json
import uuid
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class ConversationMemory:
    """Manages conversation memory and context for users (persisted to /memory)"""

    def __init__(self, memory_dir: str = "memory"):
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(exist_ok=True)
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.max_memory_days = 30
        self.summary_chunk = 120  # how many old turns to summarize at once

    # --------- session primitives --------- #
    def create_session(self, user_id: Optional[str] = None) -> str:
        session_id = str(uuid.uuid4())
        user_id = user_id or f"user_{uuid.uuid4().hex[:8]}"
        data = {
            "session_id": session_id,
            "user_id": user_id,
            "created_at": datetime.now().isoformat(),
            "last_activity": datetime.now().isoformat(),
            "messages": [],
            "context": {},
            "prescription_history": [],
            "medical_queries": [],
            "long_term_summary": "",

        }
        self.active_sessions[session_id] = data
        self._save_session(session_id)
        return session_id

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        if session_id not in self.active_sessions:
            self._load_session(session_id)
        return self.active_sessions.get(session_id)

    # --------- message storage --------- #
    def add_message(
         self,
         session_id: str,
         role: str,
         content: str,
         message_type: str = "text",
         metadata: Optional[Dict[str, Any]] = None,
     ) -> bool:
         try:
             session = self.get_session(session_id)
             if not session:
                 logger.warning(f"Session {session_id} not found")
                 return False
             msg = {
                 "id": str(uuid.uuid4()),
                 "timestamp": datetime.now().isoformat(),
                 "role": role,
                 "content": content,
                 "type": message_type,
                 "metadata": metadata or {},
             }
             session["messages"].append(msg)
             session["last_activity"] = datetime.now().isoformat()
             # bound memory: compact older messages into a long-term summary instead of dropping
             if len(session["messages"]) > 120:
                 old = session["messages"][:-80]
                 keep = session["messages"][-80:]
                 # simple deterministic compact without LLM
                 summary_lines = []
                 for m in old:
                     role = m.get("role", "user")
                     text = (m.get("content") or "").strip().replace("\n", " ")
                     if text:
                         summary_lines.append(f"{role}: {text}")
                 prefix = (session.get("long_term_summary") or "").strip()
                 new_sum = ("; ".join(summary_lines))[:12000]
                 session["long_term_summary"] = ((prefix + " ; " if prefix else "") + new_sum)[:30000]
                 session["messages"] = keep

             # persist every write
             self._save_session(session_id)
             return True
         except Exception as e:
             logger.error(f"Error adding message: {e}")
             return False


    def get_conversation_history(self, session_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        try:
            s = self.get_session(session_id)
            if not s:
                return []
            msgs = s.get("messages", [])
            return msgs[-limit:] if limit else msgs
        except Exception as e:
            logger.error(f"Error getting conversation: {e}")
            return []

    # --------- prescription & queries --------- #
    def add_prescription_record(self, session_id: str, prescription_data: Dict[str, Any]) -> bool:
        try:
            s = self.get_session(session_id)
            if not s:
                return False
            rec = {"id": str(uuid.uuid4()), "timestamp": datetime.now().isoformat(), "prescription_data": prescription_data}
            s["prescription_history"].append(rec)
            if len(s["prescription_history"]) > 12:
                s["prescription_history"] = s["prescription_history"][-12:]
            self._save_session(session_id)
            return True
        except Exception as e:
            logger.error(f"Error adding prescription: {e}")
            return False

    def get_prescription_history(self, session_id: str) -> List[Dict[str, Any]]:
        try:
            s = self.get_session(session_id)
            if not s:
                return []
            return s.get("prescription_history", [])
        except Exception as e:
            logger.error(f"Error reading prescription history: {e}")
            return []

    def add_medical_query(self, session_id: str, query: str, response: str, query_type: str = "general") -> bool:
        try:
            s = self.get_session(session_id)
            if not s:
                return False
            rec = {
                "id": str(uuid.uuid4()),
                "timestamp": datetime.now().isoformat(),
                "query": query,
                "response": response,
                "type": query_type,
            }
            s["medical_queries"].append(rec)
            if len(s["medical_queries"]) > 40:
                s["medical_queries"] = s["medical_queries"][-40:]
            self._save_session(session_id)
            return True
        except Exception as e:
            logger.error(f"Error adding query: {e}")
            return False

    # --------- summaries --------- #
    def get_context_summary(self, session_id: str) -> Dict[str, Any]:
        try:
            s = self.get_session(session_id)
            if not s:
                return {}
            return {
                "session_id": session_id,
                "user_id": s.get("user_id"),
                "created_at": s.get("created_at"),
                "last_activity": s.get("last_activity"),
                "message_count": len(s.get("messages", [])),
                "prescription_count": len(s.get("prescription_history", [])),
                "query_count": len(s.get("medical_queries", [])),
                "recent_topics": self._extract_recent_topics(s),
            }
        except Exception as e:
            logger.error(f"Error summarizing context: {e}")
            return {}

    def _extract_recent_topics(self, session: Dict[str, Any]) -> List[str]:
        topics: List[str] = []
        for m in session.get("messages", [])[-12:]:
            if m.get("role") == "user":
                t = (m.get("content") or "").lower()
                for kw in ["prescription", "medicine", "medication", "treatment", "symptom", "diagnosis", "therapy"]:
                    if kw in t and kw not in topics:
                        topics.append(kw)
        return topics[:5]

    # --------- persistence --------- #
    def _save_session(self, session_id: str) -> bool:
        try:
            s = self.active_sessions.get(session_id)
            if not s:
                return False
            f = self.memory_dir / f"{session_id}.json"
            with open(f, "w", encoding="utf-8") as out:
                json.dump(s, out, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            logger.error(f"Save error: {e}")
            return False

    def _load_session(self, session_id: str) -> bool:
        try:
            f = self.memory_dir / f"{session_id}.json"
            if not f.exists():
                return False
            with open(f, "r", encoding="utf-8") as src:
                self.active_sessions[session_id] = json.load(src)
            return True
        except Exception as e:
            logger.error(f"Load error: {e}")
            return False

    def cleanup_old_sessions(self) -> int:
        try:
            cutoff = datetime.now() - timedelta(days=self.max_memory_days)
            n = 0
            for f in self.memory_dir.glob("*.json"):
                try:
                    with open(f, "r", encoding="utf-8") as src:
                        s = json.load(src)
                    last = datetime.fromisoformat(s.get("last_activity", "1970-01-01T00:00:00"))
                    if last < cutoff:
                        f.unlink()
                        sid = f.stem
                        self.active_sessions.pop(sid, None)
                        n += 1
                except Exception as e:
                    logger.warning(f"Cleanup skip {f}: {e}")
            return n
        except Exception as e:
            logger.error(f"Cleanup error: {e}")
            return 0
