"""
Conversation Memory for maintaining chat history and context
"""
from __future__ import annotations

import json
import uuid
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class ConversationMemory:
    """Manages conversation memory and context for users (persisted to /memory)"""

    def __init__(self, memory_dir: str = "memory"):
        # Use absolute path relative to project root to avoid working directory issues
        if not os.path.isabs(memory_dir):
            # Get the project root directory (where app.py or main files are located)
            current_dir = Path(__file__).parent.parent  # Go up from mcp_server/ to project root
            self.memory_dir = current_dir / memory_dir
        else:
            self.memory_dir = Path(memory_dir)
        
        # Create directory with better error handling
        try:
            self.memory_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Memory directory initialized at: {self.memory_dir.absolute()}")
            
            # Test write permissions
            test_file = self.memory_dir / ".test_write"
            try:
                test_file.write_text("test")
                test_file.unlink()  # Remove test file
                logger.debug("Memory directory write permissions verified")
            except Exception as e:
                logger.error(f"Memory directory not writable: {e}")
                raise
                
        except Exception as e:
            logger.error(f"Failed to initialize memory directory {self.memory_dir}: {e}")
            raise
            
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
        
        # Ensure session is saved and log the result
        save_success = self._save_session(session_id)
        if save_success:
            logger.info(f"Session {session_id} created and saved successfully")
        else:
            logger.error(f"Failed to save session {session_id} to disk")
            
        return session_id

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        # Validate session_id
        if not session_id or session_id.strip() == "" or session_id == "None":
            logger.debug(f"Invalid session_id provided: {session_id}")
            return None
            
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
             # Validate session_id first
             if not session_id or session_id.strip() == "" or session_id == "None":
                 logger.debug(f"Invalid session_id provided to add_message: {session_id}")
                 return False
                 
             session = self.get_session(session_id)
             if not session:
                 # Use debug level instead of warning - this is often expected when sessions are cleaned up or not yet created
                 logger.debug(f"Session {session_id} not found when adding message")
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

             # persist every write with better error handling
             save_success = self._save_session(session_id)
             if not save_success:
                 logger.warning(f"Failed to persist session {session_id} after adding message")
             return save_success
         except Exception as e:
             logger.error(f"Error adding message to session {session_id}: {e}")
             return False


    def get_conversation_history(self, session_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        try:
            # Validate session_id first
            if not session_id or session_id.strip() == "" or session_id == "None":
                logger.debug(f"Invalid session_id provided to get_conversation_history: {session_id}")
                return []
                
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
            # Validate session_id first
            if not session_id or session_id.strip() == "" or session_id == "None":
                logger.debug(f"Invalid session_id provided to add_prescription_record: {session_id}")
                return False
                
            s = self.get_session(session_id)
            if not s:
                return False
            rec = {"id": str(uuid.uuid4()), "timestamp": datetime.now().isoformat(), "prescription_data": prescription_data}
            s["prescription_history"].append(rec)
            if len(s["prescription_history"]) > 12:
                s["prescription_history"] = s["prescription_history"][-12:]
            return self._save_session(session_id)
        except Exception as e:
            logger.error(f"Error adding prescription: {e}")
            return False

    def get_prescription_history(self, session_id: str) -> List[Dict[str, Any]]:
        try:
            # Validate session_id first
            if not session_id or session_id.strip() == "" or session_id == "None":
                logger.debug(f"Invalid session_id provided to get_prescription_history: {session_id}")
                return []
                
            s = self.get_session(session_id)
            if not s:
                return []
            return s.get("prescription_history", [])
        except Exception as e:
            logger.error(f"Error getting prescriptions: {e}")
            return []

    def add_medical_query(self, session_id: str, query: str, response: str, query_type: str = "general") -> bool:
        try:
            # Validate session_id first
            if not session_id or session_id.strip() == "" or session_id == "None":
                logger.debug(f"Invalid session_id provided to add_medical_query: {session_id}")
                return False
                
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
            if len(s["medical_queries"]) > 20:
                s["medical_queries"] = s["medical_queries"][-20:]
            return self._save_session(session_id)
        except Exception as e:
            logger.error(f"Error adding medical query: {e}")
            return False

    def get_context_summary(self, session_id: str) -> Dict[str, Any]:
        try:
            # Validate session_id first
            if not session_id or session_id.strip() == "" or session_id == "None":
                logger.debug(f"Invalid session_id provided to get_context_summary: {session_id}")
                return {}
                
            s = self.get_session(session_id)
            if not s:
                return {}

            msgs = s.get("messages", [])
            rx = s.get("prescription_history", [])

            return {
                "session_id": session_id,
                "created_at": s.get("created_at"),
                "last_activity": s.get("last_activity"),
                "message_count": len(msgs),
                "prescription_count": len(rx),
                "recent_topics": self._extract_recent_topics(s),
            }
        except Exception as e:
            logger.error(f"Error getting context summary: {e}")
            return {}

    def _extract_recent_topics(self, session: Dict[str, Any]) -> List[str]:
        try:
            msgs = session.get("messages", [])[-10:]
            topics = set()
            for m in msgs:
                if m.get("role") == "user":
                    content = (m.get("content") or "").lower()
                    if any(word in content for word in ["medicine", "drug", "medication", "prescription"]):
                        topics.add("medications")
                    if any(word in content for word in ["pain", "headache", "fever", "cold", "cough"]):
                        topics.add("symptoms")
            return list(topics)[:5]
        except Exception:
            return []

    # --------- persistence --------- #
    def _save_session(self, session_id: str) -> bool:
        try:
            s = self.active_sessions.get(session_id)
            if not s:
                logger.warning(f"Attempted to save non-existent session: {session_id}")
                return False
            
            f = self.memory_dir / f"{session_id}.json"
            
            # Create a backup if file already exists
            if f.exists():
                backup_path = self.memory_dir / f"{session_id}.json.backup"
                try:
                    f.rename(backup_path)
                except Exception as backup_error:
                    logger.debug(f"Could not create backup for {session_id}: {backup_error}")
            
            # Write the session data
            with open(f, "w", encoding="utf-8") as out:
                json.dump(s, out, indent=2, ensure_ascii=False)
            
            # Remove backup if save was successful
            backup_path = self.memory_dir / f"{session_id}.json.backup"
            if backup_path.exists():
                try:
                    backup_path.unlink()
                except Exception:
                    pass  # Ignore backup cleanup errors
                    
            logger.debug(f"Session {session_id} saved successfully to {f}")
            return True
            
        except Exception as e:
            logger.error(f"Save error for session {session_id}: {e}")
            logger.error(f"Memory directory: {self.memory_dir}")
            logger.error(f"Memory directory exists: {self.memory_dir.exists()}")
            logger.error(f"Memory directory writable: {os.access(self.memory_dir, os.W_OK)}")
            
            # Try to restore backup if save failed
            backup_path = self.memory_dir / f"{session_id}.json.backup"
            if backup_path.exists():
                try:
                    target_path = self.memory_dir / f"{session_id}.json"
                    backup_path.rename(target_path)
                    logger.info(f"Restored backup for session {session_id}")
                except Exception as restore_error:
                    logger.error(f"Failed to restore backup for {session_id}: {restore_error}")
            
            return False

    def _load_session(self, session_id: str) -> bool:
        try:
            # Validate session_id before attempting to load
            if not session_id or session_id.strip() == "" or session_id == "None":
                logger.debug(f"Invalid session_id for loading: {session_id}")
                return False
                
            f = self.memory_dir / f"{session_id}.json"
            if not f.exists():
                # Use debug level for missing files - this is often expected
                logger.debug(f"Session file not found: {f}")
                return False
            
            with open(f, "r", encoding="utf-8") as src:
                data = json.load(src)
                # Validate session data structure
                if not isinstance(data, dict) or "session_id" not in data:
                    logger.error(f"Invalid session data in {f}")
                    return False
                self.active_sessions[session_id] = data
                logger.debug(f"Session {session_id} loaded successfully from {f}")
            return True
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error in {f}: {e}")
            return False
        except Exception as e:
            logger.error(f"Load error for {session_id}: {e}")
            return False

    def cleanup_old_sessions(self) -> int:
        try:
            cutoff = datetime.now() - timedelta(days=self.max_memory_days)
            n = 0
            for f in self.memory_dir.glob("*.json"):
                try:
                    with open(f, "r", encoding="utf-8") as src:
                        s = json.load(src)
                        created = datetime.fromisoformat(s.get("created_at", ""))
                        if created < cutoff:
                            f.unlink()
                            session_id = s.get("session_id")
                            if session_id in self.active_sessions:
                                del self.active_sessions[session_id]
                            n += 1
                except Exception as e:
                    logger.debug(f"Cleanup error for {f}: {e}")
            logger.info(f"Cleaned up {n} old sessions")
            return n
        except Exception as e:
            logger.error(f"Cleanup error: {e}")
            return 0
