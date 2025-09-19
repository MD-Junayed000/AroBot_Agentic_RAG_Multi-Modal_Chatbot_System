"""
MCP (Model Context Protocol) Handler for maintaining conversation context
"""
from __future__ import annotations
import logging
from typing import Dict, Any, List, Optional

from .conversation_memory import ConversationMemory
from langsmith import traceable

logger = logging.getLogger(__name__)

class MCPHandler:
    """Handles MCP protocol for conversation context and memory"""

    def __init__(self):
        self.memory = ConversationMemory()
        self.default_context_window = 10

    @traceable(name="initialize_session")
    def initialize_session(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        try:
            sid = self.memory.create_session(user_id)
            self.memory.add_message(
                sid,
                "system",
                "AroBot session initialized. I'm ready to help with your medical questions.",
                "system",
            )
            return {"session_id": sid, "status": "initialized", "message": "Session created successfully"}
        except Exception as e:
            logger.error(f"Error init session: {e}")
            return {"error": str(e), "status": "error"}

    @traceable(name="add_user_message")
    def add_user_message(
        self, session_id: str, message: str, message_type: str = "text", metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        # Validate session_id
        if not session_id or session_id.strip() == "" or session_id == "None":
            logger.debug(f"Invalid session_id provided to add_user_message: {session_id}")
            return False
            
        # Ensure session exists before adding message
        if not self.memory.get_session(session_id):
            logger.debug(f"Session {session_id} not found when adding user message, skipping")
            return False
            
        return self.memory.add_message(session_id, "user", message, message_type, metadata)

    @traceable(name="add_assistant_response")
    def add_assistant_response(
        self, session_id: str, response: str, message_type: str = "text", metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        # Validate session_id
        if not session_id or session_id.strip() == "" or session_id == "None":
            logger.debug(f"Invalid session_id provided to add_assistant_response: {session_id}")
            return False
            
        # Ensure session exists before adding message
        if not self.memory.get_session(session_id):
            logger.debug(f"Session {session_id} not found when adding assistant response, skipping")
            return False
            
        return self.memory.add_message(session_id, "assistant", response, message_type, metadata)

    @traceable(name="get_conversation_context")
    def get_conversation_context(self, session_id: str, context_window: Optional[int] = None) -> Dict[str, Any]:
        try:
            window = context_window or self.default_context_window
            msgs = self.memory.get_conversation_history(session_id, window * 2)
            summary = self.memory.get_context_summary(session_id)
            rx = self.memory.get_prescription_history(session_id)[-3:]

            formatted = self._format_context_for_llm(msgs, summary, rx)
            return {
                "session_id": session_id,
                "context": formatted,
                "message_count": len(msgs),
                "prescription_count": len(rx),
                "recent_messages": msgs[-6:] if msgs else [],  # Add recent messages for better context
                "status": "success",
            }
        except Exception as e:
            logger.error(f"Error building context: {e}")
            return {"error": str(e), "status": "error"}

    @traceable(name="record_prescription_analysis")
    def record_prescription_analysis(self, session_id: str, prescription_data: Dict[str, Any]) -> bool:
        try:
            ok = self.memory.add_prescription_record(session_id, prescription_data)
            if ok:
                summary = self._create_prescription_summary(prescription_data)
                self.memory.add_message(
                    session_id, "system", f"Prescription analyzed: {summary}", "prescription", {"summary": summary}
                )
            return ok
        except Exception as e:
            logger.error(f"Error recording prescription: {e}")
            return False

    @traceable(name="record_medical_query")
    def record_medical_query(
        self, session_id: str, query: str, response: str, query_type: str = "general"
    ) -> bool:
        try:
            ok = self.memory.add_medical_query(session_id, query, response, query_type)
            if ok:
                self.memory.add_message(session_id, "user", query, "query")
                self.memory.add_message(session_id, "assistant", response, "response")
            return ok
        except Exception as e:
            logger.error(f"Error recording query: {e}")
            return False

    def get_user_medical_history(self, session_id: str) -> Dict[str, Any]:
        try:
            pres = self.memory.get_prescription_history(session_id)
            session = self.memory.get_session(session_id) or {}
            queries = session.get("medical_queries", [])

            # quick extraction
            conditions, medications = set(), set()
            for p in pres:
                pdata = p.get("prescription_data", {})
                ents = pdata.get("entities", {})
                for med in ents.get("medications", []):
                    if med.get("text"):
                        medications.add(med["text"][:50])

            msgs = self.memory.get_conversation_history(session_id, 60)
            for m in msgs:
                if m.get("role") == "user":
                    t = (m.get("content") or "").lower()
                    for term in ["diabetes", "hypertension", "asthma", "depression", "anxiety", "arthritis", "migraine", "allergy", "infection", "fever"]:
                        if term in t:
                            conditions.add(term)

            return {
                "session_id": session_id,
                "conditions_mentioned": list(conditions)[:10],
                "medications_found": list(medications)[:10],
                "prescription_count": len(pres),
                "query_count": len(queries),
                "status": "success",
            }
        except Exception as e:
            logger.error(f"History error: {e}")
            return {"error": str(e), "status": "error"}

    # --------- helpers --------- #
    def _format_context_for_llm(self, messages: List[Dict[str, Any]], summary: Dict[str, Any], rx: List[Dict[str, Any]]) -> str:
        try:
            parts: List[str] = []
            if summary:
                parts += [
                    "**Session Context:**",
                    f"- Session started: {summary.get('created_at', 'Unknown')}",
                    f"- Total messages: {summary.get('message_count', 0)}",
                    f"- Prescriptions analyzed: {summary.get('prescription_count', 0)}",
                ]
                
                if summary.get("recent_topics"):
                    parts.append(f"- Recent topics: {', '.join(summary['recent_topics'])}")
                    
                # include long-term summary if present
            try:
                sess = self.memory.get_session(summary.get("session_id"))
                lts = (sess or {}).get("long_term_summary", "")
                if lts:
                    parts.append("\n**Long-term summary (older turns):**")
                    parts.append(lts[:2000])  # keep prompt lean
            except Exception:
                pass

            if rx:
                parts.append("\n**Recent Prescriptions:**")
                for i, r in enumerate(rx, 1):
                    parts.append(f"{i}. {self._create_prescription_summary(r.get('prescription_data', {}))}")

            if messages:
                parts.append("\n**Recent Conversation:**")
                for m in messages[-6:]:
                    role = m.get("role", "unknown").capitalize()
                    content = (m.get("content") or "")[:220]
                    parts.append(f"{role}: {content}")

            return "\n".join(parts)
            
        
        except Exception as e:
            logger.error(f"Format context error: {e}")
            return "Context formatting error"

    def _create_prescription_summary(self, pdata: Dict[str, Any]) -> str:
        try:
            ents = pdata.get("entities", {})
            meds = ents.get("medications", [])
            if meds:
                names = [m.get("text", "Unknown")[:30] for m in meds[:3]]
                return f"Medications: {', '.join(names)}"
            return "Prescription processed (no medications clearly identified)"
        except Exception as e:
            logger.error(f"Prescription summary error: {e}")
            return "Prescription summary error"

    def cleanup_sessions(self) -> Dict[str, Any]:
        try:
            n = self.memory.cleanup_old_sessions()
            return {"cleaned_sessions": n, "status": "success"}
        except Exception as e:
            logger.error(f"Cleanup error: {e}")
            return {"error": str(e), "status": "error"}
