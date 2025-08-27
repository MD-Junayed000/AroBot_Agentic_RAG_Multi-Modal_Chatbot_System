"""
Conversation Memory for maintaining chat history and context
"""
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class ConversationMemory:
    """Manages conversation memory and context for users"""
    
    def __init__(self, memory_dir: str = "memory"):
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(exist_ok=True)
        self.active_sessions = {}
        self.max_memory_days = 30
    
    def create_session(self, user_id: str = None) -> str:
        """Create a new conversation session"""
        session_id = str(uuid.uuid4())
        if not user_id:
            user_id = f"user_{uuid.uuid4().hex[:8]}"
        
        session_data = {
            "session_id": session_id,
            "user_id": user_id,
            "created_at": datetime.now().isoformat(),
            "last_activity": datetime.now().isoformat(),
            "messages": [],
            "context": {},
            "prescription_history": [],
            "medical_queries": []
        }
        
        self.active_sessions[session_id] = session_data
        self._save_session(session_id)
        
        return session_id
    
    def add_message(self, session_id: str, role: str, content: str, 
                   message_type: str = "text", metadata: Dict = None) -> bool:
        """Add a message to the conversation"""
        try:
            if session_id not in self.active_sessions:
                self._load_session(session_id)
            
            if session_id not in self.active_sessions:
                logger.warning(f"Session {session_id} not found")
                return False
            
            message = {
                "id": str(uuid.uuid4()),
                "timestamp": datetime.now().isoformat(),
                "role": role,  # "user", "assistant", "system"
                "content": content,
                "type": message_type,  # "text", "image", "prescription"
                "metadata": metadata or {}
            }
            
            self.active_sessions[session_id]["messages"].append(message)
            self.active_sessions[session_id]["last_activity"] = datetime.now().isoformat()
            
            # Keep only recent messages in memory (last 50)
            if len(self.active_sessions[session_id]["messages"]) > 50:
                self.active_sessions[session_id]["messages"] = \
                    self.active_sessions[session_id]["messages"][-50:]
            
            self._save_session(session_id)
            return True
            
        except Exception as e:
            logger.error(f"Error adding message: {e}")
            return False
    
    def get_conversation_history(self, session_id: str, limit: int = 20) -> List[Dict]:
        """Get recent conversation history"""
        try:
            if session_id not in self.active_sessions:
                self._load_session(session_id)
            
            if session_id not in self.active_sessions:
                return []
            
            messages = self.active_sessions[session_id]["messages"]
            return messages[-limit:] if limit else messages
            
        except Exception as e:
            logger.error(f"Error getting conversation history: {e}")
            return []
    
    def add_prescription_record(self, session_id: str, prescription_data: Dict) -> bool:
        """Add prescription analysis to user's history"""
        try:
            if session_id not in self.active_sessions:
                self._load_session(session_id)
            
            if session_id not in self.active_sessions:
                return False
            
            prescription_record = {
                "id": str(uuid.uuid4()),
                "timestamp": datetime.now().isoformat(),
                "prescription_data": prescription_data
            }
            
            self.active_sessions[session_id]["prescription_history"].append(prescription_record)
            
            # Keep only recent prescriptions (last 10)
            if len(self.active_sessions[session_id]["prescription_history"]) > 10:
                self.active_sessions[session_id]["prescription_history"] = \
                    self.active_sessions[session_id]["prescription_history"][-10:]
            
            self._save_session(session_id)
            return True
            
        except Exception as e:
            logger.error(f"Error adding prescription record: {e}")
            return False
    
    def get_prescription_history(self, session_id: str) -> List[Dict]:
        """Get user's prescription history"""
        try:
            if session_id not in self.active_sessions:
                self._load_session(session_id)
            
            if session_id not in self.active_sessions:
                return []
            
            return self.active_sessions[session_id]["prescription_history"]
            
        except Exception as e:
            logger.error(f"Error getting prescription history: {e}")
            return []
    
    def add_medical_query(self, session_id: str, query: str, response: str, 
                         query_type: str = "general") -> bool:
        """Add medical query to history"""
        try:
            if session_id not in self.active_sessions:
                self._load_session(session_id)
            
            if session_id not in self.active_sessions:
                return False
            
            query_record = {
                "id": str(uuid.uuid4()),
                "timestamp": datetime.now().isoformat(),
                "query": query,
                "response": response,
                "type": query_type
            }
            
            self.active_sessions[session_id]["medical_queries"].append(query_record)
            
            # Keep only recent queries (last 20)
            if len(self.active_sessions[session_id]["medical_queries"]) > 20:
                self.active_sessions[session_id]["medical_queries"] = \
                    self.active_sessions[session_id]["medical_queries"][-20:]
            
            self._save_session(session_id)
            return True
            
        except Exception as e:
            logger.error(f"Error adding medical query: {e}")
            return False
    
    def get_context_summary(self, session_id: str) -> Dict[str, Any]:
        """Get summarized context for the session"""
        try:
            if session_id not in self.active_sessions:
                self._load_session(session_id)
            
            if session_id not in self.active_sessions:
                return {}
            
            session = self.active_sessions[session_id]
            
            summary = {
                "session_id": session_id,
                "user_id": session.get("user_id"),
                "created_at": session.get("created_at"),
                "last_activity": session.get("last_activity"),
                "message_count": len(session.get("messages", [])),
                "prescription_count": len(session.get("prescription_history", [])),
                "query_count": len(session.get("medical_queries", [])),
                "recent_topics": self._extract_recent_topics(session)
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting context summary: {e}")
            return {}
    
    def _extract_recent_topics(self, session: Dict) -> List[str]:
        """Extract recent topics from conversation"""
        topics = []
        recent_messages = session.get("messages", [])[-10:]
        
        for message in recent_messages:
            if message.get("role") == "user":
                content = message.get("content", "").lower()
                # Simple keyword extraction
                medical_keywords = [
                    "prescription", "medicine", "medication", "doctor", "treatment",
                    "symptoms", "pain", "disease", "diagnosis", "therapy"
                ]
                
                for keyword in medical_keywords:
                    if keyword in content and keyword not in topics:
                        topics.append(keyword)
        
        return topics[:5]  # Return top 5 topics
    
    def _save_session(self, session_id: str) -> bool:
        """Save session to disk"""
        try:
            if session_id not in self.active_sessions:
                return False
            
            session_file = self.memory_dir / f"{session_id}.json"
            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(self.active_sessions[session_id], f, indent=2, ensure_ascii=False)
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving session: {e}")
            return False
    
    def _load_session(self, session_id: str) -> bool:
        """Load session from disk"""
        try:
            session_file = self.memory_dir / f"{session_id}.json"
            if not session_file.exists():
                return False
            
            with open(session_file, 'r', encoding='utf-8') as f:
                session_data = json.load(f)
            
            self.active_sessions[session_id] = session_data
            return True
            
        except Exception as e:
            logger.error(f"Error loading session: {e}")
            return False
    
    def cleanup_old_sessions(self) -> int:
        """Clean up old session files"""
        try:
            cutoff_date = datetime.now() - timedelta(days=self.max_memory_days)
            cleaned_count = 0
            
            for session_file in self.memory_dir.glob("*.json"):
                try:
                    with open(session_file, 'r') as f:
                        session_data = json.load(f)
                    
                    last_activity = datetime.fromisoformat(
                        session_data.get("last_activity", "1970-01-01T00:00:00")
                    )
                    
                    if last_activity < cutoff_date:
                        session_file.unlink()
                        session_id = session_file.stem
                        if session_id in self.active_sessions:
                            del self.active_sessions[session_id]
                        cleaned_count += 1
                        
                except Exception as e:
                    logger.warning(f"Error processing session file {session_file}: {e}")
            
            return cleaned_count
            
        except Exception as e:
            logger.error(f"Error cleaning up old sessions: {e}")
            return 0
