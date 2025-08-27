"""
MCP (Model Context Protocol) Handler for maintaining conversation context
"""
from typing import Dict, Any, List, Optional
from .conversation_memory import ConversationMemory
from langsmith import traceable
import logging

logger = logging.getLogger(__name__)

class MCPHandler:
    """Handles MCP protocol for conversation context and memory"""
    
    def __init__(self):
        self.memory = ConversationMemory()
        self.default_context_window = 10
    
    @traceable(name="initialize_session")
    def initialize_session(self, user_id: str = None) -> Dict[str, Any]:
        """Initialize a new conversation session"""
        try:
            session_id = self.memory.create_session(user_id)
            
            # Add welcome message
            self.memory.add_message(
                session_id, 
                "system", 
                "AroBot session initialized. I'm ready to help with your medical questions.",
                "system"
            )
            
            return {
                "session_id": session_id,
                "status": "initialized",
                "message": "Session created successfully"
            }
            
        except Exception as e:
            logger.error(f"Error initializing session: {e}")
            return {
                "error": str(e),
                "status": "error"
            }
    
    @traceable(name="add_user_message")
    def add_user_message(self, session_id: str, message: str, 
                        message_type: str = "text", metadata: Dict = None) -> bool:
        """Add user message to conversation"""
        return self.memory.add_message(session_id, "user", message, message_type, metadata)
    
    @traceable(name="add_assistant_response")
    def add_assistant_response(self, session_id: str, response: str, 
                              message_type: str = "text", metadata: Dict = None) -> bool:
        """Add assistant response to conversation"""
        return self.memory.add_message(session_id, "assistant", response, message_type, metadata)
    
    @traceable(name="get_conversation_context")
    def get_conversation_context(self, session_id: str, context_window: int = None) -> Dict[str, Any]:
        """Get conversation context for LLM"""
        try:
            if not context_window:
                context_window = self.default_context_window
            
            # Get recent conversation history
            messages = self.memory.get_conversation_history(session_id, context_window * 2)
            
            # Get session context
            context_summary = self.memory.get_context_summary(session_id)
            
            # Get recent prescription history
            prescription_history = self.memory.get_prescription_history(session_id)[-3:]
            
            # Format for LLM consumption
            formatted_context = self._format_context_for_llm(
                messages, context_summary, prescription_history
            )
            
            return {
                "session_id": session_id,
                "context": formatted_context,
                "message_count": len(messages),
                "prescription_count": len(prescription_history),
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Error getting conversation context: {e}")
            return {
                "error": str(e),
                "status": "error"
            }
    
    @traceable(name="record_prescription_analysis")
    def record_prescription_analysis(self, session_id: str, prescription_data: Dict) -> bool:
        """Record prescription analysis in conversation memory"""
        try:
            # Add to prescription history
            success = self.memory.add_prescription_record(session_id, prescription_data)
            
            if success:
                # Add as conversation message
                summary = self._create_prescription_summary(prescription_data)
                self.memory.add_message(
                    session_id,
                    "system",
                    f"Prescription analyzed: {summary}",
                    "prescription",
                    {"prescription_id": prescription_data.get("id", "unknown")}
                )
            
            return success
            
        except Exception as e:
            logger.error(f"Error recording prescription analysis: {e}")
            return False
    
    @traceable(name="record_medical_query")
    def record_medical_query(self, session_id: str, query: str, response: str, 
                           query_type: str = "general") -> bool:
        """Record medical query and response"""
        try:
            # Record in medical query history
            success = self.memory.add_medical_query(session_id, query, response, query_type)
            
            if success:
                # Add to conversation
                self.memory.add_message(session_id, "user", query, "query")
                self.memory.add_message(session_id, "assistant", response, "response")
            
            return success
            
        except Exception as e:
            logger.error(f"Error recording medical query: {e}")
            return False
    
    def get_user_medical_history(self, session_id: str) -> Dict[str, Any]:
        """Get user's medical history from conversation"""
        try:
            prescription_history = self.memory.get_prescription_history(session_id)
            medical_queries = self.memory.get_context_summary(session_id)
            
            # Extract medical conditions and medications mentioned
            conditions = set()
            medications = set()
            
            for prescription in prescription_history:
                prescription_data = prescription.get("prescription_data", {})
                entities = prescription_data.get("entities", {})
                
                # Extract medications
                for med in entities.get("medications", []):
                    if med.get("text"):
                        medications.add(med["text"][:50])  # Truncate long names
            
            # Get conversation history to extract mentioned conditions
            messages = self.memory.get_conversation_history(session_id, 50)
            for message in messages:
                if message.get("role") == "user":
                    content = message.get("content", "").lower()
                    # Simple condition extraction (could be enhanced with NLP)
                    medical_terms = [
                        "diabetes", "hypertension", "asthma", "depression", "anxiety",
                        "arthritis", "migraine", "allergy", "infection", "fever"
                    ]
                    for term in medical_terms:
                        if term in content:
                            conditions.add(term)
            
            return {
                "session_id": session_id,
                "conditions_mentioned": list(conditions)[:10],
                "medications_found": list(medications)[:10],
                "prescription_count": len(prescription_history),
                "query_count": len(medical_queries.get("medical_queries", [])),
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Error getting user medical history: {e}")
            return {
                "error": str(e),
                "status": "error"
            }
    
    def _format_context_for_llm(self, messages: List[Dict], context_summary: Dict, 
                               prescription_history: List[Dict]) -> str:
        """Format conversation context for LLM consumption"""
        try:
            context_parts = []
            
            # Session summary
            if context_summary:
                context_parts.append("**Session Context:**")
                context_parts.append(f"- Session started: {context_summary.get('created_at', 'Unknown')}")
                context_parts.append(f"- Total messages: {context_summary.get('message_count', 0)}")
                context_parts.append(f"- Prescriptions analyzed: {context_summary.get('prescription_count', 0)}")
                
                recent_topics = context_summary.get('recent_topics', [])
                if recent_topics:
                    context_parts.append(f"- Recent topics: {', '.join(recent_topics)}")
            
            # Recent prescriptions
            if prescription_history:
                context_parts.append("\n**Recent Prescriptions:**")
                for i, prescription in enumerate(prescription_history, 1):
                    summary = self._create_prescription_summary(prescription.get("prescription_data", {}))
                    context_parts.append(f"{i}. {summary}")
            
            # Recent conversation
            if messages:
                context_parts.append("\n**Recent Conversation:**")
                for message in messages[-6:]:  # Last 6 messages
                    role = message.get("role", "unknown")
                    content = message.get("content", "")[:200]  # Truncate long messages
                    context_parts.append(f"{role.capitalize()}: {content}")
            
            return "\n".join(context_parts)
            
        except Exception as e:
            logger.error(f"Error formatting context for LLM: {e}")
            return "Context formatting error"
    
    def _create_prescription_summary(self, prescription_data: Dict) -> str:
        """Create a brief summary of prescription data"""
        try:
            entities = prescription_data.get("entities", {})
            medications = entities.get("medications", [])
            
            if medications:
                med_names = [med.get("text", "Unknown")[:30] for med in medications[:3]]
                return f"Medications: {', '.join(med_names)}"
            else:
                return "Prescription processed (no medications clearly identified)"
                
        except Exception as e:
            logger.error(f"Error creating prescription summary: {e}")
            return "Prescription summary error"
    
    def cleanup_sessions(self) -> Dict[str, Any]:
        """Clean up old sessions"""
        try:
            cleaned_count = self.memory.cleanup_old_sessions()
            return {
                "cleaned_sessions": cleaned_count,
                "status": "success"
            }
        except Exception as e:
            logger.error(f"Error cleaning up sessions: {e}")
            return {
                "error": str(e),
                "status": "error"
            }
