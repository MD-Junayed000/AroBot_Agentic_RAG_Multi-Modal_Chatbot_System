"""
RAG Agent for retrieving and generating responses using knowledge base
"""
from typing import List, Dict, Any, Optional
from core.vector_store import PineconeStore
from core.llm_handler import LLMHandler
from config.env_config import PINECONE_MEDICINE_INDEX, PINECONE_BD_PHARMACY_INDEX, PINECONE_IMAGE_INDEX
from langsmith import traceable
import logging

logger = logging.getLogger(__name__)

class RAGAgent:
    """Retrieval-Augmented Generation Agent for medical knowledge"""
    
    def __init__(self):
        # Initialize vector stores for different knowledge bases
        self.bd_pharmacy_store = PineconeStore(
            index_name=PINECONE_BD_PHARMACY_INDEX,
            dimension=384
        )
        self.medicine_store = PineconeStore(
            index_name=PINECONE_MEDICINE_INDEX,
            dimension=384
        )
        
        # Initialize LLM handler
        self.llm = LLMHandler()
    
    @traceable(name="retrieve_medical_knowledge")
    def retrieve_medical_knowledge(self, query: str, top_k: int = 5) -> List[str]:
        """Retrieve relevant medical knowledge from BD Pharmacy index (namespaced PDFs/guidelines)."""
        try:
            results = self.bd_pharmacy_store.query(query, top_k=top_k)
            return results
        except Exception as e:
            logger.error(f"Error retrieving medical knowledge: {e}")
            return []
    
    @traceable(name="retrieve_medicine_info")
    def retrieve_medicine_info(self, query: str, top_k: int = 3) -> List[str]:
        """Retrieve relevant medicine information from CSV data"""
        try:
            results = self.medicine_store.query(query, top_k=top_k)
            return results
        except Exception as e:
            logger.error(f"Error retrieving medicine info: {e}")
            return []
    
    @traceable(name="generate_medical_response")
    def generate_medical_response(self, query: str, use_medicine_db: bool = True, conversation_context: str = "") -> Dict[str, Any]:
        """Generate comprehensive medical response using RAG with conversation memory"""
        try:
            print(f"🧠 [DEBUG] RAG agent received conversation context: {len(conversation_context)} chars")
            # Retrieve context from medical knowledge base
            medical_context = self.retrieve_medical_knowledge(query, top_k=3)
            
            # Optionally retrieve medicine information
            medicine_context = []
            if use_medicine_db:
                medicine_context = self.retrieve_medicine_info(query, top_k=2)
            
            # Combine contexts
            all_context = medical_context + medicine_context
            
            # Generate response with conversation context
            response = self.llm.answer_medical_query(query, all_context, conversation_context=conversation_context)
            
            return {
                "response": response,
                "medical_sources": len(medical_context),
                "medicine_sources": len(medicine_context),
                "context_used": all_context,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Error generating medical response: {e}")
            return {
                "error": str(e),
                "status": "error"
            }
    
    @traceable(name="analyze_prescription_query")
    def analyze_prescription_query(self, prescription_text: str, user_query: str = None) -> Dict[str, Any]:
        """Analyze prescription and answer related queries"""
        try:
            # Extract medication names from prescription text
            medication_query = f"prescription analysis medications: {prescription_text}"
            
            # Retrieve medicine information for identified medications
            medicine_context = self.retrieve_medicine_info(medication_query, top_k=5)
            
            # If user has a specific query, use it; otherwise provide general analysis
            if user_query:
                query = f"Based on this prescription: {prescription_text}\nUser question: {user_query}"
            else:
                query = f"Analyze this prescription and provide information about the medications: {prescription_text}"
            
            # Generate response
            response = self.llm.answer_medical_query(query, medicine_context)
            
            return {
                "response": response,
                "prescription_text": prescription_text,
                "medicine_sources": len(medicine_context),
                "context_used": medicine_context,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Error analyzing prescription query: {e}")
            return {
                "error": str(e),
                "status": "error"
            }
    
    @traceable(name="search_medicine_by_condition")
    def search_medicine_by_condition(self, condition: str) -> Dict[str, Any]:
        """Search for medicines based on medical condition"""
        try:
            # Search for medicines that treat the condition
            query = f"treatment medication for {condition} indication therapy"
            medicine_results = self.retrieve_medicine_info(query, top_k=10)
            
            # Also search medical knowledge base for condition information
            medical_results = self.retrieve_medical_knowledge(f"{condition} treatment", top_k=3)
            
            # Generate comprehensive response
            combined_query = f"What medicines are used to treat {condition}? Provide detailed information."
            response = self.llm.answer_medical_query(combined_query, medicine_results + medical_results)
            
            return {
                "response": response,
                "condition": condition,
                "medicine_results": medicine_results,
                "medical_context": medical_results,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Error searching medicine by condition: {e}")
            return {
                "error": str(e),
                "status": "error"
            }
    
    @traceable(name="search_medicine_by_name")
    def search_medicine_by_name(self, medicine_name: str) -> Dict[str, Any]:
        """Search for specific medicine information by name"""
        try:
            query = f"{medicine_name} medicine drug medication"
            matches = self.medicine_store.query(query, top_k=5)
            
            if matches:
                context = matches[:3]  # Use top 3 matches
                response = self.llm.answer_medical_query(
                    f"Tell me about {medicine_name}. What is it used for, its dosage, and side effects?",
                    context
                )
                
                return {
                    "response": response,
                    "matches": matches,
                    "medicine_sources": len(matches),
                    "status": "success"
                }
            else:
                return {
                    "response": f"I couldn't find specific information about {medicine_name} in my medicine database.",
                    "matches": [],
                    "medicine_sources": 0,
                    "status": "success"
                }
                
        except Exception as e:
            logger.error(f"Error searching medicine by name: {e}")
            return {
                "error": str(e),
                "status": "error"
            }

    def get_knowledge_base_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge bases"""
        try:
            # This would require additional methods in PineconeStore to get index stats
            return {
                "medicine_index": PINECONE_MEDICINE_INDEX,
                "bd_pharmacy_index": PINECONE_BD_PHARMACY_INDEX,
                "image_index": PINECONE_IMAGE_INDEX,
                "status": "active"
            }
        except Exception as e:
            logger.error(f"Error getting knowledge base stats: {e}")
            return {
                "error": str(e),
                "status": "error"
            }
