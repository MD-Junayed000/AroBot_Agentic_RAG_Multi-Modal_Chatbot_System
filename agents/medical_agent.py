"""
Medical Agent - Main orchestrator for medical queries and multi-modal interactions
"""
from typing import Dict, Any, List, Union, Optional
from PIL import Image
from .rag_agent import RAGAgent
from .ocr_agent import OCRAgent
from core.llm_handler import LLMHandler
from utils.web_search import WebSearchTool
from langsmith import traceable
import logging

logger = logging.getLogger(__name__)

class MedicalAgent:
    """Main medical agent that orchestrates different capabilities"""
    
    def __init__(self):
        self.rag_agent = RAGAgent()
        self.ocr_agent = OCRAgent()
        self.llm = LLMHandler()
        self.web_search = WebSearchTool()
    
    @traceable(name="handle_image_query")
    def handle_image_query(self, image_input: Union[str, bytes, Image.Image], 
                          user_query: str = None, image_type: str = "prescription") -> Dict[str, Any]:
        """Handle any image analysis with optional user query"""
        try:
            # Handle different image types
            if image_type == "prescription":
                return self._handle_prescription_image(image_input, user_query)
            else:
                return self._handle_general_image(image_input, user_query)
        
        except Exception as e:
            logger.error(f"Error handling image query: {e}")
            return {
                "error": str(e),
                "status": "error"
            }
    
    @traceable(name="handle_prescription_query")
    def handle_prescription_query(self, image_input: Union[str, bytes, Image.Image], 
                                user_query: str = None) -> Dict[str, Any]:
        """Handle prescription image with optional user query"""
        return self.handle_image_query(image_input, user_query, "prescription")
    
    def _handle_prescription_image(self, image_input: Union[str, bytes, Image.Image], 
                                  user_query: str = None) -> Dict[str, Any]:
        """Handle prescription-specific image analysis"""
        try:
            # Step 1: OCR Analysis
            ocr_results = self.ocr_agent.process_prescription_image(image_input)
            
            if ocr_results.get('status') == 'error':
                return {
                    "error": "Failed to process prescription image",
                    "details": ocr_results.get('error'),
                    "status": "error"
                }
            
            # Step 2: LLM Vision Analysis
            llm_analysis = self.ocr_agent.analyze_prescription_with_llm(
                image_input, 
                ocr_results.get('ocr_results', {}).get('raw_text')
            )
            
            # Step 3: Extract medications for RAG search
            medications = ocr_results.get('entities', {}).get('medications', [])
            prescription_text = ocr_results.get('ocr_results', {}).get('raw_text', '')
            
            # Step 4: RAG Analysis
            rag_response = None
            if prescription_text:
                rag_response = self.rag_agent.analyze_prescription_query(
                    prescription_text, user_query
                )
            
            # Step 5: Combine results
            response = self._combine_prescription_analysis(
                ocr_results, llm_analysis, rag_response, user_query
            )
            
            return {
                "prescription_analysis": response,
                "ocr_results": ocr_results,
                "llm_analysis": llm_analysis,
                "rag_response": rag_response,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Error handling prescription query: {e}")
            return {
                "error": str(e),
                "status": "error"
            }
    
    def _handle_general_image(self, image_input: Union[str, bytes, Image.Image], 
                             user_query: str = None) -> Dict[str, Any]:
        """Handle general image analysis using vision LLM"""
        try:
            # Convert image input to appropriate format
            if isinstance(image_input, bytes):
                # Save bytes to temporary file for vision processing
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                    tmp_file.write(image_input)
                    image_path = tmp_file.name
            elif isinstance(image_input, str):
                image_path = image_input
            else:
                # PIL Image - save to temporary file
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                    image_input.save(tmp_file.name)
                    image_path = tmp_file.name
            
            # Default prompt if none provided
            if not user_query:
                user_query = "Describe what you see in this image in detail. If it's a medical image, identify any medical conditions, anatomy, or relevant medical information."
            
            # Use vision LLM for analysis
            vision_response = self.llm.generate_vision_response(
                prompt=user_query,
                image_path=image_path
            )
            
            # Clean up temporary file if created
            if isinstance(image_input, (bytes, Image.Image)):
                import os
                try:
                    os.unlink(image_path)
                except:
                    pass
            
            return {
                "analysis": vision_response,
                "type": "general_image_analysis",
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Error handling general image: {e}")
            return {
                "error": str(e),
                "status": "error"
            }
    
    @traceable(name="handle_text_query")
    def handle_text_query(self, query: str, session_id: str = None, conversation_context: str = "", use_web_search: bool = False) -> Dict[str, Any]:
        """Handle text-based medical queries with conversation memory"""
        try:
            print(f"🧠 [DEBUG] Medical agent received conversation context: {len(conversation_context)} chars")
            if conversation_context.strip():
                print(f"📝 [DEBUG] Context preview: {conversation_context[:100]}...")
            else:
                print("⚠️ [DEBUG] No conversation context received")
            # Check if user is asking for image generation
            image_gen_keywords = ['generate', 'create', 'make', 'draw', 'show me image', 'picture of', 'can you generate', 'create an image']
            if any(keyword in query.lower() for keyword in image_gen_keywords):
                return {
                    "response": "I cannot generate, create, or produce images. I'm a text-based AI medical assistant that can only analyze and describe images you provide to me. If you need information about a medication, medical condition, or anatomy, I can provide detailed descriptions and explanations based on my medical knowledge database.",
                    "type": "image_generation_not_supported",
                    "status": "success"
                }
            
            # Step 1: Enhanced RAG Response for medicine queries
            medicine_keywords = ['napa', 'paracetamol', 'acetaminophen', 'aspirin', 'ibuprofen', 'medicine', 'drug', 'medication', 'tablet', 'capsule', 'syrup']
            is_medicine_query = any(keyword in query.lower() for keyword in medicine_keywords)
            
            if is_medicine_query:
                # Enhanced medicine-specific search
                rag_response = self.rag_agent.search_medicine_by_name(query)
                if not rag_response or rag_response.get('medicine_sources', 0) == 0:
                    # Fallback to general medical search
                    rag_response = self.rag_agent.generate_medical_response(query, conversation_context=conversation_context)
            else:
                # Step 1: Regular RAG Response with conversation context
                rag_response = self.rag_agent.generate_medical_response(query, conversation_context=conversation_context)
            
            # Check if it's a general question that needs web search
            general_keywords = ['weather', 'news', 'current', 'today', 'latest', 'what is happening']
            is_general_query = any(keyword in query.lower() for keyword in general_keywords)
            
            # Step 2: Web search if requested, for general questions, or RAG doesn't have good context
            web_results = None
            if use_web_search or is_general_query or (rag_response.get('medical_sources', 0) == 0):
                if is_general_query:
                    web_results = self.web_search.search_general_info(query)
                else:
                    web_results = self.web_search.search_medical_info(query)
            
            # Step 3: Generate comprehensive response
            context = rag_response.get('context_used', [])
            if web_results and web_results.get('status') == 'success':
                web_context = [result.get('snippet', '') for result in web_results.get('results', [])[:3]]
                context.extend(web_context)
            
            final_response = self.llm.answer_medical_query(query, context)
            
            return {
                "response": final_response,
                "rag_response": rag_response,
                "web_results": web_results,
                "sources": {
                    "knowledge_base": rag_response.get('medical_sources', 0) + rag_response.get('medicine_sources', 0),
                    "web_search": len(web_results.get('results', [])) if web_results else 0
                },
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Error handling text query: {e}")
            return {
                "error": str(e),
                "status": "error"
            }
    
    @traceable(name="search_medicine_by_condition")
    def search_medicine_by_condition(self, condition: str) -> Dict[str, Any]:
        """Search for medicines that treat a specific condition"""
        try:
            # Use RAG agent for medicine search
            rag_results = self.rag_agent.search_medicine_by_condition(condition)
            
            # Optionally enhance with web search
            web_results = self.web_search.search_medical_info(f"{condition} treatment medication")
            
            # Combine results
            if rag_results.get('status') == 'success':
                response = rag_results.get('response', '')
                
                # Add web context if available
                if web_results and web_results.get('status') == 'success':
                    web_context = [result.get('snippet', '') for result in web_results.get('results', [])[:2]]
                    if web_context:
                        enhanced_query = f"Additional information about {condition} treatment: {' '.join(web_context)}"
                        enhanced_response = self.llm.generate_text_response(
                            f"Based on this additional context, enhance the previous response about {condition}: {enhanced_query}"
                        )
                        response += f"\n\nAdditional Information:\n{enhanced_response}"
                
                return {
                    "response": response,
                    "condition": condition,
                    "rag_results": rag_results,
                    "web_results": web_results,
                    "status": "success"
                }
            else:
                return rag_results
                
        except Exception as e:
            logger.error(f"Error searching medicine by condition: {e}")
            return {
                "error": str(e),
                "status": "error"
            }
    
    @traceable(name="general_medical_consultation")
    def general_medical_consultation(self, query: str, symptoms: List[str] = None, 
                                   medical_history: str = None) -> Dict[str, Any]:
        """Provide general medical consultation with comprehensive context"""
        try:
            # Build comprehensive query
            full_query = query
            if symptoms:
                full_query += f"\nSymptoms mentioned: {', '.join(symptoms)}"
            if medical_history:
                full_query += f"\nMedical history context: {medical_history}"
            
            # Get RAG response
            rag_response = self.rag_agent.generate_medical_response(full_query)
            
            # Enhance with web search for current information
            web_results = self.web_search.search_medical_info(query)
            
            # Generate comprehensive consultation response
            context = rag_response.get('context_used', [])
            if web_results and web_results.get('status') == 'success':
                web_context = [result.get('snippet', '') for result in web_results.get('results', [])[:3]]
                context.extend(web_context)
            
            consultation_prompt = f"""
            As AroBot, a medical AI assistant, provide a comprehensive consultation response for:
            Query: {full_query}
            
            Please structure your response with:
            1. Understanding of the query/condition
            2. Relevant medical information
            3. General recommendations
            4. When to seek professional medical care
            5. Disclaimer about consulting healthcare professionals
            """
            
            consultation_response = self.llm.generate_text_response(consultation_prompt)
            
            return {
                "consultation_response": consultation_response,
                "query": query,
                "symptoms": symptoms,
                "medical_history": medical_history,
                "rag_response": rag_response,
                "web_results": web_results,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Error in general medical consultation: {e}")
            return {
                "error": str(e),
                "status": "error"
            }
    
    def _combine_prescription_analysis(self, ocr_results: Dict, llm_analysis: Dict, 
                                     rag_response: Dict, user_query: str = None) -> str:
        """Combine different analysis results into a comprehensive response"""
        try:
            response_parts = []
            
            # OCR Summary
            entities = ocr_results.get('entities', {})
            medications = entities.get('medications', [])
            
            if medications:
                response_parts.append("**Medications Identified:**")
                for i, med in enumerate(medications[:5], 1):
                    med_text = med.get('text', 'Unknown medication')
                    strength = med.get('strength', '')
                    frequency = med.get('frequency', '')
                    med_info = f"{i}. {med_text}"
                    if strength:
                        med_info += f" - {strength}"
                    if frequency:
                        med_info += f" - {frequency}"
                    response_parts.append(med_info)
            
            # LLM Analysis
            if llm_analysis.get('status') == 'success':
                llm_result = llm_analysis.get('llm_analysis', {})
                if 'enhanced_analysis' in llm_result:
                    response_parts.append("\n**AI Analysis:**")
                    response_parts.append(llm_result['enhanced_analysis'])
                elif 'vision_analysis' in llm_result:
                    response_parts.append("\n**Vision Analysis:**")
                    response_parts.append(llm_result['vision_analysis'])
            
            # RAG Response
            if rag_response and rag_response.get('status') == 'success':
                response_parts.append("\n**Medical Information:**")
                response_parts.append(rag_response.get('response', ''))
            
            # User Query Response
            if user_query:
                response_parts.append(f"\n**Regarding your question: '{user_query}'**")
                # This could be enhanced with specific query handling
            
            return "\n".join(response_parts) if response_parts else "Analysis completed successfully."
            
        except Exception as e:
            logger.error(f"Error combining prescription analysis: {e}")
            return f"Error combining analysis results: {str(e)}"
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get status of all system components"""
        try:
            # Check LLM models
            llm_status = self.llm.check_model_availability()
            
            # Check knowledge bases
            kb_status = self.rag_agent.get_knowledge_base_stats()
            
            return {
                "llm_models": llm_status,
                "knowledge_bases": kb_status,
                "ocr_agent": "active",
                "web_search": "active",
                "overall_status": "operational",
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {
                "error": str(e),
                "status": "error"
            }
