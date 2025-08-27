"""
LLM Handler for text and vision models using Ollama
"""
import base64
import httpx
from typing import List, Dict, Any, Optional
from langchain_ollama import OllamaLLM
from langchain.schema import BaseMessage, HumanMessage, SystemMessage
import ollama
from config.env_config import OLLAMA_BASE_URL, OLLAMA_TEXT_MODEL, OLLAMA_VISION_MODEL
from langsmith import traceable
import logging

logger = logging.getLogger(__name__)

class LLMHandler:
    """Handles both text and vision LLMs using Ollama"""
    
    def __init__(self):
        self.base_url = OLLAMA_BASE_URL
        self.text_model = OLLAMA_TEXT_MODEL
        self.vision_model = OLLAMA_VISION_MODEL
        
        # Initialize text LLM
        self.text_llm = OllamaLLM(
            model=self.text_model,
            base_url=self.base_url,
            temperature=0.7
        )
        
        # Initialize Ollama client for vision
        self.client = ollama.Client(host=self.base_url)
    
    @traceable(name="text_completion")
    def generate_text_response(self, prompt: str, system_prompt: str = None) -> str:
        """Generate text response using text LLM"""
        try:
            messages = []
            if system_prompt:
                messages.append(SystemMessage(content=system_prompt))
            messages.append(HumanMessage(content=prompt))
            
            response = self.text_llm.invoke(prompt)
            return response
        except Exception as e:
            logger.error(f"Error in text generation: {e}")
            return f"Error generating response: {str(e)}"
    
    @traceable(name="vision_completion")
    def generate_vision_response(self, prompt: str, image_path: str = None, image_data: bytes = None) -> str:
        """Generate response using vision LLM with image input"""
        try:
            if image_data:
                # Convert bytes to base64
                image_b64 = base64.b64encode(image_data).decode('utf-8')
            elif image_path:
                # Read and encode image file
                with open(image_path, 'rb') as f:
                    image_b64 = base64.b64encode(f.read()).decode('utf-8')
            else:
                return "No image provided for vision model"
            
            response = self.client.generate(
                model=self.vision_model,
                prompt=prompt,
                images=[image_b64],
                stream=False
            )
            
            return response['response']
        except Exception as e:
            logger.error(f"Error in vision generation: {e}")
            return f"Error analyzing image: {str(e)}"
    
    @traceable(name="prescription_analysis")
    def analyze_prescription(self, image_path: str = None, image_data: bytes = None, ocr_text: str = None) -> Dict[str, Any]:
        """Analyze prescription image and extract medical information"""
        
        vision_prompt = """
        You are a medical AI assistant analyzing a prescription image. Please extract and analyze:
        
        1. Patient information (name, age, etc.)
        2. Doctor information (name, clinic, etc.)
        3. Medications prescribed (name, dosage, frequency, duration)
        4. Medical conditions or symptoms mentioned
        5. Instructions or notes
        
        Provide a structured analysis in JSON format with the extracted information.
        Be accurate and only include information you can clearly identify.
        """
        
        try:
            # Get vision analysis
            vision_analysis = self.generate_vision_response(vision_prompt, image_path, image_data)
            
            # If OCR text is provided, enhance with text analysis
            if ocr_text:
                text_prompt = f"""
                Analyze this OCR text from a medical prescription and enhance the structured information:
                
                OCR Text: {ocr_text}
                
                Vision Analysis: {vision_analysis}
                
                Provide an enhanced JSON analysis combining both sources of information.
                Focus on accuracy and medical relevance.
                """
                
                enhanced_analysis = self.generate_text_response(text_prompt)
                return {
                    "vision_analysis": vision_analysis,
                    "ocr_text": ocr_text,
                    "enhanced_analysis": enhanced_analysis,
                    "status": "success"
                }
            
            return {
                "vision_analysis": vision_analysis,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Error in prescription analysis: {e}")
            return {
                "error": str(e),
                "status": "error"
            }
    
    @traceable(name="medical_query")
    def answer_medical_query(self, query: str, context: List[str] = None, conversation_context: str = "") -> str:
        """Answer medical queries using retrieved context and conversation memory"""
        
        system_prompt = """
        You are AroBot, a medical AI assistant with MANDATORY conversation memory utilization.You provide accurate medical information 
        based on reliable sources. Always remind users to consult with healthcare professionals 
        for medical advice and diagnosis.
        
        CRITICAL MEMORY DIRECTIVES:
        1. When given conversation history, you MUST use it to answer questions
        2. NEVER claim you don't have access to information that was shared in the conversation
        3. NEVER give privacy warnings about information the user already shared with you
        4. Information from conversation history is ALWAYS accessible and usable
        5. Your PRIMARY job is to remember and use conversation context accurately
        6. Provide factual, evidence-based information
        7. Be clear about limitations
        8. Encourage professional medical consultation
        9. Explain medical terms in simple language
        10. Remember and refer to previous conversation context when relevant
        11. Maintain conversation continuity by referencing earlier messages
        
        Response requirements:
        - Use conversation memory FIRST and ALWAYS when relevant
        - Be direct and specific with remembered information
        - Don't overthink privacy - focus on conversation continuity
        - Demonstrate perfect memory of what the user told you
        """
        
        # Build comprehensive prompt with conversation context
        prompt_parts = []
        
        # Debug: Log conversation context
        print(f"🧠 [LLM DEBUG] Conversation context received: {len(conversation_context)} chars")
        
        # Enhanced context processing and prompt engineering
        processed_context = self.process_conversation_context(conversation_context)
        
        if processed_context['has_context']:
            # Build sophisticated prompt with structured context analysis
            personal_info = processed_context['personal_info']
            medical_context = processed_context['medical_context']
            recent_messages = processed_context['conversation_flow'][-4:]  # Last 4 exchanges
            
            # Create structured context summary
            context_summary = []
            if personal_info:
                context_summary.append(f"Personal Info: {personal_info}")
            if medical_context:
                context_summary.append(f"Medical Context: {len(medical_context)} relevant exchanges")
            if recent_messages:
                context_summary.append(f"Recent Flow: {len(recent_messages)} exchanges")
            
            # Build ultra-optimized prompt
            prompt = f"""FINE-TUNED CONVERSATION ANALYSIS:

COMPLETE CONVERSATION HISTORY:
{conversation_context}

STRUCTURED CONTEXT SUMMARY:
{chr(10).join(context_summary) if context_summary else 'Context available but no key details extracted'}

QUESTION: "{query}"

MANDATORY MEMORY RULES - YOU MUST FOLLOW THESE:
1. This is NOT about privacy or personal data - this is about OUR CONVERSATION
2. The user already told you their information in THIS conversation - use it!
3. If they ask about their name/department/hospital, find it in the conversation above
4. NEVER say "I don't have access" - you DO have access to our conversation history
5. NEVER give privacy warnings - this is conversation memory, not personal data access

1. MEMORY RETRIEVAL:
   - Personal identity questions: Use extracted personal info or find in conversation
   - Role/department questions: Reference department/hospital information
   - Medical questions: Combine conversation context with medical knowledge
   - General questions: Maintain conversation continuity

2. RESPONSE GENERATION:
   - Be specific and accurate with remembered information
   - Reference conversation context when relevant
   - Maintain professional medical assistant tone
   - Provide helpful, actionable responses

3. MEMORY PATTERNS:
   - Name queries: "Based on our conversation, you are  [Name]"
   - Department queries: "You work in the [Department] department"
   - Hospital queries: "You work at [Hospital]"
   - Medical queries: Use conversation context + medical knowledge

4. CONVERSATION CONTINUITY:
   - Acknowledge relevant previous exchanges
   - Build upon established context
   - Maintain consistent information across exchanges

Please provide an accurate, context-aware response:"""
        else:
            # No conversation context available
            prompt = f"""You are AroBot, a medical AI assistant.

CURRENT QUESTION: "{query}"

Please provide a helpful medical response."""
            print(f"⚠️ [LLM DEBUG] No conversation context available")
        
        print(f"🔍 [LLM DEBUG] Final prompt preview: {prompt[:200]}...")
        
        try:
            response = self.generate_text_response(prompt, system_prompt)
            return response
        except Exception as e:
            logger.error(f"Error answering medical query: {e}")
            return f"I apologize, but I encountered an error while processing your question: {str(e)}"
    
    def process_conversation_context(self, conversation_context: str) -> Dict[str, Any]:
        """Process and analyze conversation context for optimal LLM utilization"""
        
        if not conversation_context.strip():
            return {'has_context': False, 'extracted_info': {}}
        
        # Parse conversation into structured format
        lines = conversation_context.split('\n')
        structured_context = {
            'user_messages': [],
            'assistant_messages': [],
            'personal_info': {},
            'medical_context': [],
            'conversation_flow': []
        }
        
        current_speaker = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if line.startswith('User:'):
                current_speaker = 'user'
                content = line.replace('User:', '').strip()
                structured_context['user_messages'].append(content)
                
                # Extract personal information patterns
                content_lower = content.lower()
                if 'i am dr' in content_lower:
                    # Extract name pattern: "I am Dr. [First] [Last]"
                    import re
                    name_match = re.search(r'i am dr\.?\s+([a-z]+\s+[a-z]+)', content_lower)
                    if name_match:
                        structured_context['personal_info']['name'] = name_match.group(1).title()
                
                if 'department' in content_lower:
                    # Extract department pattern: "... [department] department"
                    dept_match = re.search(r'(\w+)\s+department', content_lower)
                    if dept_match:
                        structured_context['personal_info']['department'] = dept_match.group(1)
                
                if 'hospital' in content_lower or 'medical center' in content_lower:
                    # Extract hospital/medical center
                    import re
                    hospital_match = re.search(r'(?:at|from|in)\s+([a-z\s]+(?:hospital|medical center))', content_lower)
                    if hospital_match:
                        structured_context['personal_info']['hospital'] = hospital_match.group(1).title()
                
                # Medical context extraction
                medical_keywords = ['patient', 'medication', 'prescription', 'symptom', 'diagnosis', 'treatment']
                if any(keyword in content_lower for keyword in medical_keywords):
                    structured_context['medical_context'].append(content)
                    
            elif line.startswith('Assistant:') or line.startswith('AroBot:'):
                current_speaker = 'assistant'
                content = line.replace('Assistant:', '').replace('AroBot:', '').strip()
                structured_context['assistant_messages'].append(content)
            
            # Track conversation flow
            if current_speaker:
                structured_context['conversation_flow'].append({
                    'speaker': current_speaker,
                    'content': content if current_speaker == 'user' else line
                })
        
        structured_context['has_context'] = True
        return structured_context

    def check_model_availability(self) -> Dict[str, bool]:
        """Check if required models are available"""
        try:
            models = self.client.list()
            # Handle different response formats
            if hasattr(models, 'models'):
                model_list = models.models
            elif isinstance(models, dict) and 'models' in models:
                model_list = models['models']
            else:
                model_list = models
            
            # Extract model names safely
            available_models = []
            for model in model_list:
                if isinstance(model, dict):
                    available_models.append(model.get('name', ''))
                elif hasattr(model, 'name'):
                    available_models.append(model.name)
                else:
                    available_models.append(str(model))
            
            return {
                "text_model_available": self.text_model in available_models,
                "vision_model_available": self.vision_model in available_models,
                "models_found": available_models
            }
        except Exception as e:
            logger.error(f"Error checking model availability: {e}")
            return {
                "text_model_available": False,
                "vision_model_available": False,
                "error": str(e)
            }
