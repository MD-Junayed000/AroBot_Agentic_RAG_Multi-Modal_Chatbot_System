# core/agent_core.py
"""
LLM-as-Agent Core: Uses LLM to decide which tools to use for any input
"""
from __future__ import annotations

import json
import logging
from typing import Dict, Any, List, Optional, Union, Callable
from dataclasses import dataclass
from typing import get_type_hints
import inspect

from core.llm_modular import ModularLLMHandler as LLMHandler
from utils.web_search import WebSearchTool
from mcp_server.mcp_handler import MCPHandler

logger = logging.getLogger(__name__)

@dataclass
class ToolParameter:
    name: str
    type: str
    description: str
    required: bool = True
    default: Any = None

@dataclass 
class Tool:
    name: str
    description: str
    function: Callable
    parameters: List[ToolParameter]
    category: str = "general"
    priority: int = 1  # Higher priority tools are preferred
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert tool to dictionary for LLM consumption"""
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "priority": self.priority,
            "parameters": [
                {
                    "name": p.name,
                    "type": p.type,
                    "description": p.description,
                    "required": p.required,
                    "default": p.default
                }
                for p in self.parameters
            ]
        }

@dataclass
class AgentResponse:
    response: str
    tools_used: List[str]
    session_id: Optional[str] = None
    sources: Dict[str, Any] = None
    status: str = "success"

class ToolRegistry:
    """Registry for managing tools with automatic discovery and validation"""
    
    def __init__(self):
        self.tools: Dict[str, Tool] = {}
        self.categories: Dict[str, List[str]] = {}
    
    def register_tool(self, tool: Tool):
        """Register a tool in the registry"""
        self.tools[tool.name] = tool
        
        # Update category index
        if tool.category not in self.categories:
            self.categories[tool.category] = []
        if tool.name not in self.categories[tool.category]:
            self.categories[tool.category].append(tool.name)
        
        logger.info(f"Registered tool: {tool.name} (category: {tool.category})")
    
    def get_tool(self, name: str) -> Optional[Tool]:
        """Get a tool by name"""
        return self.tools.get(name)
    
    def get_tools_by_category(self, category: str) -> List[Tool]:
        """Get all tools in a category"""
        tool_names = self.categories.get(category, [])
        return [self.tools[name] for name in tool_names]
    
    def get_all_tools(self) -> List[Tool]:
        """Get all registered tools"""
        return list(self.tools.values())
    
    def get_tool_descriptions(self, category: Optional[str] = None) -> str:
        """Get formatted descriptions of tools"""
        tools = self.get_tools_by_category(category) if category else self.get_all_tools()
        
        descriptions = []
        for tool in sorted(tools, key=lambda t: (-t.priority, t.name)):
            params = ", ".join([
                f"{p.name}({p.type}): {p.description}" + ("" if p.required else " [optional]")
                for p in tool.parameters
            ])
            descriptions.append(
                f"🔧 {tool.name} [{tool.category}] (priority: {tool.priority})\n"
                f"   📝 {tool.description}\n"
                f"   📊 Parameters: {params or 'None'}"
            )
        
        return "\n\n".join(descriptions)

def tool(name: str, description: str, category: str = "general", priority: int = 1):
    """Decorator to register tool functions"""
    def decorator(func: Callable):
        # Extract parameter information from function signature
        sig = inspect.signature(func)
        type_hints = get_type_hints(func)
        
        parameters = []
        for param_name, param in sig.parameters.items():
            if param_name in ['self', 'cls']:
                continue
                
            param_type = type_hints.get(param_name, str).__name__
            required = param.default == inspect.Parameter.empty
            default = None if required else param.default
            
            # Generate description from parameter name
            param_desc = param_name.replace('_', ' ').title()
            
            parameters.append(ToolParameter(
                name=param_name,
                type=param_type,
                description=param_desc,
                required=required,
                default=default
            ))
        
        # Create tool object
        tool_obj = Tool(
            name=name,
            description=description,
            function=func,
            parameters=parameters,
            category=category,
            priority=priority
        )
        
        # Store tool metadata on function
        func._tool_metadata = tool_obj
        return func
    
    return decorator

class LLMAgent:
    """
    LLM-as-Agent: Uses LLM to decide which tools to execute based on user input
    """
    
    def __init__(self):
        self.llm = LLMHandler()
        self.web_search = WebSearchTool()
        self.mcp = MCPHandler()
        self.registry = ToolRegistry()
        self._tool_selection_cache = {}  # Simple cache for tool selection
        self._register_tools()
    
    @property
    def tools(self) -> Dict[str, Tool]:
        """Legacy property for backward compatibility"""
        return self.registry.tools
    
    def _register_tools(self):
        """Auto-register all tool functions using reflection"""
        # Get all methods that have tool metadata
        for method_name in dir(self):
            method = getattr(self, method_name)
            if hasattr(method, '_tool_metadata'):
                tool_obj = method._tool_metadata
                tool_obj.function = method  # Update function reference
                self.registry.register_tool(tool_obj)
    
    # Tool definitions using decorators
    @tool(
        name="analyze_text",
        description="Analyze text queries, medical questions, symptoms, or general conversation. Use for any text-based medical or health questions.",
        category="medical",
        priority=5
    )
    async def _tool_analyze_text(self, query: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Analyze text using the medical agent with improved handling"""
        try:
            # Check for greetings and simple queries
            query_lower = query.lower().strip()
            greeting_terms = ["hello", "hi", "hey", "good morning", "good evening", "how are you", "what's up"]
            
            if any(term in query_lower for term in greeting_terms):
                # Use LLM handler directly for greetings
                response = self.llm.greeting_response(query)
                return {"response": response, "query_type": "greeting"}
            
            # Check for "what it does" or similar capability questions
            if any(term in query_lower for term in ["what it does", "what do you do", "what can you do", "your capabilities"]):
                response = self.llm.about_response()
                return {"response": response, "query_type": "about"}
            
            # Check for anatomy/educational questions
            if any(term in query_lower for term in ["anatomy", "human body", "tell about", "explain", "what is"]):
                # Use general knowledge handler
                response = self.llm.answer_general_knowledge(query)
                return {"response": response, "query_type": "educational"}
            
            # For medical questions, use medical agent
            from agents.medical_agent import MedicalAgent
            agent = MedicalAgent()
            result = agent.handle_text_query(query, session_id=session_id)
            return result
        except Exception as e:
            logger.exception(f"Error in analyze_text tool: {e}")
            # Fallback to basic LLM response
            try:
                response = self.llm.generate_text_response(query)
                return {"response": response, "error": f"Fallback response due to: {str(e)}"}
            except Exception as e2:
                return {"error": f"Complete failure: {str(e)} | {str(e2)}"}
    
    @tool(
        name="analyze_image", 
        description="Analyze images including prescriptions, medical diagrams, anatomy charts, or general images. Performs OCR and visual analysis.",
        category="medical",
        priority=4
    )
    async def _tool_analyze_image(self, image_data: bytes, question: Optional[str] = None, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Analyze image with comprehensive prescription and medical image processing"""
        try:
            # First, always run OCR to extract text
            ocr_result = self.llm.ocr_only(image_data)
            ocr_text = ocr_result.get("raw_text", "")
            
            # Check if this looks like a prescription based on OCR content
            prescription_indicators = ["rx", "doctor", "patient", "medicine", "tablet", "capsule", "mg", "dose", "frequency"]
            is_prescription = any(indicator in ocr_text.lower() for indicator in prescription_indicators)
            
            if is_prescription or (question and "prescription" in question.lower()):
                # Comprehensive prescription analysis
                prescription_result = self.llm.analyze_prescription(image_data=image_data)
                
                if prescription_result.get("status") == "success":
                    # Extract structured data and provide comprehensive analysis
                    structured = prescription_result.get("structured", {})
                    summary = prescription_result.get("summary", "")
                    
                    # Build comprehensive response
                    response_parts = []
                    
                    if summary:
                        response_parts.append(f"**Prescription Analysis:**\n{summary}")
                    
                    # Add medication details if available
                    medications = structured.get("medications", []) if structured else []
                    if medications:
                        response_parts.append("\n**Medications Identified:**")
                        for i, med in enumerate(medications[:5], 1):  # Limit to 5 medications
                            med_name = med.get("name", "Unknown")
                            dose = med.get("dose", "")
                            frequency = med.get("frequency", "")
                            duration = med.get("duration", "")
                            
                            med_line = f"{i}. **{med_name}**"
                            if dose: med_line += f" - {dose}"
                            if frequency: med_line += f" - {frequency}"
                            if duration: med_line += f" - {duration}"
                            response_parts.append(med_line)
                    
                    # Add doctor/clinic info if available
                    if structured:
                        doctor = structured.get("doctor", "")
                        clinic = structured.get("clinic", "")
                        date = structured.get("date", "")
                        
                        if doctor or clinic or date:
                            response_parts.append("\n**Prescription Details:**")
                            if doctor: response_parts.append(f"• Doctor: {doctor}")
                            if clinic: response_parts.append(f"• Clinic: {clinic}")
                            if date: response_parts.append(f"• Date: {date}")
                    
                    # Add specific question answer if provided
                    if question and question.strip():
                        answer = self.llm.answer_over_ocr_text(question, ocr_text)
                        if answer and "can't find it" not in answer.lower():
                            response_parts.append(f"\n**Your Question:** {question}")
                            response_parts.append(f"**Answer:** {answer}")
                    
                    full_response = "\n".join(response_parts)
                    return {"response": full_response, "ocr_data": ocr_result, "structured_data": structured}
                
                else:
                    # Fallback to OCR + vision analysis
                    if question:
                        vision_response = self.llm.generate_vision_response(question, image_data=image_data)
                    else:
                        vision_response = self.llm.generate_vision_response("Analyze this medical prescription or document. What information can you extract?", image_data=image_data)
                    
                    return {"response": f"**OCR Text:**\n{ocr_text}\n\n**Visual Analysis:**\n{vision_response}"}
            
            else:
                # General image analysis
                if question:
                    vision_response = self.llm.generate_vision_response(question, image_data=image_data)
                    if ocr_text.strip():
                        response = f"**Visual Analysis:**\n{vision_response}\n\n**Text Found:**\n{ocr_text}"
                    else:
                        response = vision_response
                else:
                    brief = self.llm.default_image_brief(image_data)
                    if ocr_text.strip():
                        response = f"{brief}\n\n**Text in Image:**\n{ocr_text[:500]}{'...' if len(ocr_text) > 500 else ''}"
                    else:
                        response = brief
                
                return {"response": response, "ocr_data": ocr_result}
            
        except Exception as e:
            logger.exception(f"Error in analyze_image tool: {e}")
            # Fallback to basic OCR
            try:
                ocr_fallback = self.llm.ocr_only(image_data)
                ocr_text = ocr_fallback.get("raw_text", "")
                if ocr_text:
                    return {"response": f"I extracted this text from the image:\n\n{ocr_text}", "error": f"Partial analysis due to: {str(e)}"}
                else:
                    return {"response": "I could see the image but couldn't extract meaningful information from it.", "error": str(e)}
            except Exception as e2:
                return {"error": f"Complete analysis failure: {str(e)} | {str(e2)}"}
    
    @tool(
        name="analyze_pdf",
        description="Analyze PDF documents, extract text, summarize content, or answer questions about PDFs. Good for medical papers and documents.",
        category="documents", 
        priority=3
    )
    async def _tool_analyze_pdf(self, pdf_data: bytes, question: Optional[str] = None, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Comprehensive PDF analysis with enhanced processing"""
        try:
            from langchain_community.document_loaders import PyPDFLoader
            import tempfile
            import os
            
            # Save PDF to temp file
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(pdf_data)
                tmp_path = tmp.name
            
            try:
                loader = PyPDFLoader(tmp_path)
                docs = loader.load()
                
                # Extract text and page information
                pages_content = []
                full_text = ""
                for i, doc in enumerate(docs):
                    page_text = doc.page_content.strip()
                    if page_text:
                        pages_content.append(f"**Page {i+1}:**\n{page_text[:1000]}{'...' if len(page_text) > 1000 else ''}")
                        full_text += page_text + "\n\n"
                
                if not full_text.strip():
                    return {"response": "The PDF appears to be empty or contains no readable text."}
                
                # Analyze content type
                text_lower = full_text.lower()
                is_medical = any(term in text_lower for term in [
                    "medicine", "medical", "drug", "patient", "treatment", "diagnosis", 
                    "clinical", "pharmaceutical", "therapy", "symptoms", "disease"
                ])
                
                if question and question.strip():
                    # Answer specific question
                    response = self.llm.answer_over_pdf_text(question, full_text)
                    
                    # Add context about document structure
                    doc_info = f"\n\n**Document Info:**\n• Total pages: {len(docs)}\n• Content type: {'Medical/Clinical' if is_medical else 'General'}\n• Total characters: {len(full_text):,}"
                    response += doc_info
                    
                else:
                    # Provide comprehensive summary
                    summary_prompt = f"""Provide a comprehensive summary of this {'medical' if is_medical else ''} document. Include:

1. **Main Topic/Title**: What is this document about?
2. **Key Points**: 3-5 most important findings or points
3. **Structure**: What sections or chapters does it contain?
4. **Target Audience**: Who is this document written for?
5. **Key Takeaways**: Most important information for readers

Document text:
{full_text[:10000]}{'...(truncated)' if len(full_text) > 10000 else ''}"""
                    
                    summary = self.llm.generate_text_response(
                        summary_prompt, 
                        system_prompt="You are a medical document analyst. Provide clear, structured summaries with medical accuracy."
                    )
                    
                    # Add document statistics
                    doc_stats = f"""
**Document Statistics:**
• Total pages: {len(docs)}
• Content type: {'Medical/Clinical document' if is_medical else 'General document'}
• Word count: ~{len(full_text.split()):,}
• Character count: {len(full_text):,}

**Page Preview:**
{pages_content[0] if pages_content else 'No readable content found'}"""
                    
                    response = summary + "\n\n" + doc_stats
                
                return {
                    "response": response,
                    "document_info": {
                        "total_pages": len(docs),
                        "word_count": len(full_text.split()),
                        "char_count": len(full_text),
                        "is_medical": is_medical
                    },
                    "pages_content": pages_content[:3]  # First 3 pages for reference
                }
                
            finally:
                os.unlink(tmp_path)
                
        except Exception as e:
            logger.exception(f"Error in analyze_pdf tool: {e}")
            # Fallback attempt
            try:
                import fitz  # PyMuPDF fallback
                
                with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp:
                    tmp.write(pdf_data)
                    tmp.flush()
                    
                    doc = fitz.open(tmp.name)
                    text_content = ""
                    for page in doc:
                        text_content += page.get_text() + "\n"
                    doc.close()
                    
                    if text_content.strip():
                        if question:
                            response = self.llm.answer_over_pdf_text(question, text_content)
                        else:
                            response = f"**PDF Content Summary (via fallback method):**\n\n{text_content[:2000]}{'...' if len(text_content) > 2000 else ''}"
                        
                        return {"response": response, "error": f"Used fallback method due to: {str(e)}"}
                
            except Exception as e2:
                return {"error": f"PDF analysis failed completely: {str(e)} | Fallback also failed: {str(e2)}"}
    
    @tool(
        name="get_weather",
        description="Get current weather information for a location. Use when users ask about weather conditions.",
        category="utility",
        priority=2
    )
    async def _tool_get_weather(self, query: str) -> Dict[str, Any]:
        """Get weather information"""
        try:
            import re
            # Extract coordinates if present
            match = re.search(r"(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)", query)
            if match:
                lat, lon = float(match.group(1)), float(match.group(2))
            else:
                lat, lon = 40.73061, -73.935242  # Default NYC
            
            # Simple weather response (you can enhance this)
            response = f"Weather information for coordinates {lat:.2f}, {lon:.2f}: Current conditions available via weather API."
            return {"response": response}
        except Exception as e:
            return {"error": str(e)}
    
    @tool(
        name="get_medicine_info",
        description="Get information about medicines, drugs, dosages, prices, and pharmacy data. Use for drug and medication queries.",
        category="medical",
        priority=4
    )
    async def _tool_get_medicine_info(self, medicine_name: str, want_price: bool = False) -> Dict[str, Any]:
        """Get medicine information using vector database and web search"""
        try:
            # Check if user is asking for price specifically
            query_lower = medicine_name.lower()
            if any(term in query_lower for term in ["price", "cost", "how much", "rate", "expense"]):
                want_price = True
            
            # Get medicine information from vector database
            response = self.llm.answer_medicine(medicine_name, want_price=want_price)
            
            # If response is too short or generic, try web search for Bangladesh context
            if len(response) < 200 or "bangladesh" in query_lower:
                 try:
                     web_query = f"{medicine_name} Bangladesh medicine information price"
                     web_results = self.web_search.search_medical_info(web_query, max_results=2)
                     if web_results and web_results.get("results"):
                         web_info = "\n\n**Additional Information from Web:**\n"
                         for result in web_results["results"][:2]:
                             web_info += f"• {result.get('snippet', '')}\n"
                         response += web_info
                 except Exception as web_error:
                     logger.warning(f"Web search failed for {medicine_name}: {web_error}")
            
            return {"response": response}
        except Exception as e:
            logger.exception(f"Error in get_medicine_info tool: {e}")
            # Simple fallback
            try:
                fallback_response = self.llm.generate_text_response(
                    f"Provide basic information about the medicine {medicine_name}, including uses and general dosage guidelines."
                )
                return {"response": fallback_response, "error": f"Used fallback due to: {str(e)}"}
            except Exception as e2:
                return {"error": f"Complete failure: {str(e)} | {str(e2)}"}
    
    @tool(
        name="access_memory",
        description="Access conversation memory, recall previous interactions, or answer personal questions based on chat history.",
        category="utility",
        priority=1
    )
    async def _tool_access_memory(self, query: str, session_id: str) -> Dict[str, Any]:
        """Access conversation memory"""
        try:
            context = self.mcp.get_conversation_context(session_id)
            response = f"Conversation memory accessed. Recent context: {context.get('context', 'No recent context')[:200]}"
            return {"response": response}
        except Exception as e:
            return {"error": str(e)}
    
    @tool(
        name="web_search",
        description="Search the web for current information, news, or topics not in the knowledge base. Use when local knowledge is insufficient.",
        category="utility",
        priority=2
    )
    async def _tool_web_search(self, query: str, max_results: int = 3) -> Dict[str, Any]:
        """Search the web for information"""
        try:
            results = self.web_search.search_medical_info(query, max_results=max_results)
            
            if results and results.get("results"):
                # Format results for display
                formatted_results = []
                for i, result in enumerate(results["results"][:max_results], 1):
                    formatted_results.append(f"{i}. {result.get('title', 'No title')}: {result.get('snippet', 'No description')}")
                
                response = f"Web search results for '{query}':\n\n" + "\n\n".join(formatted_results)
                return {
                    "response": response,
                    "results": results["results"],
                    "query": query,
                    "results_count": len(results["results"])
                }
            else:
                return {"response": f"No web search results found for '{query}'"} 
        except Exception as e:
            return {"error": f"Web search failed: {str(e)}"}
    
    def get_tools_description(self) -> str:
        """Get a formatted description of all available tools"""
        return self.registry.get_tool_descriptions()
    
    def get_tools_by_category(self, category: str) -> List[Tool]:
        """Get tools filtered by category"""
        return self.registry.get_tools_by_category(category)
    
    def add_tool(self, tool: Tool):
        """Add a new tool to the agent"""
        self.registry.register_tool(tool)
    
    def clear_tool_cache(self):
        """Clear the tool selection cache"""
        self._tool_selection_cache.clear()
        logger.info("Tool selection cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            "cache_size": len(self._tool_selection_cache),
            "cache_keys": list(self._tool_selection_cache.keys()),
            "max_cache_size": 50
        }
    
    async def process_request(
        self,
        text_input: Optional[str] = None,
        image_data: Optional[bytes] = None,
        pdf_data: Optional[bytes] = None,
        session_id: Optional[str] = None
    ) -> AgentResponse:
        """
        Main entry point: LLM decides which tools to use based on input
        """
        try:
            if not session_id:
                session_id = self.mcp.initialize_session().get("session_id")
            
            # Prepare input description for the LLM
            input_description = self._describe_input(text_input, image_data, pdf_data)
            
            # Get conversation context
            conversation_context = self._get_conversation_context(session_id)
            
            # Ask LLM to decide which tools to use
            tool_selection = await self._select_tools(input_description, conversation_context)
            
            # Execute selected tools
            results = await self._execute_tools(tool_selection, text_input, image_data, pdf_data, session_id)
            
            # Generate final response
            final_response = await self._generate_final_response(results, text_input, conversation_context)
            
            # Update conversation memory
            if text_input:
                self.mcp.add_user_message(session_id, text_input)
            self.mcp.add_assistant_response(session_id, final_response)
            
            return AgentResponse(
                response=final_response,
                tools_used=[tool["name"] for tool in tool_selection],
                session_id=session_id,
                sources={"tools": tool_selection, "llm_agent": True},
                status="success"
            )
            
        except Exception as e:
            logger.exception("Error in LLM agent processing")
            return AgentResponse(
                response=f"I encountered an error while processing your request: {str(e)}",
                tools_used=[],
                session_id=session_id,
                status="error"
            )

    async def explain_request(
        self,
        text_input: Optional[str] = None,
        image_data: Optional[bytes] = None,
        pdf_data: Optional[bytes] = None,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Provide insight into how the agent would process the request without execution."""
        input_description = self._describe_input(text_input, image_data, pdf_data)
        conversation_context = ""
        if session_id:
            conversation_context = self._get_conversation_context(session_id)

        tool_selection = await self._select_tools(input_description, conversation_context)

        explanation: Dict[str, Any] = {
            "input_description": input_description,
            "conversation_context": conversation_context,
            "selected_tools": tool_selection,
            "available_tools": list(self.tools.keys()),
        }

        if text_input:
            if any(tool.get("name") == "analyze_text" for tool in tool_selection):
                try:
                    explanation["rag_context_preview"] = self.llm.gather_rag_context(text_input, limit=3)
                except Exception as exc:
                    explanation["rag_context_preview_error"] = str(exc)

        return explanation
    
    def _describe_input(self, text: Optional[str], image_data: Optional[bytes], pdf_data: Optional[bytes]) -> str:
        """Describe the input for the LLM to understand what tools might be needed"""
        descriptions = []
        
        if text:
            descriptions.append(f"Text input: '{text[:100]}{'...' if len(text) > 100 else ''}'")
        
        if image_data:
            descriptions.append(f"Image file: {len(image_data)} bytes")
        
        if pdf_data:
            descriptions.append(f"PDF file: {len(pdf_data)} bytes")
        
        return " | ".join(descriptions) if descriptions else "No input provided"
    
    def _get_conversation_context(self, session_id: str) -> str:
        """Get recent conversation context with enhanced retrieval"""
        try:
            context_data = self.mcp.get_conversation_context(session_id)
            
            # Get recent messages for better context
            recent_messages = context_data.get("recent_messages", [])
            context_parts = []
            
            # Include last few exchanges for context
            for msg in recent_messages[-6:]:  # Last 3 exchanges (user + assistant)
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role and content:
                    context_parts.append(f"{role}: {content[:200]}...")
            
            # Also include any summary context
            summary = context_data.get("context", "")
            if summary:
                context_parts.append(f"Summary: {summary}")
            
            full_context = "\n".join(context_parts)
            return full_context[:800]  # Limit but allow more space
        except Exception as e:
            logger.warning(f"Error getting conversation context: {e}")
            return ""
    
    def _generate_cache_key(self, input_description: str, conversation_context: str) -> str:
        """Generate a cache key for tool selection"""
        import hashlib
        
        # Create a simplified key based on input type and recent context
        input_type = "text"
        if "Image file" in input_description:
            input_type = "image"
        elif "PDF file" in input_description:
            input_type = "pdf"
        
        # Use last 100 chars of context to detect similar follow-ups
        context_key = conversation_context[-100:].strip().lower()
        
        # Extract key terms from input
        input_lower = input_description.lower()
        key_terms = []
        if "weather" in input_lower:
            key_terms.append("weather")
        if any(term in input_lower for term in ["medicine", "drug", "medication"]):
            key_terms.append("medicine")
        if any(term in input_lower for term in ["what", "how", "why", "explain"]):
            key_terms.append("question")
        
        # Create cache key
        cache_components = [input_type] + key_terms + [context_key[:20]]
        cache_string = "|".join(cache_components)
        
        return hashlib.md5(cache_string.encode()).hexdigest()[:12]
    
    async def _select_tools(self, input_description: str, conversation_context: str) -> List[Dict[str, Any]]:
        """Ask LLM to select which tools to use (with improved logic)"""
        
        # Check cache first for simple patterns
        cache_key = self._generate_cache_key(input_description, conversation_context)
        if cache_key in self._tool_selection_cache:
            cached_selection = self._tool_selection_cache[cache_key]
            logger.info(f"🚀 Using cached tool selection: {[t['name'] for t in cached_selection]}")
            return cached_selection
        
        # Smart pre-filtering based on clear, unambiguous input types
        if "Image file" in input_description:
            return [{"name": "analyze_image", "reason": "Image input detected", "parameters": {}}]
        elif "PDF file" in input_description:
            return [{"name": "analyze_pdf", "reason": "PDF input detected", "parameters": {}}]
        elif any(term in input_description.lower() for term in ["weather", "temperature", "rain", "sunny"]):
            return [{"name": "get_weather", "reason": "Weather query detected", "parameters": {}}]

        # For text queries, use improved LLM selection
        tools_summary = self._get_tools_summary()

        prompt = f"""Select the best tool for this medical assistant request.

AVAILABLE TOOLS:
{tools_summary}

REQUEST: {input_description}
CONTEXT: {conversation_context[-200:] if conversation_context else 'No recent context'}

SELECTION RULES:
1. For medical questions/symptoms or policy/regulation topics → analyze_text
2. For drug/medicine-specific information (dosage, brands, prices) → get_medicine_info  
3. For weather questions → get_weather
4. For general chat/greetings → analyze_text
5. For memory/history questions → access_memory

Select ONE primary tool. Return JSON format:
{{"name": "tool_name", "reason": "brief explanation"}}

Selection:"""

        try:
            response = self.llm.generate_text_response(
                prompt, 
                system_prompt="You are a medical tool selector. Always respond with valid JSON containing exactly one tool selection."
            )
            
            # Extract JSON from response
            tool_selection = self._extract_single_tool_json(response)
            if not tool_selection:
                # Fallback to smart default
                tool_selection = self._smart_fallback_selection(input_description)
            else:
                # Convert single tool to list format
                tool_selection = [tool_selection]
            
            # Cache the selection
            if len(self._tool_selection_cache) > 50:
                oldest_keys = list(self._tool_selection_cache.keys())[:10]
                for key in oldest_keys:
                    del self._tool_selection_cache[key]
            
            self._tool_selection_cache[cache_key] = tool_selection
            logger.info(f"🔧 Selected tool: {[t['name'] for t in tool_selection]}")
            
            return tool_selection
            
        except Exception as e:
            logger.warning(f"Error in tool selection: {e}")
            return self._smart_fallback_selection(input_description)
    
    def _extract_json(self, text: str) -> Optional[List[Dict[str, Any]]]:
        """Extract JSON from LLM response"""
        import re
        
        # Try to find JSON in the response
        json_match = re.search(r'\[.*\]', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        
        # Try to find individual JSON objects
        objects = re.findall(r'\{[^}]*\}', text)
        if objects:
            try:
                result = []
                for obj in objects:
                    result.append(json.loads(obj))
                return result
            except json.JSONDecodeError:
                pass
        
        return None
    
    def _get_tools_summary(self) -> str:
        """Get concise summary of available tools"""
        tools = [
            "analyze_text: Medical questions, symptoms, general health queries",
            "get_medicine_info: Drug information, prices, Bangladesh brands",
            "get_weather: Weather conditions and forecasts",
            "access_memory: Previous conversation history",
            "web_search: Current information not in knowledge base"
        ]
        return "\n".join(tools)
    
    def _extract_single_tool_json(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract single tool JSON from LLM response"""
        import re
        import json
        
        # Try to find JSON object
        json_match = re.search(r'\{[^}]*\}', text)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        return None
    
    def _smart_fallback_selection(self, input_description: str) -> List[Dict[str, Any]]:
        """Smart fallback tool selection with better logic"""
        input_lower = input_description.lower()
        
        policy_terms = [
            "policy", "policies", "regulation", "regulations", "law", "laws",
            "legal", "guideline", "guidelines", "rule", "rules", "act", "authority"
        ]

        if any(term in input_lower for term in policy_terms):
            return [{"name": "analyze_text", "reason": "Policy or regulation query detected", "parameters": {}}]
        elif any(term in input_lower for term in ["medicine", "drug", "medication", "tablet", "mg", "dosage", "dose", "capsule", "syrup"]):
            return [{"name": "get_medicine_info", "reason": "Medicine query detected", "parameters": {}}]
        elif any(term in input_lower for term in ["weather", "temperature", "rain", "sunny"]):
            return [{"name": "get_weather", "reason": "Weather query detected", "parameters": {}}]
        elif any(term in input_lower for term in ["hello", "hi", "hey", "good morning", "good evening"]):
            return [{"name": "analyze_text", "reason": "Greeting detected", "parameters": {}}]
        elif any(term in input_lower for term in ["anatomy", "human body", "organ", "muscle", "bone"]):
            return [{"name": "analyze_text", "reason": "Anatomy query detected", "parameters": {}}]
        else:
            return [{"name": "analyze_text", "reason": "General medical query", "parameters": {}}]
    
    def _fallback_tool_selection(self, input_description: str) -> List[Dict[str, Any]]:
        """Legacy fallback - use smart fallback instead"""
        return self._smart_fallback_selection(input_description)
    
    async def _execute_tools(
        self, 
        tool_selection: List[Dict[str, Any]], 
        text_input: Optional[str], 
        image_data: Optional[bytes], 
        pdf_data: Optional[bytes],
        session_id: str
    ) -> List[Dict[str, Any]]:
        """Execute the selected tools"""
        results = []
        
        for tool_spec in tool_selection:
            tool_name = tool_spec.get("name")
            if tool_name not in self.tools:
                continue
            
            try:
                tool = self.tools[tool_name]
                
                # Prepare parameters based on tool type
                if tool_name == "analyze_text" and text_input:
                    result = await tool.function(text_input, session_id)
                elif tool_name == "analyze_image" and image_data:
                    result = await tool.function(image_data, text_input, session_id)
                elif tool_name == "analyze_pdf" and pdf_data:
                    result = await tool.function(pdf_data, text_input, session_id)
                elif tool_name == "get_weather" and text_input:
                    result = await tool.function(text_input)
                elif tool_name == "get_medicine_info" and text_input:
                    result = await tool.function(text_input, False)
                elif tool_name == "access_memory" and text_input:
                    result = await tool.function(text_input, session_id)
                else:
                    result = {"error": f"Tool {tool_name} cannot be executed with provided inputs"}
                
                results.append({
                    "tool": tool_name,
                    "result": result,
                    "reason": tool_spec.get("reason", "")
                })
                
            except Exception as e:
                logger.exception(f"Error executing tool {tool_name}")
                results.append({
                    "tool": tool_name,
                    "result": {"error": str(e)},
                    "reason": tool_spec.get("reason", "")
                })
        
        return results
    
    async def _generate_final_response(self, results: List[Dict[str, Any]], original_query: Optional[str], conversation_context: str) -> str:
        """Generate final response based on tool results - improved with direct response forwarding"""
        
        if not results:
            return "I couldn't find the right tools to help with your request."
        
        # For single tool execution, return the response directly without synthesis
        if len(results) == 1:
            tool_result = results[0]["result"]
            if isinstance(tool_result, dict) and "response" in tool_result:
                return tool_result["response"]
            elif isinstance(tool_result, dict) and "error" in tool_result:
                return f"I encountered an error: {tool_result['error']}"
        
        # For multiple tools, combine intelligently
        primary_response = None
        error_messages = []
        
        for result in results:
            tool_result = result["result"]
            if isinstance(tool_result, dict):
                if "response" in tool_result and not primary_response:
                    primary_response = tool_result["response"]
                elif "error" in tool_result:
                    error_messages.append(f"Issue with {result['tool']}: {tool_result['error']}")
        
        # Return primary response with error context if needed
        if primary_response:
            if error_messages:
                return f"{primary_response}\n\n*Note: {'; '.join(error_messages)}*"
            return primary_response
        
        # Fallback synthesis only if no direct response available
        combined_results = []
        for result in results:
            tool_name = result["tool"]
            tool_result = result["result"]
            
            if isinstance(tool_result, dict):
                if "response" in tool_result:
                    combined_results.append(tool_result["response"])
                elif "error" in tool_result:
                    combined_results.append(f"Error with {tool_name}: {tool_result['error']}")
        
        if combined_results:
            return "\n\n".join(combined_results)
        
        return "I processed your request but couldn't generate a complete response."
    
