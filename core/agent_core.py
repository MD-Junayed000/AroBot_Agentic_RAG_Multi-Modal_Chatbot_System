# core/agent_core.py
"""
LLM-as-Agent Core: Uses LLM to decide which tools to use for any input
"""
from __future__ import annotations

import json
import logging
import hashlib
import time
import io
from typing import Dict, Any, List, Optional, Union, Callable
from dataclasses import dataclass
from typing import get_type_hints
import inspect
from difflib import SequenceMatcher

from core.llm_modular import ModularLLMHandler as LLMHandler
from utils.web_search import WebSearchTool
from mcp_server.mcp_handler import MCPHandler
from langsmith import traceable

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

@dataclass
class RAGCacheEntry:
    """Cache entry for RAG results with metadata"""
    query_hash: str
    context_chunks: List[str]
    query_type: str
    timestamp: float
    hit_count: int = 1

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
        
        # RAG optimization caches
        self._rag_cache: Dict[str, RAGCacheEntry] = {}
        self._query_similarity_cache: Dict[str, str] = {}  # query -> similar query hash
        self._max_rag_cache_size = 100
        self._rag_cache_ttl = 3600  # 1 hour TTL
        self._similarity_threshold = 0.8  # 80% similarity threshold
        
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
    
    def _generate_query_hash(self, query: str, query_type: str) -> str:
        """Generate a hash for query caching"""
        normalized_query = query.lower().strip()
        return hashlib.md5(f"{normalized_query}_{query_type}".encode()).hexdigest()
    
    def _find_similar_query(self, query: str) -> Optional[str]:
        """Find similar query in cache using sequence matching"""
        normalized_query = query.lower().strip()
        
        # Check if we have a cached similar query
        if normalized_query in self._query_similarity_cache:
            return self._query_similarity_cache[normalized_query]
        
        # Find most similar query in cache
        best_match = None
        best_similarity = 0.0
        
        for cached_query in self._query_similarity_cache.keys():
            similarity = SequenceMatcher(None, normalized_query, cached_query).ratio()
            if similarity > best_similarity and similarity >= self._similarity_threshold:
                best_similarity = similarity
                best_match = cached_query
        
        if best_match:
            self._query_similarity_cache[normalized_query] = best_match
            logger.info(f"Found similar query: {similarity:.2f} similarity")
            return best_match
        
        return None
    
    def _get_cached_rag_context(self, query: str, query_type: str) -> Optional[List[str]]:
        """Get cached RAG context if available and not expired"""
        query_hash = self._generate_query_hash(query, query_type)
        current_time = time.time()
        
        # Check direct cache hit
        if query_hash in self._rag_cache:
            entry = self._rag_cache[query_hash]
            if current_time - entry.timestamp < self._rag_cache_ttl:
                entry.hit_count += 1
                logger.info(f"RAG cache hit for query: {query[:50]}...")
                return entry.context_chunks
            else:
                # Remove expired entry
                del self._rag_cache[query_hash]
        
        # Check for similar query
        similar_query = self._find_similar_query(query)
        if similar_query:
            similar_hash = self._generate_query_hash(similar_query, query_type)
            if similar_hash in self._rag_cache:
                entry = self._rag_cache[similar_hash]
                if current_time - entry.timestamp < self._rag_cache_ttl:
                    entry.hit_count += 1
                    logger.info(f"RAG cache hit for similar query: {similarity:.2f} similarity")
                    return entry.context_chunks
        
        return None
    
    def _cache_rag_context(self, query: str, query_type: str, context_chunks: List[str]):
        """Cache RAG context with TTL and size management"""
        query_hash = self._generate_query_hash(query, query_type)
        current_time = time.time()
        
        # Clean expired entries
        expired_keys = [
            key for key, entry in self._rag_cache.items()
            if current_time - entry.timestamp > self._rag_cache_ttl
        ]
        for key in expired_keys:
            del self._rag_cache[key]
        
        # Manage cache size
        if len(self._rag_cache) >= self._max_rag_cache_size:
            # Remove least recently used entries (simple LRU by hit count)
            sorted_entries = sorted(
                self._rag_cache.items(),
                key=lambda x: (x[1].hit_count, x[1].timestamp)
            )
            # Remove bottom 20% of entries
            remove_count = self._max_rag_cache_size // 5
            for key, _ in sorted_entries[:remove_count]:
                del self._rag_cache[key]
        
        # Add new entry
        self._rag_cache[query_hash] = RAGCacheEntry(
            query_hash=query_hash,
            context_chunks=context_chunks,
            query_type=query_type,
            timestamp=current_time
        )
        
        # Update similarity cache
        self._query_similarity_cache[query.lower().strip()] = query.lower().strip()
    
    def _optimize_context_chunks(self, context_chunks: List[str], max_chunks: int = 3, max_chars: int = 800) -> List[str]:
        """Optimize context chunks by limiting size and count"""
        if not context_chunks:
            return []
        
        optimized_chunks = []
        total_chars = 0
        
        for chunk in context_chunks:
            if len(optimized_chunks) >= max_chunks:
                break
                
            # Truncate chunk if too long
            if len(chunk) > max_chars:
                chunk = chunk[:max_chars] + "..."
            
            # Check if adding this chunk would exceed reasonable limits
            if total_chars + len(chunk) > max_chunks * max_chars:
                break
                
            optimized_chunks.append(chunk)
            total_chars += len(chunk)
        
        logger.info(f"Optimized context: {len(optimized_chunks)} chunks, {total_chars} chars")
        return optimized_chunks
    
    def _smart_context_filtering(self, query: str, context_chunks: List[str]) -> List[str]:
        """Filter context chunks based on query type and relevance"""
        if not context_chunks:
            return []
        
        query_lower = query.lower()
        filtered_chunks = []
        
        # Define query type patterns
        medicine_patterns = ["medicine", "drug", "medication", "tablet", "capsule", "dosage", "mg", "ml"]
        symptom_patterns = ["symptom", "pain", "fever", "headache", "ache", "hurt"]
        disease_patterns = ["disease", "condition", "disorder", "syndrome", "infection"]
        
        # Determine query type
        is_medicine_query = any(pattern in query_lower for pattern in medicine_patterns)
        is_symptom_query = any(pattern in query_lower for pattern in symptom_patterns)
        is_disease_query = any(pattern in query_lower for pattern in disease_patterns)
        
        for chunk in context_chunks:
            chunk_lower = chunk.lower()
            relevance_score = 0
            
            # Medicine-specific filtering
            if is_medicine_query:
                if any(term in chunk_lower for term in ["dose", "dosage", "indication", "contraindication", "side effect"]):
                    relevance_score += 3
                if any(term in chunk_lower for term in ["tablet", "capsule", "syrup", "injection"]):
                    relevance_score += 2
            
            # Symptom-specific filtering
            if is_symptom_query:
                if any(term in chunk_lower for term in ["symptom", "sign", "manifestation", "presentation"]):
                    relevance_score += 3
                if any(term in chunk_lower for term in ["cause", "etiology", "pathophysiology"]):
                    relevance_score += 2
            
            # Disease-specific filtering
            if is_disease_query:
                if any(term in chunk_lower for term in ["disease", "condition", "disorder", "syndrome"]):
                    relevance_score += 3
                if any(term in chunk_lower for term in ["treatment", "therapy", "management"]):
                    relevance_score += 2
            
            # General medical relevance
            if any(term in chunk_lower for term in ["clinical", "medical", "health", "patient"]):
                relevance_score += 1
            
            # Only include chunks with sufficient relevance
            if relevance_score >= 1:
                filtered_chunks.append(chunk)
        
        logger.info(f"Smart filtering: {len(context_chunks)} -> {len(filtered_chunks)} chunks")
        return filtered_chunks

    # Tool definitions using decorators
    @tool(
        name="analyze_text",
        description="Analyze text queries, medical questions, symptoms, or general conversation. Use for any text-based medical or health questions.",
        category="medical",
        priority=5
    )
    @traceable(name="analyze_text_tool")
    async def _tool_analyze_text(self, query: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Analyze text using optimized RAG → LLM pipeline for medical queries"""
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
            
            # Check if this is a medicine-specific query
            medicine_indicators = [
                "medicine", "drug", "medication", "tablet", "capsule", "syrup", "injection",
                "dosage", "dose", "mg", "ml", "side effect", "contraindication",
                "paracetamol", "aspirin", "antibiotic", "vitamin", "prescription"
            ]
            
            is_medicine_query = any(indicator in query_lower for indicator in medicine_indicators)
            
            # Check for medical query indicators
            medical_indicators = [
                "symptom", "disease", "treatment", "diagnosis", "health", "medical",
                "pain", "fever", "headache", "infection", "allergy", "diabetes",
                "blood pressure", "heart", "kidney", "liver", "stomach"
            ]
            
            is_medical_query = any(indicator in query_lower for indicator in medical_indicators) or is_medicine_query
            
            if is_medical_query:
                logger.info(f"Processing medical query with optimized RAG: {query[:50]}...")
                
                # Determine query type for caching
                query_type = "medicine" if is_medicine_query else "medical"
                
                # Step 1: Check RAG cache first
                rag_context = self._get_cached_rag_context(query, query_type)
                
                if not rag_context:
                    # Step 2: Gather relevant context using RAG (only if not cached)
                    try:
                        raw_context = self.llm.gather_rag_context(query, limit=4)
                        logger.info(f"Retrieved {len(raw_context)} raw RAG context chunks")
                        
                        # Step 3: Apply smart context filtering
                        filtered_context = self._smart_context_filtering(query, raw_context)
                        
                        # Step 4: Optimize context chunks (max 3 chunks of 800 chars each)
                        rag_context = self._optimize_context_chunks(filtered_context, max_chunks=3, max_chars=800)
                        
                        # Step 5: Cache the optimized context
                        self._cache_rag_context(query, query_type, rag_context)
                        
                        logger.info(f"Optimized to {len(rag_context)} RAG context chunks")
                    except Exception as e:
                        logger.warning(f"RAG context gathering failed: {e}")
                        rag_context = []
                
                # Step 6: Use LLM to generate response with optimized RAG context
                if rag_context:
                    # Format RAG context
                    context_text = "\n\n---MEDICAL KNOWLEDGE---\n".join(rag_context)
                    
                    # Create comprehensive prompt with RAG context
                    rag_prompt = f"""
                    Using the provided MEDICAL KNOWLEDGE context, answer the following medical question comprehensively and accurately.
                    
                    Structure your response appropriately based on the question type:
                    - For medicine queries: Include uses, dosage, side effects, contraindications
                    - For symptom queries: Include possible causes, when to seek help, general management
                    - For disease queries: Include overview, symptoms, treatment options, prevention
                    
                    MEDICAL KNOWLEDGE CONTEXT:
                    {context_text}
                    
                    QUESTION: {query}
                    
                    Important: Base your response primarily on the provided medical knowledge context. If the context doesn't contain specific information needed to answer the question, indicate this clearly and provide safe general guidance. Always recommend consulting a healthcare provider for personalized medical advice.
                    """
                    
                    # Generate response using medical system prompt
                    from core.prompts.system import get_system_prompt
                    medical_response = self.llm.text_gen.generate_response(
                        rag_prompt,
                        system_prompt=get_system_prompt("medical")
                    )
                    
                    # Add source attribution
                    medical_response += f"\n\n*Information compiled from medical knowledge base ({len(rag_context)} sources) and clinical references.*"
                    
                    return {
                        "response": medical_response, 
                        "query_type": "medical_with_rag",
                        "rag_sources": len(rag_context),
                        "cache_hit": True if self._get_cached_rag_context(query, query_type) else False
                    }
                
                else:
                    # Fallback: Use medical agent without RAG context
                    logger.info("No RAG context found, using medical agent fallback")
                    from agents.medical_agent import MedicalAgent
                    agent = MedicalAgent()
                    result = agent.handle_text_query(query, session_id=session_id)
                    result["query_type"] = "medical_fallback"
                    return result
            
            # Check for anatomy/educational questions
            elif any(term in query_lower for term in ["anatomy", "human body", "tell about", "explain", "what is"]):
                # Use general knowledge handler with potential RAG context
                try:
                    # Check cache for educational queries
                    rag_context = self._get_cached_rag_context(query, "educational")
                    
                    if not rag_context:
                        raw_context = self.llm.gather_rag_context(query, limit=2)
                        if raw_context:
                            # Apply smart filtering and optimization
                            filtered_context = self._smart_context_filtering(query, raw_context)
                            rag_context = self._optimize_context_chunks(filtered_context, max_chunks=2, max_chars=600)
                            self._cache_rag_context(query, "educational", rag_context)
                    
                    if rag_context:
                        context_text = "\n".join(rag_context)
                        educational_prompt = f"""
                        Using the provided context, answer the educational question about {query}.
                        
                        CONTEXT:
                        {context_text}
                        
                        QUESTION: {query}
                        
                        Provide a clear, educational response. If the context doesn't fully answer the question, supplement with general knowledge while indicating what comes from the provided context.
                        """
                        
                        response = self.llm.text_gen.generate_response(
                            educational_prompt,
                            system_prompt=get_system_prompt("general")
                        )
                        return {"response": response, "query_type": "educational_with_context"}
                    else:
                        response = self.llm.answer_general_knowledge(query)
                        return {"response": response, "query_type": "educational"}
                except Exception as e:
                    logger.warning(f"Educational query with context failed: {e}")
                    response = self.llm.answer_general_knowledge(query)
                    return {"response": response, "query_type": "educational"}
            
            else:
                # General query - use basic LLM response
                response = self.llm.generate_text_response(query)
                return {"response": response, "query_type": "general"}
                
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
        description="Analyze images including prescriptions, medical diagrams, anatomy charts, or general images. Performs OCR and visual analysis with intelligent classification.",
        category="medical",
        priority=4
    )
    @traceable(name="analyze_image_tool")
    async def _tool_analyze_image(self, image_data: bytes, question: Optional[str] = None, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Enhanced image analysis with intelligent multi-stage classification"""
        try:
            # Get session context for better classification
            session_context = self._get_conversation_context(session_id) if session_id else ""
            
            # Initialize multimodal processor for classification
            from core.multimodal_processor import MultiModalProcessor
            processor = MultiModalProcessor()
            
            # Step 1: Classify the image using advanced classification
            classification = processor.classify_image(image_data, question, session_context)
            
            logger.info(f"Image classified as: {classification.category.value} with confidence {classification.confidence:.2f}")
            logger.info(f"Classification reasoning: {classification.reasoning}")
            
            # Step 2: Route to appropriate handler based on classification
            if classification.category.value == "prescription":
                return await self._handle_prescription_image(image_data, question, classification)
            elif classification.category.value in ["medicine_package", "medicine_bottle", "medicine_strip"]:
                return await self._handle_medicine_package_image(image_data, question, classification)
            elif classification.category.value == "lab_results":
                return await self._handle_lab_results_image(image_data, question, classification)
            elif classification.category.value == "anatomy_diagram":
                return await self._handle_anatomy_diagram_image(image_data, question, classification)
            elif classification.category.value == "medical_chart":
                return await self._handle_medical_chart_image(image_data, question, classification)
            elif classification.category.value in ["xray_scan", "medical_report"]:
                return await self._handle_medical_scan_image(image_data, question, classification)
            else:
                return await self._handle_general_image(image_data, question, classification)
            
        except Exception as e:
            logger.exception(f"Error in enhanced image analysis: {e}")
            # Fallback to basic analysis
            try:
                ocr_result = self.llm.ocr_only(image_data)
                ocr_text = ocr_result.get("raw_text", "")
                
                if question:
                    response = f"I encountered an issue with advanced analysis, but I can see some text in the image:\n\n{ocr_text[:500]}"
                    if len(ocr_text) > 500:
                        response += "..."
                else:
                    response = f"I had trouble with detailed analysis, but extracted this text: {ocr_text[:200]}{'...' if len(ocr_text) > 200 else ''}"
                
                return {"response": response, "error": f"Advanced analysis failed: {str(e)}", "status": "partial_success"}
            except Exception as e2:
                return {"error": f"Complete analysis failure: {str(e)} | {str(e2)}", "status": "error"}

    async def _handle_prescription_image(self, image_data: bytes, question: Optional[str], classification) -> Dict[str, Any]:
        """Handle prescription images with enhanced processing"""
        try:
            # Extract OCR text
            ocr_result = self.llm.ocr_only(image_data)
            ocr_text = ocr_result.get("raw_text", "")
            
            # Enhanced medicine name extraction
            extracted_medicine_names = self._extract_medicine_names_from_ocr(ocr_text)
            
            # Enhanced vision analysis for prescriptions
            vision_prompt = (
                "Analyze this prescription image carefully. Extract:\n"
                "1. Doctor's name and clinic information from the header\n"
                "2. All medicine names (look for Tab, Syp, Cap, Inj prefixes)\n"
                "3. Quantities and dosage instructions for each medicine\n"
                "4. Any special instructions or notes\n"
                "Provide a clear, structured analysis of what you can see."
            )
            vision_response = self.llm.generate_vision_response(vision_prompt, image_data=image_data)
            
            # Extract medicine names from vision analysis
            vision_extracted_names = self._extract_medicine_names_from_text(vision_response)
            all_medicine_names = list(set(extracted_medicine_names + vision_extracted_names))
            
            # Build comprehensive response
            response_parts = []
            response_parts.append(f"**Prescription Analysis** (Confidence: {classification.confidence:.1%})")
            response_parts.append(vision_response)
            
            # Process each medicine
            if all_medicine_names:
                response_parts.append(f"\n**Medicine Information:**")
                
                for i, medicine_name in enumerate(all_medicine_names[:3], 1):
                    try:
                        medicine_info_result = await self._tool_get_medicine_info(medicine_name, want_price=False)
                        if medicine_info_result.get("response"):
                            response_parts.append(f"\n**{i}. {medicine_name.title()}**")
                            response_parts.append(medicine_info_result["response"])
                    except Exception as e:
                        logger.warning(f"Failed to get info for {medicine_name}: {e}")
                        continue
            else:
                response_parts.append("\n**Note:** I had some difficulty reading the medicine names clearly. You can ask your pharmacist for details about each medication.")
            
            # Add usage instructions if requested
            if question and "how to take" in question.lower():
                instruction_response = self._generate_usage_instructions(vision_response, all_medicine_names)
                response_parts.append(f"\n{instruction_response}")
            
            response_parts.append("\n**Important:** Always follow your doctor's instructions and consult your pharmacist if you have questions!")
            
            full_response = "\n".join(response_parts)
            
            return {
                "response": full_response,
                "classification": classification.category.value,
                "confidence": classification.confidence,
                "reasoning": classification.reasoning,
                "extracted_medicines": all_medicine_names,
                "ocr_data": ocr_result,
                "status": "success"
            }
            
        except Exception as e:
            logger.exception(f"Error handling prescription image: {e}")
            return {"error": str(e), "status": "error"}

    async def _handle_medicine_package_image(self, image_data: bytes, question: Optional[str], classification) -> Dict[str, Any]:
        """Handle medicine package/bottle/strip images"""
        try:
            # Extract text from package
            ocr_result = self.llm.ocr_only(image_data)
            ocr_text = ocr_result.get("raw_text", "")
            
            # Analyze package with specific prompt
            vision_prompt = (
                "Analyze this medicine package/bottle/strip. Identify:\n"
                "1. Medicine name and brand\n"
                "2. Strength/dosage (mg, ml, etc.)\n"
                "3. Manufacturer information\n"
                "4. Expiry date if visible\n"
                "5. Batch number if visible\n"
                "6. Any usage instructions on the package"
            )
            vision_response = self.llm.generate_vision_response(vision_prompt, image_data=image_data)
            
            # Extract medicine name
            medicine_names = self._extract_medicine_names_from_text(vision_response) + self._extract_medicine_names_from_ocr(ocr_text)
            main_medicine = medicine_names[0] if medicine_names else "Unknown Medicine"
            
            # Get detailed medicine information
            medicine_details = {}
            if main_medicine != "Unknown Medicine":
                try:
                    medicine_info_result = await self._tool_get_medicine_info(main_medicine, want_price=False)
                    medicine_details = medicine_info_result
                except Exception as e:
                    logger.warning(f"Could not get details for {main_medicine}: {e}")
            
            # Build response
            response_parts = []
            response_parts.append(f"**Medicine Package Analysis** (Confidence: {classification.confidence:.1%})")
            response_parts.append(vision_response)
            
            if medicine_details.get("response"):
                response_parts.append(f"\n**Detailed Information about {main_medicine}:**")
                response_parts.append(medicine_details["response"])
            
            # Safety reminders for packages
            response_parts.append("\n**Package Safety Tips:**")
            response_parts.append("• Check expiry date before use")
            response_parts.append("• Store as directed on package")
            response_parts.append("• Keep out of reach of children")
            response_parts.append("• Follow dosage instructions carefully")
            
            full_response = "\n".join(response_parts)
            
            return {
                "response": full_response,
                "classification": classification.category.value,
                "confidence": classification.confidence,
                "reasoning": classification.reasoning,
                "extracted_medicine": main_medicine,
                "package_info": vision_response,
                "status": "success"
            }
            
        except Exception as e:
            logger.exception(f"Error handling medicine package: {e}")
            return {"error": str(e), "status": "error"}

    async def _handle_lab_results_image(self, image_data: bytes, question: Optional[str], classification) -> Dict[str, Any]:
        """Handle lab results and medical test reports"""
        try:
            # Extract text from lab results
            ocr_result = self.llm.ocr_only(image_data)
            ocr_text = ocr_result.get("raw_text", "")
            
            # Analyze lab results with specific prompt
            vision_prompt = (
                "Analyze this lab report/test results. Identify:\n"
                "1. Test names and their values\n"
                "2. Normal/reference ranges\n"
                "3. Any abnormal or flagged results\n"
                "4. Patient information if visible\n"
                "5. Date of test\n"
                "6. Laboratory name\n"
                "Provide a clear summary of the key findings."
            )
            vision_response = self.llm.generate_vision_response(vision_prompt, image_data=image_data)
            
            # Build response
            response_parts = []
            response_parts.append(f"**Lab Results Analysis** (Confidence: {classification.confidence:.1%})")
            response_parts.append(vision_response)
            
            # Add interpretation if specific question asked
            if question and any(term in question.lower() for term in ["normal", "abnormal", "meaning", "interpret"]):
                interpretation_prompt = f"""
                Based on this lab results analysis: {vision_response}
                
                Provide a simple interpretation of what these results might mean. 
                Be careful to:
                1. Explain in simple terms
                2. Highlight any values outside normal ranges
                3. Emphasize the need for doctor consultation
                4. Avoid making diagnoses
                """
                
                from core.prompts.system import get_system_prompt
                interpretation = self.llm.text_gen.generate_response(
                    interpretation_prompt,
                    system_prompt=get_system_prompt("medical")
                )
                
                response_parts.append(f"\n**Simple Interpretation:**")
                response_parts.append(interpretation)
            
            response_parts.append("\n**Important:** Lab results should always be interpreted by a qualified healthcare provider. This analysis is for informational purposes only.")
            
            full_response = "\n".join(response_parts)
            
            return {
                "response": full_response,
                "classification": classification.category.value,
                "confidence": classification.confidence,
                "reasoning": classification.reasoning,
                "lab_data": vision_response,
                "status": "success"
            }
            
        except Exception as e:
            logger.exception(f"Error handling lab results: {e}")
            return {"error": str(e), "status": "error"}

    async def _handle_anatomy_diagram_image(self, image_data: bytes, question: Optional[str], classification) -> Dict[str, Any]:
        """Handle anatomy diagrams and medical illustrations"""
        try:
            # Analyze anatomy diagram
            vision_prompt = (
                "Analyze this anatomy diagram or medical illustration. Identify:\n"
                "1. Body system or organ shown\n"
                "2. Labeled parts and structures\n"
                "3. Any pathological conditions illustrated\n"
                "4. Educational content or key points\n"
                "Provide an educational explanation of what's shown."
            )
            vision_response = self.llm.generate_vision_response(vision_prompt, image_data=image_data)
            
            # Try to get additional context from anatomy knowledge base
            anatomy_context = []
            try:
                # Extract key anatomy terms for RAG search
                anatomy_terms = self._extract_anatomy_terms(vision_response)
                if anatomy_terms:
                    for term in anatomy_terms[:2]:  # Limit to 2 terms
                        context = self.llm.gather_rag_context(f"anatomy {term}", limit=2)
                        anatomy_context.extend(context)
            except Exception as e:
                logger.warning(f"Could not get anatomy context: {e}")
            
            # Build response
            response_parts = []
            response_parts.append(f"**Anatomy Diagram Analysis** (Confidence: {classification.confidence:.1%})")
            response_parts.append(vision_response)
            
            # Add educational context if available
            if anatomy_context:
                response_parts.append(f"\n**Educational Context:**")
                for i, context in enumerate(anatomy_context[:2], 1):
                    response_parts.append(f"{i}. {context[:300]}{'...' if len(context) > 300 else ''}")
            
            # Answer specific questions about the diagram
            if question and any(term in question.lower() for term in ["function", "purpose", "what does", "how does"]):
                educational_prompt = f"""
                Based on this anatomy diagram analysis: {vision_response}
                
                Answer this question: {question}
                
                Provide an educational explanation suitable for learning anatomy.
                """
                
                from core.prompts.system import get_system_prompt
                educational_response = self.llm.text_gen.generate_response(
                    educational_prompt,
                    system_prompt=get_system_prompt("general")
                )
                
                response_parts.append(f"\n**Answer to your question:**")
                response_parts.append(educational_response)
            
            full_response = "\n".join(response_parts)
            
            return {
                "response": full_response,
                "classification": classification.category.value,
                "confidence": classification.confidence,
                "reasoning": classification.reasoning,
                "diagram_analysis": vision_response,
                "status": "success"
            }
            
        except Exception as e:
            logger.exception(f"Error handling anatomy diagram: {e}")
            return {"error": str(e), "status": "error"}

    async def _handle_medical_chart_image(self, image_data: bytes, question: Optional[str], classification) -> Dict[str, Any]:
        """Handle medical charts, graphs, and measurement tools"""
        try:
            vision_prompt = (
                "Analyze this medical chart or graph. Identify:\n"
                "1. Type of chart/measurement tool\n"
                "2. Scale or measurement units\n"
                "3. Any data points or readings\n"
                "4. Purpose or medical application\n"
                "Explain how to read or interpret this chart."
            )
            vision_response = self.llm.generate_vision_response(vision_prompt, image_data=image_data)
            
            response_parts = []
            response_parts.append(f"**Medical Chart Analysis** (Confidence: {classification.confidence:.1%})")
            response_parts.append(vision_response)
            
            full_response = "\n".join(response_parts)
            
            return {
                "response": full_response,
                "classification": classification.category.value,
                "confidence": classification.confidence,
                "reasoning": classification.reasoning,
                "status": "success"
            }
            
        except Exception as e:
            logger.exception(f"Error handling medical chart: {e}")
            return {"error": str(e), "status": "error"}

    async def _handle_medical_scan_image(self, image_data: bytes, question: Optional[str], classification) -> Dict[str, Any]:
        """Handle X-rays, CT scans, and other medical imaging"""
        try:
            vision_prompt = (
                "Analyze this medical scan image. Note:\n"
                "1. Type of scan (X-ray, CT, MRI, etc.)\n"
                "2. Body part being examined\n"
                "3. Any visible structures or abnormalities\n"
                "4. Image quality and positioning\n"
                "Provide a general description - avoid making diagnoses."
            )
            vision_response = self.llm.generate_vision_response(vision_prompt, image_data=image_data)
            
            response_parts = []
            response_parts.append(f"**Medical Scan Analysis** (Confidence: {classification.confidence:.1%})")
            response_parts.append(vision_response)
            response_parts.append("\n**Important:** Medical scans require professional interpretation by qualified radiologists or doctors. This analysis is for educational purposes only.")
            
            full_response = "\n".join(response_parts)
            
            return {
                "response": full_response,
                "classification": classification.category.value,
                "confidence": classification.confidence,
                "reasoning": classification.reasoning,
                "status": "success"
            }
            
        except Exception as e:
            logger.exception(f"Error handling medical scan: {e}")
            return {"error": str(e), "status": "error"}

    async def _handle_general_image(self, image_data: bytes, question: Optional[str], classification) -> Dict[str, Any]:
        """Handle general images with OCR and basic analysis"""
        try:
            # Extract text
            ocr_result = self.llm.ocr_only(image_data)
            ocr_text = ocr_result.get("raw_text", "")
            
            # General image analysis
            if question:
                vision_response = self.llm.generate_vision_response(question, image_data=image_data)
                if ocr_text.strip():
                    response = f"**Image Analysis:**\n{vision_response}\n\n**Text Found:**\n{ocr_text}"
                else:
                    response = vision_response
            else:
                brief = self.llm.default_image_brief(image_data)
                if ocr_text.strip():
                    response = f"{brief}\n\n**Text in Image:**\n{ocr_text[:500]}{'...' if len(ocr_text) > 500 else ''}"
                else:
                    response = brief
            
            return {
                "response": response,
                "classification": classification.category.value,
                "confidence": classification.confidence,
                "reasoning": classification.reasoning,
                "ocr_data": ocr_result,
                "status": "success"
            }
            
        except Exception as e:
            logger.exception(f"Error handling general image: {e}")
            return {"error": str(e), "status": "error"}

    def _generate_usage_instructions(self, vision_response: str, medicine_names: List[str]) -> str:
        """Generate usage instructions based on prescription analysis"""
        instruction_prompt = f"""
        Based on this prescription analysis: {vision_response}
        
        And these medicines: {', '.join(medicine_names)}
        
        Provide clear, simple instructions on how to take these medicines:
        - When to take each medicine (morning, evening, with food, etc.)
        - How many to take at once
        - How often during the day
        - Any special instructions
        
        Keep it simple and friendly, like explaining to a family member.
        """
        
        try:
            from core.prompts.system import get_system_prompt
            instructions = self.llm.text_gen.generate_response(
                instruction_prompt,
                system_prompt=get_system_prompt("medical")
            )
            return f"**How to Take Your Medicines:**\n{instructions}"
        except Exception as e:
            logger.warning(f"Could not generate usage instructions: {e}")
            return "**How to Take Your Medicines:**\nPlease follow the instructions on your prescription or consult your pharmacist for detailed guidance."


    def _extract_medicine_names_from_ocr(self, ocr_text: str) -> List[str]:
        """Extract medicine names from OCR text using enhanced patterns"""
        import re
        
        if not ocr_text:
            return []
        
        text_lower = ocr_text.lower()
        medicine_names = []
        
        # Enhanced medicine patterns for prescription format
        medicine_patterns = [
            r'tab\.?\s+(\w+)',  # Tab. or Tab
            r'syp\.?\s+(\w+)',  # Syp. or Syp  
            r'cap\.?\s+(\w+)',  # Cap. or Cap
            r'inj\.?\s+(\w+)',  # Inj. or Inj
            r'(\w+)\s+\d+\s*mg',  # Medicine name with mg
            r'(\w+)\s+\d+\s*ml',  # Medicine name with ml
            r'(\w+)\s+\d+\s*mg/\s*\w+',  # Medicine with mg/other unit
            r'(\w+)\s+\d+\s*mg\s*/\s*\w+',  # Medicine with mg / other unit
            r'(\w+)\s+\d+\s*mg\s*/\s*\w+\s*mg',  # Medicine with mg/mg format
        ]
        
        for pattern in medicine_patterns:
            matches = re.findall(pattern, text_lower)
            medicine_names.extend(matches)
        
        # Additional patterns for common medicine formats
        additional_patterns = [
            r'(\w+)\s+\d+\s*mg\s*/\s*\w+\s*mg\s*/\s*\w+',  # Complex format
            r'(\w+)\s+\d+\s*mg\s*/\s*\w+\s*mg\s*/\s*\w+\s*mg',  # Very complex format
        ]
        
        for pattern in additional_patterns:
            matches = re.findall(pattern, text_lower)
            medicine_names.extend(matches)
        
        # Clean and filter results
        cleaned_names = []
        excluded_words = {
            'tablet', 'capsule', 'syrup', 'injection', 'medicine', 'pharma', 
            'limited', 'ltd', 'the', 'and', 'for', 'tab', 'cap', 'syp', 'inj',
            'mg', 'ml', 'g', 'mcg', 'dose', 'dosage'
        }
        
        for name in medicine_names:
            name_clean = name.strip().lower()
            if (len(name_clean) >= 3 and 
                name_clean not in excluded_words and
                not re.match(r'^[^a-z]*$', name_clean) and  # Not just symbols
                len(re.sub(r'[^a-z]', '', name_clean)) >= 3):  # At least 3 letters
                cleaned_names.append(name.strip())
        
        return list(set(cleaned_names))  # Remove duplicates

    def _extract_medicine_names_from_text(self, text: str) -> List[str]:
        """Extract medicine names from analysis text"""
        import re
        
        if not text:
            return []
        
        text_lower = text.lower()
        medicine_names = []
        
        # Look for medicine names in structured responses
        medicine_patterns = [
            r'(\w+)\s+\d+\s*mg',  # Medicine name with mg
            r'(\w+)\s+\d+\s*ml',  # Medicine name with ml
            r'tab\.?\s+(\w+)',  # Tab. or Tab
            r'syp\.?\s+(\w+)',  # Syp. or Syp
            r'cap\.?\s+(\w+)',  # Cap. or Cap
            r'inj\.?\s+(\w+)',  # Inj. or Inj
        ]
        
        for pattern in medicine_patterns:
            matches = re.findall(pattern, text_lower)
            medicine_names.extend(matches)
        
        # Look for medicine names mentioned in descriptions
        description_patterns = [
            r'medicine[:\s]+(\w+)',
            r'drug[:\s]+(\w+)',
            r'medication[:\s]+(\w+)',
            r'prescribed[:\s]+(\w+)',
        ]
        
        for pattern in description_patterns:
            matches = re.findall(pattern, text_lower)
            medicine_names.extend(matches)
        
        # Clean and filter results
        cleaned_names = []
        excluded_words = {
            'tablet', 'capsule', 'syrup', 'injection', 'medicine', 'pharma', 
            'limited', 'ltd', 'the', 'and', 'for', 'tab', 'cap', 'syp', 'inj',
            'mg', 'ml', 'g', 'mcg', 'dose', 'dosage', 'patient', 'doctor'
        }
        
        for name in medicine_names:
            name_clean = name.strip().lower()
            if (len(name_clean) >= 3 and 
                name_clean not in excluded_words and
                not re.match(r'^[^a-z]*$', name_clean) and  # Not just symbols
                len(re.sub(r'[^a-z]', '', name_clean)) >= 3):  # At least 3 letters
                cleaned_names.append(name.strip())
        
        return list(set(cleaned_names))  # Remove duplicates
    def _extract_anatomy_terms(self, text: str) -> List[str]:
        """Extract anatomy-related terms from text"""
        anatomy_terms = []
        text_lower = text.lower()
        
        # Common anatomy terms
        common_terms = [
            'heart', 'lung', 'liver', 'kidney', 'brain', 'stomach', 'intestine',
            'muscle', 'bone', 'joint', 'blood', 'nerve', 'skin', 'eye', 'ear',
            'respiratory', 'cardiovascular', 'digestive', 'nervous', 'skeletal'
        ]
        
        for term in common_terms:
            if term in text_lower:
                anatomy_terms.append(term)
        
        return anatomy_terms[:3]  # Return max 3 terms
    
    @tool(
        name="get_weather",
        description="Get current weather information for a location. Use when users ask about weather conditions.",
        category="utility",
        priority=2
    )
    @traceable(name="get_weather_tool")
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
    @traceable(name="get_medicine_info_tool")
    async def _tool_get_medicine_info(self, medicine_name: str, want_price: bool = False) -> Dict[str, Any]:
        """Get comprehensive medicine information using RAG → LLM pipeline"""
        try:
            # Clean medicine name
            medicine_name_clean = medicine_name.strip().lower()
            
            # Check if user is asking for price specifically
            query_lower = medicine_name_clean
            if any(term in query_lower for term in ["price", "cost", "how much", "rate", "expense"]):
                want_price = True
            
            # Step 1: Use RAG to gather relevant medical knowledge
            logger.info(f"Gathering RAG context for medicine: {medicine_name}")
            
            # Create comprehensive search queries for RAG
            rag_queries = [
                f"{medicine_name} uses indications therapeutic effects",
                f"{medicine_name} dosage administration dose frequency",
                f"{medicine_name} side effects contraindications warnings",
                f"{medicine_name} drug interactions precautions",
                f"{medicine_name} generic name active ingredient"
            ]
            
            if want_price:
                rag_queries.append(f"{medicine_name} price cost Bangladesh pharmacy")
            
            # Gather context from multiple RAG queries
            all_rag_context = []
            for rag_query in rag_queries:
                try:
                    context_chunks = self.llm.gather_rag_context(rag_query, limit=2)
                    all_rag_context.extend(context_chunks)
                except Exception as e:
                    logger.warning(f"RAG query failed for '{rag_query}': {e}")
                    continue
            
            # Remove duplicates and limit context
            unique_contexts = []
            seen_contexts = set()
            for context in all_rag_context:
                if context and context not in seen_contexts and len(context.strip()) > 50:
                    unique_contexts.append(context)
                    seen_contexts.add(context)
                    if len(unique_contexts) >= 5:  # Limit to prevent context overflow
                        break
            
            logger.info(f"Retrieved {len(unique_contexts)} unique context chunks from RAG")
            
            # Step 2: Use LLM to generate comprehensive response with RAG context
            if unique_contexts:
                # Format RAG context
                rag_context_text = "\n\n---MEDICAL KNOWLEDGE---\n".join(unique_contexts)
                
                # Create comprehensive prompt with RAG context
                enhanced_prompt = f"""
                Using the provided MEDICAL KNOWLEDGE context, provide comprehensive information about the medicine '{medicine_name}' in the following structured format:

                **{medicine_name.title()}**

                **Generic Name:** [Active ingredient from context]
                **Drug Class:** [Therapeutic category from context]
                **Uses:**
                • [Primary indication 1 from context]
                • [Primary indication 2 from context]
                • [Additional uses from context]

                **Dosage Information:**
                • Adults: [Standard adult dose from context]
                • Children: [Pediatric dose if mentioned in context]
                • Frequency: [How often to take from context]

                **Important Information:**
                • Side Effects: [Common side effects from context]
                • Contraindications: [When not to use from context]
                • Drug Interactions: [Important interactions from context]
                • Storage: [Storage requirements from context]

                **Bangladesh Context:**
                • Availability: [Local availability if mentioned]
                • Price Range: [If price information is in context]
                • Local Brands: [Bangladesh brand names if mentioned]

                MEDICAL KNOWLEDGE CONTEXT:
                {rag_context_text}

                Important: Base your response primarily on the provided medical knowledge context. If specific information is not available in the context, indicate this clearly. Always recommend consulting a healthcare provider for personalized advice.
                """
                
                # Generate response using medical system prompt
                from core.prompts.system import get_system_prompt
                medical_system_prompt = get_system_prompt("medicine_info")
                
                enhanced_response = self.llm.text_gen.generate_response(
                    enhanced_prompt, 
                    system_prompt=medical_system_prompt
                )
                
            else:
                # Fallback: No RAG context found, use basic medicine lookup
                logger.warning(f"No RAG context found for {medicine_name}, using fallback")
                base_response = self.llm.answer_medicine(medicine_name, want_price=want_price)
                
                fallback_prompt = f"""
                Provide comprehensive information about the medicine '{medicine_name}' in a structured format:
                
                **{medicine_name.title()}**
                
                **Generic Name:** [Active ingredient if known]
                **Uses:** [Main therapeutic uses]
                **Dosage:** [General dosing guidelines]
                **Important Notes:** [Key safety information]
                
                Based on this information: {base_response}
                
                Note: Limited information available. For complete details, consult a healthcare provider or pharmacist.
                """
                
                enhanced_response = self.llm.text_gen.generate_response(
                    fallback_prompt,
                    system_prompt=get_system_prompt("medicine_info")
                )
            
            # Step 3: Enhance with web search for Bangladesh context if needed
            if len(enhanced_response) < 400 or "bangladesh" in query_lower or want_price:
                 try:
                    web_query = f"{medicine_name} Bangladesh medicine information dosage uses side effects"
                    if want_price:
                        web_query += " price"
                    
                    web_results = self.web_search.search_medical_info(web_query, max_results=3)
                    if web_results and web_results.get("results"):
                        web_info = "\n\n**Additional Information from Web Sources:**\n"
                        for i, result in enumerate(web_results["results"][:3], 1):
                            snippet = result.get('snippet', '').strip()
                            if snippet and len(snippet) > 50:
                                web_info += f"{i}. {snippet}\n"
                        
                        if len(web_info) > 50:  # Only add if we got useful info
                            enhanced_response += web_info
                 except Exception as web_error:
                     logger.warning(f"Web search failed for {medicine_name}: {web_error}")
            
            # Add safety disclaimer and source attribution
            enhanced_response += "\n\n**⚠️ Important:** This information is for educational purposes only. Always consult with a qualified healthcare provider before starting, stopping, or changing any medication."
            
            if unique_contexts:
                enhanced_response += f"\n\n*Information compiled from medical knowledge base ({len(unique_contexts)} sources) and clinical references.*"
            
            return {"response": enhanced_response, "rag_sources": len(unique_contexts)}
            
        except Exception as e:
            logger.exception(f"Error in get_medicine_info tool: {e}")
            # Enhanced fallback with structured format
            try:
                fallback_prompt = f"""
                Provide basic information about the medicine '{medicine_name}' in a structured format:
                
                **{medicine_name.title()}**
                
                **What it is:** [Brief description]
                **Common uses:** [Main therapeutic uses]
                **General dosage:** [Typical dosing guidelines]
                **Important notes:** [Key safety information]
                
                Note: For specific dosage and safety information, please consult a healthcare provider or pharmacist.
                """
                
                fallback_response = self.llm.text_gen.generate_response(
                    fallback_prompt,
                    system_prompt=get_system_prompt("default")
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
    @traceable(name="access_memory_tool")
    async def _tool_access_memory(self, query: str, session_id: str) -> Dict[str, Any]:
        """Access conversation memory with enhanced error handling and session management"""
        try:
            # Ensure session exists, create if not
            if not session_id:
                session_result = self.mcp.initialize_session()
                if session_result.get("error"):
                    return {"error": f"Failed to create session: {session_result['error']}"}
                session_id = session_result["session_id"]
            
            # Get conversation context with better error handling
            context_data = self.mcp.get_conversation_context(session_id)
            
            if context_data.get("error"):
                return {"error": f"Memory access failed: {context_data['error']}"}
            
            # Extract useful information from context
            context_text = context_data.get("context", "")
            message_count = context_data.get("message_count", 0)
            prescription_count = context_data.get("prescription_count", 0)
            
            # Generate a more helpful response based on the query
            query_lower = query.lower()
            
            if any(term in query_lower for term in ["history", "previous", "past", "before", "earlier"]):
                if message_count > 0:
                    response = f"**Conversation History:**\n{context_text}\n\n*Total messages: {message_count}*"
                else:
                    response = "No previous conversation history found. This appears to be a new session."
            elif any(term in query_lower for term in ["prescription", "medicine", "medication", "drug"]):
                if prescription_count > 0:
                    response = f"**Prescription History:**\n{context_text}\n\n*Prescriptions analyzed: {prescription_count}*"
                else:
                    response = "No prescription history found in this session."
            elif any(term in query_lower for term in ["summary", "overview", "context"]):
                response = f"**Session Summary:**\n{context_text}\n\n*Session activity: {message_count} messages, {prescription_count} prescriptions*"
            else:
                # General memory access
                if context_text:
                    response = f"**Recent Context:**\n{context_text[:500]}{'...' if len(context_text) > 500 else ''}"
                else:
                    response = "No conversation context available. This appears to be a new session."
            
            return {
                "response": response,
                "session_id": session_id,
                "message_count": message_count,
                "prescription_count": prescription_count,
                "context_length": len(context_text)
            }
            
        except Exception as e:
            logger.exception(f"Error in access_memory tool: {e}")
            # Try to create a new session as fallback
            try:
                session_result = self.mcp.initialize_session()
                if session_result.get("session_id"):
                    return {
                        "response": "I had trouble accessing your previous conversation, but I've started a new session for you. How can I help?",
                        "session_id": session_result["session_id"],
                        "error": f"Memory access failed, created new session: {str(e)}"
                    }
            except Exception as e2:
                logger.error(f"Failed to create fallback session: {e2}")
            
            return {"error": f"Memory access failed: {str(e)}"}
    
    @tool(
        name="web_search",
        description="Search the web for current information, news, or topics not in the knowledge base. Use when local knowledge is insufficient.",
        category="utility",
        priority=2
    )
    @traceable(name="web_search_tool")
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
    
    def get_rag_cache_stats(self) -> Dict[str, Any]:
        """Get RAG cache statistics for monitoring"""
        current_time = time.time()
        active_entries = sum(
            1 for entry in self._rag_cache.values()
            if current_time - entry.timestamp < self._rag_cache_ttl
        )
        
        total_hits = sum(entry.hit_count for entry in self._rag_cache.values())
        
        return {
            "total_entries": len(self._rag_cache),
            "active_entries": active_entries,
            "total_hits": total_hits,
            "cache_hit_rate": total_hits / max(len(self._rag_cache), 1),
            "similarity_cache_size": len(self._query_similarity_cache),
            "max_cache_size": self._max_rag_cache_size,
            "ttl_seconds": self._rag_cache_ttl
        }
    
    def clear_rag_cache(self):
        """Clear RAG cache for testing or memory management"""
        self._rag_cache.clear()
        self._query_similarity_cache.clear()
        logger.info("RAG cache cleared")

    @traceable(name="process_request")
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
                session_result = self.mcp.initialize_session()
                if session_result.get("error"):
                    return AgentResponse(
                        response=f"I encountered an error initializing a new session: {session_result['error']}",
                        tools_used=[],
                        session_id=None,
                        status="error"
                    )
                session_id = session_result.get("session_id")
            
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
    
    @traceable(name="select_tools")
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
    
    @traceable(name="execute_tools")
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
    
    @traceable(name="generate_final_response")
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
    
