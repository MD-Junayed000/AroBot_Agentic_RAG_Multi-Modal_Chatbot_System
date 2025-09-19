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
        description="Analyze images including prescriptions, medical diagrams, anatomy charts, or general images. Performs OCR and visual analysis.",
        category="medical",
        priority=4
    )
    @traceable(name="analyze_image_tool")
    async def _tool_analyze_image(self, image_data: bytes, question: Optional[str] = None, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Analyze image with comprehensive prescription and medical image processing"""
        try:
            # First, always run OCR to extract text
            ocr_result = self.llm.ocr_only(image_data)
            ocr_text = ocr_result.get("raw_text", "")
            
            # Enhanced medicine detection patterns
            medicine_indicators = [
                "tablet", "capsule", "mg", "ml", "dose", "dosage", "rx", "prescription",
                "medicine", "drug", "pharmaceutical", "pharma", "ltd", "limited",
                "syrup", "suspension", "injection", "cream", "ointment", "drops"
            ]
            
            # Check if this looks like a medicine/prescription based on OCR content
            is_medicine = any(indicator in ocr_text.lower() for indicator in medicine_indicators)
            
            # Enhanced medicine name extraction from OCR text
            extracted_medicine_names = self._extract_medicine_names_from_ocr(ocr_text)
            
            if is_medicine or extracted_medicine_names or (question and any(term in question.lower() for term in ["medicine", "prescription", "drug", "tablet"])):
                # Step 1: Enhanced OCR and Vision Analysis for prescriptions
                vision_prompt = (
                    "Analyze this prescription image carefully. Extract:\n"
                    "1. Doctor's name and clinic information from the header\n"
                    "2. All medicine names (look for Tab, Syp, Cap, Inj prefixes)\n"
                    "3. Quantities and dosage instructions for each medicine\n"
                    "4. Any special instructions or notes\n"
                    "Provide a clear, structured analysis of what you can see."
                )
                vision_response = self.llm.generate_vision_response(vision_prompt, image_data=image_data)
                
                # Step 2: Extract medicine names from both OCR and Vision
                all_medicine_names = extracted_medicine_names.copy()
                vision_extracted_names = self._extract_medicine_names_from_text(vision_response)
                all_medicine_names.extend(vision_extracted_names)
                
                # Clean and deduplicate medicine names
                unique_medicine_names = []
                seen_names = set()
                for name in all_medicine_names:
                    clean_name = name.strip().lower()
                    if len(clean_name) > 2 and clean_name not in seen_names:
                        unique_medicine_names.append(name)
                        seen_names.add(clean_name)
                
                logger.info(f"Extracted medicine names: {unique_medicine_names}")
                    
                # Step 3: Build comprehensive response
                response_parts = []
                
                # Add prescription analysis from vision in a friendly way
                response_parts.append("**Here's what I can see in your prescription:**")
                response_parts.append(vision_response)
                
                # Skip showing raw OCR text to users - it's confusing
                # Only show it if it contains useful readable information
                if ocr_text.strip() and len(unique_medicine_names) == 0:
                    # Only show OCR if we couldn't extract medicine names properly
                    clean_ocr = ' '.join(ocr_text.split())  # Clean up spacing
                    if len(clean_ocr) > 30:  # Only if substantial text
                        response_parts.append(f"\n**Text I could read from the image:**\n{clean_ocr}")
                
                # Step 4: RAG → LLM for each medicine with friendly presentation
                if unique_medicine_names:
                    response_parts.append(f"\n**Let me tell you about your medicines:**")
                    
                    for i, medicine_name in enumerate(unique_medicine_names[:3], 1):  # Limit to 3 medicines
                        try:
                            logger.info(f"Getting information for: {medicine_name}")
                            
                            # Use enhanced RAG → LLM pipeline
                            medicine_info_result = await self._tool_get_medicine_info(medicine_name, want_price=False)
                            
                            if medicine_info_result.get("response"):
                                response_parts.append(f"\n**Medicine #{i}: {medicine_name.title()}**")
                                response_parts.append(medicine_info_result["response"])
                                
                                # Log RAG usage (for debugging, not user-facing)
                                rag_sources = medicine_info_result.get("rag_sources", 0)
                                if rag_sources > 0:
                                    logger.info(f"Used {rag_sources} medical sources for {medicine_name}")
                            
                        except Exception as e:
                            logger.warning(f"Failed to get info for {medicine_name}: {e}")
                            continue
                else:
                    response_parts.append("\n**Note:** I had trouble reading the medicine names clearly from this image. For the best information about your medicines, you can:")
                    response_parts.append("• Ask your pharmacist about each medicine")
                    response_parts.append("• Check the medicine packaging for details")
                    response_parts.append("• Contact your doctor's office if you have questions")
                
                # Step 5: Answer specific questions in a helpful way
                if question and question.strip() and "how should i take" in question.lower():
                    instruction_prompt = f"""
                    Based on what I can see in this prescription, help the patient understand how to take their medicines.
                    
                    Prescription details: {vision_response}
                    
                    Please provide clear, simple instructions like:
                    "Here's how to take your medicines:"
                    
                    For each medicine, explain:
                    - When to take it (morning, evening, with food, etc.)
                    - How many to take at once
                    - How often during the day
                    - Any special instructions
                    
                    Keep it simple and friendly, like you're talking to a family member.
                    """
                    
                    from core.prompts.system import get_system_prompt
                    instruction_response = self.llm.text_gen.generate_response(
                        instruction_prompt,
                        system_prompt=get_system_prompt("medical")
                    )
                    
                    response_parts.append(f"\n{instruction_response}")
                
                full_response = "\n".join(response_parts)
                
                # Remove technical attribution for users - keep it simple
                full_response += "\n\n**Remember:** Always follow your doctor's instructions and ask your pharmacist if you have any questions about your medicines!"
                
                return {"response": full_response, "ocr_data": ocr_result, "extracted_medicines": unique_medicine_names}
            
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
    
    def _extract_medicine_names_from_ocr(self, ocr_text: str) -> List[str]:
        """Extract potential medicine names from OCR text with improved prescription patterns"""
        import re
        medicine_names = []
        
        # Clean OCR text first - remove noise and fix common OCR errors
        cleaned_text = ocr_text
        # Fix common OCR mistakes
        ocr_fixes = {
            r'Tsy': 'Tab',
            r'TA3': 'Tab',
            r'Ta3': 'Tab',
            r'Syp': 'Syrup',
            r'Cap': 'Capsule',
            r'Inj': 'Injection',
            r'O\s*-\)': '',  # Remove O-) pattern
            r'[-\(\)]+$': '',  # Remove trailing symbols
        }
        
        for pattern, replacement in ocr_fixes.items():
            cleaned_text = re.sub(pattern, replacement, cleaned_text, flags=re.IGNORECASE)
        
        # Enhanced patterns for prescription format
        patterns = [
            r'Tab\s+([A-Z][A-Z\s]+?)(?:\s+[-\d\(\)]|$)',  # Tab MEDICINE_NAME
            r'Tablet\s+([A-Z][A-Z\s]+?)(?:\s+[-\d\(\)]|$)',  # Tablet MEDICINE_NAME
            r'Syrup\s+([A-Z][A-Z\s]+?)(?:\s+[-\d\(\)]|$)',  # Syrup MEDICINE_NAME  
            r'Capsule\s+([A-Z][A-Z\s]+?)(?:\s+[-\d\(\)]|$)',  # Capsule MEDICINE_NAME
            r'Injection\s+([A-Z][A-Z\s]+?)(?:\s+[-\d\(\)]|$)',  # Injection MEDICINE_NAME
            r'^\s*\d+[\.\)]\s*([A-Z][A-Z\s]+?)(?:\s+[-\d\(\)]|$)',  # 1. MEDICINE_NAME
            r'\b([A-Z]{4,}[A-Z\s]*?)(?:\s+O[2-9]?|\s+M[LT]|\s+\d+|\s*$)',  # Medicine names like AFLAZEST, AZENAC
        ]
        
        # Process line by line for better extraction
        lines = cleaned_text.split('\n')
        for line in lines:
            line = line.strip()
            if len(line) < 3:
                continue
                
            for pattern in patterns:
                matches = re.findall(pattern, line, re.IGNORECASE | re.MULTILINE)
                for match in matches:
                    # Clean the match
                    cleaned = re.sub(r'\s+', ' ', match.strip())
                    # Remove trailing noise
                    cleaned = re.sub(r'[^A-Za-z\s].*$', '', cleaned).strip()
                    if len(cleaned) > 2:
                        medicine_names.append(cleaned)
        
        # Additional pattern for medicine names in the middle of noisy text
        # Look for sequences of capital letters that could be medicine names
        capital_sequences = re.findall(r'\b([A-Z]{4,}[A-Z]*)\b', cleaned_text)
        for seq in capital_sequences:
            if len(seq) >= 4 and seq not in ['TABLET', 'CAPSULE', 'SYRUP', 'INJECTION']:
                medicine_names.append(seq)
        
        # Clean and filter results
        cleaned_names = []
        excluded_words = {
            'tablet', 'capsule', 'syrup', 'injection', 'medicine', 'pharma', 
            'limited', 'ltd', 'the', 'and', 'for', 'tab', 'cap', 'syp', 'inj',
            'jkath', 'xane', 'daw', 'zof', 'ath', 'tsy'  # Common OCR errors
        }
        
        for name in medicine_names:
            name_lower = name.lower().strip()
            if (len(name) >= 4 and 
                name_lower not in excluded_words and
                not re.match(r'^[^A-Za-z]*$', name) and  # Not just symbols
                len(re.sub(r'[^A-Za-z]', '', name)) >= 3):  # At least 3 letters
                cleaned_names.append(name)
        
        return list(set(cleaned_names))  # Remove duplicates
    
    def _extract_medicine_names_from_text(self, text: str) -> List[str]:
        """Extract potential medicine names from analysis text"""
        import re
        medicine_names = []
        
        # Look for medicine names in structured responses
        patterns = [
            r'Medicine[:\s]+([A-Za-z][A-Za-z\s]+?)(?:\n|$|\s*[-•])',
            r'Brand[:\s]+([A-Za-z][A-Za-z\s]+?)(?:\n|$|\s*[-•])',
            r'Name[:\s]+([A-Za-z][A-Za-z\s]+?)(?:\n|$|\s*[-•])',
            r'\*\*([A-Za-z][A-Za-z\s]+?)\*\*',  # Bold text often contains medicine names
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            medicine_names.extend(matches)
        
        # Clean results
        cleaned_names = []
        for name in medicine_names:
            cleaned = name.strip()
            if len(cleaned) > 2 and not any(word in cleaned.lower() for word in ['information', 'analysis', 'tablet', 'capsule', 'the', 'and', 'for']):
                cleaned_names.append(cleaned)
        
        return list(set(cleaned_names))  # Remove duplicates
    
    @tool(
        name="analyze_pdf",
        description="Analyze PDF documents, extract text and images, summarize content, or answer questions about PDFs. Good for medical papers and documents with multimodal analysis.",
        category="documents", 
        priority=3
    )
    async def _tool_analyze_pdf(self, pdf_data: bytes, question: Optional[str] = None, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Comprehensive PDF analysis with optimized multimodal processing using CLIP"""
        try:
            from langchain_community.document_loaders import PyPDFLoader
            import tempfile
            import os
            import fitz  # PyMuPDF for image extraction
            from PIL import Image
            import io
            import asyncio
            from concurrent.futures import ThreadPoolExecutor
            
            # Save PDF to temp file
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(pdf_data)
                tmp_path = tmp.name
            
            try:
                # Step 1: Extract text using LangChain
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
                
                logger.info(f"Extracted text from {len(docs)} pages, {len(full_text)} characters")
                
                # Step 2: Extract and analyze images with better error handling
                image_analysis_results = []
                try:
                    doc = fitz.open(tmp_path)
                    logger.info(f"Opened PDF with {len(doc)} pages for image extraction")
                    
                    # Extract all images first
                    all_images = []
                    total_images_found = 0
                    
                    for page_idx in range(len(doc)):
                        page = doc[page_idx]
                        image_list = page.get_images(full=True)
                        total_images_found += len(image_list)
                        
                        logger.info(f"Page {page_idx + 1}: Found {len(image_list)} images")
                        
                        for img_index, img in enumerate(image_list):
                            try:
                                xref = img[0]
                                base_image = doc.extract_image(xref)
                                image_bytes = base_image["image"]
                                
                                # Check if image is valid
                                if len(image_bytes) < 100:  # Skip very small images
                                    logger.warning(f"Skipping small image on page {page_idx + 1}")
                                    continue
                                
                                pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                                
                                # Check image dimensions
                                width, height = pil_image.size
                                if width < 50 or height < 50:  # Skip very small images
                                    logger.warning(f"Skipping small image {width}x{height} on page {page_idx + 1}")
                                    continue
                                
                                all_images.append({
                                    "image": pil_image,
                                    "page_num": page_idx + 1,
                                    "img_num": img_index + 1,
                                    "size": (width, height)
                                })
                                
                                logger.info(f"Successfully extracted image {img_index + 1} from page {page_idx + 1} ({width}x{height})")
                                
                            except Exception as img_e:
                                logger.warning(f"Error extracting image {img_index + 1} on page {page_idx + 1}: {img_e}")
                                continue
                    
                    doc.close()
                    logger.info(f"Total images found: {total_images_found}, Successfully extracted: {len(all_images)}")
                    
                    # Process images in parallel with smart selection
                    if all_images:
                        # Smart image selection: prioritize first few pages and larger images
                        selected_images = self._select_relevant_images(all_images, question, max_images=10)
                        logger.info(f"Selected {len(selected_images)} images for analysis")
                        
                        # Process images in parallel
                        image_analysis_results = await self._process_images_parallel(
                            selected_images, question, max_workers=2  # Reduced workers for stability
                        )
                        logger.info(f"Successfully analyzed {len(image_analysis_results)} images")
                    else:
                        logger.warning("No valid images found in PDF")
                    
                except Exception as clip_e:
                    logger.error(f"Image processing failed: {clip_e}")
                    image_analysis_results = []
                
                # Step 3: Use CLIP for multimodal text-image correlation
                multimodal_analysis = await self._analyze_multimodal_content(
                    full_text, image_analysis_results, question
                )
                logger.info(f"Multimodal analysis completed: {multimodal_analysis.get('correlation_score', 0.0):.2f} correlation")
                
                # Step 4: Generate optimized response
                if question and question.strip():
                    response = await self._generate_optimized_pdf_answer(
                        question, full_text, image_analysis_results, multimodal_analysis
                    )
                else:
                    response = await self._generate_optimized_pdf_summary(
                        full_text, image_analysis_results, multimodal_analysis, pages_content
                    )
                
                return {
                    "response": response,
                    "document_info": {
                        "total_pages": len(docs),
                        "word_count": len(full_text.split()),
                        "char_count": len(full_text),
                        "is_medical": multimodal_analysis.get("is_medical", False),
                        "images_analyzed": len(image_analysis_results),
                        "multimodal_score": multimodal_analysis.get("correlation_score", 0.0),
                        "total_images_found": total_images_found if 'total_images_found' in locals() else 0
                    },
                    "pages_content": pages_content[:3],
                    "image_analysis": image_analysis_results[:3]
                }
                
            finally:
                os.unlink(tmp_path)
                
        except Exception as e:
            logger.exception(f"Error in analyze_pdf tool: {e}")
            # Fallback attempt
            try:
                import fitz
                
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
    
    def _select_relevant_images(self, all_images: List[Dict], question: Optional[str], max_images: int = 10) -> List[Dict]:
        """Smart selection of most relevant images with better scoring"""
        if len(all_images) <= max_images:
            return all_images
        
        # Priority scoring
        scored_images = []
        for img_data in all_images:
            score = 0
            
            # Prioritize early pages (likely to be important)
            if img_data["page_num"] <= 3:
                score += 5
            elif img_data["page_num"] <= 10:
                score += 3
            elif img_data["page_num"] <= 20:
                score += 2
            else:
                score += 1
            
            # Prioritize larger images (likely to be more important)
            width, height = img_data["size"]
            area = width * height
            if area > 200000:  # Very large images
                score += 4
            elif area > 100000:  # Large images
                score += 3
            elif area > 50000:  # Medium images
                score += 2
            elif area > 20000:  # Small images
                score += 1
            
            # If question is about specific content, prioritize accordingly
            if question:
                question_lower = question.lower()
                if any(term in question_lower for term in ["chart", "graph", "diagram", "figure", "image", "picture"]):
                    score += 3
            
            scored_images.append((score, img_data))
        
        # Sort by score and return top images
        scored_images.sort(key=lambda x: x[0], reverse=True)
        selected = [img_data for _, img_data in scored_images[:max_images]]
        
        logger.info(f"Selected {len(selected)} images from {len(all_images)} total")
        return selected
    
    async def _process_images_parallel(self, selected_images: List[Dict], question: Optional[str], max_workers: int = 2) -> List[Dict]:
        """Process multiple images in parallel for better performance"""
        try:
            if not selected_images:
                return []
            
            logger.info(f"Processing {len(selected_images)} images with {max_workers} workers")
            
            # Use ThreadPoolExecutor for parallel processing
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Create tasks for parallel execution
                tasks = []
                for img_data in selected_images:
                    task = asyncio.get_event_loop().run_in_executor(
                        executor,
                        self._analyze_single_image_sync,
                        img_data["image"],
                        img_data["page_num"],
                        img_data["img_num"],
                        question
                    )
                    tasks.append(task)
                
                # Wait for all tasks to complete with timeout
                try:
                    results = await asyncio.wait_for(
                        asyncio.gather(*tasks, return_exceptions=True),
                        timeout=60.0  # 60 second timeout
                    )
                except asyncio.TimeoutError:
                    logger.warning("Image processing timed out")
                    results = []
                
                # Filter out exceptions and None results
                valid_results = []
                for i, result in enumerate(results):
                    if isinstance(result, dict) and result.get("description"):
                        valid_results.append(result)
                    elif isinstance(result, Exception):
                        logger.warning(f"Image analysis failed for image {i+1}: {result}")
                
                logger.info(f"Successfully processed {len(valid_results)} out of {len(selected_images)} images")
                return valid_results
                
        except Exception as e:
            logger.error(f"Parallel image processing failed: {e}")
            return []
    
    def _analyze_single_image_sync(self, pil_image: Image.Image, page_num: int, img_num: int, question: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Synchronous image analysis for parallel processing"""
        try:
            # Check if CLIP is available
            if not hasattr(self.llm, 'vision') or not hasattr(self.llm.vision, 'clip_available') or not self.llm.vision.clip_available:
                logger.warning("CLIP not available, using basic image analysis")
                # Fallback to basic analysis
                return {
                    "page_number": page_num,
                    "image_number": img_num,
                    "description": f"Image {img_num} from page {page_num} (size: {pil_image.size})",
                    "clip_embedding": None,
                    "clip_similarity": 0.0,
                    "analysis_type": "general"
                }
            
            # Convert PIL image to bytes for vision analysis
            img_bytes = io.BytesIO()
            pil_image.save(img_bytes, format='JPEG', quality=75)  # Reduced quality for speed
            img_bytes = img_bytes.getvalue()
            
            # Optimized vision prompt for faster processing
            if question:
                vision_prompt = f"Describe this image from page {page_num} of a medical encyclopedia. Focus on: 1) Main content, 2) Text/labels, 3) Medical content, 4) Relevance to: '{question}'"
            else:
                vision_prompt = f"Describe this image from page {page_num} of a medical encyclopedia. Focus on: 1) Main content, 2) Text/labels, 3) Medical content, 4) Charts/diagrams"
            
            # Use vision model with optimized settings
            vision_response = self.llm.generate_vision_response(vision_prompt, image_data=img_bytes)
            
            # Generate CLIP embeddings for multimodal analysis
            clip_analysis = self.llm.vision.analyze_image_with_clip(pil_image, vision_prompt)
            
            return {
                "page_number": page_num,
                "image_number": img_num,
                "description": vision_response,
                "clip_embedding": clip_analysis.get("image_embedding"),
                "clip_similarity": clip_analysis.get("similarity_score", 0.0),
                "analysis_type": "medical" if any(term in vision_response.lower() for term in [
                    "medical", "clinical", "anatomy", "prescription", "medicine", "diagnosis", "chart", "graph", "encyclopedia"
                ]) else "general"
            }
            
        except Exception as e:
            logger.warning(f"Error analyzing image on page {page_num}: {e}")
            return None
    
    async def _analyze_multimodal_content(self, full_text: str, image_analysis_results: List[Dict], question: Optional[str]) -> Dict[str, Any]:
        """Use CLIP to analyze text-image correlation and content type"""
        try:
            if not image_analysis_results:
                logger.warning("No image analysis results for multimodal analysis")
                return {
                    "is_medical": any(term in full_text.lower() for term in [
                        "medicine", "medical", "drug", "patient", "treatment", "diagnosis", 
                        "clinical", "pharmaceutical", "therapy", "symptoms", "disease", "encyclopedia"
                    ]),
                    "correlation_score": 0.0,
                    "multimodal_insights": []
                }
            
            # Extract key text snippets for CLIP analysis
            text_snippets = self._extract_key_text_snippets(full_text, max_snippets=5)
            
            # Use CLIP to find text-image correlations
            correlations = []
            for snippet in text_snippets:
                for img_result in image_analysis_results:
                    if img_result.get("clip_embedding") is not None:
                        # Calculate similarity between text and image
                        similarity = self._calculate_text_image_similarity(snippet, img_result["clip_embedding"])
                        if similarity > 0.2:  # Lower threshold for more correlations
                            correlations.append({
                                "text_snippet": snippet,
                                "image_page": img_result["page_number"],
                                "similarity": similarity,
                                "image_description": img_result["description"]
                            })
            
            # Determine if content is medical based on both text and images
            is_medical = any(term in full_text.lower() for term in [
                "medicine", "medical", "drug", "patient", "treatment", "diagnosis", 
                "clinical", "pharmaceutical", "therapy", "symptoms", "disease", "encyclopedia"
            ])
            
            # Check image analysis for medical indicators
            if not is_medical:
                for img_result in image_analysis_results:
                    if img_result.get("analysis_type") == "medical":
                        is_medical = True
                        break
            
            logger.info(f"Multimodal analysis: {len(correlations)} correlations found, medical: {is_medical}")
            
            return {
                "is_medical": is_medical,
                "correlation_score": max([c["similarity"] for c in correlations], default=0.0),
                "multimodal_insights": correlations[:3],  # Top 3 correlations
                "text_snippets": text_snippets
            }
            
        except Exception as e:
            logger.warning(f"Multimodal analysis failed: {e}")
            return {
                "is_medical": any(term in full_text.lower() for term in [
                    "medicine", "medical", "drug", "patient", "treatment", "diagnosis", "encyclopedia"
                ]),
                "correlation_score": 0.0,
                "multimodal_insights": []
            }
    
    def _extract_key_text_snippets(self, full_text: str, max_snippets: int = 5) -> List[str]:
        """Extract key text snippets for CLIP analysis"""
        # Split into sentences and select most informative ones
        sentences = full_text.split('.')
        key_snippets = []
        
        # Prioritize sentences with medical terms
        medical_terms = ["medicine", "medical", "drug", "treatment", "diagnosis", "clinical", "patient", "therapy", "encyclopedia", "disease", "symptom"]
        
        scored_sentences = []
        for sentence in sentences:
            if len(sentence.strip()) > 20:  # Minimum length
                score = sum(1 for term in medical_terms if term in sentence.lower())
                if score > 0 or len(sentence) > 100:  # Include medical or long sentences
                    scored_sentences.append((score, sentence.strip()))
        
        # Sort by score and length
        scored_sentences.sort(key=lambda x: (x[0], len(x[1])), reverse=True)
        
        for score, sentence in scored_sentences[:max_snippets]:
            key_snippets.append(sentence[:200])  # Limit snippet length
        
        return key_snippets
    
    def _calculate_text_image_similarity(self, text_snippet: str, image_embedding: List[float]) -> float:
        """Calculate similarity between text and image using CLIP"""
        try:
            if not hasattr(self.llm.vision, 'clip_model') or self.llm.vision.clip_model is None:
                return 0.0
            
            # Tokenize text
            text_tokens = clip.tokenize([text_snippet])
            
            # Get text embedding
            with torch.no_grad():
                text_features = self.llm.vision.clip_model.encode_text(text_tokens)
                text_embedding = text_features.cpu().numpy()[0]
            
            # Calculate cosine similarity
            import numpy as np
            similarity = np.dot(text_embedding, image_embedding) / (
                np.linalg.norm(text_embedding) * np.linalg.norm(image_embedding)
            )
            
            return float(similarity)
            
        except Exception as e:
            logger.warning(f"Text-image similarity calculation failed: {e}")
            return 0.0
    
    async def _generate_optimized_pdf_summary(self, full_text: str, image_analysis_results: List[Dict], multimodal_analysis: Dict, pages_content: List[str]) -> str:
        """Generate optimized summary using multimodal analysis"""
        try:
            # Prepare efficient context
            context_parts = []
            
            # Add text context
            if full_text.strip():
                context_parts.append(f"**Text Content:**\n{full_text[:4000]}{'...' if len(full_text) > 4000 else ''}")
            
            # Add image context
            if image_analysis_results:
                image_context = "**Key Images:**\n"
                for i, img_result in enumerate(image_analysis_results[:4], 1):
                    image_context += f"Image {i} (Page {img_result['page_number']}): {img_result['description'][:150]}...\n"
                context_parts.append(image_context)
            else:
                context_parts.append("**Images:** No images were successfully analyzed from this PDF.")
            
            # Add multimodal insights
            if multimodal_analysis.get("multimodal_insights"):
                insights = "**Text-Image Correlations:**\n"
                for insight in multimodal_analysis["multimodal_insights"][:3]:
                    insights += f"• {insight['text_snippet'][:80]}... (Page {insight['image_page']})\n"
                context_parts.append(insights)
            
            # Create optimized summary prompt
            prompt = f"""
            Provide a comprehensive summary of this {'medical' if multimodal_analysis.get('is_medical') else ''} PDF document:
            
            1. **Main Topic**: What is this document about?
            2. **Key Points**: 3-5 most important findings
            3. **Visual Content**: Summary of important images/charts (if any)
            4. **Structure**: Document organization
            5. **Key Takeaways**: Most important information for readers
            
            CONTEXT:
            {chr(10).join(context_parts)}
            
            Provide a clear, structured summary that integrates text and visual information. If no images were analyzed, mention this in the visual content section.
            """
            
            # Generate summary
            system_prompt = "You are a medical document analyst. Provide clear, structured summaries with medical accuracy." if multimodal_analysis.get("is_medical") else "You are a document analyst. Provide clear, structured summaries."
            summary = self.llm.text_gen.generate_response(prompt, system_prompt=system_prompt)
            
            # Add statistics
            doc_stats = f"""
**Document Statistics:**
• Total pages: {len(pages_content)}
• Images analyzed: {len(image_analysis_results)}
• Content type: {'Medical/Clinical document' if multimodal_analysis.get('is_medical') else 'General document'}
• Multimodal correlation: {multimodal_analysis.get('correlation_score', 0.0):.2f}
• Word count: ~{len(full_text.split()):,}
• Character count: {len(full_text):,}"""
            
            return summary + "\n\n" + doc_stats
            
        except Exception as e:
            logger.error(f"Error generating optimized summary: {e}")
            # Fallback to text-only summary
            return f"**PDF Content Summary:**\n\n{full_text[:3000]}{'...' if len(full_text) > 3000 else ''}"
    
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
    
