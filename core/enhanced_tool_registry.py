# core/enhanced_tool_registry.py
"""
Enhanced Tool Registry with priority-based selection and confidence scoring
"""
import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class ToolCategory(str, Enum):
    MEDICAL = "medical"
    DOCUMENT = "document" 
    UTILITY = "utility"
    ADMIN = "admin"

@dataclass
class ToolExecutionResult:
    """Result of tool execution with metadata"""
    result: Any
    execution_time: float
    confidence: Optional[float] = None
    error: Optional[str] = None
    status: str = "success"

@dataclass
class Tool:
    """Enhanced tool definition with metadata"""
    name: str
    description: str
    category: ToolCategory
    priority: int
    function: Callable
    confidence_threshold: float = 0.0
    max_execution_time: float = 30.0
    retry_count: int = 1
    
    def __post_init__(self):
        if self.priority < 1 or self.priority > 5:
            raise ValueError("Priority must be between 1 and 5")

class ToolRegistry:
    """Enhanced tool registry with priority-based selection"""
    
    def __init__(self):
        self.tools: Dict[str, Tool] = {}
        self.tools_by_priority: Dict[int, List[Tool]] = {i: [] for i in range(1, 6)}
        self.tools_by_category: Dict[str, List[Tool]] = {}
        self.execution_stats: Dict[str, Dict[str, Any]] = {}
        self._circuit_breaker: Dict[str, float] = {}  # Tool name -> failure timestamp
        self.circuit_breaker_timeout = 300  # 5 minutes
    
    def register_tool(self, tool: Tool):
        """Register a tool with the registry"""
        if tool.name in self.tools:
            logger.warning(f"Tool {tool.name} already registered, overwriting")
        
        self.tools[tool.name] = tool
        self.tools_by_priority[tool.priority].append(tool)
        
        if tool.category.value not in self.tools_by_category:
            self.tools_by_category[tool.category.value] = []
        self.tools_by_category[tool.category.value].append(tool)
        
        # Initialize stats
        self.execution_stats[tool.name] = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "average_execution_time": 0.0,
            "total_execution_time": 0.0,
            "last_execution": None
        }
        
        logger.info(f"Registered tool: {tool.name} (category: {tool.category}, priority: {tool.priority})")
    
    def get_all_tools(self) -> List[Tool]:
        """Get all registered tools"""
        return list(self.tools.values())
    
    def get_tools_by_category(self, category: str) -> List[Tool]:
        """Get tools by category"""
        return self.tools_by_category.get(category, [])
    
    def get_tools_by_priority(self, priority: int) -> List[Tool]:
        """Get tools by priority level"""
        return self.tools_by_priority.get(priority, [])
    
    def select_tools_for_input(self, input_analysis: Dict[str, Any]) -> List[Tool]:
        """Select tools based on input analysis with confidence scoring"""
        selected_tools = []
        input_type = input_analysis.get("type", "text")
        content = input_analysis.get("content", "")
        
        # Rule-based tool selection with priority
        if input_type == "image" or "image" in input_analysis:
            # Image processing tools (high priority)
            selected_tools.extend(self._get_available_tools(["analyze_image"], priority_min=4))
            
        if input_type == "pdf" or "pdf" in input_analysis:
            # PDF processing tools (medium priority)
            selected_tools.extend(self._get_available_tools(["analyze_pdf"], priority_min=3))
        
        if input_type == "text" or content:
            # Text analysis tools (high priority for medical content)
            if self._is_medical_content(content):
                selected_tools.extend(self._get_available_tools(["analyze_text"], priority_min=4))
            else:
                selected_tools.extend(self._get_available_tools(["analyze_text"], priority_min=3))
        
        # Medicine-specific queries
        if self._is_medicine_query(content):
            selected_tools.extend(self._get_available_tools(["get_medicine_info"], priority_min=4))
        
        # Web search for current information
        if self._needs_web_search(content):
            selected_tools.extend(self._get_available_tools(["web_search"], priority_min=2))
        
        # Memory access for context
        if input_analysis.get("session_id"):
            selected_tools.extend(self._get_available_tools(["access_memory"], priority_min=1))
        
        # Remove duplicates and sort by priority
        unique_tools = list({tool.name: tool for tool in selected_tools}.values())
        return sorted(unique_tools, key=lambda t: t.priority, reverse=True)
    
    def _get_available_tools(self, tool_names: List[str], priority_min: int = 1) -> List[Tool]:
        """Get available tools that are not circuit-broken"""
        available = []
        current_time = time.time()
        
        for name in tool_names:
            if name in self.tools:
                tool = self.tools[name]
                
                # Check priority threshold
                if tool.priority < priority_min:
                    continue
                
                # Check circuit breaker
                if name in self._circuit_breaker:
                    if current_time - self._circuit_breaker[name] < self.circuit_breaker_timeout:
                        logger.warning(f"Tool {name} is circuit-broken, skipping")
                        continue
                    else:
                        # Reset circuit breaker
                        del self._circuit_breaker[name]
                
                available.append(tool)
        
        return available
    
    def _is_medical_content(self, content: str) -> bool:
        """Check if content is medical-related"""
        medical_keywords = [
            "medicine", "drug", "medication", "prescription", "symptom", "disease",
            "treatment", "diagnosis", "doctor", "patient", "health", "medical",
            "tablet", "capsule", "dose", "dosage", "mg", "ml", "side effect"
        ]
        content_lower = content.lower()
        return any(keyword in content_lower for keyword in medical_keywords)
    
    def _is_medicine_query(self, content: str) -> bool:
        """Check if content is specifically asking about medicines"""
        medicine_keywords = [
            "medicine", "drug", "medication", "tablet", "capsule", "pill",
            "brand", "generic", "price", "cost", "pharmacy", "dosage", "dose"
        ]
        content_lower = content.lower()
        return any(keyword in content_lower for keyword in medicine_keywords)
    
    def _needs_web_search(self, content: str) -> bool:
        """Check if content needs web search for current information"""
        web_indicators = [
            "latest", "recent", "current", "news", "today", "2024", "2025",
            "update", "new", "breakthrough", "research", "study"
        ]
        content_lower = content.lower()
        return any(indicator in content_lower for indicator in web_indicators)
    
    async def execute_tools_parallel(self, tools: List[Tool], input_data: Dict[str, Any]) -> List[ToolExecutionResult]:
        """Execute multiple tools in parallel with timeout and error handling"""
        if not tools:
            return []
        
        # Create execution tasks
        tasks = []
        for tool in tools:
            task = asyncio.create_task(
                self._execute_single_tool(tool, input_data),
                name=f"tool_{tool.name}"
            )
            tasks.append(task)
        
        # Execute with timeout
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=max(tool.max_execution_time for tool in tools)
            )
            
            # Process results
            execution_results = []
            for i, result in enumerate(results):
                tool = tools[i]
                
                if isinstance(result, Exception):
                    execution_results.append(ToolExecutionResult(
                        result=None,
                        execution_time=0.0,
                        error=str(result),
                        status="error"
                    ))
                    self._record_tool_failure(tool.name)
                else:
                    execution_results.append(result)
                    self._record_tool_success(tool.name, result.execution_time)
            
            return execution_results
            
        except asyncio.TimeoutError:
            logger.error(f"Tool execution timed out for tools: {[t.name for t in tools]}")
            return [
                ToolExecutionResult(
                    result=None,
                    execution_time=0.0,
                    error="Execution timed out",
                    status="timeout"
                ) for _ in tools
            ]
    
    async def _execute_single_tool(self, tool: Tool, input_data: Dict[str, Any]) -> ToolExecutionResult:
        """Execute a single tool with retry logic"""
        last_error = None
        
        for attempt in range(tool.retry_count + 1):
            try:
                start_time = time.time()
                
                # Execute tool function
                if asyncio.iscoroutinefunction(tool.function):
                    result = await tool.function(**input_data)
                else:
                    result = tool.function(**input_data)
                
                execution_time = time.time() - start_time
                
                # Extract confidence if available
                confidence = None
                if isinstance(result, dict):
                    confidence = result.get("confidence")
                
                return ToolExecutionResult(
                    result=result,
                    execution_time=execution_time,
                    confidence=confidence,
                    status="success"
                )
                
            except Exception as e:
                last_error = e
                if attempt < tool.retry_count:
                    logger.warning(f"Tool {tool.name} failed (attempt {attempt + 1}), retrying: {e}")
                    await asyncio.sleep(0.5 * (attempt + 1))  # Exponential backoff
                else:
                    logger.error(f"Tool {tool.name} failed after {tool.retry_count + 1} attempts: {e}")
        
        return ToolExecutionResult(
            result=None,
            execution_time=0.0,
            error=str(last_error),
            status="error"
        )
    
    def _record_tool_success(self, tool_name: str, execution_time: float):
        """Record successful tool execution"""
        stats = self.execution_stats[tool_name]
        stats["total_executions"] += 1
        stats["successful_executions"] += 1
        stats["total_execution_time"] += execution_time
        stats["average_execution_time"] = stats["total_execution_time"] / stats["total_executions"]
        stats["last_execution"] = time.time()
    
    def _record_tool_failure(self, tool_name: str):
        """Record failed tool execution and potentially circuit-break"""
        stats = self.execution_stats[tool_name]
        stats["total_executions"] += 1
        stats["failed_executions"] += 1
        stats["last_execution"] = time.time()
        
        # Circuit breaker logic
        failure_rate = stats["failed_executions"] / stats["total_executions"]
        if failure_rate > 0.5 and stats["total_executions"] >= 5:
            self._circuit_breaker[tool_name] = time.time()
            logger.warning(f"Circuit breaker activated for tool {tool_name}")
    
    def get_tool_stats(self) -> Dict[str, Any]:
        """Get comprehensive tool statistics"""
        return {
            "total_tools": len(self.tools),
            "tools_by_category": {
                category: len(tools) for category, tools in self.tools_by_category.items()
            },
            "tools_by_priority": {
                priority: len(tools) for priority, tools in self.tools_by_priority.items()
            },
            "execution_stats": self.execution_stats,
            "circuit_broken_tools": list(self._circuit_breaker.keys()),
            "average_success_rate": self._calculate_average_success_rate()
        }
    
    def _calculate_average_success_rate(self) -> float:
        """Calculate average success rate across all tools"""
        total_executions = sum(stats["total_executions"] for stats in self.execution_stats.values())
        total_successes = sum(stats["successful_executions"] for stats in self.execution_stats.values())
        
        if total_executions == 0:
            return 0.0
        
        return total_successes / total_executions 