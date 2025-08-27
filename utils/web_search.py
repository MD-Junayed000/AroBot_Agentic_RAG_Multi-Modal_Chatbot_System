"""
Web search utility using DuckDuckGo
"""
from typing import Dict, Any, List
from duckduckgo_search import DDGS
from langsmith import traceable
import logging

logger = logging.getLogger(__name__)

class WebSearchTool:
    """Web search tool for medical information using DuckDuckGo"""
    
    def __init__(self):
        self.ddgs = DDGS()
    
    @traceable(name="web_search")
    def search_medical_info(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """Search for medical information using DuckDuckGo"""
        try:
            # Add medical context to query
            medical_query = f"medical {query} health information"
            
            # Perform search
            results = []
            search_results = self.ddgs.text(
                medical_query, 
                max_results=max_results,
                safesearch='moderate'
            )
            
            for result in search_results:
                results.append({
                    'title': result.get('title', ''),
                    'snippet': result.get('body', ''),
                    'url': result.get('href', ''),
                    'source': self._extract_domain(result.get('href', ''))
                })
            
            return {
                "query": query,
                "results": results,
                "result_count": len(results),
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Error in web search: {e}")
            return {
                "error": str(e),
                "status": "error"
            }
    
    @traceable(name="search_medicine_info")
    def search_medicine_info(self, medicine_name: str) -> Dict[str, Any]:
        """Search for specific medicine information"""
        try:
            query = f"{medicine_name} medicine drug information uses side effects"
            return self.search_medical_info(query, max_results=3)
            
        except Exception as e:
            logger.error(f"Error searching medicine info: {e}")
            return {
                "error": str(e),
                "status": "error"
            }
    
    @traceable(name="search_disease_info")
    def search_disease_info(self, disease: str) -> Dict[str, Any]:
        """Search for disease/condition information"""
        try:
            query = f"{disease} disease condition symptoms treatment causes"
            return self.search_medical_info(query, max_results=5)
            
        except Exception as e:
            logger.error(f"Error searching disease info: {e}")
            return {
                "error": str(e),
                "status": "error"
            }
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        try:
            if url.startswith('http'):
                domain = url.split('/')[2]
                return domain
            return "unknown"
        except:
            return "unknown"
    
    @traceable(name="search_medical_news")
    def search_medical_news(self, topic: str) -> Dict[str, Any]:
        """Search for recent medical news about a topic"""
        try:
            query = f"{topic} medical news recent research study"
            
            # Use news search if available
            try:
                news_results = self.ddgs.news(query, max_results=5)
                results = []
                
                for result in news_results:
                    results.append({
                        'title': result.get('title', ''),
                        'snippet': result.get('body', ''),
                        'url': result.get('url', ''),
                        'date': result.get('date', ''),
                        'source': result.get('source', '')
                    })
                
                return {
                    "query": query,
                    "results": results,
                    "result_count": len(results),
                    "type": "news",
                    "status": "success"
                }
                
            except:
                # Fallback to regular search
                return self.search_medical_info(query)
            
        except Exception as e:
            logger.error(f"Error searching medical news: {e}")
            return {
                "error": str(e),
                "status": "error"
            }
    
    @traceable(name="search_general_info")
    def search_general_info(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """Search for general information (weather, news, etc.)"""
        try:
            results = self.ddgs.text(query, max_results=max_results)
            
            processed_results = []
            for result in results:
                processed_results.append({
                    'title': result.get('title', ''),
                    'url': result.get('href', ''),
                    'snippet': result.get('body', '')
                })
            
            return {
                'query': query,
                'results': processed_results,
                'count': len(processed_results),
                'status': 'success',
                'type': 'general'
            }
            
        except Exception as e:
            logger.error(f"Error searching general info: {e}")
            return {
                'query': query,
                'error': str(e),
                'status': 'error'
            }
