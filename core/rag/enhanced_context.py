# core/rag/enhanced_context.py
"""
Enhanced RAG Context Manager with reranking and multi-store queries
"""
import logging
import asyncio
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from core.vector_store import PineconeStore
from config.env_config import PINECONE_MEDICINE_INDEX, PINECONE_BD_PHARMACY_INDEX, PINECONE_IMAGE_INDEX

logger = logging.getLogger(__name__)

@dataclass
class ContextChunk:
    """Enhanced context chunk with metadata"""
    content: str
    source: str
    score: float
    metadata: Dict[str, Any]
    namespace: str = ""
    rerank_score: Optional[float] = None

class CrossEncoderReranker:
    """Simple reranking based on keyword matching and relevance"""
    
    def __init__(self):
        self.medical_terms = [
            "medicine", "drug", "medication", "prescription", "symptom", "disease",
            "treatment", "diagnosis", "doctor", "patient", "health", "medical",
            "tablet", "capsule", "dose", "dosage", "mg", "ml", "side effect",
            "paracetamol", "aspirin", "antibiotic", "vitamin", "insulin"
        ]
    
    async def rerank(self, query: str, chunks: List[ContextChunk]) -> List[ContextChunk]:
        """Rerank chunks based on relevance to query"""
        query_lower = query.lower()
        query_terms = set(query_lower.split())
        
        for chunk in chunks:
            # Calculate relevance score
            content_lower = chunk.content.lower()
            content_terms = set(content_lower.split())
            
            # Term overlap score
            overlap = len(query_terms.intersection(content_terms))
            overlap_score = overlap / max(len(query_terms), 1)
            
            # Medical relevance score
            medical_matches = sum(1 for term in self.medical_terms if term in content_lower)
            medical_score = min(medical_matches / 5, 1.0)  # Normalize to 0-1
            
            # Length penalty (prefer concise, relevant chunks)
            length_penalty = min(len(chunk.content) / 1000, 1.0)  # Penalize very long chunks
            
            # Combined rerank score
            chunk.rerank_score = (
                overlap_score * 0.5 +
                medical_score * 0.3 +
                chunk.score * 0.2 -
                length_penalty * 0.1
            )
        
        # Sort by rerank score
        return sorted(chunks, key=lambda c: c.rerank_score or 0, reverse=True)

class EnhancedRAGContextManager:
    """Enhanced RAG context manager with multi-store queries and reranking"""
    
    def __init__(self):
        self.vector_stores = {}
        self.reranker = CrossEncoderReranker()
        self._query_cache = {}  # Simple query cache
        self._initialize_stores()
    
    def _initialize_stores(self):
        """Initialize vector stores with error handling"""
        store_configs = [
            ("medicine", PINECONE_MEDICINE_INDEX, 384),
            ("pharmacy", PINECONE_BD_PHARMACY_INDEX, 384),
            ("anatomy", PINECONE_IMAGE_INDEX, 512)  # CLIP embeddings are 512-dim
        ]
        
        for store_name, index_name, dimension in store_configs:
            try:
                if index_name:
                    self.vector_stores[store_name] = PineconeStore(
                        index_name=index_name,
                        dimension=dimension
                    )
                    logger.info(f"Initialized {store_name} vector store")
            except Exception as e:
                logger.warning(f"Failed to initialize {store_name} store: {e}")
    
    async def gather_context(
        self,
        query: str,
        context_type: str = "medical",
        max_chunks: int = 5,
        use_cache: bool = True
    ) -> List[ContextChunk]:
        """Enhanced context gathering with reranking and caching"""
        
        # Check cache first
        cache_key = hashlib.md5(f"{query}_{context_type}_{max_chunks}".encode()).hexdigest()
        if use_cache and cache_key in self._query_cache:
            logger.debug(f"Cache hit for query: {query[:50]}...")
            return self._query_cache[cache_key]
        
        try:
            # Multi-store querying
            all_chunks = await self._query_all_stores(query, context_type)
            
            # Rerank chunks
            reranked_chunks = await self.reranker.rerank(query, all_chunks)
            
            # Deduplicate and filter
            final_chunks = self._deduplicate_and_filter(reranked_chunks, max_chunks)
            
            # Cache result
            if use_cache:
                self._query_cache[cache_key] = final_chunks
                # Limit cache size
                if len(self._query_cache) > 100:
                    # Remove oldest entries
                    oldest_key = next(iter(self._query_cache))
                    del self._query_cache[oldest_key]
            
            logger.info(f"Gathered {len(final_chunks)} context chunks for query")
            return final_chunks
            
        except Exception as e:
            logger.error(f"Error gathering context: {e}")
            return []
    
    async def _query_all_stores(self, query: str, context_type: str) -> List[ContextChunk]:
        """Query multiple vector stores simultaneously"""
        tasks = []
        
        # Determine which stores to query based on context type
        stores_to_query = self._select_stores_for_context(context_type)
        
        for store_name in stores_to_query:
            if store_name in self.vector_stores:
                task = asyncio.create_task(
                    self._query_single_store(store_name, query),
                    name=f"query_{store_name}"
                )
                tasks.append(task)
        
        if not tasks:
            return []
        
        # Execute queries in parallel
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            all_chunks = []
            for result in results:
                if isinstance(result, Exception):
                    logger.warning(f"Store query failed: {result}")
                else:
                    all_chunks.extend(result)
            
            return all_chunks
            
        except Exception as e:
            logger.error(f"Error in parallel store queries: {e}")
            return []
    
    async def _query_single_store(self, store_name: str, query: str) -> List[ContextChunk]:
        """Query a single vector store"""
        store = self.vector_stores[store_name]
        chunks = []
        
        try:
            # Query with different namespaces if applicable
            namespaces = self._get_namespaces_for_store(store_name, query)
            
            for namespace in namespaces:
                hits = store.query(query, top_k=3, namespace=namespace)
                
                for i, hit in enumerate(hits):
                    if hit and len(hit.strip()) > 50:  # Quality filter
                        chunks.append(ContextChunk(
                            content=hit,
                            source=store_name,
                            score=1.0 - (i * 0.1),  # Decreasing score by position
                            metadata={"store": store_name, "namespace": namespace},
                            namespace=namespace
                        ))
            
            return chunks
            
        except Exception as e:
            logger.warning(f"Error querying {store_name}: {e}")
            return []
    
    def _select_stores_for_context(self, context_type: str) -> List[str]:
        """Select appropriate stores based on context type"""
        store_mapping = {
            "medical": ["medicine", "pharmacy"],
            "medicine": ["medicine", "pharmacy"],
            "pharmacy": ["pharmacy"],
            "anatomy": ["anatomy"],
            "general": ["medicine", "pharmacy"]
        }
        
        return store_mapping.get(context_type, ["medicine", "pharmacy"])
    
    def _get_namespaces_for_store(self, store_name: str, query: str) -> List[str]:
        """Get appropriate namespaces for a store based on query"""
        query_lower = query.lower()
        
        if store_name == "medicine":
            return ["general", "clinical"]
        elif store_name == "pharmacy":
            if any(term in query_lower for term in ["brand", "price", "company"]):
                return ["brands"]
            return ["general"]
        elif store_name == "anatomy":
            return ["anatomy", "medical_images"]
        
        return [""]  # Default namespace
    
    def _deduplicate_and_filter(self, chunks: List[ContextChunk], max_chunks: int) -> List[ContextChunk]:
        """Remove duplicates and apply quality filters"""
        seen_content = set()
        filtered_chunks = []
        
        for chunk in chunks:
            # Create content hash for deduplication
            content_hash = hashlib.md5(chunk.content.lower().encode()).hexdigest()
            
            if content_hash not in seen_content:
                # Quality filters
                if (len(chunk.content.strip()) >= 50 and  # Minimum length
                    (chunk.rerank_score or chunk.score) > 0.1):  # Minimum relevance
                    
                    seen_content.add(content_hash)
                    filtered_chunks.append(chunk)
                    
                    if len(filtered_chunks) >= max_chunks:
                        break
        
        return filtered_chunks
    
    def get_context_summary(self, chunks: List[ContextChunk]) -> Dict[str, Any]:
        """Get summary of context chunks"""
        if not chunks:
            return {"total": 0, "sources": {}, "average_score": 0.0}
        
        sources = {}
        total_score = 0.0
        
        for chunk in chunks:
            source = chunk.source
            sources[source] = sources.get(source, 0) + 1
            total_score += chunk.rerank_score or chunk.score
        
        return {
            "total": len(chunks),
            "sources": sources,
            "average_score": total_score / len(chunks),
            "namespaces": list(set(chunk.namespace for chunk in chunks)),
            "total_content_length": sum(len(chunk.content) for chunk in chunks)
        }
    
    def clear_cache(self):
        """Clear the query cache"""
        self._query_cache.clear()
        logger.info("Context cache cleared") 