# core/rag/context.py
"""RAG context management - retrieval and relevance scoring"""

import logging
import re
from typing import List, Dict, Any, Optional, Set
from core.vector_store import PineconeStore
from config.env_config import PINECONE_MEDICINE_INDEX, PINECONE_BD_PHARMACY_INDEX

logger = logging.getLogger(__name__)

class RAGContextManager:
    """Manages RAG context retrieval and relevance scoring with optimizations"""
    
    def __init__(self):
        self.medicine_store = None
        self.bd_pharmacy_store = None
        
        try:
            self.medicine_store = PineconeStore(index_name=PINECONE_MEDICINE_INDEX, dimension=384)
        except Exception as e:
            logger.warning(f"Medicine store init failed: {e}")
            
        try:
            self.bd_pharmacy_store = PineconeStore(index_name=PINECONE_BD_PHARMACY_INDEX, dimension=384)
        except Exception as e:
            logger.warning(f"BD pharmacy store init failed: {e}")
    
    def gather_context(self, query: str, max_chunks: int = 3) -> List[str]:
        """Gather relevant context for query with optimizations"""
        contexts = []
        query_terms = set(query.lower().split())
        
        # Select best namespace
        namespace = self._select_namespace(query)
        
        # Query BD pharmacy store with optimized parameters
        if self.bd_pharmacy_store:
            try:
                hits = self.bd_pharmacy_store.query(query, top_k=2, namespace=namespace)
                for hit in hits:
                    if hit and len(hit.strip()) > 50:
                        score = self._calculate_relevance_score(hit, query_terms)
                        if score > 0.1:
                            # Truncate to 800 chars for optimization
                            truncated_hit = hit[:800] + "..." if len(hit) > 800 else hit
                            contexts.append(truncated_hit)
            except Exception as e:
                logger.warning(f"BD pharmacy query failed: {e}")
        
        # Query medicine store if needed with optimized parameters
        if len(contexts) < max_chunks and self.medicine_store:
            medicine_keywords = ["medicine", "drug", "medication", "tablet", "capsule"]
            if any(keyword in query.lower() for keyword in medicine_keywords):
                try:
                    hits = self.medicine_store.query(query, top_k=1)
                    for hit in hits:
                        if hit and len(hit.strip()) > 50:
                            score = self._calculate_relevance_score(hit, query_terms)
                            if score > 0.15:
                                # Truncate to 800 chars for optimization
                                truncated_hit = hit[:800] + "..." if len(hit) > 800 else hit
                                contexts.append(truncated_hit)
                                break
                except Exception as e:
                    logger.warning(f"Medicine store query failed: {e}")
        
        # Deduplicate and limit with optimizations
        return self._deduplicate_contexts(contexts, max_chunks)
    
    def _select_namespace(self, query: str) -> str:
        """Select appropriate namespace based on query"""
        query_lower = query.lower().strip()
        
        # Policy/Legal content
        if any(k in query_lower for k in ["law", "policy", "act", "regulation", "dgda"]):
            return "policy"
        
        # Anatomy/Educational content
        if any(k in query_lower for k in ["anatomy", "nerve", "muscle", "bone", "cell", "physiology"]):
            return "textbook"
        
        # OTC/Self-care content
        if any(k in query_lower for k in ["otc", "over the counter", "pharmacy"]):
            return "otc"
        
        # Clinical prescribing (default for medical queries)
        return "prescribing"
    
    def _calculate_relevance_score(self, text: str, query_terms: Set[str]) -> float:
        """Calculate relevance score for text chunk with optimizations"""
        if not text or not query_terms:
            return 0.0
        
        text_lower = text.lower()
        text_words = set(text_lower.split())
        
        # Term frequency score
        term_matches = len(query_terms.intersection(text_words))
        term_score = term_matches / len(query_terms) if query_terms else 0
        
        # Medical terms boost
        medical_terms = {"dose", "dosage", "indication", "side", "effect", "contraindication", 
                        "interaction", "pregnancy", "safety", "clinical", "treatment"}
        medical_matches = len(medical_terms.intersection(text_words))
        medical_boost = min(medical_matches * 0.1, 0.3)
        
        # Length penalty (optimized for 800 char limit)
        length = len(text)
        if length < 100:
            length_penalty = 0.5
        elif length > 1000:  # Reduced from 1500 to 1000 for optimization
            length_penalty = 0.7
        else:
            length_penalty = 1.0
            
        return (term_score + medical_boost) * length_penalty
    
    def _deduplicate_contexts(self, contexts: List[str], max_chunks: int) -> List[str]:
        """Remove duplicate or very similar contexts with optimizations"""
        if not contexts:
            return []
        
        unique_contexts: List[str] = []
        seen_phrase_sets: List[Set[str]] = []
        
        for context in contexts:
            if not context or not context.strip():
                continue
                
            # Extract key phrases
            key_phrases = self._extract_key_phrases(context)
            
            # Check for overlap with existing contexts
            overlap_score = 0.0
            for existing_phrases in seen_phrase_sets:
                if not existing_phrases:
                    continue
                overlap = len(key_phrases.intersection(existing_phrases))
                overlap_score = max(
                    overlap_score,
                    overlap / max(len(key_phrases) or 1, len(existing_phrases) or 1),
                )
            
            # Add if not too similar (< 70% overlap)
            if overlap_score < 0.7:
                # Truncate to 800 chars for optimization
                truncated_context = context.strip()[:800]
                unique_contexts.append(truncated_context)
                seen_phrase_sets.append(key_phrases)
                
                if len(unique_contexts) >= max_chunks:
                    break
        
        return unique_contexts
    
    def _extract_key_phrases(self, text: str) -> Set[str]:
        """Extract key phrases for similarity comparison with optimizations"""
        text_lower = text.lower()
        phrases = set()
        
        # Medical terms
        medical_pattern = r'\b(?:dose|dosage|indication|contraindication|side effect|interaction|pregnancy|safety|clinical|treatment|therapy|medication|drug|medicine)\b'
        phrases.update(re.findall(medical_pattern, text_lower))
        
        # Drug names (common endings)
        drug_pattern = r'\b[a-z]+(?:ol|in|ine|ide|ate|azole|mycin|cillin|prazole)\b'
        phrases.update(re.findall(drug_pattern, text_lower))
        
        # Quantities and measurements
        quantity_pattern = r'\b\d+\s*(?:mg|g|ml|mcg|units?|tablets?|capsules?)\b'
        phrases.update(re.findall(quantity_pattern, text_lower))
        
        return phrases
    
    def format_context_prompt(self, query: str, contexts: List[str]) -> str:
        """Format contexts into a prompt with optimizations"""
        if not contexts:
            context_block = "No relevant context available."
        else:
            # Limit context length for optimization
            limited_contexts = contexts[:3]  # Max 3 contexts
            context_block = "\n\n---\n".join(limited_contexts)
        
        return (
            "Use the CONTEXT to answer the medical question. "
            "Prefer Bangladesh-specific information when available. "
            "If context is insufficient, provide safe general guidance.\n\n"
            f"CONTEXT START\n---\n{context_block}\n---\nCONTEXT END\n\n"
            f"QUESTION: {query}\n\nAnswer:"
        )
