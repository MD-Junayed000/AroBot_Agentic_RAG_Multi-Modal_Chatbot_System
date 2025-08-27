import numpy as np
from sentence_transformers import SentenceTransformer
import torch
import logging

logger = logging.getLogger(__name__)

class Embedder:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        # Check for CUDA availability
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Use a faster, more efficient model for your RTX 3060
        # all-MiniLM-L6-v2: 384 dimensions, fast, good quality
        # Alternative: "all-mpnet-base-v2" for higher quality but slower
        try:
            self.model = SentenceTransformer(model_name, device=device)
            logger.info(f"✅ Loaded embeddings model '{model_name}' on {device}")
            if device == "cuda":
                logger.info(f"🚀 GPU acceleration enabled: {torch.cuda.get_device_name(0)}")
        except Exception as e:
            logger.warning(f"Failed to load {model_name}, falling back to default")
            self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
    
    def embed(self, texts):
        # Batch processing with progress bar for large datasets
        if len(texts) > 100:
            logger.info(f"Embedding {len(texts)} texts...")
        
        embs = self.model.encode(
            texts, 
            normalize_embeddings=True,
            batch_size=32  # Optimal for RTX 3060
        )
        return np.array(embs).tolist()