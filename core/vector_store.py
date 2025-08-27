import os, uuid, json
from typing import List, Dict
from pathlib import Path
from pinecone import Pinecone, ServerlessSpec
from .embeddings import Embedder
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

class PineconeStore:
    def __init__(self, index_name: str = None, dimension: int = 384, metric: str = "cosine"):
        self.index_name = index_name or os.environ.get("PINECONE_INDEX", "arobot-default")
        self.batch_size = 64  # Safe batch size to avoid 4MB limit
        self.max_metadata_chars = 1200  # Limit metadata size
        
        # Get API key from config
        from config.env_config import PINECONE_API_KEY
        
        # Initialize Pinecone client
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        
        # Create index if it doesn't exist
        existing_indexes = [idx.name for idx in self.pc.list_indexes()]
        if self.index_name not in existing_indexes:
            self.pc.create_index(
                name=self.index_name,
                dimension=dimension,
                metric=metric,
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
        
        self.index = self.pc.Index(self.index_name)
        self.embedder = Embedder()

    def _trim_metadata(self, metadata: Dict) -> Dict:
        """Trim metadata to keep requests under size limits"""
        trimmed = {}
        for key, value in metadata.items():
            if isinstance(value, str):
                # Trim long text fields
                trimmed[key] = value[:self.max_metadata_chars]
            else:
                trimmed[key] = value
        return trimmed

    def _get_checkpoint_file(self) -> Path:
        """Get checkpoint file path for this index"""
        return Path(f".pinecone_checkpoint_{self.index_name}.txt")

    def _save_checkpoint(self, processed_count: int):
        """Save progress checkpoint"""
        checkpoint_file = self._get_checkpoint_file()
        checkpoint_file.write_text(str(processed_count))

    def _load_checkpoint(self) -> int:
        """Load progress checkpoint"""
        checkpoint_file = self._get_checkpoint_file()
        if checkpoint_file.exists():
            try:
                return int(checkpoint_file.read_text().strip())
            except:
                return 0
        return 0

    def _clear_checkpoint(self):
        """Clear checkpoint after successful completion"""
        checkpoint_file = self._get_checkpoint_file()
        if checkpoint_file.exists():
            checkpoint_file.unlink()

    def upsert_texts(self, texts: List[str], metadatas: List[Dict]):
        """Upsert texts with safe batching and checkpointing"""
        # Load checkpoint
        start_idx = self._load_checkpoint()
        
        if start_idx > 0:
            logger.info(f"Resuming from checkpoint: {start_idx}/{len(texts)}")
            texts = texts[start_idx:]
            metadatas = metadatas[start_idx:]
        
        # Prepare vectors in batches
        total_texts = len(texts)
        for i in tqdm(range(0, total_texts, self.batch_size), desc=f"Upserting to {self.index_name}"):
            batch_texts = texts[i:i + self.batch_size]
            batch_metadatas = metadatas[i:i + self.batch_size]
            
            # Generate embeddings for this batch
            embs = self.embedder.embed(batch_texts)
            
            # Prepare vectors with trimmed metadata
            vectors = []
            for text, metadata, embedding in zip(batch_texts, batch_metadatas, embs):
                vector_id = metadata.get("id", str(uuid.uuid4()))
                trimmed_meta = self._trim_metadata(metadata)
                trimmed_meta["text"] = text[:self.max_metadata_chars]  # Store trimmed text
                
                vectors.append({
                    "id": vector_id,
                    "values": embedding,
                    "metadata": trimmed_meta
                })
            
            # Estimate payload size (rough check)
            payload_size = len(json.dumps(vectors, default=str))
            if payload_size > 3_500_000:  # 3.5MB safety margin
                logger.warning(f"Batch size may be too large ({payload_size} bytes), reducing batch size")
                # Split this batch further if needed
                mid = len(vectors) // 2
                for sub_vectors in [vectors[:mid], vectors[mid:]]:
                    if sub_vectors:
                        self.index.upsert(vectors=sub_vectors)
            else:
                # Safe to upsert
                self.index.upsert(vectors=vectors)
            
            # Save checkpoint
            processed_count = start_idx + i + len(batch_texts)
            self._save_checkpoint(processed_count)
        
        # Clear checkpoint on successful completion
        self._clear_checkpoint()
        logger.info(f"Successfully upserted {total_texts} texts to {self.index_name}")

    def query(self, query_text: str, top_k: int = 5):
        q = self.embedder.embed([query_text])[0]
        res = self.index.query(vector=q, top_k=top_k, include_metadata=True)
        hits = res.get("matches", [])
        return [h["metadata"]["text"] for h in hits]