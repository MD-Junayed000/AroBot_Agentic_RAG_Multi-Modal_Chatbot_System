import os, uuid, json
from typing import List, Dict
from pathlib import Path
from pinecone import Pinecone, ServerlessSpec
from .embeddings import Embedder
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

MAX_BYTES = 3_900_000  # keep a buffer under Pinecone's 4 MB limit

class PineconeStore:
    def __init__(self, index_name: str = None, dimension: int = 384, metric: str = "cosine", batch_size: int = None):
        self.index_name = index_name or os.environ.get("PINECONE_INDEX", "arobot-default")
        self.batch_size = batch_size or int(os.getenv("PINECONE_BATCH", "32"))
        self.max_metadata_chars = 1200
        from config.env_config import PINECONE_API_KEY
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        existing = [idx.name for idx in self.pc.list_indexes()]
        if self.index_name not in existing:
            self.pc.create_index(
                name=self.index_name,
                dimension=dimension,
                metric=metric,
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
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

    def _safe_upsert(self, vectors, namespace: str | None):
        # Recursively split until each request fits under the byte budget
        stack = [vectors]
        while stack:
            batch = stack.pop()
            if not batch:
                continue
            size = len(json.dumps(batch, default=str).encode("utf-8"))

            if size > MAX_BYTES and len(batch) > 1:
                mid = len(batch) // 2
                stack.append(batch[:mid])
                stack.append(batch[mid:])
                continue
            if size > MAX_BYTES:
                # last-resort: shrink long string metadata
                for v in batch:
                    md = v.get("metadata", {})
                    for k in list(md.keys()):
                        if isinstance(md[k], str) and len(md[k]) > 512:
                            md[k] = md[k][:512]
                # re-check not strictly necessary; send it
            self.index.upsert(vectors=batch, namespace=namespace)

    def upsert_texts(self, texts: List[str], metadatas: List[Dict], namespace: str | None = None):
        start_idx = self._load_checkpoint()
        if start_idx > 0:
            logger.info(f"Resuming from checkpoint: {start_idx}/{len(texts)}")
            texts = texts[start_idx:]
            metadatas = metadatas[start_idx:]

        total_texts = len(texts)
        for i in tqdm(range(0, total_texts, self.batch_size), desc=f"Upserting to {self.index_name}:{namespace or 'default'}"):
            batch_texts = texts[i:i + self.batch_size]
            batch_metadatas = metadatas[i:i + self.batch_size]
            embs = self.embedder.embed(batch_texts)

            vectors = []
            for text, metadata, embedding in zip(batch_texts, batch_metadatas, embs):
                vector_id = metadata.get("id", str(uuid.uuid4()))
                trimmed_meta = self._trim_metadata(metadata)
                trimmed_meta["text"] = text[:self.max_metadata_chars]
                vectors.append({"id": vector_id, "values": embedding, "metadata": trimmed_meta})

            self._safe_upsert(vectors, namespace)

            processed_count = start_idx + i + len(batch_texts)
            self._save_checkpoint(processed_count)

        self._clear_checkpoint()
        logger.info(f"Successfully upserted {total_texts} texts to {self.index_name}:{namespace or 'default'}")

    def query(self, query_text: str, top_k: int = 5, namespace: str | None = None):
        q = self.embedder.embed([query_text])[0]
        res = self.index.query(vector=q, top_k=top_k, include_metadata=True, namespace=namespace)
        hits = res.get("matches", [])
        return [h["metadata"]["text"] for h in hits]
