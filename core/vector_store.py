# core/vector_store.py
from __future__ import annotations
import os
import threading
from typing import List, Dict, Any, Optional, Tuple

try:
    from pinecone import Pinecone
    from pinecone import ServerlessSpec
except ImportError as e:
    print(f"⚠️  Pinecone import failed: {e}")
    print("Please install: pip install 'pinecone-client[grpc]>=6.0.0,<7.0.0'")
    raise

from config.env_config import (
    PINECONE_API_KEY,
    PINECONE_REGION,
    PINECONE_QUERY_TIMEOUT_S,
    PINECONE_ENABLE,
)

class PineconeStore:
    """
    Pinecone v6+ compatible wrapper with:
      - Lazy client/index creation (no calls at import time)
      - Serverless index auto-create (if missing) in configured region
      - Time-boxed query(): returns [] if Pinecone stalls past PINECONE_QUERY_TIMEOUT_S
    """
    def __init__(self, index_name: str, dimension: int = 384, metric: str = "cosine"):
        self.index_name = index_name
        self.dimension = dimension
        self.metric = metric
        self._pc: Optional[Pinecone] = None
        self._index = None
        self._disabled = not PINECONE_ENABLE
        self._embedder = None  # lazy

    # ----------- internal -----------
    def _ensure_client(self):
        if self._disabled:
            raise RuntimeError("Pinecone disabled by PINECONE_ENABLE=0")
        if not PINECONE_API_KEY:
            raise RuntimeError("PINECONE_API_KEY not set")
        if self._pc is None:
            self._pc = Pinecone(api_key=PINECONE_API_KEY)

    def _ensure_index(self):
        if self._index is not None:
            return
        self._ensure_client()
        names = [i.name for i in self._pc.list_indexes()]
        if self.index_name not in names:
            self._pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric=self.metric,
                spec=ServerlessSpec(cloud="aws", region=PINECONE_REGION),
            )
        self._index = self._pc.Index(self.index_name)

    def _ensure_embedder(self):
        """Try to initialize a sentence-transformers Embedder. Fallback to stub on failure."""
        if self._embedder is not None:
            return
        try:
            from core.embeddings import Embedder  # type: ignore
            self._embedder = Embedder()
        except Exception:
            self._embedder = None

    # ----------- public API -----------
    def list_indexes(self) -> List[str]:
        """List available Pinecone index names."""
        try:
            self._ensure_client()
            return [i.name for i in self._pc.list_indexes()]
        except Exception:
            return []

    def create_index(self, name: str, dimension: int = 384, metric: str = "cosine") -> bool:
        """Create an index if it does not exist. Returns True on success or if exists."""
        try:
            self._ensure_client()
            names = [i.name for i in self._pc.list_indexes()]
            if name not in names:
                self._pc.create_index(
                    name=name,
                    dimension=dimension,
                    metric=metric,
                    spec=ServerlessSpec(cloud="aws", region=PINECONE_REGION),
                )
            return True
        except Exception:
            return False

    def get_index_stats(self, name: Optional[str] = None) -> Dict[str, Any]:
        """Describe index stats for the given name (or this store's index)."""
        try:
            self._ensure_client()
            idx_name = name or self.index_name
            idx = self._pc.Index(idx_name)
            stats = idx.describe_index_stats()
            return self._make_json_safe(stats)
        except Exception:
            return {"error": "unavailable"}

    @staticmethod
    def _make_json_safe(value: Any) -> Any:
        """Convert Pinecone SDK objects into plain JSON-serialisable structures."""
        if value is None:
            return None
        if hasattr(value, "to_dict"):
            try:
                value = value.to_dict()
            except Exception:
                pass
        if isinstance(value, dict):
            return {k: PineconeStore._make_json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [PineconeStore._make_json_safe(v) for v in value]
        return value

    def upsert_texts(self, texts: List[str], metadatas: List[Dict[str, Any]], namespace: Optional[str] = None) -> int:
        """Embed and upsert in batches; returns number of vectors upserted."""
        try:
            self._ensure_index()
        except Exception:
            return 0

        import uuid
        batch_id = uuid.uuid4().hex[:8]
        ns = namespace or ""
        total = 0
        batch_size = int(os.getenv("PINECONE_BATCH", "64"))
        N = min(len(texts), len(metadatas))
        self._ensure_embedder()
        
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            vectors: List[Dict[str, Any]] = []
            embeddings: List[List[float]]
            
            if self._embedder is not None:
                try:
                    embeddings = self._embedder.embed(texts[start:end])
                except Exception:
                    embeddings = [self._embed_stub(texts[i]) for i in range(start, end)]
            else:
                embeddings = [self._embed_stub(texts[i]) for i in range(start, end)]
            
            for i in range(start, end):
                t = texts[i]
                md = dict(metadatas[i] or {})
                if "text" not in md:
                    md["text"] = t
                vectors.append({
                    "id": f"{self.index_name}-{batch_id}-{i}",
                    "values": embeddings[i - start],
                    "metadata": md,
                })
            
            try:
                resp = self._index.upsert(vectors=vectors, namespace=ns)
                if isinstance(resp, dict) and "upserted_count" in resp:
                    total += int(resp["upserted_count"]) or 0
                else:
                    total += len(vectors)
            except Exception:
                pass
        
        return total

    def _safe_upsert(self, vectors: List[Dict[str, Any]], namespace: str = "") -> int:
        """Upsert pre-built vectors with retries and chunking. Returns count."""
        try:
            self._ensure_index()
        except Exception:
            return 0

        batch_size = int(os.getenv("PINECONE_BATCH", "32"))
        total = 0
        for start in range(0, len(vectors), batch_size):
            chunk = vectors[start:start + batch_size]
            try:
                resp = self._index.upsert(vectors=chunk, namespace=namespace)
                if isinstance(resp, dict) and "upserted_count" in resp:
                    total += int(resp["upserted_count"]) or 0
                else:
                    total += len(chunk)
            except Exception:
                for v in chunk:
                    try:
                        self._index.upsert(vectors=[v], namespace=namespace)
                        total += 1
                    except Exception:
                        pass
        return total

    def query(self, text: str, top_k: int = 4, namespace: Optional[str] = None) -> List[str]:
        """Time-boxed Pinecone query."""
        try:
            self._ensure_index()
        except Exception:
            return []

        result_box: Dict[str, Any] = {"done": False, "hits": []}

        def _worker():
            try:
                self._ensure_embedder()
                if self._embedder is not None:
                    try:
                        qv = self._embedder.embed([text])[0]
                    except Exception:
                        qv = self._embed_stub(text)
                else:
                    qv = self._embed_stub(text)
                
                res = self._index.query(
                    vector=qv,
                    top_k=top_k,
                    include_metadata=True,
                    namespace=namespace or "",
                )
                
                hits = []
                for m in (res.get("matches") or []):
                    md = m.get("metadata") or {}
                    chunk = md.get("text") or md.get("chunk") or md.get("content") or ""
                    if chunk:
                        hits.append(str(chunk))
                result_box["hits"] = hits
            except Exception:
                result_box["hits"] = []
            finally:
                result_box["done"] = True

        th = threading.Thread(target=_worker, daemon=True)
        th.start()
        th.join(timeout=PINECONE_QUERY_TIMEOUT_S)

        return result_box["hits"] if result_box.get("done") else []

    def _embed_stub(self, text: str) -> List[float]:
        """Simple hash-based embedding stub."""
        import hashlib, math
        h = hashlib.sha256(text.encode("utf-8")).digest()
        arr = [(b / 255.0) for b in h[:32]]
        if len(arr) < self.dimension:
            arr = (arr * (math.ceil(self.dimension / len(arr))))[: self.dimension]
        else:
            arr = arr[: self.dimension]
        return arr
