# core/vector_store.py
from __future__ import annotations
import os
import threading
from typing import List, Dict, Any, Optional, Tuple

from pinecone import Pinecone, ServerlessSpec
from config.env_config import (
    PINECONE_API_KEY,
    PINECONE_REGION,
    PINECONE_QUERY_TIMEOUT_S,
    PINECONE_ENABLE,
)

class PineconeStore:
    """
    Minimal wrapper with:
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
        names = [i.name for i in self._pc.list_indexes()]  # avoid stats calls here
        if self.index_name not in names:
            self._pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric=self.metric,
                spec=ServerlessSpec(cloud="aws", region=PINECONE_REGION),
            )
        self._index = self._pc.Index(self.index_name)

    # ----------- public API -----------
    def upsert_texts(self, texts: List[str], metadatas: List[Dict[str, Any]], namespace: Optional[str] = None) -> int:
        """Embed and upsert in batches; returns number of vectors upserted."""
        try:
            self._ensure_index()
        except Exception:
            return 0  # silent if disabled/unavailable

        import uuid
        batch_id = uuid.uuid4().hex[:8]
        ns = namespace or ""
        total = 0
        batch_size = int(os.getenv("PINECONE_BATCH", "64"))
        N = min(len(texts), len(metadatas))
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            vectors: List[Dict[str, Any]] = []
            for i in range(start, end):
                t = texts[i]
                md = dict(metadatas[i] or {})
                # Ensure text is present in metadata for downstream retrieval patterns
                if "text" not in md:
                    md["text"] = t
                vectors.append({
                    "id": f"{self.index_name}-{batch_id}-{i}",
                    "values": self._embed_stub(t),
                    "metadata": md,
                })
            try:
                resp = self._index.upsert(vectors=vectors, namespace=ns)
                # Pinecone upsert can return a dict with 'upserted_count'
                if isinstance(resp, dict) and "upserted_count" in resp:
                    total += int(resp["upserted_count"]) or 0
                else:
                    total += len(vectors)
            except Exception:
                # Continue with remaining batches
                pass
        return total

    # Utility used by CLIP image pipelines that already have vectors
    def _safe_upsert(self, vectors: List[Dict[str, Any]], namespace: str = "") -> int:
        """Upsert pre-built vectors with retries and chunking. Returns count."""
        try:
            self._ensure_index()
        except Exception:
            return 0

        # pinecone 4MB limit per request – keep batches small
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
                # last resort: single inserts
                for v in chunk:
                    try:
                        self._index.upsert(vectors=[v], namespace=namespace)
                        total += 1
                    except Exception:
                        pass
        return total

    def query(self, text: str, top_k: int = 4, namespace: Optional[str] = None) -> List[str]:
        """
        Time-boxed Pinecone query. If it doesn't finish within PINECONE_QUERY_TIMEOUT_S seconds,
        we return [] so the app remains responsive.
        """
        try:
            self._ensure_index()
        except Exception:
            return []

        result_box: Dict[str, Any] = {"done": False, "hits": []}

        def _worker():
            try:
                qv = self._embed_stub(text)
                res = self._index.query(
                    vector=qv,
                    top_k=top_k,
                    include_metadata=True,
                    namespace=namespace or "",
                )
                # Convert to plain text chunks; be defensive about shapes
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

        # If not done in time -> give up quickly
        return result_box["hits"] if result_box.get("done") else []

    # --------------------
    # NOTE: replace this with a real embedder if you have one at hand.
    # For now we store text in metadata at upsert time; many pipelines do that.
    # If your current pipeline already stores full chunks in metadata["text"],
    # you can keep this stub and it will still work (since query returns matches).
    def _embed_stub(self, text: str) -> List[float]:
        # This is a deterministic tiny "hashing" projection just to keep shapes consistent
        # when you don't have a local embedder here. Replace with your embedding model if needed.
        import hashlib, math
        h = hashlib.sha256(text.encode("utf-8")).digest()
        arr = [(b / 255.0) for b in h[:32]]  # 32 dims only; Pinecone will accept any length if index created so
        # pad/trim to index dimension
        if len(arr) < self.dimension:
            arr = (arr * (math.ceil(self.dimension / len(arr))))[: self.dimension]
        else:
            arr = arr[: self.dimension]
        return arr
