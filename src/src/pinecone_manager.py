"""
PineconeManager — encapsulates all interactions with the Pinecone vector database.
Supports automatic embedding generation, duplicate filtering via cosine similarity,
and per-user isolation via metadata.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

from config.settings import (
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL,
    PINECONE_API_KEY,
    PINECONE_BASE_URL,
    PINECONE_ENVIRONMENT,
    PINECONE_INDEX_NAME,
    SIMILARITY_THRESHOLD,
    TOP_K_MEMORIES,
)

logger = logging.getLogger(__name__)


class PineconeManager:
    """
    Manages vector storage and retrieval in Pinecone with automatic embedding
    and cosine-similarity based duplicate filtering.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        environment: Optional[str] = None,
        index_name: Optional[str] = None,
        base_url: Optional[str] = None,
        embedding_model: Optional[str] = None,
        similarity_threshold: Optional[float] = None,
    ) -> None:
        self.api_key = api_key or PINECONE_API_KEY
        self.environment = environment or PINECONE_ENVIRONMENT
        self.index_name = index_name or PINECONE_INDEX_NAME
        self.base_url = (base_url or PINECONE_BASE_URL or "").strip() or None
        self.similarity_threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else SIMILARITY_THRESHOLD
        )

        if not self.api_key:
            raise ValueError("PINECONE_API_KEY is required")

        # Initialise Pinecone client (supports optional proxy via proxy_url / PINECONE_BASE_URL)
        client_kwargs: Dict[str, Any] = {"api_key": self.api_key}
        if self.base_url:
            # Official SDK uses proxy_url for HTTP/HTTPS proxies
            client_kwargs["proxy_url"] = self.base_url
            logger.info("Using proxy_url=%s for Pinecone client", self.base_url)
        try:
            self.pc = Pinecone(**client_kwargs)
        except TypeError:
            # Fallback for older SDK versions that do not accept proxy_url
            self.pc = Pinecone(api_key=self.api_key)
            if self.base_url:
                logger.warning(
                    "PINECONE_BASE_URL was provided but the installed pinecone-client "
                    "does not accept proxy_url. Proxy support may be limited."
                )

        # Ensure index exists
        self._ensure_index()

        self.index = self.pc.Index(self.index_name)

        # Load embedding model (lazy-friendly, but we load once)
        model_name = embedding_model or EMBEDDING_MODEL
        logger.info("Loading embedding model: %s", model_name)
        self.embedder = SentenceTransformer(model_name)
        # Verify dimension
        test_vec = self.embedder.encode("test", convert_to_numpy=True)
        self.dim = len(test_vec)
        if self.dim != EMBEDDING_DIMENSION:
            logger.warning(
                "Model dimension %d differs from configured EMBEDDING_DIMENSION %d. "
                "Using model dimension.",
                self.dim,
                EMBEDDING_DIMENSION,
            )

    def _ensure_index(self) -> None:
        """Create the index if it does not exist (serverless)."""
        existing = [idx.name for idx in self.pc.list_indexes()]
        if self.index_name not in existing:
            logger.info("Creating Pinecone index '%s' (dim=%d)", self.index_name, EMBEDDING_DIMENSION)
            self.pc.create_index(
                name=self.index_name,
                dimension=EMBEDDING_DIMENSION,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region=self.environment),
            )
            logger.info("Index created successfully")
        else:
            logger.info("Using existing index '%s'", self.index_name)

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------
    def embed_text(self, text: str) -> List[float]:
        """Convert text to a dense embedding vector."""
        vec = self.embedder.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        return vec.tolist()

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Batch embed multiple texts."""
        vecs = self.embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return vecs.tolist()

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------
    def upsert_raw(
        self,
        texts: Union[str, List[str]],
        user_id: str,
        metadata: Optional[Dict[str, Any]] = None,
        ids: Optional[List[str]] = None,
        namespace: Optional[str] = None,
    ) -> List[str]:
        """
        Embed text(s) and upsert into Pinecone.
        Each vector is tagged with user_id for isolation.
        Returns the list of vector IDs that were written.
        """
        if isinstance(texts, str):
            texts = [texts]
        if not texts:
            return []

        vectors = self.embed_texts(texts)
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]
        if len(ids) != len(texts):
            raise ValueError("ids length must match texts length")

        base_meta = metadata or {}
        records = []
        for i, (vid, text, vec) in enumerate(zip(ids, texts, vectors)):
            meta = {
                **base_meta,
                "user_id": str(user_id),
                "text": text,
            }
            records.append({"id": vid, "values": vec, "metadata": meta})

        ns = namespace or ""
        self.index.upsert(vectors=records, namespace=ns)
        logger.debug("Upserted %d vector(s) for user_id=%s", len(records), user_id)
        return ids

    def upsert_vectors(
        self,
        vectors: List[Dict[str, Any]],
        namespace: Optional[str] = None,
    ) -> None:
        """
        Upsert already-prepared vectors.
        Each dict must contain: id, values, and optionally metadata.
        Metadata should include user_id.
        """
        ns = namespace or ""
        self.index.upsert(vectors=vectors, namespace=ns)

    # ------------------------------------------------------------------
    # Read / search operations
    # ------------------------------------------------------------------
    def query_by_vector(
        self,
        vector: List[float],
        user_id: Optional[str] = None,
        top_k: int = TOP_K_MEMORIES,
        namespace: Optional[str] = None,
        include_metadata: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Search top-k most similar vectors.
        If user_id is given, filter results to that user only.
        """
        ns = namespace or ""
        filter_dict = {"user_id": {"$eq": str(user_id)}} if user_id else None

        response = self.index.query(
            vector=vector,
            top_k=top_k,
            namespace=ns,
            filter=filter_dict,
            include_metadata=include_metadata,
            include_values=False,
        )
        matches = response.get("matches", []) if isinstance(response, dict) else getattr(response, "matches", [])
        results = []
        for m in matches:
            item = {
                "id": m["id"] if isinstance(m, dict) else m.id,
                "score": m["score"] if isinstance(m, dict) else m.score,
            }
            meta = m.get("metadata") if isinstance(m, dict) else getattr(m, "metadata", None)
            if meta:
                item["metadata"] = meta
            results.append(item)
        return results

    def query_by_text(
        self,
        text: str,
        user_id: Optional[str] = None,
        top_k: int = TOP_K_MEMORIES,
        namespace: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Embed text and search for similar memories."""
        vector = self.embed_text(text)
        return self.query_by_vector(
            vector=vector,
            user_id=user_id,
            top_k=top_k,
            namespace=namespace,
        )

    # ------------------------------------------------------------------
    # Duplicate-aware write (core memory logic)
    # ------------------------------------------------------------------
    def store_if_new(
        self,
        text: str,
        user_id: str,
        metadata: Optional[Dict[str, Any]] = None,
        namespace: Optional[str] = None,
        update_on_duplicate: bool = False,
    ) -> Tuple[bool, Optional[str], float]:
        """
        Store text only if it is sufficiently different from existing memories
        of the same user (cosine similarity < SIMILARITY_THRESHOLD).

        Returns:
            (was_stored: bool, vector_id: Optional[str], max_similarity: float)
        """
        text = (text or "").strip()
        if not text:
            return False, None, 0.0

        # 1. Find the most similar existing memory for this user
        matches = self.query_by_text(text, user_id=user_id, top_k=1, namespace=namespace)

        max_sim = 0.0
        closest_id = None
        if matches:
            max_sim = float(matches[0]["score"])
            closest_id = matches[0]["id"]

        # 2. Decide
        if max_sim >= self.similarity_threshold:
            # Duplicate / variation
            logger.info(
                "Duplicate detected (sim=%.4f >= %.2f). Skipping write for user %s",
                max_sim,
                self.similarity_threshold,
                user_id,
            )
            if update_on_duplicate and closest_id:
                # Optionally refresh the existing vector with new text
                new_id = self.upsert_raw(
                    texts=text,
                    user_id=user_id,
                    metadata=metadata,
                    ids=[closest_id],
                    namespace=namespace,
                )[0]
                return False, new_id, max_sim
            return False, closest_id, max_sim

        # 3. New information → write
        new_ids = self.upsert_raw(
            texts=text,
            user_id=user_id,
            metadata=metadata,
            namespace=namespace,
        )
        logger.info(
            "New memory stored (sim=%.4f < %.2f) id=%s for user %s",
            max_sim,
            self.similarity_threshold,
            new_ids[0],
            user_id,
        )
        return True, new_ids[0], max_sim

    def get_relevant_context(
        self,
        query: str,
        user_id: str,
        top_k: int = TOP_K_MEMORIES,
        namespace: Optional[str] = None,
    ) -> str:
        """
        Retrieve top-k relevant memories and format them as a context string
        suitable for feeding into an LLM.
        """
        matches = self.query_by_text(query, user_id=user_id, top_k=top_k, namespace=namespace)
        if not matches:
            return ""

        lines = []
        for i, m in enumerate(matches, 1):
            text = (m.get("metadata") or {}).get("text", "")
            score = m.get("score", 0.0)
            if text:
                lines.append(f"[{i}] (score={score:.3f}) {text}")
        return "\n".join(lines)

    def delete_user_memories(self, user_id: str, namespace: Optional[str] = None) -> None:
        """Delete all vectors belonging to a user (useful for /reset)."""
        # Pinecone does not support delete-by-filter in all plans; we use a workaround
        # by querying with a dummy vector and deleting by IDs in batches if needed.
        # For simplicity we log a warning – full implementation depends on Pinecone plan.
        logger.warning(
            "delete_user_memories is a stub. Implement with index.delete(filter=...) "
            "if your Pinecone plan supports it."
        )
        # Example (if supported):
        # self.index.delete(filter={"user_id": {"$eq": str(user_id)}}, namespace=namespace or "")
