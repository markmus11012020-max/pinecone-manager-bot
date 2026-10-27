"""
Кастомные компоненты эмбеддингов для hay_v2_bot.

В Haystack 3.x компоненты sentence-transformers вынесены из ядра, поэтому здесь
используются собственные лёгкие ``@component``-обёртки. Одна и та же модель
(``EMBEDDING_MODEL`` из .env) применяется и при индексации документов, и при
поиске по запросу пользователя.

Модель загружается один раз и передаётся обоим пайплайнам через
``SentenceTransformerEmbedder`` — так не тратится память на две копии весов.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import replace
from typing import Any, Dict, List, Optional, Sequence

from haystack import Document, component

from config.settings import EMBEDDING_MODEL

logger = logging.getLogger("hay_v2_bot.embeddings")

# Одна модель на процесс: разные пайплайны делят один объект.
_MODEL_LOCK = threading.Lock()


class SentenceTransformerEmbedder:
    """
    Тонкая обёртка над ``sentence-transformers``.

    Векторы нормализуются (``normalize_embeddings=True``), поэтому косинусное
    сходство в Pinecone (metric=cosine) считается корректно.
    """

    def __init__(self, model_name: Optional[str] = None) -> None:
        self.model_name = model_name or EMBEDDING_MODEL
        logger.info("[EMBED] Загрузка модели sentence-transformers: %s", self.model_name)

        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(self.model_name)
        dimension_getter = getattr(self.model, "get_embedding_dimension", None) or getattr(
            self.model, "get_sentence_embedding_dimension", None
        )
        self.dimension = int(dimension_getter() or 0) if dimension_getter else 0
        if not self.dimension:
            probe = self.model.encode("проверка", convert_to_numpy=True)
            self.dimension = int(len(probe))

        logger.info(
            "[EMBED] Модель готова | model=%s | dim=%d",
            self.model_name,
            self.dimension,
        )

    def embed(self, text: str) -> List[float]:
        """Построить нормализованный вектор одного текста."""
        if not (text or "").strip():
            raise ValueError("Нельзя построить эмбеддинг пустого текста")
        vector = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return [float(x) for x in vector.tolist()]

    def embed_many(self, texts: Sequence[str]) -> List[List[float]]:
        """Построить нормализованные векторы для списка текстов."""
        if not texts:
            return []
        vectors = self.model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [[float(x) for x in vector.tolist()] for vector in vectors]


@component
class SentenceTransformerDocumentEmbedder:
    """Haystack-компонент: проставляет ``Document.embedding`` для чанков."""

    def __init__(
        self,
        embedder: Optional[SentenceTransformerEmbedder] = None,
        model_name: Optional[str] = None,
        batch_size: int = 32,
    ) -> None:
        self.embedder = embedder or SentenceTransformerEmbedder(model_name=model_name)
        self.batch_size = max(int(batch_size), 1)
        logger.info(
            "[EMBED] DocumentEmbedder готов | model=%s | dim=%d",
            self.embedder.model_name,
            self.embedder.dimension,
        )

    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]) -> Dict[str, Any]:
        documents = documents or []
        logger.info("[EMBED] Этап 1/2: эмбеддинги для %d чанков", len(documents))
        if not documents:
            logger.warning("[EMBED] Список документов пуст — эмбеддинги не строятся")
            return {"documents": []}

        embedded_docs: List[Document] = []
        for start in range(0, len(documents), self.batch_size):
            batch = documents[start : start + self.batch_size]
            texts = [(doc.content or "").strip() for doc in batch]
            if any(not text for text in texts):
                bad = [doc.id for doc, text in zip(batch, texts) if not text]
                logger.error("[EMBED] Пустой текст в чанках: %s", bad)
                raise ValueError(f"Пустые чанки нельзя записать в Pinecone: {bad}")

            logger.info(
                "[EMBED] Батч %d..%d из %d",
                start + 1,
                start + len(batch),
                len(documents),
            )
            vectors = self.embedder.embed_many(texts)
            # Haystack требует не мутировать Document, а создавать копию.
            for doc, vector in zip(batch, vectors):
                embedded_docs.append(replace(doc, embedding=vector))

            logger.info(
                "[EMBED] Батч готов | embedded=%d | dim=%d",
                len(vectors),
                len(vectors[0]) if vectors else 0,
            )

        logger.info(
            "[EMBED] Этап 1/2 завершён: эмбеддинги построены для %d чанков",
            len(embedded_docs),
        )
        return {"documents": embedded_docs}


@component
class SentenceTransformerTextEmbedder:
    """Haystack-компонент: эмбеддинг текстового запроса пользователя."""

    def __init__(
        self,
        embedder: Optional[SentenceTransformerEmbedder] = None,
        model_name: Optional[str] = None,
    ) -> None:
        self.embedder = embedder or SentenceTransformerEmbedder(model_name=model_name)
        logger.info(
            "[EMBED] TextEmbedder готов | model=%s | dim=%d",
            self.embedder.model_name,
            self.embedder.dimension,
        )

    @component.output_types(embedding=List[float], text=str)
    def run(self, text: str) -> Dict[str, Any]:
        text = (text or "").strip()
        if not text:
            raise ValueError("Пустой запрос нельзя превратить в эмбеддинг")
        logger.info("[EMBED] Этап 1/4: эмбеддинг запроса | символов=%d", len(text))
        embedding = self.embedder.embed(text)
        logger.info("[EMBED] Эмбеддинг запроса готов | dim=%d", len(embedding))
        return {"embedding": embedding, "text": text}


__all__ = [
    "SentenceTransformerEmbedder",
    "SentenceTransformerDocumentEmbedder",
    "SentenceTransformerTextEmbedder",
]
