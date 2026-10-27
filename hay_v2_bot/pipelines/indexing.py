"""
Пайплайн индексации файлов: Docling -> чанки -> эмбеддинги -> Pinecone.

Каждый чанк получает метаданные ``user_id``, ``file_name`` и ``page_number``
ещё в Docling-компоненте, поэтому они гарантированно доходят до Pinecone.

Запись выполняется кастомным компонентом ``PineconeDocumentWriter``: он
логирует точное место сбоя (индекс, namespace, id документа, размерность
эмбеддинга, ключи метаданных), как требуется в задаче.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from haystack import Document, Pipeline, component
from haystack.document_stores.types import DuplicatePolicy
from haystack.utils import Secret
from haystack_integrations.document_stores.pinecone import PineconeDocumentStore

from config.settings import (
    EMBEDDING_DIMENSION,
    PINECONE_API_KEY,
    PINECONE_ENVIRONMENT,
    PINECONE_INDEX_NAME,
    PINECONE_PROXY_URL,
)
from hay_v2_bot.components.docling_converter import DoclingDocumentConverter
from hay_v2_bot.components.embeddings import (
    SentenceTransformerDocumentEmbedder,
    SentenceTransformerEmbedder,
)

logger = logging.getLogger("hay_v2_bot.indexing")


def _apply_proxy_defaults() -> None:
    """
    PineconeDocumentStore не принимает ``proxy_url`` напрямую.

    Прокси из ``PINECONE_PROXY_URL`` пробрасывается через стандартные переменные
    окружения, которые читают httpx/requests внутри Pinecone SDK.
    """
    proxy = (PINECONE_PROXY_URL or "").strip()
    if not proxy:
        return
    for variable in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"):
        os.environ.setdefault(variable, proxy)
    logger.info("[INDEX] Для Pinecone прописан HTTP(S)-прокси из PINECONE_PROXY_URL")


def build_document_store(
    *,
    api_key: Optional[str] = None,
    index_name: Optional[str] = None,
    namespace: str = "",
    dimension: Optional[int] = None,
) -> PineconeDocumentStore:
    """
    Создать ``PineconeDocumentStore`` на индексе проекта.

    Namespace по умолчанию пустой — как у обычного бота и модуля ``hay``.
    Изоляция пользователей обеспечивается фильтром по ``user_id``.
    """
    _apply_proxy_defaults()

    key = api_key or PINECONE_API_KEY
    if not key:
        raise ValueError(
            "PINECONE_API_KEY не задан. Проверьте .env в корне проекта."
        )

    index = index_name or PINECONE_INDEX_NAME
    dim = dimension or EMBEDDING_DIMENSION

    logger.info(
        "[INDEX] Инициализация PineconeDocumentStore | index=%s | namespace=%r | dim=%d | metric=cosine",
        index,
        namespace or "<default>",
        dim,
    )

    store = PineconeDocumentStore(
        api_key=Secret.from_token(key),
        index=index,
        namespace=namespace,
        dimension=dim,
        metric="cosine",
        show_progress=False,
        spec={"serverless": {"cloud": "aws", "region": PINECONE_ENVIRONMENT}},
    )

    try:
        total = store.count_documents()
    except Exception as exc:  # noqa: BLE001 - нужно явно показать проблему доступа
        logger.exception(
            "[INDEX] Не удалось подключиться к Pinecone | index=%s | namespace=%r | причина=%s",
            index,
            namespace or "<default>",
            exc,
        )
        raise RuntimeError(
            f"Pinecone недоступен (index={index}, namespace={namespace or '<default>'}): {exc}"
        ) from exc

    logger.info(
        "[INDEX] PineconeDocumentStore готов | index=%s | namespace=%r | документов=%d",
        index,
        namespace or "<default>",
        total,
    )
    return store


@component
class PineconeDocumentWriter:
    """Haystack-компонент записи чанков в Pinecone с детальным логированием."""

    def __init__(
        self,
        document_store: PineconeDocumentStore,
        policy: DuplicatePolicy = DuplicatePolicy.OVERWRITE,
    ) -> None:
        self.document_store = document_store
        self.policy = policy
        logger.info(
            "[INDEX] PineconeDocumentWriter создан | index=%s | namespace=%r | policy=%s",
            self.document_store.index_name,
            getattr(self.document_store, "namespace", "") or "<default>",
            self.policy,
        )

    @component.output_types(documents_written=int)
    def run(self, documents: List[Document]) -> Dict[str, int]:
        documents = documents or []
        logger.info("[INDEX] Этап 3/3: запись в Pinecone | чанков=%d", len(documents))

        if not documents:
            logger.warning("[INDEX] Нет документов для записи — шаг пропущен")
            return {"documents_written": 0}

        for index, doc in enumerate(documents, 1):
            meta = doc.meta or {}
            logger.info(
                "[INDEX] Документ %d/%d | id=%s | user=%s | file=%s | page=%s | dim=%d | meta=%s",
                index,
                len(documents),
                doc.id,
                meta.get("user_id"),
                meta.get("file_name"),
                meta.get("page_number"),
                len(doc.embedding or []),
                sorted(meta.keys()),
            )
            if not doc.embedding:
                logger.error(
                    "[INDEX] У документа %s нет эмбеддинга — запись невозможна", doc.id
                )
                raise ValueError(f"У документа {doc.id} нет эмбеддинга")

        namespace = getattr(self.document_store, "namespace", "") or "<default>"
        try:
            written = self.document_store.write_documents(documents, policy=self.policy)
        except Exception as exc:
            first = documents[0]
            logger.exception(
                "[INDEX] ОШИБКА ЗАПИСИ В PINECONE | index=%s | namespace=%r | policy=%s | "
                "документов=%d | первый_id=%s | user=%s | file=%s | page=%s | "
                "dim=%d | meta=%s | причина=%s",
                self.document_store.index_name,
                namespace,
                self.policy,
                len(documents),
                first.id,
                (first.meta or {}).get("user_id"),
                (first.meta or {}).get("file_name"),
                (first.meta or {}).get("page_number"),
                len(first.embedding or []),
                sorted((first.meta or {}).keys()),
                exc,
            )
            raise RuntimeError(
                f"Не удалось сохранить {len(documents)} чанков в Pinecone "
                f"(index={self.document_store.index_name}, namespace={namespace}): {exc}"
            ) from exc

        logger.info(
            "[INDEX] Этап 3/3 завершён: Pinecone подтвердил запись | written=%d | index=%s | namespace=%r",
            written,
            self.document_store.index_name,
            namespace,
        )
        return {"documents_written": int(written)}


class IndexingPipeline:
    """Обёртка над Haystack-пайплайном индексации одного файла."""

    def __init__(
        self,
        *,
        document_store: Optional[PineconeDocumentStore] = None,
        embedder: Optional[SentenceTransformerEmbedder] = None,
        converter: Optional[DoclingDocumentConverter] = None,
        max_tokens: int = 512,
        policy: DuplicatePolicy = DuplicatePolicy.OVERWRITE,
    ) -> None:
        logger.info("[INDEX] Сборка пайплайна индексации...")
        self.document_store = document_store or build_document_store()
        self.embedder = embedder or SentenceTransformerEmbedder()

        self.converter = converter or DoclingDocumentConverter(
            embedding_model=self.embedder.model_name,
            # Переиспользуем токенизатор уже загруженной модели — не качаем второй.
            tokenizer=getattr(self.embedder.model, "tokenizer", None),
            max_tokens=max_tokens,
        )
        self.document_embedder = SentenceTransformerDocumentEmbedder(
            embedder=self.embedder
        )
        self.writer = PineconeDocumentWriter(
            document_store=self.document_store,
            policy=policy,
        )

        self.pipeline = self._build_pipeline()
        logger.info(
            "[INDEX] Пайплайн индексации готов | компоненты=%s | связи=converter->embedder->writer",
            list(self.pipeline.graph.nodes.keys()),
        )

    def _build_pipeline(self) -> Pipeline:
        pipeline = Pipeline()
        pipeline.add_component("converter", self.converter)
        pipeline.add_component("embedder", self.document_embedder)
        pipeline.add_component("writer", self.writer)

        pipeline.connect("converter.documents", "embedder.documents")
        pipeline.connect("embedder.documents", "writer.documents")
        return pipeline

    def run(self, *, file_path: str, user_id: str, file_name: str = "") -> Dict[str, Any]:
        """
        Разобрать и сохранить файл в Pinecone.

        :returns: ``{"documents": [...], "documents_written": N, "file_name": ...}``.
        """
        uid = str(user_id or "").strip()
        name = (file_name or "").strip()
        logger.info("=" * 78)
        logger.info(
            "[INDEX] Запуск индексации | user=%s | file=%s | path=%s",
            uid,
            name or "<из пути>",
            file_path,
        )

        try:
            result = self.pipeline.run(
                {
                    "converter": {
                        "file_path": file_path,
                        "user_id": uid,
                        "file_name": name,
                    },
                },
                include_outputs_from={"converter", "embedder", "writer"},
            )
        except Exception as exc:
            logger.exception(
                "[INDEX] ПАЙПЛАЙН ИНДЕКСАЦИИ УПАЛ | user=%s | file=%s | path=%s | причина=%s",
                uid,
                name or "<из пути>",
                file_path,
                exc,
            )
            raise

        documents = result.get("converter", {}).get("documents", []) or []
        written = int(result.get("writer", {}).get("documents_written", 0) or 0)
        logger.info(
            "[INDEX] Индексация завершена | user=%s | file=%s | чанков=%d | записано=%d",
            uid,
            name or "<из пути>",
            len(documents),
            written,
        )
        logger.info("=" * 78)
        return {
            "documents": documents,
            "documents_written": written,
            "file_name": name,
        }


def build_indexing_pipeline(**kwargs: Any) -> IndexingPipeline:
    """Фабричная функция для main.py."""
    return IndexingPipeline(**kwargs)


__all__ = [
    "build_document_store",
    "build_indexing_pipeline",
    "IndexingPipeline",
    "PineconeDocumentWriter",
]
