"""
hay/document_store.py — кастомное хранилище документов для модуля Haystack.

Отвечает за:
    * подключение к Pinecone через ``PineconeDocumentStore`` (pinecone-haystack);
    * СТРОГУЮ изоляцию контекста пользователей по ``user_id`` через метаданные —
      каждый документ помечается ``user_id``, а каждый поиск идёт с фильтром
      ``{"field": "user_id", "operator": "==", "value": <telegram_id>}``;
    * фильтрацию дубликатов по косинусному сходству (порог SIMILARITY_THRESHOLD);
    * детальное логирование каждого шага сохранения, чтобы в консоли start.bat
      было видно точное место сбоя при записи в Pinecone.

Все настройки берутся из существующего ``.env`` через ``config.settings`` —
новых констант и переменных окружения не добавляется.
"""

from __future__ import annotations

import logging
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from haystack import Document, component
from haystack.utils import Secret
from haystack_integrations.components.retrievers.pinecone import PineconeEmbeddingRetriever
from haystack_integrations.document_stores.pinecone import PineconeDocumentStore

from config.settings import (
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL,
    PINECONE_API_KEY,
    PINECONE_ENVIRONMENT,
    PINECONE_INDEX_NAME,
    PINECONE_PROXY_URL,
    SIMILARITY_THRESHOLD,
    TOP_K_MEMORIES,
)

logger = logging.getLogger("hay.document_store")

# Метка модуля для метаданных — по ней всегда можно отличить записи hay
SOURCE_TAG = "haystack_agent"

# Pinecone принимает в метаданных только простые типы: str, int, float, bool, List[str].
# None и вложенные структуры приводят к ошибке записи, поэтому значения чистим.
_PINEALONE_META_TYPES = (str, int, float, bool)


def _sanitize_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
    """Оставить только типы, поддерживаемые Pinecone, и выбросить None."""
    clean: Dict[str, Any] = {}
    for key, value in (meta or {}).items():
        if value is None:
            continue
        if isinstance(value, _PINEALONE_META_TYPES):
            clean[key] = value
        elif isinstance(value, (list, tuple)) and all(isinstance(i, str) for i in value):
            clean[key] = list(value)
        else:
            logger.debug("Метаданные '%s'=%r пропущены: тип %s не поддерживается Pinecone", key, value, type(value).__name__)
    return clean


def _apply_proxy_defaults() -> None:
    """
    PineconeDocumentStore создаёт Pinecone-клиент без параметра ``proxy_url``.

    Чтобы не терять поддержку ``PINECONE_PROXY_URL`` из .env (которую уже
    использует PineconeManager), пробрасываем прокси через переменные окружения —
    их уважают и httpx, и requests.
    """
    proxy = (PINECONE_PROXY_URL or "").strip()
    if not proxy:
        return
    for var in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"):
        os.environ.setdefault(var, proxy)
    logger.info("Для PineconeDocumentStore прописан HTTP(S)-прокси из PINECONE_PROXY_URL")


def build_document_store(
    *,
    api_key: Optional[str] = None,
    index_name: Optional[str] = None,
    namespace: str = "",
    dimension: Optional[int] = None,
) -> PineconeDocumentStore:
    """
    Создать PineconeDocumentStore на существующем индексе проекта.

    Namespace по умолчанию — пустая строка, как и в ``PineconeManager``: так модуль
    hay видит ту же векторную базу, а разделение обеспечивает фильтр по user_id.
    """
    _apply_proxy_defaults()

    key = api_key or PINECONE_API_KEY
    if not key:
        raise ValueError(
            "PINECONE_API_KEY не задан. Модуль hay не может подключиться к Pinecone. "
            "Проверьте .env в корне проекта."
        )

    index = index_name or PINECONE_INDEX_NAME
    dim = dimension or EMBEDDING_DIMENSION

    logger.info(
        "Инициализация PineconeDocumentStore | index=%s | namespace=%r | dim=%d | metric=cosine",
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

    # Заранее поднимаем соединение: первый реальный запрос к Pinecone делается
    # сразу, чтобы ошибки доступа всплыли при инициализации агента, а не в
    # момент первого сообщения пользователя.
    try:
        total = store.count_documents()
    except Exception as exc:
        logger.exception(
            "Не удалось подключиться к Pinecone | index=%s | namespace=%r | причина=%s",
            index,
            namespace or "<default>",
            exc,
        )
        raise RuntimeError(
            f"Pinecone недоступен (index={index}, namespace={namespace or '<default>'}): {exc}"
        ) from exc

    logger.info(
        "PineconeDocumentStore готов | index=%s | namespace=%r | документов в базе=%d",
        index,
        namespace or "<default>",
        total,
    )
    return store


class SentenceTransformerEmbedder:
    """
    Тонкая обёртка над sentence-transformers.

    Используется та же модель и то же векторное пространство, что и в
    PineconeManager, поэтому записи из обычного режима и из режима Haystack
    корректно сравниваются между собой по косинусному сходству.
    """

    def __init__(self, model_name: Optional[str] = None) -> None:
        self.model_name = model_name or EMBEDDING_MODEL
        logger.info("Загрузка модели эмбеддингов: %s", self.model_name)
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(self.model_name)
        probe = self.model.encode("проверка", convert_to_numpy=True)
        self.dimension = len(probe)
        logger.info("Модель эмбеддингов загружена | model=%s | dim=%d", self.model_name, self.dimension)
        if self.dimension != EMBEDDING_DIMENSION:
            logger.warning(
                "Размерность модели (%d) отличается от EMBEDDING_DIMENSION (%d) — "
                "используется размерность модели.",
                self.dimension,
                EMBEDDING_DIMENSION,
            )

    def embed(self, text: str) -> List[float]:
        vec = self.model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        return vec.tolist()

    def embed_many(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        vecs = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return [v.tolist() for v in vecs]


@component
class PineconeUserMemory:
    """
    Haystack-компонент долговременной памяти одного пользователя.

    ``run()`` выполняет два действия и подробно логирует каждый шаг:
      1. сохраняет новые факты в Pinecone (с отсечением дубликатов по косинусу);
      2. собирает релевантный контекст пользователя по его последнему сообщению.

    Изоляция: любая запись получает ``meta.user_id``, любое чтение выполняется
    с фильтром по ``user_id`` — векторы других пользователей не видны никогда.
    """

    def __init__(
        self,
        document_store: Optional[PineconeDocumentStore] = None,
        embedder: Optional[SentenceTransformerEmbedder] = None,
        similarity_threshold: Optional[float] = None,
        top_k: Optional[int] = None,
        namespace: str = "",
    ) -> None:
        self.document_store = document_store or build_document_store(namespace=namespace)
        self.embedder = embedder or SentenceTransformerEmbedder()
        self.similarity_threshold = (
            similarity_threshold if similarity_threshold is not None else SIMILARITY_THRESHOLD
        )
        self.top_k = top_k if top_k is not None else TOP_K_MEMORIES
        self.namespace = namespace
        # Фильтр user_id всегда передаётся в рантайме (filter_policy=REPLACE),
        # поэтому в инициализации ретривера он не нужен.
        self._retriever = PineconeEmbeddingRetriever(
            document_store=self.document_store,
            top_k=self.top_k,
        )
        logger.info(
            "PineconeUserMemory готов | threshold=%.2f | top_k=%d | index=%s",
            self.similarity_threshold,
            self.top_k,
            self.document_store.index_name,
        )

    # ------------------------------------------------------------------
    # Фильтры изоляции
    # ------------------------------------------------------------------
    @staticmethod
    def _user_filter(user_id: str) -> Dict[str, Any]:
        """Фильтр Haystack, который Pinecone превращает в {"user_id": {"$eq": ...}}."""
        return {"field": "user_id", "operator": "==", "value": str(user_id)}

    # ------------------------------------------------------------------
    # Запись
    # ------------------------------------------------------------------
    def _store_fact(self, text: str, user_id: str, meta: Dict[str, Any]) -> Tuple[bool, float]:
        """
        Записать один факт, если он не дублирует уже сохранённый.

        :returns: (was_stored, max_similarity)
        """
        content = (text or "").strip()
        if not content:
            return False, 0.0

        logger.info("[MEM] Проверка на дубликат | user=%s | fact=%r", user_id, content[:80])

        embedding = self.embedder.embed(content)
        logger.debug("[MEM] Эмбеддинг построен | user=%s | dim=%d", user_id, len(embedding))

        # Поиск кандидата на дубликат — строго в рамках одного пользователя.
        duplicate_search = self._retriever.run(
            query_embedding=embedding,
            filters=self._user_filter(user_id),
            top_k=1,
        )
        matches = duplicate_search.get("documents", []) or []
        max_sim = float(matches[0].score) if matches and matches[0].score is not None else 0.0

        if max_sim >= self.similarity_threshold:
            logger.info(
                "[MEM] Дубликат пропущен | user=%s | cos=%.4f >= %.2f | fact=%r",
                user_id,
                max_sim,
                self.similarity_threshold,
                content[:60],
            )
            return False, max_sim

        document = Document(
            id=str(uuid.uuid4()),
            content=content,
            embedding=embedding,
            meta=_sanitize_meta(
                {
                    "user_id": str(user_id),
                    "source": meta.get("source", SOURCE_TAG),
                    "kind": meta.get("kind", "fact"),
                    "role": meta.get("role", "user"),
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                    **meta,
                }
            ),
        )

        try:
            written = self.document_store.write_documents([document])
        except Exception as exc:
            # Точное место сбоя записи в Pinecone — важно для консоли start.bat.
            logger.exception(
                "[MEM] ОШИБКА записи в Pinecone | user=%s | index=%s | namespace=%r | "
                "document_id=%s | dim=%d | тип метаданных=%s | факт=%r",
                user_id,
                self.document_store.index_name,
                self.namespace or "<default>",
                document.id,
                len(embedding),
                sorted(document.meta.keys()),
                content[:60],
            )
            raise RuntimeError(
                f"Не удалось сохранить факт в Pinecone (user={user_id}): {exc}"
            ) from exc

        logger.info(
            "[MEM] Новый факт сохранён | user=%s | doc_id=%s | written=%s | cos_max=%.4f | fact=%r",
            user_id,
            document.id,
            written,
            max_sim,
            content[:60],
        )
        return True, max_sim

    # ------------------------------------------------------------------
    # Чтение
    # ------------------------------------------------------------------
    def _retrieve(self, query: str, user_id: str, top_k: int) -> List[Document]:
        embedding = self.embedder.embed(query)
        filters = self._user_filter(user_id)
        logger.info(
            "[MEM] Поиск контекста | user=%s | top_k=%d | filter=%s | query=%r",
            user_id,
            top_k,
            filters,
            query[:80],
        )
        result = self._retriever.run(
            query_embedding=embedding,
            filters=filters,
            top_k=top_k,
        )
        documents = result.get("documents", []) or []
        logger.info("[MEM] Найдено документов: %d | user=%s", len(documents), user_id)
        return documents

    @staticmethod
    def _document_text(doc: Document) -> str:
        """
        Текст документа.

        Векторы, записанные обычным режимом бота (PineconeManager), хранят текст в
        ``meta['text']``, а не в ``content`` — учитываем оба варианта, чтобы память
        была общей для обоих режимов.
        """
        if doc.content:
            return doc.content
        meta = doc.meta or {}
        return str(meta.get("text") or meta.get("content") or "")

    def _build_context(self, documents: List[Document], user_id: str) -> str:
        lines: List[str] = []
        for i, doc in enumerate(documents, 1):
            text = self._document_text(doc).strip()
            if not text:
                continue
            score = doc.score if doc.score is not None else 0.0
            lines.append(f"[{i}] (cos={score:.3f}) {text}")
        context = "\n".join(lines)
        logger.info(
            "[MEM] Контекст собран | user=%s | документов=%d | символов=%d",
            user_id,
            len(lines),
            len(context),
        )
        return context

    # ------------------------------------------------------------------
    # Точка входа компонента
    # ------------------------------------------------------------------
    @component.output_types(context=str, stored=int, duplicates=int, retrieved=int, query=str)
    def run(
        self,
        query: str,
        user_id: str,
        facts: Optional[List[str]] = None,
        top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Сохранить факты пользователя и вернуть релевантный контекст.

        :param query: текст последнего сообщения пользователя.
        :param user_id: Telegram user_id — ключ изоляции.
        :param facts: факты, извлечённые из сообщения (для записи в память).
        :param top_k: сколько документов подтянуть в контекст.
        """
        uid = str(user_id)
        query = (query or "").strip()
        logger.info("=" * 70)
        logger.info("[MEM] Начало работы с памятью | user=%s | query_chars=%d", uid, len(query))

        stored = 0
        duplicates = 0
        for i, fact in enumerate(facts or [], 1):
            logger.info("[MEM] Факт %d/%d | user=%s | текст=%r", i, len(facts or []), uid, fact[:80])
            try:
                was_stored, similarity = self._store_fact(
                    fact,
                    uid,
                    meta={"source": SOURCE_TAG, "kind": "fact", "role": "user"},
                )
            except Exception as exc:  # уже залогировано внутри _store_fact
                logger.error(
                    "[MEM] ШАГ %d/%d ПРОВАЛЕН | user=%s | факт=%r | ошибка=%s",
                    i,
                    len(facts or []),
                    uid,
                    fact[:80],
                    exc,
                )
                raise
            if was_stored:
                stored += 1
                logger.info("[MEM] ШАГ %d/%d OK (записан) | cos=%.4f", i, len(facts or []), similarity)
            else:
                duplicates += 1
                logger.info("[MEM] ШАГ %d/%d OK (дубликат) | cos=%.4f", i, len(facts or []), similarity)

        logger.info(
            "[MEM] Запись завершена | user=%s | сохранено=%d | дубликатов=%d",
            uid,
            stored,
            duplicates,
        )

        context = ""
        retrieved = 0
        if query:
            try:
                documents = self._retrieve(query, uid, top_k or self.top_k)
            except Exception as exc:
                logger.exception("[MEM] ОШИБКА чтения из Pinecone | user=%s | query=%r", uid, query[:80])
                logger.error(
                    "[MEM] Контекст не получен, агент ответит без памяти. Причина: %s", exc
                )
                documents = []
            context = self._build_context(documents, uid)
            retrieved = len(documents)
        else:
            logger.info("[MEM] Пустой запрос — поиск контекста пропущен | user=%s", uid)

        logger.info(
            "[MEM] Итого | user=%s | сохранено=%d | дубликатов=%d | найдено=%d",
            uid,
            stored,
            duplicates,
            retrieved,
        )
        return {
            "context": context,
            "stored": stored,
            "duplicates": duplicates,
            "retrieved": retrieved,
            "query": query,
        }