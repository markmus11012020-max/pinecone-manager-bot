"""
Кастомный компонент Haystack: Docling -> смысловые чанки.

Компонент принимает локальный путь к файлу (PDF, DOCX, PPTX, XLSX, HTML, MD,
AsciiDoc и т.п.), полностью разбирает его через Docling, режет на смысловые
чанки ``HybridChunker`` и возвращает список Haystack ``Document``.

В метаданные каждого чанка строго попадают:
    * ``user_id``     — Telegram ID владельца (изоляция контекста);
    * ``file_name``   — исходное имя файла;
    * ``page_number`` — номер страницы (0, если Docling его не отдал);
    * ``chunk_index`` — порядковый номер чанка внутри файла;
    * ``section``     — заголовки/раздел, если Docling их определил;
    * ``source``      — метка версии модуля.
"""

from __future__ import annotations

import hashlib
import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from haystack import Document, component

from config.settings import EMBEDDING_MODEL

logger = logging.getLogger("hay_v2_bot.docling")

SOURCE_TAG = "hay_v2_bot"

# Pinecone принимает в метаданных только простые типы.
_SIMPLE_META_TYPES = (str, int, float, bool)


def _sanitize_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
    """Оставить только типы, которые Pinecone принимает в metadata."""
    clean: Dict[str, Any] = {}
    for key, value in (meta or {}).items():
        if value is None:
            continue
        if isinstance(value, _SIMPLE_META_TYPES):
            clean[key] = value
        elif isinstance(value, (list, tuple)) and all(isinstance(item, str) for item in value):
            clean[key] = list(value)
        else:
            logger.debug(
                "[DOCLING] Метаданное '%s'=%r пропущено: тип %s не поддерживается Pinecone",
                key,
                value,
                type(value).__name__,
            )
    return clean


def _load_huggingface_tokenizer(model_name: str):
    """
    Загрузить токенизатор для HybridChunker.

    ``EMBEDDING_MODEL`` может быть коротким алиасом (``all-MiniLM-L6-v2``),
    который понимает sentence-transformers, но не ``transformers``. Поэтому
    пробуем сначала исходное имя, затем официальный repo_id.
    """
    from transformers import AutoTokenizer

    candidates = [model_name]
    if "/" not in model_name:
        candidates.append(f"sentence-transformers/{model_name}")

    last_error: Optional[Exception] = None
    for candidate in candidates:
        try:
            logger.info("[DOCLING] Загрузка токенизатора: %s", candidate)
            return AutoTokenizer.from_pretrained(candidate)
        except Exception as exc:  # noqa: BLE001 - нужен fallback на второй вариант
            last_error = exc
            logger.warning(
                "[DOCLING] Токенизатор '%s' недоступен: %s",
                candidate,
                exc,
            )

    raise RuntimeError(
        f"Не удалось загрузить токенизатор для EMBEDDING_MODEL={model_name!r}: {last_error}"
    )


@component
class DoclingDocumentConverter:
    """
    Haystack-компонент конвертации файла в чанки.

    ``@component.input_types``:
        * ``file_path`` (str) — локальный путь к скачанному файлу;
        * ``user_id``   (str) — Telegram ID пользователя;
        * ``file_name`` (str) — исходное имя файла (необязательно).

    ``@component.output_types``:
        * ``documents`` (List[Document]) — чанки с метаданными.
    """

    def __init__(
        self,
        embedding_model: Optional[str] = None,
        tokenizer: Optional[Any] = None,
        max_tokens: int = 512,
        merge_peers: bool = True,
        converter: Optional[Any] = None,
        chunker: Optional[Any] = None,
    ) -> None:
        self.embedding_model = embedding_model or EMBEDDING_MODEL
        self.max_tokens = max(int(max_tokens), 64)
        self.merge_peers = bool(merge_peers)

        self._tokenizer = tokenizer
        self._converter = converter
        self._chunker = chunker
        self._init_lock = threading.Lock()

        logger.info(
            "[DOCLING] Компонент создан | model=%s | max_tokens=%d | merge_peers=%s",
            self.embedding_model,
            self.max_tokens,
            self.merge_peers,
        )

    # ------------------------------------------------------------------
    # Ленивая инициализация тяжёлых объектов Docling
    # ------------------------------------------------------------------
    def _ensure_components(self) -> None:
        if self._converter is not None and self._chunker is not None:
            return

        with self._init_lock:
            if self._converter is not None and self._chunker is not None:
                return

            logger.info("[DOCLING] Этап 0/4: инициализация Docling...")
            from docling.chunking import HybridChunker
            from docling.document_converter import DocumentConverter
            from docling_core.transforms.chunker.tokenizer.huggingface import (
                HuggingFaceTokenizer,
            )

            if self._converter is None:
                self._converter = DocumentConverter()
                logger.info("[DOCLING] DocumentConverter инициализирован")

            if self._chunker is None:
                tokenizer = self._tokenizer
                if tokenizer is None:
                    tokenizer = _load_huggingface_tokenizer(self.embedding_model)

                if not isinstance(tokenizer, HuggingFaceTokenizer):
                    tokenizer = HuggingFaceTokenizer(
                        tokenizer=tokenizer,
                        max_tokens=self.max_tokens,
                    )

                self._chunker = HybridChunker(
                    tokenizer=tokenizer,
                    max_tokens=self.max_tokens,
                    merge_peers=self.merge_peers,
                )
                logger.info(
                    "[DOCLING] HybridChunker инициализирован | max_tokens=%d",
                    self.max_tokens,
                )

    # ------------------------------------------------------------------
    # Извлечение метаданных чанка
    # ------------------------------------------------------------------
    @staticmethod
    def _chunk_meta_value(chunk: Any, name: str, default: Any = None) -> Any:
        """Достать поле из ``chunk.meta`` (pydantic-модель или dict)."""
        meta = getattr(chunk, "meta", None)
        if meta is None:
            return default
        if isinstance(meta, dict):
            return meta.get(name, default)
        return getattr(meta, name, default)

    @classmethod
    def _page_number(cls, chunk: Any) -> int:
        """
        Номер страницы чанка.

        В Docling провенанс лежит в ``chunk.meta.doc_items[*].prov[*].page_no``.
        Для форматов без страниц (Markdown, DOCX в некоторых бэкендах) список
        пуст — тогда возвращаем 0.
        """
        pages: List[int] = []
        items = cls._chunk_meta_value(chunk, "doc_items", []) or []
        for item in items:
            provenance = getattr(item, "prov", None) or []
            for prov in provenance:
                page_no = getattr(prov, "page_no", None)
                if page_no is not None:
                    try:
                        pages.append(int(page_no))
                    except (TypeError, ValueError):
                        pass
        return min(pages) if pages else 0

    @classmethod
    def _section(cls, chunk: Any) -> str:
        """Заголовки чанка одной строкой (для удобства поиска и логов)."""
        headings = cls._chunk_meta_value(chunk, "headings", []) or []
        if isinstance(headings, str):
            return headings[:512]
        return " > ".join(str(item) for item in headings)[:512]

    @staticmethod
    def _document_id(user_id: str, file_name: str, page: int, index: int, text: str) -> str:
        """Детерминированный id: повторная загрузка того же файла обновит чанки."""
        digest = hashlib.sha1(
            f"{user_id}|{file_name}|{page}|{index}|{text}".encode("utf-8")
        ).hexdigest()
        return digest

    # ------------------------------------------------------------------
    # Точка входа компонента
    # ------------------------------------------------------------------
    @component.output_types(documents=List[Document])
    def run(
        self,
        file_path: str,
        user_id: str,
        file_name: str = "",
    ) -> Dict[str, Any]:
        """
        Разобрать файл и вернуть чанки с метаданными.

        :param file_path: локальный путь к файлу, скачанному из Telegram.
        :param user_id: Telegram ID — ключ строгой изоляции контекста.
        :param file_name: исходное имя файла (если пусто — берётся из пути).
        """
        uid = str(user_id or "").strip()
        if not uid:
            raise ValueError("user_id обязателен: без него невозможна изоляция контекста")

        path = Path(file_path).expanduser()
        name = (file_name or path.name or "document").strip()
        logger.info("=" * 78)
        logger.info(
            "[DOCLING] Старт | user=%s | file=%s | path=%s",
            uid,
            name,
            path,
        )

        if not path.exists() or not path.is_file():
            logger.error("[DOCLING] Файл не найден | path=%s", path)
            raise FileNotFoundError(f"Файл не найден: {path}")

        size = path.stat().st_size
        logger.info("[DOCLING] Размер файла: %.2f МБ", size / (1024 * 1024))

        self._ensure_components()

        # ---- Этап 1: конвертация Docling ----
        logger.info("[DOCLING] Этап 1/4: конвертация документа через Docling...")
        try:
            conversion = self._converter.convert(str(path))
        except Exception as exc:
            logger.exception(
                "[DOCLING] ОШИБКА конвертации | user=%s | file=%s | path=%s | причина=%s",
                uid,
                name,
                path,
                exc,
            )
            raise RuntimeError(f"Docling не смог разобрать файл '{name}': {exc}") from exc

        document = getattr(conversion, "document", None)
        status = getattr(conversion, "status", "unknown")
        if document is None:
            logger.error(
                "[DOCLING] Docling вернул пустой результат | user=%s | file=%s | status=%s",
                uid,
                name,
                status,
            )
            raise RuntimeError(f"Docling не вернул документ для '{name}' (status={status})")

        pages = getattr(document, "pages", None) or {}
        try:
            pages_count = len(pages)
        except TypeError:
            pages_count = 0
        logger.info(
            "[DOCLING] Этап 1/4 завершён | status=%s | страниц=%s",
            status,
            pages_count or "нет (формат без страниц)",
        )

        # ---- Этап 2: чанкинг ----
        logger.info("[DOCLING] Этап 2/4: разбиение на смысловые чанки...")
        try:
            raw_chunks = list(self._chunker.chunk(dl_doc=document))
        except Exception as exc:
            logger.exception(
                "[DOCLING] ОШИБКА чанкинга | user=%s | file=%s | причина=%s",
                uid,
                name,
                exc,
            )
            raise RuntimeError(f"Не удалось разбить '{name}' на чанки: {exc}") from exc

        logger.info("[DOCLING] Этап 2/4 завершён | чанков=%d", len(raw_chunks))
        if not raw_chunks:
            logger.warning(
                "[DOCLING] В документе не найдено текста | user=%s | file=%s",
                uid,
                name,
            )
            return {"documents": []}

        # ---- Этап 3: Haystack Document + метаданные ----
        logger.info("[DOCLING] Этап 3/4: формирование Haystack Document и метаданных...")
        documents: List[Document] = []
        seen_texts: set[str] = set()

        for index, chunk in enumerate(raw_chunks, 1):
            raw_text = (getattr(chunk, "text", "") or "").strip()
            if not raw_text:
                logger.warning("[DOCLING] Чанк %d пуст — пропущен", index)
                continue

            try:
                text = (self._chunker.contextualize(chunk) or raw_text).strip()
            except Exception as exc:  # noqa: BLE001 - contextualize не критичен
                logger.warning(
                    "[DOCLING] contextualize не удался для чанка %d (%s) — берём raw text",
                    index,
                    exc,
                )
                text = raw_text

            if not text:
                logger.warning("[DOCLING] Чанк %d пуст после contextualize — пропущен", index)
                continue

            text_key = hashlib.sha1(text.encode("utf-8")).hexdigest()
            if text_key in seen_texts:
                logger.info("[DOCLING] Чанк %d — точный дубликат текста, пропущен", index)
                continue
            seen_texts.add(text_key)

            page = self._page_number(chunk)
            section = self._section(chunk)
            meta = _sanitize_meta(
                {
                    "user_id": uid,
                    "file_name": name,
                    "page_number": page,
                    "chunk_index": index,
                    "section": section,
                    "source": SOURCE_TAG,
                    "embedding_model": self.embedding_model,
                }
            )
            doc_id = self._document_id(uid, name, page, index, text)

            documents.append(
                Document(
                    id=doc_id,
                    content=text,
                    meta=meta,
                )
            )
            logger.info(
                "[DOCLING] Чанк %d | page=%d | символов=%d | section=%r | id=%s",
                index,
                page,
                len(text),
                section[:80],
                doc_id,
            )

        logger.info(
            "[DOCLING] Этап 3/4 завершён | user=%s | file=%s | документов=%d (из %d чанков)",
            uid,
            name,
            len(documents),
            len(raw_chunks),
        )

        logger.info(
            "[DOCLING] Этап 4/4 готов: передаю %d чанков в пайплайн индексации",
            len(documents),
        )
        logger.info("=" * 78)
        return {"documents": documents}


__all__ = ["DoclingDocumentConverter", "SOURCE_TAG"]
