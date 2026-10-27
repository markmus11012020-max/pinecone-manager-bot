"""
Обработчики Telegram для hay_v2_bot.

Роутинг:
    * команды ``/start``, ``/help``, ``/status``;
    * текстовые сообщения -> генеративный RAG-пайплайн;
    * документы (PDF, DOCX, PPTX, XLSX, HTML, MD и др.) -> Docling -> Pinecone.

Сервис ``AssistantService`` держит оба пайплайна и общую модель эмбеддингов,
а также короткую историю диалога каждого пользователя в оперативной памяти.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import telebot
from telebot import types

from config.settings import LLM_MODEL, TOP_K_MEMORIES
from hay_v2_bot.pipelines.generation import GenerationPipeline
from hay_v2_bot.pipelines.indexing import IndexingPipeline

logger = logging.getLogger("hay_v2_bot.handlers")

# Telegram Bot API отдаёт ботам файлы до 20 МБ через getFile.
MAX_FILE_SIZE = 20 * 1024 * 1024

ALLOWED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".doc",
    ".pptx",
    ".xlsx",
    ".html",
    ".htm",
    ".md",
    ".adoc",
    ".asciidoc",
    ".ascii",
    ".csv",
}

ALLOWED_FORMATS_TEXT = "PDF, DOCX, PPTX, XLSX, HTML, Markdown, AsciiDoc, CSV"

MAX_TELEGRAM_MESSAGE = 4000
MAX_HISTORY_TURNS = 4
MAX_HISTORY_CHARS = 2500


class AssistantService:
    """Пайплайны + краткая история диалога пользователей."""

    def __init__(
        self,
        *,
        indexer: IndexingPipeline,
        generator: GenerationPipeline,
    ) -> None:
        self.indexer = indexer
        self.generator = generator
        self._histories: Dict[str, List[Tuple[str, str]]] = {}
        self._history_lock = threading.Lock()
        # Docling и Pinecone-запись не рассчитаны на параллельный вызов из
        # нескольких потоков pyTelegramBotAPI — индексацию выполняем по очереди.
        self._index_lock = threading.Lock()

    # ------------------------------------------------------------------
    @classmethod
    def build(cls) -> "AssistantService":
        """Собрать общий PineconeDocumentStore, эмбеддер и оба пайплайна."""
        from hay_v2_bot.components.embeddings import SentenceTransformerEmbedder
        from hay_v2_bot.pipelines.indexing import build_document_store

        logger.info("[BOT] Сборка сервиса: Pinecone + эмбеддинги + пайплайны...")
        embedder = SentenceTransformerEmbedder()
        document_store = build_document_store()
        indexer = IndexingPipeline(
            document_store=document_store,
            embedder=embedder,
            max_tokens=512,
        )
        generator = GenerationPipeline(
            document_store=document_store,
            embedder=embedder,
            top_k=TOP_K_MEMORIES,
        )
        logger.info("[BOT] Сервис готов: оба пайплайна используют одну модель эмбеддингов")
        return cls(indexer=indexer, generator=generator)

    # ------------------------------------------------------------------
    # История диалога (только ОЗУ, до перезапуска)
    # ------------------------------------------------------------------
    def _history_text(self, user_id: str) -> str:
        with self._history_lock:
            turns = list(self._histories.get(user_id, []))
        if not turns:
            return ""

        lines: List[str] = []
        for user_text, bot_text in turns[-MAX_HISTORY_TURNS:]:
            lines.append(f"Пользователь: {user_text[:600]}")
            lines.append(f"Помощник: {bot_text[:600]}")
        history = "\n".join(lines)
        return history[-MAX_HISTORY_CHARS:]

    def _remember(self, user_id: str, user_text: str, bot_text: str) -> None:
        with self._history_lock:
            turns = self._histories.setdefault(user_id, [])
            turns.append((user_text, bot_text))
            # Не даём истории расти бесконечно.
            if len(turns) > MAX_HISTORY_TURNS * 2:
                del turns[: len(turns) - MAX_HISTORY_TURNS]

    # ------------------------------------------------------------------
    def answer(self, user_id: str, query: str) -> Dict[str, Any]:
        uid = str(user_id)
        history = self._history_text(uid)
        result = self.generator.run(user_id=uid, query=query, history=history)
        reply = (result.get("reply") or "").strip()
        if not reply:
            reply = "Не удалось сформулировать ответ. Попробуй переформулировать вопрос."
        self._remember(uid, query, reply)
        return result

    def index_file(
        self,
        *,
        user_id: str,
        file_path: str,
        file_name: str,
    ) -> Dict[str, Any]:
        with self._index_lock:
            return self.indexer.run(
                file_path=file_path,
                user_id=str(user_id),
                file_name=file_name,
            )

    def summarize(self, documents: List[Any], file_name: str) -> str:
        return self.generator.summarize(documents, file_name=file_name)


# ---------------------------------------------------------------------------
# Отправка сообщений
# ---------------------------------------------------------------------------
def _safe_filename(name: str) -> str:
    """Оставить от имени файла только безопасные символы."""
    base = Path(name or "document").name
    base = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", base).strip(" .")
    return base or "document"


def download_telegram_file(
    bot: telebot.TeleBot,
    file_path: str,
    local_path: str,
) -> int:
    """
    Скачать файл Telegram в ``local_path`` и вернуть его размер в байтах.

    Начиная с pyTelegramBotAPI 4.36 ``TeleBot.download_file`` принимает только
    ``file_path`` и возвращает содержимое в виде ``bytes`` — файл нужно писать
    на диск самостоятельно. Старый вариант с каталогом назначения удалён из API.
    """
    payload = bot.download_file(file_path)
    if not isinstance(payload, (bytes, bytearray)):
        raise RuntimeError(
            "Telegram не вернул содержимое файла (пустой ответ Bot API)"
        )
    with open(local_path, "wb") as handle:
        handle.write(payload)
    return len(payload)


def send_text(bot: telebot.TeleBot, chat_id: int, text: str) -> None:
    """
    Отправить текст, разбивая длинные ответы.

    LLM иногда возвращает ``<`` или ``&``, которые Telegram в HTML-режиме
    считает тегами и отклоняет отправку — при ошибке повторяем без разметки.
    """
    text = (text or "").strip() or "Пустой ответ."
    chunks: List[str] = []
    while text:
        if len(text) <= MAX_TELEGRAM_MESSAGE:
            chunks.append(text)
            break
        split_at = text.rfind("\n", 0, MAX_TELEGRAM_MESSAGE)
        if split_at < MAX_TELEGRAM_MESSAGE // 2:
            split_at = text.rfind(" ", 0, MAX_TELEGRAM_MESSAGE)
        if split_at < MAX_TELEGRAM_MESSAGE // 2:
            split_at = MAX_TELEGRAM_MESSAGE
        chunks.append(text[:split_at].rstrip())
        text = text[split_at:].lstrip()

    for chunk in chunks:
        try:
            bot.send_message(chat_id, chunk)
        except Exception as exc:
            logger.warning("[BOT] HTML-отправка не удалась (%s), отправляю без разметки", exc)
            bot.send_message(chat_id, chunk, parse_mode=None)


def _typing(bot: telebot.TeleBot, chat_id: int) -> None:
    try:
        bot.send_chat_action(chat_id, "typing")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Регистрация обработчиков
# ---------------------------------------------------------------------------
def register_handlers(bot: telebot.TeleBot, assistant: AssistantService) -> None:
    """Зарегистрировать все команды и обработчики контента."""

    @bot.message_handler(commands=["start", "help"])
    def cmd_start(message: types.Message) -> None:
        user = message.from_user
        name = user.first_name or "друг"
        logger.info("/start | user=%s (%s)", user.id, name)
        send_text(
            bot,
            message.chat.id,
            f"Привет, <b>{name}</b>! 👋\n\n"
            "Я — вторая версия бота: умный помощник на <b>Haystack</b> с разбором "
            "документов через <b>Docling</b> и памятью в <b>Pinecone</b>.\n\n"
            "<b>Что умею:</b>\n"
            "• принимать файлы (PDF, DOCX, PPTX, XLSX, HTML, Markdown);\n"
            "• разбирать их на смысловые фрагменты и сохранять в Pinecone;\n"
            "• отвечать на вопросы строго по <i>твоим</i> документам "
            "(изоляция по user_id);\n"
            "• присылать короткое резюме файла в одном предложении.\n\n"
            "<b>Как пользоваться:</b>\n"
            "1. Пришли документ — я скажу, когда он изучен.\n"
            "2. Задай любой вопрос по содержимому.\n\n"
            f"<b>Модель:</b> <code>{LLM_MODEL}</code>\n"
            "Команды: /help, /status",
        )

    @bot.message_handler(commands=["status"])
    def cmd_status(message: types.Message) -> None:
        uid = str(message.from_user.id)
        total = "—"
        try:
            total = str(assistant.indexer.document_store.count_documents())
        except Exception as exc:  # noqa: BLE001 - статус не должен падать
            logger.warning("[BOT] Не удалось получить count_documents: %s", exc)

        send_text(
            bot,
            message.chat.id,
            "<b>Статус hay_v2_bot</b>\n"
            f"Пользователь: <code>{uid}</code>\n"
            f"Модель: <code>{LLM_MODEL}</code>\n"
            f"Фрагментов в индексе Pinecone: <b>{total}</b>\n"
            f"Глубина поиска: <b>{TOP_K_MEMORIES}</b> фрагментов\n\n"
            "Пришли PDF/DOCX — я разберу его через Docling.",
        )

    @bot.message_handler(content_types=["document"])
    def handle_document(message: types.Message) -> None:
        document = message.document
        uid = str(message.from_user.id)
        name = document.file_name or f"document_{document.file_unique_id}"
        extension = Path(name).suffix.lower()

        logger.info(
            "[BOT] Документ | user=%s | file=%s | ext=%s | size=%s | mime=%s",
            uid,
            name,
            extension,
            document.file_size,
            document.mime_type,
        )

        if extension not in ALLOWED_EXTENSIONS:
            send_text(
                bot,
                message.chat.id,
                f"⚠️ Формат <code>{extension or 'без расширения'}</code> не поддерживается.\n"
                f"Пришли один из: {ALLOWED_FORMATS_TEXT}.",
            )
            return

        if document.file_size and document.file_size > MAX_FILE_SIZE:
            send_text(
                bot,
                message.chat.id,
                "⚠️ Файл больше 20 МБ — Telegram не отдаёт такие файлы ботам.\n"
                "Сожми документ или разбей его на части.",
            )
            return

        send_text(
            bot,
            message.chat.id,
            "Файл получен. Запускаю анализ и сохранение. Это может занять немного времени...",
        )
        _typing(bot, message.chat.id)

        temp_dir = tempfile.mkdtemp(prefix="hay_v2_")
        safe_name = _safe_filename(name)
        local_path = os.path.join(temp_dir, safe_name)

        try:
            logger.info("[BOT] Скачивание файла | user=%s | file_id=%s", uid, document.file_id)
            file_info = bot.get_file(document.file_id)
            download_telegram_file(bot, file_info.file_path, local_path)
            downloaded_size = os.path.getsize(local_path)
            logger.info(
                "[BOT] Файл скачан | user=%s | path=%s | bytes=%d",
                uid,
                local_path,
                downloaded_size,
            )

            _typing(bot, message.chat.id)
            result = assistant.index_file(
                user_id=uid,
                file_path=local_path,
                file_name=name,
            )
            documents = result.get("documents", []) or []
            written = result.get("documents_written", 0)

            if not documents:
                send_text(
                    bot,
                    message.chat.id,
                    "⚠️ В файле не найден текст для сохранения.\n\n"
                    "Я попробовал три способа: обычный разбор, чтение текстового слоя "
                    "и полностраничный OCR — текста нет ни в одном.\n\n"
                    "Скорее всего, это скан низкого качества или пустой документ. "
                    "Попробуй прислать PDF с текстовым слоем либо скан лучшего качества.",
                )
                return

            _typing(bot, message.chat.id)
            summary = assistant.summarize(documents, name)

            logger.info(
                "[BOT] Файл обработан | user=%s | file=%s | чанков=%d | записано=%d",
                uid,
                name,
                len(documents),
                written,
            )
            send_text(
                bot,
                message.chat.id,
                "Готово. Я изучил этот файл, теперь можем его обсудить.\n\n"
                f"{summary}",
            )
        except Exception as exc:  # noqa: BLE001 - пользователю нужен внятный ответ
            logger.exception(
                "[BOT] ОШИБКА обработки файла | user=%s | file=%s | path=%s | причина=%s",
                uid,
                name,
                local_path,
                exc,
            )
            send_text(
                bot,
                message.chat.id,
                "😔 Не удалось обработать файл.\n"
                f"<code>{exc}</code>\n\n"
                "Подробности — в консоли бота.",
            )
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
            logger.info("[BOT] Временная папка удалена | path=%s", temp_dir)

    @bot.message_handler(content_types=["text"])
    def handle_text(message: types.Message) -> None:
        text = (message.text or "").strip()
        if not text or text.startswith("/"):
            return

        uid = str(message.from_user.id)
        name = message.from_user.first_name or "пользователь"
        logger.info("[BOT] Текст | user=%s (%s) | %r", uid, name, text[:160])
        _typing(bot, message.chat.id)

        try:
            result = assistant.answer(uid, text)
        except Exception as exc:  # noqa: BLE001 - ошибка должна дойти до пользователя
            logger.exception("[BOT] ОШИБКА RAG-ответа | user=%s | запрос=%r | причина=%s", uid, text[:160], exc)
            send_text(
                bot,
                message.chat.id,
                "😔 Не удалось получить ответ.\n"
                f"<code>{exc}</code>\n\n"
                "Подробности — в консоли бота.",
            )
            return

        documents = result.get("documents", []) or []
        if documents:
            logger.info(
                "[BOT] Ответ отправляется | user=%s | найдено фрагментов=%d",
                uid,
                len(documents),
            )
        else:
            logger.info("[BOT] Ответ без документов | user=%s", uid)

        send_text(bot, message.chat.id, result.get("reply") or "Пустой ответ.")

    logger.info(
        "[BOT] Обработчики зарегистрированы | команды=/start,/help,/status | "
        "контент=text,document | форматы=%s",
        ALLOWED_FORMATS_TEXT,
    )


__all__ = [
    "AssistantService",
    "download_telegram_file",
    "register_handlers",
    "send_text",
]
