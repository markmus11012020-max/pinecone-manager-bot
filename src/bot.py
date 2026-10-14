"""
Telegram bot with long-term memory powered by Pinecone + LLM (AITUNNEL).
Uses pyTelegramBotAPI.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path so "config" and "src" imports work
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import telebot
from telebot import types

from config.settings import (
    TELEGRAM_BOT_TOKEN,
    SIMILARITY_THRESHOLD,
    TOP_K_MEMORIES,
    LLM_MODEL,
)
from src.pinecone_manager import PineconeManager
from src.llm_client import LLMClient
from src.single_instance import SingleInstanceError, acquire

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("bot")

# pyTelegramBotAPI attaches its own StreamHandler to the "TeleBot" logger at
# import time. Without this, every telebot message is printed twice (once in
# telebot's own format, once via root) and its level stays at ERROR, which
# hides polling diagnostics such as "Started polling".
telebot_logger = logging.getLogger("TeleBot")
telebot_logger.handlers.clear()
telebot_logger.propagate = True
telebot_logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Initialise
# ---------------------------------------------------------------------------
if not TELEGRAM_BOT_TOKEN or TELEGRAM_BOT_TOKEN.startswith("your_"):
    logger.error(
        "TELEGRAM_BOT_TOKEN is not set. Create a .env file with a valid token from @BotFather."
    )
    sys.exit(1)

# Only one process may poll this token. Fail loudly instead of letting both
# copies cancel each other with 409 Conflict.
try:
    _instance_lock = acquire(TELEGRAM_BOT_TOKEN)
except SingleInstanceError as e:
    logger.error("%s", e)
    sys.exit(1)

bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN, parse_mode="HTML")

try:
    memory = PineconeManager()
    logger.info("PineconeManager initialised successfully")
except Exception as e:
    logger.exception("Failed to initialise PineconeManager: %s", e)
    sys.exit(1)

try:
    llm = LLMClient()
    logger.info("LLMClient initialised | model=%s", LLM_MODEL)
except Exception as e:
    logger.exception("Failed to initialise LLMClient: %s", e)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Haystack mode (module hay/)
# ---------------------------------------------------------------------------
# The Haystack agent is built lazily: importing haystack + sentence-transformers
# takes several seconds and there is no reason to pay for it if the user never
# switches to /haystack. This keeps the standard mode exactly as fast as before.
HAYSTACK_HELP = (
    "🧠 <b>Режим Haystack включён.</b>\n\n"
    "Теперь я работаю как умный персональный помощник на фреймворке Haystack:\n"
    "• помню историю диалога и факты о тебе в Pinecone (изоляция по user_id);\n"
    "• отвечаю, опираясь на найденный по косинусному сходству контекст;\n"
    "• умею вызывать инструменты: анализ облигаций РФ и мультимодальные «собаки».\n\n"
    "Просто пиши как обычно. Вернуться в обычный режим: <code>/haystackoff</code>"
)

_HAYSTACK_ASSISTANT = None  # type: ignore[var-annotated]
_HAYSTACK_FAILED_REASON: str | None = None
_HAYSTACK_FAILED_AT: float = 0.0

# После неудачной инициализации повторяем попытку не чаще раза в 5 минут:
# так временная сетевая ошибка не блокирует режим навсегда, но и не превращает
# каждое сообщение в повторную тяжёлую загрузку моделей.
_HAYSTACK_RETRY_COOLDOWN = 300.0

# Выбранный режим хранится в оперативной памяти процесса бота.
# user_id -> True, если пользователь работает через Haystack.
USER_MODES: dict[int, bool] = {}


def is_haystack_mode(user_id: int) -> bool:
    """Текущий режим пользователя (по умолчанию — стандартный)."""
    return USER_MODES.get(int(user_id), False)


def set_haystack_mode(user_id: int, enabled: bool) -> None:
    """Переключить режим пользователя."""
    USER_MODES[int(user_id)] = bool(enabled)
    logger.info(
        "Mode switch | user_id=%s | haystack=%s | активных в режиме hay=%d",
        user_id,
        enabled,
        sum(1 for v in USER_MODES.values() if v),
    )


def get_haystack_assistant():
    """Ленивая инициализация агента Haystack (создаётся один раз)."""
    global _HAYSTACK_ASSISTANT, _HAYSTACK_FAILED_REASON, _HAYSTACK_FAILED_AT

    if _HAYSTACK_ASSISTANT is not None:
        return _HAYSTACK_ASSISTANT

    now = time.monotonic()
    if _HAYSTACK_FAILED_REASON:
        waited = now - _HAYSTACK_FAILED_AT
        if waited < _HAYSTACK_RETRY_COOLDOWN:
            logger.warning(
                "Агент Haystack недоступен (ошибка %d с. назад): %s",
                int(waited),
                _HAYSTACK_FAILED_REASON,
            )
            return None
        logger.info("Повторная попытка инициализации агента Haystack...")

    logger.info("Инициализация агента Haystack (первый запуск, это может занять время)...")
    try:
        from hay.agent import HaystackAssistant

        _HAYSTACK_ASSISTANT = HaystackAssistant()
        _HAYSTACK_FAILED_REASON = None
        _HAYSTACK_FAILED_AT = 0.0
        logger.info("Агент Haystack успешно инициализирован")
        return _HAYSTACK_ASSISTANT
    except Exception as e:
        _HAYSTACK_FAILED_REASON = str(e)
        _HAYSTACK_FAILED_AT = time.monotonic()
        logger.exception("Не удалось инициализировать агент Haystack: %s", e)
        return None


def _send_haystack_text(message: types.Message, text: str) -> None:
    """
    Отправить ответ агента.

    LLM иногда возвращает «<» или «&», которые Telegram в HTML-режиме считает
    тегами и отклоняет всю отправку — при такой ошибке уходим повторно без
    разметки, чтобы пользователь всё равно получил ответ.
    """
    try:
        bot.reply_to(message, text)
    except Exception as e:
        logger.warning("HTML-отправка не удалась (%s), отправляю без разметки", e)
        bot.reply_to(message, text, parse_mode=None)


def handle_haystack_reply(message: types.Message, text: str) -> None:
    """Обработать текстовое сообщение в режиме Haystack."""
    uid = str(message.from_user.id)
    name = message.from_user.first_name or "пользователь"
    logger.info("Haystack mode | text from %s (%s): %s", uid, name, text[:80])

    try:
        bot.send_chat_action(message.chat.id, "typing")
    except Exception:
        pass

    assistant = get_haystack_assistant()
    if assistant is None:
        _send_haystack_text(
            message,
            "😔 Не удалось запустить режим Haystack.\n"
            f"<code>{_HAYSTACK_FAILED_REASON or 'неизвестная ошибка'}</code>\n\n"
            "Подробности — в консоли бота. Вернуться в обычный режим: /haystackoff",
        )
        return

    try:
        result = assistant.run(
            user_id=uid,
            query=text,
            user_name=name,
            facts=extract_facts(text),
        )
    except Exception as e:
        logger.exception("Ошибка пайплайна Haystack для user=%s: %s", uid, e)
        _send_haystack_text(
            message, f"Что-то пошло не так в пайплайне Haystack.\n<code>{e}</code>"
        )
        return

    reply = result.get("reply") or ""
    image_url = result.get("image_url") or ""

    # Инструмент «собаки» возвращает URL фотографии — отправляем её как фото.
    if image_url:
        try:
            bot.send_photo(message.chat.id, image_url, caption=reply[:1024], parse_mode="HTML")
        except Exception as e:
            logger.warning("Не удалось отправить фото %s: %s", image_url, e)
            _send_haystack_text(message, reply)
    else:
        _send_haystack_text(message, reply)

    stored = result.get("stored", 0)
    if stored:
        logger.info("Haystack | сохранено новых фактов: %d | user=%s", stored, uid)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def extract_facts(text: str) -> list[str]:
    """
    Very simple heuristic: treat each non-empty line / sentence as a potential fact.
    In a production system this would be replaced by an LLM-based fact extractor.
    """
    candidates = []
    for part in text.replace("\n", ". ").split("."):
        part = part.strip()
        if len(part) >= 15:
            candidates.append(part)
    return candidates or ([text.strip()] if text.strip() else [])


def generate_llm_reply(user_id: str, message_text: str, user_name: str) -> str:
    """Retrieve memory context and ask the LLM for a natural reply."""
    context = memory.get_relevant_context(
        message_text, user_id=str(user_id), top_k=TOP_K_MEMORIES
    )
    try:
        return llm.reply_with_memory(
            user_message=message_text,
            memory_context=context,
            user_name=user_name,
        )
    except Exception as e:
        logger.exception("LLM reply failed: %s", e)
        # Fallback without LLM
        if context:
            return (
                f"Я вспомнил следующее:\n<pre>{context}</pre>\n\n"
                f"(LLM временно недоступен: {e})"
            )
        return f"Извини, сейчас не могу сгенерировать ответ (ошибка LLM). Твоё сообщение: {message_text}"


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------
@bot.message_handler(commands=["start", "help"])
def cmd_start(message: types.Message) -> None:
    user = message.from_user
    uid = user.id
    name = user.first_name or "друг"
    logger.info("/start from user_id=%s (%s)", uid, name)

    bot.reply_to(
        message,
        f"Привет, <b>{name}</b>! 👋\n\n"
        f"Я бот с долговременной памятью (Pinecone) и отвечаю через LLM "
        f"(<code>{LLM_MODEL}</code> через AITUNNEL).\n\n"
        f"Пиши мне любые факты о себе или важные события — я запомню их "
        f"(если они достаточно отличаются от уже сохранённых) и буду учитывать "
        f"в ответах.\n\n"
        f"<b>Команды:</b>\n"
        f"/start — это сообщение\n"
        f"/memory — показать сохранённые воспоминания\n"
        f"/reset — (заглушка) очистить мою память о тебе\n"
        f"/haystack — умный помощник на фреймворке Haystack (с инструментами и памятью)\n"
        f"/haystackoff — вернуться в обычный режим\n\n"
        f"Просто пиши текст — я отвечу с учётом контекста.",
    )


# ---------------------------------------------------------------------------
# Переключение режимов: обычный <-> Haystack
# ---------------------------------------------------------------------------
@bot.message_handler(commands=["haystack"])
def cmd_haystack_on(message: types.Message) -> None:
    uid = message.from_user.id
    name = message.from_user.first_name or "друг"
    logger.info("/haystack from user_id=%s (%s)", uid, name)

    if is_haystack_mode(uid):
        bot.reply_to(message, "Режим Haystack уже включён 🙂\nОтключить: /haystackoff")
        return

    set_haystack_mode(uid, True)
    logger.info("Пользователь %s переключён в режим Haystack", uid)
    bot.reply_to(message, HAYSTACK_HELP)


@bot.message_handler(commands=["haystackoff", "haystack_off", "standard"])
def cmd_haystack_off(message: types.Message) -> None:
    uid = message.from_user.id
    logger.info("/haystackoff from user_id=%s", uid)

    if not is_haystack_mode(uid):
        bot.reply_to(message, "Ты и так в обычном режиме 🙂\nВключить Haystack: /haystack")
        return

    set_haystack_mode(uid, False)
    logger.info("Пользователь %s вернулся в обычный режим", uid)
    bot.reply_to(
        message,
        "✅ Вернулся в обычный режим.\n\n"
        "Пиши текст — отвечу через LLM с долговременной памятью в Pinecone. "
        "Снова включить умного помощника: /haystack",
    )


@bot.message_handler(commands=["haystatus"])
def cmd_haystack_status(message: types.Message) -> None:
    uid = message.from_user.id
    mode = "Haystack" if is_haystack_mode(uid) else "обычный"
    assistant_state = "не создан" if _HAYSTACK_ASSISTANT is None else "готов"
    bot.reply_to(
        message,
        f"<b>Текущий режим:</b> {mode}\n"
        f"<b>Агент Haystack:</b> {assistant_state}\n"
        f"<b>Модель:</b> <code>{LLM_MODEL}</code>\n\n"
        f"Переключить: /haystack — включить, /haystackoff — выключить.",
    )


@bot.message_handler(commands=["memory"])
def cmd_memory(message: types.Message) -> None:
    uid = str(message.from_user.id)
    matches = memory.query_by_text("что ты помнишь", user_id=uid, top_k=10)
    if not matches:
        bot.reply_to(message, "У меня пока нет сохранённых воспоминаний о тебе.")
        return

    lines = []
    for i, m in enumerate(matches, 1):
        text = (m.get("metadata") or {}).get("text", "—")
        score = m.get("score", 0.0)
        lines.append(f"{i}. <code>{text}</code> (sim={score:.3f})")
    bot.reply_to(message, "Мои воспоминания о тебе:\n\n" + "\n".join(lines))


@bot.message_handler(commands=["reset"])
def cmd_reset(message: types.Message) -> None:
    uid = str(message.from_user.id)
    memory.delete_user_memories(uid)
    bot.reply_to(
        message,
        "Команда /reset пока является заглушкой (удаление по фильтру зависит от тарифа Pinecone). "
        "В реальном проекте здесь будет очистка всех векторов пользователя.",
    )


@bot.message_handler(content_types=["text"])
def handle_text(message: types.Message) -> None:
    if not message.text or message.text.startswith("/"):
        return

    uid = str(message.from_user.id)
    text = message.text.strip()
    name = message.from_user.first_name or "пользователь"

    # Haystack mode has its own pipeline (memory + tools) — route it there.
    if is_haystack_mode(message.from_user.id):
        handle_haystack_reply(message, text)
        return

    logger.info("Text from %s: %s", uid, text[:80])

    # Show "typing..." while we work
    try:
        bot.send_chat_action(message.chat.id, "typing")
    except Exception:
        pass

    # 1. Extract potential facts and store them (with duplicate filtering)
    facts = extract_facts(text)
    stored_count = 0
    for fact in facts:
        was_stored, vid, sim = memory.store_if_new(
            text=fact,
            user_id=uid,
            metadata={"source": "user_message", "chat_id": message.chat.id},
        )
        if was_stored:
            stored_count += 1
            logger.info("Stored new fact (sim=%.3f): %s", sim, fact[:60])

    # 2. Generate natural reply via LLM + memory context
    reply = generate_llm_reply(uid, text, name)

    # Optional short note about memory write
    if stored_count:
        reply += f"\n\n✅ Сохранено новых фактов: {stored_count}"

    bot.reply_to(message, reply)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    logger.info("Starting Telegram bot polling...")
    bot.infinity_polling(timeout=60, long_polling_timeout=60)


if __name__ == "__main__":
    main()
