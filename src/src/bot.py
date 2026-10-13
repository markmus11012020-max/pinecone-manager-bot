"""
Telegram bot with long-term memory powered by Pinecone + LLM (AITUNNEL).
Uses pyTelegramBotAPI.
"""

from __future__ import annotations

import logging
import sys
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

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("bot")

# ---------------------------------------------------------------------------
# Initialise
# ---------------------------------------------------------------------------
if not TELEGRAM_BOT_TOKEN or TELEGRAM_BOT_TOKEN.startswith("your_"):
    logger.error(
        "TELEGRAM_BOT_TOKEN is not set. Create a .env file with a valid token from @BotFather."
    )
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
        f"/reset — (заглушка) очистить мою память о тебе\n\n"
        f"Просто пиши текст — я отвечу с учётом контекста.",
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
