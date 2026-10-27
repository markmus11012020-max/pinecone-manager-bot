"""
Запуск pyTelegramBotAPI для hay_v2_bot.

Используется тот же single-instance mutex, что и у основного бота: Telegram
разрешает только один getUpdates на токен, поэтому два процесса не будут
бесконечно получать 409 Conflict.
"""

from __future__ import annotations

import logging
import sys
from typing import Optional

import telebot

from config.settings import TELEGRAM_BOT_TOKEN
from hay_v2_bot.bot.handlers import AssistantService, register_handlers
from src.single_instance import SingleInstanceError, acquire

logger = logging.getLogger("hay_v2_bot.telegram")

# Мьютекс должен жить всё время работы процесса.
_INSTANCE_LOCK = None


def create_bot(token: Optional[str] = None) -> telebot.TeleBot:
    """Создать Telegram-бота и привести логи telebot к общему формату."""
    token = (token or TELEGRAM_BOT_TOKEN or "").strip()
    if not token or token.startswith("your_"):
        raise ValueError(
            "TELEGRAM_BOT_TOKEN не задан. Проверьте .env в корне проекта."
        )

    # pyTelegramBotAPI добавляет свой StreamHandler при импорте; без этого
    # сообщения дублируются и уровень остаётся ERROR.
    telebot_logger = logging.getLogger("TeleBot")
    telebot_logger.handlers.clear()
    telebot_logger.propagate = True
    telebot_logger.setLevel(logging.INFO)

    logger.info("[BOT] Создание TeleBot | parse_mode=HTML")
    return telebot.TeleBot(token, parse_mode="HTML")


def run_bot(assistant: AssistantService, token: Optional[str] = None) -> None:
    """Запустить polling (блокирующий вызов)."""
    global _INSTANCE_LOCK

    resolved_token = (token or TELEGRAM_BOT_TOKEN or "").strip()
    if not resolved_token:
        raise ValueError("TELEGRAM_BOT_TOKEN не задан")

    try:
        _INSTANCE_LOCK = acquire(resolved_token)
    except SingleInstanceError as exc:
        logger.error("[BOT] %s", exc)
        sys.exit(1)
    except Exception as exc:  # noqa: BLE001 - не блокируем запуск из-за mutex
        logger.warning("[BOT] Не удалось проверить единственный экземпляр: %s", exc)

    bot = create_bot(resolved_token)
    register_handlers(bot, assistant)

    logger.info("[BOT] Старт polling | модуль=hay_v2_bot")
    bot.infinity_polling(timeout=60, long_polling_timeout=60)


__all__ = ["create_bot", "run_bot"]
