"""
Запуск pyTelegramBotAPI для hay_v2_bot.

Используется тот же single-instance mutex, что и у основного бота: Telegram
разрешает только один getUpdates на токен, поэтому два процесса не будут
бесконечно получать 409 Conflict.
"""

from __future__ import annotations

import logging
import os
import re
import sys
import time
from typing import Optional
from urllib.parse import urlsplit, urlunsplit

import telebot
import telebot.apihelper as apihelper

from config.settings import (
    TELEGRAM_API_URL,
    TELEGRAM_BOT_TOKEN,
    TELEGRAM_PROXY_URL,
)
from hay_v2_bot.bot.handlers import AssistantService, register_handlers
from src.single_instance import SingleInstanceError, acquire

logger = logging.getLogger("hay_v2_bot.telegram")

# Мьютекс должен жить всё время работы процесса.
_INSTANCE_LOCK = None

_TOKEN_RE = re.compile(r"bot\d+:[A-Za-z0-9_-]+")


def _redact_token(text: str) -> str:
    """Скрыть токен бота в сообщениях об ошибках и логах."""
    return _TOKEN_RE.sub("bot<id>:<token>", str(text))


class _TokenRedactionFilter(logging.Filter):
    """Не дать telebot напечатать токен в URL при сетевой ошибке."""

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = _redact_token(record.msg)
        if record.args:
            record.args = tuple(
                _redact_token(arg) if isinstance(arg, str) else arg
                for arg in record.args
            )
        return True


def _mask_proxy(url: str) -> str:
    """Скрыть логин/пароль прокси в логах."""
    try:
        parts = urlsplit(url)
        if not parts.hostname:
            return url
        netloc = parts.hostname
        if parts.port:
            netloc = f"{netloc}:{parts.port}"
        if parts.username or parts.password:
            netloc = f"***@{netloc}"
        return urlunsplit((parts.scheme, netloc, "", "", ""))
    except Exception:  # noqa: BLE001 - маскировка не должна ломать запуск
        return "<invalid proxy url>"


def configure_telegram_transport() -> None:
    """
    Настроить прокси и/или альтернативный Bot API URL для telebot.

    Прямой доступ к ``api.telegram.org`` бывает заблокирован провайдером.
    В этом случае достаточно задать в ``.env``:

    * ``TELEGRAM_PROXY_URL`` — HTTP/HTTPS или SOCKS5-прокси;
    * либо ``TELEGRAM_API_URL`` — адрес своего reverse-proxy Bot API.

    Если ``TELEGRAM_PROXY_URL`` пуст, используются стандартные
    ``HTTPS_PROXY`` / ``HTTP_PROXY`` (например, от VPN-клиента).
    """
    proxy = (
        TELEGRAM_PROXY_URL
        or os.getenv("HTTPS_PROXY")
        or os.getenv("HTTP_PROXY")
        or ""
    ).strip()

    if proxy:
        apihelper.proxy = {"http": proxy, "https": proxy}
        logger.info("[BOT] Telegram прокси: %s", _mask_proxy(proxy))
    else:
        apihelper.proxy = None

    api_url = (TELEGRAM_API_URL or "").strip().rstrip("/")
    if api_url:
        apihelper.API_URL = f"{api_url}/bot{{0}}/{{1}}"
        logger.info("[BOT] Telegram API URL: %s", api_url)
    else:
        apihelper.API_URL = None


def check_telegram_connection(
    bot: telebot.TeleBot,
    *,
    attempts: int = 2,
    delay: float = 5.0,
) -> None:
    """Проверить доступность Bot API и дать внятную инструкцию при блокировке."""
    last_error: Optional[Exception] = None

    for attempt in range(1, max(int(attempts), 1) + 1):
        try:
            me = bot.get_me()
            logger.info(
                "[BOT] Telegram подключён | @%s",
                getattr(me, "username", "unknown"),
            )
            return
        except Exception as exc:  # noqa: BLE001 - нужен понятный вывод любой сетевой ошибки
            last_error = exc
            logger.warning(
                "[BOT] Проверка Telegram %d/%d не удалась: %s",
                attempt,
                max(int(attempts), 1),
                _redact_token(str(exc)),
            )
            if attempt < max(int(attempts), 1):
                time.sleep(delay)

    logger.error("=" * 78)
    logger.error("[BOT] Нет соединения с Telegram Bot API (api.telegram.org).")
    logger.error("[BOT] Это сетевая ошибка: хост недоступен напрямую из этой сети.")
    logger.error("[BOT] Включите VPN или добавьте в .env один из параметров:")
    logger.error("[BOT]   TELEGRAM_PROXY_URL=http://user:pass@host:port")
    logger.error("[BOT]   TELEGRAM_PROXY_URL=socks5h://host:port  (нужен PySocks)")
    logger.error("[BOT]   TELEGRAM_API_URL=https://your-bot-api-proxy.example.com")
    logger.error("[BOT] После изменения .env перезапустите бота.")
    logger.error("=" * 78)
    raise RuntimeError(
        "Telegram Bot API недоступен. Настройте TELEGRAM_PROXY_URL или "
        "TELEGRAM_API_URL в .env."
    ) from last_error


def create_bot(token: Optional[str] = None) -> telebot.TeleBot:
    """Создать Telegram-бота и привести логи telebot к общему формату."""
    token = (token or TELEGRAM_BOT_TOKEN or "").strip()
    if not token or token.startswith("your_"):
        raise ValueError(
            "TELEGRAM_BOT_TOKEN не задан. Проверьте .env в корне проекта."
        )

    configure_telegram_transport()

    # pyTelegramBotAPI добавляет свой StreamHandler при импорте; без этого
    # сообщения дублируются и уровень остаётся ERROR.
    telebot_logger = logging.getLogger("TeleBot")
    telebot_logger.handlers.clear()
    telebot_logger.filters.clear()
    telebot_logger.addFilter(_TokenRedactionFilter())
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
    check_telegram_connection(bot)
    register_handlers(bot, assistant)

    logger.info("[BOT] Старт polling | модуль=hay_v2_bot")
    bot.infinity_polling(timeout=60, long_polling_timeout=60)


__all__ = [
    "check_telegram_connection",
    "configure_telegram_transport",
    "create_bot",
    "run_bot",
]
