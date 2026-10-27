"""
Точка входа второй версии бота: Haystack + Docling + Pinecone.

Запуск из корня проекта:

    venv\\Scripts\\python.exe hay_v2_bot\\main.py

Проверка окружения без подключения к Telegram/Pinecone:

    venv\\Scripts\\python.exe hay_v2_bot\\main.py --check
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (  # noqa: E402
    LLM_BASE_URL,
    LLM_MODEL,
    PINECONE_INDEX_NAME,
    TELEGRAM_BOT_TOKEN,
)

logger = logging.getLogger("hay_v2_bot.main")


def configure_logging() -> None:
    """Единый формат логов в консоли (без отдельных .log-файлов)."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Шумные библиотеки не должны прятать наши этапы сохранения в Pinecone.
    for name in (
        "httpx",
        "httpx2",
        "urllib3",
        "huggingface_hub",
        "filelock",
        "PIL",
        "pinecone",
    ):
        logging.getLogger(name).setLevel(logging.WARNING)


def run_checks() -> int:
    """Проверить наличие зависимостей и ключевых настроек без сети."""
    import importlib.metadata as metadata

    logger.info("=" * 78)
    logger.info("[CHECK] Проверка окружения hay_v2_bot")

    ok = True
    for distribution in ("haystack-ai", "pinecone-haystack", "docling", "sentence-transformers"):
        try:
            version = metadata.version(distribution)
            logger.info("[CHECK] %-24s %s", distribution, version)
        except metadata.PackageNotFoundError:
            logger.error("[CHECK] %-24s НЕ УСТАНОВЛЕН", distribution)
            ok = False

    if not TELEGRAM_BOT_TOKEN or TELEGRAM_BOT_TOKEN.startswith("your_"):
        logger.error("[CHECK] TELEGRAM_BOT_TOKEN не задан")
        ok = False
    else:
        logger.info("[CHECK] TELEGRAM_BOT_TOKEN задан")

    logger.info("[CHECK] LLM_BASE_URL=%s", LLM_BASE_URL)
    logger.info("[CHECK] LLM_MODEL=%s", LLM_MODEL)
    logger.info("[CHECK] PINECONE_INDEX_NAME=%s", PINECONE_INDEX_NAME)

    try:
        from hay_v2_bot.pipelines.indexing import IndexingPipeline  # noqa: F401
        from hay_v2_bot.pipelines.generation import GenerationPipeline  # noqa: F401
        from hay_v2_bot.components.docling_converter import (  # noqa: F401
            DoclingDocumentConverter,
        )

        logger.info("[CHECK] Модули пайплайнов и Docling импортируются")
    except Exception as exc:  # noqa: BLE001 - check должен показать любую ошибку
        logger.exception("[CHECK] ОШИБКА импорта модулей: %s", exc)
        ok = False

    logger.info("[CHECK] Итог: %s", "OK" if ok else "ЕСТЬ ОШИБКИ")
    logger.info("=" * 78)
    return 0 if ok else 1


def main() -> int:
    configure_logging()

    if "--check" in sys.argv[1:]:
        return run_checks()

    logger.info("=" * 78)
    logger.info("[MAIN] Запуск hay_v2_bot")
    logger.info(
        "[MAIN] LLM_BASE_URL=%s | LLM_MODEL=%s | index=%s",
        LLM_BASE_URL,
        LLM_MODEL,
        PINECONE_INDEX_NAME,
    )

    if not TELEGRAM_BOT_TOKEN or TELEGRAM_BOT_TOKEN.startswith("your_"):
        logger.error("[MAIN] TELEGRAM_BOT_TOKEN не задан. Проверьте .env.")
        return 1

    try:
        from hay_v2_bot.bot.handlers import AssistantService
        from hay_v2_bot.bot.telegram_bot import run_bot

        logger.info("[MAIN] Этап 1/2: сборка пайплайнов, Pinecone и модели эмбеддингов...")
        assistant = AssistantService.build()
        logger.info("[MAIN] Этап 2/2: запуск Telegram polling...")
        run_bot(assistant)
    except KeyboardInterrupt:
        logger.info("[MAIN] Остановлено пользователем (Ctrl+C)")
        return 0
    except Exception as exc:  # noqa: BLE001 - единая точка логирования фатальных ошибок
        logger.exception("[MAIN] ФАТАЛЬНАЯ ОШИБКА: %s", exc)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
