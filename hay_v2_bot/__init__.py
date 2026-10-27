"""
hay_v2_bot — вторая версия Telegram-бота: Haystack + Docling + Pinecone (RAG).

Модуль полностью изолирован от текущего бота (``src/`` и ``hay/``) и запускается
отдельной точкой входа ``hay_v2_bot/main.py``. Все настройки берутся из уже
существующего ``.env`` через ``config.settings`` — новых переменных не добавляется.

Состав:
    * ``components/docling_converter.py`` — Docling -> смысловые чанки Haystack;
    * ``components/embeddings.py``         — sentence-transformers для чанков и запросов;
    * ``pipelines/indexing.py``            — индексация файла в Pinecone;
    * ``pipelines/generation.py``          — RAG-ответ по документам пользователя;
    * ``bot/``                             — pyTelegramBotAPI и обработчики команд;
    * ``main.py``                          — общая точка входа.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "2.0.0"
