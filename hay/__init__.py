"""
hay — изолированный модуль умного персонального помощника на фреймворке Haystack.

Модуль полностью самодостаточен: весь код находится в папке ``hay/`` и никак не
влияет на существующую архитектуру бота (src/, config/).

Состав модуля:
    * ``document_store.py`` — PineconeDocumentStore со строгой изоляцией по user_id.
    * ``tools.py``           — кастомные компоненты (@component) для внешних API.
    * ``agent.py``           — пайплайн Haystack и агент-помощник.

Импорт пакета намеренно лёгкий: тяжёлые зависимости (haystack, sentence-transformers)
загружаются только при первом обращении к ``build_assistant()``.
"""

from __future__ import annotations

from typing import Any

__all__ = ["build_assistant", "HaystackAssistant", "is_available"]

__version__ = "1.0.0"


def is_available() -> bool:
    """Проверить, установлены ли зависимости модуля hay."""
    try:
        import haystack  # noqa: F401
        from haystack_integrations.document_stores.pinecone import (  # noqa: F401
            PineconeDocumentStore,
        )
    except Exception:
        return False
    return True


def build_assistant(*args: Any, **kwargs: Any) -> "HaystackAssistant":
    """Собрать агента Haystack (ленивый импорт — не тормозит старт бота)."""
    from hay.agent import HaystackAssistant as _HaystackAssistant

    return _HaystackAssistant(*args, **kwargs)