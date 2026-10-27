"""Кастомные компоненты Haystack для hay_v2_bot."""

from __future__ import annotations

from hay_v2_bot.components.docling_converter import DoclingDocumentConverter
from hay_v2_bot.components.embeddings import (
    SentenceTransformerDocumentEmbedder,
    SentenceTransformerEmbedder,
    SentenceTransformerTextEmbedder,
)

__all__ = [
    "DoclingDocumentConverter",
    "SentenceTransformerEmbedder",
    "SentenceTransformerDocumentEmbedder",
    "SentenceTransformerTextEmbedder",
]
