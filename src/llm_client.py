"""
LLM client for AITUNNEL (OpenAI-compatible API).
Default: base_url=https://api.aitunnel.ru/v1, model=openai/gpt-5-nano
"""

from __future__ import annotations

import logging
from typing import List, Optional

from openai import OpenAI

from config.settings import (
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MAX_TOKENS,
    LLM_MODEL,
    LLM_TEMPERATURE,
)

logger = logging.getLogger(__name__)


class LLMClient:
    """Thin wrapper around the OpenAI-compatible chat completions API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> None:
        self.api_key = api_key or LLM_API_KEY
        self.base_url = (base_url or LLM_BASE_URL or "").rstrip("/")
        self.model = model or LLM_MODEL
        self.max_tokens = max_tokens if max_tokens is not None else LLM_MAX_TOKENS
        self.temperature = temperature if temperature is not None else LLM_TEMPERATURE

        if not self.api_key or self.api_key.startswith("sk-aitunnel-your"):
            raise ValueError(
                "LLM_API_KEY is not set. Get a key at https://aitunnel.ru and put it in .env"
            )

        # Ensure base_url ends without trailing slash issues; OpenAI SDK accepts both
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )
        logger.info(
            "LLMClient ready | base_url=%s | model=%s",
            self.base_url,
            self.model,
        )

    def chat(
        self,
        messages: List[dict],
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Send a chat completion request and return the assistant message content.
        """
        used_model = model or self.model
        try:
            response = self.client.chat.completions.create(
                model=used_model,
                messages=messages,
                max_tokens=max_tokens if max_tokens is not None else self.max_tokens,
                temperature=temperature if temperature is not None else self.temperature,
            )
            content = response.choices[0].message.content or ""
            return content.strip()
        except Exception as e:
            logger.exception("LLM request failed (model=%s): %s", used_model, e)
            raise

    def reply_with_memory(
        self,
        user_message: str,
        memory_context: str,
        user_name: str = "пользователь",
    ) -> str:
        """
        Generate a natural reply that uses long-term memory context.
        """
        system_prompt = (
            "Ты — дружелюбный и внимательный Telegram-бот-помощник с долговременной памятью. "
            "Отвечай на русском языке, кратко и по делу, если не попросили иначе. "
            "Используй сохранённые воспоминания о пользователе, когда они релевантны. "
            "Не выдумывай факты, которых нет в памяти. "
            "Если памяти нет — просто ответь на сообщение и при необходимости уточни детали."
        )

        user_content_parts = []
        if memory_context:
            user_content_parts.append(
                "Релевантные воспоминания о пользователе:\n" + memory_context
            )
        user_content_parts.append(f"Сообщение от {user_name}:\n{user_message}")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "\n\n".join(user_content_parts)},
        ]

        return self.chat(messages)
