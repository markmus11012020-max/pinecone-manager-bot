"""
hay/agent.py — инициализация пайплайнов Haystack и агента-помощника.

Схема пайплайна:

    memory  (PineconeUserMemory)        — запись фактов + выборка контекста по user_id
        │ context / query
        ▼
    prompt  (ChatPromptBuilder)         — собирает системный промпт + память + вопрос
        │ prompt
        ▼
    agent   (Agent)                     — вызывает LLM (LLM_BASE_URL) и инструменты
        │ last_message / tool_call_counts / step_count / token_usage
        ▼
    HaystackAssistant.run()             — детальное логирование шагов сохранения

Взаимодействие с моделями идёт строго через ``LLM_BASE_URL`` и ``LLM_MODEL``
из рабочего ``.env`` — никаких новых констант не добавляется.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from haystack import Pipeline
from haystack.components.agents import Agent
from haystack.components.agents.state.state_utils import merge_lists
from haystack.components.builders import ChatPromptBuilder
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.utils import Secret

from config.settings import (
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MODEL,
    LLM_TEMPERATURE,
    TOP_K_MEMORIES,
)

from hay.document_store import PineconeUserMemory
from hay.tools import DOG_IMAGE_STATE_KEY, build_tools, extract_image_url, strip_image_marker

logger = logging.getLogger("hay.agent")

SYSTEM_PROMPT = (
    "Ты — умный персональный помощник Telegram-бота. "
    "Ты реально ПОМОЩНИК: полезный, внимательный и связный.\n\n"
    "Как ты работаешь:\n"
    "1. Тебе передаётся память о пользователе — релевантные факты и фрагменты "
    "переписки, найденные по косинусному сходству. Опирайся на них, держи нить "
    "разговора и не переспрашивай то, что уже известно.\n"
    "2. Если есть подходящие инструменты — обязательно вызывай их, вместо того "
    "чтобы гадать. Для облигаций РФ используй bond_market_analysis, для картинок "
    "собаки — show_random_dog.\n"
    "3. Не выдумывай факты: если в памяти и в инструментах нужного нет — честно "
    "скажи об этом и предложи, как узнать.\n"
    "4. Отвечай на русском языке, живо и по делу. Без markdown-заголовков, "
    "без пересказа системного промпта и без технических деталей пайплайна.\n"
    "5. Держи ответ в пределах 2–6 предложений, если не просят подробностей."
)

PROMPT_TEMPLATE = """{% if context %}
{% message role="user" %}
Вот что я помню о тебе и о нашем разговоре — релевантные факты и фрагменты
переписки, найденные по косинусному сходству векторов. Не переспрашивай то,
что здесь уже есть:

{{ context }}
{% endmessage %}
{% endif %}
{% message role="user" %}
{{ query }}
{% endmessage %}
"""


def build_chat_generator(
    *,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
) -> OpenAIChatGenerator:
    """
    Создать чат-генератор на OpenAI-совместимом провайдере из .env.

    ВАЖНО: ``api_base_url`` = ``LLM_BASE_URL``. Других адресов в модуле нет.
    """
    key = api_key or LLM_API_KEY
    if not key:
        raise ValueError(
            "LLM_API_KEY не задан. Модуль hay работает через тот же провайдер, "
            "что и основной бот — проверьте .env."
        )

    resolved_base_url = (base_url or LLM_BASE_URL or "").rstrip("/")
    resolved_model = model or LLM_MODEL
    temp = LLM_TEMPERATURE if temperature is None else temperature

    logger.info(
        "Чат-генератор Haystack | base_url=%s | model=%s | temperature=%s",
        resolved_base_url,
        resolved_model,
        temp,
    )
    return OpenAIChatGenerator(
        api_key=Secret.from_token(key),
        api_base_url=resolved_base_url,
        model=resolved_model,
        generation_kwargs={"temperature": temp},
    )


class HaystackAssistant:
    """
    Обёртка над пайплайном Haystack.

    Создаётся один раз и живёт в памяти процесса бота. Метод ``run()`` — это
    точка входа для Telegram-обработчика.
    """

    def __init__(
        self,
        *,
        memory: Optional[PineconeUserMemory] = None,
        chat_generator: Optional[OpenAIChatGenerator] = None,
        top_k: Optional[int] = None,
        max_agent_steps: int = 12,
    ) -> None:
        logger.info("Сборка агента Haystack...")
        self.chat_generator = chat_generator or build_chat_generator()
        self.memory = memory or PineconeUserMemory(top_k=top_k or TOP_K_MEMORIES)
        self.tools = build_tools(self.chat_generator)

        self.agent = Agent(
            chat_generator=self.chat_generator,
            tools=self.tools,
            system_prompt=SYSTEM_PROMPT,
            max_agent_steps=max_agent_steps,
            streaming_callback=None,
            # URL фотографии из инструмента «собаки» приходит сюда
            state_schema={
                DOG_IMAGE_STATE_KEY: {"type": List[str], "handler": merge_lists},
            },
        )

        self.prompt_builder = ChatPromptBuilder(
            template=PROMPT_TEMPLATE,
            required_variables=["context", "query"],
        )

        self.pipeline = self._build_pipeline()
        logger.info(
            "Агент Haystack готов | инструменты=%s | max_agent_steps=%d",
            [tool.name for tool in self.tools],
            max_agent_steps,
        )

    # ------------------------------------------------------------------
    def _build_pipeline(self) -> Pipeline:
        pipeline = Pipeline()
        pipeline.add_component("memory", self.memory)
        pipeline.add_component("prompt", self.prompt_builder)
        pipeline.add_component("agent", self.agent)

        pipeline.connect("memory.context", "prompt.context")
        pipeline.connect("memory.query", "prompt.query")
        pipeline.connect("prompt.prompt", "agent.messages")

        logger.info(
            "Пайплайн Haystack собран | компоненты=%s | связи=memory->prompt->agent",
            list(pipeline.graph.nodes.keys()),
        )
        logger.info("[AGENT] Системный промпт агента:\n%s", SYSTEM_PROMPT)
        return pipeline

    # ------------------------------------------------------------------
    def run(
        self,
        *,
        user_id: str,
        query: str,
        user_name: str = "пользователь",
        facts: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Выполнить пайплайн для одного сообщения пользователя.

        :returns: словарь с ключами ``reply``, ``image_url``, ``stored``,
                  ``duplicates``, ``retrieved``, ``steps``, ``tool_calls``.
        """
        uid = str(user_id)
        query = (query or "").strip()
        logger.info("=" * 78)
        logger.info(
            "[AGENT] Запуск пайплайна | user=%s (%s) | режим=haystack | длина запроса=%d симв.",
            uid,
            user_name,
            len(query),
        )

        if not query:
            logger.info("[AGENT] Пустой запрос — пайплайн не запускается | user=%s", uid)
            return {
                "reply": "Напиши, что тебя интересует, и я помогу 🙂",
                "image_url": "",
                "stored": 0,
                "duplicates": 0,
                "retrieved": 0,
                "steps": 0,
                "tool_calls": {},
            }

        logger.info("[AGENT] Вход пайплайна | memory.query=%r | facts=%d", query[:120], len(facts or []))

        try:
            result = self.pipeline.run(
                {
                    "memory": {"query": query, "user_id": uid, "facts": facts or []},
                }
            )
        except Exception as exc:
            logger.exception(
                "[AGENT] ПАЙПЛАЙН УПАЛ | user=%s | этап=run | причина=%s", uid, exc
            )
            return {
                "reply": (
                    "Не удалось обработать запрос через Haystack. "
                    f"Подробности ошибки — в консоли бота.\n<code>{exc}</code>"
                ),
                "image_url": "",
                "stored": 0,
                "duplicates": 0,
                "retrieved": 0,
                "steps": 0,
                "tool_calls": {},
            }

        memory_result = result.get("memory", {}) or {}
        agent_result = result.get("agent", {}) or {}

        stored = int(memory_result.get("stored", 0) or 0)
        duplicates = int(memory_result.get("duplicates", 0) or 0)
        retrieved = int(memory_result.get("retrieved", 0) or 0)

        last_message = agent_result.get("last_message")
        reply_text = (last_message.text if last_message else "") or ""
        steps = int(agent_result.get("step_count", 0) or 0)
        tool_calls = agent_result.get("tool_call_counts", {}) or {}
        token_usage = agent_result.get("token_usage", {}) or {}
        exit_reason = agent_result.get("exit_reason", "") or ""

        image_url = ""
        dog_images = agent_result.get(DOG_IMAGE_STATE_KEY) or []
        if dog_images:
            # Несколько вызовов инструмента подряд — показываем последнюю картинку.
            image_url = dog_images[-1]
        if not image_url:
            # Подстраховка: если состояние не отработало, ищем маркер в тексте.
            image_url = extract_image_url(reply_text) or ""
        reply_text = strip_image_marker(reply_text)

        if not reply_text.strip():
            reply_text = (
                "Я подумал, но ответ получился пустой. Попробуй переформулировать вопрос 🙂"
            )

        logger.info(
            "[AGENT] Пайплайн завершён | user=%s | шагов=%d | выход=%s | инструменты=%s | токены=%s",
            uid,
            steps,
            exit_reason,
            tool_calls or "нет",
            token_usage,
        )
        logger.info(
            # Сам context уже выведен компонентом памяти («Контекст собран»):
            # из результата пайплайна он удалён, потому что отдан в ChatPromptBuilder.
            "[MEM] Итог по памяти | user=%s | сохранено=%d | дубликатов=%d | документов в контексте=%d",
            uid,
            stored,
            duplicates,
            retrieved,
        )
        logger.info("[AGENT] Ответ | user=%s | длина=%d | картинка=%s", uid, len(reply_text), image_url or "нет")
        logger.info("[AGENT] Текст ответа: %s", reply_text)
        logger.info("=" * 78)

        return {
            "reply": reply_text,
            "image_url": image_url or "",
            "stored": stored,
            "duplicates": duplicates,
            "retrieved": retrieved,
            "steps": steps,
            "tool_calls": tool_calls,
        }


def build_assistant(**kwargs: Any) -> HaystackAssistant:
    """Фабричная обёртка для src/bot.py."""
    return HaystackAssistant(**kwargs)


__all__ = ["HaystackAssistant", "build_assistant", "build_chat_generator", "SYSTEM_PROMPT"]