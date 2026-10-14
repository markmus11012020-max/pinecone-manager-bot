"""
hay/tools.py — кастомные компоненты Haystack (@component) для работы с внешними API.

Инструменты:
    1. ``BondMarketAnalyzer`` — анализ рынка корпоративных облигаций РФ.
       Бесплатные источники без ключей:
         * MOEX ISS (iss.moex.com) — доходность, цена, купон, дата ближайшего
           купонного платежа, дата погашения, объём выпуска по конкретным бумагам;
         * RSS Интерфакса (interfax.ru/rss.asp) — новостные события по облигациям.
    2. ``DogStoryteller`` — мультимодальный инструмент «Собаки»:
         * случайная картинка из бесплатного Dog API (dog.ceo);
         * распознавание породы и генерация предыстории моделью из ``LLM_MODEL``
           через ``LLM_BASE_URL`` (новых констант не добавляется).

Каждый инструмент отдаёт агенту обычную строку, а картинка помечается
маркером ``IMAGE_URL:<url>``, чтобы Telegram-бот отправил фото вместе с подписью.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional
from xml.etree import ElementTree

import requests
from haystack import component
from haystack.components.agents.state.state_utils import merge_lists
from haystack.dataclasses import ChatMessage, ImageContent
from haystack.tools import Tool

logger = logging.getLogger("hay.tools")

# Ключ состояния агента, в который инструмент «собаки» кладёт URL фотографии
DOG_IMAGE_STATE_KEY = "dog_image_url"

# Запасной маркер: если по какой-то причине состояние недоступно, бот всё равно
# сможет вытащить картинку из текста ответа.
IMAGE_MARKER = "IMAGE_URL:"
_IMAGE_URL_RE = re.compile(r"IMAGE_URL:\s*(https?://\S+)", re.IGNORECASE)

REQUEST_TIMEOUT = 20
USER_AGENT = "Mozilla/5.0 (compatible; pinecone-manager-bot/haystack; +https://github.com/)"

# Лимиты токенов для вспомогательных вызовов LLM внутри инструментов.
# LLM_MODEL — рассуждающая модель: она тратит токены на reasoning, поэтому даже
# короткому ответу («назови породу») нужен большой запас. При нехватке бюджета
# провайдер возвращает пустой текст с finish_reason=length.
BREED_MAX_TOKENS = 2000
BREED_RETRY_MAX_TOKENS = 6000
STORY_MAX_TOKENS = 2000


def extract_image_url(text: str) -> Optional[str]:
    """Достать URL картинки из текста, если инструмент её приложил."""
    if not text:
        return None
    match = _IMAGE_URL_RE.search(text)
    return match.group(1).rstrip(".,)") if match else None


def strip_image_marker(text: str) -> str:
    """Убрать служебный маркер из текста перед отправкой в Telegram."""
    return _IMAGE_URL_RE.sub("", text or "").strip()


# ===========================================================================
# 1. Анализ рынка корпоративных облигаций РФ
# ===========================================================================
@component
class BondMarketAnalyzer:
    """
    Кастомный компонент Haystack: факты о корпоративных облигациях РФ.

    Собирает два слоя информации:
        * рыночные данные по бумагам (MOEX ISS) — «что происходит с облигацией»;
        * новостные события (RSS Интерфакса) — «почему это происходит».
    """

    MOEX_URL = (
        "https://iss.moex.com/iss/engines/stock/markets/bonds/securities.json"
        "?iss.meta=off&iss.only=securities,marketdata"
    )
    INTERFAX_RSS = "https://www.interfax.ru/rss.asp"

    # Доски государственных облигаций — их исключаем, нужен корпоративный рынок
    GOVERNMENT_BOARDS = {"TQOB", "SPOB", "PACT"}

    # Ключевые слова для отбора релевантных новостей из ленты
    NEWS_KEYWORDS = (
        "облигац",
        "бонд",
        "выпуск",
        "эмитент",
        "дефолт",
        "купон",
        "доходност",
        "погашен",
        "размещен",
        "минфин",
        "moscow exchange",
        "мosбиржа",
        "московск",
        "рф",
    )

    @component.output_types(report=str, items=List[str], market=List[str])
    def run(self, query: str = "", limit: int = 5) -> Dict[str, Any]:
        """
        :param query: запрос пользователя (например, «Сбербанк» или «доходность Газпрома»).
        :param limit: сколько позиций вернуть по каждому срезу.
        """
        query = (query or "").strip()
        logger.info("[TOOL:BONDS] Старт | query=%r | limit=%s", query[:120], limit)

        market_items = self._moex_snapshot(query, limit)
        news_items = self._news(query, limit)

        items = market_items + news_items
        report = self._render(query, market_items, news_items)

        logger.info(
            "[TOOL:BONDS] Готово | рыночных фактов=%d | новостей=%d",
            len(market_items),
            len(news_items),
        )
        return {"report": report, "items": items, "market": market_items}

    # ------------------------------------------------------------------
    # MOEX ISS
    # ------------------------------------------------------------------
    def _moex_snapshot(self, query: str, limit: int) -> List[str]:
        if not query:
            return []
        try:
            response = requests.get(
                self.MOEX_URL,
                params={"q": query, "limit": max(limit, 5) * 3},
                timeout=REQUEST_TIMEOUT,
                headers={"User-Agent": USER_AGENT},
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            logger.warning("[TOOL:BONDS] MOEX ISS недоступен: %s", exc)
            return []

        securities = self._rows_to_dicts(payload.get("securities"))
        marketdata = {
            row.get("SECID"): row for row in self._rows_to_dicts(payload.get("marketdata"))
        }

        lines: List[str] = []
        for sec in securities:
            secid = sec.get("SECID")
            if not secid or sec.get("BOARDID") in self.GOVERNMENT_BOARDS:
                continue

            quote = marketdata.get(secid, {})
            name = self._clean(sec.get("SHORTNAME") or sec.get("SECNAME") or secid)
            facts: List[str] = []

            yld = quote.get("YIELD") or sec.get("YIELDATPREVWAPRICE")
            if isinstance(yld, (int, float)) and yld:
                facts.append(f"доходность {yld:.2f}%")
            price = quote.get("MARKETPRICE") or sec.get("PREVPRICE")
            if isinstance(price, (int, float)) and price:
                facts.append(f"цена {price:.2f}% от номинала")

            coupon_pct = sec.get("COUPONPERCENT")
            if isinstance(coupon_pct, (int, float)) and coupon_pct:
                facts.append(f"купон {coupon_pct:.2f}%")

            next_coupon = self._clean(sec.get("NEXTCOUPON"))
            if next_coupon and next_coupon != "0000-00-00":
                facts.append(f"следующий платёж по купону {next_coupon}")
            matdate = self._clean(sec.get("MATDATE"))
            if matdate and matdate != "0000-00-00":
                facts.append(f"погашение {matdate}")

            duration = quote.get("DURATION")
            if isinstance(duration, (int, float)) and duration:
                facts.append(f"дюрация {duration} дн.")

            isin = self._clean(sec.get("ISIN"))
            issue_size = sec.get("ISSUESIZE")
            extra = []
            if isin:
                extra.append(f"ISIN {isin}")
            if isinstance(issue_size, (int, float)) and issue_size:
                extra.append(f"объём выпуска {issue_size / 1e9:.2f} млрд руб.")

            if facts:
                tail = (" (" + "; ".join(extra) + ")") if extra else ""
                lines.append(f"{secid} — {name}: " + ", ".join(facts) + tail + ".")

            if len(lines) >= limit:
                break

        logger.info("[TOOL:BONDS] MOEX ISS вернул %d позиций", len(lines))
        return lines

    # ------------------------------------------------------------------
    # RSS Интерфакса
    # ------------------------------------------------------------------
    def _news(self, query: str, limit: int) -> List[str]:
        try:
            response = requests.get(
                self.INTERFAX_RSS, timeout=REQUEST_TIMEOUT, headers={"User-Agent": USER_AGENT}
            )
            response.raise_for_status()
            root = ElementTree.fromstring(response.content)
        except Exception as exc:
            logger.warning("[TOOL:BONDS] RSS Интерфакса недоступен: %s", exc)
            return []

        query_words = [w.lower() for w in re.findall(r"[\wА-Яа-яЁё-]{4,}", query or "")]
        lines: List[str] = []
        for item in root.iter("item"):
            title = self._clean(item.findtext("title"))
            description = self._clean(item.findtext("description"))
            link = self._clean(item.findtext("link"))
            published = self._clean(item.findtext("pubDate"))
            if not title:
                continue

            haystack_text = f"{title} {description}".lower()
            bond_relevant = any(keyword in haystack_text for keyword in self.NEWS_KEYWORDS)
            query_relevant = bool(query_words) and any(word in haystack_text for word in query_words)
            if not bond_relevant or not query_relevant:
                continue

            stamp = self._format_date(published)
            lines.append(f"{stamp}: {title}. {description[:200].strip()}{f' Источник: {link}' if link else ''}")
            if len(lines) >= limit:
                break

        logger.info("[TOOL:BONDS] Новостей отобрано: %d", len(lines))
        return lines

    # ------------------------------------------------------------------
    # Вспомогательные
    # ------------------------------------------------------------------
    @staticmethod
    def _rows_to_dicts(block: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not block or "columns" not in block:
            return []
        columns = block["columns"]
        return [dict(zip(columns, row)) for row in (block.get("data") or [])]

    @staticmethod
    def _clean(value: Any) -> str:
        if value is None:
            return ""
        text = re.sub(r"<[^>]+>", " ", str(value))
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _format_date(pub_date: str) -> str:
        if not pub_date:
            return datetime.now().strftime("%d.%m.%Y")
        for fmt in ("%a, %d %b %Y %H:%M:%S %z", "%d %b %Y %H:%M:%S %z", "%Y-%m-%dT%H:%M:%S"):
            try:
                return datetime.strptime(pub_date, fmt).strftime("%d.%m.%Y")
            except ValueError:
                continue
        return pub_date[:16]

    @staticmethod
    def _render(query: str, market: List[str], news: List[str]) -> str:
        parts: List[str] = [
            f"Анализ корпоративных облигаций РФ по запросу: «{query or 'общий обзор'}».",
            f"Данные MOEX ISS и RSS Интерфакса. Дата запроса: {datetime.now().strftime('%d.%m.%Y')}.",
        ]
        if market:
            parts.append("Рыночные факты:")
            parts.extend(f"- {item}" for item in market)
        else:
            parts.append("Рыночные факты: подходящих бумаг по запросу не найдено.")
        if news:
            parts.append("Новостные события:")
            parts.extend(f"- {item}" for item in news)
        else:
            parts.append("Новостные события: свежих релевантных публикаций не найдено.")
        parts.append(
            "Это справочные данные с биржи, не инвестиционная рекомендация. "
            "Если данных мало — честно скажи об этом и уточни, какой эмитент интересует."
        )
        return "\n".join(parts)


# ===========================================================================
# 2. Мультимодальный инструмент «Собаки»
# ===========================================================================
@component
class DogStoryteller:
    """
    Кастомный компонент Haystack: случайное фото собаки + распознавание породы
    и генерация предыстории.

    Модель берётся из ``LLM_MODEL``, обращение — строго через ``LLM_BASE_URL``
    (``OpenAIChatGenerator`` настроен на эти значения). Новых констант в код
    и окружение не добавляется.
    """

    RANDOM_IMAGE_URL = "https://dog.ceo/api/breeds/image/random"
    BREEDS_URL = "https://dog.ceo/api/breeds/list/all"

    def __init__(self, chat_generator: Any) -> None:
        self.chat_generator = chat_generator

    @component.output_types(story=str, image_url=List[str], breed=str)
    def run(self, request: str = "") -> Dict[str, Any]:
        """
        :param request: пожелание пользователя (необязательно), например «смешная собака».
        """
        logger.info("[TOOL:DOGS] Старт | request=%r", (request or "")[:120])

        image_url = self._random_image()
        if not image_url:
            story = "Не удалось загрузить фотографию собаки из Dog API. Попробуй ещё раз чуть позже."
            logger.error("[TOOL:DOGS] Dog API не ответил")
            return {"story": story, "image_url": [], "breed": ""}

        breed = self._identify_breed(image_url)
        story = self._write_story(image_url, breed, request)
        logger.info("[TOOL:DOGS] Готово | порода=%s | url=%s", breed or "не определена", image_url)

        return {
            "story": story,
            # Список — состояние агента объединяет такие значения через merge_lists.
            "image_url": [image_url],
            "breed": breed,
        }

    # ------------------------------------------------------------------
    def _random_image(self) -> str:
        try:
            response = requests.get(
                self.RANDOM_IMAGE_URL, timeout=REQUEST_TIMEOUT, headers={"User-Agent": USER_AGENT}
            )
            response.raise_for_status()
            url = response.json().get("message")
            logger.info("[TOOL:DOGS] Dog API вернул изображение: %s", url)
            return url or ""
        except Exception as exc:
            logger.warning("[TOOL:DOGS] Ошибка Dog API: %s", exc)
            return ""

    def _breed_candidates(self) -> List[str]:
        try:
            response = requests.get(
                self.BREEDS_URL, timeout=REQUEST_TIMEOUT, headers={"User-Agent": USER_AGENT}
            )
            response.raise_for_status()
            breeds = response.json().get("message", {}) or {}
            return [str(name).replace("-", " ") for name in breeds.keys()][:120]
        except Exception as exc:
            logger.warning("[TOOL:DOGS] Не удалось получить список пород: %s", exc)
            return []

    def _generate(
        self,
        messages: List[ChatMessage],
        max_tokens: int = STORY_MAX_TOKENS,
        retry_budget: Optional[int] = None,
    ) -> str:
        """
        Вызов LLM внутри инструмента.

        Рассуждающая модель может съесть весь лимит на reasoning и вернуть пустой
        текст с ``finish_reason=length`` — в этом случае один раз повторяем запрос
        с увеличенным бюджетом.
        """
        budgets = [max_tokens] + ([retry_budget] if retry_budget else [])

        for attempt, budget in enumerate(budgets):
            try:
                result = self.chat_generator.run(
                    messages=messages, generation_kwargs={"max_tokens": budget}
                )
            except Exception as exc:
                if attempt == 0 and retry_budget:
                    logger.warning(
                        "[TOOL:DOGS] Запрос не удался при max_tokens=%d (%s) — повтор с %d",
                        budget,
                        exc,
                        retry_budget,
                    )
                    continue
                raise

            replies = result.get("replies") or []
            text = (replies[0].text or "").strip() if replies else ""
            if text:
                return text
            logger.warning(
                "[TOOL:DOGS] Пустой ответ при max_tokens=%d (finish_reason=%s)",
                budget,
                (replies[0].meta or {}).get("finish_reason") if replies else "нет ответа",
            )

        return ""

    def _identify_breed(self, image_url: str) -> str:
        """Попытаться распознать породу по картинке, с откатом на текстовый режим."""
        prompt = (
            "Определи породу собаки на фотографии. "
            "Ответь только названием породы, одним словом или двумя. "
            "Если уверенности нет — напиши наиболее вероятный вариант."
        )

        # 1. Мультимодальный запрос (если модель умеет видеть картинки)
        try:
            message = ChatMessage.from_user(
                content_parts=[prompt, ImageContent.from_url(image_url)]
            )
            breed = self._generate(
                [message],
                max_tokens=BREED_MAX_TOKENS,
                retry_budget=BREED_RETRY_MAX_TOKENS,
            )
            if breed:
                logger.info("[TOOL:DOGS] Порода распознана (vision): %s", breed[:80])
                return breed[:120]
            logger.info("[TOOL:DOGS] Vision-ответ пуст — переключаюсь на текстовый откат")
        except Exception as exc:
            logger.info("[TOOL:DOGS] Модель не поддержала изображение (%s) — переключаюсь на текст", exc)

        # 2. Текстовый откат: сузим поиск по списку известных пород
        candidates = self._breed_candidates()
        if candidates:
            hint = f"Возможные породы: {', '.join(candidates[:60])}."
            try:
                breed = self._generate(
                    [ChatMessage.from_user(f"{hint}\n{prompt} Если подходит несколько — назми лучшую.")],
                    max_tokens=BREED_MAX_TOKENS,
                    retry_budget=BREED_RETRY_MAX_TOKENS,
                )
                if breed:
                    logger.info("[TOOL:DOGS] Порода предположена (text): %s", breed[:80])
                    return breed[:120]
            except Exception as exc:
                logger.warning("[TOOL:DOGS] Текстовый откат не сработал: %s", exc)

        return ""

    def _write_story(self, image_url: str, breed: str, request: str) -> str:
        breed_part = f"Порода: {breed}." if breed else "Порода точно не определена — опиши её осторожно."
        wish = f" Пожелание пользователя: {request}." if (request or "").strip() else ""
        prompt = (
            "Ты — добродушный рассказчик. Напиши на русском короткую живую предысторию "
            "этой собаки (3–5 предложений): как зовут, какой характер, где живёт, что любит. "
            "Правдиво выдумывай, опираясь на внешний вид. Без markdown-заголовков и списков.\n"
            f"{breed_part}{wish}"
        )
        try:
            message = ChatMessage.from_user(
                content_parts=[prompt, ImageContent.from_url(image_url)]
            )
            story = self._generate(
                [message], max_tokens=STORY_MAX_TOKENS, retry_budget=STORY_MAX_TOKENS * 2
            )
        except Exception as exc:
            logger.warning("[TOOL:DOGS] Мультимодальная генерация не удалась: %s", exc)
            story = ""
        if not story:
            try:
                story = self._generate(
                    [ChatMessage.from_user(prompt)],
                    max_tokens=STORY_MAX_TOKENS,
                    retry_budget=STORY_MAX_TOKENS * 2,
                )
            except Exception as inner_exc:
                logger.error("[TOOL:DOGS] LLM недоступен: %s", inner_exc)
                story = ""

        if not story:
            story = (
                f"{breed_part} Собака уже смотрит на тебя с экрана и очень надеется, "
                "что ей скорее найдут дом и вкусное лакомство."
            )

        return story


# ===========================================================================
# Сборка инструментов для агента
# ===========================================================================
def build_bond_tool() -> Tool:
    """Обернуть компонент BondMarketAnalyzer в Tool для агента Haystack."""
    analyzer = BondMarketAnalyzer()

    def bond_market_analysis(query: str = "", limit: int = 5) -> str:
        return analyzer.run(query=query, limit=limit)["report"]

    return Tool(
        name="bond_market_analysis",
        description=(
            "Анализ рынка корпоративных облигаций РФ по бесплатным данным MOEX ISS "
            "и новостям Интерфакса. Возвращает доходность, цену, купон, даты купонного "
            "платежа и погашения, а также свежие новостные события. "
            "Используй, когда пользователь спрашивает об облигациях, эмитентах, "
            "доходностях или событиях по корпоративным выпускам."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Название эмитента, бумаги или тема: «Сбербанк», «Газпром», «доходности ОФЗ».",
                },
                "limit": {
                    "type": "integer",
                    "description": "Сколько фактов и новостей вернуть (по умолчанию 5).",
                },
            },
            "required": ["query"],
        },
        function=bond_market_analysis,
    )


def build_dog_tool(chat_generator: Any) -> Tool:
    """
    Обернуть компонент DogStoryteller в Tool для агента Haystack.

    Инструмент отдаёт модели только текст (``outputs_to_string``), а URL фотографии
    уходит в состояние агента (``outputs_to_state``) — так бот надёжно отправит
    картинку, даже если агент перепишет ответ своими словами.
    """
    storyteller = DogStoryteller(chat_generator=chat_generator)

    def show_random_dog(request: str = "") -> Dict[str, Any]:
        return storyteller.run(request=request)

    return Tool(
        name="show_random_dog",
        description=(
            "Показать случайную фотографию собаки, распознать её породу и придумать "
            "короткую предысторию. Используй, когда пользователь просит собаку, пёсa, "
            "щенка, картинку для настроения или просто хочет отвлечься."
        ),
        parameters={
            "type": "object",
            "properties": {
                "request": {
                    "type": "string",
                    "description": "Пожелание к картинке или персонажу, например «смешной щенок».",
                }
            },
            "required": [],
        },
        function=show_random_dog,
        outputs_to_string={"source": "story"},
        outputs_to_state={
            DOG_IMAGE_STATE_KEY: {
                "source": "image_url",
                "type": List[str],
                "handler": merge_lists,
            }
        },
    )


def build_tools(chat_generator: Any) -> List[Tool]:
    """Полный набор инструментов агента."""
    return [build_bond_tool(), build_dog_tool(chat_generator)]