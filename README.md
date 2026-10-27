# 🤖 Bot-AI — Telegram-бот с долговременной памятью на Pinecone + LLM

Telegram-бот-помощник, который ведёт диалог, запоминает важную информацию в векторной базе **Pinecone**, извлекает её по контексту и генерирует ответы через **LLM** (провайдер [AITUNNEL](https://aitunnel.ru), OpenAI-compatible API).

Перед записью автоматически фильтрует дубликаты на основе **косинусного сходства**.

В репозитории две независимые точки входа:

- `src/bot.py` — основной бот с обычным режимом и режимом `/haystack`;
- `hay_v2_bot/main.py` — вторая версия: отдельный RAG-бот на **Haystack + Docling + Pinecone**.

---

## ✨ Возможности

### Стандартный режим
- Сохранение фактов из переписки в Pinecone с привязкой к `user_id`
- Фильтрация дубликатов / вариаций по порогу `SIMILARITY_THRESHOLD` (по умолчанию `0.85`)
- Поиск релевантных воспоминаний по тексту текущего сообщения
- Генерация естественных ответов через LLM с учётом памяти
- Изоляция памяти разных пользователей через metadata

### Режим Haystack (`/haystack`)
- Умный персональный помощник на фреймворке **[Haystack](https://haystack.deepset.ai/)**
- Пайплайн из трёх компонентов: память → промпт → агент
- **Instrument calling:** агент сам решает, когда вызвать внешний инструмент
  - `bond_market_analysis` — бесплатные данные **MOEX ISS** (доходность, цена, купон, даты
    купонного платежа и погашения) + новостные события из **RSS Интерфакса**
  - `show_random_dog` — случайная фотография из **Dog API**, распознавание породы
    и генерация предыстории моделью из `.env` (мультимодальный запрос)
- Долговременная память через `PineconeDocumentStore` со **строгой изоляцией по `user_id`**
- Детальное логирование каждого шага сохранения данных в консоль

### Вторая версия (`hay_v2_bot`)
- Отдельная точка входа — не затрагивает основной бот и модуль `hay/`
- Приём **PDF, DOCX, PPTX, XLSX, HTML, Markdown, AsciiDoc, CSV**
- Разбор документов через **[Docling](https://docling-project.github.io/docling/)**: структура,
  таблицы, заголовки и смысловые чанки через `HybridChunker`
- OCR сканов (RapidOCR) с аварийными стратегиями: `force_backend_text` для
  «упрямых» PDF и полностраничный OCR, если авто-режим не нашёл текст
- Каждый чанк хранится в Pinecone с метаданными `user_id`, `file_name`, `page_number`
  (+ `chunk_index`, `section`, `source`)
- Текстовые вопросы → RAG-поиск **строго по документам текущего `user_id`** → ответ LLM
- После загрузки файла бот присылает краткое резюме ровно одним предложением
- Общая модель эмбеддингов для индексации и поиска, все LLM-вызовы идут через `LLM_BASE_URL`
- Подробные логи всех этапов: Docling → эмбеддинги → запись в Pinecone → поиск → LLM

### Общее
- LLM-провайдер: **AITUNNEL** (`https://api.aitunnel.ru/v1`), модель по умолчанию `openai/gpt-5-nano`
- Поддержка HTTP/HTTPS-прокси для Pinecone (`PINECONE_PROXY_URL`)
- Автоматическое создание serverless-индекса при первом запуске
- Простой Windows-скрипт развёртывания (`start.bat`)

---

## 📁 Структура проекта

```text
├── config/
│   └── settings.py          # Глобальные настройки (порог сходства, токены, LLM, URL)
├── hay/                     # ★ Изолированный модуль умного помощника на Haystack
│   ├── __init__.py          # Ленивый импорт: не тормозит старт стандартного режима
│   ├── document_store.py    # PineconeDocumentStore + изоляция по user_id + логирование записи
│   ├── tools.py             # Кастомные @component: облигации РФ и мультимодальные «собаки»
│   └── agent.py             # Сборка пайплайна Haystack и агента, детальный логи run()
├── hay_v2_bot/              # ★ Вторая версия: Haystack + Docling + Pinecone (RAG)
│   ├── main.py              # Точка входа, --check и локальный разбор --inspect
│   ├── components/
│   │   ├── docling_converter.py  # Docling → смысловые чанки + метаданные
│   │   └── embeddings.py         # sentence-transformers для чанков и запросов
│   ├── pipelines/
│   │   ├── indexing.py           # файл → Docling → эмбеддинги → Pinecone
│   │   └── generation.py         # вопрос → поиск по user_id → промпт → LLM
│   └── bot/
│       ├── handlers.py           # команды, текст и документы Telegram
│       └── telegram_bot.py       # polling + single-instance mutex
├── src/
│   ├── pinecone_manager.py  # Класс PineconeManager — вся работа с векторной БД
│   ├── llm_client.py        # Клиент LLM (AITUNNEL / OpenAI-compatible)
│   └── bot.py               # Telegram-бот (pyTelegramBotAPI) + маршрутизация режимов
├── env.example              # Шаблон секретов
├── .env                     # Реальные ключи (не коммитить!)
├── requirements.txt
├── start.bat                # Авто-развёртывание и запуск v1 на Windows
├── commit.bat               # Быстрый коммит
├── push.bat                 # Быстрый push
└── README.md
```

> Папка `hay/` полностью изолирована от существующей архитектуры: обычный режим бота
> работает ровно как раньше, а весь код Haystack живёт только в `hay/`.

> Папка `hay_v2_bot/` — полностью отдельная вторая версия бота. Она использует те же
> `.env` и `config/settings.py`, но не изменяет код `src/` и `hay/`.

---

## 🚀 Быстрый старт

### 1. Клонирование и подготовка

```bash
git clone <your-repo-url>
cd <project-folder>
```

### 2. Настройка окружения

Скопируйте `.env.example` → `.env` и заполните:

```env
TELEGRAM_BOT_TOKEN=ваш_токен_от_BotFather
PINECONE_API_KEY=ваш_ключ_Pinecone
PINECONE_ENVIRONMENT=us-east-1
PINECONE_INDEX_NAME=telegram-bot-memory
# PINECONE_PROXY_URL=http://user:pass@proxy.example.com:8080    # опционально: HTTP-прокси
SIMILARITY_THRESHOLD=0.85
EMBEDDING_MODEL=all-MiniLM-L6-v2

# LLM (AITUNNEL)
LLM_BASE_URL=https://api.aitunnel.ru/v1
LLM_API_KEY=sk-aitunnel-ваш_ключ
LLM_MODEL=openai/gpt-5-nano
LLM_MAX_TOKENS=1024
LLM_TEMPERATURE=0.7
```

### 3. Запуск (Windows)

Просто запустите:

```bat
start.bat
```

Скрипт:
1. Останавливает старые процессы бота
2. Очищает кэш `__pycache__` (включая `hay\`)
3. Создаёт / переиспользует `venv`
4. Проверяет зависимости **без сети** и, если чего-то не хватает, ставит из `requirements.txt`; затем проверяет, что модуль `hay` импортируется
5. Запускает `src/bot.py`

Если venv уже собран, шаг 4 **не обращается к PyPI вообще** — запуск работает даже
при недоступном или зависающем соединении. Флаги:

| Команда | Что делает |
|---|---|
| `start.bat` | обычный запуск (без сети, если venv готов) |
| `start.bat --reinstall` | принудительно прогнать `pip install -r requirements.txt` |
| `start.bat --upgrade-pip` | обновить сам pip перед установкой зависимостей |

> Проверка смотрит на **наличие** пакетов из `requirements.txt`, а не на их версии.
> Если пакет установлен, но устарел, обновите его через `--reinstall`.

### 4. Ручной запуск (любая ОС)

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
python src/bot.py
```

### 5. Запуск второй версии (`hay_v2_bot`)

Вторая версия запускается отдельно и использует тот же `.env`:

```powershell
# из корня проекта
venv\Scripts\python.exe hay_v2_bot\main.py
```

Проверка зависимостей и настроек без подключения к Telegram и Pinecone:

```powershell
venv\Scripts\python.exe hay_v2_bot\main.py --check
```

Проверка только соединения с Telegram Bot API:

```powershell
venv\Scripts\python.exe hay_v2_bot\main.py --check-telegram
```

Локальный разбор одного файла (без Telegram и Pinecone) — удобно, если бот пишет,
что не нашёл в документе текст:

```powershell
venv\Scripts\python.exe hay_v2_bot\main.py --inspect "C:\путь\к\статье.pdf"
```

> `start.bat` по-прежнему запускает первую версию (`src\bot.py`). Так как обе версии
> используют один `TELEGRAM_BOT_TOKEN`, одновременно они работать не могут: сначала
> остановите v1, затем запустите v2 командой выше. Защита от второго экземпляра
> встроена в обе версии.

---

## 🧠 Как работает память + LLM

1. Пользователь присылает текст.
2. Бот разбивает сообщение на потенциальные факты.
3. Для каждого факта:
   - строится эмбеддинг (`sentence-transformers` / `all-MiniLM-L6-v2`)
   - выполняется поиск топ-1 похожего вектора **того же пользователя**
   - если `cosine similarity >= SIMILARITY_THRESHOLD` → дубликат (запись пропускается)
   - иначе → новый вектор записывается в Pinecone с `user_id` в metadata
4. Бот извлекает релевантный контекст из Pinecone.
5. Контекст + сообщение пользователя отправляются в LLM (AITUNNEL → `openai/gpt-5-nano`).
6. LLM генерирует естественный ответ с учётом памяти.

---

## 🧠 Режим Haystack основного бота (`/haystack`)

> Это режим **первой** версии (`src/bot.py`). Вторая версия `hay_v2_bot` — отдельный
> процесс, описанный ниже.

### Команды

| Команда        | Описание                                                  |
|----------------|-----------------------------------------------------------|
| `/haystack`    | Включить умного помощника на Haystack для себя            |
| `/haystackoff` | Вернуться в обычный режим (алиасы: `/standard`, `/haystack_off`) |
| `/haystatus`   | Показать текущий режим и состояние агента                |

Выбранный режим хранится в оперативной памяти процесса бота и действует до перезапуска.

### Схема пайплайна

```text
┌──────────────────────┐   context / query   ┌─────────────────────┐   prompt   ┌──────────────┐
│ memory               │ ──────────────────▶ │ prompt              │ ──────────▶ │ agent        │
│ PineconeUserMemory   │                     │ ChatPromptBuilder   │             │ Agent        │
│ (hay/document_store) │                     │                     │             │ + 2 tools    │
└──────────────────────┘                     └─────────────────────┘             └──────────────┘
        │
        │ пользовательский текст → эмбеддинг all-MiniLM-L6-v2
        ▼
   Pinecone (serverless, cosine, 384d)  ──►  фильтр {"user_id": {"$eq": "<telegram_id>"}}
```

1. `memory` извлекает факты из сообщения, проверяет каждый на дубликат (косинус ≥
   `SIMILARITY_THRESHOLD`) и сохраняет новые с `meta.user_id`.
2. Затем `memory` подтягивает до `TOP_K_MEMORIES` релевантных документов — **только**
   с фильтром по `user_id`, поэтому контекст одного пользователя физически недоступен
   другому.
3. `prompt` склеивает память и вопрос в сообщения для агента.
4. `agent` (Haystack `Agent`) вызывает LLM через `LLM_BASE_URL` и при необходимости
   дёргает инструменты.

### Инструменты

| Инструмент             | Что делает                                                                                | Источник данных                    |
|------------------------|-------------------------------------------------------------------------------------------|------------------------------------|
| `bond_market_analysis` | Доходность, цена, купон, дата ближайшего платежа, дата погашения, объём выпуска + новости   | MOEX ISS (`iss.moex.com`), RSS Интерфакса |
| `show_random_dog`      | Случайное фото, распознавание породы, генерация предыстории (мультимодальный запрос)      | Dog API (`dog.ceo`), модель из `.env` |

Оба инструмента — кастомные компоненты Haystack (`@component`), обёрнутые в `Tool`.
URL фотографии передаётся в состояние агента (`outputs_to_state`), а не парсится из
текста, поэтому Telegram всегда получает картинку, даже если агент переписал ответ.

### Что нужно для режима Haystack

Ничего сверх `.env` — новых переменных не добавлялось. Модуль читает те же
`PINECONE_*`, `EMBEDDING_MODEL`, `SIMILARITY_THRESHOLD`, `TOP_K_MEMORIES` и
`LLM_BASE_URL` / `LLM_MODEL` / `LLM_API_KEY`.

Агент создаётся **лениво** при первом `/haystack`: обычный режим от этого не
замедляется, а загрузка `haystack` + `sentence-transformers` не происходит зря.

### Логирование

`run()` и компонент памяти подробно пишут в консоль `start.bat` (без отдельных
`.log`-файлов): каждый шаг сохранения, сходство дубликата, найденный контекст,
вызванные инструменты, число шагов агента и потребление токенов. При сбое записи
в Pinecone печатается `user_id`, индекс, namespace, id документа и сама ошибка.

---

## 📄 Вторая версия: `hay_v2_bot` (Docling + RAG)

`hay_v2_bot` — самостоятельный Telegram-бот, а не режим основного бота. Он не
включается командами вроде `/haystack`, а запускается отдельным процессом из
`hay_v2_bot/main.py`. Настройки берутся из того же `.env`; обязательных новых
переменных не требуется (опциональные `TELEGRAM_PROXY_URL` / `TELEGRAM_API_URL`
общие для обеих версий).

### Как работает индексация

```text
Файл (PDF / DOCX / PPTX / XLSX / HTML / MD / CSV ...)
        │
        ▼
DoclingDocumentConverter  (DocumentConverter + HybridChunker)
        │  смысловые чанки + meta: user_id, file_name, page_number
        ▼
SentenceTransformerDocumentEmbedder
        │  нормализованные векторы (all-MiniLM-L6-v2, 384d)
        ▼
PineconeDocumentWriter  →  Pinecone (serverless, cosine)
```

Повторная загрузка того же файла обновляет чанки с теми же детерминированными id.

### Как работает ответ на вопрос

```text
Текст пользователя
        │
        ▼
SentenceTransformerTextEmbedder
        │
        ▼
PineconeEmbeddingRetriever
        │  фильтр: {"user_id": {"$eq": "<telegram_id>"}}
        ▼
ChatPromptBuilder  →  OpenAIChatGenerator (LLM_BASE_URL)
        │
        ▼
Ответ в Telegram
```

### Изоляция и метаданные

- Каждый чанк обязательно содержит `user_id`, `file_name`, `page_number`
  (дополнительно: `chunk_index`, `section`, `source`, `embedding_model`).
- Поиск всегда выполняется с фильтром по `user_id` текущего пользователя, поэтому
  документы других людей не попадают в контекст LLM.
- Используется тот же индекс Pinecone и namespace по умолчанию, что и у основного
  бота; разделение обеспечивается метаданными.

### Ответы при загрузке файла

Бот отправляет ровно два сообщения:

1. `Файл получен. Запускаю анализ и сохранение. Это может занять немного времени...`
2. `Готово. Я изучил этот файл, теперь можем его обсудить.` + краткое резюме
   ровно одним предложением.

Если в файле нет ни текстового слоя, ни распознаваемого OCR-текста (например,
пустой документ или скан очень низкого качества), бот сообщит об этом отдельно.

Перед этим Docling перебирает три стратегии: обычный разбор (layout + авто-OCR),
принудительное чтение нативного текстового слоя (`force_backend_text`) и
полностраничный OCR. Если чанкер не смог нарезать уже извлечённый текст, он
сохраняется плоскими фрагментами — файл не теряется.

### Команды

| Команда | Описание |
|---|---|
| `/start`, `/help` | Приветствие и справка по второй версии |
| `/status` | Модель, глубина поиска и количество фрагментов в индексе |

Любой обычный текст уходит в RAG-пайплайн, документы — в Docling и Pinecone.

### Логирование

Каждый этап пишется в консоль: конвертация Docling (включая выбранную стратегию,
status, число страниц, текстовых блоков, таблиц и символов), число чанков, батчи
эмбеддингов, запись в Pinecone (индекс, namespace, id, размерность, ключи
метаданных), параметры фильтра поиска, найденные фрагменты и ответ LLM. При
ошибке записи в Pinecone логируется точное место сбоя и причина.

### Ограничения

- Telegram Bot API отдаёт ботам файлы до **20 МБ**.
- Обе версии используют один `TELEGRAM_BOT_TOKEN`, поэтому одновременно может
  работать только одна. В каждой версии есть защита от второго экземпляра.
- Первый разбор документа может быть долгим: Docling загружает/инициализирует
  модели, а эмбеддинги считаются локально.
- `start.bat` запускает v1. Для v2 используйте команду из раздела
  «Запуск второй версии».

---

## 🛠 Основные команды бота

| Команда        | Описание                                                 |
|----------------|----------------------------------------------------------|
| `/start`       | Приветствие и краткая справка                            |
| `/memory`      | Показать сохранённые воспоминания о пользователе          |
| `/reset`       | Заглушка очистки памяти (зависит от тарифа Pinecone)      |
| `/haystack`    | Включить умного помощника на Haystack                     |
| `/haystackoff` | Вернуться в обычный режим                                 |
| `/haystatus`   | Текущий режим и состояние агента                          |

Просто пишите обычный текст — бот запомнит новые факты и ответит через LLM.

---

## 📦 Зависимости

- `pyTelegramBotAPI` — Telegram Bot API (нужна версия **4.36+**: в ней
  `download_file()` возвращает `bytes`, а не пишет файл в каталог)
- `pinecone` — официальный клиент Pinecone
- `sentence-transformers` + `torch` — локальные эмбеддинги
- `openai` — клиент для AITUNNEL (OpenAI-compatible)
- `python-dotenv` — загрузка `.env`
- `requests` — обращения к бесплатным внешним API (MOEX ISS, Интерфакс, Dog API)
- `haystack-ai` — фреймворк умного помощника (пайплайны, агенты, инструменты)
- `pinecone-haystack` — интеграция Haystack с Pinecone (`PineconeDocumentStore`)
- `docling` — разбор PDF, DOCX, PPTX, XLSX, HTML, Markdown и CSV на смысловые чанки для `hay_v2_bot`
- `PySocks` — поддержка SOCKS5-прокси для Telegram Bot API
- `numpy`

OCR для сканов идёт через `rapidocr`, который `docling` ставит вместе со
стандартными зависимостями. Отдельно ставить EasyOCR/Tesseract не нужно.

Полный список — в `requirements.txt`.

---

## 🔧 Настройки (`config/settings.py`)

| Переменная              | Описание                                      | По умолчанию                    |
|-------------------------|-----------------------------------------------|---------------------------------|
| `TELEGRAM_PROXY_URL`    | HTTP/HTTPS/SOCKS5-прокси для Telegram Bot API | пусто                           |
| `TELEGRAM_API_URL`      | Альтернативный Bot API URL (reverse proxy)    | пусто                           |
| `SIMILARITY_THRESHOLD`  | Порог косинусного сходства для дубликатов     | `0.85`                          |
| `PINECONE_PROXY_URL`    | URL HTTP/HTTPS-прокси для Pinecone SDK        | пусто                           |
| `EMBEDDING_MODEL`       | Модель sentence-transformers                  | `all-MiniLM-L6-v2`              |
| `TOP_K_MEMORIES`        | Сколько воспоминаний подтягивать для контекста| `5`                             |
| `LLM_BASE_URL`          | Base URL LLM-провайдера                       | `https://api.aitunnel.ru/v1`    |
| `LLM_API_KEY`           | API-ключ AITUNNEL                             | —                               |
| `LLM_MODEL`             | Модель по умолчанию                           | `openai/gpt-5-nano`             |
| `LLM_MAX_TOKENS`        | Макс. токенов ответа                          | `1024`                          |
| `LLM_TEMPERATURE`       | Температура генерации                         | `0.7`                           |

Вторая версия `hay_v2_bot` использует эти же переменные — отдельных настроек и
новых ключей `.env` для неё нет. Глубина поиска v2 также берётся из
`TOP_K_MEMORIES`.

---

## 📤 Git: коммит и пуш

Для удобства добавлены Windows-скрипты:

```bat
# Сделать коммит (спросит сообщение)
commit.bat

# Запушить текущую ветку
push.bat

# Или одной командой (после commit.bat)
push.bat
```

Перед первым пушем убедитесь, что remote настроен:

```bash
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git branch -M main
git push -u origin main
```

---

## 🩺 Если Telegram API недоступен

Симптомы в консоли:

```text
ERROR | TeleBot | Infinity polling exception:
HTTPSConnectionPool(host='api.telegram.org', port=443): Max retries exceeded
(Caused by ConnectTimeoutError ...)
```

Это **не ошибка кода**: провайдер или firewall блокирует прямой доступ к
`api.telegram.org`. Pinecone и LLM при этом могут работать нормально.

Варианты решения:

1. **VPN** — включите VPN, который маршрутизирует весь системный трафик, а не
   только приложение Telegram.
2. **Прокси для Bot API** — добавьте в `.env`:

   ```env
   # HTTP/HTTPS-прокси
   TELEGRAM_PROXY_URL=http://user:pass@host:port

   # или SOCKS5 (нужен пакет PySocks, он есть в requirements.txt)
   TELEGRAM_PROXY_URL=socks5h://host:port
   ```

3. **Свой reverse-proxy Bot API** — если у вас есть прокси-сервер для Telegram
   Bot API:

   ```env
   TELEGRAM_API_URL=https://your-bot-api-proxy.example.com
   ```

После изменения `.env` перезапустите бота и проверьте соединение:

```powershell
venv\Scripts\python.exe hay_v2_bot\main.py --check-telegram
```

Прокси из настроек Telegram Desktop не используется автоматически: укажите его
URL в `TELEGRAM_PROXY_URL`.

---

## 🩺 Если документ не распознаётся

### `Telebot.download_file() takes 2 positional arguments but 3 were given`

В `pyTelegramBotAPI 4.36` изменился API: метод `download_file(file_path)`
возвращает `bytes` и больше не принимает каталог назначения. Обновите пакет:

```powershell
venv\Scripts\python.exe -m pip install -U "pyTelegramBotAPI>=4.36"
```

Код `hay_v2_bot` уже рассчитан на новый API и сам записывает полученные байты
на диск.

### «В файле не найден текст для сохранения»

`hay_v2_bot` перебирает три стратегии, и в консоли видно, какая сработала:

```text
[DOCLING] Стратегия 'обычный режим (layout + авто-OCR)' | status=SUCCESS | страниц=1 | ...
[DOCLING] Стратегия 'нативный текстовый слой (force_backend_text)' | ...
[DOCLING] Стратегия 'полностраничный OCR' | ...
```

Проверить конкретный файл можно без Telegram и Pinecone:

```powershell
venv\Scripts\python.exe hay_v2_bot\main.py --inspect "C:\путь\к\статье.pdf"
```

Команда покажет `status` конвертации, количество страниц, текстовых блоков,
таблиц и символов, а также пример извлечённого текста.

Частые причины:

- **скан без текстового слоя** — сработает полностраничный OCR; на CPU это
  может занять десятки секунд на страницу;
- **сложный layout/нестандартные шрифты** — помогает стратегия
  `force_backend_text`;
- **пустой или повреждённый файл** — текста нет ни в одной стратегии, бот
  сообщит об этом и предложит прислать файл лучшего качества.

### Предупреждения huggingface_hub про symlinks

В Windows без режима разработчика `huggingface_hub` не может создавать
symlink’и в кэше моделей. На работу это не влияет, кэш просто занимает больше
места. Предупреждение можно отключить:

```powershell
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
```

---

## ⚠️ Важно

- Файл `.env` **никогда** не коммитьте (он уже в `.gitignore`).
- Первый запуск скачает модель `all-MiniLM-L6-v2` (~80 МБ).
- Для работы нужен действующий API-ключ Pinecone и токен Telegram-бота.
- Удаление памяти по `user_id` (`/reset`) зависит от возможностей вашего тарифа Pinecone.
- `start.bat` запускает **первую** версию (`src/bot.py`). Вторая версия
  (`hay_v2_bot`) запускается вручную: `venv\Scripts\python.exe hay_v2_bot\main.py`.
- Обе версии используют один `TELEGRAM_BOT_TOKEN`, поэтому одновременно должен
  работать только один процесс. Защита от второго экземпляра уже встроена.
- Первый разбор файла в `hay_v2_bot` может занять больше времени: Docling
  инициализирует модели, а эмбеддинги считаются локально.
- Сканы обрабатываются через OCR на CPU: ориентировочно 30–60 секунд на
  страницу (зависит от разрешения и загрузки машины).
- Первый запуск Docling скачивает модели layout/OCR в
  `%USERPROFILE%\.cache\huggingface` (несколько сотен мегабайт).

---

## 📄 Лицензия

MIT (или укажите свою).

---

Сделано с ❤️ для долговременной памяти в Telegram.
