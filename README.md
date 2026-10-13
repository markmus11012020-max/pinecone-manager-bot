# 🤖 Bot-AI — Telegram-бот с долговременной памятью на Pinecone + LLM

Telegram-бот-помощник, который ведёт диалог, запоминает важную информацию в векторной базе **Pinecone**, извлекает её по контексту и генерирует ответы через **LLM** (провайдер [AITUNNEL](https://aitunnel.ru), OpenAI-compatible API).

Перед записью автоматически фильтрует дубликаты на основе **косинусного сходства**.

---

## ✨ Возможности

- Сохранение фактов из переписки в Pinecone с привязкой к `user_id`
- Фильтрация дубликатов / вариаций по порогу `SIMILARITY_THRESHOLD` (по умолчанию `0.85`)
- Поиск релевантных воспоминаний по тексту текущего сообщения
- Генерация естественных ответов через LLM с учётом памяти
- LLM-провайдер: **AITUNNEL** (`https://api.aitunnel.ru/v1`), модель по умолчанию `openai/gpt-5-nano`
- Изоляция памяти разных пользователей через metadata
- Поддержка прокси (`PINECONE_BASE_URL` / `proxy_url`)
- Автоматическое создание serverless-индекса при первом запуске
- Простой Windows-скрипт развёртывания (`start.bat`)

---

## 📁 Структура проекта

```text
├── config/
│   └── settings.py          # Глобальные настройки (порог сходства, токены, LLM, URL)
├── src/
│   ├── pinecone_manager.py  # Класс PineconeManager — вся работа с векторной БД
│   ├── llm_client.py        # Клиент LLM (AITUNNEL / OpenAI-compatible)
│   └── bot.py               # Telegram-бот (pyTelegramBotAPI)
├── .env.example             # Шаблон секретов
├── .env                     # Реальные ключи (не коммитить!)
├── requirements.txt
├── start.bat                # Авто-развёртывание и запуск на Windows
├── commit.bat               # Быстрый коммит
├── push.bat                 # Быстрый push
└── README.md
```

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
PINECONE_BASE_URL=                    # опционально, для прокси
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
2. Очищает кэш `__pycache__`
3. Создаёт / переиспользует `venv`
4. Устанавливает зависимости
5. Запускает `src/bot.py`

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

## 🛠 Основные команды бота

| Команда     | Описание                                      |
|-------------|-----------------------------------------------|
| `/start`    | Приветствие и краткая справка                 |
| `/memory`   | Показать сохранённые воспоминания о пользователе |
| `/reset`    | Заглушка очистки памяти (зависит от тарифа Pinecone) |

Просто пишите обычный текст — бот запомнит новые факты и ответит через LLM.

---

## 📦 Зависимости

- `pyTelegramBotAPI` — Telegram Bot API
- `pinecone` — официальный клиент Pinecone
- `sentence-transformers` + `torch` — локальные эмбеддинги
- `openai` — клиент для AITUNNEL (OpenAI-compatible)
- `python-dotenv` — загрузка `.env`
- `numpy`

Полный список — в `requirements.txt`.

---

## 🔧 Настройки (`config/settings.py`)

| Переменная              | Описание                                      | По умолчанию                    |
|-------------------------|-----------------------------------------------|---------------------------------|
| `SIMILARITY_THRESHOLD`  | Порог косинусного сходства для дубликатов     | `0.85`                          |
| `PINECONE_BASE_URL`     | URL прокси (передаётся как `proxy_url`)       | пусто                           |
| `EMBEDDING_MODEL`       | Модель sentence-transformers                  | `all-MiniLM-L6-v2`              |
| `TOP_K_MEMORIES`        | Сколько воспоминаний подтягивать для контекста| `5`                             |
| `LLM_BASE_URL`          | Base URL LLM-провайдера                       | `https://api.aitunnel.ru/v1`    |
| `LLM_API_KEY`           | API-ключ AITUNNEL                             | —                               |
| `LLM_MODEL`             | Модель по умолчанию                           | `openai/gpt-5-nano`             |
| `LLM_MAX_TOKENS`        | Макс. токенов ответа                          | `1024`                          |
| `LLM_TEMPERATURE`       | Температура генерации                         | `0.7`                           |

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

## ⚠️ Важно

- Файл `.env` **никогда** не коммитьте (он уже в `.gitignore`).
- Первый запуск скачает модель `all-MiniLM-L6-v2` (~80 МБ).
- Для работы нужен действующий API-ключ Pinecone и токен Telegram-бота.
- Удаление памяти по `user_id` (`/reset`) зависит от возможностей вашего тарифа Pinecone.

---

## 📄 Лицензия

MIT (или укажите свою).

---

Сделано с ❤️ для долговременной памяти в Telegram.
