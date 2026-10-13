# Техническое задание (ТЗ) на разработку Telegram-бота со слотом долговременной памяти на базе Pinecone

## 1. Общие сведения о проекте
Цель проекта — создание Telegram-бота-помощника, способного вести диалог с пользователем, запоминать новую информацию в векторную базу данных Pinecone и извлекать её по контексту. Бот должен фильтровать дубликаты информации на основе косинусного сходства перед записью в хранилище.

## 2. Архитектура и структура файлов
Проект должен содержать следующую структуру каталогов и файлов:
```text
├── config/
│   └── settings.py       # Глобальные настройки и конфигурация
├── src/
│   ├── pinecone_manager.py # Модуль управления векторной БД
│   └── bot.py            # Главный модуль Telegram-бота
├── .env                  # Файл с секретами (токены, ключи API)
├── requirements.txt      # Зависимости проекта
└── start.bat             # Скрипт автоматического развертывания для Windows
```

---

## 3. Требования к компонентам системы

### 3.1. Модуль конфигурации (`config/settings.py`)
Должен содержать глобальные переменные и настройки проекта.
* **`SIMILARITY_THRESHOLD`**: Переключатель границы косинусного сходства (тип: `float`, например, `0.85`). Значения выше порога считаются дубликатом/вариацией (пропускаются или обновляются), значения ниже — новой информацией (записываются).
* **Настройки прокси и API**: Поддержка считывания `PINECONE_BASE_URL` для корректной работы через прокси-серверы.

### 3.2. Векторный менеджер (`src/pinecone_manager.py`)
Класс `PineconeManager` должен инкапсулировать всю логику взаимодействия с базой данных Pinecone.

#### Требования к реализации:
* **Инициализация**: Принимает параметры подключения (`api_key`, `environment`, `index_name`). Обязательно учитывать параметр `base_url` в конфигурации клиента API Pinecone для поддержки прокси.
* **Функции записи**:
  * Запись сырых документов (с автоматической векторизацией на стороне менеджера или встроенной модели эмбеддингов).
  * Запись подготовленных векторов напрямую.
  * Метаданные: Каждый вектор должен быть привязан к `user_id` Telegram для изоляции контекста пользователей.
* **Функции чтения**:
  * Чтение по вектору: Поиск топ-N схожих фрагментов по готовому вектору.
  * Чтение по тексту: Автоматическая конвертация текста в эмбеддинг и последующий поиск в Pinecone.

### 3.3. Логика фильтрации памяти (Косинусное сходство)
Перед отправкой данных на запись в Pinecone:
1. Выполняется текстовый/векторный поиск по текущему контексту сообщения пользователя.
2. Извлекается наиболее похожий вектор (или топ-1 совпадение).
3. Проверяется метрика косинусного сходства:
   * **`Сходство < SIMILARITY_THRESHOLD`**: Новая информация. Вызывается метод записи нового вектора в Pinecone.
   * **`Сходство >= SIMILARITY_THRESHOLD`**: Дубликат или вариация. Система игнорирует запись или обновляет существующий слот памяти (в зависимости от настроек бизнес-логики).

### 3.4. Telegram-бот (`src/bot.py`)
Разрабатывается на базе библиотеки **pyTelegramBotAPI**.
* **Основные функции**:
  * Обработка команды `/start` — инициализация сессии пользователя.
  * Перехват всех текстовых сообщений пользователя.
  * Интеграция с `PineconeManager` для сохранения важных фактов из переписки (с учетом фильтрации дубликатов).
  * Извлечение релевантного контекста из Pinecone по тексту текущего сообщения для генерации контекстного ответа (при интеграции с LLM).

---

## 4. Скрипт развертывания (`start.bat`)
Автоматический Windows-скрипт для остановки старых процессов, очистки кэша, сборки окружения и запуска бота.

```batch
@echo off
REM ============================================================================
REM Bot-AI — Windows deploy script
REM Pipeline: stop old processes -> clear cache -> build env -> install deps
REM ============================================================================
setlocal EnableDelayedExpansion
set "PROJECT_DIR=%~dp0"
cd /d "%PROJECT_DIR%"
set "VENV_DIR=%PROJECT_DIR%venv"
set "PYTHON=python"

echo ============================================================
echo Bot-AI — Windows auto-deploy
echo Project dir: %PROJECT_DIR%
echo ============================================================
echo.

REM ---- 1. Stop old processes ----
echo [1/4] Stopping old processes...
taskkill /F /IM python.exe /FI "WINDOWTITLE eq Bot-AI-Telegram*" /T >nul 2>&1
if errorlevel 1 (
    echo No old processes found.
) else (
    echo Old processes terminated.
)
echo.

REM ---- 2. Clear cache ----
echo [2/4] Clearing bytecode / cache...
if exist "%PROJECT_DIR%__pycache__" rd /s /q "%PROJECT_DIR%__pycache__"
if exist "%PROJECT_DIR%src\__pycache__" rd /s /q "%PROJECT_DIR%src\__pycache__"
if exist "%PROJECT_DIR%config\__pycache__" rd /s /q "%PROJECT_DIR%config\__pycache__"
for /d /r "%PROJECT_DIR%" %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d"
for /r "%PROJECT_DIR%" %%f in (*.pyc) do @if exist "%%f" del /q "%%f"
if exist "%PROJECT_DIR%.pytest_cache" rd /s /q "%PROJECT_DIR%.pytest_cache"
echo Cache cleared.
echo.

REM ---- 3. Build virtual env + install deps ----
echo [3/4] Preparing virtual environment...
if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo Creating venv...
    "%PYTHON%" -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo ERROR: failed to create venv. Ensure Python 3.10+ is on PATH.
        exit /b 1
    )
) else (
    echo Reusing existing venv.
)

set "VENV_PY=%VENV_DIR%\Scripts\python.exe"
set "VENV_PIP=%VENV_DIR%\Scripts\pip.exe"

echo Upgrading pip...
"%VENV_PY%" -m pip install --upgrade pip --quiet

if exist "%PROJECT_DIR%requirements.txt" (
    echo Installing dependencies from requirements.txt...
    "%VENV_PIP%" install -r "%PROJECT_DIR%requirements.txt" --quiet
    if errorlevel 1 (
        echo ERROR: dependency installation failed.
        exit /b 1
    )
) else (
    echo WARNING: requirements.txt not found.
)
echo Dependencies installed.
echo.

REM ---- 4. Launch Telegram Bot ----
echo [4/4] Launching Telegram Bot...
if not exist "%PROJECT_DIR%.env" (
    if exist "%PROJECT_DIR%.env.example" (
        echo WARNING: .env missing, copying from .env.example
        copy /y "%PROJECT_DIR%.env.example" "%PROJECT_DIR%.env" >nul
    ) else (
        echo WARNING: no .env found. App will start with default placeholders.
    )
)

title Bot-AI-Telegram
"%VENV_PY%" "%PROJECT_DIR%srcot.py"
endlocal
```

---

## 5. Стек технологий и зависимости (`requirements.txt`)
* `pyTelegramBotAPI` — библиотека для работы с Telegram Bot API.
* `pinecone-client` — официальный клиент для работы с векторной базой данных Pinecone.
* `sentence-transformers` или `openai` — для генерации эмбеддингов (текстовых векторов).
* `python-dotenv` — для менеджмента переменных окружения из файла `.env`.
