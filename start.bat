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
echo [1/5] Stopping old processes...
powershell -NoProfile -ExecutionPolicy Bypass -File "%PROJECT_DIR%stop_bot.ps1"
if errorlevel 1 (
    echo WARNING: failed to stop old instances. If the new bot reports
    echo "409 Conflict", close the old console window manually.
) else (
    echo Old processes terminated.
)
echo.

REM ---- 2. Clear cache ----
echo [2/5] Clearing bytecode / cache...
if exist "%PROJECT_DIR%__pycache__" rd /s /q "%PROJECT_DIR%__pycache__"
if exist "%PROJECT_DIR%src\__pycache__" rd /s /q "%PROJECT_DIR%src\__pycache__"
if exist "%PROJECT_DIR%config\__pycache__" rd /s /q "%PROJECT_DIR%config\__pycache__"
if exist "%PROJECT_DIR%hay\__pycache__" rd /s /q "%PROJECT_DIR%hay\__pycache__"
for /d /r "%PROJECT_DIR%" %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d"
for /r "%PROJECT_DIR%" %%f in (*.pyc) do @if exist "%%f" del /q "%%f"
if exist "%PROJECT_DIR%.pytest_cache" rd /s /q "%PROJECT_DIR%.pytest_cache"
echo Cache cleared.
echo.

REM ---- 3. Build virtual env + install deps ----
echo [3/5] Preparing virtual environment...
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

REM ---- 4. Preflight: check that the Haystack module is importable ----
echo [4/5] Checking the Haystack module (hay\)...
if exist "%PROJECT_DIR%hay\agent.py" (
    REM Текущий каталог = корень проекта, поэтому пакет hay находится в sys.path.
    "%VENV_PY%" -c "import hay.agent" >nul 2>&1
    if errorlevel 1 (
        echo WARNING: the Haystack module is not available.
        echo          /haystack will report an error until dependencies are installed.
        echo          Run: "%VENV_PY%" -m pip install -r "%PROJECT_DIR%requirements.txt"
    ) else (
        echo Haystack module OK.
    )
) else (
    echo Haystack module not present - skipping.
)
echo.

REM ---- 5. Launch Telegram Bot ----
echo [5/5] Launching Telegram Bot...
if not exist "%PROJECT_DIR%.env" (
    if exist "%PROJECT_DIR%env.example" (
        echo WARNING: .env missing, copying from env.example
        copy /y "%PROJECT_DIR%env.example" "%PROJECT_DIR%.env" >nul
    ) else (
        echo WARNING: no .env found. App will start with default placeholders.
    )
)

title Bot-AI-Telegram
"%VENV_PY%" "%PROJECT_DIR%src\bot.py"
endlocal
