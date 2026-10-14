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
REM The venv is reused across launches, so pip is only contacted when something
REM is actually missing. That keeps start.bat working on a flaky/offline link:
REM a PyPI timeout must not block a launch when every dependency is installed.
REM Optional flags: start.bat --reinstall (force pip run), start.bat --upgrade-pip
echo [3/5] Preparing virtual environment...

set "FORCE_INSTALL="
set "UPGRADE_PIP="
for %%a in (%*) do (
    if /i "%%~a"=="--reinstall" set "FORCE_INSTALL=1"
    if /i "%%~a"=="--upgrade-pip" set "UPGRADE_PIP=1"
)

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

REM Offline dependency check - never touches the network.
REM Checks every distribution listed in requirements.txt against the ones already
REM installed, by distribution name (not import name: pyTelegramBotAPI imports as
REM telebot, pinecone-haystack as haystack_integrations.components.pinecone).
REM If a package is present but too old, re-run with start.bat --reinstall.
set "DEPCHK=import sys,importlib.metadata as md,re; have={d.metadata['Name'].lower().replace('_','-') for d in md.distributions() if d.metadata['Name']}; reqs=[l.split('#')[0].strip() for l in open(r'%PROJECT_DIR%requirements.txt',encoding='utf-8') if l.split('#')[0].strip()]; miss=[r for r in reqs if re.split(r'[<>=~,;]',r)[0].strip().lower().replace('_','-') not in have]; print('Dependency check: all packages present' if not miss else 'Dependency check: missing ' + ', '.join(miss)); sys.exit(1 if miss else 0)"

if defined UPGRADE_PIP goto :upgrade_pip

if exist "%PROJECT_DIR%requirements.txt" goto :check_deps
echo WARNING: requirements.txt not found.
goto :deps_ready

:upgrade_pip
echo Upgrading pip...
"%VENV_PY%" -m pip install --upgrade pip --disable-pip-version-check --retries 2 --timeout 60 --quiet

:check_deps
"%VENV_PY%" -c "%DEPCHK%"
if errorlevel 1 goto :install_deps
if defined FORCE_INSTALL goto :install_deps
echo All dependencies present - skipping pip install.
goto :deps_ready

:install_deps
echo Installing dependencies from requirements.txt...
"%VENV_PIP%" install -r "%PROJECT_DIR%requirements.txt" --disable-pip-version-check --retries 2 --timeout 60 --quiet
if errorlevel 1 goto :install_failed
goto :deps_ready

:install_failed
echo WARNING: dependency installation failed (no network or PyPI unreachable).
"%VENV_PY%" -c "%DEPCHK%"
if not errorlevel 1 goto :deps_ready
echo ERROR: required dependencies are missing and could not be installed.
echo        Run this once the connection is back:
echo        "%VENV_PY%" -m pip install -r "%PROJECT_DIR%requirements.txt"
exit /b 1

:deps_ready
echo Dependencies ready.
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
