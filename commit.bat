@echo off
REM ============================================================================
REM Bot-AI — быстрый git commit
REM ============================================================================
setlocal EnableDelayedExpansion
cd /d "%~dp0"

echo ============================================================
echo Bot-AI — Git Commit
echo ============================================================
echo.

REM Проверяем, что это git-репозиторий
git rev-parse --is-inside-work-tree >nul 2>&1
if errorlevel 1 (
    echo [INFO] Git-репозиторий не найден. Инициализирую...
    git init
    git branch -M main
    echo [OK] git init выполнен.
    echo.
)

echo [1/3] git status
echo ------------------------------------------------------------
git status -sb
echo.

echo [2/3] Добавляю все изменения (git add -A)...
git add -A
if errorlevel 1 (
    echo ERROR: git add failed.
    exit /b 1
)
echo.

set /p COMMIT_MSG="[3/3] Введите сообщение коммита: "
if "!COMMIT_MSG!"=="" (
    echo ERROR: сообщение коммита не может быть пустым.
    exit /b 1
)

git commit -m "!COMMIT_MSG!"
if errorlevel 1 (
    echo.
    echo [WARN] Коммит не создан (возможно, нет изменений).
) else (
    echo.
    echo [OK] Коммит успешно создан.
)

echo.
echo Готово. Для отправки на GitHub запустите: push.bat
echo.
endlocal
pause
