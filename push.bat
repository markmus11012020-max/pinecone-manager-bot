@echo off
REM ============================================================================
REM Bot-AI — быстрый git push
REM ============================================================================
setlocal EnableDelayedExpansion
cd /d "%~dp0"

echo ============================================================
echo Bot-AI — Git Push
echo ============================================================
echo.

git rev-parse --is-inside-work-tree >nul 2>&1
if errorlevel 1 (
    echo ERROR: Это не git-репозиторий. Сначала запустите commit.bat
    exit /b 1
)

REM Проверяем наличие remote
git remote get-url origin >nul 2>&1
if errorlevel 1 (
    echo [WARN] Remote "origin" не настроен.
    echo.
    set /p REMOTE_URL="Введите URL репозитория (https://github.com/USER/REPO.git): "
    if "!REMOTE_URL!"=="" (
        echo ERROR: URL не указан.
        exit /b 1
    )
    git remote add origin "!REMOTE_URL!"
    echo [OK] origin добавлен: !REMOTE_URL!
    echo.
)

echo Текущая ветка:
git branch --show-current
echo.

echo Выполняю git push -u origin HEAD ...
git push -u origin HEAD
if errorlevel 1 (
    echo.
    echo ERROR: push не удался.
    echo Возможные причины:
    echo   - нет прав доступа / неверный токен
    echo   - ветка защищена
    echo   - remote URL неверный
    echo.
    echo Попробуйте вручную:
    echo   git push -u origin main
    exit /b 1
)

echo.
echo [OK] Успешно отправлено на GitHub!
echo.
endlocal
pause
