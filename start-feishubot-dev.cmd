@echo off
setlocal
cd /d "%~dp0"

set "PYTHON=%CD%\.venv\Scripts\python.exe"
set "FEISHUBOT_ENV_FILE=%CD%\.env.dev"

if not exist "%PYTHON%" (
    echo [ERROR] Python environment not found: %PYTHON%
    echo Run "uv sync --group dev" first.
    pause
    exit /b 1
)

if not exist "%FEISHUBOT_ENV_FILE%" (
    echo [ERROR] Env file not found: %FEISHUBOT_ENV_FILE%
    pause
    exit /b 1
)

echo [START] FeishuBot dev-task events (separate app, dev worktree tasks)
echo Press Ctrl+C to stop.
"%PYTHON%" -c "from feishubot.cli import main; main(['events', '--no-fallback-webhook', '--log-level', 'INFO'])"
pause
exit /b 0
