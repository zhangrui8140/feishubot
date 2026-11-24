@echo off
setlocal
cd /d "%~dp0"

set "PYTHON=%CD%\.venv\Scripts\python.exe"
set "CONFIG_URL=http://127.0.0.1:8000/admin/config"

if not exist "%PYTHON%" (
    echo [ERROR] Python environment not found: %PYTHON%
    echo Run "uv sync --group dev" first.
    pause
    exit /b 1
)

powershell.exe -NoProfile -Command "if (Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue) { exit 0 } else { exit 1 }"
if errorlevel 1 (
    echo [START] FeishuBot gateway
    start "FeishuBot Gateway" "%PYTHON%" -m uvicorn feishubot.main:app --host 127.0.0.1 --port 8000
) else (
    echo [READY] FeishuBot gateway is already listening on port 8000
)

powershell.exe -NoProfile -Command "$processes = Get-CimInstance Win32_Process | Where-Object { $_.Name -in @('python.exe', 'uv.exe', 'feishubot.exe') -and $_.CommandLine -and (($_.CommandLine -match 'feishubot(\.exe)?\s+events') -or (($_.CommandLine -match 'feishubot\.cli') -and ($_.CommandLine -match 'events'))) }; if ($processes) { exit 0 } else { exit 1 }"
if errorlevel 1 (
    echo [START] 张锐问题助手 events
    start "张锐问题助手" "%PYTHON%" -c "from feishubot.cli import main; main(['events', '--no-fallback-webhook', '--log-level', 'WARNING'])"
) else (
    echo [READY] 张锐问题助手已在运行
)

powershell.exe -NoProfile -Command "$deadline = (Get-Date).AddSeconds(60); do { if (Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue) { exit 0 }; Start-Sleep -Milliseconds 250 } while ((Get-Date) -lt $deadline); exit 1"
if errorlevel 1 (
    echo [ERROR] Gateway did not become ready within 60 seconds.
    pause
    exit /b 1
)

if /I "%~1"=="--no-open" (
    echo [READY] %CONFIG_URL%
) else (
    echo [OPEN] %CONFIG_URL%
    start "" "%CONFIG_URL%"
)
exit /b 0
