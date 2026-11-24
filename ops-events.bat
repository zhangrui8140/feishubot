@echo off
cd /d "%~dp0"
if not exist logs mkdir logs
.venv\Scripts\feishubot.exe events --no-fallback-webhook --log-level WARNING >> logs\events.log 2>&1
