@echo off
setlocal
cd /d "%~dp0"

set "PAUSE_ON_EXIT=1"
set "ARGS="
:parse
if "%~1"=="" goto run
if /I "%~1"=="--no-pause" (set "PAUSE_ON_EXIT=") else if /I "%~1"=="--skip-tests" (set "ARGS=-SkipTests")
shift
goto parse

:run
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\update-feishubot.ps1" %ARGS%
set "CODE=%ERRORLEVEL%"
if defined PAUSE_ON_EXIT pause
exit /b %CODE%
