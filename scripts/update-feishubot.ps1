param([switch]$SkipTests)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$python = Join-Path $root '.venv\Scripts\python.exe'
$env:PYTHONIOENCODING = 'utf-8'

function Get-EventsProcesses {
    Get-CimInstance Win32_Process | Where-Object {
        $_.Name -eq 'python.exe' -and $_.CommandLine -and
        $_.CommandLine -match 'feishubot\.cli' -and $_.CommandLine -match 'events'
    }
}

function Get-GatewayProcesses {
    Get-CimInstance Win32_Process | Where-Object {
        $_.Name -eq 'python.exe' -and $_.CommandLine -and
        $_.CommandLine -match 'uvicorn' -and $_.CommandLine -match 'feishubot\.main:app'
    }
}

function Read-ActiveClaim([string]$path) {
    try { return Get-Content -Raw -Encoding UTF8 $path | ConvertFrom-Json } catch { return $null }
}

if (-not $SkipTests) {
    Write-Host '[CHECK] Running tests; the current version keeps serving if they fail'
    & $python -m pytest -q -p no:cacheprovider
    if ($LASTEXITCODE -ne 0) { Write-Host '[ABORT] Tests failed'; exit 1 }
}
& $python -m compileall -q src
if ($LASTEXITCODE -ne 0) { Write-Host '[ABORT] Compile check failed'; exit 1 }

$handoffDir = (& $python -c 'from feishubot.config import settings; print(settings.qoder_handoff_dir)').Trim()
if (-not $handoffDir) { Write-Host '[ABORT] QODER_HANDOFF_DIR is empty; seamless update is disabled'; exit 1 }
$activeFile = Join-Path $handoffDir 'active.json'

$oldPids = @(Get-EventsProcesses | ForEach-Object { [int]$_.ProcessId })
$startedAt = [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds() / 1000.0

Write-Host '[START] New FeishuBot events process'
$launcher = Start-Process -FilePath $python -PassThru -WorkingDirectory $root -ArgumentList @(
    '-m', 'feishubot.cli', 'events', '--no-fallback-webhook', '--log-level', 'WARNING')

$newPid = $null
$deadline = (Get-Date).AddSeconds(120)
while ((Get-Date) -lt $deadline) {
    Start-Sleep -Seconds 1
    $claim = Read-ActiveClaim $activeFile
    if ($claim -and $claim.claimed_at -ge $startedAt -and $oldPids -notcontains [int]$claim.pid) {
        $newPid = [int]$claim.pid
        break
    }
    if ($launcher.HasExited) { break }
}

if (-not $newPid) {
    Write-Host '[ABORT] New version did not take over; stopping it and keeping the current version'
    if (-not $launcher.HasExited) { & taskkill.exe /T /F /PID $launcher.Id | Out-Null }
    exit 1
}

Start-Sleep -Seconds 5
if (-not (Get-Process -Id $newPid -ErrorAction SilentlyContinue)) {
    Write-Host "[ABORT] New events process pid=$newPid exited right after taking over"
    exit 1
}
Write-Host "[READY] New events process pid=$newPid now serves new messages"

$gateways = @(Get-GatewayProcesses)
foreach ($gateway in $gateways) { Stop-Process -Id $gateway.ProcessId -Force -ErrorAction SilentlyContinue }
Start-Process -FilePath $python -WorkingDirectory $root -ArgumentList @(
    '-m', 'uvicorn', 'feishubot.main:app', '--host', '127.0.0.1', '--port', '8000') | Out-Null
$gatewayReady = $false
$deadline = (Get-Date).AddSeconds(60)
while ((Get-Date) -lt $deadline) {
    try {
        $response = Invoke-WebRequest -UseBasicParsing -TimeoutSec 3 'http://127.0.0.1:8000/admin/config'
        if ($response.StatusCode -eq 200) { $gatewayReady = $true; break }
    } catch { }
    Start-Sleep -Seconds 1
}
if ($gatewayReady) { Write-Host '[READY] Gateway restarted: http://127.0.0.1:8000/admin/config' }
else { Write-Host '[WARN] Gateway did not answer within 60 seconds; Feishu messages are unaffected' }

$draining = @($oldPids | Where-Object { Get-Process -Id $_ -ErrorAction SilentlyContinue })
if ($draining) {
    Write-Host "[DRAIN] Old pids $($draining -join ', ') finish in-flight analyses and exit by themselves"
}
exit 0
