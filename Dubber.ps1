# Starts the review app: makes sure the environment matches the lockfile,
# makes sure Ollama is running, starts the server and opens the browser.
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root
$env:Path = [Environment]::GetEnvironmentVariable("Path", "User") + ";" + [Environment]::GetEnvironmentVariable("Path", "Machine")

$marker = Join-Path $Root ".dubber-env.json"
if (-not (Test-Path $marker) -or -not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "First run on this machine: running setup..." -ForegroundColor Cyan
    & (Join-Path $Root "setup.ps1")
    if ($LASTEXITCODE -ne 0) { exit 1 }
}
$cfg = Get-Content $marker -Raw | ConvertFrom-Json

# A code update can change the lockfile; this is a no-op when nothing changed.
Write-Host "Checking environment..." -ForegroundColor Cyan
& uv sync --extra $cfg.extra --quiet
if ($LASTEXITCODE -ne 0) { Write-Host "Environment sync failed (see above)" -ForegroundColor Red; exit 1 }

if ($cfg.ollama) {
    $up = $false
    try { Invoke-RestMethod http://localhost:11434/api/tags -TimeoutSec 2 | Out-Null; $up = $true } catch {}
    if (-not $up -and (Get-Command ollama -ErrorAction SilentlyContinue)) {
        Write-Host "Starting Ollama..." -ForegroundColor Cyan
        Start-Process ollama -ArgumentList "serve" -WindowStyle Hidden
    }
}

# Open the browser once the server answers (set DUBBER_NO_BROWSER=1 to skip).
$url = "http://127.0.0.1:5000"
if (-not $env:DUBBER_NO_BROWSER) {
    Start-Job -ScriptBlock {
        param($u)
        for ($i = 0; $i -lt 60; $i++) {
            try { Invoke-WebRequest $u -UseBasicParsing -TimeoutSec 2 | Out-Null; Start-Process $u; return } catch { Start-Sleep 1 }
        }
    } -ArgumentList $url | Out-Null
}

Write-Host "Review app at $url  (Ctrl+C to stop)" -ForegroundColor Green
& ".venv\Scripts\python.exe" "review_ui\app.py"
