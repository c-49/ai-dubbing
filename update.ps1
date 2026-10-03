# Updates the code and re-syncs the environment. Never touches models/, work/
# or your show data. If this folder is a git checkout it pulls; if you got it
# as a zip, replace src/, review_ui/ and the top-level files with the new ones
# (keeping models/, shows/ and work/) and then run this to re-sync.
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root
$env:Path = [Environment]::GetEnvironmentVariable("Path", "User") + ";" + [Environment]::GetEnvironmentVariable("Path", "Machine")

if (Test-Path ".git") {
    if (git status --porcelain --untracked-files=no) {
        Write-Host "You have uncommitted changes to tracked files; commit or stash them first." -ForegroundColor Yellow
        git status --short --untracked-files=no
        exit 1
    }
    git pull --ff-only
    if ($LASTEXITCODE -ne 0) { Write-Host "git pull failed; nothing was changed." -ForegroundColor Red; exit 1 }
} else {
    Write-Host "Not a git checkout: assuming you already copied the new files in." -ForegroundColor Yellow
}

$cfg = Get-Content ".dubber-env.json" -Raw | ConvertFrom-Json
& uv sync --extra $cfg.extra
if ($LASTEXITCODE -ne 0) { Write-Host "Environment sync failed" -ForegroundColor Red; exit 1 }
Write-Host "Updated. Models, shows and work data were not touched." -ForegroundColor Green
