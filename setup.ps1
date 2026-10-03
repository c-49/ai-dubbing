# One-time (and repeatable) setup for the dubbing project on a Windows machine.
#
#   setup.bat            -> everything, auto-detecting an NVIDIA GPU
#   setup.bat -Cpu       -> force the CPU build of torch
#   setup.bat -NoClone   -> skip the voice-cloning engine (stock Kokoro voices only)
#   setup.bat -SkipOllama
#
# Safe to re-run: every step checks what is already there. It never touches
# models/, shows/ or work/.
param(
    [switch]$Cpu,
    [switch]$NoClone,
    [switch]$SkipOllama
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

function Step($msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }
function Ok($msg)   { Write-Host "    $msg" -ForegroundColor Green }
function Warn($msg) { Write-Host "    $msg" -ForegroundColor Yellow }

# Native commands report failure through $LASTEXITCODE, not exceptions.
function Run($exe, [string[]]$arguments) {
    & $exe @arguments
    if ($LASTEXITCODE -ne 0) { throw "$exe $($arguments -join ' ') failed (exit code $LASTEXITCODE)" }
}

# True if the command exits 0. Windows PowerShell 5.1 turns any stderr output from a
# native command into a terminating error under $ErrorActionPreference = "Stop", so
# probes (which are allowed to fail noisily) must run with "Continue".
function Succeeds($exe, [string[]]$arguments) {
    $old = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try { & $exe @arguments *> $null; return ($LASTEXITCODE -eq 0) }
    finally { $ErrorActionPreference = $old }
}

function Refresh-Path {
    $env:Path = [Environment]::GetEnvironmentVariable("Path", "User") + ";" + [Environment]::GetEnvironmentVariable("Path", "Machine")
}

function Winget-Install($id) {
    if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
        throw "winget is not available. Install 'App Installer' from the Microsoft Store, or install $id by hand."
    }
    winget install --id $id -e --accept-source-agreements --accept-package-agreements
    Refresh-Path
}

# ---------------------------------------------------------------- uv
Step "uv (Python environment manager)"
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    try { Winget-Install "astral-sh.uv" } catch { Warn "winget failed ($_); using the official installer" }
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
        $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
        Refresh-Path
    }
}
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    # winget puts uv under its own Packages folder, which a fresh shell may not have on PATH yet
    $found = Get-ChildItem "$env:LOCALAPPDATA\Microsoft\WinGet\Packages" -Filter uv.exe -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($found) { $env:Path = "$($found.DirectoryName);$env:Path" }
}
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) { throw "uv could not be installed" }
Ok (uv --version)

# ---------------------------------------------------------------- hardware
Step "Hardware"
$hasNvidia = $false
if (-not $Cpu -and (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
    if (Succeeds nvidia-smi @("--query-gpu=name", "--format=csv,noheader")) {
        $gpu = (& nvidia-smi --query-gpu=name,memory.total --format=csv,noheader) -join " "
        $hasNvidia = $true
        Ok "NVIDIA GPU: $gpu"
    }
}
$extra = if ($hasNvidia) { "cuda" } else { "cpu" }
if (-not $hasNvidia) { Warn "No NVIDIA GPU used -> CPU build of torch (everything still works, just slower)" }

# ---------------------------------------------------------------- python environment
Step "Python environment (.venv, torch: $extra)"
Run uv @("sync", "--extra", $extra)
$Python = Join-Path $Root ".venv\Scripts\python.exe"
Ok (& $Python --version)

# ---------------------------------------------------------------- ffmpeg + eSpeak NG into bin/
Step "ffmpeg and eSpeak NG (bundled in bin/)"
if (-not (Test-Path "bin\ffmpeg\ffmpeg.exe")) {
    $src = Get-ChildItem "$env:LOCALAPPDATA\Microsoft\WinGet\Packages" -Directory -Filter "Gyan.FFmpeg.Shared*" -ErrorAction SilentlyContinue |
        ForEach-Object { Get-ChildItem $_.FullName -Recurse -Filter ffmpeg.exe -ErrorAction SilentlyContinue } | Select-Object -First 1
    if (-not $src) {
        Winget-Install "Gyan.FFmpeg.Shared"
        $src = Get-ChildItem "$env:LOCALAPPDATA\Microsoft\WinGet\Packages" -Directory -Filter "Gyan.FFmpeg.Shared*" |
            ForEach-Object { Get-ChildItem $_.FullName -Recurse -Filter ffmpeg.exe } | Select-Object -First 1
    }
    if (-not $src) { throw "could not find or install the shared ffmpeg build" }
    New-Item -ItemType Directory -Force "bin\ffmpeg" | Out-Null
    # the shared build's bin folder holds ffmpeg.exe, ffprobe.exe and the DLLs torchcodec needs
    Get-ChildItem $src.DirectoryName -File | Where-Object { $_.Name -notlike "ffplay*" } | Copy-Item -Destination "bin\ffmpeg"
}
Ok "ffmpeg: bin\ffmpeg"

if (-not (Test-Path "bin\espeak-ng\espeak-ng.exe")) {
    $espeak = "C:\Program Files\eSpeak NG"
    if (-not (Test-Path "$espeak\espeak-ng.exe")) { Winget-Install "eSpeak-NG.eSpeak-NG" }
    if (-not (Test-Path "$espeak\espeak-ng.exe")) { throw "could not find or install eSpeak NG" }
    New-Item -ItemType Directory -Force "bin\espeak-ng" | Out-Null
    Copy-Item "$espeak\*" "bin\espeak-ng" -Recurse -Force
}
Ok "eSpeak NG: bin\espeak-ng"

# ---------------------------------------------------------------- voice cloning engine
if ($NoClone) {
    Step "Voice-cloning engine skipped (-NoClone)"
} else {
    Step "Voice-cloning engine (OmniVoice, its own environment)"
    $EngPy = Join-Path $Root ".venv-engines\omnivoice\Scripts\python.exe"
    if (-not (Test-Path $EngPy)) {
        Run uv @("venv", ".venv-engines\omnivoice", "--python", "3.11")
    }
    if (-not (Succeeds $EngPy @("-c", "import omnivoice"))) {
        if ($hasNvidia) {
            Run uv @("pip", "install", "--python", $EngPy, "torch==2.8.0+cu126", "torchaudio==2.8.0+cu126",
                     "--index-url", "https://download.pytorch.org/whl/cu126")
        } else {
            Run uv @("pip", "install", "--python", $EngPy, "torch==2.8.0", "torchaudio==2.8.0",
                     "--index-url", "https://download.pytorch.org/whl/cpu")
        }
        Run uv @("pip", "install", "--python", $EngPy, "omnivoice", "soundfile", "pyyaml", "numpy")
    }
    Ok "OmniVoice environment ready"
}

# ---------------------------------------------------------------- Ollama (translation)
if ($SkipOllama) {
    Step "Ollama skipped (-SkipOllama)"
} else {
    Step "Ollama (translation model)"
    if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) { Winget-Install "Ollama.Ollama" }
    if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
        $guess = "$env:LOCALAPPDATA\Programs\Ollama"
        if (Test-Path "$guess\ollama.exe") { $env:Path = "$guess;$env:Path" }
    }
    if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) { throw "Ollama is not available after install" }

    $model = & $Python -c "import yaml; print(yaml.safe_load(open('config.yaml', encoding='utf-8'))['models']['ollama']['model'])"
    $up = $false
    try { Invoke-RestMethod http://localhost:11434/api/tags -TimeoutSec 3 | Out-Null; $up = $true } catch {}
    if (-not $up) {
        Start-Process ollama -ArgumentList "serve" -WindowStyle Hidden
        for ($i = 0; $i -lt 30 -and -not $up; $i++) {
            Start-Sleep 1
            try { Invoke-RestMethod http://localhost:11434/api/tags -TimeoutSec 3 | Out-Null; $up = $true } catch {}
        }
    }
    if (-not $up) { throw "Ollama did not start" }
    $have = ((& ollama list) -join "`n") -match [regex]::Escape($model)
    if (-not $have) { Run ollama @("pull", $model) }
    Ok "Ollama model ready: $model"
}

# ---------------------------------------------------------------- Hugging Face token (per user, once)
Step "Hugging Face token (speaker detection uses a gated model)"
$tokenFile = Join-Path $env:USERPROFILE ".cache\huggingface\token"
if (Test-Path $tokenFile) {
    Ok "token already saved for this user"
} else {
    Write-Host @"
    Speaker detection (pyannote) needs a free Hugging Face account and token.
      1. Create an account:  https://huggingface.co/join
      2. Accept the terms (click "Agree") on these pages:
           https://huggingface.co/pyannote/speaker-diarization-3.1
           https://huggingface.co/pyannote/segmentation-3.0
           https://huggingface.co/pyannote/speaker-diarization-community-1
      3. Create a "Read" token:  https://huggingface.co/settings/tokens
"@
    $token = Read-Host "    Paste the token here (or press Enter to skip and add it later)"
    if ($token) {
        New-Item -ItemType Directory -Force (Split-Path $tokenFile) | Out-Null
        [IO.File]::WriteAllText($tokenFile, $token.Trim())
        Ok "token saved to $tokenFile (outside the project folder)"
    } else {
        Warn "skipped; speaker detection will fail until you save a token to $tokenFile"
    }
}

# ---------------------------------------------------------------- record + verify
@{ extra = $extra; clone = (-not $NoClone); ollama = (-not $SkipOllama) } | ConvertTo-Json | Set-Content ".dubber-env.json" -Encoding UTF8

Step "Checking the install"
Run $Python @("src\check_setup.py")

Write-Host "`nSetup complete. Start the app with Dubber.bat" -ForegroundColor Green
