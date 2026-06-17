$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

$env:HF_HOME = Join-Path $root ".hf-cache"
$env:TEMP = Join-Path $root ".tmp"
$env:TMP = Join-Path $root ".tmp"

New-Item -ItemType Directory -Force -Path $env:HF_HOME | Out-Null
New-Item -ItemType Directory -Force -Path $env:TEMP | Out-Null

& ".\.venv\Scripts\python.exe" ".\qwen3_tts_server.py" @args
