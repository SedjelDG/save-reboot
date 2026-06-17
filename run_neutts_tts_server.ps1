$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$env:TEMP = Join-Path $root ".tmp"
$env:TMP = Join-Path $root ".tmp"
$env:HF_HOME = Join-Path $root ".hf-cache"
$env:PHONEMIZER_ESPEAK_LIBRARY = Join-Path $root ".venv-neutts\Lib\site-packages\neutts\espeak-ng.dll"
& (Join-Path $root ".venv-neutts\Scripts\python.exe") (Join-Path $root "neutts_tts_server.py") --host 127.0.0.1 --port 7861
