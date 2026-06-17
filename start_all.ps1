$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$tmp = Join-Path $root ".tmp"
$hfHome = Join-Path $root ".hf-cache"

New-Item -ItemType Directory -Force $tmp | Out-Null
New-Item -ItemType Directory -Force $hfHome | Out-Null

function Stop-PortListener {
  param([int]$Port)

  $lines = netstat -ano | Select-String "127.0.0.1:$Port"
  foreach ($line in $lines) {
    if ($line -notmatch "LISTENING") { continue }
    $parts = ($line.ToString() -split "\s+") | Where-Object { $_ -ne "" }
    $listenPid = [int]$parts[-1]
    $proc = Get-Process -Id $listenPid -ErrorAction SilentlyContinue
    if ($proc) {
      Stop-Process -Id $listenPid -Force
    }
  }
}

function Start-ReaderService {
  param(
    [string]$Name,
    [string]$PythonPath,
    [string]$ScriptPath,
    [int]$Port,
    [hashtable]$ExtraEnv = @{}
  )

  if (!(Test-Path $PythonPath)) {
    throw "$Name environment is missing. Run .\setup.ps1 first."
  }

  $env:TEMP = $tmp
  $env:TMP = $tmp
  $env:HF_HOME = $hfHome
  foreach ($key in $ExtraEnv.Keys) {
    Set-Item -Path "Env:$key" -Value $ExtraEnv[$key]
  }

  $log = Join-Path $root "$Name.log"
  $err = Join-Path $root "$Name.err.log"
  $argsList = @("`"$ScriptPath`"", "--host", "127.0.0.1", "--port", "$Port")
  $proc = Start-Process -FilePath $PythonPath -ArgumentList $argsList -WorkingDirectory $root -WindowStyle Hidden -PassThru -RedirectStandardOutput $log -RedirectStandardError $err

  $line = $null
  for ($attempt = 0; $attempt -lt 45; $attempt += 1) {
    Start-Sleep -Seconds 1
    if ($proc.HasExited) { break }
    $line = netstat -ano | Select-String "127.0.0.1:$Port" | Where-Object { $_ -match "LISTENING" } | Select-Object -First 1
    if ($line) { break }
  }

  if ($line) {
    $parts = ($line.ToString() -split "\s+") | Where-Object { $_ -ne "" }
    Set-Content -LiteralPath (Join-Path $root "$Name.pid") -Value $parts[-1]
    Write-Host "$Name listening on http://127.0.0.1:$Port"
  } else {
    Set-Content -LiteralPath (Join-Path $root "$Name.pid") -Value $proc.Id
    Write-Warning "$Name did not report a listener yet. Check $err."
  }
}

Stop-PortListener -Port 7860
Stop-PortListener -Port 7861

Start-ReaderService `
  -Name "kokoro_tts_server" `
  -PythonPath (Join-Path $root ".venv\Scripts\python.exe") `
  -ScriptPath (Join-Path $root "kokoro_tts_server.py") `
  -Port 7860

Start-ReaderService `
  -Name "neutts_tts_server" `
  -PythonPath (Join-Path $root ".venv-neutts\Scripts\python.exe") `
  -ScriptPath (Join-Path $root "neutts_tts_server.py") `
  -Port 7861 `
  -ExtraEnv @{ PHONEMIZER_ESPEAK_LIBRARY = (Join-Path $root ".venv-neutts\Lib\site-packages\neutts\espeak-ng.dll") }

Write-Host "Open http://127.0.0.1:7860"
