$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = if ($env:PYTHON) { $env:PYTHON } else { "python" }

function Install-Requirements {
  param(
    [string]$VenvPath,
    [string]$RequirementsPath
  )

  $pythonExe = Join-Path $VenvPath "Scripts\python.exe"
  if (!(Test-Path $pythonExe)) {
    & $python -m venv $VenvPath
  }

  & $pythonExe -m pip install --upgrade pip
  & $pythonExe -m pip install -r $RequirementsPath
}

New-Item -ItemType Directory -Force (Join-Path $root ".tmp") | Out-Null
New-Item -ItemType Directory -Force (Join-Path $root ".hf-cache") | Out-Null

Install-Requirements -VenvPath (Join-Path $root ".venv") -RequirementsPath (Join-Path $root "requirements-kokoro.txt")
Install-Requirements -VenvPath (Join-Path $root ".venv-neutts") -RequirementsPath (Join-Path $root "requirements-neutts.txt")

Write-Host "Setup complete."
Write-Host "Run .\start_all.ps1, then open http://127.0.0.1:7860"
