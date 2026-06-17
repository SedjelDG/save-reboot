$ErrorActionPreference = "Stop"

$lines = netstat -ano | Select-String ":7860"
$pids = @()

foreach ($line in $lines) {
    $parts = ($line.ToString() -split "\s+") | Where-Object { $_ -ne "" }
    if ($parts.Length -ge 5 -and $parts[1] -like "127.0.0.1:7860") {
        $pids += [int]$parts[-1]
    }
}

$pids = $pids | Sort-Object -Unique | Where-Object { $_ -gt 0 }

if (-not $pids) {
    Write-Output "No Qwen3-TTS server found on 127.0.0.1:7860."
    exit 0
}

foreach ($processId in $pids) {
    Stop-Process -Id $processId -Force
    Write-Output "Stopped process $processId."
}
