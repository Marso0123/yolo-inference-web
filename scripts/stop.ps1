$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$pidFile = Join-Path $projectRoot '.runtime\server.pid'
try {
    if (-not (Test-Path -LiteralPath $pidFile)) { Write-Host '没有本项目启动记录。'; exit 0 }
    $serverProcessId = [int](Get-Content -LiteralPath $pidFile)
    $process = Get-CimInstance Win32_Process -Filter "ProcessId = $serverProcessId" -ErrorAction SilentlyContinue
    if ($process) {
        $pythonPath = Join-Path $projectRoot '.venv\Scripts\python.exe'
        if ($process.CommandLine -notmatch [regex]::Escape($pythonPath) -or $process.CommandLine -notmatch 'run.py') { throw 'PID 对应的程序不属于本项目，未执行停止。' }
        Invoke-RestMethod -Method Post 'http://127.0.0.1:8790/api/stop' -TimeoutSec 45 | Out-Null
        Stop-Process -Id $serverProcessId
    }
    Remove-Item -LiteralPath $pidFile
    Write-Host '本项目服务已停止。'
} catch { Write-Host $_ -ForegroundColor Red; exit 1 }
