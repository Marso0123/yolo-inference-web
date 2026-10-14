param([switch]$NoBrowser)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$env:PYTHONIOENCODING = 'utf-8'
try {
    if (-not (Test-Path -LiteralPath '.runtime\install.done') -or -not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
        & (Join-Path $PSScriptRoot 'install.ps1')
        if ($LASTEXITCODE -ne 0) { throw '安装未完成，请查看上方错误。' }
    }
    $existing = Get-NetTCPConnection -State Listen -LocalPort 8790 -ErrorAction SilentlyContinue
    if ($existing) {
        $health = Invoke-RestMethod 'http://127.0.0.1:8790/api/hardware' -TimeoutSec 5
        if ($health.project_root -ne $projectRoot) { throw '端口 8790 已被其他程序占用。' }
        if (-not $NoBrowser) { Start-Process 'http://127.0.0.1:8790' }
        Write-Host '模型查看器已经运行。'
        exit 0
    }
    New-Item -ItemType Directory -Path logs -Force | Out-Null
    $pythonPath = Join-Path $projectRoot '.venv\Scripts\python.exe'
    $server = Start-Process -FilePath $pythonPath -ArgumentList @('-X', 'utf8', 'run.py') -WorkingDirectory $projectRoot -WindowStyle Hidden -RedirectStandardOutput 'logs\stdout.log' -RedirectStandardError 'logs\stderr.log' -PassThru
    Set-Content -LiteralPath '.runtime\server.pid' -Value $server.Id -Encoding ASCII
    $deadline = (Get-Date).AddSeconds(90)
    do {
        Start-Sleep -Milliseconds 500
        $server.Refresh()
        if ($server.HasExited) { throw '服务启动失败，请查看 logs\stderr.log。' }
        try {
            $health = Invoke-RestMethod 'http://127.0.0.1:8790/api/hardware' -TimeoutSec 2
            if ($health.project_root -eq $projectRoot) {
                if (-not $NoBrowser) { Start-Process 'http://127.0.0.1:8790' }
                Write-Host '已启动：http://127.0.0.1:8790'
                exit 0
            }
        } catch {}
    } while ((Get-Date) -lt $deadline)
    throw '启动超时，请查看 logs\stderr.log。'
} catch { Write-Host $_ -ForegroundColor Red; exit 1 }
