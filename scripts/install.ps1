$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$env:PYTHONIOENCODING = 'utf-8'
$env:UV_LINK_MODE = 'copy'
try {
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) { throw '未找到 uv。请先执行：python -m pip install --user uv，然后重新打开终端。' }
    $pythonPath = Join-Path $projectRoot '.venv\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $pythonPath)) {
        uv python install 3.11
        if ($LASTEXITCODE -ne 0) { throw 'Python 安装失败。' }
        uv venv --python 3.11 .venv
        if ($LASTEXITCODE -ne 0) { throw '虚拟环境创建失败。' }
    }
    Write-Host '安装本项目独立的 CUDA 推理依赖…'
    uv pip install --python $pythonPath torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu126
    if ($LASTEXITCODE -ne 0) { throw 'PyTorch 安装失败。' }
    $requirementsFile = if (Test-Path 'requirements.lock.txt') { 'requirements.lock.txt' } else { 'requirements.txt' }
    uv pip install --python $pythonPath -r $requirementsFile --extra-index-url https://download.pytorch.org/whl/cu126 --index-strategy unsafe-best-match
    if ($LASTEXITCODE -ne 0) { throw '依赖安装失败。' }
    & $pythonPath -X utf8 scripts\fetch_ffmpeg.py
    if ($LASTEXITCODE -ne 0) { throw 'FFmpeg 安装失败。' }
    & $pythonPath -X utf8 -c "import torch,ultralytics,onnxruntime,cv2; print('Dependencies ready. CUDA available:', torch.cuda.is_available())"
    if ($LASTEXITCODE -ne 0) { throw '依赖导入检查失败。' }
    New-Item -ItemType Directory -Path '.runtime' -Force | Out-Null
    Set-Content -LiteralPath '.runtime\install.done' -Value 'ready' -Encoding ASCII
    Write-Host '安装完成。请运行“启动模型查看器.cmd”。'
} catch { Write-Host $_ -ForegroundColor Red; exit 1 }
