$ErrorActionPreference = "Stop"
$PackDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvDir = Join-Path $PackDir ".venv"

Write-Host "[BeadSketch AI] Creating isolated Python environment..."
python -m venv $VenvDir
$Python = Join-Path $VenvDir "Scripts\python.exe"
& $Python -m pip install --upgrade pip
& $Python -m pip install "onnxruntime-gpu[cuda,cudnn]==1.26.0" numpy

Write-Host "[BeadSketch AI] Verifying a real model inference..."
& $Python (Join-Path $PackDir "verify_runtime.py")
if ($LASTEXITCODE -ne 0) {
  Write-Warning "CUDA inference was unavailable. The app will safely use CPU instead."
}
Write-Host "Done. The app will prefer CUDA automatically and fall back to CPU safely."
Read-Host "Press Enter to close"
