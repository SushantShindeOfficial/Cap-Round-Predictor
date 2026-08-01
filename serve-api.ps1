# Launch FastAPI CAP Predictor API
# Usage: powershell -ExecutionPolicy Bypass -File .\serve-api.ps1

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$venvPython = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    Write-Host "venv missing - running setup.ps1 first..."
    powershell -ExecutionPolicy Bypass -File (Join-Path $Root "setup.ps1")
}

if (-not (Test-Path $venvPython)) {
    Write-Error "Python venv not found at $venvPython. Run setup.ps1 first."
    exit 1
}

$env:PYTHONPATH = "src"
Write-Host "Starting API on http://127.0.0.1:8080 ..."
Write-Host "Docs: http://127.0.0.1:8080/docs"
& $venvPython -m uvicorn cap_predictor.api:app --host 127.0.0.1 --port 8080
