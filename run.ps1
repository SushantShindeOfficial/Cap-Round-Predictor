# Launch Streamlit CAP Predictor UI
# Usage: powershell -ExecutionPolicy Bypass -File .\run.ps1

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
Write-Host "Starting Streamlit on http://localhost:8501 ..."
& $venvPython -m streamlit run (Join-Path $Root "src\cap_predictor\app_streamlit.py")
