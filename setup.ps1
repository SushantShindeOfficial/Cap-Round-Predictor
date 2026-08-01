# One-click setup for Normal Cap Predictor (Windows PowerShell)
# Usage:
#   cd "E:\Sushant\LLMCHATBOT_v1_F\Normal Cap Predictor"
#   powershell -ExecutionPolicy Bypass -File .\setup.ps1

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

Write-Host "==> Project root: $Root"

$py = Get-Command python -ErrorAction SilentlyContinue
if (-not $py) {
    Write-Error "Python not found on PATH. Install Python 3.10+ and retry."
}

Write-Host "==> Creating virtual environment (.venv)"
if (-not (Test-Path ".\.venv")) {
    python -m venv .venv
}

$venvPython = Join-Path $Root ".venv\Scripts\python.exe"
$venvPip = Join-Path $Root ".venv\Scripts\pip.exe"
if (-not (Test-Path $venvPython)) {
    Write-Error "Failed to create .venv"
}

Write-Host "==> Upgrading pip"
& $venvPython -m pip install --upgrade pip

Write-Host "==> Installing requirements"
& $venvPip install -r requirements.txt

Write-Host "==> Editable install (cap_predictor package)"
& $venvPip install -e .

Write-Host ""
Write-Host "Setup complete."
Write-Host "Activate the venv, then run:"
Write-Host '  .\.venv\Scripts\Activate.ps1'
Write-Host '  python -m cap_predictor.cli parse'
Write-Host '  streamlit run src\cap_predictor\app_streamlit.py'
Write-Host ""
Write-Host "Or without activating:"
Write-Host "  .\.venv\Scripts\python.exe -m cap_predictor.cli predict --interactive --no-llm"
