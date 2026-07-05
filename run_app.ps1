$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $RepoRoot

$env:PYTHONPATH = "$RepoRoot;$RepoRoot\src"
$VenvPython = Join-Path $RepoRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $VenvPython)) {
    Write-Host "Creating local virtual environment..."
    python -m venv .venv
}

if (-not (Test-Path $VenvPython)) {
    throw "Could not find or create .venv\Scripts\python.exe"
}

& $VenvPython -c "import streamlit" 2>$null

if ($LASTEXITCODE -ne 0) {
    Write-Host "Streamlit is not installed in this repo environment."
    Write-Host "Installing dependencies from requirements.txt..."
    & $VenvPython -m pip install -r requirements.txt
}

& $VenvPython -c "import streamlit" 2>$null

if ($LASTEXITCODE -ne 0) {
    throw "Streamlit is still unavailable after dependency installation."
}

Write-Host "Starting HypnoTrust Streamlit app..."
Write-Host "PYTHONPATH=$env:PYTHONPATH"

& $VenvPython -m streamlit run .\app\app.py
