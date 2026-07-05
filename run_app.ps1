$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $RepoRoot

$env:PYTHONPATH = "$RepoRoot;$RepoRoot\src"

if (-not (Test-Path ".\.venv\Scripts\python.exe")) {
    Write-Host "Creating local virtual environment..."
    python -m venv .venv
}

$VenvPython = ".\.venv\Scripts\python.exe"

Write-Host "Installing/updating required Python packages..."
& $VenvPython -m pip install --upgrade pip
& $VenvPython -m pip install -r .\requirements.txt

Write-Host "Starting HypnoTrust Streamlit app..."
Write-Host "PYTHONPATH=$env:PYTHONPATH"

& $VenvPython -m streamlit run .\app\app.py
