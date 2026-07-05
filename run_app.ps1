$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $RepoRoot

$env:PYTHONPATH = "$RepoRoot;$RepoRoot\src"

if (Test-Path ".\.venv\Scripts\python.exe") {
    $Python = ".\.venv\Scripts\python.exe"
} else {
    $Python = "python"
}

Write-Host "Starting HypnoTrust Streamlit app..."
Write-Host "PYTHONPATH=$env:PYTHONPATH"

& $Python -m streamlit run .\app\app.py
