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

Write-Host "Running HypnoTrust tests..."
Write-Host "PYTHONPATH=$env:PYTHONPATH"

& $VenvPython -m unittest discover -s tests -p "test*.py"
