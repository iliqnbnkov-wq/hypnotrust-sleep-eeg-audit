$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $RepoRoot

$env:PYTHONPATH = "$RepoRoot;$RepoRoot\src"

if (Test-Path ".\.venv\Scripts\python.exe") {
    $Python = ".\.venv\Scripts\python.exe"
} else {
    $Python = "python"
}

Write-Host "Running HypnoTrust tests..."
Write-Host "PYTHONPATH=$env:PYTHONPATH"

& $Python -m unittest discover -s tests -p "test*.py"
