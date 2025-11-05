$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repoRoot

try {
    $pythonCmd = Get-Command python -ErrorAction Stop
} catch {
    throw "Python is not available on PATH. Install Python 3.9+ and rerun the script."
}

$venvPath = Join-Path $repoRoot '.venv-build'

if (-not (Test-Path $venvPath)) {
    & $pythonCmd.Source -m venv $venvPath
}

$venvPython = Join-Path $venvPath 'Scripts\python.exe'

& $venvPython -m pip install --upgrade pip wheel
& $venvPython -m pip install -r requirements.txt
& $venvPython -m pip install pyinstaller

$distDir = Join-Path $repoRoot 'dist'
if (-not (Test-Path $distDir)) {
    New-Item $distDir -ItemType Directory | Out-Null
}

$pyInstallerArgs = @(
    "-m", "PyInstaller",
    "main.py",
    "--noconfirm",
    "--clean",
    "--windowed",
    "--name", "cnn-dataset-annotation-tool",
    "--add-data", "cnn_dataset_annotation_tool;cnn_dataset_annotation_tool"
)

# Include sample datasets if present so demo data ships with the binary.
if (Test-Path (Join-Path $repoRoot 'datasets')) {
    $pyInstallerArgs += @("--add-data", "datasets;datasets")
}

& $venvPython $pyInstallerArgs

Write-Host ""
Write-Host "Executable build complete. Check the 'dist' folder for cnn-dataset-annotation-tool.exe."
