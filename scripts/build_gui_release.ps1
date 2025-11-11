#requires -Version 5.0
param(
    [switch] $SkipBuild
)

$ErrorActionPreference = "Stop"

$workspaceRoot = Split-Path -Parent $PSScriptRoot
$targetDir = Join-Path $workspaceRoot "target\release"
$distDir = Join-Path $workspaceRoot "dist\cvxrs-studio"
$exeSource = Join-Path $targetDir "cvxrs-gui.exe"
$exeDest = Join-Path $distDir "cvxrs-studio.exe"

if (-not $SkipBuild) {
    Write-Host "Compilando cvxrs-gui en modo release..."
    cargo build --release -p cvxrs-gui
}

if (-not (Test-Path $exeSource)) {
    throw "No se encontró $exeSource. Asegúrate de que la compilación se ejecutó correctamente."
}

New-Item -ItemType Directory -Force -Path $distDir | Out-Null
Copy-Item -Path $exeSource -Destination $exeDest -Force

Copy-Item -Path (Join-Path $workspaceRoot "examples") `
    -Destination (Join-Path $distDir "examples") `
    -Recurse -Force

Write-Host "Paquete listo en $distDir"
Write-Host "Crea un acceso directo apuntando a $exeDest para lanzar la interfaz gráfica."
