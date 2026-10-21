<#
Build Mullenizer 1.0 as a Windows one-folder application and portable ZIP.
Run on Windows x64, from any working directory:
  powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\packaging\build_release.ps1
#>
[CmdletBinding()]
param(
    [string]$TesseractSource = (Join-Path $env:ProgramFiles 'Tesseract-OCR')
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$project = Split-Path -Parent $PSScriptRoot
$python = Join-Path $project '.venv\Scripts\python.exe'
$entry = Join-Path $project 'mullenizer_entry.py'
$bundle = Join-Path $project 'dist\Mullenizer'
$release = Join-Path $project 'release'
$portableZip = Join-Path $release 'Mullenizer-1.0-portable.zip'

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw 'Missing .venv. Create it and install packaging\requirements-build.txt first.'
}
if (-not (Test-Path -LiteralPath $entry -PathType Leaf)) {
    throw "Missing entry point: $entry. Ensure you are using Mullenizer 1.0 source files."
}
if (-not (Test-Path -LiteralPath (Join-Path $project 'mullenizer_gui.py') -PathType Leaf)) {
    throw 'Missing mullenizer_gui.py.'
}
$tesseractExe = Join-Path $TesseractSource 'tesseract.exe'
$englishData = Join-Path $TesseractSource 'tessdata\eng.traineddata'
if (-not (Test-Path -LiteralPath $tesseractExe -PathType Leaf)) {
    throw "Tesseract not found at $tesseractExe; supply -TesseractSource with its install folder."
}
if (-not (Test-Path -LiteralPath $englishData -PathType Leaf)) {
    throw "Tesseract English traineddata not found: $englishData"
}

Push-Location $project
try {
    Write-Host 'Checking Python package dependencies...'
    & $python -m pip check
    if ($LASTEXITCODE -ne 0) { throw 'Python dependency check failed.' }

    Write-Host 'Building Mullenizer.exe using PyInstaller...'
    & $python -m PyInstaller `
        --noconfirm `
        --clean `
        --onedir `
        --contents-directory '_internal' `
        --console `
        --hide-console=hide-early `
        --name Mullenizer `
        mullenizer_entry.py
    if ($LASTEXITCODE -ne 0) { throw 'PyInstaller failed.' }

    # Bundle the OCR engine independently of the end user's system.
    # Copy the executable, same-directory DLLs, English data, and optional configs.
    $tessDest = Join-Path $bundle 'tesseract'
    if (Test-Path -LiteralPath $tessDest) {
        Remove-Item -LiteralPath $tessDest -Recurse -Force
    }
    $tessDataDest = Join-Path $tessDest 'tessdata'
    New-Item -ItemType Directory -Path $tessDataDest -Force | Out-Null
    Copy-Item -LiteralPath $tesseractExe -Destination $tessDest
    Copy-Item -LiteralPath $englishData -Destination $tessDataDest

    # Some Tesseract installations use osd data for orientation detection.
    $osd = Join-Path $TesseractSource 'tessdata\osd.traineddata'
    if (Test-Path -LiteralPath $osd -PathType Leaf) {
        Copy-Item -LiteralPath $osd -Destination $tessDataDest
    }
    Get-ChildItem -LiteralPath $TesseractSource -File -Filter '*.dll' | ForEach-Object {
        Copy-Item -LiteralPath $_.FullName -Destination $tessDest
    }
    foreach ($folder in @('configs', 'tessconfigs')) {
        $sourceFolder = Join-Path $TesseractSource "tessdata\$folder"
        if (Test-Path -LiteralPath $sourceFolder -PathType Container) {
            Copy-Item -LiteralPath $sourceFolder -Destination $tessDataDest -Recurse
        }
    }
    # Keep the notices available from the Tesseract installation.
    Get-ChildItem -LiteralPath $TesseractSource -File | Where-Object {
        $_.Name -match '^(LICENSE|COPYING|NOTICE)([._-]|$)'
    } | ForEach-Object {
        Copy-Item -LiteralPath $_.FullName -Destination $tessDest
    }

    $app = Join-Path $bundle 'Mullenizer.exe'
    $bundledTesseract = Join-Path $tessDest 'tesseract.exe'
    foreach ($path in @($app, $bundledTesseract, (Join-Path $tessDataDest 'eng.traineddata'))) {
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
            throw "Incomplete release; missing: $path"
        }
    }

    Write-Host 'Checking bundled Tesseract...'
    $oldTessdataPrefix = $env:TESSDATA_PREFIX
    try {
        $env:TESSDATA_PREFIX = $tessDataDest
        & $bundledTesseract --version
        if ($LASTEXITCODE -ne 0) { throw 'Bundled Tesseract --version failed.' }
        $languages = & $bundledTesseract --list-langs
        if ($LASTEXITCODE -ne 0 -or -not (@($languages) -contains 'eng')) {
            throw 'Bundled Tesseract could not find its English OCR model.'
        }
    }
    finally {
        $env:TESSDATA_PREFIX = $oldTessdataPrefix
    }

    Write-Host 'Smoke-testing the frozen entry-point dispatcher...'
    & $app --mullenizer-pipeline --help | Out-Null
    if ($LASTEXITCODE -ne 0) { throw 'Frozen pipeline --help failed.' }
    foreach ($stage in @('render', 'deskew', 'ocr')) {
        & $app --mullenizer-stage $stage --help | Out-Null
        if ($LASTEXITCODE -ne 0) { throw "Frozen $stage --help failed." }
    }

    New-Item -ItemType Directory -Path $release -Force | Out-Null
    if (Test-Path -LiteralPath $portableZip) {
        Remove-Item -LiteralPath $portableZip -Force
    }
    Compress-Archive -Path $bundle -DestinationPath $portableZip -CompressionLevel Optimal

    Write-Host ''
    Write-Host "Portable application: $bundle" -ForegroundColor Green
    Write-Host "Portable ZIP:         $portableZip" -ForegroundColor Green
    Write-Host "Next: run and test a real PDF, then compile packaging\Mullenizer.iss."
}
finally {
    Pop-Location
}
