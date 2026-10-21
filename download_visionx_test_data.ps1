# VisionX: download a small real-world image test set from COCO Val2017.
# Run this script from your VisionX project root in Windows PowerShell.
# The downloaded images are for detector smoke tests and speed benchmarking,
# not for accuracy evaluation unless you also use matching ground-truth labels.

$ErrorActionPreference = "Stop"

$destination = Join-Path (Get-Location) "assets\sample_images"
New-Item -ItemType Directory -Force -Path $destination | Out-Null

$imageIds = @(
    "000000179765", # motorcycle
    "000000190236", # computers / office
    "000000331352", # toilet
    "000000517069", # people / bench
    "000000109976", # kitchen
    "000000012667", # telephone / banana
    "000000482917", # dog
    "000000534605", # person / motorcycles
    "000000289393", # toy animals
    "000000561256", # person / mirror selfie
    "000000338325", # airplane
    "000000071711", # airplanes
    "000000152771", # bicycle
    "000000397133", # COCO sample image
    "000000252219"  # COCO sample image
)

$success = 0
$failed = @()

foreach ($id in $imageIds) {
    $fileName = "$id.jpg"
    $target = Join-Path $destination $fileName
    $url = "https://images.cocodataset.org/val2017/$fileName"

    if (Test-Path $target) {
        Write-Host "[SKIP] Already exists: $fileName" -ForegroundColor Yellow
        $success++
        continue
    }

    try {
        Invoke-WebRequest -Uri $url -OutFile $target
        $item = Get-Item $target
        if ($item.Length -lt 1000) {
            Remove-Item $target -Force
            throw "Downloaded file is unexpectedly small."
        }
        Write-Host "[OK] $fileName ($([math]::Round($item.Length / 1KB)) KB)" -ForegroundColor Green
        $success++
    }
    catch {
        if (Test-Path $target) { Remove-Item $target -Force }
        Write-Host "[FAILED] $fileName - $($_.Exception.Message)" -ForegroundColor Red
        $failed += $fileName
    }
}

$readme = @"
VisionX sample test images
==========================
Source: Microsoft COCO 2017 validation images
Official source: https://cocodataset.org/
Image hosting: https://images.cocodataset.org/val2017/

These images are intended for detector smoke tests and latency/FPS experiments.
They are not a substitute for a labeled evaluation set. Do not report precision,
recall, or mAP unless matching ground-truth annotations are used.

COCO images may have different licenses; check the COCO metadata before
redistributing any image or using it in a public/commercial release.
"@
Set-Content -Path (Join-Path $destination "README_test_data.txt") -Value $readme -Encoding UTF8

Write-Host ""
Write-Host "Download finished: $success of $($imageIds.Count) image entries available." -ForegroundColor Cyan
Write-Host "Folder: $destination"
if ($failed.Count -gt 0) {
    Write-Host "Some downloads failed. Check your internet connection and rerun the script." -ForegroundColor Yellow
    Write-Host "Failed: $($failed -join ', ')"
}
