# AlphaPose Batch Video Processing Script for PowerShell
# This script processes all videos in the PD_rawvideos directory

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "AlphaPose Batch Video Processing" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Set the paths
$InputDir = "..\PD_rawvideos\raw_videos"
$OutputDir = "..\outputs\batch_results"

# Create output directory if it doesn't exist
if (!(Test-Path $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
    Write-Host "Created output directory: $OutputDir" -ForegroundColor Green
}

Write-Host "Input directory: $InputDir" -ForegroundColor Yellow
Write-Host "Output directory: $OutputDir" -ForegroundColor Yellow
Write-Host ""

# Check if Python is available
try {
    $pythonVersion = python --version 2>&1
    Write-Host "Python version: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "Error: Python is not available in PATH" -ForegroundColor Red
    Write-Host "Please install Python or add it to your PATH" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Run the batch processing script
Write-Host "Starting batch processing..." -ForegroundColor Green
Write-Host ""

$arguments = @(
    "batch_video_demo.py",
    "--input_dir", $InputDir,
    "--output_dir", $OutputDir,
    "--mode", "normal",
    "--conf", "0.05",
    "--nms", "0.6",
    "--detbatch", "1",
    "--posebatch", "80",
    "--resume"
)

try {
    & python $arguments
    $exitCode = $LASTEXITCODE
    
    Write-Host ""
    if ($exitCode -eq 0) {
        Write-Host "========================================" -ForegroundColor Green
        Write-Host "Batch processing completed successfully!" -ForegroundColor Green
        Write-Host "Check the output directory: $OutputDir" -ForegroundColor Green
        Write-Host "========================================" -ForegroundColor Green
    } else {
        Write-Host "========================================" -ForegroundColor Red
        Write-Host "Batch processing completed with errors!" -ForegroundColor Red
        Write-Host "Exit code: $exitCode" -ForegroundColor Red
        Write-Host "========================================" -ForegroundColor Red
    }
} catch {
    Write-Host "Error running batch processing: $_" -ForegroundColor Red
}

Read-Host "Press Enter to exit"
