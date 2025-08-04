@echo off
REM Batch processing script for AlphaPose video demo on Windows
REM This script processes all videos in the PD_rawvideos directory

echo ========================================
echo AlphaPose Batch Video Processing
echo ========================================

REM Set the paths
set "INPUT_DIR=..\PD_rawvideos\raw_videos"
set "OUTPUT_DIR=..\outputs\batch_results"

REM Create output directory if it doesn't exist
if not exist "%OUTPUT_DIR%" (
    mkdir "%OUTPUT_DIR%"
)

echo Input directory: %INPUT_DIR%
echo Output directory: %OUTPUT_DIR%
echo.

REM Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python is not available in PATH
    echo Please install Python or add it to your PATH
    pause
    exit /b 1
)

REM Run the batch processing script
echo Starting batch processing...
echo.

python batch_video_demo.py ^
    --input_dir "%INPUT_DIR%" ^
    --output_dir "%OUTPUT_DIR%" ^
    --mode normal ^
    --conf 0.05 ^
    --nms 0.6 ^
    --detbatch 1 ^
    --posebatch 80 ^
    --save_video ^
    --resume

echo.
echo ========================================
echo Batch processing completed!
echo Check the output directory: %OUTPUT_DIR%
echo ========================================
pause
