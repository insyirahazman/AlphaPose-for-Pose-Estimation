# AlphaPose Batch Video Processing

This directory contains scripts to process multiple video files automatically using AlphaPose. This is particularly useful when you have many videos to process for Parkinson's Disease detection analysis.

## Files Overview

- `batch_video_demo.py` - Enhanced batch processing script with configuration support and progress tracking
- `batch_config.ini` - Configuration file for customizing processing settings
- `run_batch_video.bat` - Windows batch script for easy execution
- `run_batch_video.ps1` - PowerShell script for easy execution
- `BATCH_README.md` - This file

## Quick Start

### Method 1: Using the Batch Script (Recommended)
```bash
python batch_video_demo.py --input_dir rawvideos/raw_videos/PD --output_dir outputs/PD --cpu --vis_fast
```

### Method 2: Using the Windows Scripts
1. Double-click `run_batch_video.bat` or `run_batch_video.ps1`
2. The script will automatically process all videos in the configured input directory
3. Results will be saved in individual folders within the output directory

## Configuration

Edit `batch_config.ini` to customize processing settings:

### Key Settings:
- **INPUT_DIR**: Directory containing videos to process
- **OUTPUT_DIR**: Where to save processed videos and results
- **MODE**: Detection mode (fast/normal/accurate)
- **CONFIDENCE**: Person detection confidence threshold (0.0-1.0)
- **DETECTION_BATCH_SIZE**: Adjust based on GPU memory
- **POSE_BATCH_SIZE**: Adjust based on GPU memory
- **RESUME**: Skip already processed videos (True/False)
- **FAST_VISUALIZATION**: Enable fast rendering for better performance
- **USE_CPU**: Force CPU processing for Windows compatibility

## Command Line Options

### Batch Script Options:
```bash
python batch_video_demo.py [options]

Options:
  --config, -c          Configuration file path (default: batch_config.ini)
  --input_dir, -i       Input directory (overrides config)
  --output_dir, -o      Output directory (overrides config)
  --dry_run            Show what would be processed without processing
  --cpu                Force CPU processing (overrides config)
  --profile            Enable detailed profiling information
  --vis_fast           Enable fast visualization for faster processing
```

## Output Structure

The batch script creates individual folders for each video:

```
outputs/
├── patient_001/
│   ├── alphapose_output.mp4        # Processed video with pose overlay
│   └── alphapose-results.json      # Pose data in COCO format
├── patient_002/
│   ├── alphapose_output.mp4
│   └── alphapose-results.json
└── ...
```

## Directory Structure

```
AlphaPose-for-PD-Detection/
├── batch_video_demo.py             # Main batch processing script
├── batch_config.ini                # Configuration file
├── run_batch_video.bat             # Windows batch script
├── run_batch_video.ps1             # PowerShell script
├── video_demo.py                   # Single video processing script
└── ...

../PD_rawvideos/                    # Input videos
├── raw_videos/
│   ├── PD/
│   │   ├── front/
│   │   │   ├── PD_front_1.mp4
│   │   │   ├── PD_front_2.mp4
│   │   │   └── ...
│   │   └── side/
│   └── NP/
└── ...

../outputs/                         # Output results
├── batch_results/                  # Batch processing results
│   ├── AlphaPose_PD_front_1.avi   # Processed videos
│   ├── PD_front_1.json            # Pose data
│   └── ...
└── ...
```

## Features

### Resume Processing
- The script can resume interrupted processing
- It skips videos that have already been processed
- Checks for both video output (.mp4) and pose data (.json) in individual folders
- Compatible with both old format (AlphaPose_[video_name].mp4) and new format (alphapose_output.mp4)

### Progress Tracking
- Visual progress bar showing overall batch processing progress
- Shows current video being processed
- Real-time statistics (Success/Failed/Skipped counts)
- Estimates processing time based on video duration
- Displays processing rate and estimated time remaining
- Provides summary statistics at the end

### Error Handling
- Continues processing other videos if one fails
- Provides detailed error messages
- Option to stop or continue after failures

### Flexible Configuration
- Support for multiple video formats (mp4, avi, mov, mkv, wmv, flv)
- Recursive directory search
- Customizable processing parameters
- Both configuration file and command-line options

## Performance Tips

### GPU Memory Optimization:
- Reduce `DETECTION_BATCH_SIZE` if you get CUDA out of memory errors
- Reduce `POSE_BATCH_SIZE` if you get CUDA out of memory errors
- Start with batch sizes of 1 and increase if your GPU can handle it

### Processing Speed:
- Use `fast` mode for faster processing (lower accuracy)
- Enable `FAST_VISUALIZATION` for real-time rendering
- Use `--vis_fast` flag to speed up video rendering

### Storage:
- Set `SAVE_VIDEO = False` if you only need pose data (JSON files)
- Set `SAVE_IMAGES = True` if you want individual frame images

## Troubleshooting

### Common Issues:

1. **CUDA out of memory**
   - Reduce batch sizes in configuration
   - Use `USE_CPU = True` for CPU processing

2. **No videos found**
   - Check the input directory path
   - Verify video file extensions in configuration

3. **Permission errors**
   - Ensure output directory is writable
   - Run with administrator privileges if needed

4. **Python module errors**
   - Make sure all AlphaPose dependencies are installed
   - Check that you're in the correct directory

### Debug Mode:
Use the `--dry_run` flag to see what would be processed without actually processing:
```bash
python batch_video_demo.py --dry_run
```

## Output Files

For each processed video, you'll get organized in individual folders:
- `alphapose_output.mp4` - Processed video with pose overlay
- `alphapose-results.json` - Raw pose estimation data in COCO format

The script supports both old format files (AlphaPose_[video_name].mp4) and new format files for backward compatibility.

The JSON files contain detailed pose keypoint data that can be used for further analysis of Parkinson's Disease symptoms.

## Integration with PD Analysis

The generated JSON files can be used with Parkinson's Disease analysis tools to:
- Analyze gait patterns
- Detect tremor characteristics
- Measure movement amplitude
- Track progression over time

## Support

If you encounter issues:
1. Check the configuration file settings
2. Try processing a single video first with `video_demo.py`
3. Use `--dry_run` to verify settings
4. Check the AlphaPose documentation for model and dependency issues
