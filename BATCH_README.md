# AlphaPose Batch Video Processing

This directory contains scripts to process multiple video files automatically using AlphaPose. This is particularly useful when you have many videos to process for Parkinson's Disease detection analysis.

## Files Overview

- `batch_video_demo.py` - Basic batch processing script
- `batch_video_demo_enhanced.py` - Enhanced version with configuration file support
- `batch_config.ini` - Configuration file for customizing processing settings
- `run_batch_video.bat` - Windows batch script for easy execution
- `run_batch_video.ps1` - PowerShell script for easy execution
- `BATCH_README.md` - This file

## Quick Start

### Method 1: Using the Batch Script (Recommended for Windows)
1. Double-click `run_batch_video.bat` or `run_batch_video.ps1`
2. The script will automatically process all videos in `../PD_rawvideos/raw_videos/`
3. Results will be saved in `../outputs/batch_results/`

### Method 2: Using the Enhanced Python Script
```bash
python batch_video_demo_enhanced.py
```

### Method 3: Using the Basic Python Script
```bash
python batch_video_demo.py --input_dir ../PD_rawvideos/raw_videos --output_dir ../outputs/batch_results
```

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

## Command Line Options

### Enhanced Script Options:
```bash
python batch_video_demo_enhanced.py [options]

Options:
  --config, -c          Configuration file path (default: batch_config.ini)
  --input_dir, -i       Input directory (overrides config)
  --output_dir, -o      Output directory (overrides config)
  --dry_run            Show what would be processed without processing
```

### Basic Script Options:
```bash
python batch_video_demo.py [options]

Required:
  --input_dir, -i       Directory containing input videos
  --output_dir, -o      Directory to save processed videos

Optional:
  --mode               Detection mode: fast/normal/accurate (default: normal)
  --conf               Confidence threshold (default: 0.05)
  --nms                NMS threshold (default: 0.6)
  --detbatch           Detection batch size (default: 1)
  --posebatch          Pose estimation batch size (default: 80)
  --vis_fast           Use fast visualization
  --cpu                Use CPU instead of GPU
  --extensions         Video file extensions to process
  --resume             Skip already processed videos
```

## Directory Structure

```
AlphaPose-for-PD-Detection/
├── batch_video_demo.py              # Basic batch script
├── batch_video_demo_enhanced.py     # Enhanced batch script
├── batch_config.ini                 # Configuration file
├── run_batch_video.bat             # Windows batch script
├── run_batch_video.ps1             # PowerShell script
├── video_demo.py                   # Original single video script
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
- Checks for both video output (.avi) and pose data (.json)

### Progress Tracking
- Shows progress for each video
- Estimates processing time based on video duration
- Displays real-time processing output
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
python batch_video_demo_enhanced.py --dry_run
```

## Output Files

For each processed video, you'll get:
- `AlphaPose_[video_name].avi` - Processed video with pose overlay
- `[video_name].json` - Raw pose estimation data in COCO format

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
