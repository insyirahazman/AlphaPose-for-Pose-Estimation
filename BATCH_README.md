# 🚀 AlphaPose Batch Video Processing for Parkinson's Disease Analysis

This comprehensive guide covers the enhanced batch processing capabilities specifically designed for **Parkinson's Disease (PD) detection and analysis**. The batch processing system allows you to efficiently process multiple patient videos with advanced features like progress tracking, resume capability, and organized output management.

## 📁 Files Overview

| File | Description | Purpose |
|------|-------------|---------|
| `batch_video_demo.py` | **Main batch processing script** | Enhanced script with configuration support, progress tracking, and resume functionality |
| `batch_config.ini` | **Configuration file** | Customizable processing parameters for different analysis needs |
| `run_batch_video.bat` | **Windows batch script** | One-click execution for Windows users |
| `run_batch_video.ps1` | **PowerShell script** | Advanced Windows execution with better error handling |
| `BATCH_README.md` | **This documentation** | Comprehensive guide for batch processing |

## ⚡ Quick Start Guide

### Method 1: Command Line (Recommended)

```bash
# Basic batch processing
python batch_video_demo.py --input_dir "rawvideos/PD_videos" --output_dir "outputs/PD_analysis" --cpu --vis_fast

# With custom configuration
python batch_video_demo.py --config "pd_config.ini"

# Dry run to preview what will be processed
python batch_video_demo.py --dry_run
```

### Method 2: Windows Scripts

1. **Edit the configuration** in `batch_config.ini`
2. **Double-click** `run_batch_video.bat` or right-click → "Run with PowerShell" on `run_batch_video.ps1`
3. **Monitor progress** in the console window
4. **Check results** in the configured output directory

## 🎛️ Configuration Management

### Sample Configuration File (`batch_config.ini`)

```ini
[PATHS]
# Input directory containing patient videos
INPUT_DIR = ../rawvideos/sample_videos/Front
# Output directory for processed results
OUTPUT_DIR = ../outputs/batch_results

[DETECTION]
# Processing mode: fast (recommended), normal, accurate
MODE = fast
# Person detection confidence (0.0-1.0, lower = more sensitive)
CONFIDENCE = 0.05
# Non-maximum suppression threshold
NMS_THRESHOLD = 0.6
# Batch sizes (adjust based on available memory)
DETECTION_BATCH_SIZE = 1
POSE_BATCH_SIZE = 1

[OUTPUT]
# Save processed video with pose overlay
SAVE_VIDEO = True
# Enable fast visualization (recommended for clinical use)
FAST_VISUALIZATION = True
# Save individual frame images (not recommended for large datasets)
SAVE_IMAGES = False

[PROCESSING]
# Force CPU processing (recommended for compatibility)
USE_CPU = True
# Resume interrupted processing (skip completed videos)
RESUME = True
# Video file extensions to process
VIDEO_EXTENSIONS = mp4,avi,mov,mkv

[ADVANCED]
# Input image dimensions (lower = faster, higher = more accurate)
INPUT_DIM = 416
# Output format for pose data
OUTPUT_FORMAT = coco
# Enable detailed profiling information
SHOW_PROFILE = False
```

### Configuration for Different Scenarios

#### Clinical Analysis (High Sensitivity)
```ini
[DETECTION]
MODE = normal
CONFIDENCE = 0.03
DETECTION_BATCH_SIZE = 1
POSE_BATCH_SIZE = 1

[ADVANCED]
INPUT_DIM = 608
```

#### Fast Screening (High Speed)
```ini
[DETECTION]
MODE = fast
CONFIDENCE = 0.1
DETECTION_BATCH_SIZE = 2
POSE_BATCH_SIZE = 2

[ADVANCED]
INPUT_DIM = 320
```

#### Research Quality (High Accuracy)
```ini
[DETECTION]
MODE = accurate
CONFIDENCE = 0.05
DETECTION_BATCH_SIZE = 1
POSE_BATCH_SIZE = 1

[ADVANCED]
INPUT_DIM = 832
```

## 🔧 Command Line Options

### Complete Options Reference

```bash
python batch_video_demo.py [OPTIONS]

Required Arguments:
  None (uses configuration file or defaults)

Optional Arguments:
  --config, -c PATH          Configuration file path (default: batch_config.ini)
  --input_dir, -i PATH       Input directory (overrides config)
  --output_dir, -o PATH      Output directory (overrides config)
  
Processing Options:
  --cpu                      Force CPU processing (overrides config)
  --vis_fast                 Enable fast visualization (overrides config)
  --profile                  Enable detailed profiling information
  
Control Options:
  --dry_run                  Preview what will be processed (no actual processing)

Examples:
  # Basic usage with CPU processing
  python batch_video_demo.py --cpu
  
  # Process specific directories
  python batch_video_demo.py -i "PD_videos/front" -o "results/front_analysis"
  
  # Fast processing with profiling
  python batch_video_demo.py --vis_fast --profile --cpu
  
  # Preview processing without execution
  python batch_video_demo.py --dry_run
```

## 📊 Output Structure & Organization

### Directory Organization
```
outputs/batch_results/
├── 📁 patient_001_front/           # Individual patient folder
│   ├── 🎥 alphapose_output.mp4     # Processed video with pose overlay
│   └── 📄 alphapose-results.json   # Pose keypoints data (COCO format)
├── 📁 patient_001_side/
│   ├── 🎥 alphapose_output.mp4
│   └── 📄 alphapose-results.json
├── 📁 patient_002_front/
│   ├── 🎥 alphapose_output.mp4
│   └── 📄 alphapose-results.json
└── 📁 patient_002_side/
    ├── 🎥 alphapose_output.mp4
    └── 📄 alphapose-results.json
```

### File Naming Convention
- **Video files**: `alphapose_output.mp4` (consistent naming for easy batch analysis)
- **JSON files**: `alphapose-results.json` (COCO format pose data)
- **Folders**: Named after the original video file (without extension)

### Processing Status Indicators
- **[COMPLETE]**: Both video and JSON files exist
- **[INCOMPLETE]**: Only one file exists (will be reprocessed)
- **[MISSING]**: No output files found (will be processed)

## 🔄 Advanced Features

### Resume Functionality
The batch processor automatically detects and skips already completed videos:

```bash
# Processing status check example
Found 5 video files:
Checking existing output files...
============================================================
   1. patient_001_front.mp4 [COMPLETE]
   2. patient_001_side.mp4 [INCOMPLETE]  
   3. patient_002_front.mp4 [MISSING]
   4. patient_002_side.mp4 [MISSING]
   5. patient_003_front.mp4 [COMPLETE]

Status Summary:
  [COMPLETE] (will skip): 2
  [INCOMPLETE] (will reprocess): 1  
  [MISSING] (will process): 2
  Total videos to process: 3
```

### Progress Tracking
Real-time progress monitoring with detailed information:

```bash
================================================================================
[PROGRESS: 2/5] (40.0% complete)
Processing: patient_001_side

Input: D:\PD\rawvideos\patient_001_side.mp4
Output directory: D:\PD\outputs\patient_001_side
Using output directory: D:\PD\outputs (video_demo.py will create subfolder: patient_001_side)
Estimated processing time: 2m 30s
================================================================================

[PROGRESS UPDATE]
  Processed: 2/5 (40.0%)
  Success: 1 | Failed: 0 | Skipped: 1
  Time elapsed: 5m 12s
  Estimated remaining: 7m 48s
```

### Memory Management
Optimized settings for different hardware configurations:

#### Low Memory Systems (< 8GB RAM)
```ini
[DETECTION]
DETECTION_BATCH_SIZE = 1
POSE_BATCH_SIZE = 1

[ADVANCED]
INPUT_DIM = 320
```

#### Standard Systems (8-16GB RAM)
```ini
[DETECTION]
DETECTION_BATCH_SIZE = 1
POSE_BATCH_SIZE = 1

[ADVANCED]
INPUT_DIM = 416
```

#### High Memory Systems (> 16GB RAM)
```ini
[DETECTION]
DETECTION_BATCH_SIZE = 2
POSE_BATCH_SIZE = 2

[ADVANCED]
INPUT_DIM = 608
```

## 🔬 Parkinson's Disease Analysis Workflow

### Typical Research Pipeline

1. **Data Collection**
   ```
   📹 Record patient videos (front view, side view)
   📁 Organize by patient ID and view angle
   📝 Maintain clinical metadata
   ```

2. **Batch Processing Setup**
   ```bash
   # Configure for clinical analysis
   python batch_video_demo.py --config clinical_config.ini --dry_run
   ```

3. **Processing Execution**
   ```bash
   # Run batch processing
   python batch_video_demo.py --config clinical_config.ini
   ```

4. **Data Analysis**
   ```
   📊 Extract pose keypoints from JSON files
   📈 Calculate gait parameters
   🔬 Statistical analysis of movement patterns
   ```

### Clinical Configuration Example

```ini
[PATHS]
INPUT_DIR = /clinical_data/PD_patients/videos
OUTPUT_DIR = /clinical_data/PD_patients/pose_analysis

[DETECTION]
MODE = normal                    # Balance of speed and accuracy
CONFIDENCE = 0.03               # High sensitivity for clinical use
NMS_THRESHOLD = 0.6
DETECTION_BATCH_SIZE = 1        # Conservative for stability
POSE_BATCH_SIZE = 1

[OUTPUT]
SAVE_VIDEO = True               # Keep for clinical review
FAST_VISUALIZATION = True       # Optimize processing speed
SAVE_IMAGES = False            # Save storage space

[PROCESSING]
USE_CPU = True                 # Ensure compatibility
RESUME = True                  # Handle large datasets
VIDEO_EXTENSIONS = mp4,avi,mov

[ADVANCED]
INPUT_DIM = 608               # Higher accuracy for clinical use
OUTPUT_FORMAT = coco
SHOW_PROFILE = False
```

## 🚨 Troubleshooting Guide

### Common Issues & Solutions

#### 1. Memory Errors
**Problem**: `MemoryError` or system freezing
**Solution**: 
```ini
# Reduce batch sizes
DETECTION_BATCH_SIZE = 1
POSE_BATCH_SIZE = 1

# Lower input dimensions
INPUT_DIM = 320

# Process smaller batches
```

#### 2. CUDA Errors (on CPU-only systems)
**Problem**: CUDA-related errors despite CPU flag
**Solution**:
```bash
# Ensure CPU processing is forced
python batch_video_demo.py --cpu

# Check configuration file
USE_CPU = True
```

#### 3. File Permission Errors
**Problem**: Cannot create output directories
**Solution**:
```bash
# Run with administrator privileges (Windows)
# Check write permissions on output directory
# Ensure output path is valid and accessible
```

#### 4. Video Codec Issues
**Problem**: Cannot process certain video formats
**Solution**:
```bash
# Install additional codecs
pip install opencv-python-headless

# Convert videos to supported formats (MP4, AVI)
```

#### 5. Processing Stops/Hangs
**Problem**: Processing appears to freeze
**Solution**:
```bash
# Use resume functionality
python batch_video_demo.py --config your_config.ini

# Check for completed files in output directory
# Reduce batch sizes if memory-related
```

### Performance Optimization

#### For Large Datasets (100+ videos)
```bash
# Enable fast mode with resume
python batch_video_demo.py --vis_fast --config batch_config.ini

# Process in smaller chunks if needed
# Monitor system resources during processing
```

#### For High-Quality Analysis
```bash
# Use normal or accurate mode
# Increase input dimensions
# Ensure adequate processing time
```

## 📈 Performance Benchmarks

### Processing Speed Estimates

| Video Duration | Resolution | Mode | CPU Time (approx.) |
|----------------|------------|------|-------------------|
| 30 seconds     | 1080p      | Fast | 1-2 minutes      |
| 30 seconds     | 1080p      | Normal | 2-3 minutes    |
| 30 seconds     | 1080p      | Accurate | 3-5 minutes  |
| 2 minutes      | 1080p      | Fast | 4-8 minutes      |
| 2 minutes      | 1080p      | Normal | 8-12 minutes   |

*Note: Times vary significantly based on hardware and video complexity*

### Hardware Recommendations

#### Minimum Requirements
- **CPU**: Intel i5 or AMD Ryzen 5 (4+ cores)
- **RAM**: 8GB minimum
- **Storage**: 10GB free space per hour of video
- **OS**: Windows 10/11, Linux, macOS

#### Recommended Configuration
- **CPU**: Intel i7 or AMD Ryzen 7 (8+ cores)
- **RAM**: 16GB or more
- **Storage**: SSD with 50GB+ free space
- **OS**: Windows 11 or Ubuntu 20.04+

## 🔗 Integration with Analysis Tools

### Python Analysis Example

```python
import json
import pandas as pd
import numpy as np

def extract_gait_data(json_file):
    """Extract gait parameters from AlphaPose JSON output"""
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Extract keypoint data for analysis
    frames = []
    for frame in data:
        if 'people' in frame and len(frame['people']) > 0:
            person = frame['people'][0]  # First person detected
            keypoints = np.array(person['pose_keypoints_2d']).reshape(-1, 3)
            frames.append(keypoints)
    
    return np.array(frames)

# Process all JSON files in batch
import glob
results = {}
for json_file in glob.glob('outputs/batch_results/*/alphapose-results.json'):
    patient_id = json_file.split('/')[-2]
    results[patient_id] = extract_gait_data(json_file)
```

### R Analysis Example

```r
library(jsonlite)
library(dplyr)

# Function to load and process JSON files
process_alphapose_data <- function(json_path) {
  data <- fromJSON(json_path)
  # Process pose keypoints for gait analysis
  # Calculate stride length, cadence, etc.
  return(processed_data)
}

# Batch process all results
json_files <- list.files("outputs/batch_results", 
                         pattern = "alphapose-results.json", 
                         recursive = TRUE, 
                         full.names = TRUE)

results <- map_dfr(json_files, process_alphapose_data)
```

## 📋 Best Practices

### For Clinical Research
1. **Standardize recording conditions** (lighting, camera angle, distance)
2. **Use consistent naming conventions** for patient IDs
3. **Process in batches** to maintain consistency
4. **Backup original videos** before processing
5. **Document processing parameters** for reproducibility

### For Large Datasets
1. **Enable resume functionality** for interrupted processing
2. **Monitor disk space** during processing
3. **Process overnight** for large batches
4. **Use fast mode** for initial screening
5. **Validate random samples** for quality control

### For Accuracy
1. **Use higher input dimensions** (608 or 832)
2. **Lower confidence thresholds** (0.03-0.05)
3. **Manual verification** of critical cases
4. **Multiple view angles** when possible
5. **Consistent lighting conditions**

---

## 📞 Support & Further Information

- **Main Documentation**: [README.md](README.md)
- **Issues & Support**: [GitHub Issues](https://github.com/insyirahazman/AlphaPose-for-PD-Detection/issues)
- **Original AlphaPose**: [MVIG-SJTU/AlphaPose](https://github.com/MVIG-SJTU/AlphaPose)

---

**💡 Tip**: Start with a small batch of videos to test your configuration before processing large datasets!
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
