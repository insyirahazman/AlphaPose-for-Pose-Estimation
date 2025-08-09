# AlphaPose for Parkinson's Disease Detection

This repository is an enhanced version of [AlphaPose](https://github.com/MVIG-SJTU/AlphaPose) specifically adapted for **Parkinson's Disease (PD) detection and analysis**. It has been optimized to run on **CPU-only machines** and includes advanced batch processing capabilities for analyzing multiple patient videos efficiently.

## 💡 Overview

AlphaPose is a state-of-the-art multi-person pose estimator. This specialized version has been tailored for:
- **Parkinson's Disease research and clinical analysis**
- **CPU-only environments** (no GPU required)
- **Batch processing** of multiple patient videos
- **Gait analysis** and movement pattern detection
- **Clinical research applications**

## 🛠️ Key Features

- ✅ **Multi-person pose estimation** optimized for medical analysis
- ✅ **CPU-only processing** - no GPU dependencies
- ✅ **Batch video processing** with progress tracking and resume functionality
- ✅ **Configuration-driven processing** with customizable parameters
- ✅ **Individual patient folders** for organized output
- ✅ **JSON pose data export** for further analysis
- ✅ **Processing status tracking** (complete/incomplete/missing)
- ✅ **Memory optimization** for large video datasets
- ⚠️ Slower inference compared to GPU-based processing

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/insyirahazman/AlphaPose-for-PD-Detection.git
cd AlphaPose-for-PD-Detection
```

### 2. Install Dependencies

```bash
# Install PyTorch CPU version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other requirements
pip install -r requirements.txt
```

### 3. Basic Usage

#### Single Video Processing
```bash
python video_demo.py --video input_video.mp4 --outdir output_folder --cpu
```

#### Batch Processing (Recommended for PD Analysis)
```bash
python batch_video_demo.py --input_dir "path/to/patient/videos" --output_dir "path/to/results" --cpu --vis_fast
```

## 📁 Project Structure

```
AlphaPose-for-PD-Detection/
├── 📄 batch_video_demo.py          # Enhanced batch processing script
├── ⚙️ batch_config.ini             # Configuration file for batch processing
├── 🔧 run_batch_video.bat          # Windows batch script
├── 🔧 run_batch_video.ps1          # PowerShell script
├── 📄 video_demo.py                # Single video processing script
├── 📄 demo.py                      # Image processing demo
├── 📚 BATCH_README.md              # Detailed batch processing documentation
├── 📚 README.md                    # This file
├── 📦 requirements.txt             # Python dependencies
├── 📁 models/                      # Pre-trained models
├── 📁 examples/                    # Example images and videos
├── 📁 SPPE/                        # Single Person Pose Estimation module
├── 📁 yolo/                        # YOLO detection module
└── 📁 outputs/                     # Processing results
```

## 🎯 For Parkinson's Disease Research

### Typical Workflow
1. **Collect patient videos** (walking, movement tasks)
2. **Organize videos** by patient ID and condition
3. **Configure batch processing** parameters
4. **Run batch analysis** on all videos
5. **Extract pose data** for gait analysis
6. **Analyze movement patterns** for PD indicators

### Output Structure
```
outputs/batch_results/
├── patient_001_front/
│   ├── alphapose_output.mp4        # Video with pose overlay
│   └── alphapose-results.json      # Pose keypoints data
├── patient_001_side/
│   ├── alphapose_output.mp4
│   └── alphapose-results.json
├── patient_002_front/
└── ...
```

### Configuration for PD Analysis
```ini
[DETECTION]
MODE = fast                         # Fast processing for clinical use
CONFIDENCE = 0.05                   # Lower threshold for medical analysis
NMS_THRESHOLD = 0.6

[OUTPUT]
SAVE_VIDEO = True                   # Keep videos for clinical review
FAST_VISUALIZATION = True           # Optimize for speed
SAVE_IMAGES = False                 # Save space for large datasets

[PROCESSING]
USE_CPU = True                      # CPU-only for compatibility
RESUME = True                       # Resume interrupted processing
VIDEO_EXTENSIONS = mp4,avi,mov      # Common medical video formats
```

## 🔧 Advanced Usage

### Command Line Options
```bash
# Dry run to check what will be processed
python batch_video_demo.py --dry_run

# Process specific directory with custom output
python batch_video_demo.py -i "PD_videos/front" -o "results/front_analysis"

# Enable profiling for performance analysis
python batch_video_demo.py --profile --cpu

# Use custom configuration file
python batch_video_demo.py --config "pd_config.ini"
```

### Batch Processing Features
- **Resume capability**: Skip already processed videos
- **Progress tracking**: Real-time progress bars and status updates
- **Error handling**: Continue processing after failures
- **Memory management**: Optimized for large video datasets
- **Status checking**: Track complete/incomplete/missing results

## 📊 Output Data Format

### JSON Pose Data Structure
```json
{
  "version": "1.3",
  "people": [
    {
      "pose_keypoints_2d": [
        x1, y1, confidence1,    # Nose
        x2, y2, confidence2,    # Left Eye
        x3, y3, confidence3,    # Right Eye
        ...                     # 17 keypoints total
      ],
      "face_keypoints_2d": [...],
      "hand_left_keypoints_2d": [...],
      "hand_right_keypoints_2d": [...]
    }
  ]
}
```

### Keypoint Indices (COCO Format)
- 0: Nose, 1: Left Eye, 2: Right Eye
- 3: Left Ear, 4: Right Ear
- 5: Left Shoulder, 6: Right Shoulder
- 7: Left Elbow, 8: Right Elbow
- 9: Left Wrist, 10: Right Wrist
- 11: Left Hip, 12: Right Hip
- 13: Left Knee, 14: Right Knee
- 15: Left Ankle, 16: Right Ankle

## 🔬 Research Applications

### Gait Analysis Parameters
- **Step length**: Distance between consecutive footsteps
- **Step width**: Lateral distance between feet
- **Cadence**: Steps per minute
- **Symmetry**: Left vs right step characteristics
- **Variability**: Consistency of gait parameters

### Movement Analysis
- **Tremor detection**: High-frequency oscillations in keypoints
- **Bradykinesia**: Slowness of movement initiation and execution
- **Postural instability**: Balance and stability metrics
- **Rigidity indicators**: Reduced range of motion

## 📚 Documentation

- **[BATCH_README.md](BATCH_README.md)** - Comprehensive batch processing guide
- **[examples/](examples/)** - Sample videos and expected outputs
- **[doc/](doc/)** - Additional documentation and guides

## ⚡ Performance Tips

### For Large Datasets
- Use `--vis_fast` for faster processing
- Set `DETECTION_BATCH_SIZE = 1` for memory efficiency
- Enable `RESUME = True` for interrupted processing
- Process videos in smaller batches if memory limited

### For Clinical Use
- Use `MODE = fast` for real-time analysis
- Set `CONFIDENCE = 0.05` for sensitive detection
- Enable `SAVE_VIDEO = True` for clinical review
- Organize videos by patient ID and session

## 🤝 Contributors

**Parkinson's Disease Detection Adaptation:**
- **[insyirahazman](https://github.com/insyirahazman)** - PD-specific modifications and batch processing enhancements

**Original AlphaPose Development:**
- [Jiefeng Li](http://jeff-leaf.site/) - Core development
- [Hao-Shu Fang](https://fang-haoshu.github.io/) - Core development  
- [Yuliang Xiu](http://xiuyuliang.cn) - Core development
- [Cewu Lu](http://www.mvig.org/) - Project supervision

## 📖 References

### Technical References
1. [Original AlphaPose Repository](https://github.com/MVIG-SJTU/AlphaPose)
2. [AlphaPose PyTorch Implementation](https://github.com/MVIG-SJTU/AlphaPose/tree/pytorch)
3. [Windows Installation Guide](https://github.com/Amanbhandula/AlphaPose/blob/master/doc/win_install.md)
4. [Real-time Pose Estimation Tutorial](https://debuggercafe.com/real-time-pose-estimation-using-alphapose-pytorch-and-deep-learning/)

### Medical/Clinical References
5. [Gait Analysis in Parkinson's Disease](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6367956/)
6. [Computer Vision in Medical Diagnosis](https://www.nature.com/articles/s41591-020-0842-3)
7. [Pose Estimation for Healthcare](https://ieeexplore.ieee.org/document/9153574)

## 🔬 Research Citation

If you use this adapted version for Parkinson's Disease research, please cite:

```bibtex
@misc{alphapose-pd-2024,
  title={AlphaPose for Parkinson's Disease Detection: CPU-Optimized Batch Processing},
  author={Insyirah Azman},
  year={2024},
  howpublished={\url{https://github.com/insyirahazman/AlphaPose-for-PD-Detection}}
}
```

**Original AlphaPose Citation:**
```bibtex
@inproceedings{fang2017rmpe,
  title={{RMPE}: Regional Multi-person Pose Estimation},
  author={Fang, Hao-Shu and Xie, Shuqin and Tai, Yu-Wing and Lu, Cewu},
  booktitle={ICCV},
  year={2017}
}

@article{li2018crowdpose,
  title={CrowdPose: Efficient Crowded Scenes Pose Estimation and A New Benchmark},
  author={Li, Jiefeng and Wang, Can and Zhu, Hao and Mao, Yihuan and Fang, Hao-Shu and Lu, Cewu},
  journal={arXiv preprint arXiv:1812.00324},
  year={2018}
}
```

## 📄 License

This project maintains the same license as the original AlphaPose project. Please refer to the original repository for license details.

---

## 🆘 Support & Issues

- **Issues**: [GitHub Issues](https://github.com/insyirahazman/AlphaPose-for-PD-Detection/issues)
- **Documentation**: [BATCH_README.md](BATCH_README.md) for detailed batch processing guide
- **Original AlphaPose**: [AlphaPose Repository](https://github.com/MVIG-SJTU/AlphaPose) for core AlphaPose issues

---

**⭐ Star this repository if it helps your research!**
    }

    @inproceedings{xiu2018poseflow,
      author = {Xiu, Yuliang and Li, Jiefeng and Wang, Haoyu and Fang, Yinghong and Lu, Cewu},
      title = {{Pose Flow}: Efficient Online Pose Tracking},
      booktitle={BMVC},
      year = {2018}
    }
