# AlphaPose for Pose Estimation (CPU Version)

This repository updates and adjusts the existing [AlphaPose](https://github.com/MVIG-SJTU/AlphaPose) implementation to ensure it can run smoothly on **CPU-only machines**. This adaptation is useful for researchers or students who do not have access to an NVIDIA GPU, but still want to experiment with human pose estimation using AlphaPose.

## 💡 Overview

AlphaPose is an accurate multi-person pose estimator that achieves state-of-the-art performance on human pose tracking and detection tasks. This version has been tailored to:
- Remove or bypass CUDA dependencies.
- Run on systems without an NVIDIA GPU.
- Be suitable for testing, learning, and small-scale CPU-based applications.

## 🛠️ Features

- ✅ Multi-person pose estimation
- ✅ Runs on **CPU-only** environment
- ✅ Uses pre-trained models from original AlphaPose
- ⚠️ Slower inference compared to GPU-based processing

## 🚀 Getting Started

### 1. Clone the Repository

You will need Git installed. You can download and install [Git Bash here](https://git-scm.com/downloads) if you haven't already.

Once installed, open Git Bash and run:

git clone https://github.com/your-username/AlphaPose-CPU.git
cd AlphaPose-CPU

### 2. Install Python Dependencies

## Contributors
Pytorch version of AlphaPose is developed and maintained by [Jiefeng Li](http://jeff-leaf.site/), [Hao-Shu Fang](https://fang-haoshu.github.io/), [Yuliang Xiu](http://xiuyuliang.cn) and [Cewu Lu](http://www.mvig.org/). 

## Citation
Please cite these papers in your publications if it helps your research:

    @inproceedings{fang2017rmpe,
      title={{RMPE}: Regional Multi-person Pose Estimation},
      author={Fang, Hao-Shu and Xie, Shuqin and Tai, Yu-Wing and Lu, Cewu},
      booktitle={ICCV},
      year={2017}
    }

    @inproceedings{xiu2018poseflow,
      author = {Xiu, Yuliang and Li, Jiefeng and Wang, Haoyu and Fang, Yinghong and Lu, Cewu},
      title = {{Pose Flow}: Efficient Online Pose Tracking},
      booktitle={BMVC},
      year = {2018}
    }
