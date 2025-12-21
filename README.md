# T-Scope: Trimodal Endoscopic Sensing Platform

This repository provides the complete implementation, datasets, and results for the T-Scope system described in  *“Superelastic Tellurium Thermoelectric Coatings for Advanced Trimodal Microsensing”*.

---

## Contents
1. **EndoForce-Net**: 3D force-vector regression from thermoelectric imprint images  
2. **Painting restoration**: removal of Te-pattern occlusion from endoscopic video  
3. **Datasets & results**: in-vivo rabbit sequences and ex-vivo test sets with ground-truth force labels

---

## 🗂️ Project Structure

```
T-Scope/
├── EndoForce-Net/
│   ├── weights/
│   │   └── EndoForce_net.pth        # trained model
│   ├── rabbit/
│   │   ├── raw_frames/              # 200 original frames
│   │   └── binary_frames/           # segmented imprint images
│   ├── test/
│   │   ├── raw_frames/
│   │   ├── binary_frames/
│   │   └── ground_truth/
│   │       └── forces.csv           # X,Y,Z labels [N]
│   ├── utils/
│   │   └── EndoForce_net.py         # network definition
│   └── infer.py                     # inference script
│
├── painting/
│   ├── restore.py                   # inpainting entry point
│   ├── inputs/
│   │   ├── rabbit/
│   │   │   ├── frames/
│   │   │   └── masks/               # Te-marker masks
│   │   └── test/
│   │       ├── frames/
│   │       └── masks/
│   └── outputs/                     # restored videos
│
└── requirements.txt
└── README.md
```

---

## 🔧 Installation

## Hardware & Software Environment
- **OS**: Ubuntu 22.04.5 LTS x86_64
- **CPU**: Intel Xeon Platinum 8581C (240 cores)
- **GPU**: NVIDIA RTX A6000
- **CUDA**: 12.8
- **Memory**: 503GB RAM
- **Storage**: 7.0TB (5.5TB available)
- **Python**: 3.8.20 (CPython)

### Environment Setup

1. Create conda environment:
```bash
conda create -n tscope python=3.8 -y
conda activate tscope
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Required Dependencies
- torch >= 1.7.1
- torchvision >= 0.8.2
- numpy
- pandas
- matplotlib
- pillow
- opencv-python
- scikit-image
- [other dependencies from requirements.txt]

---

## 🚀 Usage

### Force Prediction

#### Predict forces from test dataset (with ground truth comparison):
```bash
cd EndoForce-Net
python infer.py --data test/binarydata --gt test/ground_truth/forces.csv
```

#### Predict forces from rabbit experiment data:
```bash
python infer.py --data rabbit/binarydata
```

#### Input Data Format
- **Image Format**: Binary JPG images (frame_0001.jpg - frame_0200.jpg)
- **Resolution**: Images are resized to 224x224 during preprocessing
- **Ground Truth**: CSV file with columns: `image`, `x`, `y`, `z`

### Video Inpainting

#### Process rabbit experiment video:
```bash
cd painting
python restore.py --video inputs/rabbit/prodata --mask inputs/rabbit/mask
```

#### Process test dataset video:
```bash
python restore.py --video inputs/test/prodata --mask inputs/test/mask
```

#### Output Files
- **inpaint_out.mp4**: Restored video (saved in results/)
- **masked_in.mp4**: Masked input video (saved in results/)

### ProPainter Reference
This module is based on [ProPainter](https://github.com/sczhou/ProPainter) with minor modifications for this project.

---

## 📈 Output Examples

### Force Prediction Output
```
frame_0001  Pred_X: 0.021  Pred_Y: -0.042  Pred_Z: -0.405
frame_0002  Pred_X: 0.019  Pred_Y: -0.044  Pred_Z: -0.263
...
```

### Visualization
- **Test Dataset**: Plot comparing predicted vs true forces (X, Y, Z components)
- **Rabbit Dataset**: Force prediction curves over time

---

## 📝 Data Description

### Rabbit Dataset
- **Source**: Live rabbit experiments
- **Images**: 200 frames (frame_0001.jpg - frame_0200.jpg)
- **Formats**: 
  - `prodata/`: Raw original images
  - `binarydata/`: Binarized images for model input

### Test Dataset
- **Images**: Binary images for testing
- **Ground Truth**: data.csv containing:
  - Column 1: Image name (e.g., "frame_0001")
  - Column 2-4: True X, Y, Z force values


---

## 🙏 Acknowledgments

- ProPainter module based on [sczhou/ProPainter](https://github.com/sczhou/ProPainter)
- Model architecture based on ResNet-18
