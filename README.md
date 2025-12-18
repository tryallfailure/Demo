## 📋 Project Overview

This repository contains two main components:
1. **force_predict**: A CNN-based model for predicting 3D force vectors from binary images
2. **propainter**: Video inpainting module for preprocessing and mask generation

The system processes image sequences from live rabbit experiments and test datasets to predict force vectors.
---

## 🗂️ Project Structure

```
├── force_predict/
│   ├── pth/
│   │   └── model.pth              # Trained model weights
│   ├── rabbit/                    # Live rabbit experiment data
│   │   ├── prodata/               # Original raw images
│   │   │   └── frame_0001.jpg
│   │   │   └── ... (frame_0001.jpg - frame_0200.jpg)
│   │   └── binarydata/            # Binarized images
│   │       └── frame_0001.jpg
│   │       └── ... (frame_0001.jpg - frame_0200.jpg)
│   ├── test/                      # Test dataset with ground truth
│   │   ├── prodata/               # Original test images
│   │   ├── binarydata/            # Binarized test images
│   │   └── data/
│   │       └── data.csv           # Ground truth force data
│   ├── utils/
│   │   └── model.py               # Model architecture definition
│   └── main.py                    # Main prediction script
│
├── propainter/                    # Video inpainting module
│   ├── inference_propainter.py    # Inpainting inference script
│   ├── inputs/
│   │   ├── rabbit/
│   │   │   ├── prodata/           # Input video frames
│   │   │   └── mask/              # Input masks
│   │   └── test/
│   │       ├── prodata/           # Input video frames
│   │       └── mask/              # Input masks
│   └── [other ProPainter files]
│
├── requirements.txt               # Python dependencies
└── README.md
```

---

## 🔧 Installation

### System Requirements
- **OS**: Ubuntu 22.04.5 LTS x86_64
- **CPU**: Intel Xeon Platinum 8581C (240 cores)
- **GPU**: NVIDIA RTX A6000
- **CUDA**: 12.8
- **Memory**: 503GB RAM
- **Storage**: 7.0TB (5.5TB available)
- **Python**: 3.8.20 (CPython)

### Environment Setup
1. Clone the repository:
```bash
git clone https://github.com/tryallfailure/Demo.git
cd Demo
```

2. Create conda environment:
```bash
conda create -n force_predict python=3.8 -y
conda activate force_predict
```

3. Install dependencies:
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
cd force_predict
python main.py --data test/binarydata
```

#### Predict forces from rabbit experiment data:
```bash
cd force_predict
python main.py --data rabbit/binarydata
```

#### Input Data Format
- **Image Format**: Binary JPG images (frame_0001.jpg - frame_0200.jpg)
- **Resolution**: Images are resized to 224x224 during preprocessing
- **Ground Truth**: CSV file with columns: `image`, `x`, `y`, `z`

### Video Inpainting

#### Process rabbit experiment video:
```bash
cd propainter
python inference_propainter.py --video inputs/rabbit/prodata --mask inputs/rabbit/mask
```

#### Process test dataset video:
```bash
cd propainter
python inference_propainter.py --video inputs/test/prodata --mask inputs/test/mask
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
