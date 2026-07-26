# A lightweight deep learning framework for automated wheat tiller phenotyping via multi-view imaging and spatially constrained instance segmentation

Official implementation of **WTLR-Net**, a lightweight instance segmentation network for automatic wheat tiller segmentation and counting in field environments.

WTLR-Net is developed based on the **Ultralytics YOLO11 framework** and is designed for efficient wheat tiller phenotyping under complex field conditions.

---

## Overview

WTLR-Net improves the YOLO11 segmentation framework by introducing lightweight feature enhancement strategies, including:

- **HGNetV2-based backbone enhancement**
- **CCFM (Cross-Scale Context Fusion Module)**
- **SREM (Spatial Relationship Enhancement Module)**
- **CSXB (Cross-Shaped Convolution Block)**

These improvements enhance multi-scale feature representation and spatial relationship modeling while maintaining computational efficiency, enabling accurate and lightweight wheat tiller instance segmentation and counting.

---

## Directory Structure

```
WTLR-Net
│
├── datasets
│ └── Dataset directory (not included)
│
├── ultralytics
│ │
│ ├── cfg
│ │ └── models
│ │   └── 11
│ │     ├── wtlr_net.yaml       # Model configuration
│ │     ├── yolo11.yaml         # YOLO11 detection configuration
│ │     └── yolo11-seg.yaml     # YOLO11 segmentation configuration
│ │
│ ├── nn
│ │ └── Addmodules
│ │   ├── HGNetV2.py            # HGNetV2 backbone
│ │   └── SREM.py               # SREM and CSXB related implementations
│ │
│ ├── models
│ ├── engine
│ ├── data
│ └── utils
│
├── weights
│ └── WTLR-Net.pt               # Trained model weights
│
├── train.py                    # Training script
├── predict.py                  # Inference script
│
├── requirements.txt
└── README.md
```

---

## Dataset

Experiments were conducted on a wheat tiller instance segmentation dataset collected under diverse field environments.

The dataset contains:

- Original field images
- Polygon-based segmentation annotation files in TXT format
- Training/validation/testing split files

The dataset is **not included in this repository**.

For peer review, the complete dataset is provided through a private access link available exclusively to the **editor and reviewers** for evaluation purposes.

After publication, the dataset availability will be updated according to the journal data-sharing policy.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/chmsyghr/WTLR-Net.git
cd WTLR-Net
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Model Weights

Download the pretrained model and place it into

```
weights/
```

The directory should be

```
weights/
    WTLR-Net.pt
```

---

## Training

```bash
python train.py
```

---

## Inference

```bash
python predict.py
```

The prediction results include:

- Wheat tiller instance masks
- Bounding boxes
- Automatic tiller counting results

---

## Experimental Environment

- Python 3.9
- PyTorch 2.0.1
- CUDA 11.7
- Ubuntu 20.04

---

## Acknowledgments

This work is implemented based on the Ultralytics YOLO framework.

The authors gratefully acknowledge the open-source community for providing excellent tools and resources.