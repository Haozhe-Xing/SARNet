# SARNet: Go Closer to See Better

> **Go Closer to See Better: Camouflaged Object Detection via Object Area Amplification and Figure-Ground Conversion**
>
> Paper: [IEEE Xplore](https://ieeexplore.ieee.org/document/10065514)

## Overview

SARNet (Segment-Amplify-Refine Network) is a deep learning framework for **Camouflaged Object Detection (COD)**. It addresses the challenge of detecting objects that are visually blended into their surroundings through two key modules:

- **OAA (Object Area Amplification)**: Fuses multi-level features to amplify target region representations
- **FGC (Figure-Ground Conversion)**: Progressively refines predictions via foreground-background selective attention

The network uses **PVTv2** as the backbone encoder and achieves state-of-the-art performance on major COD benchmarks.

## Architecture

```
Input Image
    │
    ▼
┌──────────┐
│  PVTv2   │ ──► 4-level multi-scale features
│ Backbone │
└──────────┘
    │
    ▼
┌──────────┐
│   OAA    │ ──► Fuse adjacent-level features (s1–s5)
│ Modules  │
└──────────┘
    │
    ▼
┌──────────┐
│   FGC    │ ──► Progressive coarse-to-fine refinement
│ Modules  │
└──────────┘
    │
    ▼
 5 prediction maps (upsampled to original resolution)
```

## Pretrained Models & Prediction Maps

| Resource | Link |
|----------|------|
| Prediction Maps | [Google Drive](https://drive.google.com/drive/folders/1rZ9IrbFz4dggik1vnK53PnNNy7l-1X_r?usp=sharing) |
| Pretrained Model | [Google Drive](https://drive.google.com/drive/folders/1n80O0RIAe4KT08SZG-UK6HsWgD_qouoQ?usp=sharing) |

## Environment Setup

```bash
# Clone the repository
git clone https://github.com/Haozhe-Xing/SARNet.git
cd SARNet

# Create and activate conda environment
conda create --name sarnet python=3.8.13
conda activate sarnet

# Install dependencies
pip install -r requirements.txt
```

### Requirements
- Python 3.8.13
- PyTorch (see `requirements.txt` for full dependency list)
- PVTv2 pretrained backbone weights (place in `PVTv2_Seg/checkpoint/`)

## Dataset Preparation

Organize the COD10K dataset as follows:

```
COD10K/
├── TrainDataset1/
│   ├── Imgs/       # Training images (.jpg)
│   └── GT/         # Ground truth masks (.png)
└── TestDataset/
    ├── CHAMELEON/
    │   ├── Imgs/
    │   └── GT/
    ├── CAMO/
    │   ├── Imgs/
    │   └── GT/
    ├── COD10K/
    │   ├── Imgs/
    │   └── GT/
    └── NC4K/
        ├── Imgs/
        └── GT/
```

Update the `root` variable in `config.py` to point to your dataset directory.

## Training

```bash
python train.py
```

Key training configurations can be modified in `train.py`:
- `pvt_name`: PVTv2 backbone variant (default: `pvt_v2_b3`)
- `args['scale']`: Input image size (default: 384)
- `args['epoch_num']`: Number of training epochs (default: 100)
- `args['lr']`: Learning rate (default: 1e-3)
- `args['optimizer']`: Optimizer type (`SGD` or `Adam`)

## Evaluation

After training, the script automatically evaluates on all test datasets and records metrics (S-measure, weighted F-measure, MAE, E-measure, F-measure) to an Excel file.

For standalone evaluation:
```bash
python new_infer.py
```

## Feature Map Visualization

To visualize intermediate feature maps as heatmaps:

```bash
cd display_heatmaps
python heatmap.py
```

To overlay prediction masks on original images:

```bash
cd display_heatmaps
python combine.py
```

## Project Structure

```
SARNet/
├── SARNet.py           # Core network architecture (OAA + FGC modules)
├── pvtv2.py            # PVTv2 backbone implementation
├── train.py            # Training pipeline
├── new_infer.py        # Inference and evaluation
├── config.py           # Dataset and model path configuration
├── datasets.py         # Dataset loading utilities
├── loss.py             # Loss functions (structure loss, IoU, Dice)
├── joint_transforms.py # Joint image-mask data augmentation
├── metric_caller.py    # Evaluation metric computation
├── excel_recorder.py   # Excel-based metric recording
├── misc.py             # Utility functions
├── excel.py            # LaTeX table formatting utility
├── requirements.txt    # Python dependencies
└── display_heatmaps/   # Visualization tools
    ├── heatmap.py      # Feature map heatmap generation
    └── combine.py      # Prediction overlay visualization
```

## Citation

If you find this work useful, please cite:

```bibtex
@article{xing2023go,
  title={Go Closer to See Better: Camouflaged Object Detection via Object Area Amplification and Figure-Ground Conversion},
  author={Xing, Haozhe and others},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  year={2023},
  doi={10.1109/TCSVT.2023.3255304}
}
```

## Acknowledgments

- [PVTv2](https://github.com/whai362/PVT) for the backbone network
- [PFNet](https://github.com/Mhaiyang/CVPR2021_PFNet) for loss functions and utility code
- [py-sod-metrics](https://github.com/lartpang/PySODMetrics) for evaluation metrics