<p align="center">
  <h1 align="center">SARNet: Go Closer to See Better 🔍</h1>
  <p align="center">
    <strong>Camouflaged Object Detection via Object Area Amplification and Figure-Ground Conversion</strong>
  </p>
  <p align="center">
    <a href="https://ieeexplore.ieee.org/document/10065514"><img src="https://img.shields.io/badge/Paper-IEEE%20TCSVT%202023-blue?style=for-the-badge&logo=ieee" alt="Paper"></a>
    <a href="https://github.com/Haozhe-Xing/SARNet"><img src="https://img.shields.io/github/stars/Haozhe-Xing/SARNet?style=for-the-badge&logo=github" alt="Stars"></a>
    <a href="https://github.com/Haozhe-Xing/SARNet/issues"><img src="https://img.shields.io/github/issues/Haozhe-Xing/SARNet?style=for-the-badge" alt="Issues"></a>
    <a href="#license"><img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License"></a>
  </p>
  <p align="center">
    <a href="#-highlights">Highlights</a> •
    <a href="#-architecture">Architecture</a> •
    <a href="#-results">Results</a> •
    <a href="#-quick-start">Quick Start</a> •
    <a href="#-citation">Citation</a>
  </p>
</p>

---

## 📰 News

| Date | Update |
|------|--------|
| **2023.03** | 🎉 Paper accepted by **IEEE TCSVT 2023**! |
| **2023.03** | 📦 Pretrained models and prediction maps released. |
| **2023.03** | 🚀 Training and inference code released. |

## ✨ Highlights

<table>
<tr>
<td width="60%">

- 🎯 **Object Area Amplification (OAA)** — Fuses adjacent-level features to amplify target region representations, enabling the network to "go closer" to camouflaged objects.
- 🔄 **Figure-Ground Conversion (FGC)** — Progressively refines predictions by selectively attending to foreground/background regions that deeper layers missed.
- 🏆 **State-of-the-Art** — Achieves competitive performance on **4 major COD benchmarks** (CAMO, CHAMELEON, COD10K, NC4K).
- ⚡ **PVTv2 Backbone** — Leverages Pyramid Vision Transformer V2 for powerful multi-scale feature extraction.

</td>
<td width="40%">

```
🦎 Can you spot the animal?

    ┌─────────────────┐
    │  ░░░▒▒▓▓██░░░░  │
    │  ░░▒▓█ 🦎 █▓▒░  │
    │  ░░░▒▒▓▓██░░░░  │
    └─────────────────┘

   SARNet: "Found it!" ✅
```

</td>
</tr>
</table>

## 🏗 Architecture

<p align="center">

```
                              SARNet Pipeline
  ┌─────────────────────────────────────────────────────────────────┐
  │                                                                 │
  │   Input (3×H×W)                                                 │
  │       │                                                         │
  │       ▼                                                         │
  │   ┌────────────────────────────────────────┐                    │
  │   │         PVTv2 Backbone Encoder         │                    │
  │   │                                        │                    │
  │   │  Stage1    Stage2    Stage3    Stage4   │                    │
  │   │   [C1]      [C2]      [C3]      [C4]   │                    │
  │   └──┬─────────┬─────────┬─────────┬───────┘                    │
  │      │         │         │         │                            │
  │      ▼         ▼         ▼         ▼                            │
  │   ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐                          │
  │   │ OAA₀ │ │ OAA₁ │ │ OAA₂ │ │ CBR  │  ◄── Feature Fusion     │
  │   └──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘                          │
  │      │        │        │        │                               │
  │      │  s2    │  s3    │  s4    │  s5     ──► Predict₄ (coarse) │
  │      │        │        │        │                               │
  │   ┌──────┐    │        │        │                               │
  │   │ OAA₃ │◄───────────────────-─┘                               │
  │   └──┬───┘    │        │                                        │
  │      │  s1    │        │                                        │
  │      │        │        │                                        │
  │      ▼        ▼        ▼        ▼                               │
  │   ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐                          │
  │   │ FGC₀ │◄│ FGC₁ │◄│ FGC₂ │◄│ FGC₃ │  ◄── Progressive       │
  │   └──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘      Refinement         │
  │      │        │        │        │                               │
  │      ▼        ▼        ▼        ▼                               │
  │  Predict₀  Predict₁ Predict₂ Predict₃                          │
  │   (fine)                        (coarse)                        │
  │                                                                 │
  └─────────────────────────────────────────────────────────────────┘
```

</p>

**Key Design Insights:**

| Module | Role | Mechanism |
|--------|------|-----------|
| **OAA** | Object Area Amplification | Fuses current-level & deeper features via dual-branch Conv+Upsample+Concat |
| **FGC** (bg mode) | Background-aware Refinement | Morphological dilation − prediction → attends to missed regions |
| **FGC** (fg mode) | Foreground-aware Refinement | Uses prediction map as attention weights for foreground enhancement |
| **CBR** | Channel Reduction | Conv → BatchNorm → ReLU on the deepest features |

## 📊 Results

### Quantitative Comparison on COD Benchmarks

> All metrics are reported using the same evaluation protocol. ↑ means higher is better, ↓ means lower is better.

| Dataset | S-measure ↑ | weighted F ↑ | MAE ↓ | mean E ↑ | mean F ↑ |
|:-------:|:-----------:|:------------:|:-----:|:--------:|:--------:|
| **CAMO** | 0.796 | 0.700 | 0.075 | 0.850 | 0.754 |
| **CHAMELEON** | 0.888 | 0.830 | 0.032 | 0.945 | 0.859 |
| **COD10K** | 0.815 | 0.667 | 0.037 | 0.886 | 0.720 |
| **NC4K** | 0.843 | 0.752 | 0.048 | 0.897 | 0.787 |

> 💡 *Please refer to the paper for full comparison tables with other methods.*

## 📦 Pretrained Models & Prediction Maps

| Resource | Backbone | Download |
|:--------:|:--------:|:--------:|
| Pretrained Model | PVTv2-B3 | [<img src="https://img.shields.io/badge/Google%20Drive-4285F4?logo=googledrive&logoColor=white" alt="Google Drive">](https://drive.google.com/drive/folders/1n80O0RIAe4KT08SZG-UK6HsWgD_qouoQ?usp=sharing) |
| Prediction Maps | — | [<img src="https://img.shields.io/badge/Google%20Drive-4285F4?logo=googledrive&logoColor=white" alt="Google Drive">](https://drive.google.com/drive/folders/1rZ9IrbFz4dggik1vnK53PnNNy7l-1X_r?usp=sharing) |

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/Haozhe-Xing/SARNet.git
cd SARNet

# Create conda environment
conda create -n sarnet python=3.8.13 -y
conda activate sarnet

# Install dependencies
pip install -r requirements.txt
```

**Requirements:** Python 3.8 · PyTorch · PVTv2 pretrained weights ([download](https://github.com/whai362/PVT))

### 2. Dataset Preparation

Download [COD10K](https://github.com/DengPingFan/SINet) and organize as:

```
<your_data_root>/
└── COD10K/
    ├── TrainDataset1/
    │   ├── Imgs/          # Training images (.jpg)
    │   └── GT/            # Ground truth masks (.png)
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

Then update the `root` path in [`config.py`](config.py).

### 3. Training

```bash
python train.py
```

<details>
<summary><b>⚙️ Training Configuration (click to expand)</b></summary>

| Parameter | Default | Description |
|-----------|---------|-------------|
| `pvt_name` | `pvt_v2_b3` | PVTv2 backbone variant |
| `args['scale']` | `384` | Input image resolution |
| `args['epoch_num']` | `100` | Number of training epochs |
| `args['lr']` | `1e-3` | Initial learning rate |
| `args['optimizer']` | `SGD` | Optimizer (`SGD` / `Adam`) |
| `args['train_batch_size']` | `2` | Training batch size |
| `args['lr_decay']` | `0.9` | Polynomial LR decay power |

</details>

### 4. Inference & Evaluation

```bash
python new_infer.py
```

Evaluation metrics (S-measure, weighted F-measure, MAE, E-measure, F-measure) are automatically saved to an Excel file.

### 5. Visualization

```bash
# Feature map heatmaps
cd display_heatmaps && python heatmap.py

# Overlay predictions on images
cd display_heatmaps && python combine.py
```

## 📁 Project Structure

```
SARNet/
├── SARNet.py             # 🧠 Core network (OAA + FGC modules)
├── pvtv2.py              # 🦴 PVTv2 backbone encoder
├── train.py              # 🏋️ Training pipeline
├── new_infer.py          # 🔍 Inference & evaluation
├── config.py             # ⚙️ Path configuration
├── datasets.py           # 📂 Dataset utilities
├── loss.py               # 📉 Loss functions (Structure, IoU, Dice)
├── joint_transforms.py   # 🔄 Joint image-mask augmentations
├── metric_caller.py      # 📏 Metric computation
├── excel_recorder.py     # 📊 Excel metric recording
├── misc.py               # 🔧 Utility functions
├── excel.py              # 📝 LaTeX table formatter
├── requirements.txt      # 📋 Python dependencies
└── display_heatmaps/     # 🎨 Visualization tools
    ├── heatmap.py        #    Feature map heatmap generation
    └── combine.py        #    Prediction overlay visualization
```

## 📖 Citation

If you find this work helpful for your research, please consider citing our paper and giving a ⭐:

```bibtex
@article{xing2023go,
  title     = {Go Closer to See Better: Camouflaged Object Detection via Object Area Amplification and Figure-Ground Conversion},
  author    = {Xing, Haozhe and Wang, Haiyu and Li, Yanye and Ling, Haibin},
  journal   = {IEEE Transactions on Circuits and Systems for Video Technology},
  volume    = {33},
  number    = {10},
  pages     = {5595--5608},
  year      = {2023},
  publisher = {IEEE},
  doi       = {10.1109/TCSVT.2023.3255304}
}
```

## 🙏 Acknowledgments

We sincerely thank the following open-source projects:

- [PVTv2](https://github.com/whai362/PVT) — Pyramid Vision Transformer backbone
- [PFNet](https://github.com/Mhaiyang/CVPR2021_PFNet) — Loss functions and utility code
- [py-sod-metrics](https://github.com/lartpang/PySODMetrics) — Evaluation metrics library

## 📬 Contact

If you have any questions, please feel free to open an [issue](https://github.com/Haozhe-Xing/SARNet/issues) or contact us.

---

<p align="center">
  If you find this project useful, please consider giving it a ⭐.<br>
  It helps others discover this work!
</p>