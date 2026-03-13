<p align="right">
  <a href="README.md">English</a> | <a href="README_zh.md">中文</a>
</p>

<p align="center">
  <h1 align="center">SARNet：更近一步，看得更清 🔍</h1>
  <p align="center">
    <strong>基于目标区域放大和图-底转换的伪装目标检测</strong>
  </p>
  <p align="center">
    <a href="https://ieeexplore.ieee.org/document/10065514"><img src="https://img.shields.io/badge/论文-IEEE%20TCSVT%202023-blue?style=for-the-badge&logo=ieee" alt="论文"></a>
    <a href="https://github.com/Haozhe-Xing/SARNet"><img src="https://img.shields.io/github/stars/Haozhe-Xing/SARNet?style=for-the-badge&logo=github" alt="Stars"></a>
    <a href="https://github.com/Haozhe-Xing/SARNet/issues"><img src="https://img.shields.io/github/issues/Haozhe-Xing/SARNet?style=for-the-badge" alt="Issues"></a>
    <a href="#license"><img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License"></a>
  </p>
  <p align="center">
    <a href="#-亮点">亮点</a> •
    <a href="#-网络架构">架构</a> •
    <a href="#-实验结果">结果</a> •
    <a href="#-可视化">可视化</a> •
    <a href="#-快速开始">快速开始</a> •
    <a href="#-引用">引用</a>
  </p>
</p>

<p align="center">
  <img src="figures/fig1_motivation.png" width="85%" alt="Motivation">
  <br>
  <em>图 1：本文提出的"搜索-放大-识别"（SAR）范式。与以往的"搜索-辨别"方法不同，SARNet 通过 OAA 模块引入"放大"阶段，并通过 FGC 模块实现"识别"阶段，从而逐步检测高度伪装的目标。</em>
</p>

---

## 👋 为什么选择这个项目？

> **刚开始接触伪装目标检测（COD）？** 那你来对地方了！

SARNet 是一个**对初学者友好、同时具备科研水准**的 COD 项目。无论你是第一次探索伪装目标检测的学生，还是寻找可靠 baseline 的研究者，这个仓库都能满足你的需求：

- 📖 **代码清晰、注释完善** — 每个模块都有详尽的英文注释文档，从数据加载到模型推理的完整流程一目了然。
- 🎨 **开箱即用的可视化工具** — 我们开源了生成**特征图热力图**和**预测叠加图**的脚本（详见[可视化](#-可视化)部分），帮助你从视觉上理解模型的工作原理——而不仅仅是看数字。
- 🧩 **模块化架构** — OAA 和 FGC 模块相互独立、即插即用，方便你将其嵌入自己的网络进行实验。
- 🚀 **端到端工作流** — 训练、推理、评估、可视化一应俱全。只需克隆仓库、配置路径、运行即可！
- 📊 **自动化评估** — 推理后自动计算 S-measure、F-measure、MAE、E-measure 等指标，并保存为 Excel 文件。

> 💡 *如果这是你的第一个 COD 项目，我们建议先从[快速开始](#-快速开始)部分入手，然后探索[可视化](#-可视化)工具，建立对伪装目标检测工作原理的直觉理解。*

## ✨ 亮点

- 🎯 **目标区域放大（OAA）** — 融合相邻层级特征以放大目标区域表示，使网络能够"更近一步"观察伪装目标。
- 🔄 **图-底转换（FGC）** — 通过选择性关注深层网络遗漏的前景/背景区域，逐步精化预测结果。
- 🏆 **SOTA 性能** — 在 **4 个主流 COD 基准**（CAMO、CHAMELEON、COD10K、NC4K）上取得了具有竞争力的性能。
- ⚡ **PVTv2 骨干网络** — 利用 Pyramid Vision Transformer V2 进行强大的多尺度特征提取。
- 🎨 **开源可视化工具** — 我们提供了开箱即用的**特征图热力图生成**和**预测叠加可视化**脚本（详见[可视化](#-可视化)部分）。

## 🏗 网络架构

<p align="center">
  <img src="figures/fig3_architecture.png" width="90%" alt="SARNet Architecture">
  <br>
  <em>图 3：SARNet 整体网络架构。PVTv2 骨干网络提取多尺度特征，然后通过目标区域放大（OAA）模块融合并放大目标特征。图-底转换模块（FFGC、EFGC）通过关注前景/背景区域逐步精化预测结果。</em>
</p>

**核心设计：**

| 模块 | 作用 | 机制 |
|------|------|------|
| **OAA** | 目标区域放大 | 通过双分支 Conv+上采样+拼接 融合当前层与深层特征 |
| **FGC** (bg 模式) | 背景感知精化 | 形态学膨胀 − 预测图 → 关注遗漏区域 |
| **FGC** (fg 模式) | 前景感知精化 | 使用预测图作为注意力权重增强前景 |
| **CBR** | 通道缩减 | 对最深层特征执行 Conv → BatchNorm → ReLU |

## 📊 实验结果

### COD 基准数据集定量对比

> 所有指标均使用相同评估协议。↑ 表示越高越好，↓ 表示越低越好。

| 数据集 | S-measure ↑ | weighted F ↑ | MAE ↓ | mean E ↑ | mean F ↑ |
|:------:|:-----------:|:------------:|:-----:|:--------:|:--------:|
| **CAMO** | 0.796 | 0.700 | 0.075 | 0.850 | 0.754 |
| **CHAMELEON** | 0.888 | 0.830 | 0.032 | 0.945 | 0.859 |
| **COD10K** | 0.815 | 0.667 | 0.037 | 0.886 | 0.720 |
| **NC4K** | 0.843 | 0.752 | 0.048 | 0.897 | 0.787 |

> 💡 *完整的与其他方法的对比表格请参阅论文。*

### 定性对比

<p align="center">
  <img src="figures/fig6_visual_comparison.png" width="90%" alt="Visual Comparison">
  <br>
  <em>图 6：与最先进方法的视觉对比。SARNet 生成更精确、更完整的分割掩码，尤其对于具有复杂伪装模式的目标表现突出。本方法有效处理了小目标、与背景纹理相似的目标以及多个伪装实例等挑战场景。</em>
</p>

## 🎨 可视化

> **📢 我们开源了论文中使用的所有可视化工具！** 你可以使用 `display_heatmaps/` 目录下提供的脚本复现下面展示的特征热力图和预测叠加图。

### 特征图热力图

<p align="center">
  <img src="figures/fig7_heatmap.png" width="90%" alt="Feature Map Heatmaps">
  <br>
  <em>图 7：不同阶段的特征图可视化。热力图展示了 OAA 和 FGC 模块如何逐步聚焦于伪装目标。暖色表示更高的激活值，表明深层特征关注更广泛的区域，而精化后的特征则精确定位目标边界。</em>
</p>

使用开源脚本生成特征图热力图：

```bash
cd display_heatmaps && python heatmap.py
```

> 该脚本加载中间特征图，应用色彩映射变换，并将热力图叠加到原始图像上。详见 [`display_heatmaps/heatmap.py`](display_heatmaps/heatmap.py)。

### 特征可视化分析

<p align="center">
  <img src="figures/fig8_feature_visualization.png" width="90%" alt="Feature Visualization">
  <br>
  <em>图 8：展示 OAA 和 FGC 模块效果的详细特征可视化。(a-b) OAA 前后的特征图展示了目标区域注意力的放大效果。(c-d) FGC 前后的特征图展示了精化的图-底分离效果。</em>
</p>

### 预测叠加图

将预测图叠加到原始图像上进行定性分析：

```bash
cd display_heatmaps && python combine.py
```

> 该脚本生成输入图像、真值掩码和模型预测的并排对比图。详见 [`display_heatmaps/combine.py`](display_heatmaps/combine.py)。

---

## 📦 预训练模型与预测图

| 资源 | 骨干网络 | 下载链接 |
|:----:|:-------:|:-------:|
| 预训练模型 | PVTv2-B3 | [<img src="https://img.shields.io/badge/Google%20Drive-4285F4?logo=googledrive&logoColor=white" alt="Google Drive">](https://drive.google.com/drive/folders/1n80O0RIAe4KT08SZG-UK6HsWgD_qouoQ?usp=sharing) |
| 预测图 | — | [<img src="https://img.shields.io/badge/Google%20Drive-4285F4?logo=googledrive&logoColor=white" alt="Google Drive">](https://drive.google.com/drive/folders/1rZ9IrbFz4dggik1vnK53PnNNy7l-1X_r?usp=sharing) |

## 🚀 快速开始

### 1. 环境配置

```bash
# 克隆仓库
git clone https://github.com/Haozhe-Xing/SARNet.git
cd SARNet

# 创建 conda 环境
conda create -n sarnet python=3.8.13 -y
conda activate sarnet

# 安装依赖
pip install -r requirements.txt
```

**依赖要求：** Python 3.8 · PyTorch · PVTv2 预训练权重（[下载地址](https://github.com/whai362/PVT)）

### 2. 数据集准备

下载 [COD10K](https://github.com/DengPingFan/SINet) 数据集并按如下结构组织：

```
<你的数据根目录>/
└── COD10K/
    ├── TrainDataset1/
    │   ├── Imgs/          # 训练图像 (.jpg)
    │   └── GT/            # 真值掩码 (.png)
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

然后更新 [`config.py`](config.py) 中的 `root` 路径。

### 3. 训练

```bash
python train.py
```

<details>
<summary><b>⚙️ 训练配置参数（点击展开）</b></summary>

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `pvt_name` | `pvt_v2_b3` | PVTv2 骨干网络变体 |
| `args['scale']` | `384` | 输入图像分辨率 |
| `args['epoch_num']` | `100` | 训练轮数 |
| `args['lr']` | `1e-3` | 初始学习率 |
| `args['optimizer']` | `SGD` | 优化器（`SGD` / `Adam`） |
| `args['train_batch_size']` | `2` | 训练批次大小 |
| `args['lr_decay']` | `0.9` | 多项式学习率衰减幂次 |

</details>

### 4. 推理与评估

```bash
python new_infer.py
```

评估指标（S-measure、weighted F-measure、MAE、E-measure、F-measure）会自动计算并保存到 Excel 文件。

### 5. 可视化

我们提供了开源可视化工具，用于复现论文中的所有可视化结果。详见[可视化](#-可视化)部分的详细示例和说明。

```bash
# 特征图热力图（复现论文图 7 和图 8）
cd display_heatmaps && python heatmap.py

# 预测叠加图（复现视觉对比结果）
cd display_heatmaps && python combine.py
```

## 📁 项目结构

```
SARNet/
├── SARNet.py             # 🧠 核心网络（OAA + FGC 模块）
├── pvtv2.py              # 🦴 PVTv2 骨干网络编码器
├── train.py              # 🏋️ 训练流程
├── new_infer.py          # 🔍 推理与评估
├── config.py             # ⚙️ 路径配置
├── datasets.py           # 📂 数据集工具
├── loss.py               # 📉 损失函数（Structure、IoU、Dice）
├── joint_transforms.py   # 🔄 图像-掩码联合数据增强
├── metric_caller.py      # 📏 指标计算
├── excel_recorder.py     # 📊 Excel 指标记录
├── misc.py               # 🔧 工具函数
├── excel.py              # 📝 LaTeX 表格格式化
├── requirements.txt      # 📋 Python 依赖
└── display_heatmaps/     # 🎨 可视化工具
    ├── heatmap.py        #    特征图热力图生成
    └── combine.py        #    预测叠加可视化
```

## 📖 引用

如果本工作对你的研究有所帮助，请考虑引用我们的论文并给个 ⭐：

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

## 🙏 致谢

我们衷心感谢以下开源项目：

- [PVTv2](https://github.com/whai362/PVT) — Pyramid Vision Transformer 骨干网络
- [PFNet](https://github.com/Mhaiyang/CVPR2021_PFNet) — 损失函数与工具代码
- [py-sod-metrics](https://github.com/lartpang/PySODMetrics) — 评估指标库

## 📬 联系方式

如有任何问题，欢迎提交 [Issue](https://github.com/Haozhe-Xing/SARNet/issues) 或直接联系我们。

---

<p align="center">
  如果你觉得这个项目对你有帮助，请给个 ⭐<br>
  这能帮助更多人发现这项工作！
</p>