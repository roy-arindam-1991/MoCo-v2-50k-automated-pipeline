# DEEPCTSEG: MoCo-v2-50k-automated-pipeline (50k)

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE) [![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/) [![PyTorch](https://img.shields.io/badge/PyTorch-2.1.2-EE4C2C.svg)](https://pytorch.org/) [![Platform](https://img.shields.io/badge/HPC-BlueBear%20%7C%20A100-7B2FBE.svg)](https://www.birmingham.ac.uk/research/arc/bear) [![Pipeline](https://img.shields.io/badge/Pipeline-MoCo%20v2%20%E2%86%92%20U--Net-00B4D8.svg)](https://github.com/roy-arindam-1991/MoCo-v2-50k-automated-pipeline/blob/main) [![Build](https://img.shields.io/badge/Build-Passing-brightgreen.svg)](https://github.com/roy-arindam-1991/MoCo-v2-50k-automated-pipeline/blob/main)

---

## Pipeline Developers

Arindam Roy, [Poulami Ghosh](https://github.com/g-Poulami)

---

## Overview

A fully automated, annotation-free pipeline for semantic segmentation of fossil CT data combining:

1. **MoCo v2 contrastive pretraining** on unlabelled CT slices to learn fossil-matrix feature representations without any manual annotation
2. **Rule-based deterministic masking** to generate reproducible coarse spatial priors
3. **Knowledge fusion into a modified U-Net** to refine coarse masks into high-precision segmentation overlays
4. **3D mesh generation** via marching cubes for downstream morphometric analysis

The pipeline reduces per-specimen processing from ~100 person-hours to **1--3 minutes** at inference.

---

## Repository Structure

```
MoCo-v2-50k-automated-pipeline/
|
+-- config/
|   +-- README.md                    # Configuration guide and key parameter reference
|   +-- config.yaml                  # All hyperparameters and paths
|
+-- data_preprocessing/
|   +-- README.md                    # Data access, MorphoSource accession, NMS request
|   +-- __init__.py
|   +-- preprocessing.py             # Resize, normalise, z-standardise CT slices
|   +-- resize_and_normalise.py      # Resize and normalise pipeline
|   +-- tiff_to_hdf5.py              # Convert TIFF stacks to HDF5 (.h5) volumes
|   +-- dataset_split.py             # Fixed-seed train / val / U-Net partition split
|   +-- hdf5_converter.py            # HDF5 I/O utilities
|   +-- run_preprocessing.sh         # Runs all data scripts in order on BlueBear HPC
|
+-- mocov2/
|   +-- README.md                    # MoCo v2 architecture, augmentation rationale, training
|   +-- __init__.py
|   +-- model.py                     # ResNet-50 encoder + momentum encoder + queue
|   +-- augmentations.py             # Domain-specific stochastic augmentation pipeline
|   +-- loss.py                      # InfoNCE (noise-contrastive estimation) loss
|   +-- train.py                     # MoCo v2 pre-training loop
|
+-- mocov2_validation/
|   +-- README.md                    # How to run Grad-CAM and UMAP sanity checks
|   +-- __init__.py
|   +-- metrics.py                   # Dice-Sorensen coefficient, IoU (Jaccard Index)
|   +-- grad_cam.py                  # Grad-CAM heatmaps over CT slices
|   +-- umap_viz.py                  # 2D UMAP of learned features + KDE overlay
|
+-- masking/
|   +-- README.md                    # 5-stage deterministic pipeline, fixed params, usage
|   +-- __init__.py
|   +-- rule_based_masking.py        # Otsu > region grow > morph ops > component filter
|
+-- knowledge_fusion/
|   +-- README.md                    # Dual-stream transfer: MoCo v2 to U-Net encoder init
|   +-- __init__.py
|   +-- transfer_weights.py          # Loads MoCo v2 checkpoint, maps encoder_q to U-Net
|   +-- run_transfer.sh              # Validates weight transfer before U-Net training starts
|
+-- unet/
|   +-- README.md                    # Modified U-Net architecture, loss, training details
|   +-- __init__.py
|   +-- model.py                     # Deep encoder, bottleneck, skip connections
|   +-- loss.py                      # BCEWithLogitsLoss
|   +-- train.py                     # U-Net training loop, Adam, cosine LR
|
+-- mesh/
|   +-- README.md                    # 3D mesh generation pipeline and output metrics
|   +-- __init__.py
|   +-- generate_mesh.py             # Stack masks to 3D vol to marching cubes to STL
|   +-- mesh_construction.py         # End-to-end mesh construction entry point
|
+-- registration/
|   +-- README.md                    # CloudCompare PPR > ICP > C2C workflow
|   +-- __init__.py
|   +-- registration_stats.py        # Parses CloudCompare CSV export
|   +-- run_registration.sh          # Batch-calls registration_stats.py for all specimens
|
+-- bash_scripts/
|   +-- README.md                    # SLURM job scripts, stage order, HPC details
|   +-- 01_preprocess.sh             # Stage 1: data preprocessing on SLURM
|   +-- 02_train_mocov2.sh           # Stage 2: MoCo v2 pre-training
|   +-- 03_mocov2_validation.sh      # Stage 3: Grad-CAM and UMAP sanity checks
|   +-- 04_generate_masks.sh         # Stage 4: rule-based coarse masking
|   +-- 05_transfer_weights.sh       # Stage 5: MoCo v2 to U-Net knowledge fusion
|   +-- 06_train_unet.sh             # Stage 6: U-Net training
|   +-- 07_run_inference.sh          # Stage 7: inference and mesh generation
|
+-- requirements.txt
+-- README.md
```

---

## Pipeline Workflow

```
Raw CT TIFF Slices
        |
        v
+------------------------------------------+
|  STAGE 1 - Data Preprocessing            |
|                                          |
|  Proportional sampling                   |
|  Resize to 224x224 px                    |
|  16-bit to 8-bit conversion              |
|  Normalise and z-standardise             |
|  TIFF stacks to HDF5 volumes             |
+---------------+--------------------------+
                |
       +--------+--------+
       v                 v
+------------------+  +--------------------------+
|  STAGE 2         |  |  STAGE 4                 |
|  MoCo v2         |  |  Rule-Based Masking      |
|  Pre-training    |  |                          |
|                  |  |  Intensity normalise     |
|  ResNet-50       |  |  Gaussian/median blur    |
|  + MLP head      |  |  Otsu thresholding       |
|  Momentum enc.   |  |  Region growing          |
|  Queue K=8192    |  |  Morphological ops       |
|  InfoNCE loss    |  |  Component filtering     |
|                  |  |                          |
+------+-----------+  |  Coarse binary masks     |
       |              |  Fixed parameters        |
       v              |  Fully reproducible      |
+------------------+  +-------------+------------+
|  STAGE 3         |                |
|  MoCo v2         |                |
|  Validation      |                |
|                  |                |
|  Grad-CAM        |                |
|  UMAP + KDE      |                |
+------+-----------+                |
       |                            |
       +-------------+--------------+
                     v
+-----------------------------------------+
|  STAGE 5 - Knowledge Fusion             |
|                                         |
|  MoCo v2 encoder_q ResNet-50 weights    |
|  initialise U-Net encoder               |
|                                         |
|  Dual-stream transfer:                  |
|    Textural features from MoCo v2       |
|    Spatial priors from coarse masks     |
+------------------+----------------------+
                   |
                   v
+-----------------------------------------+
|  STAGE 6 - U-Net Segmentation           |
|                                         |
|  Deep encoder with skip connections     |
|  Expanded bottleneck                    |
|  Loss: BCEWithLogitsLoss                |
|  Optimiser: Adam + cosine LR            |
+------------------+----------------------+
                   |
                   v
+-----------------------------------------------+
|  STAGE 7 - 3D Mesh Generation                 |
|                                               |
|  Stack 2D masks to 3D volume                  |
|  Region growing and Otsu intensity filter     |
|  Morphological cleanup                        |
|  Marching cubes surface extraction            |
|  Laplacian smoothing                          |
|  Mesh simplification via quadric decimation   |
|  Watertight STL output                        |
+------------------+----------------------------+
                   |
                   v
+-----------------------------------------+
|  STAGE 8 - Geometric Registration       |
|            (CloudCompare v2)            |
|                                         |
|  PPR > manual landmarking               |
|  ICP > rigid alignment (scale = 1.0)   |
|  C2C > signed distance analysis         |
|                                         |
|  External specimens validated           |
|  against manual thresholding baseline   |
+-----------------------------------------+
```

---

## Installation

```bash
git clone https://github.com/roy-arindam-1991/MoCo-v2-50k-automated-pipeline.git
cd MoCo-v2-50k-automated-pipeline
pip install -r requirements.txt
```

---

## Quick Start

```bash
# Stage 1 - preprocess raw TIFF stacks
bash bash_scripts/01_preprocess.sh

# Stage 2 - pre-train MoCo v2 on unlabelled slices
bash bash_scripts/02_train_mocov2.sh

# Stage 3 - MoCo v2 sanity checks (Grad-CAM + UMAP)
bash bash_scripts/03_mocov2_validation.sh

# Stage 4 - generate rule-based coarse masks
bash bash_scripts/04_generate_masks.sh

# Stage 5 - MoCo v2 to U-Net knowledge fusion
bash bash_scripts/05_transfer_weights.sh

# Stage 6 - train U-Net
bash bash_scripts/06_train_unet.sh

# Stage 7 - run inference and generate 3D meshes
bash bash_scripts/07_run_inference.sh
```

---

## Data

Training data: 50,626 CT images from the Middle Jurassic Kilmaluag Formation, Skye, Scotland. Specimens housed at **National Museums Scotland (NMS)**. Data available on request.

---

## Hardware

Trained on a single **NVIDIA A100 GPU** via BlueBear HPC (University of Birmingham).

| Stage | Time |
|---|---|
| MoCo v2 pre-training (600 epochs) | ~7.8 hrs |
| MoCo v2 validation (Grad-CAM + UMAP) | ~4 min |
| U-Net training (500 epochs) | ~6 hrs 10 min |
| Peak GPU memory | ~3.26 GB |
| Mesh generation per specimen | 1--3 min |

---

## License

This project is licensed under the **GNU General Public License v3.0** - see [LICENSE](LICENSE) for full terms.
All derivative works must remain open-source under GPL-3.0.
