# bash_scripts/

SLURM batch scripts to run each stage of the pipeline on BlueBear HPC (University of Birmingham).
Scripts must be submitted in order — each stage depends on outputs from the previous.

## Scripts

| File | Stage | Description |
|---|---|---|
| `01_preprocess.sh` | 1 | Data preprocessing: resize, normalise, TIFF → HDF5 |
| `02_train_mocov2.sh` | 2 | MoCo v2 contrastive pre-training on A100 GPU |
| `03_mocov2_validation.sh` | 3 | Grad-CAM heatmaps and UMAP embedding sanity checks |
| `04_train_unet.sh` | 4 | U-Net training via MoCo v2 knowledge fusion |
| `05_run_inference.sh` | 5 | End-to-end inference on new CT specimens |

## Usage

```bash
bash bash_scripts/01_preprocess.sh
bash bash_scripts/02_train_mocov2.sh
bash bash_scripts/03_mocov2_validation.sh
bash bash_scripts/04_train_unet.sh
bash bash_scripts/05_run_inference.sh
```

## HPC Details

- **Cluster**: BlueBear HPC, University of Birmingham
- **GPU**: NVIDIA A100
- **Scheduler**: SLURM
- All scripts activate the correct module environment before execution
