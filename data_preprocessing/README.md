# data/

This module handles all preprocessing steps to prepare raw CT TIFF stacks for training.

## Scripts

| File | Description |
|---|---|
| `preprocessing.py` | Master preprocessing wrapper |
| `resize_and_normalise.py` | Resize slices to 224×224, normalise to [0,1], z-standardise |
| `tiff_to_hdf5.py` | Convert TIFF stacks → HDF5 (.h5) volumes for fast I/O |
| `dataset_split.py` | Fixed-seed split: 19 train / 2 val / 3 U-Net datasets |
| `hdf5_converter.py` | HDF5 I/O utilities |
| `run_preprocessing.sh` | Runs all scripts in order on BlueBear HPC via SLURM |

## Usage

```bash
bash data/run_preprocessing.sh
```

## Data Access

Raw CT volumes are available on request.  
Specimens are housed at **National Museums Scotland (NMS)**.  
See the root `README.md` for full accession numbers.
