# data_preprocessing/

This module handles all preprocessing steps to prepare raw CT TIFF stacks for training.

## Scripts

| File | Description |
|---|---|
| `preprocessing.py` | Master preprocessing wrapper |
| `tiff_to_hdf5.py` | Convert TIFF stacks → HDF5 (.h5) volumes for fast I/O |
| `hdf5_converter.py` | HDF5 I/O utilities |

## Usage

```bash
bash bash_scripts/01_preprocess.sh
```

## Data Access

Raw CT volumes are available on request.
Specimens are housed at **National Museums Scotland (NMS)**.
See the root `README.md` for full accession numbers.
