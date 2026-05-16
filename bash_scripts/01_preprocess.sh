#!/usr/bin/env bash
set -euo pipefail
echo "=== Stage 1: Data Preprocessing (TIFF → HDF5, resize, normalise) ==="
python data_preprocessing/tiff_to_hdf5.py
python data_preprocessing/preprocessing.py
echo "HDF5 volumes → data/hdf5/"
