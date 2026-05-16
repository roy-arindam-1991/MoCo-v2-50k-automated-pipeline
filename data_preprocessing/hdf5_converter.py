"""
data/hdf5_converter.py
======================
Convert preprocessed CT TIFF stacks into HDF5 (.h5) containers.

Motivation (per manuscript):
    The split CT image stacks originally consist of thousands of individual 2D TIFF
    files. Converting them into HDF5 eliminates significant file-system overhead and
    directory-latency, enables data chunking and 'out-of-core' processing for high-
    speed random access on specific volumetric regions without exceeding RAM limits,
    and uses built-in lossless compression for a storage-efficient pipeline.

Schema per HDF5 file (one file = one specimen):
    /volume         float32 (N_slices, 224, 224)  — preprocessed CT stack
    /meta/n_slices  int
    /meta/specimen  str
"""

from __future__ import annotations

import logging
from pathlib import Path

import h5py
import numpy as np
import yaml

from data.preprocessing import preprocess_specimen

logger = logging.getLogger(__name__)


def load_config(config_path: str | Path = "config/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def convert_specimen_to_hdf5(
    tiff_dir: str | Path,
    output_path: str | Path,
    specimen_name: str = "",
    config_path: str | Path = "config/config.yaml",
) -> Path:
    """
    Preprocess one specimen's TIFF stack and write it to an HDF5 file.

    Parameters
    ----------
    tiff_dir    : directory of raw .tif slices
    output_path : destination .h5 file
    specimen_name : optional label stored in /meta/specimen
    config_path : path to config.yaml

    Returns
    -------
    Path of the written HDF5 file
    """
    cfg = load_config(config_path)
    pcfg = cfg["preprocessing"]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    volume = preprocess_specimen(tiff_dir, config_path=config_path)  # (N, H, W)
    n_slices = volume.shape[0]

    chunk_size = tuple(pcfg["hdf5_chunk_size"])          # e.g. (1, 224, 224)
    compression = pcfg["compression"]                    # "gzip"
    compression_opts = pcfg["compression_opts"]          # 4

    with h5py.File(output_path, "w") as f:
        f.create_dataset(
            "volume",
            data=volume,
            chunks=chunk_size,
            compression=compression,
            compression_opts=compression_opts,
            dtype=np.float32,
        )
        meta = f.require_group("meta")
        meta.attrs["n_slices"] = n_slices
        meta.attrs["specimen"] = specimen_name or str(Path(tiff_dir).name)
        meta.attrs["image_size"] = pcfg["image_size"]
        meta.attrs["z_mean"] = pcfg["z_mean"]
        meta.attrs["z_std"] = pcfg["z_std"]

    logger.info("Wrote %d slices → %s (%.1f MB)", n_slices, output_path,
                output_path.stat().st_size / 1e6)
    return output_path


def batch_convert(
    raw_root: str | Path,
    hdf5_root: str | Path,
    config_path: str | Path = "config/config.yaml",
) -> list[Path]:
    """
    Convert every specimen sub-directory under *raw_root* to an HDF5 file.

    Directory convention:
        raw_root/
            specimen_A/   ← one subdirectory per specimen
                0001.tif
                0002.tif
                ...

    Parameters
    ----------
    raw_root  : parent directory containing per-specimen TIFF subdirectories
    hdf5_root : output directory for .h5 files (mirrors raw_root structure)

    Returns
    -------
    List of paths to written HDF5 files
    """
    raw_root = Path(raw_root)
    hdf5_root = Path(hdf5_root)
    written = []

    specimen_dirs = sorted(d for d in raw_root.iterdir() if d.is_dir())
    if not specimen_dirs:
        raise FileNotFoundError(f"No specimen subdirectories found in {raw_root}")

    for spec_dir in specimen_dirs:
        out = hdf5_root / f"{spec_dir.name}.h5"
        logger.info("Converting %s …", spec_dir.name)
        convert_specimen_to_hdf5(
            tiff_dir=spec_dir,
            output_path=out,
            specimen_name=spec_dir.name,
            config_path=config_path,
        )
        written.append(out)

    logger.info("Converted %d specimens to HDF5 under %s", len(written), hdf5_root)
    return written


class FossilHDF5Dataset:
    """
    Lightweight dataset wrapper for random-access loading of CT slices from HDF5.

    Supports out-of-core access: only the requested slice is loaded into memory.
    """

    def __init__(self, hdf5_paths: list[str | Path]):
        self.files: list[h5py.File] = []
        self._index: list[tuple[int, int]] = []   # (file_idx, slice_idx)

        for i, p in enumerate(hdf5_paths):
            f = h5py.File(str(p), "r")
            self.files.append(f)
            n = f["volume"].shape[0]
            self._index.extend((i, j) for j in range(n))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> np.ndarray:
        file_idx, slice_idx = self._index[idx]
        # Out-of-core: h5py reads only the requested slice from disk
        return self.files[file_idx]["volume"][slice_idx]   # (224, 224)

    def close(self):
        for f in self.files:
            f.close()

    def __del__(self):
        self.close()
