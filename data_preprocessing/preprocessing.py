"""
data/preprocessing.py
=====================
CT slice preprocessing pipeline.

All parameters are taken from config/config.yaml (preprocessing section).

Steps (applied identically to every slice, per the manuscript):
    1. Load raw TIFF slice
    2. Resize to 224 × 224 px using bilinear interpolation
    3. Pixel-intensity normalisation to [0, 1]
    4. Z-standardisation: (x − μ) / σ  where μ = σ = 0.5
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import tifffile
import yaml
from PIL import Image

logger = logging.getLogger(__name__)


def load_config(config_path: str | Path = "config/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Individual transforms
# ---------------------------------------------------------------------------

def resize_slice(img: np.ndarray, size: tuple[int, int] = (224, 224)) -> np.ndarray:
    """Resize a 2-D CT slice to *size* (H, W) using bilinear interpolation."""
    pil = Image.fromarray(img.astype(np.float32))
    pil = pil.resize((size[1], size[0]), resample=Image.BILINEAR)
    return np.array(pil, dtype=np.float32)


def normalise(img: np.ndarray,
              norm_min: float = 0.0,
              norm_max: float = 1.0) -> np.ndarray:
    """
    Min-max normalise a slice to [norm_min, norm_max].

    Clips values outside the observed range before scaling to avoid
    artefact-driven extremes dominating the range.
    """
    lo, hi = float(img.min()), float(img.max())
    if hi == lo:
        return np.full_like(img, norm_min, dtype=np.float32)
    scaled = (img - lo) / (hi - lo)                              # → [0, 1]
    return (scaled * (norm_max - norm_min) + norm_min).astype(np.float32)


def z_standardise(img: np.ndarray,
                  mean: float = 0.5,
                  std: float = 0.5) -> np.ndarray:
    """Apply z-standardisation: (x − mean) / std."""
    return ((img - mean) / std).astype(np.float32)


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def preprocess_slice(
    tiff_path: str | Path,
    image_size: tuple[int, int] = (224, 224),
    norm_min: float = 0.0,
    norm_max: float = 1.0,
    z_mean: float = 0.5,
    z_std: float = 0.5,
) -> np.ndarray:
    """
    Load one TIFF slice and apply the full preprocessing pipeline.

    Parameters
    ----------
    tiff_path : path to the .tif / .tiff file
    image_size : target (H, W) after resizing
    norm_min, norm_max : min-max normalisation bounds
    z_mean, z_std : z-standardisation parameters

    Returns
    -------
    Preprocessed float32 array of shape (*image_size,)
    """
    raw = tifffile.imread(str(tiff_path)).astype(np.float32)
    # Collapse channel dim if scanner wrote a 3-channel TIFF
    if raw.ndim == 3:
        raw = raw.mean(axis=-1)

    img = resize_slice(raw, size=image_size)
    img = normalise(img, norm_min=norm_min, norm_max=norm_max)
    img = z_standardise(img, mean=z_mean, std=z_std)
    return img


def preprocess_specimen(
    tiff_dir: str | Path,
    config_path: str | Path = "config/config.yaml",
) -> np.ndarray:
    """
    Preprocess all TIFF slices for one specimen.

    Parameters
    ----------
    tiff_dir : directory containing ordered .tif / .tiff slices
    config_path : path to config.yaml

    Returns
    -------
    Float32 array of shape (N_slices, H, W)
    """
    cfg = load_config(config_path)["preprocessing"]
    tiff_dir = Path(tiff_dir)
    slices = sorted(tiff_dir.glob("*.tif")) + sorted(tiff_dir.glob("*.tiff"))
    if not slices:
        raise FileNotFoundError(f"No TIFF files found in {tiff_dir}")

    volume = []
    for tiff in slices:
        arr = preprocess_slice(
            tiff,
            image_size=tuple(cfg["image_size"]),
            norm_min=cfg["norm_min"],
            norm_max=cfg["norm_max"],
            z_mean=cfg["z_mean"],
            z_std=cfg["z_std"],
        )
        volume.append(arr)

    logger.info("Preprocessed %d slices from %s", len(volume), tiff_dir)
    return np.stack(volume, axis=0)   # (N, H, W)
