"""
masking/rule_based_masking.py
==============================
Fully deterministic, rule-based pipeline for coarse binary mask generation.

"A fully deterministic, rule-based pipeline generated coarse binary masks of
fossilised bone from each CT slice, providing standardised spatial guidance for
subsequent U-Net training. [...] Given the same input CT slice, the algorithm
always produces the identical mask, eliminating operator subjectivity and
annotation fatigue inherent in manual ground truth generation."
— manuscript §Rule-based pipeline

Pipeline stages (five sequential):
    1. Intensity normalisation via percentile clipping
    2. Gaussian + median blurring (noise / artefact suppression)
    3. Otsu thresholding + outward region expansion (seed identification)
    4. Morphological refinement (top-hat, erosion/dilation cycles)
    5. Connected component filtering (size + seed-overlap criteria)

All parameters are loaded from config/config.yaml (masking section).
Parameters were determined empirically on the training partition and held
fixed thereafter — they are NOT tuned per specimen.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import yaml
from scipy import ndimage as ndi
from skimage import filters, morphology
from skimage.measure import label, regionprops

logger = logging.getLogger(__name__)


def load_config(config_path: str | Path = "config/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Individual pipeline stages
# ---------------------------------------------------------------------------

def stage1_normalise(img: np.ndarray,
                     p_low: float = 2,
                     p_high: float = 98) -> np.ndarray:
    """
    Stage 1: Percentile clipping + linear rescale to [0, 1].

    Suppresses outlier voxels (e.g. beam hardening spikes, scanner artefacts)
    and enhances the radiodensity range occupied by mineralised bone.
    """
    lo = np.percentile(img, p_low)
    hi = np.percentile(img, p_high)
    clipped = np.clip(img, lo, hi)
    if hi == lo:
        return np.zeros_like(img, dtype=np.float32)
    return ((clipped - lo) / (hi - lo)).astype(np.float32)


def stage2_blur(img: np.ndarray,
                gaussian_sigma: float = 1.5,
                median_kernel: int = 3) -> np.ndarray:
    """
    Stage 2: Sequential Gaussian blur then median filter.

    Gaussian blur: attenuates high-frequency noise and stochastic graininess
    from high-resolution µCT acquisition.
    Median filter: removes salt-and-pepper noise while preserving bone edges
    better than additional Gaussian smoothing.
    """
    img_gauss = filters.gaussian(img, sigma=gaussian_sigma)
    img_med = ndi.median_filter(img_gauss, size=median_kernel)
    return img_med.astype(np.float32)


def stage3_otsu_seeds(img: np.ndarray,
                      nbins: int = 256,
                      dilation_radius: int = 3) -> np.ndarray:
    """
    Stage 3: Otsu thresholding + morphological dilation for seed expansion.

    Otsu's method finds an automatic global threshold from the intensity
    histogram, maximising between-class variance to separate bone from matrix.
    The subsequent dilation expands seed regions outward to capture the full
    spatial extent of bone in each slice, compensating for the conservative
    nature of global thresholding on low-contrast fossil material.

    Returns
    -------
    Binary mask (uint8, 0/1) after seed identification + expansion
    """
    threshold = filters.threshold_otsu(img, nbins=nbins)
    binary = (img > threshold).astype(np.uint8)
    if dilation_radius > 0:
        struct = morphology.disk(dilation_radius)
        binary = morphology.binary_dilation(binary, footprint=struct).astype(np.uint8)
    return binary


def stage4_morphological_refinement(mask: np.ndarray,
                                     tophat_size: int = 15,
                                     erosion_iter: int = 1,
                                     dilation_iter: int = 2) -> np.ndarray:
    """
    Stage 4: Top-hat transform + erosion/dilation cycles.

    Top-hat transform isolates bright bone structures against a slowly
    varying matrix background, filling internal gaps and suppressing
    disconnected artefactual fragments.
    Subsequent erosion removes thin spurious connections; dilation restores
    and slightly expands genuine bone boundaries.

    Returns
    -------
    Refined binary mask (uint8, 0/1)
    """
    selem = morphology.disk(tophat_size)
    # White top-hat: bright details not captured by the structural element
    tophat = morphology.white_tophat(mask.astype(np.float32), footprint=selem)
    refined = (mask.astype(np.float32) + tophat).clip(0, 1)

    # Binary erosion then dilation
    for _ in range(erosion_iter):
        refined = morphology.binary_erosion(refined > 0.5).astype(np.float32)
    for _ in range(dilation_iter):
        refined = morphology.binary_dilation(refined > 0.5).astype(np.float32)

    return refined.astype(np.uint8)


def stage5_component_filter(mask: np.ndarray,
                             seeds: np.ndarray,
                             min_area: float = 50,
                             voxel_area_mm2: float = 1.0) -> np.ndarray:
    """
    Stage 5: Connected component analysis — retain only valid bone regions.

    Two criteria must both be satisfied for a component to be retained:
      (a) Physical area ≥ min_area (mm²), computed via voxel_area_mm2
      (b) Spatial overlap with seed regions from Stage 3

    This eliminates noise artefacts and matrix fragments that passed through
    earlier stages without genuine bone-seed overlap.

    Parameters
    ----------
    mask          : binary mask after stage 4
    seeds         : binary seed mask from stage 3 (before dilation)
    min_area      : minimum component area in mm² (default from config)
    voxel_area_mm2: area of one pixel in mm² (scanner-dependent)

    Returns
    -------
    Filtered binary mask (uint8, 0/1)
    """
    min_pixels = max(1, int(min_area / voxel_area_mm2))
    labeled = label(mask)
    output = np.zeros_like(mask, dtype=np.uint8)

    for region in regionprops(labeled):
        if region.area < min_pixels:
            continue
        # Check overlap with seed mask
        component_mask = labeled == region.label
        if not np.any(seeds & component_mask):
            continue
        output[component_mask] = 1

    return output


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def generate_mask(
    img: np.ndarray,
    config_path: str | Path = "config/config.yaml",
    voxel_area_mm2: float = 1.0,
) -> np.ndarray:
    """
    Apply the full 5-stage deterministic masking pipeline to one CT slice.

    Parameters
    ----------
    img            : 2D float array, shape (H, W) — preprocessed CT slice
    config_path    : path to config.yaml
    voxel_area_mm2 : pixel area in mm² for physical size filtering

    Returns
    -------
    Binary mask (uint8, 0/1), shape (H, W)
    """
    cfg = load_config(config_path)["masking"]

    s1 = stage1_normalise(img,
                          p_low=cfg["percentile_low"],
                          p_high=cfg["percentile_high"])

    s2 = stage2_blur(s1,
                     gaussian_sigma=cfg["gaussian_sigma"],
                     median_kernel=cfg["median_kernel_size"])

    seeds = stage3_otsu_seeds(s2,
                              nbins=cfg["otsu_nbins"],
                              dilation_radius=0)           # undilated seeds for S5

    s3 = stage3_otsu_seeds(s2,
                           nbins=cfg["otsu_nbins"],
                           dilation_radius=cfg["dilation_radius"])

    s4 = stage4_morphological_refinement(s3,
                                          tophat_size=cfg["tophat_kernel_size"],
                                          erosion_iter=cfg["erosion_iterations"],
                                          dilation_iter=cfg["dilation_iterations"])

    s5 = stage5_component_filter(s4, seeds,
                                  min_area=cfg["min_area_mm2"],
                                  voxel_area_mm2=voxel_area_mm2)
    return s5


def generate_masks_for_volume(
    volume: np.ndarray,
    config_path: str | Path = "config/config.yaml",
    voxel_area_mm2: float = 1.0,
) -> np.ndarray:
    """
    Apply the masking pipeline to every slice in a CT volume.

    Parameters
    ----------
    volume         : (N, H, W) float array — preprocessed CT stack
    config_path    : path to config.yaml
    voxel_area_mm2 : pixel area in mm²

    Returns
    -------
    Binary mask volume (uint8), shape (N, H, W)
    """
    masks = np.stack(
        [generate_mask(volume[i], config_path, voxel_area_mm2)
         for i in range(volume.shape[0])],
        axis=0,
    )
    logger.info("Generated masks for %d slices", volume.shape[0])
    return masks
