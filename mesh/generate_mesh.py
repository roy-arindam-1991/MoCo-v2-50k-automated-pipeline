"""
mesh/generate_mesh.py
=====================
3D mesh generation from stacked 2D U-Net segmentation masks.

Pipeline (per manuscript §3D mesh generation):
    1. Stack 2D masks to form a 3D binary volume
    2. Region growing from high-confidence bone voxels
    3. Otsu-based intensity filtering to exclude non-bone regions
    4. Morphological operations — connect bone fragments, eliminate noise
    5. Marching cubes (iso-level = 0.5) → triangulated surface mesh
    6. Export watertight STL suitable for morphometrics, FEA, MDA, 3D printing

Typical runtime: 1–3 minutes per specimen on a single GPU server.
Mesh sizes (Table 1): 5–8.4 million vertices, 9.9–16.7 million faces.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import yaml
from skimage import filters, morphology
from skimage.measure import marching_cubes

logger = logging.getLogger(__name__)


def load_config(config_path: str | Path = "config/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Volume refinement steps
# ---------------------------------------------------------------------------

def region_grow(mask_volume: np.ndarray,
                ct_volume: np.ndarray,
                threshold: float = 0.4) -> np.ndarray:
    """
    Step 2: Region growing from high-confidence bone seeds.

    Seeds: voxels where mask value exceeds *threshold* (i.e. the U-Net
    predicted bone with high probability before binarisation).
    The grown region fills connected voxels that share bone-like intensity
    with the seeds, recapturing fine bone margins that conservative
    binarisation might exclude.

    Parameters
    ----------
    mask_volume : (Z, Y, X) float — raw U-Net sigmoid output [0, 1]
    ct_volume   : (Z, Y, X) float — original preprocessed CT intensities
    threshold   : confidence threshold for seed selection

    Returns
    -------
    Binary grown volume (bool)
    """
    seeds = mask_volume > threshold
    # Use seeds' CT intensity stats to define valid neighbour range
    seed_ct = ct_volume[seeds]
    if len(seed_ct) == 0:
        return seeds
    lo = max(0.0, seed_ct.mean() - 2.0 * seed_ct.std())
    hi = min(1.0, seed_ct.mean() + 2.0 * seed_ct.std())

    from skimage.morphology import binary_dilation, ball
    grown = seeds.copy()
    candidate = (ct_volume >= lo) & (ct_volume <= hi)
    # Iterative expansion (fast approximation of seeded region growing)
    for _ in range(5):
        expanded = binary_dilation(grown, footprint=ball(1))
        grown = grown | (expanded & candidate)
    return grown


def otsu_intensity_filter(volume: np.ndarray,
                           ct_volume: np.ndarray) -> np.ndarray:
    """
    Step 3: Otsu-based intensity filtering to exclude non-bone regions.

    Refines the grown region by applying a global Otsu threshold to the
    CT values within it, dropping voxels below the threshold.
    """
    ct_in_mask = ct_volume[volume > 0]
    if len(ct_in_mask) == 0:
        return volume
    otsu_thresh = filters.threshold_otsu(ct_in_mask)
    return (volume > 0) & (ct_volume >= otsu_thresh)


def morphological_cleanup(binary_volume: np.ndarray,
                           closing_radius: int = 2,
                           min_fragment_voxels: int = 50) -> np.ndarray:
    """
    Step 4: Morphological closing + small fragment removal.

    Closing (dilation then erosion) bridges small gaps between bone
    fragments and smooths jagged surface boundaries.
    Fragment removal eliminates isolated noise clusters smaller than
    min_fragment_voxels.
    """
    from skimage.morphology import ball, binary_closing, remove_small_objects
    struct = ball(closing_radius)
    closed = binary_closing(binary_volume, footprint=struct)
    cleaned = remove_small_objects(closed, min_size=min_fragment_voxels)
    return cleaned.astype(np.uint8)


# ---------------------------------------------------------------------------
# Marching cubes
# ---------------------------------------------------------------------------

def extract_surface_mesh(
    binary_volume: np.ndarray,
    voxel_size_mm: tuple[float, float, float] = (1.0, 1.0, 1.0),
    iso_level: float = 0.5,
    step_size: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Step 5: Marching cubes surface extraction.

    "The marching cubes algorithm was employed to convert the 3D volume
    into a triangulated surface mesh using an iso-level of 0.5. This
    resulted in watertight 3D meshes suitable for further analysis,
    visualization, or 3D printing."
    — manuscript §3D mesh generation

    Parameters
    ----------
    binary_volume  : (Z, Y, X) uint8 — cleaned binary mask
    voxel_size_mm  : (dz, dy, dx) spacing in mm (from CT header)
    iso_level      : marching cubes iso-surface value (default 0.5)
    step_size      : voxel step size (1 = full resolution)

    Returns
    -------
    vertices : (V, 3) float64 — vertex coordinates in mm
    faces    : (F, 3) int     — triangle face indices
    normals  : (V, 3) float64 — vertex normals
    """
    verts, faces, normals, _ = marching_cubes(
        binary_volume,
        level=iso_level,
        spacing=voxel_size_mm,
        step_size=step_size,
        allow_degenerate=False,
    )
    return verts, faces, normals


# ---------------------------------------------------------------------------
# STL export
# ---------------------------------------------------------------------------

def save_stl(
    vertices: np.ndarray,
    faces: np.ndarray,
    normals: np.ndarray,
    output_path: str | Path,
) -> Path:
    """
    Export the triangulated mesh to STL (Standard Tessellation Language).

    STL files are the standard deliverable for downstream morphometrics
    (GM, FEA, MDA) and 3D printing workflows.
    """
    try:
        from stl import mesh as stl_mesh
        fossil_mesh = stl_mesh.Mesh(np.zeros(faces.shape[0], dtype=stl_mesh.Mesh.dtype))
        for i, f in enumerate(faces):
            fossil_mesh.vectors[i] = vertices[f]
        fossil_mesh.save(str(output_path))
    except ImportError:
        # Fallback: write binary STL manually
        _write_binary_stl(vertices, faces, normals, output_path)

    logger.info("Saved mesh → %s  (%d vertices, %d faces)",
                output_path, len(vertices), len(faces))
    return Path(output_path)


def _write_binary_stl(vertices, faces, normals, path):
    import struct
    with open(path, "wb") as f:
        f.write(b"\0" * 80)                            # header
        f.write(struct.pack("<I", len(faces)))          # triangle count
        for i, face in enumerate(faces):
            n = normals[face].mean(axis=0)
            n /= (np.linalg.norm(n) + 1e-8)
            f.write(struct.pack("<3f", *n))
            for vi in face:
                f.write(struct.pack("<3f", *vertices[vi]))
            f.write(struct.pack("<H", 0))               # attribute byte count


# ---------------------------------------------------------------------------
# End-to-end mesh generation
# ---------------------------------------------------------------------------

def generate_mesh(
    mask_volume_raw: np.ndarray,
    ct_volume: np.ndarray,
    output_path: str | Path,
    voxel_size_mm: tuple[float, float, float] = (1.0, 1.0, 1.0),
    config_path: str | Path = "config/config.yaml",
) -> dict:
    """
    Full 3D mesh generation pipeline for one specimen.

    Parameters
    ----------
    mask_volume_raw : (Z, Y, X) float — raw sigmoid output from U-Net [0, 1]
    ct_volume       : (Z, Y, X) float — preprocessed CT intensities
    output_path     : destination .stl file
    voxel_size_mm   : (dz, dy, dx) in mm
    config_path     : path to config.yaml

    Returns
    -------
    dict with mesh statistics (n_vertices, n_faces, runtime_s)
    """
    cfg = load_config(config_path)["mesh"]
    t0 = time.time()

    logger.info("Step 1: Binarising U-Net output …")
    binary = (mask_volume_raw > 0.5).astype(np.uint8)

    logger.info("Step 2: Region growing …")
    grown = region_grow(mask_volume_raw, ct_volume,
                        threshold=cfg["region_grow_threshold"])

    logger.info("Step 3: Otsu intensity filtering …")
    filtered = otsu_intensity_filter(grown.astype(np.uint8), ct_volume)

    logger.info("Step 4: Morphological cleanup …")
    cleaned = morphological_cleanup(
        filtered,
        closing_radius=cfg["closing_radius"],
        min_fragment_voxels=cfg["min_fragment_voxels"],
    )

    logger.info("Step 5: Marching cubes …")
    verts, faces, normals = extract_surface_mesh(
        cleaned,
        voxel_size_mm=voxel_size_mm,
        iso_level=cfg["iso_level"],
        step_size=cfg["step_size"],
    )

    logger.info("Step 6: Saving STL …")
    save_stl(verts, faces, normals, output_path)

    runtime = time.time() - t0
    logger.info("Mesh generation complete in %.1f s", runtime)

    return {
        "n_vertices": len(verts),
        "n_faces":    len(faces),
        "runtime_s":  runtime,
        "output_path": str(output_path),
    }
