"""
inference.py
============
End-to-end inference pipeline: raw CT TIFF stack → segmentation masks → 3D mesh.

Usage
-----
    python inference.py --ct_dir path/to/tiffs/ --output_dir outputs/
    
    # or from Python:
    from inference import FossilSegmentationPipeline
    pipeline = FossilSegmentationPipeline.from_config("config/config.yaml")
    pipeline.run(ct_dir="path/to/tiffs/", output_dir="outputs/")

Typical runtime: 1–3 minutes per specimen on a single GPU.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from data.preprocessing import preprocess_specimen
from mesh.generate_mesh import generate_mesh
from unet.model import FossilUNet

logger = logging.getLogger(__name__)


def load_config(config_path: str | Path = "config/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


class FossilSegmentationPipeline:
    """
    Fully automated, annotation-free fossil CT segmentation pipeline.

    Loads trained U-Net weights and runs the complete inference workflow:
        preprocessing → U-Net segmentation → 3D mesh generation

    Parameters
    ----------
    unet_ckpt   : path to trained U-Net checkpoint (unet_best.pt)
    config_path : path to config.yaml
    device      : torch device (auto-detected if None)
    """

    def __init__(
        self,
        unet_ckpt: str | Path,
        config_path: str | Path = "config/config.yaml",
        device: torch.device | None = None,
    ):
        self.config_path = config_path
        self.cfg = load_config(config_path)
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        logger.info("Device: %s", self.device)

        logger.info("Loading U-Net from %s …", unet_ckpt)
        self.model = FossilUNet.from_checkpoint(unet_ckpt, map_location=self.device)
        self.model.to(self.device).eval()
        logger.info("U-Net loaded.")

    @classmethod
    def from_config(
        cls,
        config_path: str | Path = "config/config.yaml",
        device: torch.device | None = None,
    ) -> "FossilSegmentationPipeline":
        cfg = load_config(config_path)
        unet_ckpt = Path(cfg["paths"]["unet_ckpt_dir"]) / "unet_best.pt"
        return cls(unet_ckpt=unet_ckpt, config_path=config_path, device=device)

    def run(
        self,
        ct_dir: str | Path,
        output_dir: str | Path,
        specimen_name: str = "",
        voxel_size_mm: tuple[float, float, float] = (1.0, 1.0, 1.0),
        save_masks: bool = True,
    ) -> dict:
        """
        Run the full inference pipeline for one specimen.

        Parameters
        ----------
        ct_dir        : directory containing raw TIFF slices
        output_dir    : directory for outputs (masks, mesh, log)
        specimen_name : optional label (defaults to ct_dir folder name)
        voxel_size_mm : (dz, dy, dx) physical voxel spacing in mm
        save_masks    : whether to save the 2D U-Net mask stack as .npy

        Returns
        -------
        dict with timing and mesh statistics
        """
        ct_dir = Path(ct_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        specimen_name = specimen_name or ct_dir.name

        t_total = time.time()

        # ---- Step 1: Preprocess ----
        logger.info("[%s] Preprocessing CT slices …", specimen_name)
        t0 = time.time()
        volume = preprocess_specimen(ct_dir, config_path=self.config_path)
        # volume: (N, H, W) float32
        logger.info("  %d slices preprocessed in %.1f s", volume.shape[0],
                    time.time() - t0)

        # ---- Step 2: U-Net segmentation ----
        logger.info("[%s] Running U-Net segmentation …", specimen_name)
        t0 = time.time()
        mask_volume_raw = self._segment(volume)   # (N, H, W) float32 [0,1]
        logger.info("  Segmentation complete in %.1f s", time.time() - t0)

        if save_masks:
            mask_path = output_dir / f"{specimen_name}_masks.npy"
            np.save(mask_path, mask_volume_raw)
            logger.info("  Masks saved → %s", mask_path)

        # ---- Step 3: 3D mesh ----
        logger.info("[%s] Generating 3D mesh …", specimen_name)
        stl_path = output_dir / f"{specimen_name}.stl"
        mesh_stats = generate_mesh(
            mask_volume_raw=mask_volume_raw,
            ct_volume=volume,
            output_path=stl_path,
            voxel_size_mm=voxel_size_mm,
            config_path=self.config_path,
        )

        total_time = time.time() - t_total
        logger.info("[%s] Pipeline complete in %.1f s  "
                    "(%d vertices, %d faces)",
                    specimen_name, total_time,
                    mesh_stats["n_vertices"], mesh_stats["n_faces"])

        return {
            "specimen": specimen_name,
            "n_slices": volume.shape[0],
            "total_time_s": total_time,
            **mesh_stats,
        }

    @torch.no_grad()
    def _segment(
        self,
        volume: np.ndarray,
        batch_size: int = 8,
    ) -> np.ndarray:
        """
        Run the U-Net on a CT
