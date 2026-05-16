#!/usr/bin/env bash
set -euo pipefail
echo "=== Stage 3: MoCo v2 Validation (Grad-CAM + UMAP, ~4 min on A100) ==="
python mocov2_validation/validate.py
echo "Grad-CAM overlays → outputs/gradcam/"
echo "UMAP projection   → outputs/umap.svg"
echo "UMAP coordinates  → outputs/umap.csv"
