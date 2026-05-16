#!/usr/bin/env bash
set -euo pipefail
echo "=== Stage 4: U-Net Training via MoCo v2 Knowledge Fusion (500 epochs, ~6 hrs on A100) ==="
python mocov2_unet/train.py
echo "Trained model → checkpoints/unet/unet_model.pth"
