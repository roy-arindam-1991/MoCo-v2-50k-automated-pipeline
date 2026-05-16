#!/usr/bin/env bash
set -euo pipefail
echo "=== Stage 2: MoCo v2 Pre-training (600 epochs, ~7.8 hrs on A100) ==="
python mocov2_training/train.py
echo "Best checkpoint → checkpoints/mocov2/srxtm_moco_r50_best.pth"
