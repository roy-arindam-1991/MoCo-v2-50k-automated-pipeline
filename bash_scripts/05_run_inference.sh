#!/usr/bin/env bash
# Usage: bash bash_scripts/05_run_inference.sh path/to/tiffs/ outputs/
set -euo pipefail
CT_DIR=${1:-"data/raw/new_specimen/"}
OUT_DIR=${2:-"outputs/"}
echo "=== Stage 5: Inference on $CT_DIR ==="
python inference.py --ct_dir "$CT_DIR" --output_dir "$OUT_DIR"
echo "Results → $OUT_DIR"
