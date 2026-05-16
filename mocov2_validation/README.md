# mocov2_validation/

Qualitative evaluation of MoCo v2 learned representations via Grad-CAM and UMAP.

## Scripts

| File | Description |
|---|---|
| `validate.py` | Embedding extraction, Grad-CAM heatmaps, UMAP projection |

## Usage

```bash
bash bash_scripts/03_mocov2_validation.sh
```

## Outputs

- `outputs/gradcam/` — per-slice Grad-CAM TIFF overlays (3,765 validation slices)
- `outputs/umap.svg` — 2D UMAP projection coloured by kernel density estimate
- `outputs/umap.csv` — UMAP coordinates with slice filenames

## Pipeline

1. **Embedding extraction** — frozen encoder_q produces 128-d embeddings for all val slices
2. **Grad-CAM** — gradient-weighted heatmaps at layer4[-1], masked by bone signal
3. **UMAP** — 2D projection of all embeddings (n_neighbors=30) with KDE density colouring
