# mesh/

3D mesh generation from stacked 2D segmentation masks produced by the U-Net.

## Pipeline

1. Stack per-slice 2D masks into a 3D binary volume
2. Apply region growing and Otsu intensity filtering for volume refinement
3. Run marching cubes algorithm (iso-level = 0.5) to extract surface mesh
4. Export as watertight STL file for downstream morphometric analysis

## Scripts

| File | Description |
|---|---|
| `generate_mesh.py` | Full mesh generation: stack → refine → marching cubes → STL export |

## Usage

```bash
bash scripts/05_run_inference.sh
```

## Output

- Watertight `.stl` mesh per specimen
- Typical runtime: **1–3 minutes per specimen** on A100 GPU
