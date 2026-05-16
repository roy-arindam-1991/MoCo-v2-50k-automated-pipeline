# masking/

Deterministic rule-based pipeline to generate coarse binary bone masks from CT slices.  
No training required — fully reproducible with fixed parameters.

## Pipeline Stages

1. **Otsu thresholding** — automatic intensity-based foreground separation
2. **Region growing** — expand seed regions based on intensity similarity
3. **Morphological operations** — erosion, dilation, closing to clean boundaries
4. **Connected component filtering** — remove small spurious components
5. **Mask refinement** — final binary mask output per slice

## Scripts

| File | Description |
|---|---|
| `rule_based_masking.py` | Full 5-stage deterministic masking pipeline |
| `run_masking.sh` | Applies masking across all U-Net partition datasets |

## Usage

```bash
bash scripts/03_generate_masks.sh
```
