# config/

Central configuration for all pipeline stages.

## Files

| File | Description |
|---|---|
| `config.yaml` | All hyperparameters, file paths, and training settings |

## Key Parameters

```yaml
# Data
image_size: 224
normalise: true
z_standardise: true

# MoCo v2
mocov2_epochs: 600
mocov2_batch_size: 128
mocov2_lr: 0.015
mocov2_temperature: 0.07
mocov2_queue_size: 8192
mocov2_momentum: 0.999

# U-Net
unet_epochs: 500
unet_optimiser: Adam
unet_scheduler: cosine

# Mesh
marching_cubes_iso: 0.5
```

## Notes

- All scripts load from `config/config.yaml` at runtime
- Edit this file to change paths, hyperparameters, or output directories before running any stage
