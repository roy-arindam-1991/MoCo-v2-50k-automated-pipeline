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

# SimCLR
simclr_epochs: 250
simclr_batch_size: 64
simclr_lr: 3.0e-4
simclr_temperature: 0.5

# U-Net
unet_epochs: 500
unet_optimiser: AdamW
unet_scheduler: OneCycleLR

# Mesh
marching_cubes_iso: 0.5
```

## Notes

- All scripts load from `config/config.yaml` at runtime
- Edit this file to change paths, hyperparameters, or output directories before running any stage
