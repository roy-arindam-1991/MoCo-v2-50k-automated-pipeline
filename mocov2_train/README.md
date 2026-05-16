# mocov2_train/

Self-supervised contrastive pre-training using MoCo v2 on unlabelled fossil CT slices.

## Architecture

- **Query encoder**: ResNet-50 backbone + 2-layer MLP projection head (2048 → 2048 → 128)
- **Key encoder**: Identical architecture, updated via momentum (not backprop)
- **Queue**: 128-d × K=8192 normalised embeddings (FIFO ring buffer)
- **Loss**: InfoNCE (Noise-Contrastive Estimation)

## Scripts

| File | Description |
|---|---|
| `train.py` | Training loop: 600 epochs, batch size 128, SGD, cosine annealing LR |

## Training

```bash
bash bash_scripts/02_train_mocov2.sh
```

## Augmentation Strategy

Augmentations are applied independently to each of the two views per slice:

- Random resized crop (scale 0.2–1.0)
- Horizontal and vertical flip
- Random rotation ±180°
- Gaussian blur (kernel=5, σ=0.5–3.0, p=0.6)

## Checkpointing

- Saved every 10 epochs to `checkpoints/mocov2/`
- Automatically resumes from latest checkpoint on restart
