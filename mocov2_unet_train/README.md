# mocov2_unet_train/

Modified U-Net for fine segmentation of fossil CT slices, initialised with MoCo v2 pre-trained weights via knowledge fusion.

## Architecture

- **Encoder**: 4-stage U-Net encoder (1→64→128→256→512), initialised from MoCo v2 checkpoint
- **Bottleneck**: DoubleConv(512→1024)
- **Decoder**: 4 upsampling stages with skip connections
- **Loss**: BCEWithLogitsLoss

## Scripts

| File | Description |
|---|---|
| `train.py` | 500 epochs, Adam optimiser, cosine LR, best-checkpoint saving |

## Training

```bash
bash bash_scripts/04_train_unet.sh
```

## Results

| Metric | Epoch 500 |
|---|---|
| Dice Score | ~0.85–0.99 |
| IoU | ~0.75–0.97 |
| Best batch Dice | ≥ 0.985 |

## Notes

- MoCo v2 encoder weights must be available before training — see `checkpoints/mocov2/`
- Supports `--resume` flag for automatic checkpoint resumption
