#!/usr/bin/env python3

import os
import time
import logging
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import h5py
import tifffile
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from skimage import exposure

try:
    from umap import UMAP
except ImportError:
    from umap.umap_ import UMAP


def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, f"val_log_r50_{time.strftime('%Y%m%d-%H%M%S')}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )
    logging.info(f"Logging initialized: {log_file}")


def enhance_bone_visibility(img_np):
    p2, p98 = np.percentile(img_np, (1, 99))
    img = np.clip(img_np, p2, p98)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    img = exposure.equalize_adapthist(img, clip_limit=0.03)
    return (img * 255).astype(np.uint8)


def ensure_uint8(img_np, max_dim=1024):
    img = enhance_bone_visibility(img_np)
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        img = Image.fromarray(img)
        img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
        img = np.array(img)
    return img


def generate_masked_gradcam(activations, gradients, target_shape, image_u8):
    weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
    heatmap = torch.sum(weights * activations, dim=1, keepdim=True)
    heatmap = F.relu(heatmap)
    heatmap = F.interpolate(heatmap, size=target_shape[:2], mode="bilinear", align_corners=False)
    heatmap = heatmap.squeeze().cpu().numpy()
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    bone_mask = np.clip(image_u8.astype(float) / 255.0, 0.1, 1.0)
    return heatmap * bone_mask


def save_overlay(original_u8, heatmap, path, alpha=0.35):
    rgb = np.stack([original_u8] * 3, axis=-1)
    cmap = plt.colormaps["jet"]
    heat_color = (cmap(heatmap)[:, :, :3] * 255).astype(np.uint8)
    overlay = (rgb * (1 - alpha) + heat_color * alpha).astype(np.uint8)
    tifffile.imwrite(path, overlay)


class InferenceHDF5Dataset(Dataset):
    def __init__(self, h5_path, transform):
        self.h5_path = h5_path
        self.transform = transform
        self.h5_file = None
        self.samples = []
        with h5py.File(h5_path, "r") as f:
            f.visititems(
                lambda n, o: self.samples.append(n)
                if isinstance(o, h5py.Dataset) and o.ndim >= 2 else None
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, "r")
        name = self.samples[idx]
        data = self.h5_file[name][:]
        if data.ndim == 3:
            data = data[data.shape[0] // 2]
        img_u8 = ensure_uint8(data)
        img = Image.fromarray(img_u8).convert("RGB")
        return self.transform(img), name.replace("/", "_"), img_u8

    def __del__(self):
        if self.h5_file is not None:
            try:
                self.h5_file.close()
            except Exception:
                pass


def collate_fn(batch):
    return (
        torch.stack([b[0] for b in batch]),
        [b[1] for b in batch],
        [b[2] for b in batch]
    )


def run_analysis(h5_file, model_path, output_dir, batch_size):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Grayscale(3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = InferenceHDF5Dataset(h5_file, transform)
    logging.info(f"Dataset loaded: {len(dataset)} samples from {h5_file}")

    loader = DataLoader(dataset, batch_size=batch_size, num_workers=10, collate_fn=collate_fn)

    # ── Load checkpoint ──────────────────────────────────────────────────────
    # Training saves model.state_dict() from the full MoCo wrapper, so keys are:
    #   "encoder_q.conv1.weight", "encoder_q.layer1...", "encoder_k...", "queue"
    # mlp=False (default, --mlp not passed in SLURM), so fc = Linear(2048, 128).
    # We extract encoder_q weights only and load into plain resnet50(num_classes=128).
    logging.info(f"Loading checkpoint: {model_path}")
    ckpt = torch.load(model_path, map_location=device)
    sd = ckpt["state_dict"]

    # Strip DataParallel prefix if present
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    logging.info(f"Checkpoint raw key sample (first 5): {list(sd.keys())[:5]}")

    # Extract encoder_q only
    encoder_sd = {k[len("encoder_q."):]: v for k, v in sd.items() if k.startswith("encoder_q.")}
    if not encoder_sd:
        raise RuntimeError(
            f"No 'encoder_q.*' keys found. Prefixes present: {sorted({k.split('.')[0] for k in sd.keys()})}"
        )
    logging.info(f"encoder_q key sample (first 5): {list(encoder_sd.keys())[:5]}")

    # Plain ResNet50 — fc=Linear(2048,128), no MLP head replacement
    encoder = models.resnet50(num_classes=128)
    dim_mlp = encoder.fc.weight.shape[1]
    encoder.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), nn.Linear(dim_mlp, 128))
    encoder = encoder.to(device)
    logging.info(f"Model key sample (first 5): {list(encoder.state_dict().keys())[:5]}")

    encoder.load_state_dict(encoder_sd, strict=True)
    encoder.eval()
    logging.info("Checkpoint loaded successfully (strict=True)")

    # ── Grad-CAM hooks ───────────────────────────────────────────────────────
    grads, acts = {}, {}

    def hook_fn(name, is_grad):
        def hook(m, i, o):
            if is_grad:
                grads[name] = o[0].detach() if isinstance(o, tuple) else o.detach()
            else:
                acts[name] = o.detach()
        return hook

    target_layer = encoder.layer4[-1]
    target_layer.register_forward_hook(hook_fn("layer", False))
    target_layer.register_full_backward_hook(hook_fn("layer", True))

    os.makedirs(os.path.join(output_dir, "gradcam"), exist_ok=True)
    embeddings, names_all = [], []

    # ── Inference loop ───────────────────────────────────────────────────────
    for imgs, names, u8s in tqdm(loader, total=len(dataset) // batch_size):
        imgs = imgs.to(device)
        acts.clear()
        grads.clear()

        output = encoder(imgs)
        score = output[:, output.detach().mean(0).argmax()].sum()
        encoder.zero_grad()
        score.backward()

        embeddings.append(output.detach().cpu().numpy())
        names_all.extend(names)

        for i in range(len(names)):
            heatmap = generate_masked_gradcam(
                acts["layer"][i].unsqueeze(0),
                grads["layer"][i].unsqueeze(0),
                u8s[i].shape[:2],
                u8s[i]
            )
            save_overlay(u8s[i], heatmap, os.path.join(output_dir, "gradcam", f"{names[i]}.tif"))

    logging.info(f"Inference complete. {len(names_all)} embeddings extracted.")

    # ── UMAP ─────────────────────────────────────────────────────────────────
    logging.info("Running UMAP...")
    emb = np.vstack(embeddings)
    n_neighbors = min(30, len(emb) - 1)
    embedding_2d = UMAP(n_neighbors=n_neighbors).fit_transform(emb)

    plt.figure(figsize=(10, 8))
    xy = embedding_2d.T
    z = gaussian_kde(xy)(xy)
    idx = z.argsort()
    plt.scatter(embedding_2d[idx, 0], embedding_2d[idx, 1], c=z[idx], s=10, cmap="viridis")
    plt.colorbar(label="Density")
    plt.title("MoCo v2 ResNet50 — UMAP Embedding")
    plt.tight_layout()

    umap_svg = os.path.join(output_dir, "umap.svg")
    plt.savefig(umap_svg)
    plt.close()
    logging.info(f"UMAP plot saved: {umap_svg}")

    umap_csv = os.path.join(output_dir, "umap.csv")
    pd.DataFrame({"file": names_all, "x": embedding_2d[:, 0], "y": embedding_2d[:, 1]}).to_csv(umap_csv, index=False)
    logging.info(f"UMAP coordinates saved: {umap_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5_file",    required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    setup_logging(args.output_dir)
    run_analysis(args.h5_file, args.model_path, args.output_dir, args.batch_size)
