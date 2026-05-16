import os
import h5py
import numpy as np
import re
from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import logging
import time
import argparse
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

# ===========================
# Helper: Dataset Discovery
# ===========================
def _is_image_like(dset: h5py.Dataset) -> bool:
    if not isinstance(dset, h5py.Dataset): return False
    if dset.dtype.kind not in ("u", "i", "f"): return False
    if dset.ndim == 2:
        h, w = dset.shape
        return h >= 32 and w >= 32
    if dset.ndim == 3:
        n, h, w = dset.shape
        return n >= 1 and h >= 32 and w >= 32
    if dset.ndim == 4:
        n, h, w, c = dset.shape
        return n >= 1 and h >= 32 and w >= 32 and c in (1, 3)
    return False

def discover_image_datasets(h5_path: str):
    found = []
    with h5py.File(h5_path, "r") as f:
        def visit(name, obj):
            if isinstance(obj, h5py.Dataset) and _is_image_like(obj):
                shape = tuple(obj.shape)
                n = 1 if obj.ndim == 2 else int(shape[0])
                found.append({"path": "/" + name, "n": n, "shape": shape})
        f.visititems(visit)
    return found

# ===========================
# Optimized Dataset Class
# ===========================
class H5SliceDataset(Dataset):
    def __init__(self, h5_path, transform=None, is_final_eval=False):
        self.h5_path = h5_path
        self.transform = transform
        self.is_final_eval = is_final_eval
        self.h5_file = None 

        logging.info(f"Scanning HDF5: {self.h5_path}...")
        self.metadata = discover_image_datasets(h5_path)
        
        self.index_map = []
        for meta in self.metadata:
            for i in range(meta['n']):
                self.index_map.append((meta['path'], i))
        
        if len(self.index_map) == 0:
            raise ValueError(f"No valid image datasets found in {h5_path}!")
        logging.info(f"Found {len(self.metadata)} datasets with {len(self.index_map)} total slices.")

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, 'r')
            
        dset_path, slice_idx = self.index_map[idx]
        dset = self.h5_file[dset_path]
        
        if dset.ndim == 2:
            data = dset[:].astype(np.float32)
        elif dset.ndim == 3:
            data = dset[slice_idx, :, :].astype(np.float32)
        elif dset.ndim == 4:
            data = dset[slice_idx, :, :, 0].astype(np.float32)

        p_low, p_high = np.percentile(data, (1, 99))
        data = np.clip(data, p_low, p_high)
        data = (data - p_low) / (p_high - p_low + 1e-8)
        img = Image.fromarray((data * 255).astype(np.uint8))

        if self.transform:
            x1 = self.transform(img)
            x2 = x1 if self.is_final_eval else self.transform(img)
            return x1, x2, 0
        return img, img, 0

# ===========================
# Augmentations & MoCo Model
# ===========================
def get_srxtm_augmentation(size=224):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomResizedCrop(size=size, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=180),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.5, 3.0))], p=0.6),
        transforms.ToTensor()
    ])

class MoCo(nn.Module):
    def __init__(self, base_encoder, dim=128, K=65536, m=0.999, T=0.07, mlp=False):
        super(MoCo, self).__init__()
        self.K, self.m, self.T = K, m, T
        self.encoder_q = base_encoder(num_classes=dim)
        self.encoder_k = base_encoder(num_classes=dim)
        if mlp:
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
            self.encoder_k = base_encoder(num_classes=dim)
            self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        if ptr + batch_size > self.K:
            remainder = self.K - ptr
            self.queue[:, ptr:] = keys[:remainder].T
            self.queue[:, :batch_size - remainder] = keys[remainder:].T
            ptr = (ptr + batch_size) % self.K
        else:
            self.queue[:, ptr:ptr + batch_size] = keys.T
            ptr = (ptr + batch_size) % self.K
        self.queue_ptr[0] = ptr

    def forward(self, im_q, im_k):
        q = nn.functional.normalize(self.encoder_q(im_q), dim=1)
        with torch.no_grad():
            self._momentum_update_key_encoder()
            k = nn.functional.normalize(self.encoder_k(im_k), dim=1)
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        logits = torch.cat([l_pos, l_neg], dim=1) / self.T
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(logits.device)
        self._dequeue_and_enqueue(k)
        return logits, labels

# ===========================
# Checkpoint & Training Utilities
# ===========================
def find_latest_checkpoint(output_dir):
    if not os.path.exists(output_dir): return None, 0
    checkpoint_files = [f for f in os.listdir(output_dir) if "srxtm_moco_r50_e" in f and f.endswith(".pth")]
    if not checkpoint_files: return None, 0
    epochs = [int(re.findall(r'\d+', f)[0]) for f in checkpoint_files]
    latest_epoch = max(epochs)
    latest_file = os.path.join(output_dir, f"srxtm_moco_r50_e{latest_epoch}.pth")
    return latest_file, latest_epoch

def setup_logger(output_dir):
    log_dir = os.path.join(output_dir, "log")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"srxtm_moco_log_{time.strftime('%Y%m%d-%H%M%S')}.txt")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
                        handlers=[logging.FileHandler(log_file), logging.StreamHandler()])
    return log_file

def save_metrics_plot(train_losses, train_accs, output_dir):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, color='tab:red', label='Loss')
    plt.title('MoCo V2 Loss (ResNet50)')
    plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.6); plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, color='tab:blue', label='Accuracy')
    plt.title('MoCo V2 Accuracy (ResNet50)')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy')
    plt.grid(True, linestyle='--', alpha=0.6); plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "srxtm_moco_r50_metrics.svg"), format='svg')
    plt.close()

def train_one_epoch(model, dataloader, optimizer, epoch, args):
    model.train()
    total_loss, total_samples, top1_correct = 0, 0, 0
    log_interval = max(1, len(dataloader) // 5)
    for batch_idx, (img_q, img_k, _) in enumerate(dataloader, 1):
        img_q, img_k = img_q.to(args.device), img_k.to(args.device)
        logits, labels = model(img_q, img_k)
        loss = F.cross_entropy(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        acc1 = (logits.argmax(dim=1) == labels).float().sum()
        top1_correct += acc1.item()
        total_samples += img_q.size(0)
        if batch_idx % log_interval == 0:
            logging.info(f"Epoch [{epoch+1}] Batch {batch_idx}/{len(dataloader)} Loss: {loss.item():.4f}")
    return total_loss / len(dataloader), top1_correct / total_samples

def train_moco(args):
    transform = get_srxtm_augmentation()
    dataset = H5SliceDataset(args.h5_file, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, 
                        num_workers=args.num_workers, drop_last=True, pin_memory=True)

    model = MoCo(models.resnet50, dim=args.moco_dim, K=args.moco_k, mlp=args.mlp).to(args.device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=1e-4, momentum=0.9)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    start_epoch = 0
    train_losses, train_accs = [], []
    checkpoint_path, latest_epoch = find_latest_checkpoint(args.output_dir)
    
    if checkpoint_path and not args.force_restart:
        logging.info(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=args.device)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint.get('epoch', 0)
        train_losses = checkpoint.get('loss_history', [])
        train_accs = checkpoint.get('acc_history', [])
        logging.info(f"Resumed from epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs):
        loss, acc = train_one_epoch(model, loader, optimizer, epoch, args)
        scheduler.step()
        train_losses.append(loss); train_accs.append(acc)
        logging.info(f"Epoch {epoch+1} Complete. Loss: {loss:.4f} Acc: {acc:.4f}")
        save_metrics_plot(train_losses, train_accs, args.output_dir)
        
        if (epoch + 1) % 10 == 0:
            checkpoint_file = os.path.join(args.output_dir, f"srxtm_moco_r50_e{epoch+1}.pth")
            torch.save({
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch + 1,
                'loss_history': train_losses,
                'acc_history': train_accs
            }, checkpoint_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.015)
    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--moco_dim", type=int, default=128)
    parser.add_argument("--moco_k", type=int, default=8192)
    parser.add_argument("--mlp", action="store_true")
    parser.add_argument("--force_restart", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    setup_logger(args.output_dir)
    train_moco(args)