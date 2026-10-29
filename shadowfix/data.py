"""ISTD dataset loading.

ISTD stores each sample as three images with the same file name:
    <split>/<split>_A/  shadow image
    <split>/<split>_B/  binary shadow mask
    <split>/<split>_C/  shadow-free ground truth
"""
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def find_split_dir(data_root, split):
    """Return the folder that holds <split>_A, <split>_B and <split>_C.

    Accepts the extracted archive layout (data/ISTD_Dataset/train) as well as data/train.
    """
    root = Path(data_root)
    candidates = [root / split, root / "ISTD_Dataset" / split]
    candidates += [p for p in root.glob(f"*/{split}") if p.is_dir()]
    for c in candidates:
        if (c / f"{split}_A").is_dir() and (c / f"{split}_C").is_dir():
            return c
    raise FileNotFoundError(
        f"Could not find {split}_A and {split}_C under {root}. "
        "Expected folders like train/train_A, train/train_B, train/train_C."
    )


def list_triplets(data_root, split):
    """List (shadow, mask or None, target) path triplets for a split, sorted by name."""
    split_dir = find_split_dir(data_root, split)
    a_dir, b_dir, c_dir = (split_dir / f"{split}_{k}" for k in "ABC")
    records = []
    for a in sorted(a_dir.iterdir()):
        if a.suffix.lower() not in IMAGE_EXTS:
            continue
        c = _same_stem(c_dir, a.stem)
        if c is None:
            continue
        b = _same_stem(b_dir, a.stem) if b_dir.is_dir() else None
        records.append((a, b, c))
    if not records:
        raise FileNotFoundError(f"No image triplets found in {split_dir}")
    return records


def _same_stem(folder, stem):
    for ext in (".png", ".jpg", ".jpeg", ".bmp"):
        p = folder / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def split_train_val(records, val_fraction=0.1, seed=42):
    """Deterministic train/validation split (1,330 ISTD triplets -> 1,197 / 133)."""
    records = list(records)
    random.Random(seed).shuffle(records)
    n_val = max(1, round(len(records) * val_fraction))
    return records[n_val:], records[:n_val]


def load_rgb(path, size=None):
    img = Image.open(path).convert("RGB")
    if size:
        img = img.resize((size, size), Image.BICUBIC)
    return img


def load_mask(path, size=None, like=None):
    """Binary mask in {0, 1}. Missing masks become all zeros."""
    if path is None:
        w, h = like.size
        return Image.fromarray(np.zeros((h, w), np.uint8))
    m = Image.open(path).convert("L")
    if size:
        m = m.resize((size, size), Image.NEAREST)
    return m


def to_tensor(img):
    arr = np.asarray(img, dtype=np.float32) / 255.0
    if arr.ndim == 2:
        arr = arr[None]
    else:
        arr = arr.transpose(2, 0, 1)
    return torch.from_numpy(np.ascontiguousarray(arr))


def to_image(tensor):
    """CHW float tensor in [0, 1] -> PIL RGB image."""
    arr = tensor.detach().clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
    return Image.fromarray((arr * 255.0 + 0.5).astype(np.uint8))


class ISTDDataset(Dataset):
    """Returns dicts with shadow (3,H,W), mask (1,H,W), target (3,H,W) in [0, 1]."""

    def __init__(self, records, image_size=256, augment=False):
        self.records = list(records)
        self.image_size = image_size
        self.augment = augment

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        a_path, b_path, c_path = self.records[idx]
        shadow = load_rgb(a_path, self.image_size)
        target = load_rgb(c_path, self.image_size)
        mask = load_mask(b_path, self.image_size, like=shadow)

        shadow, mask, target = to_tensor(shadow), to_tensor(mask), to_tensor(target)
        mask = (mask > 0.5).float()

        if self.augment:
            if random.random() < 0.5:
                shadow, mask, target = shadow.flip(-1), mask.flip(-1), target.flip(-1)
            if random.random() < 0.5:
                shadow, mask, target = shadow.flip(-2), mask.flip(-2), target.flip(-2)

        return {"shadow": shadow, "mask": mask, "target": target, "name": a_path.stem}
