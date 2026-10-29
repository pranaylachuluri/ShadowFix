"""Remove shadows from your own images with a trained checkpoint.

    python inference.py --checkpoint runs/baseline_final/best.pth --input photo.jpg --output-dir outputs
    python inference.py --checkpoint runs/mask_guided/best.pth --input photos/ --mask-dir masks/ --output-dir outputs

The model runs at the training resolution (256x256). The predicted correction is upsampled
and applied to the full-resolution photo, so the output keeps the original size and detail.
Mask-guided checkpoints need a shadow mask per image (same file name, white = shadow).
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from shadowfix.checkpoint import load_model
from shadowfix.data import IMAGE_EXTS, to_image, to_tensor
from shadowfix.model import model_input


def find_mask(mask_dir, stem):
    for ext in (".png", ".jpg", ".jpeg", ".bmp"):
        p = Path(mask_dir) / f"{stem}{ext}"
        if p.exists():
            return p
    return None


@torch.no_grad()
def restore(model, config, image, mask=None, device="cpu"):
    """PIL RGB image (any size) -> restored PIL RGB image of the same size."""
    size = config["image_size"]
    full = to_tensor(image).unsqueeze(0).to(device)
    small = F.interpolate(full, size=(size, size), mode="bicubic", align_corners=False).clamp(0, 1)
    if mask is None:
        m = torch.zeros(1, 1, size, size, device=device)
    else:
        m = to_tensor(mask.convert("L").resize((size, size), Image.NEAREST)).unsqueeze(0).to(device)
        m = (m > 0.5).float()
    pred = model(model_input(small, m, config.get("use_mask_input", False))).float()
    # Upsample the correction (prediction minus input) rather than the low-resolution image
    delta = F.interpolate(pred - small, size=full.shape[-2:], mode="bilinear", align_corners=False)
    return to_image((full + delta)[0])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--input", required=True, help="an image file or a folder of images")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--mask", help="shadow mask for a single input image")
    p.add_argument("--mask-dir", help="folder of shadow masks named like the input images")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config = load_model(args.checkpoint, device)
    needs_mask = config.get("use_mask_input", False)

    src = Path(args.input)
    inputs = sorted(f for f in src.iterdir() if f.suffix.lower() in IMAGE_EXTS) if src.is_dir() else [src]
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for path in inputs:
        mask_path = Path(args.mask) if args.mask else (find_mask(args.mask_dir, path.stem) if args.mask_dir else None)
        if needs_mask and mask_path is None:
            print(f"Skipping {path.name}: this checkpoint is mask-guided and no mask was given")
            continue
        image = Image.open(path).convert("RGB")
        mask = Image.open(mask_path) if mask_path else None
        result = restore(model, config, image, mask, device)
        dest = out / f"{path.stem}.png"
        result.save(dest)
        print(f"Saved: {dest}")


if __name__ == "__main__":
    main()
