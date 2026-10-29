"""One forward and backward pass on real ISTD data to catch path or shape problems before training.

    python sanity_check.py --data-root data --image-size 128 --batch-size 2
"""
import argparse

import torch
from torch.utils.data import DataLoader

from shadowfix.data import ISTDDataset, list_triplets
from shadowfix.losses import ShadowRemovalLoss
from shadowfix.model import build_model, model_input


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", required=True)
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--use-mask-input", action="store_true")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    loader = DataLoader(ISTDDataset(list_triplets(args.data_root, "train"), args.image_size), batch_size=args.batch_size)
    batch = next(iter(loader))
    shadow, mask, target = (batch[k].to(device) for k in ("shadow", "mask", "target"))

    model = build_model({"use_mask_input": args.use_mask_input}).to(device)
    pred = model(model_input(shadow, mask, args.use_mask_input))
    loss, details = ShadowRemovalLoss()(pred, target, mask)
    loss.backward()

    print(f"Input shape: {tuple(shadow.shape)}")
    print(f"Prediction shape: {tuple(pred.shape)}")
    print(f"Loss details: {details}")
    assert pred.shape == target.shape and torch.isfinite(loss)
    print("Sanity check passed.")


if __name__ == "__main__":
    main()
