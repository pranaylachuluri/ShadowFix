"""Evaluate a checkpoint on the ISTD test split.

Reports PSNR, SSIM and PSNR inside / outside the shadow mask for the shadow input,
the model and (optionally) the traditional colour-matching baseline, and saves
metrics_summary.csv, per_image_metrics.csv and side-by-side comparison images.

    python evaluate.py --data-root data --checkpoint runs/mask_guided/best.pth \
        --output-dir evaluation/mask_guided --num-visuals 20 --include-traditional
"""
import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from shadowfix.checkpoint import load_model
from shadowfix.data import ISTDDataset, list_triplets
from shadowfix.metrics import all_metrics
from shadowfix.model import model_input
from shadowfix.traditional import traditional_correction

METRICS = ["psnr", "ssim", "shadow_psnr", "nonshadow_psnr"]


def save_comparison(path, name, panels):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(panels), figsize=(3.6 * len(panels), 3.9))
    for ax, (title, img) in zip(axes, panels):
        arr = img.detach().cpu().clamp(0, 1)
        if arr.shape[0] == 1:
            ax.imshow(arr[0], cmap="gray", vmin=0, vmax=1)
        else:
            ax.imshow(arr.permute(1, 2, 0).numpy())
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle(name)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-visuals", type=int, default=20, help="comparison images to save")
    p.add_argument("--include-traditional", action="store_true")
    p.add_argument("--num-workers", type=int, default=2)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config = load_model(args.checkpoint, device)
    size, use_mask = config["image_size"], config.get("use_mask_input", False)

    records = list_triplets(args.data_root, args.split)
    with_masks = sum(r[1] is not None for r in records)
    print(f"Test records: {len(records)} triplets ({with_masks} with masks)")
    print(f"Device: {device}; image size: {size}")
    loader = DataLoader(ISTDDataset(records, size), batch_size=args.batch_size, num_workers=args.num_workers)

    out = Path(args.output_dir)
    (out / "comparisons").mkdir(parents=True, exist_ok=True)
    methods = ["Shadow input", "U-Net model"] + (["Traditional correction"] if args.include_traditional else [])
    per_image, saved = [], 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluate"):
            shadow, mask, target = (batch[k].to(device) for k in ("shadow", "mask", "target"))
            outputs = {"Shadow input": shadow, "U-Net model": model(model_input(shadow, mask, use_mask)).float()}
            if args.include_traditional:
                outputs["Traditional correction"] = traditional_correction(shadow, mask)
            scores = {m: all_metrics(img, target, mask) for m, img in outputs.items()}
            for i, name in enumerate(batch["name"]):
                for m in methods:
                    per_image.append({"image": name, "method": m, **{k: float(scores[m][k][i]) for k in METRICS}})
                if saved < args.num_visuals:
                    panels = [("Shadow input", shadow[i]), ("Shadow mask", mask[i])]
                    if args.include_traditional:
                        panels.append(("Traditional", outputs["Traditional correction"][i]))
                    panels += [("Model output", outputs["U-Net model"][i]), ("Ground truth", target[i])]
                    save_comparison(out / "comparisons" / f"{name}.png", name, panels)
                    saved += 1

    summary = []
    for m in methods:
        rows = [r for r in per_image if r["method"] == m]
        summary.append({"method": m, **{k: sum(r[k] for r in rows) / len(rows) for k in METRICS}})

    for name, rows in (("metrics_summary.csv", summary), ("per_image_metrics.csv", per_image)):
        with open(out / name, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    print("\nEvaluation summary")
    print("-" * 82)
    print(f"{'Method':<28}{'PSNR':>9}{'SSIM':>11}{'Shadow PSNR':>16}{'Non-shadow':>16}")
    for r in summary:
        print(f"{r['method']:<28}{r['psnr']:>9.3f}{r['ssim']:>11.4f}{r['shadow_psnr']:>16.3f}{r['nonshadow_psnr']:>16.3f}")
    print(f"\nSaved evaluation files to: {out}")


if __name__ == "__main__":
    main()
