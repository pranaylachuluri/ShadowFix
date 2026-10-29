"""Train a ShadowFix U-Net on ISTD.

Examples:
    # RGB baseline
    python train.py --data-root data --output-dir runs/baseline --epochs 50 --amp
    # Mask-guided U-Net (best model)
    python train.py --data-root data --output-dir runs/mask_guided --epochs 50 \
        --use-mask-input --shadow-weight 4 --edge-weight 0.05 --amp
    # Residual RGB model, then continue it to 100 epochs
    python train.py ... --output-mode residual --shadow-weight 6 --edge-weight 0.10
    python train.py ... --epochs 100 --resume runs/baseline_final/last.pth
"""
import argparse
import csv
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from shadowfix.checkpoint import save_checkpoint
from shadowfix.data import ISTDDataset, list_triplets, split_train_val
from shadowfix.losses import ShadowRemovalLoss
from shadowfix.metrics import psnr, ssim
from shadowfix.model import build_model, count_parameters, model_input


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", required=True, help="folder containing the ISTD train/ and test/ splits")
    p.add_argument("--output-dir", required=True, help="where checkpoints and the training history are written")
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=50, help="total epochs (also when resuming)")
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--base-channels", type=int, default=32)
    p.add_argument("--output-mode", choices=["direct", "residual"], default="direct")
    p.add_argument("--use-mask-input", action="store_true", help="feed the shadow mask as a 4th channel")
    p.add_argument("--shadow-weight", type=float, default=1.0)
    p.add_argument("--nonshadow-weight", type=float, default=1.0)
    p.add_argument("--edge-weight", type=float, default=0.0)
    p.add_argument("--perceptual-weight", type=float, default=0.0)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--resume", help="checkpoint to continue from (weights, optimizer and history)")
    p.add_argument("--save-every-epoch", action="store_true", help="also keep epoch_XXX.pth for every epoch")
    p.add_argument("--amp", action="store_true", help="mixed precision on CUDA")
    return p.parse_args()


@torch.no_grad()
def validate(model, loader, criterion, device, use_mask):
    model.eval()
    losses, psnrs, ssims = [], [], []
    for batch in loader:
        shadow, mask, target = (batch[k].to(device) for k in ("shadow", "mask", "target"))
        pred = model(model_input(shadow, mask, use_mask)).float()
        loss, _ = criterion(pred, target, mask)
        losses.append(loss.item() * len(shadow))
        psnrs += psnr(pred, target).tolist()
        ssims += ssim(pred, target).tolist()
    n = len(psnrs)
    return sum(losses) / n, sum(psnrs) / n, sum(ssims) / n


def plot_history(history, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, [h["train_loss"] for h in history], label="train")
    axes[0].plot(epochs, [h["val_loss"] for h in history], label="validation")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[1].plot(epochs, [h["val_psnr"] for h in history], color="tab:orange")
    axes[1].set_title("Validation PSNR (dB)")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = args.amp and device.type == "cuda"

    config = {k: getattr(args, k) for k in (
        "image_size", "base_channels", "output_mode", "use_mask_input", "shadow_weight",
        "nonshadow_weight", "edge_weight", "perceptual_weight", "learning_rate", "batch_size", "seed")}

    records = list_triplets(args.data_root, "train")
    train_records, val_records = split_train_val(records, args.val_fraction, args.seed)
    train_loader = DataLoader(ISTDDataset(train_records, args.image_size, augment=True), batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers, pin_memory=device.type == "cuda", drop_last=True)
    val_loader = DataLoader(ISTDDataset(val_records, args.image_size), batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers)

    model = build_model(config).to(device)
    criterion = ShadowRemovalLoss(args.shadow_weight, args.nonshadow_weight, args.edge_weight,
                                  args.perceptual_weight).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=4)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_epoch, best_psnr, history = 1, float("-inf"), []
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        if ckpt.get("optimizer"):
            optimizer.load_state_dict(ckpt["optimizer"])
        if ckpt.get("scheduler"):
            scheduler.load_state_dict(ckpt["scheduler"])
        if ckpt.get("scaler") and use_amp:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        best_psnr = ckpt.get("best_psnr", best_psnr)
        history = ckpt.get("history", [])
        print(f"Resumed from {args.resume} at epoch {ckpt['epoch']}")

    (out / "config.json").write_text(json.dumps(config, indent=2))
    print(f"Device: {device}; AMP: {use_amp}")
    print(f"Train / validation triplets: {len(train_records)} / {len(val_records)}")
    print(f"Model parameters: {count_parameters(model) / 1e6:.2f}M")

    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        model.train()
        running, seen, t0 = 0.0, 0, time.time()
        bar = tqdm(train_loader, desc="Train", leave=False)
        for batch in bar:
            shadow, mask, target = (batch[k].to(device, non_blocking=True) for k in ("shadow", "mask", "target"))
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                pred = model(model_input(shadow, mask, args.use_mask_input))
            loss, _ = criterion(pred.float(), target, mask)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item() * len(shadow)
            seen += len(shadow)
            bar.set_postfix(loss=f"{running / seen:.4f}")

        train_loss = running / max(seen, 1)
        val_loss, val_psnr, val_ssim = validate(model, val_loader, criterion, device, args.use_mask_input)
        scheduler.step(val_psnr)
        lr = optimizer.param_groups[0]["lr"]
        print(f"train_loss={train_loss:.5f} | val_loss={val_loss:.5f} | PSNR={val_psnr:.3f} | "
              f"SSIM={val_ssim:.4f} | lr={lr:.2e}")
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_psnr": val_psnr,
                        "val_ssim": val_ssim, "lr": lr, "seconds": round(time.time() - t0, 1)})

        is_best = val_psnr > best_psnr
        best_psnr = max(best_psnr, val_psnr)
        state = (model, optimizer, scheduler, scaler if use_amp else None, epoch, best_psnr, config, history)
        save_checkpoint(out / "last.pth", *state)
        if args.save_every_epoch:
            save_checkpoint(out / f"epoch_{epoch:03d}.pth", *state)
        if is_best:
            save_checkpoint(out / "best.pth", *state)
            print(f"Saved new best checkpoint: {out / 'best.pth'}")

        with open(out / "training_history.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)
        plot_history(history, out / "training_history.png")

    print(f"\nTraining complete. Outputs are in: {out}")


if __name__ == "__main__":
    main()
