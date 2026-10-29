"""Save / load checkpoints together with the model configuration."""
from pathlib import Path

import torch

from .model import build_model


def save_checkpoint(path, model, optimizer, scheduler, scaler, epoch, best_psnr, config, history):
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer else None,
            "scheduler": scheduler.state_dict() if scheduler else None,
            "scaler": scaler.state_dict() if scaler else None,
            "epoch": epoch,
            "best_psnr": best_psnr,
            "config": config,
            "history": history,
        },
        Path(path),
    )


def load_model(checkpoint_path, device="cpu"):
    """Rebuild the model from a checkpoint. Returns (model, config)."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt["config"]
    model = build_model(config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, config
