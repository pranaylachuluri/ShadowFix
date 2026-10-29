"""PSNR / SSIM, overall and inside / outside the shadow mask. Images are in [0, 1]."""
import torch
import torch.nn.functional as F


def psnr(pred, target, mask=None, eps=1e-10):
    """Per-image PSNR (dB). With a mask, the MSE is taken over masked pixels only."""
    se = (pred - target) ** 2
    if mask is None:
        mse = se.flatten(1).mean(1)
    else:
        m = mask.expand_as(se)
        mse = (se * m).flatten(1).sum(1) / m.flatten(1).sum(1).clamp_min(1.0)
    return 10 * torch.log10(1.0 / (mse + eps))


def _gaussian_window(size=11, sigma=1.5, channels=3, device="cpu"):
    coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window = (g[:, None] * g[None, :]).expand(channels, 1, size, size).contiguous()
    return window


def ssim(pred, target, size=11, sigma=1.5):
    """Per-image SSIM averaged over channels (Wang et al. 2004, Gaussian window)."""
    c = pred.shape[1]
    w = _gaussian_window(size, sigma, c, pred.device).to(pred.dtype)
    pad = size // 2
    mu_x = F.conv2d(pred, w, padding=pad, groups=c)
    mu_y = F.conv2d(target, w, padding=pad, groups=c)
    sxx = F.conv2d(pred * pred, w, padding=pad, groups=c) - mu_x ** 2
    syy = F.conv2d(target * target, w, padding=pad, groups=c) - mu_y ** 2
    sxy = F.conv2d(pred * target, w, padding=pad, groups=c) - mu_x * mu_y
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    s = ((2 * mu_x * mu_y + c1) * (2 * sxy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (sxx + syy + c2))
    return s.flatten(1).mean(1)


def all_metrics(pred, target, mask):
    """Dict of per-image tensors: psnr, ssim, shadow_psnr, nonshadow_psnr."""
    return {
        "psnr": psnr(pred, target),
        "ssim": ssim(pred, target),
        "shadow_psnr": psnr(pred, target, mask),
        "nonshadow_psnr": psnr(pred, target, 1 - mask),
    }
