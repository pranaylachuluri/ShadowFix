"""Hand-crafted baseline: match the shadow region's color statistics to the lit region.

For each channel, pixels inside the shadow mask are shifted and scaled so their mean and
standard deviation match the non-shadow pixels, then the boundary is feathered.
"""
import torch
import torch.nn.functional as F


def traditional_correction(shadow, mask, feather=7):
    """shadow (B,3,H,W) and mask (B,1,H,W) in [0, 1] -> corrected (B,3,H,W)."""
    out = shadow.clone()
    for i in range(shadow.shape[0]):
        m = mask[i, 0] > 0.5
        if m.sum() < 10 or (~m).sum() < 10:
            continue
        for ch in range(3):
            x = shadow[i, ch]
            s_mean, s_std = x[m].mean(), x[m].std().clamp_min(1e-3)
            l_mean, l_std = x[~m].mean(), x[~m].std().clamp_min(1e-3)
            out[i, ch][m] = (x[m] - s_mean) * (l_std / s_std) + l_mean
    # Feather the mask so the correction blends into the boundary
    k = feather | 1
    soft = F.avg_pool2d(mask, k, stride=1, padding=k // 2)
    return (soft * out + (1 - soft) * shadow).clamp(0, 1)
