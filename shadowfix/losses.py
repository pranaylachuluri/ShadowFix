"""Region-weighted reconstruction loss with optional edge and perceptual terms."""
import torch
import torch.nn.functional as F
from torch import nn


def _gradients(x):
    dx = x[..., :, 1:] - x[..., :, :-1]
    dy = x[..., 1:, :] - x[..., :-1, :]
    return dx, dy


class VGGPerceptual(nn.Module):
    """L1 distance between VGG16 relu2_2 / relu3_3 features (loads ImageNet weights)."""

    def __init__(self):
        super().__init__()
        from torchvision.models import VGG16_Weights, vgg16

        features = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:16].eval()
        for p in features.parameters():
            p.requires_grad_(False)
        self.features = features
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, pred, target):
        x = (pred - self.mean) / self.std
        y = (target - self.mean) / self.std
        loss = 0.0
        for i, layer in enumerate(self.features):
            x, y = layer(x), layer(y)
            if i in (8, 15):
                loss = loss + F.l1_loss(x, y)
        return loss


class ShadowRemovalLoss(nn.Module):
    """L = w_s * L1(shadow pixels) + w_n * L1(non-shadow pixels) [+ w_e * edge + w_p * perceptual].

    With shadow_weight == nonshadow_weight == 1 this is plain L1 over the image.
    """

    def __init__(self, shadow_weight=1.0, nonshadow_weight=1.0, edge_weight=0.0, perceptual_weight=0.0):
        super().__init__()
        self.shadow_weight = shadow_weight
        self.nonshadow_weight = nonshadow_weight
        self.edge_weight = edge_weight
        self.perceptual_weight = perceptual_weight
        self.perceptual = VGGPerceptual() if perceptual_weight > 0 else None

    def forward(self, pred, target, mask):
        err = (pred - target).abs()
        details = {}
        if self.shadow_weight == self.nonshadow_weight:
            rec = self.shadow_weight * err.mean()
        else:
            m = mask.expand_as(err)
            shadow = (err * m).sum() / m.sum().clamp_min(1.0)
            nonshadow = (err * (1 - m)).sum() / (1 - m).sum().clamp_min(1.0)
            rec = self.shadow_weight * shadow + self.nonshadow_weight * nonshadow
        details["reconstruction"] = rec
        total = rec

        if self.edge_weight > 0:
            pdx, pdy = _gradients(pred)
            tdx, tdy = _gradients(target)
            edge = (pdx - tdx).abs().mean() + (pdy - tdy).abs().mean()
            details["edge"] = edge
            total = total + self.edge_weight * edge

        if self.perceptual is not None:
            perc = self.perceptual(pred, target)
            details["perceptual"] = perc
            total = total + self.perceptual_weight * perc

        details["total"] = total
        return total, {k: float(v.detach()) for k, v in details.items()}
