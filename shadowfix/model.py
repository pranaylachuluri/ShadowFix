"""U-Net used for every ShadowFix variant.

With base_channels=32 the network has about 7.76M parameters. Variants:
  * RGB (direct):  3-channel shadow image in, restored image out
  * RGB residual:  the network predicts a correction that is added to the input
  * mask-guided:   the shadow mask is concatenated as a 4th input channel
"""
import torch
from torch import nn


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))

    def forward(self, x):
        return self.block(x)


class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        # Pad when the input size is not divisible by 16
        dy, dx = skip.shape[-2] - x.shape[-2], skip.shape[-1] - x.shape[-1]
        if dy or dx:
            x = nn.functional.pad(x, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        return self.conv(torch.cat([skip, x], dim=1))


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_channels=32, output_mode="direct"):
        super().__init__()
        if output_mode not in ("direct", "residual"):
            raise ValueError("output_mode must be 'direct' or 'residual'")
        self.output_mode = output_mode
        c = base_channels
        self.inc = DoubleConv(in_channels, c)
        self.down1 = Down(c, c * 2)
        self.down2 = Down(c * 2, c * 4)
        self.down3 = Down(c * 4, c * 8)
        self.down4 = Down(c * 8, c * 16)
        self.up1 = Up(c * 16, c * 8)
        self.up2 = Up(c * 8, c * 4)
        self.up3 = Up(c * 4, c * 2)
        self.up4 = Up(c * 2, c)
        self.outc = nn.Conv2d(c, out_channels, 1)

    def forward(self, x):
        rgb = x[:, :3]
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        y = self.up1(x5, x4)
        y = self.up2(y, x3)
        y = self.up3(y, x2)
        y = self.up4(y, x1)
        y = self.outc(y)
        if self.output_mode == "residual":
            return (rgb + torch.tanh(y)).clamp(0, 1)
        return torch.sigmoid(y)


def build_model(config):
    return UNet(
        in_channels=4 if config.get("use_mask_input") else 3,
        out_channels=3,
        base_channels=config.get("base_channels", 32),
        output_mode=config.get("output_mode", "direct"),
    )


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_input(shadow, mask, use_mask_input):
    return torch.cat([shadow, mask], dim=1) if use_mask_input else shadow
