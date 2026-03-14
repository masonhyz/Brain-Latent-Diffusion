import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


def _match_size(src: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Center-crop (or pad) src so spatial dims match ref. Shape: [B, C, D, H, W]."""
    sd, sh, sw = src.shape[-3:]
    rd, rh, rw = ref.shape[-3:]

    dd, dh, dw = sd - rd, sh - rh, sw - rw
    d0, h0, w0 = max(dd // 2, 0), max(dh // 2, 0), max(dw // 2, 0)
    d1, h1, w1 = d0 + min(rd, sd), h0 + min(rh, sh), w0 + min(rw, sw)
    src = src[..., d0:d1, h0:h1, w0:w1]

    sd, sh, sw = src.shape[-3:]
    pd, ph, pw = rd - sd, rh - sh, rw - sw
    if pd > 0 or ph > 0 or pw > 0:
        pad = (
            max(pw // 2, 0), max(pw - pw // 2, 0),
            max(ph // 2, 0), max(ph - ph // 2, 0),
            max(pd // 2, 0), max(pd - pd // 2, 0),
        )
        src = F.pad(src, pad)
    return src


class DoubleConv3d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = DoubleConv3d(in_ch, out_ch)
        self.pool = nn.MaxPool3d(2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.conv(x)
        return h, self.pool(h)


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = DoubleConv3d(out_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = _match_size(x, skip)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)
