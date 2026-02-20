import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# 3D UNet (small, solid baseline)
# -------------------------
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

    def forward(self, x):
        return self.net(x)


def _match_size(src: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """
    Center-crop (or pad) src so that src spatial dims match ref spatial dims.
    src/ref shape: [B, C, D, H, W]
    """
    sd, sh, sw = src.shape[-3:]
    rd, rh, rw = ref.shape[-3:]

    # If src is larger: crop
    dd = sd - rd
    dh = sh - rh
    dw = sw - rw

    d0 = max(dd // 2, 0)
    h0 = max(dh // 2, 0)
    w0 = max(dw // 2, 0)

    d1 = d0 + min(rd, sd)
    h1 = h0 + min(rh, sh)
    w1 = w0 + min(rw, sw)

    src = src[..., d0:d1, h0:h1, w0:w1]

    # If src is smaller: pad
    sd, sh, sw = src.shape[-3:]
    pd = rd - sd
    ph = rh - sh
    pw = rw - sw
    if pd > 0 or ph > 0 or pw > 0:
        pad = (
            max(pw // 2, 0), max(pw - pw // 2, 0),
            max(ph // 2, 0), max(ph - ph // 2, 0),
            max(pd // 2, 0), max(pd - pd // 2, 0),
        )
        src = F.pad(src, pad)

    return src


class UNet3D(nn.Module):
    """
    Basic 3D UNet.
    For fMRI-as-channels, set in_channels = T (timepoints) from your data.
    """
    def __init__(self, in_channels: int, out_channels: int, base: int = 32):
        super().__init__()
        self.enc1 = DoubleConv3d(in_channels, base)
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = DoubleConv3d(base, base * 2)
        self.pool2 = nn.MaxPool3d(2)
        self.enc3 = DoubleConv3d(base * 2, base * 4)
        self.pool3 = nn.MaxPool3d(2)

        self.bottleneck = DoubleConv3d(base * 4, base * 8)

        self.up3 = nn.ConvTranspose3d(base * 8, base * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv3d(base * 8, base * 4)
        self.up2 = nn.ConvTranspose3d(base * 4, base * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv3d(base * 4, base * 2)
        self.up1 = nn.ConvTranspose3d(base * 2, base, kernel_size=2, stride=2)
        self.dec1 = DoubleConv3d(base * 2, base)

        self.out = nn.Conv3d(base, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b = self.bottleneck(self.pool3(e3))

        d3 = self.up3(b)
        d3 = _match_size(d3, e3)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = _match_size(d2, e2)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = _match_size(d1, e1)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.out(d1)
