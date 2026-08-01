import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
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


# ─────────────────────────────────────────────────────────────────────────────
# Shared 3-D diffusion primitives
#
# These were previously copy-pasted across the diffusion model files
# (cdm3d / ldm / ldm3d / ldm_7tcdm3d). They are kept here so the four models
# share one definition. All class internals (submodule names) are unchanged
# from the originals, so existing checkpoints load without modification.
# ─────────────────────────────────────────────────────────────────────────────

def group_norm_groups(ch: int, max_g: int = 8) -> int:
    """Largest divisor of ch that is ≤ max_g (a safe GroupNorm group count)."""
    for g in range(min(ch, max_g), 0, -1):
        if ch % g == 0:
            return g
    return 1


def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """t: (B,) int timesteps → (B, dim) sinusoidal embeddings."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, dtype=torch.float32, device=t.device) / (half - 1)
    )
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)   # (B, half)
    return torch.cat([args.sin(), args.cos()], dim=1)     # (B, dim)


def make_beta_schedule(T: int = 1000, schedule: str = "cosine") -> torch.Tensor:
    if schedule == "linear":
        return torch.linspace(1e-4, 0.02, T)
    # cosine (Nichol & Dhariwal 2021)
    s = 0.008
    steps = torch.arange(T + 1, dtype=torch.float64) / T
    alphas_bar = torch.cos((steps + s) / (1 + s) * math.pi / 2) ** 2
    alphas_bar = alphas_bar / alphas_bar[0]
    betas = 1.0 - alphas_bar[1:] / alphas_bar[:-1]
    return betas.clamp(0.0, 0.999).float()


# Pitfall: stride-2 3-D down/upsample — Conv3d/ConvTranspose3d (not the 2-D ops).
def Downsample3D(dim: int) -> nn.Module:
    return nn.Conv3d(dim, dim, kernel_size=4, stride=2, padding=1)


def Upsample3D(dim: int) -> nn.Module:
    return nn.ConvTranspose3d(dim, dim, kernel_size=4, stride=2, padding=1)


class Block3D(nn.Module):
    """3×3×3 conv + GroupNorm + SiLU with optional scale-shift conditioning."""
    def __init__(self, dim: int, dim_out: int, groups: int = 8):
        super().__init__()
        self.proj = nn.Conv3d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(group_norm_groups(dim_out, groups), dim_out)
        self.act  = nn.SiLU()

    def forward(self, x: torch.Tensor, scale_shift=None) -> torch.Tensor:
        x = self.proj(x)
        x = self.norm(x)
        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        return self.act(x)


class ResnetBlock3D(nn.Module):
    """3-D ResBlock with scale-shift time conditioning (7TCDM style)."""
    def __init__(self, dim: int, dim_out: int, *, time_emb_dim: int = None, groups: int = 8):
        super().__init__()
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2)) \
            if time_emb_dim is not None else None
        self.block1   = Block3D(dim,     dim_out, groups)
        self.block2   = Block3D(dim_out, dim_out, groups)
        self.res_conv = nn.Conv3d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor = None) -> torch.Tensor:
        scale_shift = None
        if self.mlp is not None and t_emb is not None:
            # 5-D broadcast: 'b c -> b c 1 1 1' (three spatial singleton dims).
            ss = rearrange(self.mlp(t_emb), 'b c -> b c 1 1 1')
            scale_shift = ss.chunk(2, dim=1)
        h = self.block1(x, scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)


class ResBlock3D(nn.Module):
    """Pre-norm residual block with scale-shift time injection (LDM-style).

    ``gn_max`` sets the GroupNorm group ceiling; image-space (ldm.py) uses 8,
    latent-space (ldm3d.py) uses 32 — preserved per call site.
    """
    def __init__(self, in_ch: int, out_ch: int, t_dim: int, gn_max: int = 8):
        super().__init__()
        self.norm1  = nn.GroupNorm(group_norm_groups(in_ch, gn_max), in_ch)
        self.conv1  = nn.Conv3d(in_ch, out_ch, 3, padding=1)
        self.t_proj = nn.Linear(t_dim, out_ch * 2)   # scale + shift
        self.norm2  = nn.GroupNorm(group_norm_groups(out_ch, gn_max), out_ch)
        self.conv2  = nn.Conv3d(out_ch, out_ch, 3, padding=1)
        self.skip   = nn.Conv3d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.act    = nn.SiLU()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act(self.norm1(x)))
        scale, shift = self.t_proj(t_emb).chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale[:, :, None, None, None]) + shift[:, :, None, None, None]
        h = self.conv2(self.act(h))
        return h + self.skip(x)
