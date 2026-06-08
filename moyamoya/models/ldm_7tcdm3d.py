"""
3-D adaptation of the 7TCDM diffusion U-Net used as the denoiser inside the
same Latent Diffusion Model framework as ldm3d.py (shared AutoencoderKL3D +
PairedLatentDiffusion wrapper).

Key differences from ldm3d.py's LatentDiffusionUNet3D:
  - 3 ResBlocks per encoder level (vs 2) + 1 extra after each Downsample
  - 4 mid ResBlocks
  - Initial depthwise conv output is concatenated back at the final layer
  - Scale-shift time conditioning (7TCDM style)

Key changes from 7TCDM's 2-D Unet3D:
  - All Conv2d -> Conv3d; Downsample / Upsample become 3-D
  - Time-embedding rearranged to 'b c 1 1 1' (5-D)
  - SpatialLinearAttention / CrossAttention removed (7TCDM uses them off in
    practice; replaced here by the third ResBlock per level)
  - _match_size guards against 1-voxel mismatches on odd spatial dims
"""

import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .ldm3d import (
    AutoencoderKL3D,
    PairedLatentDiffusion,
    build_autoencoder,
    sinusoidal_embedding,
)
from ..modules import _match_size


# ─────────────────────────────────────────────────────────────────────────────
# GroupNorm helper
# ─────────────────────────────────────────────────────────────────────────────

def _ng(ch: int, max_g: int = 8) -> int:
    """Largest divisor of ch that is <= max_g (for GroupNorm)."""
    for g in range(min(ch, max_g), 0, -1):
        if ch % g == 0:
            return g
    return 1


# ─────────────────────────────────────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────────────────────────────────────

def Downsample3D(dim: int) -> nn.Module:
    return nn.Conv3d(dim, dim, kernel_size=4, stride=2, padding=1)


def Upsample3D(dim: int) -> nn.Module:
    return nn.ConvTranspose3d(dim, dim, kernel_size=4, stride=2, padding=1)


class Block3D(nn.Module):
    def __init__(self, dim: int, dim_out: int, groups: int = 8):
        super().__init__()
        self.proj = nn.Conv3d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(_ng(dim_out, groups), dim_out)
        self.act  = nn.SiLU()

    def forward(self, x: torch.Tensor, scale_shift=None) -> torch.Tensor:
        x = self.proj(x)
        x = self.norm(x)
        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        return self.act(x)


class ResnetBlock3D(nn.Module):
    def __init__(self, dim: int, dim_out: int, *, time_emb_dim: int = None, groups: int = 8):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2))
            if time_emb_dim is not None else None
        )
        self.block1   = Block3D(dim,     dim_out, groups)
        self.block2   = Block3D(dim_out, dim_out, groups)
        self.res_conv = nn.Conv3d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor = None) -> torch.Tensor:
        scale_shift = None
        if self.mlp is not None:
            t = self.mlp(t_emb)
            t = rearrange(t, 'b c -> b c 1 1 1')
            scale_shift = t.chunk(2, dim=1)
        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)


# ─────────────────────────────────────────────────────────────────────────────
# 3-D U-Net denoiser ported from 7TCDM
# ─────────────────────────────────────────────────────────────────────────────

class Unet3DFrom7TCDM(nn.Module):
    """
    3-D ResNet U-Net denoiser, architecture ported from 7TCDM's Unet3D.

    in_channels  = 2 * embed_dim  (z_t cat z_pre, conditioning via concat)
    out_channels = embed_dim      (predicted noise)

    Per encoder level: ResBlock x2 → save skip → Downsample → ResBlock x1
    Per decoder level: cat skip → ResBlock x1 → Upsample → ResBlock x2
    Mid: 4 ResBlocks
    Final: cat with init_conv output → ResBlock → depthwise → 1x1
    """

    def __init__(
        self,
        dim:              int   = 32,
        in_channels:      int   = 8,
        out_channels:     int   = 4,
        dim_mults:        tuple = (1, 2, 4, 8),
        init_kernel_size: int   = 3,
        resnet_groups:    int   = 8,
    ):
        super().__init__()
        init_dim  = dim
        time_dim  = dim * 4
        pad       = init_kernel_size // 2

        # ── initial conv (1×1 projection + depthwise) ───────────────────────
        self.init_conv = nn.Sequential(
            nn.Conv3d(in_channels, init_dim, 1),
            nn.Conv3d(init_dim, init_dim, init_kernel_size,
                      groups=init_dim, padding=pad),
        )

        # ── time embedding ───────────────────────────────────────────────────
        self.time_dim = dim
        self.time_mlp = nn.Sequential(
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        dims        = [init_dim, *[dim * m for m in dim_mults]]
        in_out      = list(zip(dims[:-1], dims[1:]))
        num_levels  = len(in_out)
        block       = partial(ResnetBlock3D, time_emb_dim=time_dim, groups=resnet_groups)

        # ── encoder ──────────────────────────────────────────────────────────
        self.downs = nn.ModuleList()
        for ind, (d_in, d_out) in enumerate(in_out):
            is_last = ind >= num_levels - 1
            self.downs.append(nn.ModuleList([
                block(d_in,  d_out),           # b1: change channels
                block(d_out, d_out),            # b2: process
                Downsample3D(d_out) if not is_last else nn.Identity(),
                block(d_out, d_out),            # b3: after downsample
            ]))

        # ── bottleneck ───────────────────────────────────────────────────────
        mid_dim  = dims[-1]
        self.mid = nn.ModuleList([block(mid_dim, mid_dim) for _ in range(4)])

        # ── decoder ──────────────────────────────────────────────────────────
        self.ups = nn.ModuleList()
        for ind, (d_in, d_out) in enumerate(reversed(in_out)):
            is_last = ind >= num_levels - 1
            self.ups.append(nn.ModuleList([
                block(d_out * 2, d_in),         # b1: merge skip + process
                Upsample3D(d_in) if not is_last else nn.Identity(),
                block(d_in, d_in),              # b2: after upsample
                block(d_in, d_in),              # b3: extra
            ]))

        # ── output head ──────────────────────────────────────────────────────
        # init_conv output is concatenated here (initial shortcut)
        self.final_block = ResnetBlock3D(init_dim * 2, init_dim, groups=resnet_groups)
        self.final_dw    = nn.Conv3d(init_dim, init_dim, init_kernel_size,
                                     groups=init_dim, padding=pad)
        self.final_out   = nn.Conv3d(init_dim, out_channels, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 2*embed_dim, D', H', W')  — cat([z_t, z_pre]) in latent space
        t: (B,) integer timesteps
        Returns: (B, embed_dim, D', H', W') — predicted noise
        """
        t_emb = sinusoidal_embedding(t, self.time_dim)
        t_emb = self.time_mlp(t_emb)

        x = self.init_conv(x)
        r = x.clone()

        skips = []
        for b1, b2, down, b3 in self.downs:
            x = b1(x, t_emb)
            x = b2(x, t_emb)
            skips.append(x)         # save skip before downsample
            x = down(x)
            x = b3(x, t_emb)

        for m in self.mid:
            x = m(x, t_emb)

        for b1, up, b2, b3 in self.ups:
            skip = skips.pop()
            x    = _match_size(x, skip)
            x    = torch.cat([x, skip], dim=1)
            x    = b1(x, t_emb)
            x    = up(x)
            x    = b2(x, t_emb)
            x    = b3(x, t_emb)

        x = _match_size(x, r)
        x = torch.cat([x, r], dim=1)
        x = self.final_block(x)
        x = self.final_dw(x)
        return self.final_out(x)


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def build_paired_latent_diffusion_7tcdm(
    # autoencoder (identical to ldm3d)
    z_channels:       int   = 4,
    embed_dim:        int   = 4,
    ae_ch:            int   = 64,
    ae_ch_mult:       tuple = (1, 2, 4),
    ae_res_blocks:    int   = 2,
    ae_resolution:    int   = 64,
    kl_weight:        float = 1e-6,
    # 7TCDM-style 3-D denoiser
    diff_dim:         int   = 32,
    diff_dim_mults:   tuple = (1, 2, 4, 8),
    init_kernel_size: int   = 3,
    resnet_groups:    int   = 8,
    # diffusion schedule
    T:                int   = 1000,
    schedule:         str   = "cosine",
) -> PairedLatentDiffusion:
    ae = build_autoencoder(
        z_channels=z_channels,
        embed_dim=embed_dim,
        ch=ae_ch,
        ch_mult=ae_ch_mult,
        num_res_blocks=ae_res_blocks,
        resolution=ae_resolution,
        kl_weight=kl_weight,
    )
    denoiser = Unet3DFrom7TCDM(
        dim=diff_dim,
        in_channels=2 * embed_dim,
        out_channels=embed_dim,
        dim_mults=diff_dim_mults,
        init_kernel_size=init_kernel_size,
        resnet_groups=resnet_groups,
    )
    return PairedLatentDiffusion(ae=ae, denoiser=denoiser, T=T, schedule=schedule)
