"""
CDM3D — 3D port of 7TCDM's Unet3D + GaussianDiffusion.

Key design choices carried from 7TCDM (single stage, image space):
  - x₀-prediction (denoiser outputs clean image, not noise)
  - Continuous timesteps t ∈ [0,1] with ᾱ(t) = cos²((t+s)/(1+s)·π/2)
  - Loss = MSE(x̂₀, x₀) + λ_ssim · (1 − SSIM3D(x̂₀, x₀))
  - Conditioning by concatenation: input = cat([x_pre, x_t], dim=1)
  - EMA model weights used for inference

Critical 2D → 3D changes and pitfalls:

  1. Conv2d/ConvTranspose2d → Conv3d/ConvTranspose3d everywhere.
     Missed instances produce silent wrong shapes that only crash at matmul.

  2. Time embedding broadcast: 'b c -> b c 1 1' becomes 'b c -> b c 1 1 1'.
     One missing dim silently broadcasts incorrectly across batch items.

  3. alpha_cum broadcast: [:, None, None, None] → [:, None, None, None, None].
     Same issue — one missing dim causes incorrect noising of entire batches.

  4. nonzero_mask in p_sample: reshape(b,1,1,1) → reshape(b,1,1,1,1).

  5. Attention is dropped. 7TCDM disables attention via use_sparse_linear_attn=False.
     In 3D, flattening D×H×W for attention is infeasible (91*109*91 = 904k tokens).
     Slots remain as nn.Identity() to keep the architecture comment accurate.

  6. GroupNorm requires ch % groups == 0. 3D U-Nets change channel counts at
     every level; use _ng() to pick the largest valid group count.

  7. Odd spatial dims: stride-2 conv on size 91 → 45, and ConvTranspose on
     45 → 90 (not 91). Guard every skip-cat with _match_size().

  8. Depthwise (Block2d → BlockDW3D): groups=in_channels must equal in_channels,
     which is always true for a depthwise conv. Keep groups=dim in Conv3d.

  9. init_kernel_size=7 becomes 7×7×7. Fine at full resolution (depthwise, few
     params), but reduce to 3 on GPUs with <12 GB for the full (91,109,91) volume.

  10. SSIM on z-score data: avg_pool3d Gaussian approximation + default C1/C2
      constants are tuned for [0,1] data. For z-score (-3…3) the loss is still
      a useful relative gradient; absolute SSIM values may differ from standard.
"""

import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ng(ch: int, max_g: int = 8) -> int:
    """Largest divisor of ch that is ≤ max_g (safe GroupNorm groups)."""
    for g in range(min(ch, max_g), 0, -1):
        if ch % g == 0:
            return g
    return 1


def _match_size(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Trim or pad x to match ref's spatial dims (handles stride-2 off-by-one)."""
    if x.shape[2:] == ref.shape[2:]:
        return x
    return F.interpolate(x, size=ref.shape[2:], mode="trilinear", align_corners=False)


# ─────────────────────────────────────────────────────────────────────────────
# Continuous cosine schedule (from math_utils.py)
# ─────────────────────────────────────────────────────────────────────────────

def get_alpha_cum(t: torch.Tensor) -> torch.Tensor:
    """ᾱ(t) for t ∈ [0,1]. t=0 → ᾱ≈1 (clean), t=1 → ᾱ≈0 (pure noise)."""
    return torch.cos((t + 0.008) / 1.008 * math.pi / 2).pow(2).clamp(1e-4, 1 - 1e-4)


def get_z_t(x0: torch.Tensor, t: torch.Tensor):
    """Forward diffusion: z_t = √ᾱ(t)·x₀ + √(1−ᾱ(t))·ε.  t ∈ [0,1]."""
    # Pitfall #3: must be 5-D for (B, C, D, H, W) — was 4-D in original 2D code.
    ac = get_alpha_cum(t)[:, None, None, None, None]
    eps = torch.randn_like(x0)
    return ac.sqrt() * x0 + (1 - ac).sqrt() * eps, eps


def _get_eps_from_x0(x0: torch.Tensor, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    ac = get_alpha_cum(t)[:, None, None, None, None]
    return (xt - ac.sqrt() * x0) / (1 - ac).sqrt().clamp_min(1e-6)


def ddim_step(x0_pred: torch.Tensor, xt: torch.Tensor,
              t_cur: torch.Tensor, t_prev: torch.Tensor) -> torch.Tensor:
    """One deterministic DDIM step (η=0), x₀-parameterisation."""
    eps  = _get_eps_from_x0(x0_pred, xt, t_cur)
    ac   = get_alpha_cum(t_prev)[:, None, None, None, None]
    return ac.sqrt() * x0_pred + (1 - ac).sqrt() * eps


# ─────────────────────────────────────────────────────────────────────────────
# 3-D SSIM loss (differentiable, avg_pool3d Gaussian approximation)
# ─────────────────────────────────────────────────────────────────────────────

def ssim3d_loss(x: torch.Tensor, y: torch.Tensor, window_size: int = 7) -> torch.Tensor:
    """Returns 1 − SSIM(x, y) as a differentiable scalar.
    Uses avg_pool3d as a fast Gaussian approximation.
    Note: C1/C2 constants assume [0,1] data; for z-score input the loss is still
    a valid relative gradient but absolute SSIM values won't match standard metrics.
    """
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    pad = window_size // 2

    def pool(t: torch.Tensor) -> torch.Tensor:
        return F.avg_pool3d(t, window_size, stride=1, padding=pad)

    mu_x, mu_y   = pool(x), pool(y)
    mu2_x, mu2_y = mu_x * mu_x, mu_y * mu_y
    mu_xy        = mu_x * mu_y

    sig2_x = pool(x * x) - mu2_x
    sig2_y = pool(y * y) - mu2_y
    sig_xy = pool(x * y) - mu_xy

    ssim_map = ((2 * mu_xy + C1) * (2 * sig_xy + C2)) / \
               ((mu2_x + mu2_y + C1) * (sig2_x + sig2_y + C2))
    return 1.0 - ssim_map.mean()


# ─────────────────────────────────────────────────────────────────────────────
# Building blocks (all 3-D)
# ─────────────────────────────────────────────────────────────────────────────

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        emb  = math.log(10000) / (half - 1)
        emb  = torch.exp(torch.arange(half, device=x.device) * -emb)
        emb  = x[:, None] * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class Block3D(nn.Module):
    """Standard 3×3×3 conv + GroupNorm + SiLU with optional scale-shift conditioning."""
    def __init__(self, dim: int, dim_out: int, groups: int = 8):
        super().__init__()
        self.proj = nn.Conv3d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(_ng(dim_out, groups), dim_out)
        self.act  = nn.SiLU()

    def forward(self, x: torch.Tensor, scale_shift=None) -> torch.Tensor:
        x = self.norm(self.proj(x))
        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        return self.act(x)


class ResnetBlock3D(nn.Module):
    """Standard 3-D ResBlock with scale-shift time conditioning."""
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
            # Pitfall #2: must rearrange to 'b c 1 1 1' for 5-D (was 'b c 1 1' in 2D).
            ss = rearrange(self.mlp(t_emb), 'b c -> b c 1 1 1')
            scale_shift = ss.chunk(2, dim=1)
        h = self.block1(x, scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)


class BlockDW3D(nn.Module):
    """Depthwise-separable 3-D conv block (ported from Block2d in 7TCDM)."""
    def __init__(self, dim: int, dim_out: int, groups: int = 8):
        super().__init__()
        # Pitfall #8: groups must equal in_channels for depthwise conv.
        self.proj = nn.Sequential(
            nn.Conv3d(dim, dim, 3, padding=1, groups=dim),  # depthwise
            nn.Conv3d(dim, dim_out, 1),                      # pointwise
        )
        self.norm = nn.GroupNorm(_ng(dim_out, groups), dim_out)
        self.act  = nn.SiLU()

    def forward(self, x: torch.Tensor, scale_shift=None) -> torch.Tensor:
        x = self.norm(self.proj(x))
        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        return self.act(x)


class ResnetBlockDW3D(nn.Module):
    """Depthwise 3-D ResBlock — cheaper per-channel than ResnetBlock3D."""
    def __init__(self, dim: int, dim_out: int, *, time_emb_dim: int = None, groups: int = 8):
        super().__init__()
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2)) \
            if time_emb_dim is not None else None
        self.block1   = BlockDW3D(dim,     dim_out, groups)
        self.block2   = BlockDW3D(dim_out, dim_out, groups)
        self.res_conv = nn.Conv3d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor = None) -> torch.Tensor:
        scale_shift = None
        if self.mlp is not None and t_emb is not None:
            ss = rearrange(self.mlp(t_emb), 'b c -> b c 1 1 1')
            scale_shift = ss.chunk(2, dim=1)
        h = self.block1(x, scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)


# Pitfall #1: stride-2 3-D down/upsample — was Conv2d/ConvTranspose2d in original.
def Downsample3D(dim: int) -> nn.Module:
    return nn.Conv3d(dim, dim, kernel_size=4, stride=2, padding=1)


def Upsample3D(dim: int) -> nn.Module:
    return nn.ConvTranspose3d(dim, dim, kernel_size=4, stride=2, padding=1)


# ─────────────────────────────────────────────────────────────────────────────
# 3-D U-Net denoiser (faithful 3-D port of 7TCDM's Unet3D)
# ─────────────────────────────────────────────────────────────────────────────

class Unet3D_CDM(nn.Module):
    """
    3-D conditional denoiser, faithful port of 7TCDM's Unet3D.

    Encoder per level: ResnetBlock3D×2 → save skip → Downsample3D → ResnetBlockDW3D
    Bottleneck:        ResnetBlock3D → ResnetBlockDW3D×2 → ResnetBlock3D
    Decoder per level: cat(skip) → ResnetBlock3D → Upsample3D → ResnetBlock3D → ResnetBlockDW3D
    Output head:       cat(init_conv) → ResnetBlock3D → DW conv → 1×1×1

    Attention slots (SpatialLinearAttention, CrossAttention) are nn.Identity() —
    they were disabled in the original training.py; in 3-D they are infeasible anyway
    (91×109×91 = 904k tokens per sequence).
    """

    def __init__(
        self,
        dim:              int   = 32,
        dim_mults:        tuple = (1, 2, 4, 8),
        in_channels:      int   = 2,   # cat([x_pre, x_t])
        out_channels:     int   = 1,   # predicted x₀
        init_kernel_size: int   = 7,
        resnet_groups:    int   = 8,
    ):
        super().__init__()

        init_dim  = dim
        init_pad  = init_kernel_size // 2
        time_dim  = dim * 4

        # Pitfall #1: was nn.Conv2d in original.
        self.init_conv0 = nn.Sequential(
            nn.Conv3d(in_channels, init_dim, 1),
            nn.Conv3d(init_dim, init_dim, init_kernel_size,
                      groups=init_dim, padding=init_pad),
        )

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        dims       = [init_dim, *[dim * m for m in dim_mults]]
        in_out     = list(zip(dims[:-1], dims[1:]))
        num_levels = len(in_out)

        block    = partial(ResnetBlock3D,   time_emb_dim=time_dim, groups=resnet_groups)
        block_dw = partial(ResnetBlockDW3D, time_emb_dim=time_dim, groups=resnet_groups)

        # ── encoder ──────────────────────────────────────────────────────────
        self.downs = nn.ModuleList()
        for ind, (d_in, d_out) in enumerate(in_out):
            is_last = ind >= num_levels - 1
            self.downs.append(nn.ModuleList([
                block(d_in, d_out),
                block(d_out, d_out),
                nn.Identity(),                              # was SpatialLinearAttention
                nn.Identity(),                              # was CrossAttention
                block_dw(d_out, d_out),                    # was ResnetBlock2d (temporal conv)
                Downsample3D(d_out) if not is_last else nn.Identity(),
            ]))

        # ── bottleneck ────────────────────────────────────────────────────────
        mid_dim = dims[-1]
        self.mid_block1      = block(mid_dim, mid_dim)
        self.mid_dw1         = block_dw(mid_dim, mid_dim)  # was mid_spatial_attn (ResnetBlock2d)
        self.mid_dw2         = block_dw(mid_dim, mid_dim)  # was mid_temporal_conv
        self.mid_block2      = block(mid_dim, mid_dim)

        # ── decoder ───────────────────────────────────────────────────────────
        self.ups = nn.ModuleList()
        for ind, (d_in, d_out) in enumerate(reversed(in_out)):
            is_last = ind >= num_levels - 1
            self.ups.append(nn.ModuleList([
                block(d_out * 2, d_in),                    # merges skip: d_out×2 → d_in
                block(d_in, d_in),
                nn.Identity(),
                nn.Identity(),
                block_dw(d_in, d_in),
                Upsample3D(d_in) if not is_last else nn.Identity(),
            ]))

        # ── output head ───────────────────────────────────────────────────────
        block_plain = partial(ResnetBlock3D, groups=resnet_groups)
        self.final_conv0 = nn.Sequential(
            block_plain(init_dim * 2, init_dim),            # merges init_conv shortcut
            nn.Conv3d(init_dim, init_dim, init_kernel_size,
                      groups=init_dim, padding=init_pad),   # depthwise
            nn.Conv3d(init_dim, out_channels, 1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 2, D, H, W)  — cat([x_pre, x_t])
        t: (B,) float        — time index (0 … T as float, fed to sinusoidal emb)
        Returns: (B, 1, D, H, W) — predicted x₀
        """
        x    = self.init_conv0(x)
        r    = x.clone()              # save for final skip
        t_emb = self.time_mlp(t)

        skips = []
        for b1, b2, _, __, dw, down in self.downs:
            x = b1(x, t_emb)
            x = b2(x, t_emb)
            skips.append(x)           # save at pre-downsample resolution
            x = down(x)
            x = dw(x, t_emb)

        x = self.mid_block1(x, t_emb)
        x = self.mid_dw1(x, t_emb)
        x = self.mid_dw2(x, t_emb)
        x = self.mid_block2(x, t_emb)

        for b1, b2, _, __, dw, up in self.ups:
            skip = skips.pop()
            # Pitfall #7: stride-2 on odd dims causes ±1 mismatch; guard here.
            x = _match_size(x, skip)
            x = torch.cat([x, skip], dim=1)
            x = b1(x, t_emb)         # d_out×2 → d_in
            x = up(x)                 # upsample at d_in
            x = b2(x, t_emb)
            x = dw(x, t_emb)

        x = _match_size(x, r)
        x = torch.cat([x, r], dim=1)
        return self.final_conv0(x)


# ─────────────────────────────────────────────────────────────────────────────
# EMA
# ─────────────────────────────────────────────────────────────────────────────

class EMA:
    def __init__(self, beta: float = 0.999):
        self.beta = beta

    def update(self, ema_model: nn.Module, model: nn.Module) -> None:
        for ep, p in zip(ema_model.parameters(), model.parameters()):
            ep.data.mul_(self.beta).add_(p.data, alpha=1 - self.beta)


# ─────────────────────────────────────────────────────────────────────────────
# Gaussian diffusion wrapper (x₀-prediction, continuous timesteps)
# ─────────────────────────────────────────────────────────────────────────────

class GaussianDiffusion3D(nn.Module):
    """
    x₀-prediction diffusion in image space, mirroring 7TCDM's GaussianDiffusion.

    Training: sample t ∈ [0,1] uniformly, noise x_post to x_t, predict x_post.
    Loss:     MSE + λ_ssim·(1−SSIM3D).
    Inference: DDIM sampling with CFG (guidance_scale > 1 requires cfg_drop_prob > 0).
    """

    def __init__(
        self,
        denoise_fn:    nn.Module,
        timesteps:     int   = 1000,
        ddim_steps:    int   = 50,
        ssim_weight:   float = 0.5,
        cfg_drop_prob: float = 0.1,
    ):
        super().__init__()
        self.denoise_fn    = denoise_fn
        self.T             = timesteps
        self.ddim_steps    = ddim_steps
        self.ssim_weight   = ssim_weight
        self.cfg_drop_prob = cfg_drop_prob

    def p_losses(self, x_post: torch.Tensor, x_pre: torch.Tensor) -> torch.Tensor:
        B, device = x_post.size(0), x_post.device

        # Continuous t ∈ [0,1]; denoiser receives t·T as its "time index" for
        # the sinusoidal embedding — same range as integer timesteps.
        t = torch.rand(B, device=device)

        x_t, _ = get_z_t(x_post, t)

        # CFG: zero out conditioning for a random fraction of the batch.
        x_pre_in = x_pre.clone()
        if self.cfg_drop_prob > 0.0:
            drop = torch.rand(B, device=device) < self.cfg_drop_prob
            x_pre_in[drop] = 0.0

        x_in    = torch.cat([x_pre_in, x_t], dim=1)  # (B, 2, D, H, W)
        x0_pred = self.denoise_fn(x_in, t * self.T)

        loss = F.mse_loss(x0_pred, x_post)
        if self.ssim_weight > 0.0:
            loss = loss + self.ssim_weight * ssim3d_loss(x0_pred, x_post)
        return loss

    def forward(self, x_post: torch.Tensor, x_pre: torch.Tensor) -> torch.Tensor:
        return self.p_losses(x_post, x_pre)

    @torch.no_grad()
    def sample(
        self,
        x_pre:          torch.Tensor,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        """
        DDIM sampling (η=0, x₀-parameterisation) with optional CFG.
        Returns the predicted x_post in the same space as x_pre.

        The iterative DDIM loop divides by √(1−ᾱ) (≈0.01 at late steps) and
        accumulates over many steps; under fp16 autocast these intermediates
        overflow to inf → nan.  We therefore force fp32 here and clamp/sanitise
        every step so a single bad value cannot poison the whole trajectory.
        """
        # Pitfall: never run the DDIM loop under fp16 autocast — it is unstable.
        with torch.cuda.amp.autocast(enabled=False):
            x_pre = x_pre.float()
            B, _, D, H, W = x_pre.shape
            device = x_pre.device

            # t_seq[0]=1.0 (pure noise) … t_seq[-1]=0.0 (clean).
            t_seq = torch.linspace(1.0, 0.0, self.ddim_steps + 1, device=device)

            x = torch.randn(B, 1, D, H, W, device=device)

            for i in range(self.ddim_steps):
                t_cur  = t_seq[i].expand(B)
                t_prev = t_seq[i + 1].expand(B)

                x_in    = torch.cat([x_pre, x], dim=1)
                x0_pred = self.denoise_fn(x_in, t_cur * self.T).float()

                if guidance_scale != 1.0:
                    x0_unc  = self.denoise_fn(
                        torch.cat([torch.zeros_like(x_pre), x], dim=1), t_cur * self.T
                    ).float()
                    x0_pred = x0_unc + guidance_scale * (x0_pred - x0_unc)

                x0_pred = torch.nan_to_num(
                    x0_pred, nan=0.0, posinf=5.0, neginf=-5.0
                ).clamp(-5.0, 5.0)

                # On the last step t_prev=0 → ddim_step returns ≈x0_pred directly
                # (ᾱ(0)≈1, so √ᾱ·x0 + √(1-ᾱ)·eps ≈ x0).  No special casing needed.
                x = ddim_step(x0_pred, x, t_cur, t_prev)
                x = torch.nan_to_num(x, nan=0.0, posinf=5.0, neginf=-5.0)

            return x


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def build_cdm3d(
    dim:              int   = 32,
    dim_mults:        tuple = (1, 2, 4, 8),
    init_kernel_size: int   = 7,
    resnet_groups:    int   = 8,
    timesteps:        int   = 1000,
    ddim_steps:       int   = 50,
    ssim_weight:      float = 0.5,
    cfg_drop_prob:    float = 0.1,
) -> GaussianDiffusion3D:
    unet = Unet3D_CDM(
        dim=dim,
        dim_mults=dim_mults,
        in_channels=2,
        out_channels=1,
        init_kernel_size=init_kernel_size,
        resnet_groups=resnet_groups,
    )
    return GaussianDiffusion3D(
        denoise_fn=unet,
        timesteps=timesteps,
        ddim_steps=ddim_steps,
        ssim_weight=ssim_weight,
        cfg_drop_prob=cfg_drop_prob,
    )
