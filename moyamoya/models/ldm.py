"""
Paired Diffusion Model for post-surgery CBF prediction.

Image-space diffusion conditioned on x_pre:
    input = cat([x_t, x_pre], dim=1)   # 2 channels, full image space

The denoiser U-Net sees x_pre at full resolution at every layer through skip
connections, so the conditioning signal propagates at every spatial scale.
With eta=0 DDIM at inference the model behaves like deterministic regression.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules import (
    _match_size,
    sinusoidal_embedding,
    make_beta_schedule,
    ResBlock3D,
)


# ─────────────────────────────────────────────────────────────────────────────
# Denoising U-Net
# ─────────────────────────────────────────────────────────────────────────────

class DiffusionUNet3D(nn.Module):
    """
    U-Net denoiser in image space.
    Input: cat([x_t, x_pre], dim=1) — 2 channels — predicts noise for x_t.

    Args:
        in_ch:    2 (noisy image + conditioning image concatenated)
        out_ch:   1 (predicted noise, same shape as x_t)
        base:     base channel count for the U-Net
        t_dim:    sinusoidal time embedding dimension
        n_levels: number of down/up levels (spatial resolution halved each time)
    """
    def __init__(
        self,
        in_ch: int = 2,
        out_ch: int = 1,
        base: int = 64,
        t_dim: int = 256,
        n_levels: int = 2,
    ):
        super().__init__()
        self.t_dim = t_dim
        self.time_emb = nn.Sequential(
            nn.Linear(t_dim, t_dim * 2),
            nn.SiLU(),
            nn.Linear(t_dim * 2, t_dim),
        )

        self.enc_in = nn.Conv3d(in_ch, base, 3, padding=1)

        self.enc_blocks = nn.ModuleList()
        self.downs       = nn.ModuleList()
        enc_chs = []
        ch = base
        for _ in range(n_levels):
            self.enc_blocks.append(nn.ModuleList([
                ResBlock3D(ch, ch, t_dim),
                ResBlock3D(ch, ch, t_dim),
            ]))
            self.downs.append(nn.Conv3d(ch, ch * 2, 3, stride=2, padding=1))
            enc_chs.append(ch)
            ch *= 2

        self.mid = nn.ModuleList([
            ResBlock3D(ch, ch, t_dim),
            ResBlock3D(ch, ch, t_dim),
        ])

        self.ups       = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        for skip_ch in reversed(enc_chs):
            self.ups.append(nn.ConvTranspose3d(ch, skip_ch, kernel_size=2, stride=2))
            self.dec_blocks.append(nn.ModuleList([
                ResBlock3D(skip_ch * 2, skip_ch, t_dim),
                ResBlock3D(skip_ch, skip_ch, t_dim),
            ]))
            ch = skip_ch

        self.out = nn.Conv3d(ch, out_ch, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        x: (B, in_ch, D, H, W)
        t: (B,) int timesteps in [0, T)
        Returns: (B, out_ch, D, H, W) predicted noise
        """
        t_emb = sinusoidal_embedding(t, self.t_dim)
        t_emb = self.time_emb(t_emb)

        h = self.enc_in(x)
        skips = []
        for (r1, r2), down in zip(self.enc_blocks, self.downs):
            h = r1(h, t_emb)
            h = r2(h, t_emb)
            skips.append(h)
            h = down(h)

        for r in self.mid:
            h = r(h, t_emb)

        for (r1, r2), up, skip in zip(self.dec_blocks, self.ups, reversed(skips)):
            h = up(h)
            h = _match_size(h, skip)
            h = torch.cat([h, skip], dim=1)
            h = r1(h, t_emb)
            h = r2(h, t_emb)

        return self.out(h)


# ─────────────────────────────────────────────────────────────────────────────
# Paired Diffusion Model
# ─────────────────────────────────────────────────────────────────────────────

class PairedDiffusion(nn.Module):
    """
    Image-space diffusion model for paired pre→post prediction.

    Diffusion runs directly in image space; no VAE is needed.
    The denoiser receives cat([x_t, x_pre], dim=1) so x_pre is visible
    at every spatial scale through the U-Net skip connections.
    """
    def __init__(
        self,
        denoiser: DiffusionUNet3D,
        T: int = 1000,
        schedule: str = "cosine",
    ):
        super().__init__()
        self.denoiser = denoiser
        self.T        = T

        betas      = make_beta_schedule(T, schedule)
        alphas     = 1.0 - betas
        alphas_bar = alphas.cumprod(dim=0)

        self.register_buffer("betas",                     betas)
        self.register_buffer("alphas_bar",                alphas_bar)
        self.register_buffer("sqrt_alphas_bar",           alphas_bar.sqrt())
        self.register_buffer("sqrt_one_minus_alphas_bar", (1 - alphas_bar).sqrt())

    # ── diffusion helpers ───────────────────────────────────────────────────

    def q_sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward process: x_t = sqrt(ᾱ_t)*x_0 + sqrt(1-ᾱ_t)*ε."""
        if noise is None:
            noise = torch.randn_like(x0)
        a  = self.sqrt_alphas_bar[t][:, None, None, None, None]
        sa = self.sqrt_one_minus_alphas_bar[t][:, None, None, None, None]
        return a * x0 + sa * noise, noise

    def p_loss(
        self,
        x_post: torch.Tensor,
        x_pre: torch.Tensor,
        cfg_drop_prob: float = 0.15,
    ) -> torch.Tensor:
        """Training loss for the denoiser with classifier-free guidance dropout."""
        B          = x_post.size(0)
        t          = torch.randint(0, self.T, (B,), device=x_post.device)
        x_t, noise = self.q_sample(x_post, t)

        # Drop conditioning for a random subset of the batch
        keep = (torch.rand(B, device=x_post.device) >= cfg_drop_prob).float()
        x_pre_in = x_pre * keep[:, None, None, None, None]

        x_in       = torch.cat([x_t, x_pre_in], dim=1)    # (B, 2, D, H, W)
        pred_noise = self.denoiser(x_in, t)
        return F.mse_loss(pred_noise, noise)

    # ── sampling ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def sample_ddim(
        self,
        x_pre: torch.Tensor,
        steps: int = 50,
        eta: float = 0.0,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        """
        DDIM sampling (Song et al. 2020) conditioned on x_pre.
        eta=0 → fully deterministic; eta=1 → DDPM-like stochastic.
        guidance_scale > 1 applies classifier-free guidance (requires cfg_drop_prob > 0 at training).
        Returns x0_pred in image space, same shape as x_pre.
        """
        B      = x_pre.size(0)
        device = x_pre.device

        # uniform timestep sub-sequence τ ⊂ [0, T)
        tau = torch.linspace(self.T - 1, 0, steps, dtype=torch.long, device=device)

        # start from pure Gaussian noise
        x = torch.randn_like(x_pre)
        zero_cond = torch.zeros_like(x_pre)  # loop-invariant CFG conditioning

        for i, t_val in enumerate(tau):
            t_batch    = t_val.expand(B)
            x_in       = torch.cat([x, x_pre], dim=1)
            pred_noise = self.denoiser(x_in, t_batch)

            if guidance_scale != 1.0:
                x_in_uncond = torch.cat([x, zero_cond], dim=1)
                pred_uncond = self.denoiser(x_in_uncond, t_batch)
                pred_noise  = pred_uncond + guidance_scale * (pred_noise - pred_uncond)

            ab         = self.alphas_bar[t_val]
            x0_pred    = (x - (1 - ab).sqrt() * pred_noise) / ab.sqrt().clamp_min(1e-8)
            x0_pred    = x0_pred.clamp(-10.0, 10.0)

            if i < steps - 1:
                t_next  = tau[i + 1]
                ab_next = self.alphas_bar[t_next]
                dir_xt  = (1 - ab_next - eta**2 * (1 - ab_next)).sqrt() * pred_noise
                noise   = eta * (1 - ab_next).sqrt() * torch.randn_like(x)
                x       = ab_next.sqrt() * x0_pred + dir_xt + noise
            else:
                x = x0_pred

        return x

    @torch.no_grad()
    def generate(
        self,
        x_pre: torch.Tensor,
        ddim_steps: int = 50,
        eta: float = 0.0,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        """
        End-to-end: x_pre → x_post_pred.
        x_pre: (B, 1, D, H, W)  (already z-score normalised)
        Returns: x_post_pred with same shape as x_pre, background zeroed.
        """
        x_post = self.sample_ddim(x_pre, steps=ddim_steps, eta=eta, guidance_scale=guidance_scale)
        # zero background using the pre-surgery brain mask
        brain_mask = (x_pre != 0).float()
        return x_post * brain_mask


# ─────────────────────────────────────────────────────────────────────────────
# Factory helper
# ─────────────────────────────────────────────────────────────────────────────

def build_paired_diffusion(
    base: int = 64,
    t_dim: int = 256,
    n_levels: int = 2,
    T: int = 1000,
    schedule: str = "cosine",
) -> PairedDiffusion:
    denoiser = DiffusionUNet3D(
        in_ch=2,
        out_ch=1,
        base=base,
        t_dim=t_dim,
        n_levels=n_levels,
    )
    return PairedDiffusion(denoiser=denoiser, T=T, schedule=schedule)
