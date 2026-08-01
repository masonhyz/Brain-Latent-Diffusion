"""
Paired Latent Diffusion Model for post-surgery CBF prediction.

Architecture (true LDM, as in Rombach et al. 2022):
    Stage 1 – KL-regularised 3D autoencoder
        Encoder : (B, 1, D, H, W) → (B, 2*z_ch, D/f, H/f, W/f)   [mean + log_var]
        Decoder : (B,   z_ch, D/f, H/f, W/f) → (B, 1, D, H, W)

    Stage 2 – Latent diffusion U-Net (3D)
        Denoiser input : cat([z_t, z_pre], dim=1)   (2*z_ch channels)
        Conditioning   : z_pre = encoder_mean(x_pre)  (deterministic)
        Loss           : MSE(ε_pred, ε) in latent space
        Sampling       : DDIM with optional stochasticity

The Encoder / Decoder blocks come from the 3D-converted CompVis code
(third_party/latent-diffusion/ldm/modules/diffusionmodules/model.py).
All Conv2d, ConvTranspose2d, and spatial-attention ops have been replaced
with their 3D counterparts; padding and reshape dimensions updated accordingly.

Public API (mirrors ldm.PairedDiffusion):
    build_paired_latent_diffusion(**kwargs) → PairedLatentDiffusion
    model.p_loss_ae(x)                 → autoencoder reconstruction loss
    model.p_loss(x_post, x_pre)        → diffusion denoising loss (latent space)
    model.generate(x_pre, ...)         → x_post_pred in image space
"""

import sys
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── make the cloned CompVis code importable ───────────────────────────────────
_LDM_ROOT = Path(__file__).parent.parent.parent / "third_party" / "latent-diffusion"
if str(_LDM_ROOT) not in sys.path:
    sys.path.insert(0, str(_LDM_ROOT))

from ldm.modules.diffusionmodules.model import Encoder, Decoder   # 3D versions

from ..modules import (
    _match_size,
    sinusoidal_embedding,
    make_beta_schedule,
    ResBlock3D,
)

# Latent-space blocks use a GroupNorm ceiling of 32 (vs 8 in image space).
_LatentResBlock = partial(ResBlock3D, gn_max=32)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — 3D KL Autoencoder
# ─────────────────────────────────────────────────────────────────────────────

class DiagonalGaussian:
    """Thin wrapper around (mean, log_var) tensors — mirrors CompVis distribution."""
    def __init__(self, parameters: torch.Tensor):
        # parameters: (B, 2*z_ch, D, H, W) — first half = mean, second = log_var
        self.mean, self.log_var = parameters.chunk(2, dim=1)
        self.log_var = self.log_var.clamp(-30.0, 20.0)
        self.std = (0.5 * self.log_var).exp()

    def sample(self) -> torch.Tensor:
        return self.mean + self.std * torch.randn_like(self.mean)

    def mode(self) -> torch.Tensor:
        return self.mean

    def kl(self) -> torch.Tensor:
        """KL divergence from N(0,1), summed over all dims, averaged over batch."""
        # Compute in float32: std.pow(2) = exp(log_var) can reach ~4.85e8 under
        # the [-30, 20] clamp, which overflows float16 (~65504) under AMP.
        mean = self.mean.float()
        log_var = self.log_var.float()
        return 0.5 * (mean.pow(2) + log_var.exp() - log_var - 1).sum() / mean.shape[0]


class AutoencoderKL3D(nn.Module):
    """
    KL-regularised 3D variational autoencoder.

    Uses the 3D-converted CompVis Encoder / Decoder from model.py.
    quant_conv maps encoder output → (mean, log_var) in embed_dim space.
    post_quant_conv maps sampled z → decoder input.
    """
    def __init__(
        self,
        in_channels:    int = 1,
        z_channels:     int = 4,
        embed_dim:      int = 4,
        ch:             int = 64,
        ch_mult:        tuple = (1, 2, 4),
        num_res_blocks: int = 2,
        attn_resolutions: list = [],
        resolution:     int = 64,
        dropout:        float = 0.0,
        kl_weight:      float = 1e-6,
    ):
        super().__init__()
        self.kl_weight = kl_weight

        ddconfig = dict(
            in_channels=in_channels,
            out_ch=in_channels,
            z_channels=z_channels,
            embed_dim=embed_dim,
            ch=ch,
            ch_mult=ch_mult,
            num_res_blocks=num_res_blocks,
            attn_resolutions=attn_resolutions,
            resolution=resolution,
            dropout=dropout,
            double_z=True,
        )

        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)

        # 1×1×1 pointwise convolutions — 3D
        self.quant_conv      = nn.Conv3d(2 * z_channels, 2 * embed_dim, 1)
        self.post_quant_conv = nn.Conv3d(embed_dim, z_channels, 1)

        self.embed_dim = embed_dim

    def encode(self, x: torch.Tensor) -> DiagonalGaussian:
        h = self.encoder(x)
        moments = self.quant_conv(h)
        return DiagonalGaussian(moments)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        z = self.post_quant_conv(z)
        return self.decoder(z)

    def forward(self, x: torch.Tensor, sample_posterior: bool = True):
        posterior = self.encode(x)
        z = posterior.sample() if sample_posterior else posterior.mode()
        recon = self.decode(z)
        # Odd spatial dims: asymmetric padding in Downsample means the decode
        # path can be 1 voxel off (either direction). Force exact size match.
        if recon.shape[2:] != x.shape[2:]:
            recon = F.interpolate(recon, size=x.shape[2:], mode="trilinear", align_corners=False)
        return recon, posterior

    def rec_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Combined reconstruction + KL loss for Stage-1 training."""
        recon, posterior = self(x, sample_posterior=True)
        rec  = F.mse_loss(recon, x)
        kl   = posterior.kl()
        return rec + self.kl_weight * kl


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — Latent-space 3D Diffusion U-Net
# ─────────────────────────────────────────────────────────────────────────────

class LatentDiffusionUNet3D(nn.Module):
    """
    3D U-Net denoiser operating in the latent space of AutoencoderKL3D.

    Input channels: 2 * embed_dim  (noisy latent z_t  +  conditioning latent z_pre)
    Output channels:    embed_dim  (predicted noise for z_t)
    """
    def __init__(
        self,
        embed_dim: int = 4,
        base:      int = 64,
        t_dim:     int = 256,
        n_levels:  int = 3,
    ):
        super().__init__()
        in_ch  = 2 * embed_dim
        out_ch = embed_dim

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
                _LatentResBlock(ch, ch, t_dim),
                _LatentResBlock(ch, ch, t_dim),
            ]))
            self.downs.append(nn.Conv3d(ch, ch * 2, 3, stride=2, padding=1))
            enc_chs.append(ch)
            ch *= 2

        self.mid = nn.ModuleList([
            _LatentResBlock(ch, ch, t_dim),
            _LatentResBlock(ch, ch, t_dim),
        ])

        self.ups        = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        for skip_ch in reversed(enc_chs):
            self.ups.append(nn.ConvTranspose3d(ch, skip_ch, kernel_size=2, stride=2))
            self.dec_blocks.append(nn.ModuleList([
                _LatentResBlock(skip_ch * 2, skip_ch, t_dim),
                _LatentResBlock(skip_ch, skip_ch, t_dim),
            ]))
            ch = skip_ch

        self.out = nn.Conv3d(ch, out_ch, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 2*embed_dim, D', H', W')  cat([z_t, z_pre])
        t: (B,) int timesteps
        Returns: (B, embed_dim, D', H', W') predicted noise
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
# Full Paired Latent Diffusion Model
# ─────────────────────────────────────────────────────────────────────────────

class PairedLatentDiffusion(nn.Module):
    """
    Full 3D latent diffusion model for paired pre→post fMRI prediction.

    Stage 1 (AutoencoderKL3D): trained separately via rec_loss().
    Stage 2 (LatentDiffusionUNet3D): trained via p_loss() with encoder frozen.

    At inference, generate() encodes x_pre, runs DDIM in latent space, then
    decodes back to image space.
    """
    def __init__(
        self,
        ae:       AutoencoderKL3D,
        denoiser: LatentDiffusionUNet3D,
        T:        int = 1000,
        schedule: str = "cosine",
    ):
        super().__init__()
        self.ae       = ae
        self.denoiser = denoiser
        self.T        = T

        betas      = make_beta_schedule(T, schedule)
        alphas     = 1.0 - betas
        alphas_bar = alphas.cumprod(dim=0)

        self.register_buffer("betas",                     betas)
        self.register_buffer("alphas_bar",                alphas_bar)
        self.register_buffer("sqrt_alphas_bar",           alphas_bar.sqrt())
        self.register_buffer("sqrt_one_minus_alphas_bar", (1 - alphas_bar).sqrt())

    # ── autoencoder helpers ─────────────────────────────────────────────────

    def p_loss_ae(self, x: torch.Tensor) -> torch.Tensor:
        """Stage-1 loss: reconstruction + KL."""
        return self.ae.rec_loss(x)

    @torch.no_grad()
    def encode_mean(self, x: torch.Tensor) -> torch.Tensor:
        """Deterministic encoding (mode of posterior)."""
        return self.ae.encode(x).mode()

    # ── diffusion helpers ────────────────────────────────────────────────────

    def q_sample(
        self,
        z0:    torch.Tensor,
        t:     torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward diffusion: z_t = sqrt(ᾱ_t)*z_0 + sqrt(1-ᾱ_t)*ε."""
        if noise is None:
            noise = torch.randn_like(z0)
        a  = self.sqrt_alphas_bar[t][:, None, None, None, None]
        sa = self.sqrt_one_minus_alphas_bar[t][:, None, None, None, None]
        return a * z0 + sa * noise, noise

    def p_loss(
        self,
        x_post: torch.Tensor,
        x_pre:  torch.Tensor,
        cfg_drop_prob: float = 0.15,
    ) -> torch.Tensor:
        """
        Stage-2 training loss with classifier-free guidance dropout.
        Encodes both volumes, adds noise to z_post, predicts it conditioned on z_pre.
        """
        B = x_post.size(0)
        with torch.no_grad():
            z_post = self.ae.encode(x_post).sample()
            z_pre  = self.ae.encode(x_pre).mode()

        # Drop conditioning for a random subset of the batch
        keep = (torch.rand(B, device=x_post.device) >= cfg_drop_prob).float()
        z_pre_in = z_pre * keep[:, None, None, None, None]

        t          = torch.randint(0, self.T, (B,), device=x_post.device)
        z_t, noise = self.q_sample(z_post, t)
        z_in       = torch.cat([z_t, z_pre_in], dim=1)   # (B, 2*embed_dim, D', H', W')
        pred_noise = self.denoiser(z_in, t)
        return F.mse_loss(pred_noise, noise)

    # ── sampling ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def sample_ddim(
        self,
        z_pre:          torch.Tensor,
        steps:          int   = 50,
        eta:            float = 0.0,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        """
        DDIM sampling in latent space conditioned on z_pre.
        eta=0 → fully deterministic; eta=1 → DDPM-like stochastic.
        guidance_scale > 1 applies classifier-free guidance (requires cfg_drop_prob > 0 at training).
        Returns z0_pred (latent), same spatial shape as z_pre.
        """
        B      = z_pre.size(0)
        device = z_pre.device

        tau = torch.linspace(self.T - 1, 0, steps, dtype=torch.long, device=device)
        z   = torch.randn_like(z_pre)
        zero_cond = torch.zeros_like(z_pre)  # loop-invariant CFG conditioning

        for i, t_val in enumerate(tau):
            t_batch    = t_val.expand(B)
            z_in       = torch.cat([z, z_pre], dim=1)
            pred_noise = self.denoiser(z_in, t_batch)

            if guidance_scale != 1.0:
                z_in_uncond = torch.cat([z, zero_cond], dim=1)
                pred_uncond = self.denoiser(z_in_uncond, t_batch)
                pred_noise  = pred_uncond + guidance_scale * (pred_noise - pred_uncond)

            ab      = self.alphas_bar[t_val]
            z0_pred = (z - (1 - ab).sqrt() * pred_noise) / ab.sqrt().clamp_min(1e-8)
            z0_pred = z0_pred.clamp(-10.0, 10.0)

            if i < steps - 1:
                t_next  = tau[i + 1]
                ab_next = self.alphas_bar[t_next]
                dir_xt  = (1 - ab_next - eta**2 * (1 - ab_next)).sqrt() * pred_noise
                noise   = eta * (1 - ab_next).sqrt() * torch.randn_like(z)
                z       = ab_next.sqrt() * z0_pred + dir_xt + noise
            else:
                z = z0_pred

        return z

    @torch.no_grad()
    def generate(
        self,
        x_pre:          torch.Tensor,
        ddim_steps:     int   = 50,
        eta:            float = 0.0,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        """
        End-to-end prediction: x_pre → x_post_pred.
        x_pre: (B, 1, D, H, W) — z-score normalised
        Returns x_post_pred in image space, background zeroed.
        """
        z_pre  = self.encode_mean(x_pre)
        z_post = self.sample_ddim(z_pre, steps=ddim_steps, eta=eta, guidance_scale=guidance_scale)
        x_post = self.ae.decode(z_post)

        # Odd dims: decode may be 1 voxel off from the original input size.
        if x_post.shape[2:] != x_pre.shape[2:]:
            x_post = F.interpolate(x_post, size=x_pre.shape[2:], mode="trilinear", align_corners=False)

        brain_mask = (x_pre != 0).float()
        return x_post * brain_mask


# ─────────────────────────────────────────────────────────────────────────────
# Factory helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_autoencoder(
    z_channels:     int   = 4,
    embed_dim:      int   = 4,
    ch:             int   = 64,
    ch_mult:        tuple = (1, 2, 4),
    num_res_blocks: int   = 2,
    attn_resolutions: list = [],
    resolution:     int   = 64,
    kl_weight:      float = 1e-6,
) -> AutoencoderKL3D:
    return AutoencoderKL3D(
        in_channels=1,
        z_channels=z_channels,
        embed_dim=embed_dim,
        ch=ch,
        ch_mult=ch_mult,
        num_res_blocks=num_res_blocks,
        attn_resolutions=attn_resolutions,
        resolution=resolution,
        kl_weight=kl_weight,
    )


def build_paired_latent_diffusion(
    # autoencoder
    z_channels:     int   = 4,
    embed_dim:      int   = 4,
    ae_ch:          int   = 64,
    ae_ch_mult:     tuple = (1, 2, 4),
    ae_res_blocks:  int   = 2,
    ae_resolution:  int   = 64,
    kl_weight:      float = 1e-6,
    # latent U-Net
    diff_base:      int   = 64,
    t_dim:          int   = 256,
    n_levels:       int   = 3,
    # diffusion schedule
    T:              int   = 1000,
    schedule:       str   = "cosine",
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
    denoiser = LatentDiffusionUNet3D(
        embed_dim=embed_dim,
        base=diff_base,
        t_dim=t_dim,
        n_levels=n_levels,
    )
    return PairedLatentDiffusion(ae=ae, denoiser=denoiser, T=T, schedule=schedule)
