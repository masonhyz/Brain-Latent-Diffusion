import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from moyamoya.modules import DoubleConv3d, _match_size, DownBlock, UpBlock


def _make_mlp(in_dim: int, hidden: int, out_dim: int, n_layers: int = 2, dropout: float = 0.0) -> nn.Module:
    layers = []
    d = in_dim
    for i in range(max(n_layers - 1, 0)):
        layers += [nn.Linear(d, hidden), nn.SiLU()]
        if dropout > 0:
            layers += [nn.Dropout(dropout)]
        d = hidden
    layers += [nn.Linear(d, out_dim)]
    return nn.Sequential(*layers)


# -------------------------
# CVAE with UNet-like capacity, UNet-like interface
# -------------------------
class CVAE3D(nn.Module):
    """
    Conditional VAE that matches the *interface* of your UNet3D:
      - __init__(in_channels, out_channels, base=32, ...)
      - forward(x) -> tensor of shape [B, out_channels, D, H, W]

    Internally it:
      1) Encodes x to a latent distribution q(z|x)
      2) Samples z (reparameterization)
      3) Decodes z to reconstruct y_hat with out_channels

    Notes:
      - This is a *plain* VAE unless you pass a condition vector c to forward().
        To keep the same interface as UNet3D, `forward(x)` works with c=None.
      - During training you should also use `model.loss(...)` to add KL + recon loss.
      - If you want conditioning (e.g., age/sex/diagnosis), pass `c` to forward and loss.

    Typical usage:
      model = CVAE3D(in_channels=T, out_channels=1, base=32, latent_dim=256)
      y_hat = model(x)                               # same as UNet interface
      loss, logs = model.loss(x, y, recon="l1")       # includes KL

    Shapes:
      x: [B, in_channels, D, H, W]
      c: [B, cond_dim] (optional)
      out: [B, out_channels, D, H, W]
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        base: int = 32,
        latent_dim: int = 256,
        cond_dim: int = 0,            # set >0 to enable conditioning via vector c
        cond_hidden: int = 128,
        kl_weight: float = 1.0,
        sample_during_eval: bool = False,  # if False, use mean at eval for determinism
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base = base
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim
        self.kl_weight = kl_weight
        self.sample_during_eval = sample_during_eval

        # ---------- Encoder (UNet-like down path) ----------
        self.enc1 = DoubleConv3d(in_channels, base)
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = DoubleConv3d(base, base * 2)
        self.pool2 = nn.MaxPool3d(2)
        self.enc3 = DoubleConv3d(base * 2, base * 4)
        self.pool3 = nn.MaxPool3d(2)

        self.bottleneck = DoubleConv3d(base * 4, base * 8)

        # We'll turn the bottleneck feature map into a vector via global pooling
        self.global_pool = nn.AdaptiveAvgPool3d(1)  # -> [B, C, 1, 1, 1]
        enc_vec_dim = base * 8

        # Optional conditioning embedding (vector c)
        if cond_dim > 0:
            self.cond_enc = _make_mlp(cond_dim, cond_hidden, cond_hidden, n_layers=2)
            enc_in = enc_vec_dim + cond_hidden
        else:
            self.cond_enc = None
            enc_in = enc_vec_dim

        # Latent posterior parameters
        self.fc_mu = nn.Linear(enc_in, latent_dim)
        self.fc_logvar = nn.Linear(enc_in, latent_dim)

        # ---------- Decoder ----------
        # Optional conditioning into decoder (concat to z)
        if cond_dim > 0:
            self.cond_dec = _make_mlp(cond_dim, cond_hidden, cond_hidden, n_layers=2)
            dec_in = latent_dim + cond_hidden
        else:
            self.cond_dec = None
            dec_in = latent_dim

        # Map latent vector back to a small 3D feature map; we target the bottleneck resolution.
        # We'll infer spatial size at runtime from encoder skips, so use a linear layer to C only,
        # then expand to [B, C, d, h, w] by broadcasting and refine with convs.
        self.fc_z_to_c = nn.Linear(dec_in, base * 8)

        # Up path mirrors UNet; we will create a constant feature map then upsample,
        # fusing encoder skips (acts like a conditional generator).
        self.up3 = UpBlock(in_ch=base * 8, skip_ch=base * 4, out_ch=base * 4)
        self.up2 = UpBlock(in_ch=base * 4, skip_ch=base * 2, out_ch=base * 2)
        self.up1 = UpBlock(in_ch=base * 2, skip_ch=base, out_ch=base)

        self.out = nn.Conv3d(base, out_channels, kernel_size=1)

        # Cache for last forward pass (for training loops that want access)
        self.last_mu: Optional[torch.Tensor] = None
        self.last_logvar: Optional[torch.Tensor] = None

    # ---- VAE internals ----
    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x: torch.Tensor, c: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        # Down path (collect skips)
        e1 = self.enc1(x)               # [B, base, D, H, W]
        e2 = self.enc2(self.pool1(e1))  # [B, 2b, ...]
        e3 = self.enc3(self.pool2(e2))  # [B, 4b, ...]
        b = self.bottleneck(self.pool3(e3))  # [B, 8b, ...]

        # Vectorize
        h = self.global_pool(b).flatten(1)  # [B, 8b]

        if self.cond_enc is not None:
            if c is None:
                raise ValueError("cond_dim>0 but c=None was passed.")
            cemb = self.cond_enc(c)
            h = torch.cat([h, cemb], dim=1)

        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar, (e1, e2, e3)

    def decode(self, z: torch.Tensor, skips: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], c: Optional[torch.Tensor] = None) -> torch.Tensor:
        e1, e2, e3 = skips

        if self.cond_dec is not None:
            if c is None:
                raise ValueError("cond_dim>0 but c=None was passed.")
            cemb = self.cond_dec(c)
            z = torch.cat([z, cemb], dim=1)

        # Determine bottleneck spatial size from e3 (because after pool3, b is e3 pooled once)
        # e3 spatial dims: [B, 4b, d3, h3, w3] -> after pool3: [B, 4b, d3/2, h3/2, w3/2]
        d3, h3, w3 = e3.shape[-3:]
        d_b, h_b, w_b = max(d3 // 2, 1), max(h3 // 2, 1), max(w3 // 2, 1)

        cfeat = self.fc_z_to_c(z)              # [B, 8b]
        x = cfeat[:, :, None, None, None].expand(-1, -1, d_b, h_b, w_b)  # [B, 8b, d_b, h_b, w_b]

        # Up path with skips
        x = self.up3(x, e3)  # -> [B, 4b, d3, h3, w3]
        x = self.up2(x, e2)
        x = self.up1(x, e1)

        return self.out(x)

    # ---- Interface-matching forward ----
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Matches UNet3D forward signature: forward(x) -> y_hat.
        For conditioning, call forward_cond(x, c).
        """
        mu, logvar, skips = self.encode(x, c=None)
        self.last_mu, self.last_logvar = mu, logvar

        if self.training or self.sample_during_eval:
            z = self.reparameterize(mu, logvar)
        else:
            z = mu  # deterministic at eval by default

        return self.decode(z, skips, c=None)

    # ---- Optional conditioned forward (doesn't break UNet-like interface) ----
    def forward_cond(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Conditioned forward for when you actually want CVAE behavior:
          y_hat = model.forward_cond(x, c)
        """
        if self.cond_dim <= 0:
            raise ValueError("This model was created with cond_dim=0 (no conditioning).")

        mu, logvar, skips = self.encode(x, c=c)
        self.last_mu, self.last_logvar = mu, logvar

        if self.training or self.sample_during_eval:
            z = self.reparameterize(mu, logvar)
        else:
            z = mu

        return self.decode(z, skips, c=c)

    # ---- Loss helper ----
    def kl_divergence(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # KL(q(z|x) || N(0,I)) per-sample
        # = 0.5 * sum(exp(logvar) + mu^2 - 1 - logvar)
        kl = 0.5 * torch.sum(torch.exp(logvar) + mu**2 - 1.0 - logvar, dim=1)
        return kl.mean()

    def loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        c: Optional[torch.Tensor] = None,
        recon: str = "l1",
        beta: Optional[float] = None,  # override kl_weight (beta-VAE)
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute recon + KL loss.

        Args:
          x: input [B, in_ch, D, H, W]
          y: target [B, out_ch, D, H, W]
          c: optional condition [B, cond_dim]
          recon: "l1" or "mse"
          beta: if provided, uses this KL weight instead of self.kl_weight

        Returns:
          total_loss, logs dict
        """
        if c is None:
            y_hat = self.forward(x)
        else:
            y_hat = self.forward_cond(x, c)

        mu = self.last_mu
        logvar = self.last_logvar
        assert mu is not None and logvar is not None, "Internal error: mu/logvar not set."

        if recon.lower() == "l1":
            recon_loss = F.l1_loss(y_hat, y)
        elif recon.lower() in ("mse", "l2"):
            recon_loss = F.mse_loss(y_hat, y)
        else:
            raise ValueError(f"Unknown recon loss '{recon}'. Use 'l1' or 'mse'.")

        kl = self.kl_divergence(mu, logvar)
        w = self.kl_weight if beta is None else float(beta)
        total = recon_loss + w * kl

        logs = {
            "loss_total": float(total.detach().cpu()),
            "loss_recon": float(recon_loss.detach().cpu()),
            "loss_kl": float(kl.detach().cpu()),
            "kl_weight": w,
        }
        return total, logs

    # ---- Sampling helper (optional) ----
    @torch.no_grad()
    def sample(
        self,
        shape_like: torch.Tensor,
        c: Optional[torch.Tensor] = None,
        z: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Sample a reconstruction-like volume with same spatial size as `shape_like` input.
        This uses skips from encoding `shape_like` to set spatial sizes (useful when you want
        subject-structured generation). If you want unconditional generation, you could
        pass a dummy volume and ignore semantics.
        """
        if c is not None and self.cond_dim <= 0:
            raise ValueError("c was provided but model has cond_dim=0.")

        # Use encoder skips to define spatial sizes; ignore mu/logvar except for skips
        _, _, skips = self.encode(shape_like, c=c)

        if z is None:
            z = torch.randn(shape_like.shape[0], self.latent_dim, device=shape_like.device, dtype=shape_like.dtype)

        return self.decode(z, skips, c=c)
