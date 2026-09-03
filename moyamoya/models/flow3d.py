"""
Flow3D — conditional flow matching for paired pre→post CBF prediction.

Algorithm (follows the official Conditional Flow Matching code)
─────────────────────────────────────────────────────────────────────────────
This is the ``ConditionalFlowMatcher`` of Lipman et al. 2023 ("Flow Matching for
Generative Modeling") / Tong et al. 2024 ("Improving and generalizing flow-based
generative models with minibatch optimal transport", the ``torchcfm`` reference
implementation), applied to a *paired* prediction task. The official recipe is:

    x0 ~ source,  x1 = x_post,  t ~ U(0,1),  ε ~ N(0, I)
    x_t = (1-t)·x0 + t·x1 + σ·ε          ← Gaussian probability path, CONSTANT σ
    u_t = x1 - x0                        ← conditional velocity (clean target)
    loss = E ‖ v_θ(x_t, t | cond) - u_t ‖²

The two things that matter and were wrong before:

  1. **The velocity net is conditioned on x_pre.** The task is to predict x_post
     *from a specific x_pre*, so the network must see x_pre — otherwise it can
     only learn E[u_t | x_t], the field that transports the *marginals* while
     scrambling the pairing, and in practice the loss barely moves off its
     initial value because the pre→post residual is not recoverable from x_t
     alone. x_pre is supplied by channel concatenation, exactly as every
     diffusion model in this repo conditions its denoiser. This is the fix that
     makes training actually descend.

  2. **The regression target is u_t = x1 - x0, and σ is a constant** (official
     ``ConditionalFlowMatcher``). The σ·ε perturbs the *sample location* only; it
     is deliberately NOT added to the target. An earlier version used a
     time-varying γ(t)=σ·sin(πt) and a pathwise target (x1-x0)+γ'(t)·z, which is
     a valid stochastic-interpolant variant but inflates the loss with the
     irreducible variance of γ'(t)·z and is not what the official code does.

Source distribution
─────────────────────────────────────────────────────────────────────────────
A flow model is free to choose its source, and here the natural one is the
pre-op volume itself (``source="pre"``, a data-to-data *bridge*):

    x0 = x_pre,  sampling integrates dx/dt = v_θ(x_t, t | x_pre) from x(0)=x_pre.

Two consequences:

  * With the output conv zero-initialised (``zero_init_out``) an untrained model
    returns exactly x_pre, i.e. training *starts* at the identity baseline — the
    strongest predictor on this dataset (see the metrics docstring) — and only
    has to learn the residual from there.
  * The path is short (masked ‖x_post-x_pre‖ is small), so the ODE is nearly
    straight and 4–16 NFE suffices instead of 50 DDIM steps.

``source="noise"`` recovers standard CFM from a Gaussian (x0 ~ N(0,I)) for a
like-for-like comparison against the diffusion models with the architecture held
fixed — only the objective and the sampler differ.

Why σ > 0 is required for the bridge
─────────────────────────────────────────────────────────────────────────────
With ``source="pre"``, ``condition="concat"`` and **σ = 0** the path is the
deterministic line x_t = (1-t)·x_pre + t·x_post, so a network handed both x_t and
x_pre can read the target off algebraically:

    x_post - x_pre  =  (x_t - x_pre) / t .

That drives the *training* loss to ~0 while the *sampling* trajectory collapses:
the velocity at t=0 is 0/0, so an Euler step from x(0)=x_pre has no defined
direction and the prediction never leaves the input (measured: MAE stuck at the
identity value while train-loss looks excellent). σ > 0 removes the degeneracy —
with x_t = (1-t)·x_pre + t·x_post + σ·ε the target x_post is no longer determined
by (x_t, x_pre, t), so the network must regress the genuine conditional mean
E[x_post - x_pre | x_t, x_pre]; at t→0 that is E[x_post - x_pre | x_pre], a
well-defined initial velocity. This is the standard image-to-image-bridge fix
(cf. I2SB, stochastic interpolants). The default σ is therefore non-zero and the
factory warns on the σ=0 + bridge + concat combination.

The bridge noise σ·ε is masked by ``(x_pre != 0) | (x_post != 0)``. Under the
default whole-volume normalisation (``zero_background=False``, matching the
7TCDM3D/LDM models) the background is a constant ~-2.4 plateau, not 0, so that
mask is every voxel and the noise is applied everywhere — the model simply learns
a ~0 velocity on the background plateau and sampling keeps it there. Only under
``zero_background=True`` (exact-zero background) does the mask actually confine
the noise to the brain. The Gaussian *source* of ``source="noise"`` is never
masked (there it IS the source, and sampling starts from a full randn).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cdm3d import Unet3D_CDM, ssim3d_loss


# ─────────────────────────────────────────────────────────────────────────────
# Interpolant helpers
# ─────────────────────────────────────────────────────────────────────────────

def _b(v: torch.Tensor) -> torch.Tensor:
    """(B,) → (B,1,1,1,1) so it broadcasts against a 5-D volume batch."""
    return v[:, None, None, None, None]


def sample_t(
    batch: int,
    device,
    dist: str = "uniform",
    logit_mean: float = 0.0,
    logit_std: float = 1.0,
) -> torch.Tensor:
    """Draw training times t ∈ (0,1).

    ``uniform``      — t ~ U(0,1). The right default for the bridge: the path is
                       short and homogeneous, so no region of it is special.
    ``logit_normal`` — t = sigmoid(m + s·ε), the SD3 schedule. Concentrates
                       samples near t=0.5, which helps when the path is long
                       (i.e. ``source="noise"``) and the endpoints are trivial.
    """
    if dist == "uniform":
        return torch.rand(batch, device=device)
    if dist == "logit_normal":
        return torch.sigmoid(logit_mean + logit_std * torch.randn(batch, device=device))
    raise ValueError(f"unknown t distribution: {dist!r} (use 'uniform' or 'logit_normal')")


def change_weight_map(
    x_pre:  torch.Tensor,
    x_post: torch.Tensor,
    gamma:  float,
    eps:    float = 1e-6,
) -> torch.Tensor:
    """Per-voxel loss weight emphasising where the scan actually changes.

        w = 1 + γ·|x_post − x_pre| / scale,      scale = per-sample mean |Δ|
        w ← w / mean(w)                           (renormalised, per sample)

    The renormalisation makes ``w`` average to 1 over every sample, so the loss
    magnitude — and therefore the effective learning rate — is identical to the
    unweighted MSE; γ only *redistributes* gradient toward the voxels that differ
    between pre and post. That is the whole fix: with a flat weight the vast
    near-zero-Δ majority dominates and v≡0 (copy x_pre) minimises the loss, so the
    sparse surgical edits are never learned. γ=0 returns all-ones (plain MSE).

    Δ is normalised by its own per-sample mean, so the emphasis is scale-invariant
    across subjects: a subject with a small absolute change still has its changed
    voxels up-weighted relative to its own unchanged ones. *Which* subjects to
    dwell on is the sampler's job (see moyamoya/data.py), not this weight's.

    A pure function of the data — carries no gradient. Shapes (B,C,D,H,W) (or any
    (B, …) tensor); the reduction is over every non-batch dim.
    """
    c = (x_post - x_pre).abs()
    dims = tuple(range(1, c.ndim))
    scale = c.mean(dim=dims, keepdim=True).clamp_min(eps)
    w = 1.0 + gamma * (c / scale)
    return w / w.mean(dim=dims, keepdim=True).clamp_min(eps)


# ─────────────────────────────────────────────────────────────────────────────
# Coherent-change objective — the "learn where + learn the real detail" fix
# ─────────────────────────────────────────────────────────────────────────────
#
# The empirical finding behind this whole block (measured on all 235 pairs):
# the top-few-% "change" a subject actually undergoes is roughly HALF unpredictable
# noise — ~60% high-frequency, 2.2× concentrated on x_pre's structural edges (a
# misregistration signature, since pre/post are separate acquisitions), sign-balanced
# and only weakly lateralised. The part a model *can* learn is the coherent,
# low-frequency, subject-specific edit (an oracle smooth edit already removes ~62%
# of the change-ROI error). So the objective must (a) supervise WHERE the change is
# as an imbalance-robust detection problem, and (b) supervise WHAT changes at the
# scales where it is predictable — instead of pouring MSE gradient into the noisy
# voxel tail (change_weight_map's failure mode: it up-weights the least predictable
# voxels) or synthesising plausible-but-wrong speckle (the adversarial risk).

def gaussian_blur3d(x: torch.Tensor, sigma: float, truncate: float = 2.0) -> torch.Tensor:
    """Separable 3-D Gaussian blur over the spatial dims of an (B,C,D,H,W) volume.

    Three depthwise 1-D convolutions (one per axis), so cost is linear in the
    kernel radius rather than cubic. ``sigma`` is in voxels; ``sigma<=0`` is a
    no-op. Used to isolate the *coherent* (low-frequency) part of the change,
    which is the part that is actually predictable from x_pre.
    """
    if not sigma or sigma <= 0:
        return x
    radius = max(1, int(truncate * float(sigma) + 0.5))
    coords = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
    k = torch.exp(-(coords ** 2) / (2.0 * float(sigma) ** 2))
    k = k / k.sum()
    C = x.shape[1]
    for ax in range(3):                                    # over D, H, W
        ksz = [1, 1, 1]; ksz[ax] = k.numel()
        kernel = k.view(1, 1, *ksz).repeat(C, 1, 1, 1, 1)
        pad = [0, 0, 0]; pad[ax] = radius
        x = F.conv3d(x, kernel, padding=tuple(pad), groups=C)
    return x


def coherent_change_target(
    x_pre:  torch.Tensor,
    x_post: torch.Tensor,
    sigma:  float = 2.0,
    q:      float = 0.995,
    eps:    float = 1e-6,
) -> torch.Tensor:
    """Soft label in [0,1] marking WHERE the *coherent* change happens.

        s = gaussian_blur(|x_post − x_pre|, sigma)   ← drop the high-freq noise
        target = clamp(s / quantile(s, q), 0, 1)     ← per-sample soft occupancy

    Blurring first is the whole point: the raw |Δ| tail is dominated by
    edge/registration speckle, so a detector trained on it would chase noise; the
    blurred field keeps only the spatially-coherent edit. No brain mask is needed —
    both volumes share the same background plateau (and the same paired
    augmentation), so Δ≈0 there and the target is already ≈0 off-brain. A pure
    function of the data (no gradient). Shapes (B,C,D,H,W). Computed in fp32 (it
    runs under bf16 autocast, and ``torch.quantile`` requires float/double).
    """
    # .float() AFTER the blur: the conv inside runs under bf16 autocast, and
    # torch.quantile below rejects half/bf16.
    s = gaussian_blur3d((x_post - x_pre).float().abs(), sigma).float()
    flat = s.flatten(1)
    scale = torch.quantile(flat, q, dim=1).clamp_min(eps)
    scale = scale.view(-1, *([1] * (s.ndim - 1)))
    return (s / scale).clamp(0.0, 1.0)


def soft_dice_loss(p: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    """1 − soft Dice between a predicted map ``p``∈[0,1] and a soft ``target``∈[0,1].

    The imbalance-robust localisation loss: Dice normalises by the size of the
    predicted+target regions, so it is not swamped by the ~95% of voxels that do
    not change — exactly the regime where a re-weighted MSE fails. Per-sample
    reduction; ``eps`` (Laplace smoothing) keeps an all-zero target well-defined.
    """
    dims = tuple(range(1, p.ndim))
    num = 2.0 * (p * target).sum(dims) + eps
    den = (p + target).sum(dims) + eps
    return (1.0 - num / den).mean()


def edge_downweight_map(
    x_pre: torch.Tensor,
    gamma: float,
    q:     float = 0.8,
    floor: float = 0.1,
    eps:   float = 1e-6,
) -> torch.Tensor:
    """Per-voxel weight in (0,1] that *down*-weights x_pre's high-gradient edges.

        w = clamp(1 − γ·|∇x_pre|/quantile(|∇x_pre|, q), floor, 1)

    The change-ROI is 2.2× enriched on structural edges, where a sub-voxel
    pre/post misalignment manufactures large apparent change that no model can
    predict. Softly discounting those voxels keeps the regression from being
    dominated by registration artefacts. γ=0 returns all-ones. |∇| is a simple
    first-difference magnitude; a pure function of x_pre (no gradient).
    """
    if gamma <= 0:
        return torch.ones_like(x_pre)
    xf = x_pre.float()                                  # fp32: quantile needs it
    gm = torch.zeros_like(xf)
    for ax in range(2, xf.ndim):
        df = xf - torch.roll(xf, 1, dims=ax)
        gm = gm + df * df
    gm = gm.sqrt()
    scale = torch.quantile(gm.flatten(1), q, dim=1).clamp_min(eps)
    scale = scale.view(-1, *([1] * (xf.ndim - 1)))
    w = (1.0 - gamma * (gm / scale)).clamp(floor, 1.0)
    return w.to(x_pre.dtype)


def multiscale_change_loss(
    v:       torch.Tensor,
    u:       torch.Tensor,
    weights,
    voxel_w: torch.Tensor = None,
) -> torch.Tensor:
    """Multi-scale MSE between predicted change ``v`` and target change ``u``.

    ``weights[i]`` weights the residual after ``i`` successive ×2 average-pools, so
    higher-index terms score progressively coarser (lower-frequency) structure —
    the coherent, predictable part of the surgical edit. Averaging over a pooled
    field is a low-pass filter, so the coarse terms are (near-)blind to the
    high-frequency registration speckle that a single full-resolution MSE otherwise
    chases. ``voxel_w`` (e.g. an edge down-weight) is applied at the full-resolution
    scale only. Normalised by Σweights so the loss magnitude tracks a plain MSE.
    ``weights=[1.0]`` recovers exactly ``(voxel_w·(v−u)²).mean()``.
    """
    total = v.new_zeros(())
    norm = 0.0
    cv, cu = v, u
    for i, wt in enumerate(weights):
        if wt > 0:
            se = (cv - cu).pow(2)
            if i == 0 and voxel_w is not None:
                term = (voxel_w * se).mean()
            else:
                term = se.mean()
            total = total + wt * term
            norm += wt
        if i < len(weights) - 1:
            cv = F.avg_pool3d(cv, 2)
            cu = F.avg_pool3d(cu, 2)
    return total / max(norm, 1e-8)


# ─────────────────────────────────────────────────────────────────────────────
# Adversarial detail term (hinge-GAN)
# ─────────────────────────────────────────────────────────────────────────────

def d_hinge_loss(real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
    """Discriminator hinge loss — push real logits ≥ +1 and fake logits ≤ −1.

    The hinge form (Lim & Ye 2017 / Miyato 2018) is the stable default for
    spectral-normalised GANs: once a patch is confidently correct it stops
    contributing gradient, which keeps the discriminator from running away on a
    small dataset.
    """
    return torch.relu(1.0 - real_logits).mean() + torch.relu(1.0 + fake_logits).mean()


def g_hinge_loss(fake_logits: torch.Tensor) -> torch.Tensor:
    """Generator hinge loss — raise the discriminator's score on the fakes."""
    return -fake_logits.mean()


# ─────────────────────────────────────────────────────────────────────────────
# EMA
# ─────────────────────────────────────────────────────────────────────────────

class EMA:
    """Exponential moving average of model weights, with linear warmup.

    ``beta`` is ramped from 0 over the first ``warmup`` updates so the shadow
    copy is not dragged by the (meaningless) initial weights — with 235 subjects
    an epoch is ~100 steps, so a fixed beta=0.999 would otherwise take many
    epochs just to forget the initialisation.
    """

    def __init__(self, beta: float = 0.999, warmup: int = 500):
        self.beta = beta
        self.warmup = max(int(warmup), 0)
        self.step = 0

    def _beta(self) -> float:
        if self.step >= self.warmup:
            return self.beta
        return min(self.beta, self.step / max(self.warmup, 1))

    @torch.no_grad()
    def update(self, ema_model: nn.Module, model: nn.Module) -> None:
        b = self._beta()
        self.step += 1
        for ep, p in zip(ema_model.parameters(), model.parameters()):
            ep.data.mul_(b).add_(p.data, alpha=1 - b)
        # Buffers (GroupNorm has none, but keep the copy exact if that changes).
        for eb, bf in zip(ema_model.buffers(), model.buffers()):
            eb.data.copy_(bf.data)


# ─────────────────────────────────────────────────────────────────────────────
# Gated-residual velocity — the "learn WHERE, then WHAT" factorisation
# ─────────────────────────────────────────────────────────────────────────────

class GatedVelocityNet(nn.Module):
    """Wrap a 2-output-channel backbone into a gated-residual velocity field.

        (gate_logit, residual) = backbone(x, t)
        gate = sigmoid(gate_logit)                 ← WHERE to edit, in [0,1]
        v    = gate · residual                     ← WHAT edit, only where gate>0

    The factorisation is the point. For the bridge the velocity *is* the change
    field (u_t = x_post − x_pre), so multiplying a localisation map into a residual
    forces the network to first decide *where* this subject changes and to return
    ~0 (identity) everywhere else — instead of smearing a timid edit across the
    whole brain, the mode-averaging failure of a plain regression. The gate is a
    genuine, supervisable change-detector (see :func:`coherent_change_target`): even
    when the residual's fine detail is uncertain, a crisp gate still edits the right
    place at the right extent, which is exactly the "learn when and where" ask.

    ``forward`` returns just the velocity, so this is a drop-in for the plain net
    in the sampler and the one-step head; :meth:`gate_velocity` also returns the
    gate for the detection loss and for visualisation.
    """

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def gate_velocity(self, x: torch.Tensor, t: torch.Tensor):
        out = self.backbone(x, t)
        gate = torch.sigmoid(out[:, 0:1])
        res = out[:, 1:2]
        return gate, gate * res

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.gate_velocity(x, t)[1]


# ─────────────────────────────────────────────────────────────────────────────
# Conditional flow matching wrapper
# ─────────────────────────────────────────────────────────────────────────────

class ConditionalFlowMatching3D(nn.Module):
    """Conditional flow matching in image space for paired pre→post prediction.

    Implements the official ``ConditionalFlowMatcher`` path/target
    (x_t = (1-t)·x0 + t·x1 + σ·ε, u_t = x1 - x0) with the velocity network
    conditioned on x_pre.

    Args:
        net: velocity network mapping (B,C,D,H,W) and (B,) times → (B,1,D,H,W),
            with C = 2 for ``condition="concat"`` and C = 1 for ``"none"``.
        source: ``"pre"`` transports x_pre → x_post (bridge, the default);
            ``"noise"`` transports N(0,I) → x_post (standard CFM).
        condition: how x_pre reaches the network. ``"concat"`` — channel
            concatenation, the default and effectively required: the task is
            conditional on x_pre, and without it training does not descend.
            ``"none"`` — the network sees x_t alone; kept only for ablation.
        sigma: constant σ in x_t = (1-t)·x0 + t·x1 + σ·ε. For the bridge this
            must be > 0 (see the module docstring); it smooths the velocity field
            around the path and removes the σ=0 algebraic degeneracy.
        t_dist / logit_mean / logit_std: training-time distribution over t.
        change_weight: γ in the per-voxel loss weight w = 1 + γ·(|Δ|/scale),
            where Δ = x_post − x_pre is the *change map* (the very thing the bridge
            regresses, since u_t = x_post − x_pre) and scale is Δ's per-sample mean.
            w is then renormalised to mean 1 per sample, so the loss magnitude —
            and hence the effective learning rate — is unchanged; γ only
            *redistributes* emphasis toward the voxels that actually change between
            pre- and post-op. This is the fix for the identity-collapse failure:
            with a flat MSE the near-zero Δ of the vast unchanged majority drowns
            out the sparse edits, so v≈0 (copy x_pre) is the minimiser. γ=0
            recovers the plain MSE exactly. Only applied for source="pre" (Δ is
            only the target velocity for the bridge). See :meth:`flow_loss`.
        l1_weight: weight of an L1 term on the velocity residual, added to the
            MSE. L1 is more robust to the heavy-tailed voxels at vessel edges.
            Change-weighted with the same w as the MSE term.
        ssim_weight: weight of (1 − SSIM3D) between the *one-step x₁ estimate*
            and x_post. See :meth:`flow_loss`.
        cfg_drop_prob: probability of zeroing the conditioning channel, for
            classifier-free guidance. Only coherent for ``source="noise"``.
        gated: if the ``net`` is a :class:`GatedVelocityNet`, its output is a gated
            residual v = gate·residual and the gate is a supervisable change
            detector (see ``det_weight``). Set automatically by the factory.
        det_weight: weight of the change-detection loss — a soft-Dice between the
            gate map and :func:`coherent_change_target` (the *coherent*,
            low-frequency change region). This is the "learn WHERE" term; it is the
            imbalance-robust localisation loss the re-weighted MSE could not be.
            Requires a gated net and source="pre". 0 = off.
        det_sigma: Gaussian σ (voxels) that defines the coherent-change target the
            detector is trained on. Larger = smoother/coarser change region.
        ms_weights: per-scale weights for :func:`multiscale_change_loss` on the
            change field (v vs u_t). ``None`` or ``[1.0]`` = plain full-resolution
            MSE. Extra coarse scales (e.g. ``[1,1,1]``) put gradient on the coherent
            edit and away from the high-frequency registration speckle — the
            "learn the real detail" term. Bridge only.
        edge_downweight: γ for :func:`edge_downweight_map`, softly discounting the
            regression on x_pre's structural edges (where misregistration fakes
            change). Applied at the full-resolution scale. 0 = off.
        time_scale: t is multiplied by this before the sinusoidal embedding, so
            a t ∈ [0,1] lands in the same numeric range the U-Net's embedding
            was designed for (integer timesteps 0…1000).
    """

    def __init__(
        self,
        net:           nn.Module,
        source:        str   = "pre",
        condition:     str   = "concat",
        sigma:         float = 0.3,
        t_dist:        str   = "uniform",
        logit_mean:    float = 0.0,
        logit_std:     float = 1.0,
        change_weight: float = 5.0,
        l1_weight:     float = 0.0,
        ssim_weight:   float = 0.0,
        cfg_drop_prob: float = 0.0,
        gated:         bool  = False,
        det_weight:    float = 0.0,
        det_sigma:     float = 2.0,
        ms_weights:    tuple = None,
        edge_downweight: float = 0.0,
        steps:         int   = 8,
        solver:        str   = "heun",
        time_scale:    float = 1000.0,
    ):
        super().__init__()
        if source not in ("pre", "noise"):
            raise ValueError(f"source must be 'pre' or 'noise', got {source!r}")
        if condition not in ("none", "concat"):
            raise ValueError(f"condition must be 'none' or 'concat', got {condition!r}")
        if source == "noise" and condition == "none":
            raise ValueError(
                "source='noise' with condition='none' is unconditional generation: "
                "x_t starts as pure noise, so the network would never see x_pre and "
                "could not know which subject to predict.")
        self.net           = net
        self.source        = source
        self.condition     = condition
        self.sigma         = float(sigma)
        self.t_dist        = t_dist
        self.logit_mean    = float(logit_mean)
        self.logit_std     = float(logit_std)
        self.change_weight = float(change_weight)
        self.l1_weight     = float(l1_weight)
        self.ssim_weight   = float(ssim_weight)
        self.cfg_drop_prob = float(cfg_drop_prob)
        self.gated         = bool(gated)
        self.det_weight    = float(det_weight)
        self.det_sigma     = float(det_sigma)
        self.ms_weights    = tuple(ms_weights) if ms_weights else None
        self.edge_downweight = float(edge_downweight)
        self.steps         = int(steps)
        self.solver        = solver
        self.time_scale    = float(time_scale)

    # ── training ─────────────────────────────────────────────────────────────

    def flow_loss(self, x_post: torch.Tensor, x_pre: torch.Tensor,
                  return_components: bool = False):
        """Conditional flow matching loss (official ConditionalFlowMatcher).

        Regresses v_θ onto the conditional velocity u_t = x1 - x0 of the Gaussian
        path x_t = (1-t)·x0 + t·x1 + σ·ε. The minimiser is the marginal velocity
        E[u_t | x_t, cond] that transports the source to the target — that
        equivalence is the whole point of flow matching, and it is why the
        unobservable ε does not bias the result.

        On top of the base regression this method carries the coherent-change
        objective (bridge only), each term optional:

          * **multi-scale regression** (``ms_weights``) on the change field v vs u_t
            — scores coarse (low-frequency) structure as well as full resolution, so
            the gradient lands on the *predictable* coherent edit rather than the
            high-frequency registration speckle a single-scale MSE chases.
          * **edge down-weighting** (``edge_downweight``) — discounts the
            full-resolution error on x_pre's structural edges, where a sub-voxel
            pre/post misalignment fakes change.
          * **change detection** (``det_weight``, gated net only) — a soft-Dice
            between the gate map and :func:`coherent_change_target`, teaching the
            network WHERE this subject changes as an imbalance-robust detection task.

        With a plain net, ``ms_weights`` unset and ``change_weight`` the only knob,
        this reduces exactly to the previous change-weighted MSE.

        Returns the scalar loss, or ``(loss, components)`` with the per-term values
        for logging when ``return_components``.
        """
        B, device = x_post.size(0), x_post.device
        t = sample_t(B, device, self.t_dist, self.logit_mean, self.logit_std)

        # Confines the bridge noise to signal voxels. Only bites under
        # zero_background=True (exact-zero background); with the default
        # whole-volume normalisation this is every voxel (see module docstring).
        mask = ((x_pre != 0) | (x_post != 0)).to(x_post.dtype)

        x0 = x_pre if self.source == "pre" else torch.randn_like(x_post)

        # σ·ε, masked to the brain for the bridge so the background stays 0.
        eps = torch.randn_like(x_post)
        if self.source == "pre":
            eps = eps * mask
        tb = _b(t)

        x_t = (1.0 - tb) * x0 + tb * x_post + self.sigma * eps
        u_t = x_post - x0                                   # clean official target

        if self.condition == "concat":
            x_pre_in = x_pre
            if self.cfg_drop_prob > 0.0:
                keep = (torch.rand(B, device=device) >= self.cfg_drop_prob).to(x_pre.dtype)
                x_pre_in = x_pre * _b(keep)
            net_in = torch.cat([x_pre_in, x_t], dim=1)
        else:
            net_in = x_t

        # Gated net also returns the change-localisation map for the detection term.
        gate = None
        if self.gated:
            gate, v = self.net.gate_velocity(net_in, t * self.time_scale)
        else:
            v = self.net(net_in, t * self.time_scale)

        # ── regression on the change field (v vs u_t) ─────────────────────────
        # Per-voxel full-resolution weight = edge down-weight × change-weight (both
        # optional). change_weight_map is kept for backward compatibility but the
        # coherent-change model leaves it at 0 — up-weighting the raw |Δ| tail is
        # what made the old model chase noise; the multi-scale terms are the fix.
        bridge = self.source == "pre"
        voxel_w = None
        if bridge and self.edge_downweight > 0.0:
            voxel_w = edge_downweight_map(x_pre, self.edge_downweight)
        if bridge and self.change_weight > 0.0:
            cw = change_weight_map(x_pre, x_post, self.change_weight)
            voxel_w = cw if voxel_w is None else voxel_w * cw

        if self.ms_weights:
            reg = multiscale_change_loss(v, u_t, self.ms_weights, voxel_w)
        else:
            se = (v - u_t).pow(2)
            reg = se.mean() if voxel_w is None else (voxel_w * se).mean()
        loss = reg
        comp = {"reg": float(reg.detach())}

        if self.l1_weight > 0.0:
            ae = (v - u_t).abs()
            l1 = ae.mean() if voxel_w is None else (voxel_w * ae).mean()
            loss = loss + self.l1_weight * l1
            comp["l1"] = float(l1.detach())

        if self.ssim_weight > 0.0:
            # x̂₁ from a single Euler step to t=1: x_t + (1-t)·v. This is the
            # model's current estimate of the target, so a structural (SSIM)
            # term can be applied to it even though the network's own output is
            # a velocity. Exact when the path is straight, which for the bridge
            # it very nearly is.
            x1_hat = x_t + (1.0 - tb) * v
            s = ssim3d_loss(x1_hat, x_post)
            loss = loss + self.ssim_weight * s
            comp["ssim"] = float(s.detach())

        # ── change detection (learn WHERE) ────────────────────────────────────
        if bridge and self.gated and self.det_weight > 0.0:
            target = coherent_change_target(x_pre, x_post, self.det_sigma)
            det = soft_dice_loss(gate.float(), target)
            loss = loss + self.det_weight * det
            comp["det"] = float(det.detach())

        return (loss, comp) if return_components else loss

    def forward(self, x_post: torch.Tensor, x_pre: torch.Tensor) -> torch.Tensor:
        return self.flow_loss(x_post, x_pre)

    # ── adversarial detail term ──────────────────────────────────────────────

    def onestep_prediction(self, x_pre: torch.Tensor) -> torch.Tensor:
        """One Euler step from t=0: x̂_post = x_pre + v_θ(x_pre, 0 | x_pre).

        The cheapest *differentiable* estimate of x_post, and the right image to
        hand a discriminator. The interpolant estimate x̂₁ = x_t + (1−t)·v used by
        the SSIM term is ≈ x_post as t→1 — a trivially-real "fake" that would
        destabilise the GAN — whereas this is a genuine prediction at every stage
        of training and is exactly the first step the sampler takes, so pushing it
        to look real sharpens the trajectory itself. Bridge only (``source="pre"``).
        """
        if self.source != "pre":
            raise ValueError("onestep_prediction is only defined for source='pre'")
        t0 = torch.zeros(x_pre.size(0), device=x_pre.device)
        net_in = torch.cat([x_pre, x_pre], dim=1) if self.condition == "concat" else x_pre
        v = self.net(net_in, t0 * self.time_scale)
        return x_pre + v

    @torch.no_grad()
    def predict_change_map(self, x_pre: torch.Tensor) -> torch.Tensor:
        """The gate at t=0: the model's prediction of WHERE this subject changes.

        For a gated net this is ``sigmoid(gate_logit)`` evaluated with x_t=x_pre —
        the honest "from the pre-op scan alone, where will surgery alter the
        perfusion" map, and the localisation used at the first sampling step. Useful
        for visual/quantitative validation of the detection head. Returns ``None``
        for a non-gated model (no explicit gate exists).
        """
        if not self.gated:
            return None
        t0 = torch.zeros(x_pre.size(0), device=x_pre.device)
        net_in = torch.cat([x_pre, x_pre], dim=1) if self.condition == "concat" else x_pre
        gate, _ = self.net.gate_velocity(net_in, t0 * self.time_scale)
        return gate

    # ── sampling ─────────────────────────────────────────────────────────────

    def _velocity(
        self,
        x:              torch.Tensor,
        t:              torch.Tensor,
        x_pre:          torch.Tensor,
        guidance_scale: float,
        zero_cond:      torch.Tensor,
    ) -> torch.Tensor:
        if self.condition == "none":
            return self.net(x, t * self.time_scale).float()
        v = self.net(torch.cat([x_pre, x], dim=1), t * self.time_scale).float()
        if guidance_scale != 1.0:
            v_u = self.net(torch.cat([zero_cond, x], dim=1), t * self.time_scale).float()
            v = v_u + guidance_scale * (v - v_u)
        return v

    @torch.no_grad()
    def sample(
        self,
        x_pre:          torch.Tensor,
        steps:          int   = None,
        solver:         str   = None,
        guidance_scale: float = 1.0,
        init_noise:     float = 0.0,
        generator:      torch.Generator = None,
        return_traj:    bool  = False,
    ) -> torch.Tensor:
        """Integrate dx/dt = v_θ(x, t | x_pre) from t=0 to t=1.

        Args:
            steps: number of integration steps. NFE is ``steps`` for euler,
                ``2·steps`` for heun, ``4·steps`` for rk4.
            solver: ``euler`` (1st order), ``heun`` (2nd, the default — best
                accuracy per NFE here), or ``rk4`` (4th).
            guidance_scale: classifier-free guidance on the velocity. Only
                coherent for ``source="noise"`` models trained with
                ``cfg_drop_prob > 0``.
            init_noise: std of Gaussian perturbation added to x(0), masked to
                the brain. The ODE is deterministic given its start, so this is
                the knob that produces an ensemble of distinct predictions.
            return_traj: also return the list of intermediate states.

        Returns the predicted x_post. The final ``x * brain`` only removes the
        background under ``zero_background=True`` inputs (where the background is
        exact 0); with the default whole-volume normalisation ``brain`` is every
        voxel, so the full volume is returned — as ``cdm3d.sample`` does.
        """
        steps  = int(steps or self.steps)
        solver = solver or self.solver

        # fp32 throughout: integration accumulates over steps and the whole
        # point of the low-NFE claim is that the solver error, not the dtype,
        # is what limits accuracy.
        with torch.autocast(device_type=x_pre.device.type, enabled=False):
            x_pre = x_pre.float()
            device = x_pre.device
            brain = (x_pre != 0).float()

            if self.source == "pre":
                x = x_pre.clone()
            else:
                x = torch.randn(x_pre.shape, device=device, generator=generator,
                                dtype=x_pre.dtype)

            if init_noise > 0.0:
                n = torch.randn(x.shape, device=device, generator=generator, dtype=x.dtype)
                x = x + init_noise * n * (brain if self.source == "pre" else 1.0)

            zero_cond = torch.zeros_like(x_pre)
            ts = torch.linspace(0.0, 1.0, steps + 1, device=device)
            traj = [x.clone()] if return_traj else None

            for i in range(steps):
                t0, t1 = ts[i], ts[i + 1]
                h = (t1 - t0)
                B = x.shape[0]
                tt0 = t0.expand(B)

                if solver == "euler":
                    x = x + h * self._velocity(x, tt0, x_pre, guidance_scale, zero_cond)

                elif solver == "heun":
                    v0 = self._velocity(x, tt0, x_pre, guidance_scale, zero_cond)
                    x_e = x + h * v0
                    v1 = self._velocity(x_e, t1.expand(B), x_pre, guidance_scale, zero_cond)
                    x = x + h * 0.5 * (v0 + v1)

                elif solver == "rk4":
                    tm = (t0 + 0.5 * h).expand(B)
                    k1 = self._velocity(x, tt0, x_pre, guidance_scale, zero_cond)
                    k2 = self._velocity(x + 0.5 * h * k1, tm, x_pre, guidance_scale, zero_cond)
                    k3 = self._velocity(x + 0.5 * h * k2, tm, x_pre, guidance_scale, zero_cond)
                    k4 = self._velocity(x + h * k3, t1.expand(B), x_pre, guidance_scale, zero_cond)
                    x = x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

                else:
                    raise ValueError(f"unknown solver: {solver!r} (euler|heun|rk4)")

                if return_traj:
                    traj.append(x.clone())

            x = x * brain
            return (x, traj) if return_traj else x

    @torch.no_grad()
    def sample_sde(
        self,
        x_pre:          torch.Tensor,
        steps:          int   = None,
        gamma_scale:    float = 1.0,
        guidance_scale: float = 1.0,
        generator:      torch.Generator = None,
        return_traj:    bool  = False,
    ) -> torch.Tensor:
        """Stochastic sampling of the model's *own* probability path (bridge only).

        The principled ensemble knob. The ODE in :meth:`sample` is deterministic,
        so an ensemble needs an arbitrary ``init_noise`` perturbation; here the
        stochasticity is the trained path's own σ instead. The model was trained
        on the Gaussian path x_t = (1−t)·x_pre + t·x_post + σ·ε, whose score is
        recoverable from the learned velocity (Gaussian-path identity, cf.
        stochastic interpolants, Albergo & Vanden-Eijnden 2023):

            E[x_post | x, t] = x_pre + v_θ(x, t)
            E[μ_t   | x, t] = (1−t)·x_pre + t·E[x_post | x, t] = x_pre + t·v_θ
            s_θ(x, t) = ∇ log p_t(x | x_pre) = −(x − x_pre − t·v_θ(x, t)) / σ²

        Any Langevin-corrected SDE  dx = [v_θ + γ·s_θ] dt + √(2γ) dW  then shares
        the ODE's time marginals p_t, for any γ ≥ 0 — the flow-matching analogue
        of sampling a DDPM with different seeds. Concretely (Euler–Maruyama):

            x(0) = x_pre + σ·ε          ← p_0 of the training path, not a knob
            per step: x += h·(v + γ·s) + √(2γh)·ε,   γ = gamma_scale · σ²
            return x_pre + v_θ(x(1), 1)  ← denoised endpoint, E[x_post | x(1)]

        The final denoise matters: p_1 is the target *convolved with* N(0, σ²),
        so the raw endpoint carries an additive σ floor that is path noise, not
        predictive uncertainty; conditioning it out via the t=1 velocity leaves
        the spread that comes from the velocity field responding differently
        along different trajectories. ``gamma_scale=1`` (γ = σ²) relaxes toward
        the path marginal with unit rate over the whole run — the natural
        default; 0 recovers the ODE from a σ-perturbed start. NFE = steps + 1.

        Bridge only (``source="pre"``): the score identity above needs the known
        endpoint x_pre; for ``source="noise"`` E[x0 | x] is not recoverable from
        u = x1 − x0 alone. Requires the σ > 0 the bridge is trained with anyway.
        Noise is brain-masked exactly as the training σ·ε is.
        """
        if self.source != "pre":
            raise ValueError("sample_sde is only defined for source='pre'")
        if self.sigma <= 0:
            raise ValueError("sample_sde needs the trained σ > 0 "
                             "(the score is −residual/σ²)")
        steps = int(steps or self.steps)

        with torch.autocast(device_type=x_pre.device.type, enabled=False):
            x_pre = x_pre.float()
            device = x_pre.device
            brain = (x_pre != 0).float()
            sig2 = self.sigma ** 2
            gamma = float(gamma_scale) * sig2
            B = x_pre.shape[0]

            def noise():
                n = torch.randn(x_pre.shape, device=device, generator=generator,
                                dtype=x_pre.dtype)
                return n * brain

            x = x_pre + self.sigma * noise()
            zero_cond = torch.zeros_like(x_pre)
            ts = torch.linspace(0.0, 1.0, steps + 1, device=device)
            traj = [x.clone()] if return_traj else None

            for i in range(steps):
                t0, t1 = ts[i], ts[i + 1]
                h = t1 - t0
                v = self._velocity(x, t0.expand(B), x_pre, guidance_scale,
                                   zero_cond)
                if gamma > 0:
                    score = -(x - x_pre - t0 * v) / sig2
                    x = (x + h * (v + gamma * score)
                         + torch.sqrt(2 * gamma * h) * noise())
                else:
                    x = x + h * v
                if return_traj:
                    traj.append(x.clone())

            v1 = self._velocity(x, ts[-1].expand(B), x_pre, guidance_scale,
                                zero_cond)
            x = (x_pre + v1) * brain
            return (x, traj) if return_traj else x

    @torch.no_grad()
    def generate(
        self,
        x_pre:          torch.Tensor,
        steps:          int   = None,
        solver:         str   = None,
        guidance_scale: float = 1.0,
        init_noise:     float = 0.0,
        **kwargs,
    ) -> torch.Tensor:
        """Alias of :meth:`sample` matching the other models' ``generate`` name.

        Accepts and ignores ``ddim_steps`` so the shared training/eval plumbing
        written for the diffusion models can call this unchanged.
        """
        if steps is None and "ddim_steps" in kwargs:
            steps = kwargs["ddim_steps"]
        return self.sample(x_pre, steps=steps, solver=solver,
                           guidance_scale=guidance_scale, init_noise=init_noise)


# ─────────────────────────────────────────────────────────────────────────────
# Conditional 3D PatchGAN discriminator (for the adversarial detail term)
# ─────────────────────────────────────────────────────────────────────────────

class PatchDiscriminator3D(nn.Module):
    """Conditional 3D PatchGAN, spectral-normalised.

    Judges realism *locally*: strided convs shrink the volume to a grid of logits,
    one per receptive-field patch, so the adversarial gradient rewards realistic
    high-frequency texture patch-by-patch rather than making one global real/fake
    call — the property that makes PatchGANs sharpen fine detail (Isola et al.
    2017, pix2pix). It is conditioned on x_pre by channel concatenation, so it
    scores "is this a plausible post-op volume *for this pre-op input*", pushing
    the plausibility of the surgical edit and not mere image realism.

    Spectral norm on every conv bounds the Lipschitz constant — the main thing
    keeping a GAN stable on ~200 volumes — and is the only normalisation, to avoid
    interacting with a second one. Deliberately shallow/narrow for the data size.
    """

    def __init__(self, in_channels: int = 2, dim: int = 32, n_layers: int = 3):
        super().__init__()
        sn = nn.utils.spectral_norm
        layers = [sn(nn.Conv3d(in_channels, dim, 4, stride=2, padding=1)),
                  nn.LeakyReLU(0.2, inplace=True)]
        ch = dim
        for _ in range(1, n_layers):                       # more stride-2 blocks
            nch = min(ch * 2, dim * 8)
            layers += [sn(nn.Conv3d(ch, nch, 4, stride=2, padding=1)),
                       nn.LeakyReLU(0.2, inplace=True)]
            ch = nch
        nch = min(ch * 2, dim * 8)                          # stride-1 head → logits
        layers += [sn(nn.Conv3d(ch, nch, 4, stride=1, padding=1)),
                   nn.LeakyReLU(0.2, inplace=True),
                   sn(nn.Conv3d(nch, 1, 4, stride=1, padding=1))]
        self.net = nn.Sequential(*layers)

    def forward(self, x_pre: torch.Tensor, img: torch.Tensor) -> torch.Tensor:
        """(x_pre, img) → patch logits. Both (B,1,D,H,W); concatenated to 2 ch."""
        return self.net(torch.cat([x_pre, img], dim=1))


def build_discriminator3d(dim: int = 32, n_layers: int = 3,
                          in_channels: int = 2) -> PatchDiscriminator3D:
    """Build the conditional 3D PatchGAN for the adversarial detail term."""
    return PatchDiscriminator3D(in_channels=in_channels, dim=dim, n_layers=n_layers)


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def build_flow3d(
    # velocity network (the 7TCDM-derived U-Net, reused verbatim from cdm3d)
    dim:              int   = 32,
    dim_mults:        tuple = (1, 2, 4, 8),
    init_kernel_size: int   = 7,
    resnet_groups:    int   = 8,
    zero_init_out:    bool  = True,
    # flow
    source:           str   = "pre",
    condition:        str   = "concat",
    sigma:            float = 0.3,
    t_dist:           str   = "uniform",
    logit_mean:       float = 0.0,
    logit_std:        float = 1.0,
    change_weight:    float = 5.0,
    l1_weight:        float = 0.0,
    ssim_weight:      float = 0.0,
    cfg_drop_prob:    float = 0.0,
    gated:            bool  = False,
    det_weight:       float = 0.0,
    det_sigma:        float = 2.0,
    ms_weights:       tuple = None,
    edge_downweight:  float = 0.0,
    gate_init_bias:   float = -2.0,
    steps:            int   = 8,
    solver:           str   = "heun",
) -> ConditionalFlowMatching3D:
    """Build the image-space conditional flow matching model.

    The velocity network is ``Unet3D_CDM`` — the same architecture CDM3D uses as
    its denoiser — so a CDM3D-vs-Flow3D comparison isolates the objective and
    the sampler. Its single output channel is read as a velocity rather than an
    x₀ estimate.

    ``condition="concat"`` is the default for both sources: the network is
    conditioned on x_pre. This is what makes training descend (see the module
    docstring); ``condition="none"`` is kept only for ablation.

    ``gated=True`` gives the backbone a second output channel and wraps it in a
    :class:`GatedVelocityNet` (v = gate·residual), so the model factorises WHERE it
    edits (the supervisable gate) from WHAT it edits (the residual). The gate is
    then trained by the ``det_weight`` detection loss.
    """
    if condition is None:
        condition = "concat"

    in_ch = 2 if condition == "concat" else 1
    backbone = Unet3D_CDM(
        dim=dim,
        dim_mults=dim_mults,
        in_channels=in_ch,
        out_channels=2 if gated else 1,        # (gate_logit, residual) when gated
        init_kernel_size=init_kernel_size,
        resnet_groups=resnet_groups,
    )

    if zero_init_out:
        # Zero the final 1×1×1 conv so v_θ ≡ 0 at initialisation. For the bridge
        # this makes the untrained model return exactly x_pre — training starts
        # at the identity baseline instead of at noise, and the first gradient
        # steps only ever have to explain the residual. For the gated net the
        # residual channel is 0 (→ v=0) regardless of the gate; the gate bias is
        # set negative so the gate starts sparse (mostly "no change"), a sensible
        # prior for a localisation map over a near-identity volume.
        last = backbone.final_conv0[-1]
        nn.init.zeros_(last.weight)
        if last.bias is not None:
            nn.init.zeros_(last.bias)
            if gated:
                last.bias.data[0] = float(gate_init_bias)   # gate logit
                last.bias.data[1] = 0.0                      # residual

    net = GatedVelocityNet(backbone) if gated else backbone

    return ConditionalFlowMatching3D(
        net=net,
        source=source,
        condition=condition,
        sigma=sigma,
        t_dist=t_dist,
        logit_mean=logit_mean,
        logit_std=logit_std,
        change_weight=change_weight,
        l1_weight=l1_weight,
        ssim_weight=ssim_weight,
        cfg_drop_prob=cfg_drop_prob,
        gated=gated,
        det_weight=det_weight,
        det_sigma=det_sigma,
        ms_weights=ms_weights,
        edge_downweight=edge_downweight,
        steps=steps,
        solver=solver,
    )


# ── checkpoint loading ────────────────────────────────────────────────────────

_BUILD_KEYS = (
    "dim", "dim_mults", "init_kernel_size", "resnet_groups", "zero_init_out",
    "source", "condition", "sigma", "t_dist", "logit_mean", "logit_std",
    "change_weight", "l1_weight", "ssim_weight", "cfg_drop_prob",
    "gated", "det_weight", "det_sigma", "ms_weights", "edge_downweight",
    "steps", "solver",
)


def build_from_args(args: dict, device="cpu") -> ConditionalFlowMatching3D:
    """Rebuild the exact model variant from a checkpoint's ``args`` dict."""
    kw = {k: args[k] for k in _BUILD_KEYS if k in args}
    if "dim_mults" in kw:
        kw["dim_mults"] = tuple(kw["dim_mults"])
    if kw.get("ms_weights"):
        kw["ms_weights"] = tuple(kw["ms_weights"])
    # Checkpoints predating `condition` always concatenated x_pre.
    kw.setdefault("condition", "concat")
    # zero_init_out only affects initialisation; weights are loaded over it.
    return build_flow3d(**kw).to(device)


def load_flow3d_checkpoint(ckpt_path, device="cpu", use_ema: bool = True):
    """Load a Flow3D checkpoint → ``(model, raw)``, model in eval mode.

    Prefers the EMA weights when present (``use_ema``), which is what the
    training script validates and reports against.
    """
    raw = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_from_args(raw["args"], device=device)
    state = raw["ema"] if (use_ema and raw.get("ema") is not None) else raw["model"]
    model.net.load_state_dict(state)
    model.eval()
    return model, raw
