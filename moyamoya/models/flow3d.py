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
        self.steps         = int(steps)
        self.solver        = solver
        self.time_scale    = float(time_scale)

    # ── training ─────────────────────────────────────────────────────────────

    def flow_loss(self, x_post: torch.Tensor, x_pre: torch.Tensor) -> torch.Tensor:
        """Conditional flow matching loss (official ConditionalFlowMatcher).

        Regresses v_θ onto the conditional velocity u_t = x1 - x0 of the Gaussian
        path x_t = (1-t)·x0 + t·x1 + σ·ε. The minimiser is the marginal velocity
        E[u_t | x_t, cond] that transports the source to the target — that
        equivalence is the whole point of flow matching, and it is why the
        unobservable ε does not bias the result.

        The squared (and optional L1) residual is weighted per voxel by
        :func:`change_weight_map` (γ = ``self.change_weight``) so the sparse
        surgical edits are not averaged away by the near-identity majority — the
        cause of the copy-x_pre collapse. The weight averages to 1 per sample, so
        with γ=0 this is exactly ``F.mse_loss(v, u_t)``.
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

        v = self.net(net_in, t * self.time_scale)

        # Change-weighted regression. w up-weights the voxels that actually differ
        # between pre and post so the sparse edits are not averaged away by the
        # near-identity majority (the copy-x_pre collapse). Only for the bridge,
        # where Δ = x_post − x_pre is literally the target velocity; γ=0 or
        # source="noise" fall back to the flat MSE. w averages to 1 per sample, so
        # (w·se).mean() matches F.mse_loss(v, u_t) when γ=0.
        w = (change_weight_map(x_pre, x_post, self.change_weight)
             if (self.change_weight > 0.0 and self.source == "pre") else None)

        diff = v - u_t
        se = diff.pow(2)
        loss = se.mean() if w is None else (w * se).mean()
        if self.l1_weight > 0.0:
            ae = diff.abs()
            loss = loss + self.l1_weight * (ae.mean() if w is None else (w * ae).mean())

        if self.ssim_weight > 0.0:
            # x̂₁ from a single Euler step to t=1: x_t + (1-t)·v. This is the
            # model's current estimate of the target, so a structural (SSIM)
            # term can be applied to it even though the network's own output is
            # a velocity. Exact when the path is straight, which for the bridge
            # it very nearly is.
            x1_hat = x_t + (1.0 - tb) * v
            loss = loss + self.ssim_weight * ssim3d_loss(x1_hat, x_post)

        return loss

    def forward(self, x_post: torch.Tensor, x_pre: torch.Tensor) -> torch.Tensor:
        return self.flow_loss(x_post, x_pre)

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
    """
    if condition is None:
        condition = "concat"

    net = Unet3D_CDM(
        dim=dim,
        dim_mults=dim_mults,
        in_channels=2 if condition == "concat" else 1,
        out_channels=1,
        init_kernel_size=init_kernel_size,
        resnet_groups=resnet_groups,
    )

    if zero_init_out:
        # Zero the final 1×1×1 conv so v_θ ≡ 0 at initialisation. For the bridge
        # this makes the untrained model return exactly x_pre — training starts
        # at the identity baseline instead of at noise, and the first gradient
        # steps only ever have to explain the residual.
        last = net.final_conv0[-1]
        nn.init.zeros_(last.weight)
        if last.bias is not None:
            nn.init.zeros_(last.bias)

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
        steps=steps,
        solver=solver,
    )


# ── checkpoint loading ────────────────────────────────────────────────────────

_BUILD_KEYS = (
    "dim", "dim_mults", "init_kernel_size", "resnet_groups", "zero_init_out",
    "source", "condition", "sigma", "t_dist", "logit_mean", "logit_std",
    "change_weight", "l1_weight", "ssim_weight", "cfg_drop_prob", "steps", "solver",
)


def build_from_args(args: dict, device="cpu") -> ConditionalFlowMatching3D:
    """Rebuild the exact model variant from a checkpoint's ``args`` dict."""
    kw = {k: args[k] for k in _BUILD_KEYS if k in args}
    if "dim_mults" in kw:
        kw["dim_mults"] = tuple(kw["dim_mults"])
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
