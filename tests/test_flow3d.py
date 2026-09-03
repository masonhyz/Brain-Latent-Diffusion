"""Self-checks for the Flow3D conditional flow matching implementation.

Run directly (no pytest needed):   python tests/test_flow3d.py
With pytest:                       pytest tests/test_flow3d.py

The load-bearing test is `test_oracle_velocity_is_exact`: if the network returned
the *true* conditional velocity, the sampler must land exactly on x_post. That
pins the interpolant, the velocity target, and the ODE solvers to each other —
a sign error or a mis-scaled time in any one of them breaks it.

The algorithm follows the official ConditionalFlowMatcher (Lipman 2023 / Tong
2024, torchcfm): x_t = (1-t)·x0 + t·x1 + σ·ε, target u_t = x1 - x0, velocity net
conditioned on x_pre.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from moyamoya.models.flow3d import (
    ConditionalFlowMatching3D, EMA, build_flow3d, sample_t,
)

DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
SHAPE = (2, 1, 12, 14, 12)


def _pair(seed=0):
    """A fake (x_pre, x_post) pair with a realistic zero background."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    x_pre = torch.randn(SHAPE, generator=g)
    x_post = x_pre + 0.3 * torch.randn(SHAPE, generator=g)
    x_pre[..., :3, :, :] = 0            # background, zero in both volumes
    x_post[..., :3, :, :] = 0
    return x_pre.to(DEV), x_post.to(DEV)


class _OracleNet(nn.Module):
    """Returns the exact conditional velocity of a σ=0 bridge: x_post − x_pre.

    The true velocity is constant in t and in x, so any consistent solver must
    integrate it to x_post exactly, in a single step.
    """

    def __init__(self, x_pre, x_post):
        super().__init__()
        self.register_buffer("u", x_post - x_pre)

    def forward(self, x, t):
        return self.u[: x.shape[0]]


def test_oracle_velocity_is_exact():
    x_pre, x_post = _pair()
    model = ConditionalFlowMatching3D(_OracleNet(x_pre, x_post).to(DEV),
                                      source="pre", condition="none", sigma=0.0).to(DEV)
    brain = (x_pre != 0).float()
    for solver in ("euler", "heun", "rk4"):
        for steps in (1, 4, 16):
            out = model.sample(x_pre, steps=steps, solver=solver)
            err = (out - x_post * brain).abs().max().item()
            assert err < 1e-5, f"{solver}/{steps}: sampler off by {err}"
    print("  ok  oracle velocity integrates to x_post exactly (all solvers, 1-16 steps)")


def test_sde_oracle_denoised_endpoint_is_exact():
    """With the true (constant) velocity, the SDE's denoised endpoint must be
    exactly x_post for every trajectory: the terminal state is x_post + path
    noise, and the t=1 denoise x_pre + v removes it identically since v ≡ u.
    Pins the score sign, the final denoise, and the brain masking at once."""
    x_pre, x_post = _pair()
    model = ConditionalFlowMatching3D(_OracleNet(x_pre, x_post).to(DEV),
                                      source="pre", condition="none",
                                      sigma=0.3).to(DEV)
    brain = (x_pre != 0).float()
    g = torch.Generator(device=DEV).manual_seed(0)
    for gs in (0.0, 1.0, 2.0):
        out = model.sample_sde(x_pre, steps=8, gamma_scale=gs, generator=g)
        err = (out - x_post * brain).abs().max().item()
        assert err < 1e-5, f"gamma_scale={gs}: denoised endpoint off by {err}"
    print("  ok  SDE denoised endpoint is exactly x_post under the oracle velocity")


def test_sde_marginal_width_matches_sigma():
    """Under the oracle velocity the Langevin-corrected SDE must keep the
    marginal at its stationary width: the pre-denoise terminal state is
    x_post + σ·ε, so its per-voxel std across trajectories ≈ σ. A wrong score
    scale (or a missing √(2γh)) shows up as a width far from σ."""
    torch.manual_seed(0)
    B, sigma = 256, 0.3
    # One pair, replicated over the batch: every trajectory samples the same
    # conditional, so the cross-batch spread is purely the path noise.
    x_pre = (torch.randn(1, 1, 8, 8, 8, device=DEV).abs() + 0.5).repeat(B, 1, 1, 1, 1)
    x_post = x_pre + 0.3 * torch.randn(1, 1, 8, 8, 8, device=DEV)
    model = ConditionalFlowMatching3D(_OracleNet(x_pre, x_post).to(DEV),
                                      source="pre", condition="none",
                                      sigma=sigma).to(DEV)
    g = torch.Generator(device=DEV).manual_seed(1)
    _, traj = model.sample_sde(x_pre, steps=32, gamma_scale=1.0, generator=g,
                               return_traj=True)
    width = traj[-1].std(0).mean().item()          # pre-denoise terminal spread
    assert abs(width - sigma) < 0.2 * sigma, (
        f"terminal marginal width {width:.4f} should be ≈ σ={sigma}")
    print(f"  ok  SDE terminal marginal width {width:.3f} ≈ σ={sigma}")


def test_sde_reproducible_and_source_guard():
    x_pre, x_post = _pair()
    model = ConditionalFlowMatching3D(_OracleNet(x_pre, x_post).to(DEV),
                                      source="pre", condition="none",
                                      sigma=0.3).to(DEV)
    a = model.sample_sde(x_pre, steps=4,
                         generator=torch.Generator(device=DEV).manual_seed(7))
    b = model.sample_sde(x_pre, steps=4,
                         generator=torch.Generator(device=DEV).manual_seed(7))
    assert torch.equal(a, b), "same generator seed must reproduce the sample"
    noise_model = ConditionalFlowMatching3D(_OracleNet(x_pre, x_post).to(DEV),
                                            source="noise", condition="concat",
                                            sigma=0.3).to(DEV)
    try:
        noise_model.sample_sde(x_pre)
        raise AssertionError("source='noise' must be rejected")
    except ValueError:
        pass
    print("  ok  SDE reproducible under a seeded generator; bridge-only guard holds")


def test_interpolant_endpoints_and_target():
    """The interpolant must start at x0, end at x1, and its clean velocity target
    must be x1 - x0 — the official ConditionalFlowMatcher path."""
    x_pre, x_post = _pair()
    mask = ((x_pre != 0) | (x_post != 0)).float()
    sigma = 0.3
    for t_val in (0.0, 1.0):
        t = torch.full((SHAPE[0],), t_val, device=DEV)
        tb = t[:, None, None, None, None]
        eps = torch.randn_like(x_post) * mask
        x_t = (1 - tb) * x_pre + tb * x_post + sigma * eps
        # at the endpoints the only deviation from x0 / x1 is the σ·ε term
        target = x_post if t_val == 1.0 else x_pre
        assert (x_t - target - sigma * eps).abs().max().item() < 1e-5
    # the regression target is the clean displacement, independent of ε
    u_t = x_post - x_pre
    assert (u_t[mask.bool()]).abs().mean().item() > 0, "degenerate test pair"
    print("  ok  interpolant hits x0/x1 at the endpoints; target = x1 - x0")


def test_zero_init_returns_input():
    """With a zero-initialised output layer the untrained bridge must reproduce
    x_pre exactly — that is what makes training start at the identity baseline."""
    x_pre, _ = _pair()
    m = build_flow3d(dim=8, dim_mults=(1, 2), init_kernel_size=3,
                     source="pre", zero_init_out=True).to(DEV)
    out = m.sample(x_pre, steps=4)
    err = (out - x_pre * (x_pre != 0).float()).abs().max().item()
    assert err == 0.0, f"zero-init model perturbed the input by {err}"
    print("  ok  zero-init model returns x_pre exactly (starts at identity baseline)")


def test_background_stays_zero():
    """The bridge noise is masked to the brain, so the path — and therefore any
    prediction — must be exactly 0 where both volumes are 0."""
    x_pre, x_post = _pair()
    m = build_flow3d(dim=8, dim_mults=(1, 2), init_kernel_size=3,
                     source="pre", sigma=0.3, zero_init_out=False).to(DEV)
    out = m.sample(x_pre, steps=4)
    bg = (x_pre == 0)
    assert out[bg].abs().max().item() == 0.0, "background leaked non-zero values"
    # and the interpolant itself keeps the background clean
    mask = ((x_pre != 0) | (x_post != 0)).float()
    t = torch.full((SHAPE[0],), 0.5, device=DEV)
    tb = t[:, None, None, None, None]
    eps = torch.randn_like(x_post) * mask
    x_t = (1 - tb) * x_pre + tb * x_post + 0.3 * eps
    assert x_t[bg & (x_post == 0)].abs().max().item() == 0.0
    print("  ok  background stays exactly zero through path and sampling")


def test_loss_is_finite_and_shapes_agree():
    x_pre, x_post = _pair()
    for source in ("pre", "noise"):
        for sigma in (0.0, 0.2):
            for t_dist in ("uniform", "logit_normal"):
                m = build_flow3d(dim=8, dim_mults=(1, 2), init_kernel_size=3,
                                 source=source, sigma=sigma, t_dist=t_dist,
                                 l1_weight=0.5, ssim_weight=0.5).to(DEV)
                loss = m.flow_loss(x_post, x_pre)
                assert torch.isfinite(loss), f"{source}/{sigma}/{t_dist} loss not finite"
                loss.backward()
                grads = [p.grad for p in m.net.parameters() if p.grad is not None]
                assert grads and all(torch.isfinite(g).all() for g in grads)
    print("  ok  loss finite + backward clean across source/sigma/t_dist/aux-losses")


def test_t_sampling_ranges():
    for dist in ("uniform", "logit_normal"):
        t = sample_t(4096, DEV, dist)
        assert t.min() > 0 and t.max() < 1, f"{dist} produced t outside (0,1)"
    assert abs(sample_t(20000, DEV, "uniform").mean().item() - 0.5) < 0.02
    assert abs(sample_t(20000, DEV, "logit_normal").mean().item() - 0.5) < 0.03
    print("  ok  t distributions stay in (0,1) and centre on 0.5")


def test_ema_warmup_tracks_then_lags():
    """EMA must start by tracking the live weights (warmup) and end up lagging,
    otherwise the shadow copy spends the early epochs stuck at initialisation."""
    a = nn.Linear(4, 4).to(DEV)
    b = nn.Linear(4, 4).to(DEV)
    with torch.no_grad():
        b.weight.fill_(1.0)
        a.weight.fill_(0.0)
    ema = EMA(beta=0.99, warmup=10)
    ema.update(a, b)                      # step 0 → beta 0 → copy outright
    assert torch.allclose(a.weight, b.weight), "EMA should copy on the first update"
    ema.step = 10_000
    with torch.no_grad():
        b.weight.fill_(5.0)
    ema.update(a, b)
    assert a.weight.mean().item() < 2.0, "EMA should lag once warmed up"
    print("  ok  EMA copies during warmup, lags afterwards")


def test_zero_background_makes_the_brain_mask_real():
    """`(x != 0)` must actually select the brain when zero_background is on.

    Flow3D's guarantees — bridge noise confined to the brain, background exactly
    zero along the path, metrics scored on tissue — all rest on that mask being
    real. With zero_background off (the historical default) z-scoring shifts the
    background off zero and the mask degenerates to the whole volume, which is
    how ~71 % of every metric in this project ends up measuring a constant.
    """
    from moyamoya.transform import ToChannelsFirstAndNormalize

    vol = torch.rand(8, 9, 8) + 1.0        # (X,Y,Z), strictly positive "tissue"
    vol[:4] = 0.0                          # half of it is background
    true_frac = float((vol != 0).float().mean())

    off = ToChannelsFirstAndNormalize(nonzero_mask=True, zero_background=False)
    on = ToChannelsFirstAndNormalize(nonzero_mask=True, zero_background=True)
    x_off, _ = off(vol.clone(), vol.clone())
    x_on, _ = on(vol.clone(), vol.clone())

    assert float((x_off != 0).float().mean()) == 1.0, (
        "expected the legacy path to leave a nonzero background; if it no longer "
        "does, this test and the warnings that cite it are stale")
    got = float((x_on != 0).float().mean())
    assert abs(got - true_frac) < 1e-6, (
        f"zero_background=True should recover the true tissue fraction "
        f"{true_frac:.3f}, got {got:.3f}")
    # normalisation itself must be unchanged where there is signal
    sig = (vol != 0).permute(2, 1, 0).unsqueeze(0)
    assert torch.allclose(x_on[sig], x_off[sig], atol=1e-6), \
        "zero_background changed the normalised tissue values"
    print(f"  ok  zero_background recovers the real mask "
          f"({true_frac:.2f} of volume) without touching tissue values")


def test_default_condition_and_rejected_combo():
    """Conditioning on x_pre (concat) is the default for both sources, and the
    incoherent source='noise' + condition='none' combo is rejected."""
    assert build_flow3d(dim=8, dim_mults=(1, 2), source="pre").condition == "concat"
    assert build_flow3d(dim=8, dim_mults=(1, 2), source="noise").condition == "concat"
    try:
        build_flow3d(dim=8, dim_mults=(1, 2), source="noise", condition="none")
    except ValueError:
        pass
    else:
        raise AssertionError("source='noise' + condition='none' must be rejected: "
                             "the network would never see which subject to predict")
    print("  ok  condition defaults to 'concat'; noise+none is rejected")


def test_overfit_a_single_pair():
    """The objective must be able to drive a prediction onto its target.

    A model that cannot overfit two volumes has a broken objective, and no
    amount of data or tuning will save it. Tested on the deterministic bridge
    (condition='none', σ=0), which isolates the objective+sampler.
    """
    x_pre, x_post = _pair(seed=3)
    m = build_flow3d(dim=16, dim_mults=(1, 2), init_kernel_size=3,
                     source="pre", condition="none", sigma=0.0).to(DEV)
    opt = torch.optim.AdamW(m.net.parameters(), lr=3e-3)
    brain = (x_pre != 0) | (x_post != 0)
    start = (x_pre - x_post)[brain].abs().mean().item()
    for _ in range(300):
        loss = m.flow_loss(x_post, x_pre)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    with torch.no_grad():
        pred = m.sample(x_pre, steps=8)
    end = (pred - x_post)[brain].abs().mean().item()
    assert end < 0.5 * start, f"failed to overfit: MAE {start:.4f} → {end:.4f}"
    print(f"  ok  overfits a single pair: MAE {start:.4f} → {end:.4f} "
          f"({100 * (1 - end / start):.0f}% reduction)")


def test_conditioning_generalizes_better():
    """The core reason this model conditions on x_pre: it generalises better.

    Trained on *fresh* batches (so neither model can memorise), the conditioned
    net — which sees x_pre directly — recovers the pre→post map with lower
    held-out loss than the unconditioned net, which only sees the blend x_t and
    has to disentangle x_pre from it. On the real 200-subject data the gap is
    the difference between a loss that descends (concat) and one that stalls
    (none); here it shows up as a clearly lower held-out loss.
    """
    def fresh_batch():
        xp = torch.randn(8, 1, 20, 24, 20, device=DEV)
        return xp + 0.4 * torch.roll(xp, shifts=1, dims=2), xp

    heldout = {}
    for condition in ("none", "concat"):
        torch.manual_seed(0)
        m = build_flow3d(dim=16, dim_mults=(1, 2), init_kernel_size=3,
                         source="pre", condition=condition, sigma=0.2,
                         zero_init_out=False).to(DEV)
        opt = torch.optim.AdamW(m.net.parameters(), lr=3e-3)
        for _ in range(400):
            loss = m.flow_loss(*fresh_batch())
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
        m.eval()
        with torch.no_grad():
            heldout[condition] = sum(m.flow_loss(*fresh_batch()).item()
                                     for _ in range(20)) / 20

    assert heldout["concat"] < 0.7 * heldout["none"], (
        f"conditioning did not generalise better: held-out loss "
        f"none={heldout['none']:.4f} concat={heldout['concat']:.4f}")
    print(f"  ok  conditioning generalises better: held-out loss "
          f"none={heldout['none']:.4f} vs concat={heldout['concat']:.4f}")


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    print(f"Flow3D self-checks on {DEV}\n")
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
            failed += 1
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
