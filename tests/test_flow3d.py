"""Self-checks for the Flow3D conditional flow matching implementation.

Run directly (no pytest needed):   python tests/test_flow3d.py
With pytest:                       pytest tests/test_flow3d.py

The load-bearing test is `test_oracle_velocity_is_exact`: if the network returned
the *true* conditional velocity, the sampler must land exactly on x_post. That
pins the interpolant, the velocity target, and the ODE solvers to each other —
a sign error or a mis-scaled time in any one of them breaks it.
"""

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from moyamoya.models.flow3d import (
    ConditionalFlowMatching3D, EMA, build_flow3d, gamma, gamma_dot, sample_t,
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
                                      source="pre", sigma=0.0).to(DEV)
    brain = (x_pre != 0).float()
    for solver in ("euler", "heun", "rk4"):
        for steps in (1, 4, 16):
            out = model.sample(x_pre, steps=steps, solver=solver)
            err = (out - x_post * brain).abs().max().item()
            assert err < 1e-5, f"{solver}/{steps}: sampler off by {err}"
    print("  ok  oracle velocity integrates to x_post exactly (all solvers, 1-16 steps)")


def test_gamma_boundary_conditions():
    """γ must vanish at both endpoints, or the path does not start at x_pre and
    end at x_post; γ' must stay bounded, which is why sin(πt) is used instead of
    the √(t(1-t)) Brownian bridge."""
    t = torch.linspace(0, 1, 101)
    g = gamma(t, 0.3)
    assert g[0].abs() < 1e-6 and g[-1].abs() < 1e-6, "gamma must be 0 at t=0 and t=1"
    assert abs(g.max().item() - 0.3) < 1e-5, "gamma should reach sigma at t=0.5"
    assert gamma_dot(t, 0.3).abs().max().item() <= 0.3 * math.pi + 1e-5
    # numerical derivative agrees with the analytic one
    num = (gamma(t[1:], 0.3) - gamma(t[:-1], 0.3)) / (t[1] - t[0])
    ana = gamma_dot((t[1:] + t[:-1]) / 2, 0.3)
    assert (num - ana).abs().max().item() < 1e-3, "gamma_dot != d(gamma)/dt"
    print("  ok  gamma(0)=gamma(1)=0, |gamma'| bounded, derivative consistent")


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
    t = torch.full((SHAPE[0],), 0.5, device=DEV)
    mask = ((x_pre != 0) | (x_post != 0)).float()
    z = torch.randn_like(x_post) * mask
    x_t = (1 - 0.5) * x_pre + 0.5 * x_post + gamma(t, 0.3)[0] * z
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


def test_no_algebraic_shortcut():
    """The bridge must not be handed x_pre as a separate input channel.

    Along the training path x_t = (1-t)·x_pre + t·x_post, so a network given both
    x_t and x_pre can return (x_t − x_pre)/t and score a near-zero loss while
    learning nothing about the pre→post relationship. The probe: train on pairs
    whose residual is *pure unpredictable noise* of variance 0.25. An honest
    model cannot beat that floor by much. A shortcut model blows straight
    through it — which is the signature to catch, because in real training the
    same collapse hides behind a plausible-looking loss curve.
    """
    torch.manual_seed(0)
    x_pre = torch.randn(8, 1, 24, 28, 24, device=DEV)
    x_post = x_pre + 0.5 * torch.randn_like(x_pre)      # residual ~ N(0, 0.25)
    floor = 0.25

    losses = {}
    for condition in ("concat", "none"):
        torch.manual_seed(0)
        m = build_flow3d(dim=16, dim_mults=(1, 2), init_kernel_size=3,
                         source="pre", condition=condition, sigma=0.0,
                         zero_init_out=False).to(DEV)
        assert m.net.init_conv0[0].in_channels == (2 if condition == "concat" else 1)
        opt = torch.optim.AdamW(m.net.parameters(), lr=3e-3)
        for _ in range(400):
            loss = m.flow_loss(x_post, x_pre)
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
        losses[condition] = loss.item()

    assert losses["concat"] < 0.1 * floor, (
        "expected the concat model to exploit the shortcut; if it no longer "
        "does, this probe has stopped testing anything")
    assert losses["none"] > losses["concat"] * 3, (
        f"condition='none' reached {losses['none']:.4f} vs concat "
        f"{losses['concat']:.4f} — x_pre is still leaking into the network")
    print(f"  ok  no algebraic shortcut: concat={losses['concat']:.4f} (cheats "
          f"past the {floor} floor), none={losses['none']:.4f}")


def test_default_condition_per_source():
    """The safe conditioning must be the default, not something to remember."""
    assert build_flow3d(dim=8, dim_mults=(1, 2), source="pre").condition == "none"
    assert build_flow3d(dim=8, dim_mults=(1, 2), source="noise").condition == "concat"
    try:
        build_flow3d(dim=8, dim_mults=(1, 2), source="noise", condition="none")
    except ValueError:
        pass
    else:
        raise AssertionError("source='noise' + condition='none' must be rejected: "
                             "the network would never see which subject to predict")
    print("  ok  condition defaults to 'none' for the bridge, 'concat' for noise")


def test_overfit_a_single_pair():
    """The objective must be able to drive a prediction onto its target.

    A model that cannot overfit two volumes has a broken objective, and no
    amount of data or tuning will save it.
    """
    x_pre, x_post = _pair(seed=3)
    m = build_flow3d(dim=16, dim_mults=(1, 2), init_kernel_size=3,
                     source="pre", sigma=0.0).to(DEV)
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
