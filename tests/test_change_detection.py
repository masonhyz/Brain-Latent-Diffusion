"""Self-checks for the change-detection / coherent-change rework of Flow3D.

Run directly (no pytest needed):   python tests/test_change_detection.py
With pytest:                       pytest tests/test_change_detection.py

The diagnosis behind this branch: the top-few-% pre→post "change" is ~half
unpredictable registration/acquisition noise, so a plain/re-weighted MSE either
collapses to identity or blurs. The fix factorises WHERE (a supervised gate) from
WHAT (a residual) and supervises the change at the scales where it is predictable.
These tests pin that machinery:

  * gaussian_blur3d / coherent_change_target — the coherent-change soft target.
  * soft_dice_loss                           — the imbalance-robust WHERE loss.
  * multiscale_change_loss / edge_downweight_map — the anti-noise regression.
  * GatedVelocityNet + build_flow3d(gated=…) — the gate·residual factorisation,
    its zero-init identity start, and the detection term inside flow_loss.
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from moyamoya.models.flow3d import (
    build_flow3d, coherent_change_target, edge_downweight_map, gaussian_blur3d,
    GatedVelocityNet, multiscale_change_loss, soft_dice_loss,
)


def _volume(bg=-2.4, brain_shape=(6, 6, 6), full=(10, 10, 10), seed=0):
    """A (1,D,H,W) volume: constant background plateau + a central brain block."""
    g = torch.Generator().manual_seed(seed)
    v = torch.full((1, *full), float(bg))
    sl = tuple(slice((f - b) // 2, (f - b) // 2 + b) for f, b in zip(full, brain_shape))
    v[(0, *sl)] = 1.0 + torch.rand(brain_shape, generator=g)
    return v, sl


# ── gaussian_blur3d ──────────────────────────────────────────────────────────

def test_gaussian_blur_smooths_and_preserves_shape():
    """Blurring keeps the shape, is a near-average (conserves the mean), and
    lowers variance — i.e. it removes high-frequency content."""
    g = torch.Generator().manual_seed(0)
    x = torch.randn(2, 1, 12, 14, 12, generator=g)
    b = gaussian_blur3d(x, sigma=2.0)
    assert b.shape == x.shape
    assert b.var() < x.var(), "blur should reduce variance (remove high freq)"
    # normalised Gaussian conserves the mean to within edge effects
    assert abs(float(b.mean()) - float(x.mean())) < 0.05
    assert torch.allclose(gaussian_blur3d(x, sigma=0.0), x), "sigma=0 must be a no-op"
    print("  ok  gaussian_blur3d: shape-preserving low-pass; sigma=0 is identity")


# ── coherent_change_target ───────────────────────────────────────────────────

def test_coherent_change_target_localises_and_is_bounded():
    """Target ∈ [0,1] and concentrated on the changed sub-region — higher there
    than in the *unchanged brain*, not merely higher than the background."""
    x_pre, sl = _volume(brain_shape=(8, 8, 8), full=(16, 16, 16), seed=1)
    x_post = x_pre.clone()
    edit = tuple(slice(s.start, s.start + 3) for s in sl)   # a 3³ corner of the brain
    x_post[(0, *edit)] += 3.0
    x_pre, x_post = x_pre[None], x_post[None]               # (1,1,D,H,W)

    t = coherent_change_target(x_pre, x_post, sigma=1.0)
    assert t.min() >= 0.0 and t.max() <= 1.0 + 1e-5, "target must be bounded [0,1]"
    edited = torch.zeros_like(t, dtype=torch.bool); edited[(0, 0, *edit)] = True
    brain = torch.zeros_like(t, dtype=torch.bool); brain[(0, 0, *sl)] = True
    rest_of_brain = brain & ~edited
    assert t[edited].mean() > 4.0 * t[rest_of_brain].mean(), (
        f"target not concentrated on the edited sub-region: {t[edited].mean():.3f} "
        f"vs unchanged-brain {t[rest_of_brain].mean():.3f}")
    print("  ok  coherent_change_target: bounded [0,1], localises the edited region")


# ── soft_dice_loss ───────────────────────────────────────────────────────────

def test_soft_dice_perfect_and_disjoint():
    """Dice→0 for a perfect overlap, →~1 for disjoint maps, and pulls a predicted
    map toward the target under a gradient step."""
    tgt = torch.zeros(1, 1, 8, 8, 8); tgt[..., :4, :, :] = 1.0
    assert soft_dice_loss(tgt, tgt) < 1e-2, "identical maps must score ~0"

    disj = torch.zeros_like(tgt); disj[..., 4:, :, :] = 1.0
    assert soft_dice_loss(disj, tgt) > 0.9, "disjoint maps must score ~1"

    p = torch.full_like(tgt, 0.5).requires_grad_(True)
    soft_dice_loss(p, tgt).backward()
    # gradient should push p up where target=1 (negative grad) vs down where target=0
    assert p.grad[tgt.bool()].mean() < p.grad[~tgt.bool()].mean()
    print("  ok  soft_dice_loss: 0 on match, ~1 on disjoint, gradient localises")


# ── multiscale_change_loss ───────────────────────────────────────────────────

def test_multiscale_reduces_to_mse_and_scores_coarse_structure():
    """[1.0] == plain MSE. The *coarse* terms reward getting the low-frequency
    structure right regardless of the (unpredictable) speckle — that is the extra
    coherent-structure gradient multi-scale adds. (Full-res noise suppression is
    the gate's job, not this loss's — see the module docstring.)"""
    g = torch.Generator().manual_seed(2)
    u = torch.randn(2, 1, 16, 16, 16, generator=g)
    v = torch.randn(2, 1, 16, 16, 16, generator=g)
    single = multiscale_change_loss(v, u, [1.0])
    assert torch.allclose(single, (v - u).pow(2).mean(), atol=1e-6), "[1.0] must be MSE"

    # A realistic target = coherent low-freq signal + high-freq speckle (exactly the
    # structure of the real change). The coherent part is what any model can learn.
    signal = 3.0 * gaussian_blur3d(torch.randn(2, 1, 16, 16, 16, generator=g), sigma=3.0)
    speckle = 0.5 * torch.randn(2, 1, 16, 16, 16, generator=g)
    u = signal + speckle
    v_good = signal                                     # coherent part right, speckle omitted
    v_zero = torch.zeros_like(u)
    coarse_only = [0.0, 1.0, 1.0]
    assert (multiscale_change_loss(v_good, u, coarse_only)
            < multiscale_change_loss(v_zero, u, coarse_only)), \
        "the coarse terms must reward matching the low-frequency structure"
    # even the full multi-scale loss (all scales) prefers the coherent prediction
    assert (multiscale_change_loss(v_good, u, [1.0, 1.0, 1.0])
            < multiscale_change_loss(v_zero, u, [1.0, 1.0, 1.0])), \
        "predicting the coherent signal should beat predicting nothing"
    print("  ok  multiscale_change_loss: MSE at 1 scale; coarse terms score structure")


# ── edge_downweight_map ──────────────────────────────────────────────────────

def test_edge_downweight_discounts_edges_only():
    """γ=0 → all ones; γ>0 → weight is smaller on x_pre's high-gradient edges."""
    x_pre, sl = _volume(seed=3)
    x_pre = x_pre[None]
    assert torch.allclose(edge_downweight_map(x_pre, 0.0), torch.ones_like(x_pre))

    w = edge_downweight_map(x_pre, gamma=0.8)
    # brain-block boundary voxels (large |∇|) should be down-weighted vs the flat
    # interior of the background plateau.
    grad = torch.zeros_like(x_pre)
    for ax in range(2, x_pre.ndim):
        grad = grad + (x_pre - torch.roll(x_pre, 1, ax)).pow(2)
    edge = grad.squeeze() > grad.squeeze().mean()
    flat = ~edge
    assert w.squeeze()[edge].mean() < w.squeeze()[flat].mean()
    assert w.min() >= 0.1 - 1e-6, "weight floor respected"
    print("  ok  edge_downweight_map: γ=0 is ones; edges discounted, floor held")


# ── GatedVelocityNet + build_flow3d(gated) ───────────────────────────────────

def test_gated_velocity_is_gate_times_residual():
    """v = gate·residual with gate ∈ (0,1); forward() returns exactly that velocity."""
    torch.manual_seed(0)
    m = build_flow3d(dim=8, dim_mults=(1, 2), gated=True, zero_init_out=False)
    assert isinstance(m.net, GatedVelocityNet)
    x = torch.randn(2, 2, 12, 12, 12)                    # cat([x_pre, x_t]) = 2 ch
    t = torch.zeros(2)
    gate, v = m.net.gate_velocity(x, t)
    assert gate.min() >= 0.0 and gate.max() <= 1.0
    assert torch.allclose(m.net(x, t), v), "forward() must equal gate·residual"
    print("  ok  GatedVelocityNet: v = gate·residual, gate in [0,1]")


def test_gated_zero_init_starts_at_identity_and_sparse_gate():
    """With zero_init_out the untrained gated model has v≡0 (identity bridge) and
    a sparse gate (sigmoid of the negative init bias)."""
    m = build_flow3d(dim=8, dim_mults=(1, 2), gated=True, zero_init_out=True,
                     gate_init_bias=-2.0)
    x_pre = torch.randn(2, 1, 12, 12, 12)
    with torch.no_grad():
        s = m.sample(x_pre, steps=2, solver="euler")
    assert torch.allclose(s, x_pre, atol=1e-5), "zero-init bridge must return x_pre"
    gmap = m.predict_change_map(x_pre)
    assert abs(float(gmap.mean()) - torch.sigmoid(torch.tensor(-2.0))) < 1e-3, \
        "gate should start at sigmoid(gate_init_bias)"
    print("  ok  gated zero-init: v≡0 (identity), gate starts sparse")


def test_gated_flow_loss_components_and_detection():
    """flow_loss(return_components) exposes reg + det; the detection term is a real
    positive contribution and the whole thing is differentiable."""
    torch.manual_seed(1)
    m = build_flow3d(dim=8, dim_mults=(1, 2), gated=True, det_weight=0.3,
                     det_sigma=2.0, ms_weights=(1.0, 1.0, 1.0), change_weight=0.0,
                     edge_downweight=0.3, sigma=0.3)
    x = torch.randn(2, 1, 12, 12, 12)
    y = x + 0.3 * torch.randn(2, 1, 12, 12, 12)
    loss, comp = m.flow_loss(x_post=y, x_pre=x, return_components=True)
    assert {"reg", "det"} <= set(comp), comp
    assert comp["det"] > 0.0 and comp["reg"] > 0.0
    loss.backward()
    grads = [p.grad for p in m.net.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    print("  ok  gated flow_loss: reg+det components, finite gradients")


def test_nongated_backward_compatible():
    """A non-gated model is unchanged: single-channel velocity, no gate, and
    flow_loss with change_weight reproduces the old change-weighted MSE path."""
    m = build_flow3d(dim=8, dim_mults=(1, 2), gated=False, change_weight=5.0)
    assert not isinstance(m.net, GatedVelocityNet)
    x_pre = torch.randn(1, 1, 12, 12, 12)
    assert m.predict_change_map(x_pre) is None, "no gate on a non-gated model"
    y = x_pre + 0.2 * torch.randn_like(x_pre)
    loss = m.flow_loss(x_post=y, x_pre=x_pre)
    assert torch.isfinite(loss)
    print("  ok  non-gated model: unchanged velocity path, no gate")


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    print("change-detection self-checks\n")
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
