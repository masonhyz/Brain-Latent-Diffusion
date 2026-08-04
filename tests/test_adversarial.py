"""Self-checks for the adversarial detail term (the anti-blur fix).

Run directly (no pytest needed):   python tests/test_adversarial.py
With pytest:                       pytest tests/test_adversarial.py

The change-weighted regression still blurred, because any MSE-type objective
predicts the *mean* edit. A conditional PatchGAN on the one-step prediction adds
back the high-frequency detail. These tests pin the pieces: the discriminator
produces a real patch grid, the hinge losses point the right way, the one-step
prediction is the clean differentiable fake, and a full G/D step runs stably.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from moyamoya.models.flow3d import (
    build_discriminator3d, build_flow3d, d_hinge_loss, g_hinge_loss,
)

DEV = "cpu"                      # tiny volumes; CPU keeps the test deterministic
SHAPE = (2, 1, 48, 48, 48)


def _pair(seed=0):
    g = torch.Generator().manual_seed(seed)
    x_pre = torch.randn(SHAPE, generator=g)
    x_post = x_pre + 0.3 * torch.randn(SHAPE, generator=g)
    x_pre[..., :6, :, :] = 0
    x_post[..., :6, :, :] = 0
    return x_pre.to(DEV), x_post.to(DEV)


def test_discriminator_emits_a_patch_grid():
    """A PatchGAN must output a *grid* of logits (local judgements), not one
    scalar — that locality is what makes it sharpen detail."""
    D = build_discriminator3d(dim=16, n_layers=3).to(DEV)
    x_pre, x_post = _pair()
    logits = D(x_pre, x_post)
    assert logits.shape[0] == SHAPE[0] and logits.shape[1] == 1
    grid = logits.shape[2:]
    assert all(g > 1 for g in grid), f"expected a patch grid, got {tuple(grid)}"
    print(f"  ok  discriminator emits a {tuple(grid)} patch grid per volume")


def test_hinge_losses_point_the_right_way():
    """D-hinge is minimised by real≫0, fake≪0; G-hinge just raises fake."""
    real_good = torch.full((2, 1, 4, 4, 4), 2.0)
    fake_good = torch.full((2, 1, 4, 4, 4), -2.0)
    l_good = d_hinge_loss(real_good, fake_good).item()
    l_bad = d_hinge_loss(fake_good, real_good).item()   # swapped → worse
    assert l_good < l_bad, f"d_hinge not discriminating: {l_good} vs {l_bad}"
    assert abs(l_good) < 1e-6, f"confident-correct hinge should be ~0, got {l_good}"
    # generator wants the discriminator's fake score up ⇒ loss decreases with score
    assert g_hinge_loss(torch.ones(8)) < g_hinge_loss(-torch.ones(8))
    print("  ok  hinge losses point the right way (D separates, G lifts fakes)")


def test_onestep_prediction_is_the_clean_fake():
    """One-step prediction returns x_pre at init (zero-velocity) and is only
    defined for the bridge."""
    x_pre, _ = _pair()
    m = build_flow3d(dim=8, dim_mults=(1, 2), init_kernel_size=3,
                     source="pre", zero_init_out=True).to(DEV)
    pred = m.onestep_prediction(x_pre)
    assert pred.shape == x_pre.shape
    assert (pred - x_pre).abs().max().item() == 0.0, "zero-init one-step must be x_pre"

    mn = build_flow3d(dim=8, dim_mults=(1, 2), init_kernel_size=3,
                      source="noise").to(DEV)
    try:
        mn.onestep_prediction(x_pre)
    except ValueError:
        pass
    else:
        raise AssertionError("onestep_prediction must reject source='noise'")
    print("  ok  one-step prediction: x_pre at init, bridge-only")


def test_full_adversarial_step_is_stable():
    """One full generator+discriminator update (mirroring the training loop) must
    run with finite losses and finite grads on both networks."""
    x_pre, x_post = _pair(seed=1)
    m = build_flow3d(dim=8, dim_mults=(1, 2), init_kernel_size=3,
                     source="pre", zero_init_out=False).to(DEV)
    D = build_discriminator3d(dim=8, n_layers=3).to(DEV)
    opt = torch.optim.AdamW(m.net.parameters(), lr=1e-3)
    opt_d = torch.optim.AdamW(D.parameters(), lr=1e-3, betas=(0.5, 0.9))

    opt.zero_grad(set_to_none=True)
    loss_cfm = m.flow_loss(x_post, x_pre)
    loss_cfm.backward()
    x1 = m.onestep_prediction(x_pre)
    loss_d = d_hinge_loss(D(x_pre, x_post), D(x_pre, x1.detach()))
    opt_d.zero_grad(set_to_none=True); loss_d.backward(); opt_d.step()
    g_adv = g_hinge_loss(D(x_pre, x1))
    (0.05 * g_adv).backward()
    opt.step()

    for name, val in [("cfm", loss_cfm), ("d", loss_d), ("g_adv", g_adv)]:
        assert torch.isfinite(val), f"{name} loss not finite"
    g_grads = [p.grad for p in m.net.parameters() if p.grad is not None]
    d_grads = [p.grad for p in D.parameters() if p.grad is not None]
    assert g_grads and all(torch.isfinite(g).all() for g in g_grads), "bad G grads"
    assert d_grads and all(torch.isfinite(g).all() for g in d_grads), "bad D grads"
    print("  ok  full adversarial step runs; losses + grads on G and D all finite")


def test_discriminator_detects_blur():
    """The adversarial signal must be real and must specifically detect *blur* —
    that is the failure mode it exists to fix. With a sharp real (x_post) and a
    blurred fake (smoothed x_post), a few D updates must open up a score margin
    (real > fake). The hinge value itself stays near its floor because spectral
    norm caps the logit magnitude — that is the stability feature, not a failure —
    so the meaningful signal is the *margin*, which is what the generator follows.
    """
    import torch.nn.functional as F
    _, x_post = _pair(seed=2)
    blur = F.avg_pool3d(x_post, kernel_size=3, stride=1, padding=1)   # kills detail
    D = build_discriminator3d(dim=8, n_layers=3).to(DEV)
    opt_d = torch.optim.AdamW(D.parameters(), lr=2e-3, betas=(0.5, 0.9))

    def margin():
        return (D(x_post, x_post).mean() - D(x_post, blur).mean()).item()

    start = margin()
    for _ in range(60):
        loss_d = d_hinge_loss(D(x_post, x_post), D(x_post, blur))
        opt_d.zero_grad(set_to_none=True); loss_d.backward(); opt_d.step()
    end = margin()
    assert end > start and end > 0.1, f"D did not learn to detect blur: {start:.3f}→{end:.3f}"
    print(f"  ok  discriminator learns to detect blur (real−fake margin {start:+.3f} → {end:+.3f})")


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    print(f"adversarial self-checks on {DEV}\n")
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
