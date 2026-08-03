"""Self-checks for the change-aware training additions (the identity-collapse fix).

Run directly (no pytest needed):   python tests/test_change_aware.py
With pytest:                       pytest tests/test_change_aware.py

Three coordinated pieces, tested here:

  * change_weight_map  — per-voxel loss weight that up-weights where pre→post
                         actually changes (moyamoya/models/flow3d.py).
  * change_mask / change_region_report — score only the region surgery altered,
                         the metric the whole-volume numbers are blind to
                         (moyamoya/metrics.py).
  * change_sampler     — oversample the big-change subjects (moyamoya/data.py).

The whole point is that a flat MSE over ~71%-background, near-identity volumes is
minimised by copying x_pre; these tests pin the machinery that redirects the
gradient (and the evaluation) onto the sparse voxels that carry the edits.
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from moyamoya.models.flow3d import change_weight_map
from moyamoya.metrics import (
    change_mask, change_region_report, foreground_mask,
)
from moyamoya.data import change_magnitudes, change_sampler


def _volume(bg=-2.4, brain_shape=(6, 6, 6), full=(10, 10, 10), seed=0):
    """A (1,D,H,W) volume: a constant background plateau with a central brain
    block of distinct positive values — the structure both preprocessing
    conventions produce (a modal background + real tissue)."""
    g = torch.Generator().manual_seed(seed)
    v = torch.full((1, *full), float(bg))
    sl = tuple(slice((f - b) // 2, (f - b) // 2 + b) for f, b in zip(full, brain_shape))
    v[(0, *sl)] = 1.0 + torch.rand(brain_shape, generator=g)
    return v, sl


# ── change_weight_map ────────────────────────────────────────────────────────

def test_change_weight_map_mean_one_and_gamma0():
    """w averages to 1 per sample (so the loss scale / LR is unchanged), and
    γ=0 gives all-ones — i.e. exactly the plain MSE."""
    g = torch.Generator().manual_seed(1)
    x_pre = torch.randn(3, 1, 8, 8, 8, generator=g)
    x_post = x_pre + 0.3 * torch.randn(3, 1, 8, 8, 8, generator=g)

    w = change_weight_map(x_pre, x_post, gamma=5.0)
    per_sample_mean = w.mean(dim=(1, 2, 3, 4))
    assert torch.allclose(per_sample_mean, torch.ones(3), atol=1e-5), per_sample_mean
    assert (w > 0).all()

    w0 = change_weight_map(x_pre, x_post, gamma=0.0)
    assert torch.allclose(w0, torch.ones_like(w0), atol=1e-6), "γ=0 must be plain MSE"
    print("  ok  change_weight_map: mean-1 per sample; γ=0 is all-ones (plain MSE)")


def test_change_weight_map_emphasizes_change():
    """Voxels that change more get more weight; a fixed per-voxel error therefore
    contributes more loss in the region of change than outside it."""
    x_pre, sl = _volume(seed=2)
    x_post = x_pre.clone()
    x_post[(0, *sl)] += 3.0                          # a big edit in the brain block
    x_pre, x_post = x_pre[None], x_post[None]        # add batch dim → (1,1,D,H,W)

    w = change_weight_map(x_pre, x_post, gamma=5.0)
    changed = torch.zeros_like(w, dtype=torch.bool)
    changed[(0, 0, *sl)] = True
    assert w[changed].mean() > 3.0 * w[~changed].mean(), (
        f"change region under-weighted: {w[changed].mean():.3f} vs "
        f"{w[~changed].mean():.3f}")

    # a constant error everywhere ⇒ its weighted contribution is larger in-region
    se = torch.full_like(w, 0.5)
    assert (w * se)[changed].mean() > (w * se)[~changed].mean()
    print("  ok  change_weight_map concentrates loss on the changed voxels")


# ── foreground / change ROI ──────────────────────────────────────────────────

def test_foreground_mask_both_background_conventions():
    """foreground_mask must find the brain whether the background is a nonzero
    plateau (zero_background=False) or exact 0 (zero_background=True)."""
    for bg in (-2.4, 0.0):
        v, sl = _volume(bg=bg, seed=3)
        fg = foreground_mask(v)
        truth = np.zeros((10, 10, 10), dtype=bool)
        truth[sl] = True
        assert np.array_equal(fg, truth), f"bg={bg}: foreground != brain block"
    print("  ok  foreground_mask detects the brain under both background conventions")


def test_change_mask_selects_the_most_changed_voxels():
    """change_mask picks ~frac of the brain, and exactly the highest-change end."""
    x_pre, sl = _volume(seed=4)
    x_post = x_pre.clone()
    # graded, all-distinct change over the brain block so the top-frac is unambiguous
    n = int(np.prod([s.stop - s.start for s in sl]))
    grade = torch.linspace(0.1, 5.0, n).reshape([s.stop - s.start for s in sl])
    x_post[(0, *sl)] += grade

    frac = 0.1
    roi = change_mask(x_pre, x_post, frac=frac)
    brain = np.zeros((10, 10, 10), dtype=bool)
    brain[sl] = True

    assert roi[~brain].sum() == 0, "ROI leaked outside the brain"
    roi_frac_of_brain = roi.sum() / brain.sum()
    assert 0.07 < roi_frac_of_brain < 0.14, roi_frac_of_brain

    diff = np.abs((x_post - x_pre).numpy().squeeze())
    sel, unsel = diff[roi], diff[brain & ~roi]
    assert sel.min() >= unsel.max(), "selected voxels are not the most-changed ones"
    print(f"  ok  change_mask selects the top {frac:.0%} most-changed brain voxels")


def test_change_region_report_identity_vs_perfect():
    """The report must read ≈0 improvement for the copy baseline and a large
    positive improvement for a perfect prediction."""
    x_pre, sl = _volume(seed=5)
    x_post = x_pre.clone()
    x_post[(0, *sl)] += 2.0

    ident = change_region_report(x_pre, x_pre, x_post)      # pred = copy x_pre
    assert abs(ident["change_mae_improvement"]) < 1e-6
    assert abs(ident["change_mae"] - ident["identity_change_mae"]) < 1e-6

    perfect = change_region_report(x_post, x_pre, x_post)   # pred = x_post
    assert perfect["change_mae"] < 1e-6
    assert perfect["change_mae_improvement"] > 1.0, perfect["change_mae_improvement"]
    assert perfect["change_roi_frac"] > 0
    print("  ok  change_region_report: copy→0 improvement, perfect→large improvement")


# ── change-emphasis sampler ──────────────────────────────────────────────────

def test_change_sampler_oversamples_big_change_subjects():
    """The sampler must weight — and therefore draw — big-change subjects more."""
    small_a, sl = _volume(seed=6)
    small_b, _ = _volume(seed=7)
    big, _ = _volume(seed=8)

    def _pair(v, delta):
        y = v.clone()
        y[(0, *sl)] += delta
        return v, y

    # index 2 changes ~10× more than 0 and 1
    subset = [_pair(small_a, 0.1), _pair(small_b, 0.1), _pair(big, 2.0)]

    g = change_magnitudes(subset)
    assert g.argmax() == 2, f"global change vector wrong: {g}"

    sampler, g2 = change_sampler(subset, beta=1.0)
    w = sampler.weights.numpy()
    assert w[2] > w[0] and w[2] > w[1], f"big-change subject under-weighted: {w}"

    draws = torch.multinomial(sampler.weights, 6000, replacement=True)
    counts = torch.bincount(draws, minlength=3)
    assert counts[2] > counts[0] and counts[2] > counts[1], counts
    print(f"  ok  change_sampler oversamples the big-change subject "
          f"(draws {counts.tolist()} for change {np.round(g, 2).tolist()})")


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    print("change-aware self-checks\n")
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
