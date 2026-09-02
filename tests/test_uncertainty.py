"""Self-checks for the uncertainty-trust-signal math (moyamoya/uncertainty.py).

Run directly (no pytest needed):   python tests/test_uncertainty.py
With pytest:                       pytest tests/test_uncertainty.py

The load-bearing tests are the two exactness ones: a *perfect* uncertainty map
(unc ∝ err) must give AUSE = 0 and AUROC = 1, and a constant (information-free)
map must give AUSE = ause_random and AUROC = 0.5. Those pin the curve
construction, the normalisation, and the tie handling to the definitions.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from moyamoya.uncertainty import (
    auroc, bootstrap_ci, recall_at_budget, retention_curve, sparsification,
)

RNG = np.random.default_rng(0)


def test_sparsification_perfect_uncertainty():
    err = RNG.exponential(1.0, size=5000)
    sp = sparsification(err, unc=err.copy())        # unc ranks exactly like err
    assert abs(sp["ause"]) < 1e-9, f"perfect ranking must give AUSE 0, got {sp['ause']}"
    assert np.allclose(sp["by_uncertainty"], sp["by_error"])
    assert np.isclose(sp["by_uncertainty"][0], 1.0)  # normalised to full-set mean
    print("  ok  perfect uncertainty → AUSE = 0")


def test_sparsification_constant_uncertainty():
    err = RNG.exponential(1.0, size=5000)
    sp = sparsification(err, unc=np.zeros_like(err))
    # A constant map carries no ranking information: with stable sort the
    # "removal" order is arbitrary-but-fixed, so the curve hovers at ~1 and the
    # AUSE approaches the area above the oracle.
    assert abs(sp["ause"] - sp["ause_random"]) < 0.05 * sp["ause_random"], (
        f"constant unc: AUSE {sp['ause']:.4f} should ≈ ause_random "
        f"{sp['ause_random']:.4f}")
    print("  ok  constant uncertainty → AUSE ≈ ause_random")


def test_sparsification_oracle_monotone():
    err = RNG.exponential(1.0, size=2000)
    sp = sparsification(err, unc=RNG.random(2000))
    assert np.all(np.diff(sp["by_error"]) <= 1e-12), "oracle curve must be non-increasing"
    assert sp["ause"] >= -1e-12, "no ranking can beat the oracle"
    print("  ok  oracle curve monotone and lower-bounds the model curve")


def test_auroc_exact_values():
    # Perfectly separated → 1; anti-separated → 0; all-tied → exactly 0.5.
    y = np.array([0, 0, 0, 1, 1])
    assert auroc([1, 2, 3, 4, 5], y) == 1.0
    assert auroc([5, 4, 3, 2, 1], y) == 0.0
    assert auroc([7, 7, 7, 7, 7], y) == 0.5
    assert np.isnan(auroc([1, 2, 3], [0, 0, 0]))    # undefined without positives
    print("  ok  AUROC exact on separable / tied / degenerate cases")


def test_recall_at_budget():
    y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1])
    s_good = np.arange(10, dtype=float)             # positives score highest
    assert recall_at_budget(s_good, y, 0.2) == 1.0
    assert recall_at_budget(s_good, y, 0.1) == 0.5
    s_bad = -s_good                                 # positives score lowest
    assert recall_at_budget(s_bad, y, 0.2) == 0.0
    print("  ok  recall@budget catches ranked positives")


def test_retention_curve():
    e = RNG.exponential(1.0, size=40)
    rc = retention_curve(scores=e.copy(), errors=e)  # perfect subject ranking
    assert np.allclose(rc["by_uncertainty"], rc["by_error"])
    assert np.isclose(rc["by_uncertainty"][0], e.mean())    # f=0: keep everyone
    assert np.all(np.diff(rc["by_error"]) <= 1e-12)
    assert np.allclose(rc["random"], e.mean())
    print("  ok  retention curve: f=0 is full-set mean, oracle monotone")


def test_bootstrap_ci_brackets_point_estimate():
    s = np.concatenate([RNG.normal(0, 1, 50), RNG.normal(2, 1, 50)])
    y = np.concatenate([np.zeros(50), np.ones(50)])
    point = auroc(s, y)
    lo, hi = bootstrap_ci(auroc, s, y, n_boot=500, seed=1)
    assert lo <= point <= hi, f"CI [{lo:.3f}, {hi:.3f}] must bracket {point:.3f}"
    assert 0.5 < lo, "separated classes: CI should exclude chance"
    print(f"  ok  bootstrap CI [{lo:.3f}, {hi:.3f}] brackets AUROC {point:.3f}")


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    print("Uncertainty self-checks\n")
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
