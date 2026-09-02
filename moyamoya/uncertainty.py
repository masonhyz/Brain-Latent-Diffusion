"""Uncertainty-as-a-trust-signal — does the model know when it is wrong?

The clinical-trust question this module answers is *not* "how accurate is the
prediction" but "can the model flag its own failures". The mechanism: the flow
sampler's probability-flow ODE is deterministic given its start, so perturbing
x(0) (``init_noise``) yields an ensemble of distinct predictions per subject.
The ensemble mean is the prediction; the per-voxel std across members is a
*predicted* uncertainty map, available at inference time with no ground truth.
Everything here quantifies how well that predicted uncertainty tracks the
*actual* error of the ensemble mean, at two granularities:

  * **voxel level** — within one subject, do high-variance voxels coincide with
    high-error voxels? Measured by the sparsification curve and its AUSE
    (Ilg et al. 2018, "Uncertainty estimates ... for optical flow"): remove the
    most-uncertain voxels first and watch the error of the remainder fall; an
    oracle that removes by true error lower-bounds the curve, and the area
    between the two (AUSE, lower = better, 0 = perfect ranking) is the summary.
  * **subject level** — across patients, does a scalar uncertainty score
    (computed from the pre-op scan and the ensemble alone) rank the bad
    predictions first? Measured by AUROC / recall-at-budget against a "failure"
    label and by the error-retention curve: if a clinician reviews only the
    top-f most-uncertain predictions, how much error leaves the automated
    stream, and how many of the true failures are caught?

All functions are pure numpy (torch-free) so they are trivially testable and
usable from any eval script. Curves are evaluated on a fixed fraction grid so
per-subject curves can be averaged across subjects.
"""

import numpy as np
from scipy.stats import rankdata


# ─────────────────────────────────────────────────────────────────────────────
# Voxel level — sparsification / AUSE
# ─────────────────────────────────────────────────────────────────────────────

def _retained_mean_curve(err: np.ndarray, order: np.ndarray,
                         fracs: np.ndarray) -> np.ndarray:
    """Mean of ``err`` after removing the first ``f·N`` entries of ``order``.

    ``order`` ranks entries most-suspect-first; the curve value at fraction f is
    the mean error over what *remains* when the top f are discarded — the error
    a downstream consumer sees if the flagged part is handled elsewhere.
    """
    e = err[order]
    n = e.size
    # tail_sum[k] = sum of e[k:], so retained mean after removing k = tail_sum[k]/(n-k)
    tail_sum = np.concatenate([np.cumsum(e[::-1])[::-1], [0.0]])
    ks = np.minimum((fracs * n).astype(int), n - 1)     # always retain ≥ 1
    return tail_sum[ks] / (n - ks)


def sparsification(err: np.ndarray, unc: np.ndarray, n_points: int = 50) -> dict:
    """Sparsification analysis of a predicted-uncertainty map against true error.

    Sorts voxels most-uncertain-first and computes the mean error of the
    retained set as the removed fraction grows (``by_uncertainty``); the same
    with the true error as the ranking gives the oracle bound (``by_error``).
    Both curves are normalised by the full-set mean error (their f=0 value), so
    they start at 1 and AUSE — the area between them over f ∈ [0, max frac] —
    is scale-free and comparable across subjects. A flat ``by_uncertainty``
    curve (uncertainty carries no ranking information) has AUSE equal to the
    area above the oracle, reported as ``ause_random`` for reference.

    Args:
        err: 1-D per-voxel error of the prediction (e.g. |mean − target|).
        unc: 1-D per-voxel predicted uncertainty (e.g. ensemble std), same size.
        n_points: size of the fraction grid (fixed, so curves average across
            subjects).

    Returns:
        {"fracs", "by_uncertainty", "by_error", "ause", "ause_random"} — curves
        normalised to the f=0 error; NaN-filled when ``err`` is degenerate.
    """
    err = np.asarray(err, dtype=np.float64).ravel()
    unc = np.asarray(unc, dtype=np.float64).ravel()
    if err.size != unc.size:
        raise ValueError(f"size mismatch: err {err.size} vs unc {unc.size}")
    fracs = np.linspace(0.0, 0.99, n_points)
    base = err.mean()
    if err.size == 0 or not np.isfinite(base) or base <= 0:
        nan = np.full(n_points, np.nan)
        return {"fracs": fracs, "by_uncertainty": nan, "by_error": nan,
                "ause": float("nan"), "ause_random": float("nan")}

    curve_u = _retained_mean_curve(err, np.argsort(-unc, kind="stable"), fracs) / base
    curve_e = _retained_mean_curve(err, np.argsort(-err, kind="stable"), fracs) / base
    ause = float(np.trapezoid(curve_u - curve_e, fracs))
    ause_random = float(np.trapezoid(1.0 - curve_e, fracs))
    return {"fracs": fracs, "by_uncertainty": curve_u, "by_error": curve_e,
            "ause": ause, "ause_random": ause_random}


# ─────────────────────────────────────────────────────────────────────────────
# Subject level — failure flagging and error retention
# ─────────────────────────────────────────────────────────────────────────────

def auroc(scores, labels) -> float:
    """AUROC of ``scores`` for ranking ``labels`` (1 = positive), tie-aware.

    The Mann-Whitney identity: AUROC = P(score⁺ > score⁻) + ½P(tie), computed
    from average ranks — exact, no thresholds, no sklearn. NaN when either
    class is empty (AUROC is undefined, not 0.5).
    """
    s = np.asarray(scores, dtype=np.float64)
    pos = np.asarray(labels).astype(bool)
    n1, n0 = int(pos.sum()), int((~pos).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def recall_at_budget(scores, labels, budget: float) -> float:
    """Fraction of positives caught when the top-``budget`` by score are flagged.

    The deployment quantity: "review the ``budget`` most-uncertain predictions —
    how many of the actual failures does that catch?" Flags ⌈budget·n⌉ items
    (at least 1). Ties are broken by index, matching a fixed review queue.
    """
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels).astype(bool)
    if y.sum() == 0:
        return float("nan")
    k = max(1, int(np.ceil(budget * s.size)))
    flagged = np.argsort(-s, kind="stable")[:k]
    return float(y[flagged].sum() / y.sum())


def retention_curve(scores, errors) -> dict:
    """Subject-level error retention as the review budget grows.

    At review fraction f the top-f most-uncertain subjects go to a human and the
    rest stay automated; the curve reports the mean error of the automated
    remainder. ``oracle`` reviews by true error (the bound), ``random`` is the
    flat full-set mean (a budget spent on a random subset does not change the
    expected retained error). Evaluated at f = k/n for k = 0 … n−1.

    Returns {"fracs", "by_uncertainty", "by_error", "random"} in raw error units
    (not normalised — subject MAEs share a scale, unlike per-subject voxel
    errors).
    """
    s = np.asarray(scores, dtype=np.float64)
    e = np.asarray(errors, dtype=np.float64)
    n = e.size
    fracs = np.arange(n) / n
    return {
        "fracs": fracs,
        "by_uncertainty": _retained_mean_curve(e, np.argsort(-s, kind="stable"), fracs),
        "by_error": _retained_mean_curve(e, np.argsort(-e, kind="stable"), fracs),
        "random": np.full(n, e.mean()),
    }


def bootstrap_ci(stat_fn, *arrays, n_boot: int = 2000, seed: int = 0,
                 ci: float = 0.95) -> tuple:
    """Percentile bootstrap CI for ``stat_fn(*arrays)`` resampled jointly.

    Resamples subjects (rows) with replacement, keeping the arrays aligned —
    the right unit for n≈35 subject-level statistics, where the point estimate
    alone would overstate certainty. Non-finite resample values (e.g. an AUROC
    draw with no positives) are dropped. Returns (lo, hi).
    """
    rng = np.random.default_rng(seed)
    arrays = [np.asarray(a) for a in arrays]
    n = arrays[0].shape[0]
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        v = stat_fn(*(a[idx] for a in arrays))
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return (float("nan"), float("nan"))
    lo, hi = np.percentile(vals, [100 * (1 - ci) / 2, 100 * (1 + ci) / 2])
    return (float(lo), float(hi))
