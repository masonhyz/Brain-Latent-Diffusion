"""Flow3D uncertainty-as-a-trust-signal — does the ensemble know when it's wrong?

Samples an ensemble of K perturbed-start trajectories per subject (the
probability-flow ODE is deterministic, so ``--init_noise`` on x(0) is the
stochasticity knob), takes the ensemble mean as the prediction and the
per-voxel std as a *predicted uncertainty map*, then quantifies how well that
uncertainty tracks the actual error — the "flag your own failures" result:

  * voxel level   — sparsification curves + AUSE, per-voxel error↔std
                    correlation (over brain voxels; the union mask is ~71%
                    background where both are trivially ~0).
  * subject level — a scalar uncertainty score per subject computed WITHOUT
                    ground truth (mean/p99/top-1% ensemble std over the pre-op
                    brain), scored by Spearman vs true MAE, AUROC for flagging
                    the worst ``--worst_frac`` of predictions (and the
                    predictions that lose to the identity baseline), recall at
                    a ``--budget`` review budget, and the error-retention curve.

    # the headline run: held-out split, 20-member ensemble
    python scripts/eval_flow3d_uncertainty.py \
        --ckpt runs/flow3d_2026-08-02_21-07-41/best_mae.pt --n_samples 20

    # pick the init_noise operating point first
    python scripts/eval_flow3d_uncertainty.py \
        --ckpt runs/flow3d_2026-08-02_21-07-41/best_mae.pt \
        --sweep_init_noise 0.05 0.1 0.2 0.3 --n_samples 8

Outputs (under --out_dir, default outputs/uncertainty/<run name>):
    per_subject.csv        per-subject metrics, scores, and voxel-level stats
    summary.json           config + all aggregate trust-signal numbers
    fig_trust_triage.png   THE figure: failures caught vs review budget +
                           error retention of the auto-accepted remainder
    fig_sparsification.png voxel-level sparsification (mean over subjects)
    fig_subject_scatter.png subject uncertainty vs subject error
    fig_qualitative.png    slices: pre / post / prediction / error / uncertainty
"""

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).parent.parent))

from moyamoya.data import reconstruct_val_split
from moyamoya.dataset import PrePostFMRI
from moyamoya.metrics import (
    change_mask, compute_metrics, foreground_mask, union_mask,
)
from moyamoya.models.flow3d import gaussian_blur3d, load_flow3d_checkpoint
from moyamoya.transform import ToChannelsFirstAndNormalize
from moyamoya.uncertainty import (
    auroc, bootstrap_ci, recall_at_budget, retention_curve, sparsification,
)
from moyamoya.utils import get_device

# Validated categorical palette (dataviz skill): model = blue, second label =
# orange; oracle/random are reference bounds, not series, so they stay gray.
C_MODEL, C_ALT = "#2a78d6", "#eb6834"
C_ORACLE, C_RANDOM, C_INK = "#6f6e66", "#a5a49b", "#3d3d3a"

# Subject-level trust scores, all computable at deployment (no ground truth):
# three summaries of the ensemble std over the pre-op brain, plus pred_change —
# the size of the edit the model attempted — as a non-ensemble comparison.
SCORES = ("unc_mean", "unc_p99", "unc_top1", "pred_change")
PRIMARY = "unc_mean"


def subject_seed(base: int, sid: str) -> int:
    """Stable per-subject RNG seed, so a subject's ensemble reproduces
    regardless of dataset ordering."""
    h = int(hashlib.sha1(sid.encode()).hexdigest()[:8], 16)
    return (base + h) % (2 ** 31 - 1)


@torch.no_grad()
def sample_ensemble(model, x, args, steps):
    """K stochastic trajectories for one subject → (K, D, H, W) fp32 CPU.

    ``--sampler ode`` integrates the deterministic ODE from an
    ``--init_noise``-perturbed start (the tuned-knob ensemble); ``--sampler
    sde`` samples the model's own probability path via the score-derived
    Langevin SDE (see ``ConditionalFlowMatching3D.sample_sde``) — the trained σ
    is the stochasticity, no knob. Batches ``chunk`` members at a time; a
    single per-subject generator makes the K draws distinct and reproducible.
    """
    g = torch.Generator(device=x.device).manual_seed(
        subject_seed(args.sample_seed, args._sid))
    outs = []
    for i in range(0, args.n_samples, args.chunk):
        b = min(args.chunk, args.n_samples - i)
        xb = x.repeat(b, 1, 1, 1, 1)
        if args.sampler == "sde":
            p = model.sample_sde(xb, steps=steps,
                                 gamma_scale=args.sde_gamma_scale,
                                 guidance_scale=args.guidance_scale,
                                 generator=g)
        else:
            p = model.sample(xb, steps=steps, solver=args.solver,
                             guidance_scale=args.guidance_scale,
                             init_noise=args.init_noise, generator=g)
        outs.append(p.squeeze(1).float().cpu())
    return torch.cat(outs, 0)


def evaluate_subject(model, x, y, sid, args, steps):
    """Ensemble one subject and compute every per-subject quantity.

    Returns (row dict, extras dict) — ``extras`` carries the arrays the
    aggregate figures need (sparsification curve, maps for the qualitative
    panel).
    """
    xb = x.unsqueeze(0).to(args.device)
    args._sid = sid
    preds = sample_ensemble(model, xb, args, steps)
    mean = preds.mean(0)                                   # (D,H,W)
    std = preds.std(0, correction=1)

    x3, y3 = x.squeeze(0), y.squeeze(0)
    err = (mean - y3).abs()

    # Whole-volume (union) metrics — the repo's headline convention — plus the
    # identity baseline on the same voxels.
    m_union = union_mask(x3, y3)
    mm = compute_metrics(mean, y3, m_union)
    bm = compute_metrics(x3, y3, m_union)
    mae_single = float((preds[0] - y3).abs()[m_union.bool()].mean())

    # Brain-only view for the voxel-level signal: the union mask is ~71%
    # background where error and std are both trivially ~0 and would inflate
    # any correlation.
    fg = torch.from_numpy(foreground_mask(x3, y3))
    e_fg = err[fg].numpy()
    s_fg = std[fg].numpy()
    vox_pearson = float(pearsonr(s_fg, e_fg).statistic)
    vox_spearman = float(spearmanr(s_fg, e_fg).statistic)
    sp = sparsification(e_fg, s_fg)

    # The same at the coherent scale: raw |error| is roughly half high-frequency
    # registration speckle (see the change-signal diagnosis) that no ensemble
    # can rank, so also score error↔std after a σ=``--smooth_sigma`` Gaussian —
    # co-localisation of the coherent error with the coherent uncertainty.
    blur = lambda v: gaussian_blur3d(v[None, None], args.smooth_sigma)[0, 0]
    err_sm, std_sm = blur(err), blur(std)
    e_sm, s_sm = err_sm[fg].numpy(), std_sm[fg].numpy()
    vox_pearson_sm = float(pearsonr(s_sm, e_sm).statistic)
    vox_spearman_sm = float(spearmanr(s_sm, e_sm).statistic)
    sp_sm = sparsification(e_sm, s_sm)

    # The change ROI — the top-5% most-changed brain voxels, where ~8× the
    # error density lives — is the hardest place for the signal to work.
    roi = torch.from_numpy(change_mask(x3, y3))
    roi_pearson = float(pearsonr(std[roi].numpy(), err[roi].numpy()).statistic) \
        if int(roi.sum()) > 10 else float("nan")

    # Subject-level uncertainty scores. Deployment-valid: computed from the
    # ensemble std over the PRE-op brain only — no ground truth touched.
    fg_pre = torch.from_numpy(foreground_mask(x3))
    s_pre = std[fg_pre].numpy()
    scores = {
        "unc_mean": float(s_pre.mean()),
        "unc_p99": float(np.quantile(s_pre, 0.99)),
        "unc_top1": float(np.sort(s_pre)[-max(1, s_pre.size // 100):].mean()),
        # Not an uncertainty: the size of the edit the model attempted. A
        # GT-free competitor — if it flags failures as well as the ensemble
        # std does, the variance adds nothing over "big edit = risky".
        "pred_change": float((mean - x3).abs()[fg_pre].mean()),
    }

    row = {
        "id": sid, **{k: mm[k] for k in ("mae", "mse", "psnr", "ssim")},
        **{f"identity_{k}": bm[k] for k in ("mae", "mse", "psnr", "ssim")},
        "mae_single": mae_single,
        "mae_fg": float(e_fg.mean()),
        "identity_mae_fg": float((x3 - y3).abs()[fg].mean()),
        **scores,
        "vox_pearson": vox_pearson, "vox_spearman": vox_spearman,
        "vox_ause": sp["ause"], "vox_ause_random": sp["ause_random"],
        "vox_pearson_sm": vox_pearson_sm, "vox_spearman_sm": vox_spearman_sm,
        "vox_ause_sm": sp_sm["ause"], "vox_ause_random_sm": sp_sm["ause_random"],
        "roi_pearson": roi_pearson,
    }
    extras = {"sp_curve": sp, "sp_curve_sm": sp_sm,
              "x": x3.numpy(), "y": y3.numpy(), "mean": mean.numpy(),
              "err_sm": err_sm.numpy(), "std_sm": std_sm.numpy(),
              "fg": fg.numpy()}
    return row, extras


# ─────────────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────────────

def _style(ax):
    """Recessive axes: no top/right spines, light grid behind the data."""
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, color="#e8e7e2", linewidth=0.8)
    ax.set_axisbelow(True)


TRIAGE_SCORES = (("unc_mean", "ensemble variance (unc_mean)", C_MODEL),
                 ("pred_change", "attempted-edit size (pred_change)", C_ALT))


def fig_trust_triage(rows, worst, out_path, budget):
    """The killer figure: review by a GT-free trust score, catch the failures.

    Both deployment-valid scores are shown — the ensemble variance the paper is
    about, and the attempted-edit size it largely proxies (they are strongly
    rank-correlated; the edit size is the stronger subject-level flag). Left —
    recall of the worst-``worst_frac`` predictions vs review budget, with the
    chance diagonal and the operating point at ``budget`` marked. Right — mean
    MAE of the auto-accepted remainder vs the same budget, with the true-error
    oracle and random triage as bounds.
    """
    n = len(rows)
    mae = np.array([r["mae"] for r in rows])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    revs = np.arange(0, n + 1) / n
    k = int(np.ceil(budget * n))
    for si, (key, name, color) in enumerate(TRIAGE_SCORES):
        score = np.array([r[key] for r in rows])
        order = np.argsort(-score, kind="stable")
        caught = np.concatenate([[0], np.cumsum(worst[order])]) / worst.sum()
        ax.plot(revs, caught, color=color, linewidth=2, label=name)
        ax.plot(k / n, caught[k], "o", color=color, markersize=8)
        ax.annotate(f"{caught[k]:.0%} caught at {budget:.0%} review",
                    (k / n, caught[k]), textcoords="offset points",
                    xytext=(10, -4 - 14 * si), fontsize=9, color=C_INK)
    ax.plot([0, 1], [0, 1], color=C_RANDOM, linewidth=1.5, linestyle=":",
            label="random triage")
    ax.axvline(budget, color=C_RANDOM, linewidth=1, linestyle="--")
    ax.set_xlabel("fraction of predictions sent for review\n(highest score first)")
    ax.set_ylabel("fraction of true failures caught")
    ax.set_title("The model triages its own outputs", fontsize=11, color=C_INK)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    _style(ax)

    ax = axes[1]
    for key, name, color in TRIAGE_SCORES:
        score = np.array([r[key] for r in rows])
        rc = retention_curve(score, mae)
        ax.plot(rc["fracs"], rc["by_uncertainty"], color=color, linewidth=2,
                label=f"triage by {name.split(' (')[0]}")
    rc = retention_curve(mae, mae)
    ax.plot(rc["fracs"], rc["by_error"], color=C_ORACLE, linewidth=1.5,
            linestyle="--", label="oracle (triage by true error)")
    ax.plot(rc["fracs"], rc["random"], color=C_RANDOM, linewidth=1.5,
            linestyle=":", label="random triage")
    ax.axvline(budget, color=C_RANDOM, linewidth=1, linestyle="--")
    ax.set_xlabel("fraction of predictions sent for review")
    ax.set_ylabel("MAE of auto-accepted predictions")
    ax.set_title("Error left in the automated stream", fontsize=11, color=C_INK)
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    _style(ax)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def fig_sparsification(curves, curves_sm, smooth_sigma, out_path):
    """Voxel-level sparsification, averaged over subjects (shared frac grid).

    Two panels: raw per-voxel maps, and the σ-smoothed maps where the
    registration speckle — unrankable by construction — is suppressed.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    panels = ((axes[0], curves, "raw voxels"),
              (axes[1], curves_sm, f"coherent scale (σ={smooth_sigma:g} smooth)"))
    for ax, cs, tag in panels:
        fracs = cs[0]["fracs"]
        cu = np.nanmean([c["by_uncertainty"] for c in cs], axis=0)
        ce = np.nanmean([c["by_error"] for c in cs], axis=0)
        ause = float(np.nanmean([c["ause"] for c in cs]))
        ause_rand = float(np.nanmean([c["ause_random"] for c in cs]))
        ax.plot(fracs, cu, color=C_MODEL, linewidth=2,
                label="remove by ensemble std")
        ax.plot(fracs, ce, color=C_ORACLE, linewidth=1.5, linestyle="--",
                label="oracle (remove by true error)")
        ax.axhline(1.0, color=C_RANDOM, linewidth=1.5, linestyle=":",
                   label="random removal")
        ax.fill_between(fracs, ce, cu, color=C_MODEL, alpha=0.12)
        ax.annotate(f"AUSE = {ause:.3f}\n(random = {ause_rand:.3f})",
                    (0.42, 0.6), xycoords="axes fraction", fontsize=10,
                    color=C_INK)
        ax.set_xlabel("fraction of brain voxels removed\n(most uncertain first)")
        ax.set_title(tag, fontsize=11, color=C_INK)
        _style(ax)
    axes[0].set_ylabel("relative MAE of remaining voxels")
    axes[0].legend(frameon=False, fontsize=9, loc="lower left")
    fig.suptitle("Where the ensemble disagrees is where it is wrong",
                 fontsize=12, color=C_INK)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def fig_subject_scatter(rows, worst, out_path, rho, ci):
    """Subject uncertainty score vs subject error, failures highlighted."""
    score = np.array([r[PRIMARY] for r in rows])
    mae = np.array([r["mae"] for r in rows])
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    ok = ~worst
    ax.scatter(score[ok], mae[ok], s=42, color=C_MODEL, alpha=0.75,
               edgecolors="white", linewidths=0.8, label="predictions")
    ax.scatter(score[worst], mae[worst], s=60, color=C_ALT, edgecolors="white",
               linewidths=0.8, label="true failures (worst 15% MAE)")
    ax.annotate(f"Spearman ρ = {rho:.2f}\n95% CI [{ci[0]:.2f}, {ci[1]:.2f}]",
                (0.04, 0.82), xycoords="axes fraction", fontsize=10, color=C_INK)
    ax.set_xlabel(f"subject uncertainty score ({PRIMARY}, no ground truth)")
    ax.set_ylabel("subject MAE (whole volume)")
    ax.set_title("Uncertainty predicts per-subject error", fontsize=11, color=C_INK)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    _style(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def fig_qualitative(picked, out_path):
    """Mid-axial slices: pre / post / prediction / |error| / uncertainty.

    Error and uncertainty use the magma sequential map, each scaled to its own
    99th percentile — the comparison is spatial pattern, not magnitude.
    """
    cols = ("pre-op", "post-op (truth)", "prediction (mean)",
            "|error| (smoothed)", "uncertainty (smoothed std)")
    fig, axes = plt.subplots(len(picked), 5,
                             figsize=(12.5, 2.7 * len(picked)))
    axes = np.atleast_2d(axes)
    for r, (tag, row, ex) in enumerate(picked):
        z = ex["x"].shape[0] // 2
        anat = [ex["x"][z], ex["y"][z], ex["mean"][z]]
        lo, hi = np.percentile(np.stack(anat), [1, 99])
        fgz = ex["fg"][z]
        for c, img in enumerate(anat + [ex["err_sm"][z], ex["std_sm"][z]]):
            ax = axes[r, c]
            if c < 3:
                ax.imshow(img, cmap="gray", vmin=lo, vmax=hi)
            else:
                # Brain-masked, per-map in-brain p2–p98 stretch: the std
                # rides a large constant floor (init noise the ~0 background
                # velocity never removes) and is several times smaller than
                # the error, so a [0, p99] scale renders it flat — the claim
                # is within-brain spatial co-localisation, not magnitude.
                if fgz.any():
                    vmin, vmax = np.percentile(img[fgz], [2, 98])
                else:
                    vmin, vmax = 0.0, 1.0
                shown = np.where(fgz, img, vmin)
                ax.imshow(shown, cmap="magma", vmin=vmin, vmax=vmax)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if r == 0:
                ax.set_title(cols[c], fontsize=10, color=C_INK)
        axes[r, 0].set_ylabel(
            f"{tag}\n{row['id']}\nr={row['vox_pearson_sm']:.2f}",
            fontsize=9, color=C_INK)
    fig.suptitle("Uncertainty co-localises with error "
                 "(smoothed maps; per-voxel Pearson r at the coherent scale)",
                 fontsize=12, color=C_INK)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",    type=str, default="fmri")
    p.add_argument("--ckpt",         type=str, default="runs/flow3d/best_mae.pt")
    p.add_argument("--out_dir",      type=str, default=None,
                   help="Default: outputs/uncertainty/<run name>")
    p.add_argument("--pre_dirname",  type=str, default="pre_surgery")
    p.add_argument("--post_dirname", type=str, default="6_months_post_surgery")
    p.add_argument("--full", action="store_true",
                   help="All 235 subjects instead of the held-out split "
                        "(mixes training subjects in — context only, not the "
                        "headline)")
    p.add_argument("--limit", type=int, default=None,
                   help="Evaluate only the first N subjects of the scope "
                        "(smoke tests)")
    # ensemble
    p.add_argument("--sampler", type=str, default="ode", choices=["ode", "sde"],
                   help="ode: deterministic ODE + init_noise perturbation. "
                        "sde: score-derived Langevin SDE on the trained path "
                        "(σ is the stochasticity; init_noise unused)")
    p.add_argument("--sde_gamma_scale", type=float, default=1.0,
                   help="γ = scale·σ² in the SDE churn term; 1 = unit "
                        "relaxation over the run")
    p.add_argument("--n_samples",  type=int, default=16,
                   help="Ensemble members per subject")
    p.add_argument("--init_noise", type=float, default=None,
                   help="Std of the x(0) perturbation. Default: the σ the "
                        "checkpoint was trained with (in-distribution at t=0, "
                        "where the training path is x_pre + σ·ε)")
    p.add_argument("--chunk",      type=int, default=8,
                   help="Ensemble members per forward batch")
    p.add_argument("--sweep_init_noise", type=float, nargs="+", default=None,
                   help="Evaluate the trust signal at each of these init_noise "
                        "values and print a table instead of the full run")
    p.add_argument("--steps",  type=int, default=None)
    p.add_argument("--solver", type=str, default=None,
                   choices=["euler", "heun", "rk4"])
    p.add_argument("--guidance_scale", type=float, default=None)
    p.add_argument("--sample_seed", type=int, default=0)
    p.add_argument("--use_ema", dest="use_ema", action="store_true")
    p.add_argument("--no-ema",  dest="use_ema", action="store_false")
    p.set_defaults(use_ema=True)
    # trust-signal definitions
    p.add_argument("--smooth_sigma", type=float, default=2.0,
                   help="Gaussian σ (voxels) for the coherent-scale voxel "
                        "analysis — matches the coherent-change target")
    p.add_argument("--worst_frac", type=float, default=0.15,
                   help="'Failure' = this fraction of subjects with worst MAE")
    p.add_argument("--budget", type=float, default=0.15,
                   help="Review budget for the recall-at-budget headline")
    p.add_argument("--n_qual", type=int, default=4,
                   help="Subjects in the qualitative panel figure")
    p.add_argument("--no-figs", dest="figs", action="store_false")
    p.set_defaults(figs=True)
    args = p.parse_args()

    args.device = get_device()
    model, raw = load_flow3d_checkpoint(args.ckpt, device=args.device,
                                        use_ema=args.use_ema)
    ck = raw.get("args", {})
    steps = args.steps or ck.get("steps", 8)
    args.solver = args.solver or ck.get("solver", "heun")
    if args.guidance_scale is None:
        args.guidance_scale = ck.get("guidance_scale", 1.0)
    if args.init_noise is None:
        args.init_noise = float(ck.get("sigma", 0.3))

    print(f"Checkpoint: {args.ckpt}  ({'EMA' if args.use_ema else 'online'} weights)")
    noise_desc = (f"gamma_scale={args.sde_gamma_scale} (σ={ck.get('sigma')})"
                  if args.sampler == "sde" else f"init_noise={args.init_noise}")
    print(f"Ensemble: sampler={args.sampler} K={args.n_samples} {noise_desc} "
          f"solver={args.solver} steps={steps} chunk={args.chunk} "
          f"device={args.device}")

    zero_bg = ck.get("zero_background", False)
    ds = PrePostFMRI(root_dir=args.data_root, pre_dirname=args.pre_dirname,
                     post_dirname=args.post_dirname,
                     transform=ToChannelsFirstAndNormalize(
                         nonzero_mask=True, zero_background=zero_bg),
                     strict=False, return_paths=True)
    if args.full:
        indices, scope = list(range(len(ds))), "full dataset (incl. training)"
    else:
        seed, vf = ck.get("seed", 42), ck.get("val_frac", 0.15)
        _, val = reconstruct_val_split(ds, vf, seed, n_folds=ck.get("n_folds"),
                                       fold=ck.get("fold"))
        indices = list(val.indices)
        scope = f"held-out validation split (seed={seed}, val_frac={vf})"
    if args.limit:
        indices = indices[:args.limit]
        scope += f" [first {len(indices)}]"
    print(f"Scope: {scope} — {len(indices)} subjects\n")

    suffix = "_sde" if args.sampler == "sde" else ""
    out_dir = Path(args.out_dir or
                   f"outputs/uncertainty/{Path(args.ckpt).parent.name}{suffix}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── sweep mode: pick the init_noise operating point ──────────────────────
    if args.sweep_init_noise:
        print(f"{'noise':>6} {'MAE':>8} {'MAEid':>8} {'voxSp':>7} {'AUSE':>7} "
              f"{'subjSp':>7} {'AUROC':>6}")
        sweep = {}
        for nv in args.sweep_init_noise:
            args.init_noise = float(nv)
            rows = []
            for i in indices:
                x, y, meta = ds[i]
                row, _ = evaluate_subject(model, x, y, meta["id"], args, steps)
                rows.append(row)
            mae = np.array([r["mae"] for r in rows])
            unc = np.array([r[PRIMARY] for r in rows])
            k = max(1, int(np.ceil(args.worst_frac * len(rows))))
            worst = mae >= np.sort(mae)[-k]
            agg = {
                "mae": float(mae.mean()),
                "identity_mae": float(np.mean([r["identity_mae"] for r in rows])),
                "vox_spearman": float(np.mean([r["vox_spearman"] for r in rows])),
                "ause": float(np.mean([r["vox_ause"] for r in rows])),
                "subj_spearman": float(spearmanr(unc, mae).statistic),
                "auroc_worst": auroc(unc, worst),
            }
            sweep[str(nv)] = agg
            print(f"{nv:>6} {agg['mae']:>8.4f} {agg['identity_mae']:>8.4f} "
                  f"{agg['vox_spearman']:>7.3f} {agg['ause']:>7.3f} "
                  f"{agg['subj_spearman']:>7.3f} {agg['auroc_worst']:>6.3f}")
        with open(out_dir / "init_noise_sweep.json", "w") as f:
            json.dump({"scope": scope, "n_samples": args.n_samples,
                       "sweep": sweep}, f, indent=2)
        print(f"\nSweep saved: {out_dir / 'init_noise_sweep.json'}")
        return

    # ── main evaluation ──────────────────────────────────────────────────────
    rows, extras = [], []
    for n, i in enumerate(indices, 1):
        x, y, meta = ds[i]
        row, ex = evaluate_subject(model, x, y, meta["id"], args, steps)
        rows.append(row)
        extras.append(ex)
        if n % 5 == 0 or n == len(indices):
            print(f"  [{n}/{len(indices)}] MAE={np.mean([r['mae'] for r in rows]):.4f} "
                  f"vox r={np.mean([r['vox_pearson'] for r in rows]):.3f} "
                  f"AUSE={np.mean([r['vox_ause'] for r in rows]):.3f}")

    mae = np.array([r["mae"] for r in rows])
    ident = np.array([r["identity_mae"] for r in rows])
    k = max(1, int(np.ceil(args.worst_frac * len(rows))))
    worst = mae >= np.sort(mae)[-k]                # worst k by whole-volume MAE
    loses = mae > ident                            # loses to copy-x_pre

    # Subject-level trust signal, every candidate score.
    subj = {}
    for sc in SCORES:
        u = np.array([r[sc] for r in rows])
        subj[sc] = {
            "spearman_vs_mae": float(spearmanr(u, mae).statistic),
            "pearson_vs_mae": float(pearsonr(u, mae).statistic),
            "auroc_worst": auroc(u, worst),
            "auroc_loses_identity": auroc(u, loses),
            f"recall_at_{args.budget:.0%}_worst":
                recall_at_budget(u, worst, args.budget),
            f"recall_at_{args.budget:.0%}_loses_identity":
                recall_at_budget(u, loses, args.budget),
        }
    u = np.array([r[PRIMARY] for r in rows])
    primary_ci = {
        "spearman_vs_mae":
            bootstrap_ci(lambda a, b: spearmanr(a, b).statistic, u, mae),
        "auroc_worst": bootstrap_ci(auroc, u, worst),
        f"recall_at_{args.budget:.0%}_worst":
            bootstrap_ci(lambda a, b: recall_at_budget(a, b, args.budget),
                         u, worst),
    }
    upc = np.array([r["pred_change"] for r in rows])
    pred_change_ci = {
        "spearman_vs_mae":
            bootstrap_ci(lambda a, b: spearmanr(a, b).statistic, upc, mae),
        "auroc_worst": bootstrap_ci(auroc, upc, worst),
        f"recall_at_{args.budget:.0%}_worst":
            bootstrap_ci(lambda a, b: recall_at_budget(a, b, args.budget),
                         upc, worst),
    }

    vox = {k2: float(np.nanmean([r[k2] for r in rows]))
           for k2 in ("vox_pearson", "vox_spearman", "vox_ause",
                      "vox_ause_random", "vox_pearson_sm", "vox_spearman_sm",
                      "vox_ause_sm", "vox_ause_random_sm", "roi_pearson")}
    # Fraction of the oracle's achievable error-removal the ensemble ranking
    # actually delivers (1 = perfect ranking, 0 = uninformative).
    vox["rel_ause_gain"] = 1.0 - vox["vox_ause"] / vox["vox_ause_random"]
    vox["rel_ause_gain_sm"] = 1.0 - vox["vox_ause_sm"] / vox["vox_ause_random_sm"]

    # ── report ───────────────────────────────────────────────────────────────
    print(f"\n=== {scope} — n={len(rows)}, K={args.n_samples}, "
          f"sampler={args.sampler} ({noise_desc}) ===")
    print(f"MAE: ensemble mean {mae.mean():.4f}  single sample "
          f"{np.mean([r['mae_single'] for r in rows]):.4f}  identity "
          f"{ident.mean():.4f}  (wins {int((mae < ident).sum())}/{len(rows)})")
    print(f"\nVoxel level (brain): Pearson r={vox['vox_pearson']:.3f}  "
          f"Spearman={vox['vox_spearman']:.3f}  AUSE={vox['vox_ause']:.3f} "
          f"(random={vox['vox_ause_random']:.3f}, gain "
          f"{vox['rel_ause_gain']:.1%})  change-ROI r={vox['roi_pearson']:.3f}")
    print(f"  coherent scale (σ={args.smooth_sigma:g}): "
          f"Pearson r={vox['vox_pearson_sm']:.3f}  "
          f"Spearman={vox['vox_spearman_sm']:.3f}  "
          f"AUSE={vox['vox_ause_sm']:.3f} "
          f"(random={vox['vox_ause_random_sm']:.3f}, gain "
          f"{vox['rel_ause_gain_sm']:.1%})")
    print(f"\nSubject level ({len(rows)} subjects, {int(worst.sum())} worst-"
          f"{args.worst_frac:.0%} failures, {int(loses.sum())} lose to identity):")
    print(f"{'score':>10} {'ρ(U,MAE)':>9} {'AUROC-worst':>12} "
          f"{'AUROC-loses':>12} {'recall@' + format(args.budget, '.0%'):>10}")
    for sc in SCORES:
        s = subj[sc]
        print(f"{sc:>10} {s['spearman_vs_mae']:>9.3f} {s['auroc_worst']:>12.3f} "
              f"{s['auroc_loses_identity']:>12.3f} "
              f"{s[f'recall_at_{args.budget:.0%}_worst']:>10.3f}")
    print(f"\nPrimary ({PRIMARY}) 95% bootstrap CIs: "
          + "  ".join(f"{k2} [{v[0]:.3f}, {v[1]:.3f}]"
                      for k2, v in primary_ci.items()))
    print("pred_change 95% bootstrap CIs: "
          + "  ".join(f"{k2} [{v[0]:.3f}, {v[1]:.3f}]"
                      for k2, v in pred_change_ci.items()))

    # ── outputs ──────────────────────────────────────────────────────────────
    with open(out_dir / "per_subject.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    summary = {
        "timestamp": datetime.now().isoformat(),
        "ckpt": args.ckpt, "use_ema": args.use_ema, "scope": scope,
        "n_subjects": len(rows),
        "ensemble": {"sampler": args.sampler, "n_samples": args.n_samples,
                     "init_noise": (None if args.sampler == "sde"
                                    else args.init_noise),
                     "sde_gamma_scale": (args.sde_gamma_scale
                                         if args.sampler == "sde" else None),
                     "solver": args.solver, "steps": steps,
                     "chunk": args.chunk, "sample_seed": args.sample_seed,
                     "smooth_sigma": args.smooth_sigma},
        "failure_defs": {"worst_frac": args.worst_frac,
                         "n_worst": int(worst.sum()),
                         "n_loses_identity": int(loses.sum()),
                         "budget": args.budget},
        "accuracy": {"mae_ensemble": float(mae.mean()),
                     "mae_single": float(np.mean([r["mae_single"] for r in rows])),
                     "mae_identity": float(ident.mean()),
                     "mae_wins_vs_identity": int((mae < ident).sum())},
        "voxel_level": vox,
        "subject_level": subj,
        "primary_score": PRIMARY,
        "primary_bootstrap_ci95": primary_ci,
        "pred_change_bootstrap_ci95": pred_change_ci,
        "train_args": ck,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    if args.figs:
        fig_trust_triage(rows, worst, out_dir / "fig_trust_triage.png",
                         args.budget)
        fig_sparsification([e["sp_curve"] for e in extras],
                           [e["sp_curve_sm"] for e in extras],
                           args.smooth_sigma, out_dir / "fig_sparsification.png")
        rho = subj[PRIMARY]["spearman_vs_mae"]
        fig_subject_scatter(rows, worst, out_dir / "fig_subject_scatter.png",
                            rho, primary_ci["spearman_vs_mae"])
        # Qualitative rows: most/least uncertain + worst error + median error.
        by_unc = np.argsort(-u, kind="stable")
        by_err = np.argsort(-mae, kind="stable")
        want = [("most uncertain", by_unc[0]), ("worst error", by_err[0]),
                ("median error", by_err[len(rows) // 2]),
                ("least uncertain", by_unc[-1])]
        seen, picked = set(), []
        for tag, i in want:
            if i not in seen and len(picked) < args.n_qual:
                seen.add(i)
                picked.append((tag, rows[i], extras[i]))
        fig_qualitative(picked, out_dir / "fig_qualitative.png")
        print(f"\nFigures: {out_dir}/fig_*.png")

    print(f"Per-subject CSV: {out_dir / 'per_subject.csv'}")
    print(f"Summary: {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
