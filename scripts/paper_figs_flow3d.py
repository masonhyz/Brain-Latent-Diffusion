"""Paper figures for the flow-matching + uncertainty sections.

Two figures that do not come out of the evaluation scripts:

  ``flow_config_ablation.png`` — validation MAE per epoch for the flow-matching
      training configurations, expressed as a *ratio to each run's own identity
      baseline* (the runs use different splits, so raw MAE is not comparable
      across them; the ratio is). A companion bar panel gives the conditioning
      ablation's train-loss trajectory, which is the cleanest evidence that
      conditioning is what makes the objective descend.

  ``flow_uncertainty_pipeline.png`` — schematic of the bridge + score-derived
      SDE ensemble and the two trust signals it produces.

Run from the repo root:  python scripts/paper_figs_flow3d.py --out_dir <dir>
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

RUNS_DIR = Path("/data1/masonhu/runs")

# Validated categorical palette (dataviz skill): blue = the selected model,
# orange = the second series, remaining slots for the ablation arms.
C_BLUE, C_ORANGE, C_AQUA = "#2a78d6", "#eb6834", "#1baf7a"
C_YELLOW, C_MAGENTA = "#eda100", "#e87ba4"
C_INK, C_MUTE, C_GRID = "#3d3d3a", "#6f6e66", "#e8e7e2"

# (run dir, legend label, colour, linewidth) — the fully-trained arms.
ARMS = [
    ("flow3d_2026-08-02_21-07-41", "bridge, $\\sigma$=0.3, uniform $t$ (selected)", C_BLUE, 2.2),
    ("flow3d_sharp",               "bridge, $\\sigma$=0.2, logit-normal $t$",       C_ORANGE, 1.6),
    ("flow3d_struct",              "bridge + L1 + SSIM terms",                      C_AQUA, 1.6),
    ("flow3d_nd_bridge",           "bridge + change-weighted MSE",                  C_YELLOW, 1.6),
    ("flow3d_noise_dual",          "source = noise (standard CFM)",                 C_MAGENTA, 1.6),
]


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, color=C_GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _load(run):
    h = json.load(open(RUNS_DIR / run / "hparams.json"))
    rows = list(csv.DictReader(open(RUNS_DIR / run / "metrics.csv")))
    return h, rows


def _series(rows, key):
    ep, val = [], []
    for r in rows:
        v = r.get(key)
        if v and v not in ("", "nan"):
            ep.append(int(r["epoch"]))
            val.append(float(v))
    return np.array(ep), np.array(val)


def fig_config_ablation(out_path):
    """Left: val MAE / identity MAE per epoch, per configuration. Right: the
    conditioning ablation's train-loss trajectory."""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.1),
                             gridspec_kw={"width_ratios": [1.55, 1]})

    ax = axes[0]
    top = 1.42
    for run, label, color, lw in ARMS:
        h, rows = _load(run)
        ident = h["identity_baseline"]["mae"]
        ep, mae = _series(rows, "mae")
        if ep.size == 0:
            continue
        ratio = mae / ident
        if ratio.max() > top:
            # This arm never approaches the baseline; clip it to the axis and
            # say so rather than stretching the scale for a curve that is
            # uniformly off the chart.
            label += f" \u2014 off scale ({ratio.min():.1f}\u2013{ratio.max():.1f})"
            ratio = np.clip(ratio, None, top)
        ax.plot(ep, ratio, color=color, linewidth=lw, label=label)
    ax.axhline(1.0, color=C_MUTE, linewidth=1.5, linestyle="--")
    ax.annotate("identity baseline (copy $x_{\\mathrm{pre}}$)", (0.985, 1.0),
                xycoords=("axes fraction", "data"), ha="right", va="bottom",
                fontsize=8.5, color=C_MUTE)
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation MAE / identity MAE")
    ax.set_ylim(0.75, 1.46)
    ax.set_title("Only the conditioned bridge beats copying the input",
                 fontsize=10.5, color=C_INK)
    # Upper band is empty (every in-scale arm sits below 1.1), so the legend
    # goes there rather than over the selected model's curve.
    ax.legend(frameon=False, fontsize=7.6, loc="upper right",
              bbox_to_anchor=(1.0, 0.90))
    _style(ax)

    ax = axes[1]
    for run, label, color in [
            ("flow3d_2026-08-02_20-25-21", "condition = concat($x_{\\mathrm{pre}}$)", C_BLUE),
            ("flow3d_2026-08-02_16-54-37", "condition = none", C_ORANGE)]:
        _, rows = _load(run)
        ep, tl = _series(rows, "train_loss")
        n = min(8, ep.size)
        ax.plot(ep[:n], tl[:n], color=color, linewidth=2.2, marker="o",
                markersize=4, label=label)
    ax.set_xlabel("epoch")
    ax.set_ylabel("training loss")
    ax.set_title("Conditioning is what makes the loss descend",
                 fontsize=10.5, color=C_INK)
    ax.legend(frameon=False, fontsize=8.5, loc="center right")
    _style(ax)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def _box(ax, xy, w, h, text, face, edge, fontsize=8.5, weight="normal"):
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                                facecolor=face, edgecolor=edge, linewidth=1.3))
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center",
            fontsize=fontsize, color=C_INK, weight=weight, linespacing=1.45)


def _arrow(ax, a, b, color=C_MUTE, style="-|>", lw=1.5, ls="-"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=13,
                                 color=color, linewidth=lw, linestyle=ls,
                                 shrinkA=2, shrinkB=2))


def fig_pipeline(out_path):
    """Schematic: the pre->post bridge, the score-derived SDE ensemble, and the
    two trust signals read off the ensemble."""
    fig, ax = plt.subplots(figsize=(11.5, 4.0))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    tint_b, tint_o, tint_g = "#eaf2fc", "#fdeee7", "#f2f1ed"

    # ── row 1: the bridge / velocity field ───────────────────────────────────
    ax.text(0.005, 0.955, "A.  Conditional flow matching as a pre$\\rightarrow$post bridge",
            fontsize=10, color=C_INK, weight="bold")
    _box(ax, (0.01, 0.60), 0.135, 0.24,
         "pre-op CBF\n$x_{\\mathrm{pre}}$", tint_b, C_BLUE, weight="bold")
    _box(ax, (0.20, 0.60), 0.20, 0.24,
         "interpolant\n$x_t=(1{-}t)x_{\\mathrm{pre}}+t\\,x_{\\mathrm{post}}+\\sigma\\varepsilon$",
         tint_g, C_MUTE, fontsize=8)
    _box(ax, (0.45, 0.60), 0.20, 0.24,
         "3D U-Net velocity\n$v_\\theta(x_t,t \\mid x_{\\mathrm{pre}})$", tint_b, C_BLUE)
    _box(ax, (0.70, 0.60), 0.135, 0.24,
         "target\n$u_t=x_{\\mathrm{post}}-x_{\\mathrm{pre}}$", tint_g, C_MUTE, fontsize=8)
    _box(ax, (0.865, 0.60), 0.125, 0.24,
         "$\\mathcal{L}=\\|v_\\theta-u_t\\|^2$", tint_g, C_MUTE, fontsize=8.5)
    for a, b in [((0.145, 0.72), (0.20, 0.72)), ((0.40, 0.72), (0.45, 0.72)),
                 ((0.65, 0.72), (0.70, 0.72)), ((0.835, 0.72), (0.865, 0.72))]:
        _arrow(ax, a, b)
    # conditioning path
    _arrow(ax, (0.078, 0.60), (0.078, 0.53), color=C_BLUE)
    ax.plot([0.078, 0.55], [0.53, 0.53], color=C_BLUE, linewidth=1.5)
    _arrow(ax, (0.55, 0.53), (0.55, 0.60), color=C_BLUE)
    ax.text(0.30, 0.545, "channel-concatenated conditioning", fontsize=7.8,
            color=C_BLUE, ha="center")

    # ── row 2: SDE ensemble → trust signals ──────────────────────────────────
    ax.text(0.005, 0.44, "B.  Inference: score-derived SDE ensemble $\\rightarrow$ two trust signals",
            fontsize=10, color=C_INK, weight="bold")
    _box(ax, (0.01, 0.06), 0.135, 0.30,
         "$x_{\\mathrm{pre}}$\n(new patient)", tint_b, C_BLUE, weight="bold")
    _box(ax, (0.185, 0.06), 0.245, 0.30,
         "$K$ SDE trajectories (trained $v_\\theta$)\n"
         "$dx=[v_\\theta+\\gamma s_\\theta]dt+\\sqrt{2\\gamma}\\,dW$\n"
         "$s_\\theta=-(x-x_{\\mathrm{pre}}-t v_\\theta)/\\sigma^2$",
         tint_o, C_ORANGE, fontsize=7.8)
    _box(ax, (0.47, 0.225, ), 0.20, 0.135,
         "prediction  $\\bar{x}=\\mathrm{mean}_K$", tint_b, C_BLUE, fontsize=8.5)
    _box(ax, (0.47, 0.06), 0.20, 0.135,
         "uncertainty  $u=\\mathrm{std}_K$", tint_o, C_ORANGE, fontsize=8.5)
    _box(ax, (0.715, 0.225), 0.275, 0.135,
         "voxel map: where not to trust\n(sparsification / AUSE)", tint_o, C_ORANGE, fontsize=8)
    _box(ax, (0.715, 0.06), 0.275, 0.135,
         "subject score: triage queue\n(AUROC / recall @ budget)", tint_o, C_ORANGE, fontsize=8)
    for a, b in [((0.145, 0.21), (0.185, 0.21)),
                 ((0.43, 0.21), (0.47, 0.29)), ((0.43, 0.21), (0.47, 0.13)),
                 ((0.67, 0.29), (0.715, 0.29)), ((0.67, 0.13), (0.715, 0.13))]:
        _arrow(ax, a, b, color=C_ORANGE)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", type=str,
                   default="/data1/masonhu/Neurips_genai4health/figures")
    a = p.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig_config_ablation(out / "flow_config_ablation.png")
    fig_pipeline(out / "flow_uncertainty_pipeline.png")


if __name__ == "__main__":
    main()
