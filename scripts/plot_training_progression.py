#!/usr/bin/env python
"""
All five model runs on a single training-progression figure — every logged
quantity, both stages, no re-training (reads each run's metrics_stage{1,2}.csv).

Layout (3x3), color = model (consistent with the comparison boxplots, winner
highlighted), one shared legend in the open top-right cell:

  Stage 1 · Autoencoder        AE train loss | AE val loss | [legend]
  Stage 2 · Latent diffusion   diff train    | diff val    | MAE
                               MSE           | PSNR        | SSIM

Usage:
  python scripts/plot_training_progression.py --out outputs/compare_5models/training_progression_all.png
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
import seaborn as sns

# run dir -> publication name, in left-to-right / ascending-richness order
RUNS = [
    ("runs/cfg1_baseline",         "cldm-base"),
    ("runs/cfg2_big_denoiser",     "cldm-big"),
    ("runs/cfg3_rich_latent",      "cldm-rich"),
    ("runs/cfg5_richae_strongcfg", "cldm-rich-strong"),
    ("runs/cfg5_strong_cfg",       "cldm-strong"),
]

SEQ_CMAP = "crest"
WINNER_COLOR = "#D1495B"

# panel spec: (stage, column, title, log-y?, robust-ylim?)
S1 = [
    ("stage1", "train_loss", "AE train loss",   True,  False),
    ("stage1", "val_loss",   "AE val loss",     True,  False),
]
S2 = [
    ("stage2", "train_loss", "Diffusion train loss", True,  False),
    ("stage2", "val_loss",   "Diffusion val loss",   True,  False),
    ("stage2", "mae",        "MAE  ↓",          False, True),
    ("stage2", "mse",        "MSE  ↓",          False, True),
    ("stage2", "psnr",       "PSNR  ↑ (dB)",    False, False),
    ("stage2", "ssim",       "SSIM  ↑",         False, False),
]


def build_palette(order):
    *rest, winner = order
    rest_cols = sns.color_palette(SEQ_CMAP, n_colors=max(len(rest), 1))
    pal = dict(zip(rest, rest_cols))
    pal[winner] = WINNER_COLOR
    return pal, winner


def set_style():
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor":   "white",
        "savefig.dpi":      300,
        "font.family":      "sans-serif",
        "font.sans-serif":  ["Helvetica", "Arial", "DejaVu Sans"],
        "axes.titlesize":   14,
        "axes.titleweight": "bold",
        "axes.labelsize":   12,
        "xtick.labelsize":  11,
        "ytick.labelsize":  11,
        "axes.edgecolor":   "#444444",
        "axes.linewidth":   1.0,
        "grid.color":       "#DCDCDC",
        "grid.linewidth":   0.7,
    })


def load_runs():
    data = {}
    for run_dir, label in RUNS:
        d = {}
        for stage in ("stage1", "stage2"):
            f = Path(run_dir) / f"metrics_{stage}.csv"
            if f.exists():
                d[stage] = pd.read_csv(f)
        data[label] = d
    return data


def draw_panel(ax, data, order, palette, winner, spec):
    stage, col, title, logy, robust = spec
    pooled = []
    for label in order:
        df = data.get(label, {}).get(stage)
        if df is None or col not in df.columns:
            continue
        ep, y = df["epoch"].to_numpy(), df[col].to_numpy(dtype=float)
        if len(y) == 0 or np.all(np.isnan(y)):
            continue
        w = max(1, len(y) // 60)
        ys = pd.Series(y).rolling(w, center=True, min_periods=1).mean().to_numpy()
        is_win = label == winner
        if w > 1:                                  # faint raw trace behind smoothed
            ax.plot(ep, y, color=palette[label], alpha=0.07, lw=0.7, zorder=1)
        kw = dict(color=palette[label], solid_capstyle="round")
        if is_win:
            ax.plot(ep, ys, lw=3.0, alpha=1.0, zorder=6,
                    path_effects=[pe.Stroke(linewidth=4.6, foreground="white"),
                                  pe.Normal()], **kw)
        else:
            ax.plot(ep, ys, lw=1.7, alpha=0.9, zorder=3, **kw)
        pooled.append(ys)

    ax.set_title(title, pad=6)
    ax.set_xlabel("Epoch", labelpad=1)
    ax.grid(True, alpha=0.6)
    if logy:
        ax.set_yscale("log")
    elif robust and pooled:
        allv = np.concatenate(pooled)
        lo, hi = np.nanpercentile(allv, [2, 98])
        if hi > lo:
            pad = 0.06 * (hi - lo)
            ax.set_ylim(lo - pad, hi + pad)
    sns.despine(ax=ax)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/compare_5models/training_progression_all.png")
    ap.add_argument("--title", default="Training progression across model variants")
    args = ap.parse_args()

    order = [label for _, label in RUNS]
    data = load_runs()
    palette, winner = build_palette(order)

    set_style()
    fig, axes = plt.subplots(3, 3, figsize=(14.5, 11.2))
    fig.subplots_adjust(left=0.065, right=0.985, top=0.825, bottom=0.062,
                        hspace=0.50, wspace=0.255)

    # Stage 1 -> row 0 cols 0,1 ; legend -> (0,2) ; Stage 2 -> remaining 6 cells
    s2_cells = [axes[1, 0], axes[1, 1], axes[1, 2],
                axes[2, 0], axes[2, 1], axes[2, 2]]
    for ax, spec in zip([axes[0, 0], axes[0, 1]], S1):
        draw_panel(ax, data, order, palette, winner, spec)
    for ax, spec in zip(s2_cells, S2):
        draw_panel(ax, data, order, palette, winner, spec)

    # legend lives in the open (0,2) cell
    leg_ax = axes[0, 2]
    leg_ax.axis("off")
    handles = [Line2D([0], [0], color=palette[m],
                      lw=3.4 if m == winner else 2.0,
                      path_effects=[pe.Stroke(linewidth=5.0, foreground="white"),
                                    pe.Normal()] if m == winner else None)
               for m in order]
    leg = leg_ax.legend(handles, order, title="Model", loc="center",
                        frameon=False, fontsize=13, title_fontsize=13.5,
                        handlelength=1.8, labelspacing=0.7, borderaxespad=0)
    leg.get_title().set_fontweight("bold")
    for txt in leg.get_texts():
        if txt.get_text() == winner:
            txt.set_fontweight("bold")

    # band labels + divider between Stage 1 (row 0) and Stage 2 (rows 1-2)
    p0 = axes[0, 0].get_position()
    p1 = axes[1, 0].get_position()
    y_div = (p0.y0 + p1.y1) / 2
    fig.add_artist(Line2D([0.03, 0.985], [y_div, y_div], color="#BBBBBB",
                          lw=1.2, linestyle=(0, (6, 4))))
    fig.text(0.065, p0.y1 + 0.030, "Stage 1  ·  Autoencoder reconstruction",
             ha="left", va="bottom", fontsize=15, fontweight="bold",
             color="#3A3A3A")
    fig.text(0.065, p1.y1 + 0.026, "Stage 2  ·  Conditional latent diffusion",
             ha="left", va="bottom", fontsize=15, fontweight="bold",
             color="#3A3A3A")

    fig.suptitle(args.title, x=0.5, y=0.975, fontsize=20, fontweight="bold")
    fig.text(0.5, 0.938,
             "smoothed (moving avg) over faint raw curves   ·   losses on log scale"
             "   ·   MAE/MSE y-axis clipped to 2–98th pct",
             ha="center", fontsize=11.5, color="#666666")
    fig.text(0.5, 0.012,
             "cldm-rich-strong reuses a Stage-1 autoencoder, so it has no Stage-1 curve.",
             ha="center", fontsize=10, color="#888888", style="italic")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
