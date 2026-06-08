#!/usr/bin/env python
"""
Publication-style grouped boxplots from a metrics_long.csv produced by
compare_models_boxplot.py — reads the CSV only, no re-evaluation.

One panel per metric (MAE / MSE / PSNR / SSIM); one colored box per model, with
jittered per-subject points, a mean marker, and a single shared legend. Models
are renamed to their publication names (see RENAME) and ordered by complexity.

Usage:
  python scripts/plot_compare_boxplots.py \
      --csv outputs/compare_5models/metrics_long.csv \
      --out outputs/compare_5models/compare_boxplots_pro.png
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns

# CSV label -> publication name. cfg5_richae = rich AE + strong CFG (rich-strong);
# cfg5_strong = strong CFG only.
RENAME = {
    "cfg1":        "cldm-base",
    "cfg2":        "cldm-big",
    "cfg3":        "cldm-rich",
    "cfg5_richae": "cldm-rich-strong",
    "cfg5_strong": "cldm-strong",
}
# left-to-right order (ascending model "richness")
ORDER = ["cldm-base", "cldm-big", "cldm-rich", "cldm-rich-strong", "cldm-strong"]

# metric key, display name, better-direction arrow, y-axis unit
METRICS = [
    ("mae",  "MAE",  "↓", ""),
    ("mse",  "MSE",  "↓", ""),
    ("psnr", "PSNR", "↑", "dB"),
    ("ssim", "SSIM", "↑", ""),
]

# Cool sequential map for the comparison models; the best model (last in ORDER)
# is pulled out in a warm accent so it reads as the highlighted winner.
SEQ_CMAP = "crest"
WINNER_COLOR = "#D1495B"


def build_palette(order):
    """Sequential cool colors for all-but-last; warm accent for the winner."""
    *rest, winner = order
    rest_cols = sns.color_palette(SEQ_CMAP, n_colors=max(len(rest), 1))
    pal = dict(zip(rest, rest_cols))
    pal[winner] = WINNER_COLOR
    return pal, winner


def set_style():
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update({
        "figure.facecolor":  "white",
        "axes.facecolor":    "white",
        "savefig.dpi":       300,
        "font.family":       "sans-serif",
        "font.sans-serif":   ["Helvetica", "Arial", "DejaVu Sans"],
        "axes.titlesize":    16,
        "axes.titleweight":  "bold",
        "axes.labelsize":    14,
        "xtick.labelsize":   13,
        "ytick.labelsize":   13,
        "axes.edgecolor":    "#444444",
        "axes.linewidth":    1.0,
        "grid.color":        "#D9D9D9",
        "grid.linewidth":    0.7,
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="metrics_long.csv path")
    ap.add_argument("--out", default=None,
                    help="output PNG (default: <csv dir>/compare_boxplots_pro.png)")
    ap.add_argument("--title", default="Latent-diffusion model comparison")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df["model"] = df["model"].map(lambda m: RENAME.get(m, m))
    seen = set(df["model"])
    order = [m for m in ORDER if m in seen] + sorted(seen - set(ORDER))
    df = df[df["model"].isin(order)].copy()
    palette, winner = build_palette(order)

    n = int(df.groupby("model")["id"].nunique().max())

    set_style()
    fig, axes = plt.subplots(1, len(METRICS),
                             figsize=(2.1 * len(METRICS) + 0.4, 5.1))
    axes = np.atleast_1d(axes)

    for ax, (key, name, arrow, unit) in zip(axes, METRICS):
        sub = df[["model", key]].rename(columns={key: "value"})
        sns.boxplot(
            data=sub, x="model", y="value", order=order,
            hue="model", hue_order=order, palette=palette, legend=False,
            width=0.74, saturation=1.0, fliersize=0, ax=ax,
            boxprops=dict(edgecolor="#2B2B2B", linewidth=1.0, alpha=0.92),
            medianprops=dict(color="#111111", linewidth=1.7),
            whiskerprops=dict(color="#555555", linewidth=1.0),
            capprops=dict(color="#555555", linewidth=1.0),
            showmeans=True,
            meanprops=dict(marker="D", markerfacecolor="white",
                           markeredgecolor="#111111", markersize=5,
                           markeredgewidth=1.0),
        )
        sns.stripplot(
            data=sub, x="model", y="value", order=order, ax=ax,
            color="#1A1A1A", size=2.2, alpha=0.28, jitter=0.12, linewidth=0,
        )
        # emphasize the winner's box (last position): bold opaque outline
        if ax.patches:
            wp = ax.patches[len(order) - 1]
            wp.set_alpha(1.0)
            wp.set_linewidth(2.2)
            wp.set_edgecolor("#111111")
        ax.set_title(f"{name}  {arrow}", pad=8)
        ax.set_xlabel("")
        ax.set_ylabel(unit, labelpad=2)
        ax.tick_params(axis="x", which="both", length=0, labelbottom=False)
        ax.tick_params(axis="y", pad=1)
        ax.grid(axis="y", alpha=0.6)
        ax.grid(axis="x", visible=False)
        ax.margins(x=0.04)
        sns.despine(ax=ax)

    handles = [Patch(facecolor=palette[m], edgecolor="#111111",
                     linewidth=2.2 if m == winner else 1.0, label=m)
               for m in order]
    leg = fig.legend(handles=handles, loc="lower center", ncol=len(order),
                     frameon=False, bbox_to_anchor=(0.5, 0.01),
                     handlelength=1.3, handleheight=1.3, columnspacing=1.4,
                     fontsize=13)
    for txt in leg.get_texts():        # bold the winner in the legend
        if txt.get_text() == winner:
            txt.set_fontweight("bold")

    fig.subplots_adjust(left=0.055, right=0.99, top=0.80, bottom=0.15, wspace=0.30)
    fig.suptitle(args.title, x=0.5, y=0.975, fontsize=19, fontweight="bold")
    fig.text(0.5, 0.895,
             f"n = {n} held-out validation subjects    ·    z-score, brain-masked "
             f"(SSIM over full volume)    ·    ◇ mean,  — median",
             ha="center", fontsize=12, color="#555555")

    out = Path(args.out) if args.out else \
        Path(args.csv).with_name("compare_boxplots_pro.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)

    # console summary (mean ± std per model per metric)
    print(f"\n{'model':<18}" + "".join(f"{n:>16}" for _, n, _, _ in METRICS))
    for m in order:
        g = df[df["model"] == m]
        cells = "".join(f"{g[k].mean():>8.3f}±{g[k].std():<7.3f}"
                        for k, _, _, _ in METRICS)
        print(f"{m:<18}{cells}")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
