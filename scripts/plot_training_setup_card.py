#!/usr/bin/env python
"""
Render the cldm-strong training setup as a poster-ready "spec panel" PNG:
sectioned grid of value-over-label cards. Pure presentation (values hard-coded
from runs/cfg5_strong_cfg/hparams_stage{1,2}.json).

Usage:
  python scripts/plot_training_setup_card.py --out outputs/compare_5models/training_setup_card.png
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ACCENT   = "#D1495B"   # ties to the highlighted cldm-strong color
CARD_BG  = "#F5F7F9"
CARD_EDGE= "#E2E6EA"
VAL_CLR  = "#1F2A37"
LBL_CLR  = "#6B7280"
SEC_CLR  = "#111827"
RULE_CLR = "#E5E7EB"

# (section title, [(value, label), ...]) — counts are multiples of 3
SECTIONS = [
    ("Data", [
        ("221 / 39", "Train / Val"), ("64³", "Resolution"), ("seed 42", "Split RNG")]),
    ("Autoencoder · Stage 1", [
        ("64", "Base channels"), ("1·2·4", "Channel mult."), ("4×16³", "Latent shape"),
        ("1e-6", "KL weight"), ("3", "Res-blocks / lvl"), ("200", "Epochs")]),
    ("Denoiser · Stage 2", [
        ("64", "Base channels"), ("1·2·4·8", "Channel mult."), ("869", "Epochs")]),
    ("Diffusion & guidance", [
        ("1000", "Diffusion steps T"), ("cosine", "Noise schedule"), ("50", "DDIM steps"),
        ("η = 0", "DDIM sampling"), ("0.25", "CFG drop prob"), ("5.0", "Guidance scale")]),
    ("Optimization", [
        ("AdamW", "Optimizer"), ("3e-4", "Learning rate"), ("1e-4", "Weight decay"),
        ("2", "Batch size"), ("bf16", "Precision"), ("1.0", "Grad clip")]),
]

# layout (inches)
NCOL = 3
LEFT, RIGHT = 0.35, 0.35
CONTENT_W = 8.4
COL_GAP = 0.18
CARD_W = (CONTENT_W - (NCOL - 1) * COL_GAP) / NCOL
CARD_H, CARD_VGAP = 0.86, 0.16
HEAD_H, HEAD_GAP = 0.40, 0.12
SEC_GAP = 0.24
TITLE_H, TITLE_GAP = 0.52, 0.30
TOP_PAD, BOT_PAD = 0.18, 0.26
TOTAL_W = LEFT + CONTENT_W + RIGHT


def card_rows(cards):
    return [cards[i:i + NCOL] for i in range(0, len(cards), NCOL)]


def total_height():
    h = TOP_PAD + TITLE_H + TITLE_GAP + BOT_PAD
    for _, cards in SECTIONS:
        nrows = len(card_rows(cards))
        h += SEC_GAP + HEAD_H + HEAD_GAP + nrows * CARD_H + (nrows - 1) * CARD_VGAP
    return h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/compare_5models/training_setup_card.png")
    ap.add_argument("--title", default="Training Setup — cldm-strong")
    ap.add_argument("--accent", default=ACCENT, help="accent color for the dots/bars")
    args = ap.parse_args()
    accent = args.accent

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "savefig.dpi": 300,
    })

    H = total_height()
    fig = plt.figure(figsize=(TOTAL_W, H))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, TOTAL_W); ax.set_ylim(0, H)
    ax.axis("off")

    def ytop(cur):  # convert distance-from-top to data-y
        return H - cur

    cur = TOP_PAD
    # title with accent bar
    ax.add_patch(FancyBboxPatch((LEFT, ytop(cur + TITLE_H) + 0.06), 0.10, TITLE_H - 0.12,
                                boxstyle="round,pad=0,rounding_size=0.03",
                                facecolor=accent, edgecolor="none"))
    ax.text(LEFT + 0.24, ytop(cur + TITLE_H / 2), args.title,
            ha="left", va="center", fontsize=19, fontweight="bold", color=SEC_CLR)
    cur += TITLE_H + TITLE_GAP

    for title, cards in SECTIONS:
        cur += SEC_GAP
        # section header: small accent square + bold text + thin rule
        hy = ytop(cur + HEAD_H / 2)
        ax.add_patch(FancyBboxPatch((LEFT, hy - 0.075), 0.15, 0.15,
                                    boxstyle="round,pad=0,rounding_size=0.04",
                                    facecolor=accent, edgecolor="none"))
        ax.text(LEFT + 0.27, hy, title, ha="left", va="center",
                fontsize=13, fontweight="bold", color=SEC_CLR)
        ax.plot([LEFT, LEFT + CONTENT_W], [ytop(cur + HEAD_H)] * 2,
                color=RULE_CLR, lw=1.1)
        cur += HEAD_H + HEAD_GAP

        for row in card_rows(cards):
            for c, (val, lbl) in enumerate(row):
                x = LEFT + c * (CARD_W + COL_GAP)
                y0 = ytop(cur + CARD_H)
                ax.add_patch(FancyBboxPatch((x, y0), CARD_W, CARD_H,
                             boxstyle="round,pad=0,rounding_size=0.07",
                             facecolor=CARD_BG, edgecolor=CARD_EDGE, lw=1.1))
                cx = x + CARD_W / 2
                ax.text(cx, y0 + CARD_H * 0.60, val, ha="center", va="center",
                        fontsize=17, fontweight="bold", color=VAL_CLR)
                ax.text(cx, y0 + CARD_H * 0.24, lbl, ha="center", va="center",
                        fontsize=10, color=LBL_CLR)
            cur += CARD_H + CARD_VGAP
        cur -= CARD_VGAP  # last row has no trailing gap

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out}  ({TOTAL_W:.1f} x {H:.1f} in)")


if __name__ == "__main__":
    main()
