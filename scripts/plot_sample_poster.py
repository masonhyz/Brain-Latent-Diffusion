"""
plot_sample_poster.py

Generates a poster-quality 3×2 grid showing one subject's pre- and
post-surgery CBF volumes: Axial / Coronal / Sagittal × Pre / Post.

Usage:
    python scripts/plot_sample_poster.py
    python scripts/plot_sample_poster.py --subject 2020_001 --out poster_sample.png
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ── colormap ──────────────────────────────────────────────────────────────────
# Dark-blue → cyan → white warm ramp  (works well for CBF on dark background)
_CBF_COLORS = [
    (0.00, "#000000"),
    (0.30, "#1a1a6e"),
    (0.55, "#2166ac"),
    (0.72, "#4dac26"),
    (0.85, "#f7d53d"),
    (1.00, "#ffffff"),
]
CBF_CMAP = LinearSegmentedColormap.from_list(
    "cbf",
    [(v, c) for v, c in _CBF_COLORS],
)


# ── helpers ───────────────────────────────────────────────────────────────────

def load_vol(path: Path) -> np.ndarray:
    """Load NIfTI, reorder to [D, H, W] = [Z, Y, X], return float32."""
    raw = nib.load(str(path)).get_fdata(dtype="float32")   # (X, Y, Z)
    return raw.transpose(2, 1, 0)                          # (Z, Y, X)


def percentile_norm(vol: np.ndarray, lo: float = 1, hi: float = 99.5) -> np.ndarray:
    """Percentile stretch using non-zero voxels only → [0, 1]."""
    mask = vol > 0
    if mask.any():
        p_lo, p_hi = np.percentile(vol[mask], [lo, hi])
    else:
        p_lo, p_hi = vol.min(), vol.max()
    return np.clip((vol - p_lo) / (p_hi - p_lo + 1e-8), 0, 1)


def get_slices(vol: np.ndarray, offsets=(0, 0, 0)):
    """
    vol: [D, H, W]
    Returns centre slices (± offset) as dict.
    Each slice is already in display orientation (origin='lower').
    """
    D, H, W = vol.shape
    dz, dy, dx = offsets
    axial    = vol[D // 2 + dz, :, :]          # [H, W]
    coronal  = vol[:, H // 2 + dy, :]          # [D, W]
    sagittal = vol[:, :, W // 2 + dx]          # [D, H]
    return {"Axial": axial, "Coronal": coronal, "Sagittal": sagittal}


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--subject",  default="2020_001")
    p.add_argument("--data_root", default="fmri")
    p.add_argument("--out",      default="poster_sample.png")
    p.add_argument("--dpi",      type=int, default=300)
    # slice offsets from centre (tune per subject if needed)
    p.add_argument("--dz", type=int, default=5,
                   help="Axial slice offset from centre (+up)")
    p.add_argument("--dy", type=int, default=0)
    p.add_argument("--dx", type=int, default=0)
    args = p.parse_args()

    root     = Path(args.data_root)
    pre_path = root / "pre_surgery"  / f"{args.subject}.nii.gz"
    post_path= root / "6_months_post_surgery" / f"{args.subject}.nii.gz"

    if not pre_path.exists():
        sys.exit(f"Pre file not found: {pre_path}")
    if not post_path.exists():
        sys.exit(f"Post file not found: {post_path}")

    pre_vol  = percentile_norm(load_vol(pre_path))
    post_vol = percentile_norm(load_vol(post_path))

    offsets = (args.dz, args.dy, args.dx)
    pre_slices  = get_slices(pre_vol,  offsets)
    post_slices = get_slices(post_vol, offsets)

    views = ["Axial", "Coronal", "Sagittal"]
    cols  = [("Pre-surgery",          pre_slices),
             ("Post-surgery (6 mo)",  post_slices)]

    # ── layout ────────────────────────────────────────────────────────────────
    BG = "#0d0d0d"
    plt.rcParams.update({
        "figure.facecolor": BG,
        "axes.facecolor":   BG,
        "text.color":       "white",
        "font.family":      "DejaVu Sans",
    })

    fig = plt.figure(figsize=(8, 11), facecolor=BG)

    # title area + grid
    gs_outer = gridspec.GridSpec(
        2, 1, figure=fig,
        height_ratios=[0.06, 1],
        hspace=0.04,
    )
    ax_title = fig.add_subplot(gs_outer[0])
    ax_title.axis("off")
    ax_title.text(
        0.5, 0.5,
        f"Subject {args.subject}  ·  Pre vs Post-surgery CBF",
        ha="center", va="center",
        fontsize=14, fontweight="bold", color="white",
        transform=ax_title.transAxes,
    )

    gs = gridspec.GridSpecFromSubplotSpec(
        3, 2,
        subplot_spec=gs_outer[1],
        hspace=0.06, wspace=0.04,
    )

    for col_idx, (col_name, slices) in enumerate(cols):
        for row_idx, view in enumerate(views):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            img = slices[view]

            ax.imshow(
                img, cmap=CBF_CMAP,
                origin="lower",
                vmin=0, vmax=1,
                interpolation="lanczos",
                aspect="equal",
            )
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

            # row label on left column
            if col_idx == 0:
                ax.set_ylabel(
                    view, fontsize=11, color="#cccccc",
                    labelpad=8, rotation=90, va="center",
                )

            # column header above first row
            if row_idx == 0:
                ax.set_title(
                    col_name,
                    fontsize=12, color="white", fontweight="bold",
                    pad=6,
                )

            # thin border accent — gold for pre, cyan for post
            accent = "#d4af37" if col_idx == 0 else "#4fc3f7"
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor(accent)
                spine.set_linewidth(0.8)

    # colorbar strip at bottom
    cbar_ax = fig.add_axes([0.15, 0.015, 0.70, 0.012])
    sm = plt.cm.ScalarMappable(cmap=CBF_CMAP)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(["Low", "Mid", "High"], color="#aaaaaa", fontsize=8)
    cbar.outline.set_edgecolor("#444444")
    cbar_ax.set_title("Relative CBF (a.u.)", color="#aaaaaa", fontsize=8, pad=4)

    out = Path(args.out)
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight", facecolor=BG)
    print(f"Saved → {out}  ({args.dpi} dpi)")


if __name__ == "__main__":
    main()
