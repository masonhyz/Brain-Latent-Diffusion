"""
fmri_3d_html_viz.py
────────────────────
Auto-refreshing HTML visualizer for 3D CycleGAN training.

Grid layout per snapshot:

              real_A (pre)  |  fake_B (gen. post)  |  real_B (real post)
    ──────────────────────────────────────────────────────────────────────
    sagittal  |     img      |        img           |       img
    axial     |     img      |        img           |       img
    coronal   |     img      |        img           |       img

Usage:
    from fmri_3d_html_viz import VolumeVisualizer

    viz = VolumeVisualizer(web_dir="checkpoints/my_run/web")
    viz.save_volumes(epoch=1, iters=500, volumes={
        "real_A": real_A_tensor,
        "fake_B": fake_B_tensor,
        "real_B": real_B_tensor,
    })
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── helpers ──────────────────────────────────────────────────────────────────

def _to_numpy_3d(t) -> np.ndarray:
    if isinstance(t, torch.Tensor):
        t = t.detach().cpu().float().numpy()
    t = np.asarray(t, dtype=np.float32)
    while t.ndim > 3:
        t = t[0]
    return t  # (D, H, W)


def _norm01(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    return (arr - lo) / (hi - lo) if hi > lo else np.zeros_like(arr)


# Rows: views. Cols: volumes.
_COLS = [
    ("real_A", "real_A\n(pre)"),
    ("fake_B", "fake_B\n(generated post)"),
    ("real_B", "real_B\n(real post)"),
]
_ROWS = [
    ("sagittal", lambda vol: vol[:, :, vol.shape[2] // 2]),
    ("axial",    lambda vol: vol[vol.shape[0] // 2, :, :]),
    ("coronal",  lambda vol: vol[:, vol.shape[1] // 2, :]),
]


class VolumeVisualizer:
    """
    Saves a 3×3 PNG per visualization step and maintains an auto-refreshing
    HTML index at web_dir/index.html.
    """

    def __init__(
        self,
        web_dir: str | Path,
        title: str = "3D CycleGAN Training",
        refresh_secs: int = 30,
    ):
        self.web_dir      = Path(web_dir)
        self.img_dir      = self.web_dir / "images"
        self.title        = title
        self.refresh_secs = refresh_secs
        self._entries: List[dict] = []

        self.web_dir.mkdir(parents=True, exist_ok=True)
        self.img_dir.mkdir(parents=True, exist_ok=True)

    # ── public API ────────────────────────────────────────────────────────

    def save_volumes(self, epoch: int, iters: int, volumes: Dict[str, object]) -> None:
        fname = f"ep{epoch:04d}_it{iters:07d}.png"
        label = f"Epoch {epoch} | Iter {iters}"

        fig = self._make_figure(volumes, label)
        fig.savefig(str(self.img_dir / fname), dpi=100, bbox_inches="tight",
                    facecolor="#1a1a1a")
        plt.close(fig)

        self._entries.insert(0, {"label": label, "img_rel": f"images/{fname}"})
        self._write_html()

    # ── figure ────────────────────────────────────────────────────────────

    def _make_figure(self, volumes: dict, suptitle: str) -> plt.Figure:
        """3 rows (views) × 3 cols (real_A, fake_B, real_B)."""
        n_rows, n_cols = len(_ROWS), len(_COLS)
        cell_size = 3.2
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(n_cols * cell_size, n_rows * cell_size),
            facecolor="#1a1a1a",
            squeeze=False,
        )
        fig.suptitle(suptitle, fontsize=11, color="#dddddd", y=1.02)

        for c, (key, col_label) in enumerate(_COLS):
            vol_data = volumes.get(key)
            vol = _to_numpy_3d(vol_data) if vol_data is not None else None

            for r, (view_name, slice_fn) in enumerate(_ROWS):
                ax = axes[r][c]
                ax.set_facecolor("#111111")
                ax.axis("off")

                if vol is not None:
                    sl = slice_fn(vol)
                    ax.imshow(_norm01(sl), cmap="gray", origin="upper",
                              interpolation="bilinear", aspect="equal")
                else:
                    ax.text(0.5, 0.5, "n/a", ha="center", va="center",
                            color="#666", transform=ax.transAxes)

                if r == 0:
                    ax.set_title(col_label, fontsize=9, color="#cccccc", pad=4)
                if c == 0:
                    ax.set_ylabel(view_name, fontsize=9, color="#aaaaaa",
                                  rotation=0, labelpad=40, va="center")

        fig.tight_layout()
        return fig

    # ── HTML ──────────────────────────────────────────────────────────────

    def _write_html(self) -> None:
        rows = "\n".join(
            f'<tr><td class="lbl">{e["label"]}</td>'
            f'<td><img src="{e["img_rel"]}" loading="lazy"></td></tr>'
            for e in self._entries
        )
        html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta http-equiv="refresh" content="{self.refresh_secs}">
  <title>{self.title}</title>
  <style>
    body  {{ background:#111; color:#ddd; font-family:monospace; margin:16px; }}
    h1    {{ font-size:1.05em; color:#adf; margin-bottom:2px; }}
    p     {{ font-size:0.78em; color:#777; margin:0 0 14px; }}
    table {{ border-collapse:collapse; width:100%; }}
    td    {{ padding:5px 8px; border-bottom:1px solid #222; vertical-align:middle; }}
    td.lbl {{ white-space:nowrap; color:#888; font-size:0.82em; width:150px; }}
    img   {{ max-width:100%; border:1px solid #333; }}
  </style>
</head>
<body>
  <h1>{self.title}</h1>
  <p>Refreshes every {self.refresh_secs}s &nbsp;|&nbsp; newest first
     &nbsp;|&nbsp; cols: real_A (pre) / fake_B (generated post) / real_B (real post)
     &nbsp;|&nbsp; rows: sagittal / axial / coronal</p>
  <table><tbody>{rows}</tbody></table>
</body>
</html>"""
        (self.web_dir / "index.html").write_text(html, encoding="utf-8")
