import argparse
import csv
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

p = argparse.ArgumentParser()
p.add_argument("--run", default="runs/cfg2_big_denoiser")
p.add_argument("--drop_last", type=int, default=0,
               help="Drop the last N stage-2 epochs (e.g. corrupted final epochs)")
args = p.parse_args()
RUN = args.run


def load_csv(path):
    with open(path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    cols = {k: [] for k in rows[0]}
    for row in rows:
        for k, v in row.items():
            try:
                cols[k].append(float(v))
            except (ValueError, TypeError):
                cols[k].append(float("nan"))
    return {k: np.array(v) for k, v in cols.items()}


s2 = load_csv(f"{RUN}/metrics_stage2.csv")
if args.drop_last > 0:                                   # drop corrupted trailing epochs
    s2 = {k: v[:-args.drop_last] for k, v in s2.items()}

# stage 1 only exists for runs that trained their own AE (not when --ae_ckpt is reused)
has_s1 = os.path.exists(f"{RUN}/metrics_stage1.csv")
if has_s1:
    s1 = load_csv(f"{RUN}/metrics_stage1.csv")
    mask1 = ~np.isnan(s1["train_loss"])
    s1 = {k: v[mask1] for k, v in s1.items()}

fig = plt.figure(figsize=(16, 10))
fig.suptitle(f"{RUN.split('/')[-1]} — Training Progression", fontsize=14, fontweight="bold", y=0.98)

gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.42, wspace=0.35)


def smooth(arr, window):
    # edge-corrected moving average: divide by the actual #points overlapping the
    # window at each position (avoids the mode="same" zero-padding dive at the ends).
    if window <= 1:
        return arr
    num = np.convolve(arr, np.ones(window), mode="same")
    den = np.convolve(np.ones_like(arr), np.ones(window), mode="same")
    return num / den


# ── Stage 1: AE loss (only for runs that trained their own AE) ────────────────
if has_s1:
    ax0 = fig.add_subplot(gs[0, :2])
    ax0.plot(s1["epoch"], s1["train_loss"], label="Train", linewidth=1.8)
    ax0.plot(s1["epoch"], s1["val_loss"],   label="Val",   linewidth=1.8, linestyle="--")
    ax0.set_title("Stage 1 — Autoencoder Loss")
    ax0.set_xlabel("Epoch")
    ax0.set_ylabel("Loss")
    ax0.legend()
    ax0.grid(True, alpha=0.3)

# ── Stage 2: diffusion loss (spans the full top row when there's no Stage 1) ──
ax1 = fig.add_subplot(gs[0, 2:] if has_s1 else gs[0, :])
w = max(1, len(s2["epoch"]) // 50)
ax1.plot(s2["epoch"], s2["train_loss"], alpha=0.35, linewidth=0.8, color="tab:blue")
ax1.plot(s2["epoch"], s2["val_loss"],   alpha=0.35, linewidth=0.8, color="tab:orange")
ax1.plot(s2["epoch"], smooth(s2["train_loss"], w), label="Train (smooth)", linewidth=1.8, color="tab:blue")
ax1.plot(s2["epoch"], smooth(s2["val_loss"],   w), label="Val (smooth)",   linewidth=1.8, color="tab:orange", linestyle="--")
ax1.set_title("Stage 2 — Diffusion Loss")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.legend()
ax1.grid(True, alpha=0.3)

# ── Stage 2 image metrics ─────────────────────────────────────────────────────
metrics = [
    ("mae",  "MAE",      "tab:red",    False),
    ("mse",  "MSE",      "tab:brown",  False),
    ("psnr", "PSNR (dB)","tab:green",  True),
    ("ssim", "SSIM",     "tab:purple", True),
]
for i, (col, label, color, higher_better) in enumerate(metrics):
    ax = fig.add_subplot(gs[1, i])
    ax.plot(s2["epoch"], s2[col], color=color, linewidth=0.7, alpha=0.3)
    s = smooth(s2[col], w)
    ax.plot(s2["epoch"], s, color=color, linewidth=2.0)
    # mark best
    best_idx = np.nanargmax(s) if higher_better else np.nanargmin(s)
    ax.axvline(s2["epoch"][best_idx], color="gray", linestyle=":", linewidth=1.0)
    best_val = s[best_idx]
    ax.set_title(f"Stage 2 — {label}\nbest: {best_val:.4f} @ ep {int(s2['epoch'][best_idx])}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(label)
    ax.grid(True, alpha=0.3)

out = f"{RUN}/training_progression.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved → {out}")
plt.close()
