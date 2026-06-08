"""
Exploratory Data Analysis for the moyamoya pre/post-surgery fMRI dataset.

Run from project root:
    python scripts/eda.py [--data_root fmri] [--out_dir outputs/eda] [--n_sample 5]
"""
import argparse
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ── helpers ──────────────────────────────────────────────────────────────────

def load_nii(path):
    img = nib.load(str(path))
    return img.get_fdata(dtype="float32"), img.header.get_zooms()[:3], img.shape

def brain_mask(vol, threshold=0):
    return vol > threshold

def stats_for_vol(vol):
    mask = brain_mask(vol)
    brain = vol[mask]
    return {
        "n_nonzero": int(mask.sum()),
        "n_total": int(vol.size),
        "brain_frac": float(mask.sum() / vol.size),
        "min": float(vol.min()),
        "max": float(vol.max()),
        "mean_global": float(vol.mean()),
        "mean_brain": float(brain.mean()) if len(brain) > 0 else np.nan,
        "std_brain": float(brain.std()) if len(brain) > 0 else np.nan,
        "p5_brain": float(np.percentile(brain, 5)) if len(brain) > 0 else np.nan,
        "p95_brain": float(np.percentile(brain, 95)) if len(brain) > 0 else np.nan,
    }


def collect_stats(data_dir, label):
    records = []
    files = sorted(Path(data_dir).glob("*.nii.gz"))
    print(f"  Loading {label} ({len(files)} files)…", flush=True)
    for i, f in enumerate(files):
        vol, zooms, shape = load_nii(f)
        s = stats_for_vol(vol)
        s["subject"] = f.stem.replace(".nii", "")
        s["year"] = s["subject"].split("_")[0]
        s["label"] = label
        s["shape"] = shape
        s["voxel_mm"] = zooms
        records.append(s)
        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{len(files)}", flush=True)
    return records


# ── main ─────────────────────────────────────────────────────────────────────

def main(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_root = Path(args.data_root)

    pre_dir  = data_root / "pre_surgery"
    post_dir = data_root / "6_months_post_surgery"

    # ── 1. Collect per-subject statistics ────────────────────────────────────
    print("Collecting statistics…")
    pre_stats  = collect_stats(pre_dir,  "pre")
    post_stats = collect_stats(post_dir, "post")

    pre_names  = {r["subject"] for r in pre_stats}
    post_names = {r["subject"] for r in post_stats}
    paired     = pre_names & post_names
    only_pre   = pre_names - post_names
    only_post  = post_names - pre_names

    # Build lookup dicts
    pre_map  = {r["subject"]: r for r in pre_stats}
    post_map = {r["subject"]: r for r in post_stats}

    # ── 2. Print text summary ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print(f"  Pre-surgery volumes   : {len(pre_stats)}")
    print(f"  Post-surgery volumes  : {len(post_stats)}")
    print(f"  Fully paired subjects : {len(paired)}")
    if only_pre:
        print(f"  Pre only (no post)    : {len(only_pre)}  — {sorted(only_pre)[:5]}{'…' if len(only_pre)>5 else ''}")
    if only_post:
        print(f"  Post only (no pre)    : {len(only_post)} — {sorted(only_post)[:5]}{'…' if len(only_post)>5 else ''}")

    # Volume shape / voxel size (should be uniform)
    shapes = {r["shape"] for r in pre_stats + post_stats}
    zooms  = {r["voxel_mm"] for r in pre_stats + post_stats}
    print(f"\n  Volume shape(s)       : {shapes}")
    print(f"  Voxel size(s) (mm)    : {zooms}")

    # Year breakdown
    from collections import Counter
    pre_years  = Counter(r["year"] for r in pre_stats)
    post_years = Counter(r["year"] for r in post_stats)
    all_years  = sorted(set(pre_years) | set(post_years))
    print("\n  Subjects per cohort year:")
    print(f"    {'Year':<8} {'Pre':>6} {'Post':>6}")
    for y in all_years:
        print(f"    {y:<8} {pre_years.get(y,0):>6} {post_years.get(y,0):>6}")

    # CBF statistics
    def stat_table(records, name):
        vals = lambda k: np.array([r[k] for r in records])
        print(f"\n  {name} CBF statistics (brain-masked voxels):")
        print(f"    {'Metric':<20} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
        for k, label in [
            ("mean_brain",  "mean (ml/100g/min)"),
            ("std_brain",   "std"),
            ("p5_brain",    "5th pct"),
            ("p95_brain",   "95th pct"),
            ("brain_frac",  "brain fraction"),
            ("max",         "volume max"),
        ]:
            v = vals(k)
            print(f"    {label:<20} {v.mean():>10.3f} {v.std():>10.3f} {v.min():>10.3f} {v.max():>10.3f}")

    stat_table(pre_stats,  "Pre-surgery")
    stat_table(post_stats, "Post-surgery")

    # Paired delta
    paired_list = sorted(paired)
    delta_means = [post_map[s]["mean_brain"] - pre_map[s]["mean_brain"] for s in paired_list]
    print(f"\n  Post–Pre CBF change (paired, brain mean):")
    print(f"    Mean Δ = {np.mean(delta_means):.3f}   Std = {np.std(delta_means):.3f}")
    print(f"    Subjects with increased CBF: {sum(d > 0 for d in delta_means)} / {len(delta_means)}")
    print(f"    Subjects with decreased CBF: {sum(d < 0 for d in delta_means)} / {len(delta_means)}")
    print("=" * 60)

    # ── 3. Figure 1: CBF distribution comparison ─────────────────────────────
    pre_means  = [r["mean_brain"]  for r in pre_stats]
    post_means = [r["mean_brain"]  for r in post_stats]
    pre_stds   = [r["std_brain"]   for r in pre_stats]
    post_stds  = [r["std_brain"]   for r in post_stats]
    pre_maxs   = [r["max"]         for r in pre_stats]
    post_maxs  = [r["max"]         for r in post_stats]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    for ax, pre_v, post_v, xlabel in zip(
        axes,
        [pre_means, pre_stds, pre_maxs],
        [post_means, post_stds, post_maxs],
        ["Brain-mean CBF", "Brain-std CBF", "Volume max CBF"],
    ):
        bins = np.linspace(
            min(min(pre_v), min(post_v)),
            max(max(pre_v), max(post_v)),
            40,
        )
        ax.hist(pre_v,  bins=bins, alpha=0.6, label="Pre",  color="steelblue",  edgecolor="none")
        ax.hist(post_v, bins=bins, alpha=0.6, label="Post", color="tomato",     edgecolor="none")
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel("# subjects", fontsize=9)
        ax.legend(fontsize=9)
        ax.set_title(f"{xlabel} distribution", fontsize=10)

    fig.suptitle("Pre- vs Post-surgery CBF distributions (N=260 each)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "fig1_cbf_distributions.png", dpi=150)
    plt.close(fig)
    print(f"\nSaved: {out_dir / 'fig1_cbf_distributions.png'}")

    # ── 4. Figure 2: Paired CBF change ───────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # scatter: post vs pre mean
    ax = axes[0]
    ax.scatter(pre_means, post_means, s=10, alpha=0.6, color="mediumpurple")
    lim = [min(min(pre_means), min(post_means)) * 0.95, max(max(pre_means), max(post_means)) * 1.05]
    ax.plot(lim, lim, "k--", lw=0.8)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Pre-surgery brain-mean CBF"); ax.set_ylabel("Post-surgery brain-mean CBF")
    ax.set_title("Post vs Pre brain-mean CBF")

    # histogram of deltas
    ax = axes[1]
    ax.hist(delta_means, bins=30, color="seagreen", edgecolor="white", linewidth=0.5)
    ax.axvline(0, color="k", lw=1, ls="--")
    ax.set_xlabel("Post – Pre brain-mean CBF"); ax.set_ylabel("# subjects")
    ax.set_title(f"CBF change distribution  (mean={np.mean(delta_means):.2f})")

    # year-wise boxplot of deltas
    ax = axes[2]
    year_deltas = {}
    for s, d in zip(paired_list, delta_means):
        y = s.split("_")[0]
        year_deltas.setdefault(y, []).append(d)
    years_sorted = sorted(year_deltas)
    ax.boxplot([year_deltas[y] for y in years_sorted], labels=years_sorted, patch_artist=True,
               medianprops={"color": "black"})
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("Cohort year"); ax.set_ylabel("Post – Pre brain-mean CBF")
    ax.set_title("CBF change by cohort year")

    fig.suptitle("Paired Pre→Post CBF Change (N=260 paired subjects)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "fig2_paired_cbf_change.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {out_dir / 'fig2_paired_cbf_change.png'}")

    # ── 5. Figure 3: Brain mask consistency ──────────────────────────────────
    pre_fracs  = [r["brain_frac"]  for r in pre_stats]
    post_fracs = [r["brain_frac"]  for r in post_stats]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    bins = np.linspace(min(min(pre_fracs), min(post_fracs)),
                       max(max(pre_fracs), max(post_fracs)), 30)
    axes[0].hist(pre_fracs,  bins=bins, alpha=0.6, color="steelblue", label="Pre")
    axes[0].hist(post_fracs, bins=bins, alpha=0.6, color="tomato",    label="Post")
    axes[0].set_xlabel("Brain voxel fraction (>0)"); axes[0].set_ylabel("# subjects")
    axes[0].set_title("Brain mask fill fraction"); axes[0].legend()

    ax = axes[1]
    ax.scatter(pre_fracs, post_fracs, s=10, alpha=0.5, color="orange")
    lim2 = [min(min(pre_fracs), min(post_fracs)) * 0.98, max(max(pre_fracs), max(post_fracs)) * 1.02]
    ax.plot(lim2, lim2, "k--", lw=0.8)
    ax.set_xlim(lim2); ax.set_ylim(lim2)
    ax.set_xlabel("Pre brain fraction"); ax.set_ylabel("Post brain fraction")
    ax.set_title("Mask consistency: Post vs Pre")

    fig.suptitle("Brain Mask Fill Fraction", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "fig3_brain_mask.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {out_dir / 'fig3_brain_mask.png'}")

    # ── 6. Figure 4: Sample orthogonal slice views ────────────────────────────
    rng = np.random.default_rng(42)
    sample_subjects = sorted(rng.choice(sorted(paired), size=min(args.n_sample, len(paired)), replace=False))

    for subj in sample_subjects:
        pre_vol,  _, _ = load_nii(pre_dir  / f"{subj}.nii.gz")
        post_vol, _, _ = load_nii(post_dir / f"{subj}.nii.gz")

        fig, axes = plt.subplots(2, 3, figsize=(12, 7))
        cx, cy, cz = [s // 2 for s in pre_vol.shape]
        slices = [
            (pre_vol[cx, :, :], f"Pre sagittal x={cx}"),
            (pre_vol[:, cy, :], f"Pre coronal y={cy}"),
            (pre_vol[:, :, cz], f"Pre axial z={cz}"),
            (post_vol[cx, :, :], f"Post sagittal x={cx}"),
            (post_vol[:, cy, :], f"Post coronal y={cy}"),
            (post_vol[:, :, cz], f"Post axial z={cz}"),
        ]
        vmax = max(pre_vol.max(), post_vol.max())
        for ax, (sl, title) in zip(axes.flat, slices):
            im = ax.imshow(sl.T, origin="lower", cmap="hot", aspect="auto", vmin=0, vmax=vmax)
            ax.set_title(title, fontsize=9); ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.suptitle(f"Subject {subj} — CBF (ml/100g/min)", fontsize=11)
        fig.tight_layout()
        out_path = out_dir / f"fig4_sample_{subj}.png"
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        print(f"Saved: {out_path}")

    # ── 7. Figure 5: Year-wise CBF distributions ─────────────────────────────
    fig, axes = plt.subplots(1, len(all_years), figsize=(4 * len(all_years), 4), sharey=False)
    if len(all_years) == 1:
        axes = [axes]
    for ax, yr in zip(axes, all_years):
        pre_y  = [r["mean_brain"] for r in pre_stats  if r["year"] == yr]
        post_y = [r["mean_brain"] for r in post_stats if r["year"] == yr]
        bins = np.linspace(min(min(pre_y, default=[0]), min(post_y, default=[0])),
                           max(max(pre_y, default=[1]), max(post_y, default=[1])), 25)
        ax.hist(pre_y,  bins=bins, alpha=0.6, color="steelblue", label="Pre")
        ax.hist(post_y, bins=bins, alpha=0.6, color="tomato",    label="Post")
        ax.set_title(f"Year {yr}\n(n={len(pre_y)})", fontsize=10)
        ax.set_xlabel("Brain-mean CBF"); ax.legend(fontsize=8)
    fig.suptitle("Brain-mean CBF by cohort year", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "fig5_cbf_by_year.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {out_dir / 'fig5_cbf_by_year.png'}")

    print(f"\nAll figures saved to {out_dir}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="EDA for moyamoya fMRI dataset")
    ap.add_argument("--data_root", default="fmri",   help="Root folder containing pre/post subfolders")
    ap.add_argument("--out_dir",   default="outputs/eda", help="Output directory for figures")
    ap.add_argument("--n_sample",  type=int, default=5, help="Number of sample subjects to visualize")
    main(ap.parse_args())
