"""Run-logging plumbing shared by the training scripts.

Console teeing, hparam capture, per-epoch CSV, W&B, and the orthogonal-slice
visualisation grid — the boilerplate that was previously copy-pasted between
``train_cdm3d.py`` and ``train_ldm_7tcdm3d.py``. New scripts import it from
here; the existing two are left alone so their in-flight runs keep working.

Nothing in here is allowed to break training: W&B in particular is wrapped so
that a missing package, a missing login, or a network failure prints one line
and training continues.
"""

import csv
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── console teeing ────────────────────────────────────────────────────────────

class _Tee:
    """Duplicate stream writes to the real console and a logfile."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self._streams:
            s.flush()


def install_run_logger(out_dir: Path, name: str = "train") -> Path:
    """Tee stdout/stderr into ``out_dir/<name>.log`` (append mode).

    Appended rather than truncated so re-running into the same directory keeps
    earlier console history, each run delimited by a banner.
    """
    log_path = Path(out_dir) / f"{name}.log"
    logfile = open(log_path, "a", buffering=1)   # line-buffered
    logfile.write(f"\n===== run started {datetime.now().isoformat()} =====\n")
    sys.stdout = _Tee(sys.__stdout__, logfile)
    sys.stderr = _Tee(sys.__stderr__, logfile)
    return log_path


def git_commit():
    """Short git SHA of the working tree, or None if unavailable."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent.parent,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return None


def save_hparams(args, model_name: str, out_dir: Path, extra: dict = None) -> Path:
    """Write args + split + git SHA + timestamp to ``out_dir/hparams.json``."""
    hp = {
        "model": model_name,
        "timestamp": datetime.now().isoformat(),
        "git_commit": git_commit(),
        "args": vars(args),
    }
    if extra:
        hp.update(extra)
    path = Path(out_dir) / "hparams.json"
    with open(path, "w") as f:
        json.dump(hp, f, indent=2, default=str)
    print(f"  Hyperparameters saved → {path}")
    return path


# ── per-epoch metrics CSV ─────────────────────────────────────────────────────

class MetricsCSV:
    """Append-only per-epoch scalar log — the durable history of a run.

    The plots are all rebuilt from this file, so it is the artifact that must
    survive; a crashed run still leaves a readable history behind.
    """

    def __init__(self, path: Path, columns):
        self.path = Path(path)
        self.columns = list(columns)
        with open(self.path, "w", newline="") as f:
            csv.writer(f).writerow(self.columns)

    def append(self, row: dict) -> None:
        with open(self.path, "a", newline="") as f:
            csv.writer(f).writerow([
                _fmt(row.get(c, float("nan"))) for c in self.columns
            ])

    def load(self) -> dict:
        rows = list(csv.DictReader(open(self.path)))
        if not rows:
            return {}
        out = {k: [] for k in rows[0]}
        for r in rows:
            for k, v in r.items():
                try:
                    out[k].append(float(v))
                except (TypeError, ValueError):
                    out[k].append(float("nan"))
        return {k: np.array(v) for k, v in out.items()}


def _fmt(v):
    return f"{v:.6f}" if isinstance(v, float) else v


# ── Weights & Biases (never fatal) ────────────────────────────────────────────

def _has_wandb_creds() -> bool:
    if os.environ.get("WANDB_API_KEY"):
        return True
    netrc = Path.home() / ".netrc"
    try:
        return netrc.is_file() and "api.wandb.ai" in netrc.read_text()
    except Exception:
        return False


def init_wandb(args, out_dir: Path, project: str, config_extra: dict = None):
    """Start a W&B run, or return None if disabled/unavailable."""
    if not getattr(args, "wandb", False):
        return None
    try:
        import wandb
    except Exception:
        print("[wandb] not installed — skipping (`pip install wandb`). Training continues.")
        return None

    mode = getattr(args, "wandb_mode", "online")
    if mode == "online" and not _has_wandb_creds():
        print("[wandb] online requested but not logged in → logging OFFLINE instead. "
              "Run `wandb login` once, then `wandb sync <run dir>`. Training continues.")
        mode = "offline"

    out_dir = Path(out_dir)
    group = getattr(args, "wandb_group", None)
    if not group:
        name = out_dir.parent.name if re.fullmatch(r"fold\d+", out_dir.name) else out_dir.name
        group = re.sub(r"_fold\d+$|_cv$", "", name)

    config = dict(vars(args))
    if config_extra:
        config.update(config_extra)
    fold = getattr(args, "fold", None)
    tags = ["kfold" if fold is not None else "holdout"]
    if fold is not None:
        tags.append(f"fold{fold}")

    try:
        run = wandb.init(
            project=project, entity=getattr(args, "wandb_entity", None),
            name=out_dir.name, group=group, tags=tags, config=config,
            dir=str(out_dir), mode=mode, reinit=True,
        )
        print(f"[wandb] project '{project}' | group '{group}' | run '{run.name}' | mode {mode}")
        return run
    except Exception as e:
        print(f"[wandb] init failed ({e}); continuing without W&B.")
        return None


def wandb_log(run, data: dict, step: int) -> None:
    if run is not None:
        try:
            run.log(data, step=step)
        except Exception:
            pass


def wandb_log_image(run, key: str, path, step: int) -> None:
    if run is None:
        return
    try:
        import wandb
        run.log({key: wandb.Image(str(path))}, step=step)
    except Exception:
        pass


def wandb_finish(run) -> None:
    if run is not None:
        try:
            run.finish()
        except Exception:
            pass


# ── visualisation ─────────────────────────────────────────────────────────────

def _slices(vol: np.ndarray) -> dict:
    D, H, W = vol.shape
    return {"Axial": vol[D // 2], "Coronal": vol[:, H // 2], "Sagittal": vol[:, :, W // 2]}


def save_grid(x_np, y_np, pred_np, title: str, save_path: Path) -> None:
    """4×4 grid: three orthogonal views × (pre | prediction | post GT | error).

    The error column is |prediction − GT| on a fixed 0…1 scale, which is what
    makes it possible to see at a glance whether a model is actually improving
    on the input or just reproducing it.
    """
    cols = ["Pre-surgery", "Prediction", "Post-surgery GT", "|error|"]
    views = ["Axial", "Coronal", "Sagittal"]
    xs, ys, ps = _slices(x_np), _slices(y_np), _slices(pred_np)

    fig, axes = plt.subplots(3, 4, figsize=(13, 10), constrained_layout=True)
    for r, view in enumerate(views):
        err = np.abs(ps[view] - ys[view])
        for c, img in enumerate([xs[view], ps[view], ys[view], err]):
            ax = axes[r, c]
            ax.imshow(img, cmap="inferno" if c == 3 else "gray",
                      origin="lower", vmin=0, vmax=1)
            ax.axis("off")
            if r == 0:
                ax.set_title(cols[c], fontsize=11)
        axes[r, 0].set_ylabel(view, fontsize=10)

    fig.suptitle(title, fontsize=12)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def smooth(arr: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return arr
    return np.convolve(arr, np.ones(window) / window, mode="same")


def plot_progression(csv_log: "MetricsCSV", out_dir: Path, title: str,
                     baseline: dict = None) -> Path:
    """Loss + MAE/MSE/PSNR/SSIM curves, rebuilt from the metrics CSV.

    When ``baseline`` is given (the identity predictor's scores) each metric
    panel draws it as a dashed reference line — so a run that never crosses it
    is visibly a run that lost to copying the input.
    """
    d = csv_log.load()
    if not d or "epoch" not in d:
        return None
    out_dir = Path(out_dir)

    fig, axes = plt.subplots(1, 5, figsize=(24, 4.2), constrained_layout=True)
    fig.suptitle(title, fontsize=13, fontweight="bold")

    w = max(1, len(d["epoch"]) // 50)
    ax = axes[0]
    ax.plot(d["epoch"], d["train_loss"], alpha=0.3, lw=0.8, color="tab:blue")
    ax.plot(d["epoch"], d["val_loss"], alpha=0.3, lw=0.8, color="tab:orange")
    ax.plot(d["epoch"], smooth(d["train_loss"], w), lw=1.8, color="tab:blue", label="Train")
    ax.plot(d["epoch"], smooth(d["val_loss"], w), lw=1.8, color="tab:orange",
            ls="--", label="Val")
    ax.set_title("Flow matching loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(alpha=0.3)

    for i, (col, label, color, hib) in enumerate([
        ("mae", "MAE", "tab:red", False),
        ("mse", "MSE", "tab:brown", False),
        ("psnr", "PSNR (dB)", "tab:green", True),
        ("ssim", "SSIM", "tab:purple", True),
    ]):
        ax = axes[i + 1]
        if col not in d:
            continue
        valid = ~np.isnan(d[col])
        ep, val = d["epoch"][valid], d[col][valid]
        if len(val) == 0:
            ax.set_title(f"{label}\n(no metric epochs yet)")
            ax.grid(alpha=0.3)
            continue
        wv = max(1, len(ep) // 50)
        ax.plot(ep, val, color=color, lw=0.7, alpha=0.3)
        s = smooth(val, wv)
        ax.plot(ep, s, color=color, lw=2.0)
        best = int(np.nanargmax(s) if hib else np.nanargmin(s))
        ax.axvline(ep[best], color="gray", ls=":", lw=1.0)
        sub = f"best {s[best]:.4f} @ ep {int(ep[best])}"
        if baseline and col in baseline:
            ax.axhline(baseline[col], color="k", ls="--", lw=1.2, alpha=0.7)
            beat = (s[best] > baseline[col]) if hib else (s[best] < baseline[col])
            sub += f"  |  identity {baseline[col]:.4f} {'✓ beaten' if beat else '✗ NOT beaten'}"
        ax.set_title(f"{label}\n{sub}", fontsize=10)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)

    path = out_dir / "training_progression.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → training progression saved: {path}")
    return path
