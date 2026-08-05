"""Shared dataset / dataloader wiring for the paired pre→post models.

Centralises the train/val split and loader construction that was previously
copy-pasted across every train and eval script, so training and evaluation
provably use the *same* held-out validation subjects.
"""

import numpy as np
import torch
from torch.utils.data import (
    DataLoader, Dataset, Subset, WeightedRandomSampler, random_split,
)

from .dataset import PrePostFMRI
from .transform import (
    ToChannelsFirstAndNormalize,
    PairedCompose,
    PairedRandomFlip,
    PairedRandomRotate3D,
    PairedGaussianNoise,
)


def kfold_split(ds, n_folds: int, fold: int, seed: int):
    """Deterministic k-fold ``(train_subset, val_subset)`` for one fold.

    A single seeded permutation of all sample indices is partitioned into
    ``n_folds`` disjoint, near-equal contiguous chunks (uneven sizes handled like
    ``np.array_split``: the first ``len(ds) % n_folds`` chunks get one extra).
    Fold ``fold`` is the held-out validation chunk; the remaining chunks are the
    training set.

    The permutation depends only on ``(seed, n_folds)`` — *not* on ``fold`` — so
    across ``fold = 0..n_folds-1`` the validation chunks are mutually disjoint and
    cover every subject exactly once, and any ``(seed, n_folds, fold)`` reproduces
    the identical split at eval time.
    """
    if n_folds < 2:
        raise ValueError(f"n_folds must be >= 2, got {n_folds}")
    if not (0 <= fold < n_folds):
        raise ValueError(f"fold must be in [0, {n_folds}), got {fold}")
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(ds), generator=g)
    chunks = torch.tensor_split(perm, n_folds)
    val_idx = chunks[fold].tolist()
    train_idx = [int(i) for j, c in enumerate(chunks) if j != fold for i in c.tolist()]
    return Subset(ds, train_idx), Subset(ds, [int(i) for i in val_idx])


def reconstruct_val_split(ds, val_frac: float, seed: int, n_folds=None, fold=None):
    """Deterministic ``(train_subset, val_subset)`` split.

    Two modes, both pure functions of their arguments so a checkpoint's
    validation subjects reproduce exactly at eval time:

      * **k-fold** — when ``fold`` is not None: hold out fold ``fold`` of
        ``n_folds`` (see :func:`kfold_split`). ``val_frac`` is ignored.
      * **holdout** — otherwise: hold out a random ``val_frac`` fraction. This is
        the legacy single-split behaviour, kept for backward compatibility so old
        checkpoints (and the other models that share this helper) reconstruct the
        same subjects they always did.
    """
    if fold is not None:
        if n_folds is None:
            raise ValueError("n_folds is required when fold is set")
        return kfold_split(ds, n_folds, fold, seed)
    n_val = max(1, int(len(ds) * val_frac))
    n_train = len(ds) - n_val
    g = torch.Generator().manual_seed(seed)
    return random_split(ds, [n_train, n_val], generator=g)


class AugmentedSubset(Dataset):
    """Wrap a Subset and apply a paired augmentation on top of its transform."""
    def __init__(self, subset, aug):
        self.subset = subset
        self.aug    = aug

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        x, y = self.subset[idx]
        return self.aug(x, y)


# Default per-transform strengths. Each is "0 = disabled"; a strength of 0 makes
# the corresponding transform a no-op, so a script can turn any single transform
# off just by passing its flag as 0.
#
# The suite is deliberately restricted to transforms that are valid for a paired
# pre→post prediction task: spatial transforms (flip, rotate) applied IDENTICALLY
# to both volumes so voxel correspondence is preserved, plus input-only Gaussian
# noise (target left clean). Independent intensity scale/shift/gamma were removed
# — they perturb the target unpredictably (label noise) and are redundant given
# per-volume z-scoring. See moyamoya/transform.py for the full rationale.
#
# These fallback defaults (flip only) apply to scripts that share build_loaders
# but define no --aug_* flags (e.g. train_cdm3d.py); the 7TCDM training script
# opts into rotate + noise via its own CLI defaults.
AUG_DEFAULTS = {
    "flip_p":     0.5,   # prob of left-right flip (shared spatial)
    "rotate_deg": 0.0,   # max |rotation| per axis, degrees (shared spatial)
    "noise_std":  0.0,   # input-only Gaussian noise std (z-scored units)
}


def build_augmentation(flip_p=0.5, rotate_deg=0.0, noise_std=0.0):
    """Assemble the paired train-time augmentation from per-transform strengths.

    Spatial transforms (flip, rotate) come first and are applied *identically* to
    x and y so their voxel correspondence is preserved; input-only Gaussian noise
    comes last (target stays clean). Any strength of 0 drops that transform, so an
    all-zero config yields an empty (identity) pipeline.
    """
    tfms = []
    if flip_p > 0:
        tfms.append(PairedRandomFlip(p=flip_p))
    if rotate_deg > 0:
        tfms.append(PairedRandomRotate3D(max_deg=rotate_deg))
    if noise_std > 0:
        tfms.append(PairedGaussianNoise(std=noise_std, input_only=True))
    return PairedCompose(tfms)


def build_augmentation_from_args(args):
    """Build the augmentation from ``--aug_*`` args, falling back to
    :data:`AUG_DEFAULTS` for any a script doesn't define (keeps the other models,
    which never added these flags, on the flip-only fallback)."""
    return build_augmentation(**{
        k: getattr(args, f"aug_{k}", v) for k, v in AUG_DEFAULTS.items()
    })


def default_augmentation():
    """Minimal valid suite (flip only); kept for callers that want it without an
    args object. See :func:`build_augmentation`."""
    return build_augmentation()


def change_magnitudes(subset, coherent: bool = False, sigma: float = 2.0) -> np.ndarray:
    """Per-subject global change ``mean|x_post − x_pre|`` over the brain, in order.

    Loads every volume in ``subset`` once (with the base normalising transform, no
    augmentation), so it runs at loader-construction time. Restricting the mean to
    the brain (see :func:`~moyamoya.metrics.foreground_mask`) is what makes it a
    real change measure: under whole-volume normalisation the background is a
    near-constant plateau, so averaging over it would just add the same offset to
    every subject and wash out their differences.

    ``coherent=True`` measures the *coherent* change instead: it Gaussian-blurs |Δ|
    (σ voxels) before averaging, so the score reflects big *real* edits rather than
    the high-frequency registration speckle that inflates raw |Δ|. This is the
    sounder thing to oversample on — raw-|Δ| oversampling partly oversamples the
    noisiest subjects (see the change-signal diagnosis).
    """
    from .metrics import foreground_mask
    from .models.flow3d import gaussian_blur3d
    mags = []
    for x, y in subset:
        fg = foreground_mask(x, y)
        d = (y - x).abs()
        if coherent:
            d = gaussian_blur3d(d.unsqueeze(0), sigma)[0]
        d = d.numpy().squeeze()
        mags.append(float(d[fg].mean()) if fg.any() else 0.0)
    return np.asarray(mags, dtype=np.float64)


def change_sampler(subset, beta: float, coherent: bool = False):
    """A ``WeightedRandomSampler`` that oversamples the big-change subjects.

        weight_i ∝ 1 + β · (g_i / median g)

    with g_i the subject's global change (:func:`change_magnitudes`). β=0 is
    uniform; larger β spends more of each epoch on the subjects where pre and post
    genuinely differ — the pairs that carry the "make a real edit" signal, which
    is otherwise drowned out by the near-identity majority. Kept deliberately mild
    by default: with only ~200 subjects, aggressively oversampling the handful of
    large-change cases overfits them. The finer-grained voxel weighting
    (:func:`~moyamoya.models.flow3d.change_weight_map`) is the safer primary lever;
    this is the complementary coarse one.

    ``coherent=True`` weights by the coherent (smoothed) change, not raw |Δ| — the
    recommended measure, so the oversampling targets real edits not misregistration.

    Returns ``(sampler, g)`` — ``g`` is the change vector, for logging.
    """
    g = change_magnitudes(subset, coherent=coherent)
    med = float(np.median(g))
    if med <= 0:
        med = float(g.mean()) if g.mean() > 0 else 1.0
    w = np.clip(1.0 + beta * (g / med), 1e-6, None)
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(w, dtype=torch.double),
        num_samples=len(g), replacement=True,
    )
    return sampler, g


def build_loaders(args, augment: bool = False):
    """Build (train_dl, val_dl) from ``args`` (data_root, val_frac, seed,
    batch_size, num_workers). When ``augment`` is set, the train split is wrapped
    with the paired augmentation built from ``--aug_*`` args (see
    :func:`build_augmentation_from_args`)."""
    # zero_background defaults off so the diffusion models keep their historical
    # preprocessing; only scripts that define the flag (Flow3D) opt in. See
    # ToChannelsFirstAndNormalize._zscore for why it matters — with it off, every
    # "(x != 0)" brain mask downstream actually selects the whole volume.
    tfm = ToChannelsFirstAndNormalize(
        nonzero_mask=True,
        zero_background=getattr(args, "zero_background", False),
    )
    ds  = PrePostFMRI(root_dir=args.data_root, transform=tfm, strict=False)

    # ``getattr`` defaults keep the other models (which never define these args)
    # on the legacy holdout path; only the 7TCDM pipeline sets --fold.
    train_subset, val_subset = reconstruct_val_split(
        ds, args.val_frac, args.seed,
        n_folds=getattr(args, "n_folds", None),
        fold=getattr(args, "fold", None),
    )
    train_ds = (AugmentedSubset(train_subset, build_augmentation_from_args(args))
                if augment else train_subset)

    # persistent_workers avoids re-spawning workers every epoch (a real cost with
    # small datasets + many epochs); only valid when num_workers > 0. Defaults to
    # True but stays off for the other models, which don't define the arg.
    persistent = getattr(args, "persistent_workers", False) and args.num_workers > 0

    # Optional change-emphasis sampler: oversample the subjects whose post-op scan
    # genuinely differs from the pre-op one, so the "make a real edit" signal is
    # not drowned out by the near-identity majority. Off unless --change_sampler_beta
    # > 0. Replaces shuffle (a sampler and shuffle=True are mutually exclusive).
    beta = float(getattr(args, "change_sampler_beta", 0.0) or 0.0)
    coherent = bool(getattr(args, "change_sampler_coherent", False))
    sampler = None
    if beta > 0:
        print(f"[change-sampler] computing per-subject "
              f"{'coherent ' if coherent else ''}change over {len(train_subset)} "
              f"training volumes (one-time)...")
        sampler, g = change_sampler(train_subset, beta, coherent=coherent)
        print(f"[change-sampler] beta={beta} coherent={coherent}  global change: "
              f"min={g.min():.3f} median={np.median(g):.3f} max={g.max():.3f} — "
              f"oversampling large-change subjects (weight ∝ 1+beta·g/median).")

    train_dl = DataLoader(train_ds, batch_size=args.batch_size,
                          shuffle=(sampler is None), sampler=sampler,
                          num_workers=args.num_workers, pin_memory=True,
                          persistent_workers=persistent)
    val_dl   = DataLoader(val_subset, batch_size=1, shuffle=False,
                          num_workers=args.num_workers, pin_memory=True,
                          persistent_workers=persistent)
    return train_dl, val_dl
