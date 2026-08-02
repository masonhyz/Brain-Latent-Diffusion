"""Shared dataset / dataloader wiring for the paired pre→post models.

Centralises the train/val split and loader construction that was previously
copy-pasted across every train and eval script, so training and evaluation
provably use the *same* held-out validation subjects.
"""

import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split

from .dataset import PrePostFMRI
from .transform import (
    ToChannelsFirstAndNormalize,
    PairedCompose,
    PairedRandomFlip,
    PairedRandomRotate3D,
    PairedRandomIntensityScale,
    PairedRandomIntensityShift,
    PairedRandomGamma,
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
# off just by passing its flag as 0. These conservative defaults reproduce the
# original augmentation (flip + intensity-scale only); the 7TCDM training script
# opts into the richer suite (rotate/shift/gamma/noise) via its own CLI defaults.
AUG_DEFAULTS = {
    "flip_p":          0.5,   # prob of left-right flip
    "rotate_deg":      0.0,   # max |rotation| per axis, degrees
    "intensity_scale": 0.1,   # multiplicative scale ∈ [1-s, 1+s]
    "intensity_shift": 0.0,   # additive shift ∈ [-s, s]
    "gamma":           0.0,   # gamma ∈ [1-g, 1+g] (sign-preserving)
    "noise_std":       0.0,   # additive Gaussian noise std (z-scored units)
}


def build_augmentation(flip_p=0.5, rotate_deg=0.0, intensity_scale=0.1,
                       intensity_shift=0.0, gamma=0.0, noise_std=0.0):
    """Assemble the paired train-time augmentation from per-transform strengths.

    Spatial transforms (flip, rotate) come first and are applied *identically* to
    x and y; intensity transforms then noise follow (noise last so it isn't
    rescaled). Any strength of 0 drops that transform, so an all-zero config
    yields an empty (identity) pipeline.
    """
    tfms = []
    if flip_p > 0:
        tfms.append(PairedRandomFlip(p=flip_p))
    if rotate_deg > 0:
        tfms.append(PairedRandomRotate3D(max_deg=rotate_deg))
    if intensity_scale > 0:
        tfms.append(PairedRandomIntensityScale(
            scale_range=(1.0 - intensity_scale, 1.0 + intensity_scale)))
    if intensity_shift > 0:
        tfms.append(PairedRandomIntensityShift(max_shift=intensity_shift))
    if gamma > 0:
        tfms.append(PairedRandomGamma(gamma=gamma))
    if noise_std > 0:
        tfms.append(PairedGaussianNoise(std=noise_std))
    return PairedCompose(tfms)


def build_augmentation_from_args(args):
    """Build the augmentation from ``--aug_*`` args, falling back to
    :data:`AUG_DEFAULTS` for any a script doesn't define (keeps the other models,
    which never added these flags, on the original flip + intensity-scale suite)."""
    return build_augmentation(**{
        k: getattr(args, f"aug_{k}", v) for k, v in AUG_DEFAULTS.items()
    })


def default_augmentation():
    """Original minimal suite (flip + intensity-scale); kept for callers that
    want it without an args object. See :func:`build_augmentation`."""
    return build_augmentation()


def build_loaders(args, augment: bool = False):
    """Build (train_dl, val_dl) from ``args`` (data_root, val_frac, seed,
    batch_size, num_workers). When ``augment`` is set, the train split is wrapped
    with the paired augmentation built from ``--aug_*`` args (see
    :func:`build_augmentation_from_args`)."""
    tfm = ToChannelsFirstAndNormalize(nonzero_mask=True)
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

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=True,
                          persistent_workers=persistent)
    val_dl   = DataLoader(val_subset, batch_size=1, shuffle=False,
                          num_workers=args.num_workers, pin_memory=True,
                          persistent_workers=persistent)
    return train_dl, val_dl
