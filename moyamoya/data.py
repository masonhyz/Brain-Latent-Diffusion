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
    PairedRandomIntensityScale,
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


def default_augmentation():
    return PairedCompose([
        PairedRandomFlip(p=0.5),
        PairedRandomIntensityScale(scale_range=(0.9, 1.1)),
    ])


def build_loaders(args, augment: bool = False):
    """Build (train_dl, val_dl) from ``args`` (data_root, val_frac, seed,
    batch_size, num_workers). ``augment`` adds flip + intensity-scale to train."""
    tfm = ToChannelsFirstAndNormalize(nonzero_mask=True)
    ds  = PrePostFMRI(root_dir=args.data_root, transform=tfm, strict=False)

    # ``getattr`` defaults keep the other models (which never define these args)
    # on the legacy holdout path; only the 7TCDM pipeline sets --fold.
    train_subset, val_subset = reconstruct_val_split(
        ds, args.val_frac, args.seed,
        n_folds=getattr(args, "n_folds", None),
        fold=getattr(args, "fold", None),
    )
    train_ds = AugmentedSubset(train_subset, default_augmentation()) if augment else train_subset

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=True)
    val_dl   = DataLoader(val_subset, batch_size=1, shuffle=False,
                          num_workers=args.num_workers, pin_memory=True)
    return train_dl, val_dl
