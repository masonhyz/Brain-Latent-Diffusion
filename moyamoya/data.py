"""Shared dataset / dataloader wiring for the paired pre→post models.

Centralises the train/val split and loader construction that was previously
copy-pasted across every train and eval script, so training and evaluation
provably use the *same* held-out validation subjects.
"""

import torch
from torch.utils.data import DataLoader, Dataset, random_split

from .dataset import PrePostFMRI
from .transform import (
    ToChannelsFirstAndNormalize,
    PairedCompose,
    PairedRandomFlip,
    PairedRandomIntensityScale,
)


def reconstruct_val_split(ds, val_frac: float, seed: int):
    """Deterministic (train_subset, val_subset) split.

    Identical formula to every train/eval script, so a checkpoint's validation
    subjects can be reproduced exactly at eval time from (val_frac, seed).
    """
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

    train_subset, val_subset = reconstruct_val_split(ds, args.val_frac, args.seed)
    train_ds = AugmentedSubset(train_subset, default_augmentation()) if augment else train_subset

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=True)
    val_dl   = DataLoader(val_subset, batch_size=1, shuffle=False,
                          num_workers=args.num_workers, pin_memory=True)
    return train_dl, val_dl
