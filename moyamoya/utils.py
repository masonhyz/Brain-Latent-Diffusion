import os
import random
import numpy as np
import torch


def get_device() -> torch.device:
    """Return the active CUDA device, controlled by the GPU env var (default: 1)."""
    if torch.cuda.is_available():
        gpu = int(os.environ.get("GPU", 1))
        return torch.device(f"cuda:{gpu}")
    return torch.device("cpu")


def resolve_seed(seed: int | None = None) -> int:
    """Return a concrete RNG seed, drawing a random one when ``seed`` is None.

    Passing None picks a fresh seed each run from the OS entropy source instead
    of pinning every run to the same constant. The caller MUST log the returned
    value for reproducibility — the training scripts save it into hparams + the
    checkpoint, so a run can be recreated exactly by rerunning with
    ``--seed <logged value>`` (same train/val split and initialization).

    The seed stays in ``[1, 2**32 - 1]``: below numpy's 2**32 ceiling and never
    0, so it survives ``args.seed or <default>`` fallbacks elsewhere.
    """
    if seed is None:
        seed = random.SystemRandom().randint(1, 2**32 - 1)
    return int(seed)


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True  # faster if shapes are consistent
    torch.backends.cudnn.deterministic = False


def l1_loss(pred, target):
    return (pred - target).abs().mean()


def make_union_mask(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    x,y: [B,C,D,H,W]
    Returns boolean mask of voxels to include in loss.
    Union of nonzero voxels in either input or target.
    """
    return (x != 0) | (y != 0)


def masked_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    pred,target: [B,C,D,H,W]
    mask: boolean [B,C,D,H,W] or broadcastable
    """
    diff = (pred - target).abs()
    diff = diff * mask.to(diff.dtype)
    denom = mask.to(diff.dtype).sum().clamp_min(eps)
    return diff.sum() / denom
