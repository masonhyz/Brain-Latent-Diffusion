import random
import numpy as np
import torch


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
