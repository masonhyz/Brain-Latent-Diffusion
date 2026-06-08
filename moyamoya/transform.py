import random
import torch
from typing import Callable, List, Tuple


class ToChannelsFirstAndNormalize:
    """
    Converts NIfTI-shaped tensors to PyTorch 3D conv format and normalizes.

    - 3D: (X,Y,Z)  -> (C=1, D=Z, H=Y, W=X)
    - 4D: (X,Y,Z,T)-> (C=T, D=Z, H=Y, W=X)  (treat T as channels)

    Normalization: z-score per-sample (optionally within nonzero mask).
    """

    def __init__(self, eps: float = 1e-6, nonzero_mask: bool = True):
        self.eps = eps
        self.nonzero_mask = nonzero_mask

    def _reorder(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 3:
            # (X,Y,Z) -> (Z,Y,X) then add channel -> (1,Z,Y,X)
            t = t.permute(2, 1, 0).unsqueeze(0)
            return t
        elif t.ndim == 4:
            # (X,Y,Z,T) -> (T,Z,Y,X)
            t = t.permute(3, 2, 1, 0)
            return t
        else:
            raise ValueError(f"Expected 3D or 4D tensor, got shape {tuple(t.shape)}")

    def _zscore(self, t: torch.Tensor) -> torch.Tensor:
        if self.nonzero_mask:
            mask = t != 0
            if mask.any():
                vals = t[mask]
                mean = vals.mean()
                std = vals.std(unbiased=False).clamp_min(self.eps)
                t = (t - mean) / std
            else:
                # fallback: global
                mean = t.mean()
                std = t.std(unbiased=False).clamp_min(self.eps)
                t = (t - mean) / std
        else:
            mean = t.mean()
            std = t.std(unbiased=False).clamp_min(self.eps)
            t = (t - mean) / std
        return t

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self._reorder(x)
        y = self._reorder(y)
        x = self._zscore(x)
        y = self._zscore(y)
        return x, y


class PairedRandomFlip:
    """
    Random left-right (W-axis) flip applied identically to x and y.

    After ToChannelsFirstAndNormalize the layout is [C, D, H, W] where
    W corresponds to the original X (left-right) direction.  Brains are
    roughly bilaterally symmetric so this is anatomically valid.
    """

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if random.random() < self.p:
            x = torch.flip(x, dims=[-1])
            y = torch.flip(y, dims=[-1])
        return x, y


class PairedRandomIntensityScale:
    """
    Independent per-volume random multiplicative scaling (applied after z-score).

    Each volume gets its own scale factor drawn from Uniform(lo, hi), so the
    model sees different relative intensity contrasts and becomes more robust
    to inter-scanner / inter-session variability.
    """

    def __init__(self, scale_range: Tuple[float, float] = (0.9, 1.1)):
        self.lo, self.hi = scale_range

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        sx = random.uniform(self.lo, self.hi)
        sy = random.uniform(self.lo, self.hi)
        return x * sx, y * sy


class PairedCompose:
    """Chain multiple paired transforms."""

    def __init__(self, transforms: List[Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]]):
        self.transforms = transforms

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        for t in self.transforms:
            x, y = t(x, y)
        return x, y
