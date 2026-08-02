import math
import random
import torch
import torch.nn.functional as F
from typing import Callable, List, Tuple


class ToChannelsFirstAndNormalize:
    """
    Converts NIfTI-shaped tensors to PyTorch 3D conv format and normalizes.

    - 3D: (X,Y,Z)  -> (C=1, D=Z, H=Y, W=X)
    - 4D: (X,Y,Z,T)-> (C=T, D=Z, H=Y, W=X)  (treat T as channels)

    Normalization: z-score per-sample (optionally within nonzero mask).
    """

    def __init__(self, eps: float = 1e-6, nonzero_mask: bool = True,
                 zero_background: bool = False):
        """
        Args:
            nonzero_mask: compute the z-score statistics over nonzero voxels only.
            zero_background: restore exact zeros outside the acquired volume after
                normalising. **Off by default, and that default is a trap** —
                see :meth:`_zscore`.
        """
        self.eps = eps
        self.nonzero_mask = nonzero_mask
        self.zero_background = zero_background

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
        """Per-volume z-score.

        ``zero_background`` decides whether the background survives as zeros.

        With it OFF (the historical default), the shift ``t - mean`` is applied
        to the *whole* tensor, so background voxels — exactly 0 on input — come
        out at ``-mean/std ≈ -2.4``, i.e. **nonzero**. Every downstream
        ``(x != 0)`` "brain mask" in this project therefore selects 100 % of the
        volume rather than the ~29 % that is brain, and every metric documented
        as brain-masked is in fact whole-volume. The difference is not cosmetic:
        on the seed=42 val split the identity baseline scores MAE 0.2231 /
        SSIM 0.8382 whole-volume but MAE 0.4687 / SSIM 0.5176 on brain tissue —
        roughly 71 % of each metric was measuring how well a constant background
        is reproduced.

        The default stays OFF so the diffusion models and their checkpoints keep
        behaving exactly as they always did. Flow3D turns it ON.
        """
        mask = (t != 0) if self.nonzero_mask else torch.ones_like(t, dtype=torch.bool)
        vals = t[mask] if mask.any() else t
        std = vals.std(unbiased=False).clamp_min(self.eps)
        out = (t - vals.mean()) / std
        if self.zero_background and self.nonzero_mask and mask.any():
            out = out * mask
        return out

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


# NOTE ON INTENSITY AUGMENTATION
# ------------------------------
# Earlier revisions had per-volume intensity transforms (scale / shift / gamma)
# that drew their parameters *independently* for x (pre) and y (post). For a
# paired pre→post prediction task that is invalid: it perturbs the target by a
# random amount the model cannot infer from the input, i.e. it injects label
# noise and asks the model to hit a moving target. It is also largely redundant
# here because ToChannelsFirstAndNormalize already z-scores each volume, which
# normalises absolute scale/offset away. Those transforms were removed. The only
# intensity-side augmentation kept is input-only Gaussian noise (see
# PairedGaussianNoise), which leaves the target clean.


class PairedRandomRotate3D:
    """
    Small random 3-D rotation applied **identically** to x and y.

    This is a *spatial* transform, so pre and post must receive the exact same
    warp or their voxel-wise correspondence (which the pre→post model relies on)
    is destroyed — hence one rotation matrix is sampled per call and used for
    both volumes. Angles are drawn independently per axis from
    Uniform(-max_deg, max_deg); resampling is trilinear with zero padding.

    Operates on [C, D, H, W] tensors (the layout produced by
    :class:`ToChannelsFirstAndNormalize`).
    """

    def __init__(self, max_deg: float = 10.0, p: float = 1.0):
        self.max_deg = float(max_deg)
        self.p = p

    @staticmethod
    def _rot_matrix(rx: float, ry: float, rz: float) -> torch.Tensor:
        cx, sx = math.cos(rx), math.sin(rx)
        cy, sy = math.cos(ry), math.sin(ry)
        cz, sz = math.cos(rz), math.sin(rz)
        Rx = torch.tensor([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        Ry = torch.tensor([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        Rz = torch.tensor([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
        return Rz @ Ry @ Rx

    def _apply(self, vol: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        # affine_grid/grid_sample want a batch dim: [C,D,H,W] -> [1,C,D,H,W]
        v = vol.unsqueeze(0)
        grid = F.affine_grid(theta, v.shape, align_corners=False)
        out = F.grid_sample(v, grid, mode="bilinear",
                            padding_mode="zeros", align_corners=False)
        return out.squeeze(0)

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.max_deg <= 0 or random.random() >= self.p:
            return x, y
        m = math.radians(self.max_deg)
        R = self._rot_matrix(random.uniform(-m, m),
                             random.uniform(-m, m),
                             random.uniform(-m, m))
        # theta is the 3x4 affine (rotation | zero-translation), shape [1,3,4]
        theta = torch.cat([R, torch.zeros(3, 1)], dim=1).unsqueeze(0).to(x.dtype)
        return self._apply(x, theta), self._apply(y, theta)


class PairedGaussianNoise:
    """
    Additive Gaussian noise on the **input only** (default); the target is left
    clean.

    For a pre→post prediction task the target must stay pristine — corrupting it
    teaches the model to reproduce a noise realisation it cannot see. Perturbing
    only the input is the valid form: the model learns to map a noisy input to
    the clean target, i.e. robustness to acquisition noise. ``std`` is relative
    to the z-scored data's unit standard deviation.

    ``input_only=False`` also noises the target (independent field) — kept only
    for self-supervised use (e.g. an autoencoder reconstructing each volume to
    itself, where there is no cross-volume target); do not use it for stage-2
    paired prediction.
    """

    def __init__(self, std: float = 0.1, input_only: bool = True):
        self.std = float(std)
        self.input_only = input_only

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.std <= 0:
            return x, y
        x = x + torch.randn_like(x) * self.std
        if not self.input_only:
            y = y + torch.randn_like(y) * self.std
        return x, y


class PairedCompose:
    """Chain multiple paired transforms."""

    def __init__(self, transforms: List[Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]]):
        self.transforms = transforms

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        for t in self.transforms:
            x, y = t(x, y)
        return x, y
