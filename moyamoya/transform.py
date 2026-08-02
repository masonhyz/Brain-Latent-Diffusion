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
        # Stats over nonzero voxels when masking (falling back to global if the
        # volume is all-zero); over the whole volume otherwise.
        mask = (t != 0) if self.nonzero_mask else torch.ones_like(t, dtype=torch.bool)
        vals = t[mask] if mask.any() else t
        std = vals.std(unbiased=False).clamp_min(self.eps)
        return (t - vals.mean()) / std

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


class PairedRandomIntensityShift:
    """
    Independent per-volume additive intensity shift (applied after z-score).

    Each volume gets its own offset drawn from Uniform(-max_shift, max_shift).
    Like :class:`PairedRandomIntensityScale`, shifts are drawn *independently*
    for x and y so the model sees varied absolute-intensity offsets.
    """

    def __init__(self, max_shift: float = 0.1):
        self.max_shift = float(max_shift)

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.max_shift <= 0:
            return x, y
        return (x + random.uniform(-self.max_shift, self.max_shift),
                y + random.uniform(-self.max_shift, self.max_shift))


class PairedRandomGamma:
    """
    Independent per-volume random gamma (contrast) adjustment.

    Data is z-scored (zero-centred, signed), so a plain power would be undefined
    for negatives. We apply a **sign-preserving** gamma, ``sign(v)*|v|**g``, which
    is a smooth monotonic contrast warp valid on signed data. ``g`` is drawn from
    Uniform(1-gamma, 1+gamma); g<1 boosts low-magnitude detail, g>1 suppresses it.
    """

    def __init__(self, gamma: float = 0.1):
        self.gamma = float(gamma)

    def _apply(self, v: torch.Tensor) -> torch.Tensor:
        g = random.uniform(1.0 - self.gamma, 1.0 + self.gamma)
        return v.sign() * v.abs().pow(g)

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.gamma <= 0:
            return x, y
        return self._apply(x), self._apply(y)


class PairedGaussianNoise:
    """
    Additive Gaussian noise, drawn independently for x and y.

    Acquisition noise is physically independent between the two scans, so the
    noise fields are sampled independently (not shared). ``std`` is relative to
    the z-scored data's unit standard deviation.
    """

    def __init__(self, std: float = 0.1):
        self.std = float(std)

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.std <= 0:
            return x, y
        return (x + torch.randn_like(x) * self.std,
                y + torch.randn_like(y) * self.std)


class PairedCompose:
    """Chain multiple paired transforms."""

    def __init__(self, transforms: List[Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]]):
        self.transforms = transforms

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        for t in self.transforms:
            x, y = t(x, y)
        return x, y
