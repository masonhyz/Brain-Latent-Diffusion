# from __future__ import annotations

# from typing import Dict, Optional, Callable, Tuple, Any
# import torch
# from torch.utils.data import Dataset


# class CycleGANDictWrapper(Dataset):
#     """
#     Wraps a paired dataset that returns (x, y) or (x, y, meta) into the
#     dict format expected by junyanz/pytorch-CycleGAN-and-pix2pix:

#         {
#           "A": <tensor>,        # domain A sample
#           "B": <tensor>,        # domain B sample
#           "A_paths": <str>,     # path/identifier for A
#           "B_paths": <str>,     # path/identifier for B
#         }

#     Notes
#     -----
#     - This does NOT add paired supervision to CycleGAN; it only formats data.
#     - Your underlying dataset can be paired (pre/post). CycleGAN will still
#       train with its usual unpaired objectives unless you modify the model.
#     - For 3D volumes, ensure tensors are shaped [C, D, H, W] per sample (so
#       DataLoader batches to [B, C, D, H, W]).
#     """

#     def __init__(
#         self,
#         paired_dataset: Dataset,
#         direction: str = "AtoB",
#         a_key: str = "A",
#         b_key: str = "B",
#         a_paths_key: str = "A_paths",
#         b_paths_key: str = "B_paths",
#         default_a_id_prefix: str = "A_",
#         default_b_id_prefix: str = "B_",
#         postprocess: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
#     ):
#         """
#         Args:
#             paired_dataset: Your dataset (e.g., PrePostFMRI) yielding:
#                 - (x, y) OR
#                 - (x, y, meta) where meta is a dict containing paths/ids.
#             direction: "AtoB" or "BtoA". If "BtoA", swaps domains so that:
#                 A becomes post and B becomes pre.
#             a_key/b_key: Keys for tensors in output dict (normally "A"/"B").
#             a_paths_key/b_paths_key: Keys for path strings in output dict.
#             default_a_id_prefix/default_b_id_prefix: Used if meta has no paths.
#             postprocess: Optional callable to edit/augment the output dict
#                          after formatting (e.g., ensure channel dim).
#         """
#         if direction not in ("AtoB", "BtoA"):
#             raise ValueError("direction must be 'AtoB' or 'BtoA'")
#         self.ds = paired_dataset
#         self.direction = direction
#         self.a_key = a_key
#         self.b_key = b_key
#         self.a_paths_key = a_paths_key
#         self.b_paths_key = b_paths_key
#         self.default_a_id_prefix = default_a_id_prefix
#         self.default_b_id_prefix = default_b_id_prefix
#         self.postprocess = postprocess

#     def __len__(self) -> int:
#         return len(self.ds)

#     @staticmethod
#     def _ensure_channel_first(x: torch.Tensor) -> torch.Tensor:
#         """
#         Ensures x is channel-first for 3D volumes.
#         Expected shapes:
#           - 3D: [D,H,W]  -> [1,D,H,W]
#           - 4D: [C,D,H,W] (leave as-is)
#         """
#         if x.ndim == 3:
#             return x.unsqueeze(0)
#         return x

#     def __getitem__(self, idx: int) -> Dict[str, Any]:
#         item = self.ds[idx]

#         # Accept (x, y) or (x, y, meta)
#         if isinstance(item, (tuple, list)) and len(item) == 2:
#             x, y = item
#             meta = {}
#         elif isinstance(item, (tuple, list)) and len(item) == 3:
#             x, y, meta = item
#             meta = meta or {}
#         else:
#             raise ValueError(
#                 "paired_dataset must return (x, y) or (x, y, meta). "
#                 f"Got type={type(item)} with len={len(item) if hasattr(item, '__len__') else 'NA'}"
#             )

#         # Make sure tensors are torch.Tensor
#         if not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor):
#             raise TypeError(f"Expected torch.Tensor outputs. Got x={type(x)}, y={type(y)}")

#         # Ensure channel-first (esp. if your PrePostFMRI left channel dim commented out)
#         x = self._ensure_channel_first(x)
#         y = self._ensure_channel_first(y)

#         # Extract "paths" if provided
#         # Your PrePostFMRI meta uses: pre_path, post_path, filename, id
#         pre_path = meta.get("pre_path")
#         post_path = meta.get("post_path")

#         # Provide stable fallbacks
#         if pre_path is None:
#             pre_path = meta.get("filename") or meta.get("id") or f"{self.default_a_id_prefix}{idx}"
#         if post_path is None:
#             post_path = meta.get("filename") or meta.get("id") or f"{self.default_b_id_prefix}{idx}"

#         # Direction controls swapping A/B
#         if self.direction == "AtoB":
#             A, B = x, y
#             A_path, B_path = str(pre_path), str(post_path)
#         else:  # "BtoA"
#             A, B = y, x
#             A_path, B_path = str(post_path), str(pre_path)

#         out: Dict[str, Any] = {
#             self.a_key: A,
#             self.b_key: B,
#             self.a_paths_key: A_path,
#             self.b_paths_key: B_path,
#         }

#         if self.postprocess is not None:
#             out = self.postprocess(out)

#         return out
        

from __future__ import annotations

from typing import Dict, Optional, Callable, Any, Tuple
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class CycleGANDictWrapper(Dataset):
    """
    Wraps a paired dataset that returns (x, y) or (x, y, meta) into the
    dict format expected by junyanz/pytorch-CycleGAN-and-pix2pix:

        {
          "A": <tensor>,        # domain A sample
          "B": <tensor>,        # domain B sample
          "A_paths": <str>,     # path/identifier for A
          "B_paths": <str>,     # path/identifier for B
        }

    Added feature:
      - Pad volumes from (91,109,91) to (92,109,92) on the D and W axes
        (and in general: pad D to pad_to[0], H to pad_to[1], W to pad_to[2]).

    Notes:
      - Padding happens AFTER ensuring channel-first, so tensors are [C,D,H,W].
      - Uses zero-padding, symmetric as much as possible.
    """

    def __init__(
        self,
        paired_dataset: Dataset,
        direction: str = "AtoB",
        a_key: str = "A",
        b_key: str = "B",
        a_paths_key: str = "A_paths",
        b_paths_key: str = "B_paths",
        default_a_id_prefix: str = "A_",
        default_b_id_prefix: str = "B_",
        postprocess: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        pad_to: Tuple[int, int, int] = (92, 109, 92),  # (D,H,W)
        pad_mode: str = "constant",
        pad_value: float = 0.0,
        strict_pad: bool = True,
    ):
        """
        Args:
            paired_dataset: Your dataset (e.g., PrePostFMRI) yielding:
                - (x, y) OR
                - (x, y, meta) where meta is a dict containing paths/ids.
            direction: "AtoB" or "BtoA". If "BtoA", swaps domains so that:
                A becomes post and B becomes pre.
            postprocess: Optional callable to edit/augment the output dict
                         after formatting.
            pad_to: Target spatial size (D,H,W) after padding.
            pad_mode: Passed to torch.nn.functional.pad (usually "constant").
            pad_value: Fill value when pad_mode == "constant".
            strict_pad: If True, raise if an input is larger than pad_to on any axis
                        (since this wrapper only pads, does not crop).
        """
        if direction not in ("AtoB", "BtoA"):
            raise ValueError("direction must be 'AtoB' or 'BtoA'")
        self.ds = paired_dataset
        self.direction = direction
        self.a_key = a_key
        self.b_key = b_key
        self.a_paths_key = a_paths_key
        self.b_paths_key = b_paths_key
        self.default_a_id_prefix = default_a_id_prefix
        self.default_b_id_prefix = default_b_id_prefix
        self.postprocess = postprocess

        self.pad_to = pad_to
        self.pad_mode = pad_mode
        self.pad_value = pad_value
        self.strict_pad = strict_pad

    def __len__(self) -> int:
        return len(self.ds)

    @staticmethod
    def _ensure_channel_first(x: torch.Tensor) -> torch.Tensor:
        """
        Ensures x is channel-first for 3D volumes.
        Expected shapes:
          - 3D: [D,H,W]  -> [1,D,H,W]
          - 4D: [C,D,H,W] (leave as-is)
        """
        if x.ndim == 3:
            return x.unsqueeze(0)
        return x

    def _pad_to_shape(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pads x (shape [C,D,H,W]) to self.pad_to (D,H,W).
        Only pads (no cropping). Uses symmetric padding where possible.
        """
        if x.ndim != 4:
            raise ValueError(f"Expected [C,D,H,W] tensor, got shape {tuple(x.shape)}")

        C, D, H, W = x.shape
        tD, tH, tW = self.pad_to

        if self.strict_pad:
            if D > tD or H > tH or W > tW:
                raise ValueError(
                    f"Input volume larger than pad_to. "
                    f"Got (D,H,W)=({D},{H},{W}) but pad_to={self.pad_to}. "
                    f"Set strict_pad=False or change pad_to / crop upstream."
                )

        pad_D = max(0, tD - D)
        pad_H = max(0, tH - H)
        pad_W = max(0, tW - W)

        if pad_D == 0 and pad_H == 0 and pad_W == 0:
            return x

        # F.pad padding order for 4D (C,D,H,W) is:
        # (W_left, W_right, H_left, H_right, D_left, D_right)
        w_left = pad_W // 2
        w_right = pad_W - w_left
        h_left = pad_H // 2
        h_right = pad_H - h_left
        d_left = pad_D // 2
        d_right = pad_D - d_left

        if self.pad_mode == "constant":
            return F.pad(x, (w_left, w_right, h_left, h_right, d_left, d_right), mode="constant", value=self.pad_value)
        else:
            # e.g. "replicate", "reflect" (reflect requires pad < dimension)
            return F.pad(x, (w_left, w_right, h_left, h_right, d_left, d_right), mode=self.pad_mode)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.ds[idx]

        # Accept (x, y) or (x, y, meta)
        if isinstance(item, (tuple, list)) and len(item) == 2:
            x, y = item
            meta = {}
        elif isinstance(item, (tuple, list)) and len(item) == 3:
            x, y, meta = item
            meta = meta or {}
        else:
            raise ValueError(
                "paired_dataset must return (x, y) or (x, y, meta). "
                f"Got type={type(item)} with len={len(item) if hasattr(item, '__len__') else 'NA'}"
            )

        if not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor outputs. Got x={type(x)}, y={type(y)}")

        # Ensure channel-first: [C,D,H,W]
        x = self._ensure_channel_first(x)
        y = self._ensure_channel_first(y)

        # Pad to target (D,H,W)
        x = self._pad_to_shape(x)
        y = self._pad_to_shape(y)

        # Extract paths
        pre_path = meta.get("pre_path")
        post_path = meta.get("post_path")

        if pre_path is None:
            pre_path = meta.get("filename") or meta.get("id") or f"{self.default_a_id_prefix}{idx}"
        if post_path is None:
            post_path = meta.get("filename") or meta.get("id") or f"{self.default_b_id_prefix}{idx}"

        # Direction controls swapping A/B
        if self.direction == "AtoB":
            A, B = x, y
            A_path, B_path = str(pre_path), str(post_path)
        else:  # "BtoA"
            A, B = y, x
            A_path, B_path = str(post_path), str(pre_path)

        out: Dict[str, Any] = {
            self.a_key: A,
            self.b_key: B,
            self.a_paths_key: A_path,
            self.b_paths_key: B_path,
        }

        if self.postprocess is not None:
            out = self.postprocess(out)

        return out