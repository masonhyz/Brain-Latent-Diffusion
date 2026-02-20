import os
from pathlib import Path
from typing import Callable, Optional, Tuple, List, Union, Dict

import torch
from torch.utils.data import Dataset
import nibabel as nib


class PrePostFMRI(Dataset):
    """
    Paired dataset:
      input  = pre_surgery/<ID>.nii.gz
      target = 6_months_post_surgery/<ID>.nii.gz

    Assumes filenames match exactly between folders, e.g. 2020_076.nii.gz exists in both.
    """

    def __init__(
        self,
        root_dir: Union[str, Path],
        pre_dirname: str = "pre_surgery",
        post_dirname: str = "6_months_post_surgery",
        transform: Optional[Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]] = None,
        load_dtype: torch.dtype = torch.float32,
        strict: bool = True,
        return_paths: bool = False,
    ):
        """
        Args:
            root_dir: Path to the folder containing the two subfolders, e.g. ".../fmri".
            pre_dirname: Subfolder name for inputs.
            post_dirname: Subfolder name for targets.
            transform: Optional callable applied to (x, y) tensors. Should return (x, y).
                       Use this for normalization/cropping/augmentation (paired transforms).
            load_dtype: dtype to cast loaded volumes to.
            strict: If True, require every pre file has a matching post file (and vice versa).
                    If False, use intersection only.
            return_paths: If True, __getitem__ returns (x, y, meta) where meta includes paths + id.
        """
        self.root_dir = Path(root_dir)
        self.pre_dir = self.root_dir / pre_dirname
        self.post_dir = self.root_dir / post_dirname

        if not self.pre_dir.is_dir():
            raise FileNotFoundError(f"Pre-surgery folder not found: {self.pre_dir}")
        if not self.post_dir.is_dir():
            raise FileNotFoundError(f"Post-surgery folder not found: {self.post_dir}")

        self.transform = transform
        self.load_dtype = load_dtype
        self.return_paths = return_paths

        pre_files = sorted([p for p in self.pre_dir.glob("*.nii.gz")])
        post_files = sorted([p for p in self.post_dir.glob("*.nii.gz")])

        pre_map = {p.name: p for p in pre_files}
        post_map = {p.name: p for p in post_files}

        pre_names = set(pre_map.keys())
        post_names = set(post_map.keys())

        if strict:
            missing_in_post = sorted(list(pre_names - post_names))
            missing_in_pre = sorted(list(post_names - pre_names))
            if missing_in_post or missing_in_pre:
                msg = []
                if missing_in_post:
                    msg.append(f"{len(missing_in_post)} files missing in '{post_dirname}': e.g. {missing_in_post[:5]}")
                if missing_in_pre:
                    msg.append(f"{len(missing_in_pre)} files missing in '{pre_dirname}': e.g. {missing_in_pre[:5]}")
                raise ValueError("Paired dataset mismatch.\n" + "\n".join(msg))
            names = sorted(list(pre_names))
        else:
            names = sorted(list(pre_names & post_names))

        self.pairs: List[Tuple[str, Path, Path]] = [(n, pre_map[n], post_map[n]) for n in names]

    def __len__(self) -> int:
        return len(self.pairs)

    @staticmethod
    def _load_nii(path: Path) -> torch.Tensor:
        """
        Loads a NIfTI file to a torch tensor with shape [D, H, W] (or whatever the NIfTI contains),
        without adding a channel dimension.
        """
        img = nib.load(str(path))
        data = img.get_fdata(dtype="float32")  # numpy float32
        # NOTE: fMRI is often 4D (X,Y,Z,T). This keeps whatever shape is in the file.
        x = torch.from_numpy(data)
        return x

    def __getitem__(self, idx: int):
        sample_id, pre_path, post_path = self.pairs[idx]

        x = self._load_nii(pre_path).to(self.load_dtype)   # input
        y = self._load_nii(post_path).to(self.load_dtype)  # target

        # Common convention: add channel dim if 3D (or 4D). You can customize as needed.
        # If your model expects [C, ...], uncomment:
        # if x.ndim >= 3:
        #     x = x.unsqueeze(0)
        # if y.ndim >= 3:
        #     y = y.unsqueeze(0)

        if self.transform is not None:
            x, y = self.transform(x, y)

        if self.return_paths:
            meta: Dict[str, str] = {
                "id": sample_id.replace(".nii.gz", ""),
                "pre_path": str(pre_path),
                "post_path": str(post_path),
                "filename": sample_id,
            }
            return x, y, meta

        return x, y


# ---- Example usage ----
# dataset = PrePostFMRI(root_dir="/path/to/fmri", strict=True, return_paths=True)
# x, y, meta = dataset[0]
# print(meta, x.shape, y.shape)
#
# loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4)
