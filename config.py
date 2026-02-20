from dataclasses import dataclass


@dataclass
class Config:
    data_root: str = "fmri"  # folder containing pre_surgery and 6_months_post_surgery
    out_dir: str = "./runs/new_run"
    seed: int = 42

    # training
    epochs: int = 20
    lr: float = 3e-4
    batch_size: int = 4  # for 3D volumes often 1–2; A6000 can handle more depending on shape
    num_workers: int = 8
    val_frac: float = 0.15

    # model
    base_channels: int = 128  # increase to 48/64 if you want more capacity
    amp: bool = True
