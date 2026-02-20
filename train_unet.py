import os
from typing import Optional
import torch
from torch.utils.data import DataLoader, random_split
from unet import UNet3D
from dataset import PrePostFMRI
from config import Config
from transform import ToChannelsFirstAndNormalize
from utils import seed_everything, make_union_mask, masked_l1
import json


@torch.no_grad()
def validate(model, loader, device, amp: bool):
    model.eval()
    total = 0.0
    total_id = 0.0
    n = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)  # [B,C,D,H,W]
        y = y.to(device, non_blocking=True)

        mask = make_union_mask(x, y)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
            pred_delta = model(x)     # model predicts delta
            pred = x + pred_delta     # reconstruct predicted post
            loss = masked_l1(pred, y, mask)

            # identity baseline: pred = x
            id_loss = masked_l1(x, y, mask)

        total += float(loss) * x.size(0)
        total_id += float(id_loss) * x.size(0)
        n += x.size(0)

    return total / max(n, 1), total_id / max(n, 1)



def train(
    model,
    train_loader,
    val_loader,
    device,
    epochs: int,
    lr: float,
    out_dir: str,
    amp: bool = True,
    grad_clip: Optional[float] = 1.0,
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    best_val = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        running_id = 0.0
        n = 0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)  # [B,C,D,H,W]
            y = y.to(device, non_blocking=True)

            mask = make_union_mask(x, y)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                pred_delta = model(x)
                pred = x + pred_delta
                loss = masked_l1(pred, y, mask)

                # identity baseline loss for monitoring (no grad needed but cheap)
                id_loss = masked_l1(x, y, mask)

            scaler.scale(loss).backward()

            if grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            scaler.step(optimizer)
            scaler.update()

            running += float(loss) * x.size(0)
            running_id += float(id_loss) * x.size(0)
            n += x.size(0)

        train_loss = running / max(n, 1)
        train_id = running_id / max(n, 1)

        val_loss, val_id = validate(model, val_loader, device, amp)

        print(
            f"Epoch {epoch:03d} | "
            f"train maskedL1: {train_loss:.5f} (id={train_id:.5f}) | "
            f"val maskedL1: {val_loss:.5f} (id={val_id:.5f})"
        )

        torch.save(
            {"epoch": epoch, "model": model.state_dict(), "opt": optimizer.state_dict()},
            os.path.join(out_dir, "last.pt"),
        )

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {"epoch": epoch, "model": model.state_dict(), "opt": optimizer.state_dict(), "best_val": best_val},
                os.path.join(out_dir, "best.pt"),
            )
            print(f"  ✓ saved best.pt (val={best_val:.5f})")


def main():
    cfg = Config()
    os.makedirs(cfg.out_dir, exist_ok=True)
    with open(os.path.join(cfg.out_dir, "config.json"), "w") as f:
        json.dump(cfg.__dict__, f, indent=2)

    seed_everything(cfg.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    transform = ToChannelsFirstAndNormalize(nonzero_mask=True)

    dataset = PrePostFMRI(
        root_dir=cfg.data_root,
        strict=True,
        transform=transform,
        return_paths=False,
    )

    # Infer in/out channels from a single sample
    x0, y0 = dataset[0]
    in_ch = x0.shape[0]   # after transform, x0 is [C,D,H,W]
    out_ch = y0.shape[0]
    print("Example shapes:", tuple(x0.shape), tuple(y0.shape), "| in_ch:", in_ch, "out_ch:", out_ch)

    # Split
    n_total = len(dataset)
    n_val = max(1, int(cfg.val_frac * n_total))
    n_train = n_total - n_val
    train_ds, val_ds = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(cfg.seed),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=(cfg.num_workers > 0),
        prefetch_factor=2 if cfg.num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=(cfg.num_workers > 0),
        prefetch_factor=2 if cfg.num_workers > 0 else None,
    )

    model = UNet3D(in_channels=in_ch, out_channels=out_ch, base=cfg.base_channels).to(device)
    print("Params:", sum(p.numel() for p in model.parameters()) / 1e6, "M")

    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=cfg.epochs,
        lr=cfg.lr,
        out_dir=cfg.out_dir,
        amp=cfg.amp and (device == "cuda"),
        grad_clip=1.0,
    )


if __name__ == "__main__":
    main()
