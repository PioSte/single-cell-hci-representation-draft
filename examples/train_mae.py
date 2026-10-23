"""Train the channel-aware MAE on the synthetic variable-channel dataset.

    uv run python examples/train_mae.py --steps 200
"""

from __future__ import annotations

import argparse
import time

import torch
from torch.utils.data import DataLoader

from fuse_channels import ChannelFusionMAE
from fuse_channels.data import SyntheticChannelDataset, collate_fn


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--image-size", type=int, default=16)
    p.add_argument("--patch-size", type=int, default=8)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-memory", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--c-max", type=int, default=8)
    p.add_argument("--channel-ids", type=int, default=32)
    p.add_argument("--p-patch", type=float, default=0.5)
    p.add_argument("--p-channel", type=float, default=0.5)
    p.add_argument("--patch-ratio", type=float, default=0.75)
    p.add_argument(
        "--mode-sampling", choices=("batch", "sample"), default="batch"
    )
    p.add_argument(
        "--hcs",
        action="store_true",
        help="HCS-only DCP (sets --p-patch 0 --p-channel 1)",
    )
    args = p.parse_args()

    p_patch, p_channel = args.p_patch, args.p_channel
    if args.hcs:
        p_patch, p_channel = 0.0, 1.0

    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = SyntheticChannelDataset(
        n_samples=256,
        image_size=args.image_size,
        c_max=args.c_max,
        n_channel_ids=args.channel_ids,
    )
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True
    )

    model = ChannelFusionMAE(
        image_size=args.image_size,
        patch_size=args.patch_size,
        dim=args.dim,
        encoder_depth=args.depth,
        decoder_depth=2,
        n_heads=args.n_heads,
        n_memory=args.n_memory,
        num_channel_ids=args.channel_ids,
        p_patch=p_patch,
        p_channel=p_channel,
        patch_ratio=args.patch_ratio,
        mode_sampling=args.mode_sampling,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)

    step = 0
    t0 = time.time()
    model.train()
    while step < args.steps:
        for images, ids in loader:
            images = [im.to(device) for im in images]
            ids = [i.to(device) for i in ids]
            out = model(images, ids)
            opt.zero_grad()
            out["loss"].backward()
            opt.step()
            step += 1
            if step % 20 == 0 or step == 1:
                print(
                    f"step {step:5d}  loss {out['loss'].item():.4f}  "
                    f"({time.time() - t0:.1f}s)"
                )
            if step >= args.steps:
                break

    print("done.")


if __name__ == "__main__":
    main()
