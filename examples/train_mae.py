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
    p.add_argument("--patch-size", type=int, default=8, help="canonical patch size p0")
    p.add_argument(
        "--patch-sizes",
        type=int,
        nargs="+",
        default=None,
        help="FlexiViT patch sizes, e.g. --patch-sizes 4 8 16 (default: fixed p0)",
    )
    p.add_argument(
        "--no-flexi-random",
        action="store_true",
        help="do not sample one patch size per training step",
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="training sampler keeps p with (H//p)*(W//p) <= max_tokens",
    )
    p.add_argument(
        "--head-mode",
        choices=("per_p", "canonical"),
        default="per_p",
        help="decoder reconstruction head; canonical is the legacy ablation",
    )
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-memory", type=int, default=4)
    p.add_argument(
        "--fusion-attn-dim",
        type=int,
        default=None,
        help="Q/K/V attention width (default: 3*dim, Panopticon's D_attn=3D)",
    )
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
    p.add_argument(
        "--loss",
        choices=("mae", "cha_maevit"),
        default="mae",
        help="mae: patch-normalized MSE (default); "
        "cha_maevit: raw-pixel L2 + Fourier L1 ablation",
    )
    p.add_argument(
        "--pos-embed",
        choices=("fractional", "learned", "sincos"),
        default="fractional",
        help="fractional = encoder RoPE + decoder fractional PE (target); "
        "learned/sincos = legacy fixed-grid table ablation",
    )
    p.add_argument(
        "--scale-conditioning",
        choices=("add", "adaln", "none"),
        default="add",
        help="log(p) / log(um per px) conditioning",
    )
    p.add_argument("--pixel-size", type=float, default=None, help="um per pixel")
    p.add_argument(
        "--pixel-size-range",
        type=float,
        nargs=2,
        default=None,
        metavar=("MIN", "MAX"),
        help="sample per-sample um/px uniformly from this range",
    )
    p.add_argument("--rope-base", type=float, default=100.0)
    p.add_argument("--rope-ref-extent", type=float, default=32.0)
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
        pixel_size=args.pixel_size,
        pixel_size_range=args.pixel_size_range,
        return_pixel_size=(
            args.pixel_size is not None or args.pixel_size_range is not None
        ),
    )
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True
    )

    model = ChannelFusionMAE(
        image_size=args.image_size,
        patch_size=args.patch_size,
        patch_sizes=args.patch_sizes,
        flexi_random=not args.no_flexi_random,
        max_tokens=args.max_tokens,
        head_mode=args.head_mode,
        dim=args.dim,
        encoder_depth=args.depth,
        decoder_depth=2,
        n_heads=args.n_heads,
        n_memory=args.n_memory,
        fusion_attn_dim=args.fusion_attn_dim,
        num_channel_ids=args.channel_ids,
        p_patch=p_patch,
        p_channel=p_channel,
        patch_ratio=args.patch_ratio,
        mode_sampling=args.mode_sampling,
        loss=args.loss,
        pos_embed=args.pos_embed,
        scale_conditioning=args.scale_conditioning,
        pixel_size=args.pixel_size,
        rope_base=args.rope_base,
        rope_ref_extent=args.rope_ref_extent,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)

    step = 0
    t0 = time.time()
    model.train()
    last_batch = None
    while step < args.steps:
        for batch in loader:
            if len(batch) == 3:
                images, ids, pixel_sizes = batch
            else:
                images, ids = batch
                pixel_sizes = None
            images = [im.to(device) for im in images]
            ids = [i.to(device) for i in ids]
            last_batch = (images, ids, pixel_sizes)
            out = model(images, ids, pixel_sizes=pixel_sizes)
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

    if last_batch is not None and last_batch[2] is not None:
        # Diagnostic: does the scale conditioner actually use µm/px?  A ratio
        # near 1 means the model is ignoring the scale input.
        images, ids, pixel_sizes = last_batch
        model.eval()
        with torch.no_grad():
            gen = torch.Generator(device=images[0].device).manual_seed(123)
            correct = model(
                images, ids, generator=gen, pixel_sizes=pixel_sizes
            )["loss"].item()
            gen = torch.Generator(device=images[0].device).manual_seed(123)
            wrong = model(
                images, ids, generator=gen, pixel_sizes=list(reversed(pixel_sizes))
            )["loss"].item()
        print(
            f"scale sensitivity: correct {correct:.4f}, shuffled {wrong:.4f}, "
            f"ratio {wrong / max(correct, 1e-12):.3f}"
        )

    print("done.")


if __name__ == "__main__":
    main()
