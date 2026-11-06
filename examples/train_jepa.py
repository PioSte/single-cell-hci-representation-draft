"""Train the channel-fusion JEPA on the synthetic variable-channel dataset.

The only difference from the MAE loop is the objective and the EMA target
update after each optimizer step. The defaults keep the regularizer regime
honest: 4 locations per 16x16 image at patch 8, so a batch of 32 gives up to
~128 regularizer rows against `--dim 32` (rows > dim). A small batch with a
wide embedding makes both regularizers' sampling floor dominate their value.

    uv run python examples/train_jepa.py --steps 200
    uv run python examples/train_jepa.py --steps 200 --regularizer sigreg --reg-weight 0.01
    uv run python examples/train_jepa.py --steps 200 --no-ema --regularizer sigreg --reg-weight 0.01

Every report line includes the within-sample feature spread and effective rank
(`feature_spread`): a run whose `jepa_loss` falls while `within_std` collapses is
degenerating, not learning. That failure is invisible to both loss terms and to
a probe over the pooled latents.
"""

from __future__ import annotations

import argparse
import time

import torch
from torch.utils.data import DataLoader

from fuse_channels import JEPAConfig, build_jepa, feature_spread
from fuse_channels.data import SyntheticChannelDataset, collate_fn


def _reg_kwargs(args) -> dict:
    """Regularizer constructor kwargs implied by the CLI flags."""
    if args.regularizer == "sigreg":
        return {"reduce": args.reg_reduce}
    if args.regularizer == "vicreg":
        return {"bias_correct": args.bias_correct}
    return {}


def _report(step: int, out: dict, elapsed: float) -> str:
    """One report line: losses, regularizer regime, and the collapse diagnostic."""
    rows = out["reg_rows"]
    dim = out["reg_dim"]
    parts = [
        f"step {step:5d}",
        f"loss {out['loss'].item():.4f}",
        f"jepa {out['jepa_loss'].item():.4f}",
        f"reg {out['reg_loss'].item():.4f}",
        f"rows {rows}",
        f"dim {dim}",
    ]
    if out["pred"] is not None and rows > 1:
        spread = feature_spread(out["pred"], out.get("target_group_id"))
        parts.append(f"within_std {spread['within_sample_std']:.4f}")
        parts.append(f"eff_rank {spread['effective_rank']:.2f}")
    parts.append(f"({elapsed:.1f}s)")
    return "  ".join(parts)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--image-size", type=int, default=16)
    p.add_argument("--patch-size", type=int, default=8, help="canonical patch size p0")
    p.add_argument("--patch-sizes", type=int, nargs="+", default=None)
    p.add_argument("--no-flexi-random", action="store_true")
    p.add_argument("--dim", type=int, default=32)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-memory", type=int, default=4)
    p.add_argument("--n-latents", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--c-max", type=int, default=8)
    p.add_argument("--channel-ids", type=int, default=32)
    p.add_argument("--pixel-size", type=float, default=1.0, help="um per pixel")
    p.add_argument("--pixel-size-range", type=float, nargs=2, default=None, metavar=("MIN", "MAX"))
    p.add_argument("--target-blocks", type=int, default=4)
    p.add_argument("--channel-dropout", action="store_true")
    p.add_argument("--loss", choices=("l1", "l2"), default="l1")
    p.add_argument("--ema-momentum", type=float, default=0.996)
    p.add_argument(
        "--no-ema", action="store_true", help="no EMA target encoder; needs a regularizer"
    )
    p.add_argument("--regularizer", choices=("none", "sigreg", "vicreg"), default="none")
    p.add_argument("--reg-weight", type=float, default=0.0)
    p.add_argument(
        "--reg-reduce",
        choices=("mean", "sum"),
        default="mean",
        help="SIGReg row convention; 'sum' is LeJEPA Algorithm 1 verbatim",
    )
    p.add_argument(
        "--bias-correct",
        action="store_true",
        help="subtract VICReg's (dim-1)/(rows-1) covariance sampling floor",
    )
    p.add_argument(
        "--no-projector",
        action="store_true",
        help="predict in encoder feature space instead of the projector space",
    )
    p.add_argument("--proj-dim", type=int, default=None)
    args = p.parse_args()

    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = SyntheticChannelDataset(
        n_samples=256,
        image_size=args.image_size,
        c_max=args.c_max,
        n_channel_ids=args.channel_ids,
        pixel_size=None if args.pixel_size_range else args.pixel_size,
        pixel_size_range=args.pixel_size_range,
        return_pixel_size=True,
    )
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True
    )

    model = build_jepa(
        jepa=JEPAConfig(
            loss=args.loss,
            use_ema=not args.no_ema,
            ema_momentum=args.ema_momentum,
            regularizer=args.regularizer,
            reg_weight=args.reg_weight,
            reg_kwargs=_reg_kwargs(args),
            use_projector=not args.no_projector,
            proj_dim=args.proj_dim,
            target_blocks=args.target_blocks,
            channel_dropout=args.channel_dropout,
        ),
        image_size=args.image_size,
        patch_size=args.patch_size,
        patch_sizes=args.patch_sizes,
        flexi_random=not args.no_flexi_random,
        dim=args.dim,
        encoder_depth=args.depth,
        decoder_depth=2,
        n_heads=args.n_heads,
        n_memory=args.n_memory,
        n_latents=args.n_latents,
        num_channel_ids=args.channel_ids,
        pixel_size=args.pixel_size,
    ).to(device)

    # The target encoder is frozen; only the student + predictor are optimized.
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.05
    )

    step = 0
    t0 = time.time()
    model.train()
    while step < args.steps:
        for images, ids, pixel_sizes in loader:
            images = [im.to(device) for im in images]
            ids = [i.to(device) for i in ids]
            out = model(images, ids, pixel_sizes=pixel_sizes)
            opt.zero_grad()
            out["loss"].backward()
            opt.step()
            model.update_target_encoder()
            step += 1
            if step % 20 == 0 or step == 1:
                print(_report(step, out, time.time() - t0))
            if step >= args.steps:
                break


if __name__ == "__main__":
    main()
