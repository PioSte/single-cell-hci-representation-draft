"""Measure how much channel identity leaks into the trained descriptor.

The descriptor is a PMA over encoder patch tokens.  Those tokens carry the
fusion identity skip, so a panel-composition signal can enter the descriptor.
This script quantifies it without training:

1. same content, different channel IDs -> identity contribution;
2. different content, same channel IDs -> content contribution;
3. repeat with ``identity_skip`` zeroed;
4. optionally fit a linear probe from descriptor to the channel-ID multi-hot
   vector.  A positive TPR means panel composition is readable from the
   descriptor.

Run with ``uv run python examples/descriptor_identity_probe.py``.
"""

from __future__ import annotations

import argparse

import torch

from fuse_channels import ChannelFusionMAE


def _encode(model, images, ids, patch_size):
    with torch.no_grad():
        _, desc = model.encode(images, ids, patch_sizes=patch_size)
    return desc


def _probe(desc, labels, ridge=1e-2):
    n = len(desc)
    n_train = int(0.8 * n)
    xtr, xte = desc[:n_train], desc[n_train:]
    ytr, yte = labels[:n_train], labels[n_train:]
    w = torch.linalg.solve(
        xtr.T @ xtr + ridge * torch.eye(xtr.shape[1]), xtr.T @ ytr
    )
    pred = xte @ w
    ss_res = ((pred - yte) ** 2).sum(0)
    ss_tot = ((yte - yte.mean(0, keepdim=True)) ** 2).sum(0)
    valid = ss_tot > 1e-12
    r2 = (
        (1 - ss_res[valid] / ss_tot[valid]).mean().item()
        if valid.any()
        else float("nan")
    )
    pos = (yte > 0.5).sum().clamp_min(1)
    neg = (yte <= 0.5).sum().clamp_min(1)
    tpr = (((pred > 0.5) & (yte > 0.5)).sum() / pos).item()
    tnr = (((pred <= 0.5) & (yte <= 0.5)).sum() / neg).item()
    return r2, tpr, tnr


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--image-size", type=int, default=32)
    p.add_argument("--patch-size", type=int, default=8)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--channels", type=int, default=3)
    p.add_argument("--channel-ids", type=int, default=16)
    p.add_argument("--samples", type=int, default=600)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    model = ChannelFusionMAE(
        image_size=args.image_size,
        patch_size=args.patch_size,
        patch_sizes=(args.patch_size,),
        dim=args.dim,
        encoder_depth=2,
        decoder_depth=1,
        n_heads=4,
        n_memory=2,
        num_channel_ids=args.channel_ids,
        flexi_random=False,
    ).eval()

    g = torch.Generator().manual_seed(args.seed)
    base = [
        torch.randn(args.channels, 1, args.image_size, args.image_size, generator=g)
        for _ in range(4)
    ]
    ids_a = [
        (torch.arange(k * args.channels, (k + 1) * args.channels) % args.channel_ids)
        for k in range(4)
    ]
    ids_b = [
        torch.randperm(args.channel_ids)[: args.channels].sort().values
        for _ in range(4)
    ]

    def rel(a, b):
        return ((a - b).norm(dim=-1) / a.norm(dim=-1)).mean().item()

    d_a = _encode(model, base, ids_a, args.patch_size)
    d_b = _encode(model, base, ids_b, args.patch_size)
    other = [
        torch.randn(args.channels, 1, args.image_size, args.image_size, generator=g)
        for _ in range(4)
    ]
    d_c = _encode(model, other, ids_a, args.patch_size)
    print(f"same content, different IDs: {rel(d_a, d_b):.4f}")
    print(f"different content, same IDs: {rel(d_a, d_c):.4f}")

    with torch.no_grad():
        model.fusion.identity_skip.weight.zero_()
    images, ids, labels = [], [], []
    for _ in range(args.samples):
        sel = torch.randperm(args.channel_ids)[: args.channels].sort().values
        images.append(
            torch.randn(args.channels, 1, args.image_size, args.image_size)
        )
        ids.append(sel)
        y = torch.zeros(args.channel_ids)
        y[sel] = 1
        labels.append(y)
    labels = torch.stack(labels)
    desc = _encode(model, images, ids, args.patch_size)
    r2, tpr, tnr = _probe(desc, labels)
    print(f"normal channel-set probe: R2 {r2:.3f}, TPR {tpr:.3f}, TNR {tnr:.3f}")

    with torch.no_grad():
        model.fusion.identity_skip.weight.zero_()
    d_a0 = _encode(model, base, ids_a, args.patch_size)
    d_b0 = _encode(model, base, ids_b, args.patch_size)
    print(f"zero identity_skip, same content/different IDs: {rel(d_a0, d_b0):.4f}")
    desc0 = _encode(model, images, ids, args.patch_size)
    r2, tpr, tnr = _probe(desc0, labels)
    print(f"zero-skip channel-set probe: R2 {r2:.3f}, TPR {tpr:.3f}, TNR {tnr:.3f}")


if __name__ == "__main__":
    main()
