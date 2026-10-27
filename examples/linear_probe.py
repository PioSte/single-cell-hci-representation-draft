"""Frozen-encoder linear probe on a small labeled synthetic task.

    uv run python examples/linear_probe.py

The dataset is :class:`SyntheticChannelDataset` (small defaults).  The label is
the binary presence of ``--target-channel`` in a sample's channel-id set.  This
is a non-trivial channel-set property: it depends on which identities are
present, not directly on the panel size, and a channel-count or mean-pixel
shortcut cannot read it off (channel count only changes its prior).

The script optionally pretrains the MAE on the train split for a few steps,
freezes every model parameter, extracts ``encode(pool="attention")``
descriptors with ``torch.no_grad``, standardizes them using train statistics,
and trains a single linear layer with Adam.  It prints train and test accuracy.
"""

from __future__ import annotations

import argparse
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

from fuse_channels import ChannelFusionMAE
from fuse_channels.data import SyntheticChannelDataset, collate_fn


def channel_presence_labels(
    channel_ids: Sequence[torch.Tensor], target_channel: int
) -> torch.Tensor:
    """Return ``[B]`` float labels: 1 when ``target_channel`` is present."""
    return torch.tensor(
        [float(bool((ids == target_channel).any())) for ids in channel_ids]
    )


def split_indices(
    n_samples: int, test_frac: float = 0.25, seed: int = 0
) -> Tuple[List[int], List[int]]:
    """Deterministic train/test split of dataset indices."""
    if not 0.0 < test_frac < 1.0:
        raise ValueError(f"test_frac must be in (0, 1), got {test_frac}")
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_samples, generator=g).tolist()
    n_test = max(1, int(round(n_samples * test_frac)))
    return perm[n_test:], perm[:n_test]


def _loader(
    ds: Dataset,
    indices: Sequence[int],
    batch_size: int,
    shuffle: bool,
    seed: int = 0,
) -> DataLoader:
    return DataLoader(
        Subset(ds, list(indices)),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        generator=torch.Generator().manual_seed(seed),
    )


def pretrain(
    model: ChannelFusionMAE,
    ds: Dataset,
    indices: Sequence[int],
    steps: int,
    batch_size: int = 8,
    lr: float = 1e-3,
    seed: int = 0,
) -> float:
    """Train the MAE on ``indices`` for ``steps`` batches; return the last loss."""
    if steps <= 0:
        return float("nan")
    loader = _loader(ds, indices, batch_size, shuffle=True, seed=seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    model.train()
    last_loss = float("nan")
    step = 0
    while step < steps:
        for images, ids in loader:
            out = model(images, ids)
            opt.zero_grad()
            out["loss"].backward()
            opt.step()
            last_loss = out["loss"].item()
            step += 1
            if step >= steps:
                break
    return last_loss


def extract_descriptors(
    model: ChannelFusionMAE,
    ds: Dataset,
    indices: Sequence[int],
    target_channel: int,
    batch_size: int = 32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return raw ``[N, D]`` descriptors and ``[N]`` labels."""
    loader = _loader(ds, indices, batch_size, shuffle=False)
    model.eval()
    features, labels = [], []
    with torch.no_grad():
        for images, ids in loader:
            _, pooled = model.encode(images, ids, pool="attention")
            features.append(pooled)
            labels.append(channel_presence_labels(ids, target_channel))
    return torch.cat(features, dim=0), torch.cat(labels, dim=0)


class LinearProbe(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)


def train_linear_probe(
    features: torch.Tensor,
    labels: torch.Tensor,
    steps: int = 50,
    lr: float = 1e-2,
) -> Tuple[LinearProbe, torch.Tensor, torch.Tensor]:
    """Fit train-set standardization and a linear classifier on frozen features."""
    mean = features.mean(dim=0)
    std = features.std(dim=0).clamp_min(1e-6)
    x = (features - mean) / std

    head = LinearProbe(features.shape[1])
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    for _ in range(steps):
        opt.zero_grad()
        loss = loss_fn(head(x), labels)
        loss.backward()
        opt.step()
    return head, mean, std


def probe_accuracy(
    head: LinearProbe,
    features: torch.Tensor,
    labels: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> float:
    with torch.no_grad():
        logits = head((features - mean) / std)
        predictions = (logits > 0).float()
    return float((predictions == labels).float().mean())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--samples", type=int, default=96)
    p.add_argument("--image-size", type=int, default=16)
    p.add_argument("--patch-size", type=int, default=8, help="canonical patch size p0")
    p.add_argument(
        "--patch-sizes",
        type=int,
        nargs="+",
        default=None,
        help="FlexiViT patch sizes available during pretraining "
        "(evaluation uses the canonical --patch-size)",
    )
    p.add_argument("--c-max", type=int, default=8)
    p.add_argument("--channel-ids", type=int, default=32)
    p.add_argument(
        "--target-channel",
        type=int,
        default=0,
        help="probe label: binary presence of this channel id",
    )
    p.add_argument("--test-frac", type=float, default=0.25)
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-memory", type=int, default=4)
    p.add_argument(
        "--fusion-attn-dim",
        type=int,
        default=None,
        help="Q/K/V attention width (default: 3*dim, Panopticon's D_attn=3D)",
    )
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--pretrain-steps", type=int, default=10)
    p.add_argument("--pretrain-lr", type=float, default=1e-3)
    p.add_argument("--probe-steps", type=int, default=50)
    p.add_argument("--probe-lr", type=float, default=1e-2)
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
    )
    p.add_argument("--pixel-size", type=float, default=None, help="um per pixel")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if not 0 <= args.target_channel < args.channel_ids:
        raise ValueError("--target-channel must be in [0, --channel-ids)")

    torch.manual_seed(args.seed)
    ds = SyntheticChannelDataset(
        n_samples=args.samples,
        image_size=args.image_size,
        c_max=args.c_max,
        n_channel_ids=args.channel_ids,
        seed=args.seed,
    )
    train_idx, test_idx = split_indices(args.samples, args.test_frac, args.split_seed)
    if not train_idx or not test_idx:
        raise ValueError("empty train or test split; increase --samples")

    train_labels = channel_presence_labels(
        [ds[i][1] for i in train_idx], args.target_channel
    )
    test_labels = channel_presence_labels(
        [ds[i][1] for i in test_idx], args.target_channel
    )
    print(
        f"train: {len(train_idx)} samples ({int(train_labels.sum())} positive), "
        f"test: {len(test_idx)} samples ({int(test_labels.sum())} positive)"
    )

    model = ChannelFusionMAE(
        image_size=args.image_size,
        patch_size=args.patch_size,
        patch_sizes=args.patch_sizes,
        dim=args.dim,
        encoder_depth=args.depth,
        decoder_depth=1,
        n_heads=args.n_heads,
        n_memory=args.n_memory,
        fusion_attn_dim=args.fusion_attn_dim,
        num_channel_ids=args.channel_ids,
        pos_embed=args.pos_embed,
        scale_conditioning=args.scale_conditioning,
        pixel_size=args.pixel_size,
    )

    if args.pretrain_steps > 0:
        last_loss = pretrain(
            model,
            ds,
            train_idx,
            args.pretrain_steps,
            batch_size=args.batch_size,
            lr=args.pretrain_lr,
            seed=args.seed,
        )
        print(f"pretrain: {args.pretrain_steps} steps, final loss {last_loss:.4f}")

    model.eval()
    model.requires_grad_(False)

    train_features, train_targets = extract_descriptors(
        model, ds, train_idx, args.target_channel
    )
    test_features, test_targets = extract_descriptors(
        model, ds, test_idx, args.target_channel
    )
    assert torch.equal(train_targets, train_labels)
    assert torch.equal(test_targets, test_labels)

    head, mean, std = train_linear_probe(
        train_features, train_targets, steps=args.probe_steps, lr=args.probe_lr
    )
    train_acc = probe_accuracy(head, train_features, train_targets, mean, std)
    test_acc = probe_accuracy(head, test_features, test_targets, mean, std)
    print(f"train accuracy: {train_acc:.4f}")
    print(f"test accuracy:  {test_acc:.4f}")


if __name__ == "__main__":
    main()
