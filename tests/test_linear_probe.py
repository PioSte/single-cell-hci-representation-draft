import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples.linear_probe import (
    extract_features,
    probe_accuracy,
    split_indices,
    train_linear_probe,
)
from fuse_channels import ChannelFusionMAE
from fuse_channels.data import SyntheticChannelDataset


def test_linear_probe_smoke():
    torch.manual_seed(0)
    image_size, patch_size, dim = 8, 4, 16
    ds = SyntheticChannelDataset(
        n_samples=16,
        image_size=image_size,
        c_max=4,
        n_channel_ids=8,
        pixel_size=1.0,
        seed=1,
    )
    train_idx, test_idx = split_indices(len(ds), test_frac=0.25, seed=0)
    assert train_idx and test_idx
    assert not set(train_idx) & set(test_idx)

    model = ChannelFusionMAE(
        image_size=image_size,
        patch_size=patch_size,
        dim=dim,
        encoder_depth=1,
        decoder_depth=1,
        n_heads=2,
        n_memory=2,
        num_channel_ids=8,
        pixel_size=1.0,
    )
    model.eval()
    model.requires_grad_(False)

    features, labels = extract_features(model, ds, train_idx, target_channel=0, batch_size=8)
    assert features.shape == (len(train_idx), dim)
    assert labels.shape == (len(train_idx),)
    assert torch.isfinite(features).all()

    head, mean, std = train_linear_probe(features, labels, steps=5, lr=1e-2)
    assert head.linear.weight.shape == (1, dim)
    assert mean.shape == std.shape == (dim,)

    train_acc = probe_accuracy(head, features, labels, mean, std)
    assert math.isfinite(train_acc)
    assert 0.0 <= train_acc <= 1.0

    test_features, test_labels = extract_features(
        model, ds, test_idx, target_channel=0, batch_size=8
    )
    assert test_features.shape == (len(test_idx), dim)
    assert test_labels.shape == (len(test_idx),)
    test_acc = probe_accuracy(head, test_features, test_labels, mean, std)
    assert math.isfinite(test_acc)
    assert 0.0 <= test_acc <= 1.0
