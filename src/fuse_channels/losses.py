"""Reconstruction loss for the channel-aware MAE.

The target patch is normalized before computing the MSE, matching the standard
MAE objective.  There is no per-channel normalization: ChA-MAEViT uses plain
pixel L2 plus a Fourier L1 term.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def normalized_patch_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """MSE between predicted and per-patch-normalized target pixels.

    Parameters
    ----------
    pred, target:
        ``[K, patch_dim]`` predictions and raw target pixels for ``K`` masked
        channel-patches.
    """
    target = target.float()
    mean = target.mean(dim=-1, keepdim=True)
    std = target.std(dim=-1, keepdim=True, unbiased=False)
    norm_target = (target - mean) / (std + eps)
    return nn.functional.mse_loss(pred, norm_target)
