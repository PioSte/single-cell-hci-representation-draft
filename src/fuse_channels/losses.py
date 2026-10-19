"""Reconstruction losses for the channel-aware MAE.

Standard MAE normalizes the target patch before computing the MSE.  For
multi-channel imaging the intensity ranges of different channels differ by
orders of magnitude, so we additionally support **per-channel** normalization
(grouping targets by channel identity) so that bright channels do not dominate
the objective.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


def per_channel_normalized_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    channel_id: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
    per_channel: bool = False,
) -> torch.Tensor:
    """MSE between predicted and normalized target patches.

    Parameters
    ----------
    pred, target:
        ``[K, patch_dim]`` predictions and raw target pixels for ``K`` masked
        channel-patches.
    channel_id:
        ``[K]`` marker id of each target.  Required when ``per_channel=True``.
    per_channel:
        if True, normalize each target using per-channel mean/std computed over
        the batch; otherwise normalize each patch independently.
    """
    target = target.float()
    if per_channel:
        assert channel_id is not None
        norm_target = target.clone()
        for c in torch.unique(channel_id):
            sel = channel_id == c
            group = target[sel]
            mean = group.mean()
            std = group.std(unbiased=False)
            norm_target[sel] = (group - mean) / (std + eps)
    else:
        mean = target.mean(dim=-1, keepdim=True)
        std = target.std(dim=-1, keepdim=True, unbiased=False)
        norm_target = (target - mean) / (std + eps)
    return nn.functional.mse_loss(pred, norm_target)
