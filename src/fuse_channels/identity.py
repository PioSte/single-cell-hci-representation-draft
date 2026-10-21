"""Channel-identity embeddings.

Channel identity matters for this architecture: ChannelViT's channel-embedding
ablation shows that replacing per-channel embeddings with a shared mean
embedding severely degrades performance.  A learnable per-marker embedding is
the supported identity path for a closed/fixed panel.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ChannelIdentity(nn.Module):
    """Map marker ids to ``[N, dim]`` additive identity vectors."""

    def __init__(self, dim: int, num_markers: int) -> None:
        super().__init__()
        if num_markers <= 0:
            raise ValueError("ChannelIdentity requires a positive num_markers")
        self.dim = dim
        self.embed = nn.Embedding(num_markers, dim)
        nn.init.trunc_normal_(self.embed.weight, std=0.02)

    def forward(self, channel_ids: torch.Tensor) -> torch.Tensor:
        """Return ``[N, dim]`` identity vectors for ``[N]`` marker ids."""
        return self.embed(channel_ids)
