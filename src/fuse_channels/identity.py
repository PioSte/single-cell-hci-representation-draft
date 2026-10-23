"""Channel-identity embeddings.

Channel identity matters for this architecture: ChannelViT's channel-embedding
ablation shows that replacing per-channel embeddings with a shared mean
embedding severely degrades performance.  A learnable per-channel-id embedding
is the supported identity path for a closed/fixed panel.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ChannelEmbedding(nn.Module):
    """Map channel ids to ``[N, dim]`` additive channel-embedding vectors."""

    def __init__(self, dim: int, num_channel_ids: int) -> None:
        super().__init__()
        if num_channel_ids <= 0:
            raise ValueError("ChannelEmbedding requires a positive num_channel_ids")
        self.dim = dim
        self.embed = nn.Embedding(num_channel_ids, dim)
        nn.init.trunc_normal_(self.embed.weight, std=0.02)

    def forward(self, channel_ids: torch.Tensor) -> torch.Tensor:
        """Return ``[N, dim]`` additive embeddings for ``[N]`` channel ids."""
        return self.embed(channel_ids)
