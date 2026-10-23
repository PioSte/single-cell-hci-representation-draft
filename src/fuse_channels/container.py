"""Container describing a batch of variable-channel token sets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class ChannelSet:
    """A flattened, padding-free batch of per-channel patch tokens.

    All tensors are indexed over the ``T`` *real* tokens in the batch.  Tokens
    are conceptually ordered ``(sample, channel, patch)`` but consumers should
    rely on the index tensors rather than on the ordering.

    Attributes
    ----------
    tokens:
        ``[T, D]`` patch embeddings (already projected; channel embedding and
        position optional).
    sample_idx:
        ``[T]`` sample id in ``[0, n_samples)``.
    channel_idx:
        ``[T]`` *panel position* of the channel within its sample, in
        ``[0, C_b)``.
    patch_idx:
        ``[T]`` spatial patch position in ``[0, n_patches)``.
    channel_id:
        ``[T]`` channel *identity key* indexing the embedding table (constant
        across the patches of a channel).  Unlike ``channel_idx`` this is a
        dataset-level channel id, not a panel position.
    channel_group_id:
        ``[T]`` global channel index across the whole batch (constant across a
        channel's patches).  Unlike ``channel_id`` this is a *position* in the
        concatenated batch, so it uniquely identifies one channel instance even
        when the same identity appears twice.  Used by :class:`IdentityGatedSum`
        to evaluate its channel MLP once per channel.
    channel_embed:
        ``[T, D]`` additive channel-embedding vector (learnable), if already
        computed.
    n_samples, n_patches:
        batch size and (fixed) number of spatial patches.
    """

    tokens: torch.Tensor
    sample_idx: torch.Tensor
    channel_idx: torch.Tensor
    patch_idx: torch.Tensor
    channel_id: torch.Tensor
    n_samples: int
    n_patches: int
    channel_embed: Optional[torch.Tensor] = None
    channel_group_id: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------ #
    def location_group_ids(self) -> torch.Tensor:
        """Group id per token for *per-spatial-location* channel attention."""
        return self.sample_idx * self.n_patches + self.patch_idx
