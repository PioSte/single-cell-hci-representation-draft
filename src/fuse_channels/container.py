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
        ``[T, D]`` patch embeddings (already projected; identity/pos optional).
    sample_idx:
        ``[T]`` sample id in ``[0, n_samples)``.
    channel_idx:
        ``[T]`` channel position within its sample in ``[0, C_b)``.
    patch_idx:
        ``[T]`` spatial patch position in ``[0, n_patches)``.
    channel_id:
        ``[T]`` marker identity id (constant across the patches of a channel).
    identity:
        ``[T, D]`` additive channel-identity vector (learnable or
        text/sequence conditioned), if already computed.
    n_samples, n_patches:
        batch size and (fixed) number of spatial patches.
    seq_len:
        ``[n_samples]`` number of visible tokens per sample.
    """

    tokens: torch.Tensor
    sample_idx: torch.Tensor
    channel_idx: torch.Tensor
    patch_idx: torch.Tensor
    channel_id: torch.Tensor
    n_samples: int
    n_patches: int
    identity: Optional[torch.Tensor] = None
    seq_len: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------ #
    def to(self, device) -> "ChannelSet":
        def mv(x):
            return x.to(device) if torch.is_tensor(x) else x

        return ChannelSet(
            tokens=mv(self.tokens),
            sample_idx=mv(self.sample_idx),
            channel_idx=mv(self.channel_idx),
            patch_idx=mv(self.patch_idx),
            channel_id=mv(self.channel_id),
            n_samples=self.n_samples,
            n_patches=self.n_patches,
            identity=mv(self.identity),
            seq_len=mv(self.seq_len),
        )

    def location_segments(self) -> torch.Tensor:
        """Segment id per token for *per-spatial-location* channel attention."""
        return self.sample_idx * self.n_patches + self.patch_idx

    def channel_segments(self) -> torch.Tensor:
        """Segment id per token for *per-sample* (global) channel attention."""
        return self.sample_idx
