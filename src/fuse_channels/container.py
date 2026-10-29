"""Container describing a batch of variable-channel token sets."""

from __future__ import annotations

from dataclasses import dataclass
import math
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
        batch size and the **maximum** number of spatial patches in the batch
        (a scalar keeps the original fixed-grid API; with variable patch sizes
        each sample may have a smaller ``n_patches_per_sample``).
    grid_h, grid_w:
        optional ``[n_samples]`` per-sample grid extents.
    n_patches_per_sample:
        optional ``[n_samples]`` patch counts.
    patch_row, patch_col:
        optional ``[T]`` row-major grid coordinates.  With a single patch size
        these are derivable from ``patch_idx`` and ``grid_w``.
    patch_size:
        optional ``[T]`` patch size in pixels for each token.
    pixel_size:
        ``[T]`` physical pixel size (micrometres per pixel). Required metadata:
        the tokenizer fills it from ``pixel_sizes`` and the model refuses to run
        without it, since every image has a physical pixel size.
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
    grid_h: Optional[torch.Tensor] = None
    grid_w: Optional[torch.Tensor] = None
    n_patches_per_sample: Optional[torch.Tensor] = None
    patch_row: Optional[torch.Tensor] = None
    patch_col: Optional[torch.Tensor] = None
    patch_size: Optional[torch.Tensor] = None
    pixel_size: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------ #
    def per_sample_of(self, values: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """Collapse a per-token ``[T]`` tensor to per-sample ``[n_samples]``.

        Patch size and pixel size are constant within a sample, so the last
        write wins.  Returns ``None`` when ``values`` is ``None``.
        """
        if values is None:
            return None
        out = values.new_zeros(self.n_samples)
        out[self.sample_idx] = values
        return out

    def location_group_ids(self) -> torch.Tensor:
        """Group id per token for *per-spatial-location* channel attention.

        ``n_patches`` is the batch maximum, so this is a unique dense group id
        even when samples have different grid sizes.
        """
        return self.sample_idx * self.n_patches + self.patch_idx

    def grid_coords(self):
        """Return ``(rows, cols, grid_h, grid_w)`` per token, inventing a square grid.

        Tokenizers fill these fields.  Hand-built :class:`ChannelSet` objects
        used by unit tests only have ``patch_idx``; the fallback keeps those
        usable by assuming a square grid.
        """
        if self.patch_row is not None and self.patch_col is not None:
            rows = self.patch_row
            cols = self.patch_col
        else:
            side = math.isqrt(self.n_patches)
            if side * side != self.n_patches:
                raise ValueError(
                    "ChannelSet has no patch_row/patch_col and n_patches is not a square"
                )
            rows = self.patch_idx // side
            cols = self.patch_idx % side
        if self.grid_h is not None and self.grid_w is not None:
            gh = self.grid_h[self.sample_idx]
            gw = self.grid_w[self.sample_idx]
        else:
            side = math.isqrt(self.n_patches)
            gh = torch.full_like(self.patch_idx, side)
            gw = torch.full_like(self.patch_idx, side)
        return rows, cols, gh, gw
