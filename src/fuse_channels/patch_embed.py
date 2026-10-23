"""Channel-tied patch embedding and the channel-set tokenizer.

The projection is **tied across channels** (one shared kernel applied to every
single-channel image); ChannelViT shows tied filters beat untied ones and give
better robustness to varying channel sets.  A *shared* spatial positional
embedding is added so that position and channel identity stay disentangled.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .container import ChannelSet
from .identity import ChannelEmbedding


class ChannelTiedPatchEmbed(nn.Module):
    """``Conv2d(1 -> dim)`` shared by all channels, plus shared spatial pos."""

    def __init__(self, image_size: int, patch_size: int, dim: int) -> None:
        super().__init__()
        assert image_size % patch_size == 0, "image_size must be divisible by patch_size"
        self.image_size = image_size
        self.patch_size = patch_size
        self.n_patches = (image_size // patch_size) ** 2
        self.proj = nn.Conv2d(1, dim, kernel_size=patch_size, stride=patch_size)
        self.spatial_pos = nn.Parameter(torch.zeros(1, self.n_patches, dim))
        nn.init.trunc_normal_(self.spatial_pos, std=0.02)

    def forward(
        self,
        images: Sequence[torch.Tensor],
        channel_ids: Sequence[torch.Tensor],
        visible: Optional[Sequence[torch.Tensor]] = None,
    ):
        """Patchify a batch of variable-channel images.

        Parameters
        ----------
        images:
            list of ``[C_b, 1, H, W]`` tensors (one image per channel).
        channel_ids:
            list of ``[C_b]`` channel ids.
        visible:
            optional list of ``[C_b, n_patches]`` bool masks selecting which
            channel-patch tokens to *keep*.  Masked tokens are omitted, never
            zeroed.

        Returns
        -------
        tuple ``(tokens, meta)`` where ``meta`` holds ``sample_idx``,
        ``channel_idx``, ``patch_idx``, ``channel_id`` and ``channel_global``.
        """
        device = images[0].device
        counts = torch.tensor([im.shape[0] for im in images], device=device)
        n_samples = len(images)

        x = torch.cat([im.to(device) for im in images], dim=0)  # [ΣC,1,H,W]
        feat = self.proj(x)  # [ΣC, dim, h, w]
        feat = feat.flatten(2).transpose(1, 2)  # [ΣC, N, dim]
        feat = feat + self.spatial_pos  # shared across channels
        total_ch = feat.shape[0]
        feat = feat.reshape(total_ch * self.n_patches, -1)  # [ΣC*N, dim]

        sample_of_channel = torch.repeat_interleave(torch.arange(n_samples, device=device), counts)
        channel_idx_per_ch = torch.cat(
            [torch.arange(c, device=device) for c in counts.tolist()]
        )
        channel_id_per_ch = torch.cat([c.to(device) for c in channel_ids])

        sample_idx = sample_of_channel.repeat_interleave(self.n_patches)
        channel_idx = channel_idx_per_ch.repeat_interleave(self.n_patches)
        channel_id = channel_id_per_ch.repeat_interleave(self.n_patches)
        channel_global = torch.arange(total_ch, device=device).repeat_interleave(
            self.n_patches
        )
        patch_idx = torch.arange(self.n_patches, device=device).repeat(total_ch)

        if visible is not None:
            keep = torch.cat([v.reshape(-1).to(device) for v in visible], dim=0)
            feat = feat[keep]
            sample_idx = sample_idx[keep]
            channel_idx = channel_idx[keep]
            channel_id = channel_id[keep]
            channel_global = channel_global[keep]
            patch_idx = patch_idx[keep]

        meta = dict(
            sample_idx=sample_idx,
            channel_idx=channel_idx,
            patch_idx=patch_idx,
            channel_id=channel_id,
            channel_global=channel_global,
        )
        return feat, meta


class ChannelTokenizer(nn.Module):
    """Patch embed -> additive channel embedding -> :class:`ChannelSet`.

    The channel embedding is a learnable per-channel-id table.
    """

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        dim: int,
        channel_embedding: ChannelEmbedding,
    ) -> None:
        super().__init__()
        self.patch_embed = ChannelTiedPatchEmbed(image_size, patch_size, dim)
        self.n_patches = self.patch_embed.n_patches
        self.channel_embed = channel_embedding

    def forward(
        self,
        images: Sequence[torch.Tensor],
        channel_ids: Sequence[torch.Tensor],
        visible: Optional[Sequence[torch.Tensor]] = None,
    ) -> ChannelSet:
        n_samples = len(images)
        tokens, meta = self.patch_embed(images, channel_ids, visible)
        channel_global = meta["channel_global"]

        ids_per_ch = torch.cat([c.to(tokens.device) for c in channel_ids])
        embed_per_ch = self.channel_embed(ids_per_ch)
        channel_embed = embed_per_ch[channel_global]

        return ChannelSet(
            tokens=tokens + channel_embed,
            sample_idx=meta["sample_idx"],
            channel_idx=meta["channel_idx"],
            patch_idx=meta["patch_idx"],
            channel_id=meta["channel_id"],
            n_samples=n_samples,
            n_patches=self.n_patches,
            channel_embed=channel_embed,
        )

    # ------------------------------------------------------------------ #
    @staticmethod
    def extract_patches(images: Sequence[torch.Tensor], patch_size: int) -> torch.Tensor:
        """Unfold ``[C,1,H,W]`` channels into ``[ΣC, patch_size**2, N]`` pixels."""
        return torch.cat(
            [
                F.unfold(im.to(images[0].device), patch_size, stride=patch_size)
                for im in images
            ],
            dim=0,
        )
