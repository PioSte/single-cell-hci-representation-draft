"""Channel-tied patch embedding and the channel-set tokenizer.

The projection is **tied across channels** (one shared kernel applied to every
single-channel image); ChannelViT shows tied filters beat untied ones and give
better robustness to varying channel sets.  The kernel is learned at a
canonical patch size and **PI-resized** to the requested patch size for each
sample (FlexiViT).  There is no spatial positional embedding here: the encoder
gets positions from RoPE and the decoder from fractional absolute PE.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .container import ChannelSet
from .flexi import pi_resize_patch_embed
from .identity import ChannelEmbedding


def _hw(grid) -> Tuple[int, int]:
    if isinstance(grid, int):
        return grid, grid
    h, w = grid
    return int(h), int(w)


class ChannelTiedPatchEmbed(nn.Module):
    """``Conv2d(1 -> dim)`` shared by all channels, with PI-resized kernels.

    ``patch_size`` is the canonical size whose kernel is stored in ``self.proj``.
    Calling with another ``patch_sizes`` value PI-resizes that kernel on the
    fly.  No positional embedding is attached.
    """

    def __init__(
        self,
        image_size,
        patch_size: int,
        dim: int,
        patch_sizes: Optional[Sequence[int]] = None,
    ) -> None:
        super().__init__()
        self.image_h, self.image_w = _hw(image_size)
        if self.image_h % patch_size or self.image_w % patch_size:
            raise ValueError(
                "image_size must be divisible by patch_size; got "
                f"image_size={(self.image_h, self.image_w)}, patch_size={patch_size}"
            )
        self.image_size = image_size
        self.patch_size = int(patch_size)
        self.patch_sizes = tuple(sorted(set(int(p) for p in (patch_sizes or [patch_size]))))
        if self.patch_size not in self.patch_sizes:
            self.patch_sizes = tuple(sorted({*self.patch_sizes, self.patch_size}))
        for p in self.patch_sizes:
            if p < 1:
                raise ValueError(f"patch sizes must be positive, got {p}")
        self.n_patches = (self.image_h // self.patch_size) * (
            self.image_w // self.patch_size
        )
        self.proj = nn.Conv2d(1, dim, kernel_size=self.patch_size, stride=self.patch_size)

    def _project(self, x: torch.Tensor, patch_size: int) -> torch.Tensor:
        """Patchify ``[C, 1, H, W]`` and project to ``[C, N, dim]``."""
        p = int(patch_size)
        if p == self.patch_size:
            return self.proj(x)
        weight = pi_resize_patch_embed(self.proj.weight, p)
        return F.conv2d(x, weight, bias=self.proj.bias, stride=p)

    def forward(
        self,
        images: Sequence[torch.Tensor],
        channel_ids: Sequence[torch.Tensor],
        visible: Optional[Sequence[torch.Tensor]] = None,
        patch_sizes=None,
    ):
        """Patchify a batch of variable-channel images.

        Parameters
        ----------
        images:
            list of ``[C_b, 1, H, W]`` tensors (one image per channel).
        channel_ids:
            list of ``[C_b]`` channel ids.
        visible:
            optional list of ``[C_b, N_b]`` bool masks selecting which
            channel-patch tokens to *keep*.  Masked tokens are omitted, never
            zeroed.
        patch_sizes:
            optional int or per-sample sequence of patch sizes.  ``None`` uses
            the canonical ``patch_size``.

        Returns
        -------
        tuple ``(tokens, meta)``.  ``meta`` carries all token index tensors plus
        per-sample grid metadata.
        """
        device = images[0].device
        n_samples = len(images)
        ps = self._resolve_patch_sizes(n_samples, patch_sizes, device)

        tokens_list = []
        sample_idx, channel_idx, patch_idx = [], [], []
        channel_id, channel_global, patch_row, patch_col = [], [], [], []
        patch_size_tok = []
        grid_h_list, grid_w_list, n_patches_list = [], [], []
        channel_offset = 0

        for b, (im, ids, p) in enumerate(zip(images, channel_ids, ps)):
            C, _, H, W = im.shape
            p = int(p)
            if H % p or W % p:
                raise ValueError(
                    f"sample {b}: image {(H, W)} is not divisible by patch size {p}"
                )
            gh, gw = H // p, W // p
            N = gh * gw
            if N < 1:
                raise ValueError(f"sample {b}: patch size {p} leaves no tokens")
            x = im.to(device)
            feat = self._project(x, p)  # [C, dim, gh, gw]
            feat = feat.flatten(2).transpose(1, 2).reshape(C * N, -1)

            c_idx = torch.arange(C, device=device).repeat_interleave(N)
            p_idx = torch.arange(N, device=device).repeat(C)
            s_idx = torch.full((C * N,), b, dtype=torch.long, device=device)
            c_id = ids.to(device).repeat_interleave(N)
            c_global = torch.arange(
                channel_offset, channel_offset + C, device=device
            ).repeat_interleave(N)

            if visible is not None:
                v = visible[b].to(device)
                if tuple(v.shape) != (C, N):
                    raise ValueError(
                        f"visible[{b}] must have shape {(C, N)} for patch size {p}, "
                        f"got {tuple(v.shape)}"
                    )
                keep = v.reshape(-1)
                feat = feat[keep]
                c_idx = c_idx[keep]
                p_idx = p_idx[keep]
                s_idx = s_idx[keep]
                c_id = c_id[keep]
                c_global = c_global[keep]

            tokens_list.append(feat)
            sample_idx.append(s_idx)
            channel_idx.append(c_idx)
            patch_idx.append(p_idx)
            channel_id.append(c_id)
            channel_global.append(c_global)
            patch_row.append(p_idx // gw)
            patch_col.append(p_idx % gw)
            patch_size_tok.append(
                torch.full_like(p_idx, p, dtype=torch.float32, device=device)
            )
            grid_h_list.append(gh)
            grid_w_list.append(gw)
            n_patches_list.append(N)
            channel_offset += C

        tokens = torch.cat(tokens_list, dim=0) if tokens_list else images[0].new_zeros(0, self.proj.out_channels)
        meta = dict(
            sample_idx=torch.cat(sample_idx, dim=0) if sample_idx else torch.empty(0, dtype=torch.long, device=device),
            channel_idx=torch.cat(channel_idx, dim=0) if channel_idx else torch.empty(0, dtype=torch.long, device=device),
            patch_idx=torch.cat(patch_idx, dim=0) if patch_idx else torch.empty(0, dtype=torch.long, device=device),
            channel_id=torch.cat(channel_id, dim=0) if channel_id else torch.empty(0, dtype=torch.long, device=device),
            channel_global=torch.cat(channel_global, dim=0) if channel_global else torch.empty(0, dtype=torch.long, device=device),
            patch_row=torch.cat(patch_row, dim=0) if patch_row else torch.empty(0, dtype=torch.long, device=device),
            patch_col=torch.cat(patch_col, dim=0) if patch_col else torch.empty(0, dtype=torch.long, device=device),
            patch_size=torch.cat(patch_size_tok, dim=0) if patch_size_tok else torch.empty(0, device=device),
            grid_h=torch.tensor(grid_h_list, dtype=torch.long, device=device),
            grid_w=torch.tensor(grid_w_list, dtype=torch.long, device=device),
            n_patches_per_sample=torch.tensor(n_patches_list, dtype=torch.long, device=device),
            n_patches=max(n_patches_list) if n_patches_list else 0,
            patch_sizes=ps,
        )
        return tokens, meta

    def _resolve_patch_sizes(self, n_samples, patch_sizes, device):
        if patch_sizes is None:
            return [self.patch_size] * n_samples
        if isinstance(patch_sizes, int):
            return [int(patch_sizes)] * n_samples
        ps = [int(p) for p in patch_sizes]
        if len(ps) != n_samples:
            raise ValueError(
                f"expected {n_samples} patch sizes, got {len(ps)}"
            )
        return ps


class ChannelTokenizer(nn.Module):
    """Patch embed -> additive channel embedding -> :class:`ChannelSet`.

    The channel embedding is a learnable per-channel-id table.  The returned
    :class:`ChannelSet` carries per-token grid coordinates and per-sample grid
    metadata so downstream modules never need a fixed ``image_size`` lookup.
    """

    def __init__(
        self,
        image_size,
        patch_size: int,
        dim: int,
        channel_embedding: ChannelEmbedding,
        patch_sizes: Optional[Sequence[int]] = None,
    ) -> None:
        super().__init__()
        self.patch_embed = ChannelTiedPatchEmbed(
            image_size, patch_size, dim, patch_sizes=patch_sizes
        )
        self.n_patches = self.patch_embed.n_patches
        self.patch_size = self.patch_embed.patch_size
        self.patch_sizes = self.patch_embed.patch_sizes
        self.channel_embed = channel_embedding

    def forward(
        self,
        images: Sequence[torch.Tensor],
        channel_ids: Sequence[torch.Tensor],
        visible: Optional[Sequence[torch.Tensor]] = None,
        patch_sizes=None,
    ) -> ChannelSet:
        n_samples = len(images)
        tokens, meta = self.patch_embed(
            images, channel_ids, visible, patch_sizes=patch_sizes
        )
        channel_global = meta["channel_global"]

        ids_per_ch = torch.cat([c.to(tokens.device) for c in channel_ids])
        embed_per_ch = self.channel_embed(ids_per_ch)
        channel_embed = embed_per_ch[channel_global]

        return ChannelSet(
            tokens=tokens,
            sample_idx=meta["sample_idx"],
            channel_idx=meta["channel_idx"],
            patch_idx=meta["patch_idx"],
            channel_id=meta["channel_id"],
            n_samples=n_samples,
            n_patches=meta["n_patches"],
            channel_embed=channel_embed,
            channel_group_id=channel_global,
            grid_h=meta["grid_h"],
            grid_w=meta["grid_w"],
            n_patches_per_sample=meta["n_patches_per_sample"],
            patch_row=meta["patch_row"],
            patch_col=meta["patch_col"],
            patch_size=meta["patch_size"],
        )

    # ------------------------------------------------------------------ #
    @staticmethod
    def extract_patches(images: Sequence[torch.Tensor], patch_size) -> torch.Tensor:
        """Unfold ``[C,1,H,W]`` channels into ``[ΣC, patch_size**2, N]`` pixels.

        ``patch_size`` may be an int or one size per sample.  Mixed sizes must
        all produce the same number of pixels per patch (i.e. the same ``p``);
        for genuinely mixed patch sizes use :func:`extract_patches_list`.
        """
        if isinstance(patch_size, int):
            ps = [patch_size] * len(images)
        else:
            ps = [int(p) for p in patch_size]
        if len(set(ps)) == 1:
            p = ps[0]
            return torch.cat(
                [
                    F.unfold(im.to(images[0].device), p, stride=p)
                    for im in images
                ],
                dim=0,
            )
        return torch.cat(
            [
                F.unfold(im.to(images[0].device), p, stride=p)
                for im, p in zip(images, ps)
            ],
            dim=0,
        )

    @staticmethod
    def extract_patches_list(
        images: Sequence[torch.Tensor], patch_size
    ) -> List[torch.Tensor]:
        """Per-sample ``[C_b, p_b**2, N_b]`` patch tensors (variable ``p``)."""
        ps = (
            [patch_size] * len(images)
            if isinstance(patch_size, int)
            else [int(p) for p in patch_size]
        )
        return [
            F.unfold(im.to(images[0].device), p, stride=p)
            for im, p in zip(images, ps)
        ]
