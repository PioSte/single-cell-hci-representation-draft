"""The channel-fusion tokenizer: the padding-free front end, as one stage.

``ChannelFusionTokenizer`` composes the two stages that turn a variable-channel
image set into a fixed-width token set:

1. :class:`~fuse_channels.patch_embed.ChannelTokenizer` -- channel-tied
   FlexiViT patch projection, token omission by an explicit ``visible`` mask,
   and the per-channel-id identity lookup (attached, never added to content);
2. a per-location channel fusion -- :class:`~fuse_channels.fusion.ChannelFusion`
   by default, or the ``O(C)`` :class:`~fuse_channels.fusion.IdentityGatedSum`
   ablation.

Contract: **one row per present ``(sample, patch)`` location, of fixed width
``D``**, invariant to the channel count ``C`` and to channel order, with no
padding rows and no positional embedding. The output is ``[K_f, D]`` plus the
``[B, N]`` presence index and the :class:`~fuse_channels.container.ChannelSet`
whose geometry metadata (grid, patch size, pixel size) later stages consume.

The stage stops at the fused token set on purpose. It excludes:

* scale conditioning (``log(p)`` / ``log(pixel_size)``), which is added to the
  fused tokens as encoder input;
* the encoder (2-D axial RoPE and attention);
* spatial pooling to fixed size latent;
* the decoder and its reconstruction head.

Position is born downstream, in the encoder's RoPE and the decoder's fractional
absolute PE, so nothing here depends on a fixed grid.

Both components are passed in built, so this stage can wrap the modules of an
existing model without duplicating parameters; use :meth:`build` for the
convenience constructor. :class:`~fuse_channels.mae.ChannelFusionMAE` composes
the same two modules directly under its own attribute names, which keeps its
state dict unchanged; wrapping them here is the way to apply this named stage to
those weights.
"""

from __future__ import annotations

import torch.nn as nn

from .container import ChannelSet
from .fusion import ChannelFusion, IdentityGatedSum
from .identity import ChannelEmbedding
from .patch_embed import ChannelTokenizer


class ChannelFusionTokenizer(nn.Module):
    """``{per-channel images, channel ids} -> ([K_f, D], present [B, N], ChannelSet)``.

    Parameters
    ----------
    tokenizer:
        a built :class:`~fuse_channels.patch_embed.ChannelTokenizer`.
    fusion:
        a built :class:`~fuse_channels.fusion.ChannelFusion` or
        :class:`~fuse_channels.fusion.IdentityGatedSum`.

    The default is positional-embedding free. ``pos_embed="learned"`` /
    ``"sincos"`` on the fusion module restores the legacy fixed-grid table, which
    makes the stage fixed-grid only.
    """

    def __init__(self, tokenizer: ChannelTokenizer, fusion: nn.Module) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.fusion = fusion

    # ------------------------------------------------------------------ #
    @classmethod
    def build(
        cls,
        image_size,
        patch_size: int,
        dim: int,
        channel_embedding: ChannelEmbedding,
        *,
        n_heads: int = 8,
        patch_sizes=None,
        fusion: str = "attention",
        fusion_attn_dim: int | None = None,
        fusion_pos_embed: str = "none",
        fusion_dropout: float = 0.0,
        fusion_n_patches: int | None = None,
    ) -> ChannelFusionTokenizer:
        """Construct the stage from configuration (``channel_embedding`` is shared).

        ``fusion_n_patches`` sizes the legacy table-based fusion PE; it defaults
        to the canonical patch count, which is where the table lives. The
        target architecture passes ``fusion_pos_embed="none"`` and needs no
        patch-count configuration at all.
        """
        tokenizer = ChannelTokenizer(
            image_size,
            patch_size,
            dim,
            channel_embedding,
            patch_sizes=patch_sizes,
        )
        if fusion_n_patches is None and fusion_pos_embed != "none":
            fusion_n_patches = tokenizer.n_patches
        if fusion == "attention":
            pooled = ChannelFusion(
                dim,
                n_heads=n_heads,
                n_patches=fusion_n_patches,
                dropout=fusion_dropout,
                pos_embed=fusion_pos_embed,
                attn_dim=fusion_attn_dim,
            )
        elif fusion == "identity_gated":
            pooled = IdentityGatedSum(
                dim,
                n_patches=fusion_n_patches,
                pos_embed=fusion_pos_embed,
            )
        else:
            raise ValueError(f"unknown fusion '{fusion}'; expected 'attention' or 'identity_gated'")
        return cls(tokenizer, pooled)

    # ------------------------------------------------------------------ #
    @property
    def dim(self) -> int:
        return self.tokenizer.patch_embed.proj.out_channels

    @property
    def patch_size(self) -> int:
        return self.tokenizer.patch_size

    @property
    def patch_sizes(self):
        return self.tokenizer.patch_sizes

    @property
    def n_patches(self) -> int:
        return self.tokenizer.n_patches

    # ------------------------------------------------------------------ #
    def forward(
        self,
        images,
        channel_ids,
        visible=None,
        patch_sizes=None,
        pixel_sizes=None,
    ):
        """Tokenize and fuse one batch of variable-channel samples.

        ``images`` is a list of ``[C_b, 1, H, W]`` planes, ``channel_ids`` the
        matching ``[C_b]`` identity keys. ``visible`` (list of ``[C_b, N_b]``
        bool) drops channel-patches by omission -- dropped rows do not exist, so
        a location with no visible channel is absent from the output.
        ``patch_sizes`` is an int or per-sample sequence (FlexiViT);
        ``pixel_sizes`` is per-sample µm/px, carried on the ``ChannelSet`` for
        the metric positional embeddings downstream.

        Returns
        -------
        tuple ``(fused, present, cs)``: ``fused`` is ``[K_f, D]`` in the order of
        ``present.nonzero()``, ``present`` is the ``[B, N]`` presence index, and
        ``cs`` is the tokenized :class:`~fuse_channels.container.ChannelSet`.
        """
        cs: ChannelSet = self.tokenizer(
            images,
            channel_ids,
            visible=visible,
            patch_sizes=patch_sizes,
            pixel_sizes=pixel_sizes,
        )
        fused, present = self.fusion(cs)
        return fused, present, cs
