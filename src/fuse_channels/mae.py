"""Channel-aware masked autoencoder built on the padding-free fusion path.

Pipeline
--------
``images`` -> DCP masking (token omission) -> channel-tied patch embed +
additive channel embedding -> padding-free channel fusion (+ post-fusion
positional embedding) -> memory-augmented transformer encoder -> trained PMA
descriptor -> channel-aware shared decoder -> patch-normalized pixel loss.

The descriptor is computed in :meth:`forward` from the encoder's *content*
tokens only (``tok_out``, not the memory tokens), appended as one extra
per-sample token to the decoder KV, and therefore receives gradients from the
reconstruction loss.  ``encode(pool="attention")`` returns that same
descriptor.  ``pool="mean"`` remains the content-only mean ablation.

Masking follows ChA-MAEViT Eq. 1 (patch, channel, or combined regions; see
:mod:`fuse_channels.masking`).  Masking omits rows from the flattened token
tensor; no mask tokens and no attention masks are used.  The encoder only ever
sees *visible* tokens.  Channels dropped by the channel region stay in
``images`` and are reconstructed as targets, so HCS and DCP's channel masking
are the same mechanism.

``p_patch=0, p_channel=1`` is HCS-only, ``p_patch=1, p_channel=0`` is
patch-only, and ``p_patch=p_channel=0`` is the combined region.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn

from .decoder import ChannelAwareDecoder
from .encoder import TransformerEncoder
from .fusion import ChannelFusion, IdentityGatedSum, MultiheadAttentionPooling
from .identity import ChannelEmbedding
from .losses import normalized_patch_mse
from .masking import DCPVisibilitySampler
from .patch_embed import ChannelTokenizer


class ChannelFusionMAE(nn.Module):
    def __init__(
        self,
        image_size: int,
        patch_size: int,
        dim: int = 192,
        encoder_depth: int = 4,
        decoder_depth: int = 2,
        n_heads: int = 6,
        n_memory: int = 4,
        n_queries: int = 1,
        num_channel_ids: int = 16,
        fusion: str = "attention",
        p_patch: float = 0.5,
        p_channel: float = 0.5,
        patch_ratio: float = 0.75,
        mode_sampling: str = "batch",
        min_visible_channels: int = 1,
        max_masked_channels_frac: float = 1.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.n_queries = n_queries

        channel_embedding = ChannelEmbedding(dim, num_channel_ids)
        self.frontend = ChannelTokenizer(image_size, patch_size, dim, channel_embedding)
        n_patches = self.frontend.n_patches
        self.n_patches = n_patches

        if fusion == "attention":
            self.fusion = ChannelFusion(
                dim, n_heads=n_heads, n_patches=n_patches, dropout=dropout
            )
        elif fusion == "identity_gated":
            self.fusion = IdentityGatedSum(dim, n_patches=n_patches)
        else:
            raise ValueError(f"unknown fusion '{fusion}'")

        self.encoder = TransformerEncoder(
            dim, encoder_depth, n_heads, n_memory=n_memory, dropout=dropout
        )
        # n_queries controls only the descriptor pooler; the readout is averaged
        # back to [B, D] so the descriptor can be one decoder KV token.
        self.pooler = MultiheadAttentionPooling(
            dim,
            n_heads=n_heads,
            n_queries=n_queries,
            dropout=dropout,
            post_sab=n_queries > 1,
            readout="mean",
        )
        self.decoder = ChannelAwareDecoder(
            dim, decoder_depth, n_heads, patch_dim=patch_size * patch_size,
            n_patches=n_patches, dropout=dropout,
        )
        self.masker = DCPVisibilitySampler(
            n_patches,
            p_patch=p_patch,
            p_channel=p_channel,
            patch_ratio=patch_ratio,
            mode_sampling=mode_sampling,
            min_visible_channels=min_visible_channels,
            max_masked_channels_frac=max_masked_channels_frac,
        )

    # ------------------------------------------------------------------ #
    def forward(
        self,
        images: Sequence[torch.Tensor],
        channel_ids: Sequence[torch.Tensor],
        visible: Optional[Sequence[torch.Tensor]] = None,
        generator: Optional[torch.Generator] = None,
    ):
        B = len(images)
        device = images[0].device

        if visible is None:
            visible = self.masker(images, generator=generator)

        cs = self.frontend(images, channel_ids, visible)
        fused, present = self.fusion(cs)  # [B, N, D] incl. post-fusion PE

        # -- padding-free encoder input: only present locations ---------- #
        enc_tokens = fused[present]  # [Kf, D]
        enc_group_id = torch.arange(B, device=device).repeat_interleave(present.sum(1))
        mem_out, tok_out = self.encoder(enc_tokens, enc_group_id, B)

        # -- trained descriptor from content tokens only ----------------- #
        desc = self.pooler(tok_out, enc_group_id, B)  # [B, D]

        enc_all = torch.cat([mem_out, tok_out, desc], dim=0)
        mem_group_id = torch.arange(B, device=device).repeat_interleave(
            self.encoder.n_memory
        )
        desc_group_id = torch.arange(B, device=device)
        enc_all_group_id = torch.cat(
            [mem_group_id, enc_group_id, desc_group_id], dim=0
        )

        # -- decoder targets: the masked channel-patches ----------------- #
        counts = torch.tensor([im.shape[0] for im in images], device=device)
        offsets = torch.zeros(B + 1, dtype=torch.long, device=device)
        offsets[1:] = torch.cumsum(counts, dim=0)

        tgt_group_id, tgt_ch, tgt_patch = [], [], []
        for b in range(B):
            m = ~visible[b]
            ci, pi = m.nonzero(as_tuple=True)
            tgt_group_id.append(torch.full_like(ci, b))
            tgt_ch.append(ci)
            tgt_patch.append(pi)
        tgt_group_id = torch.cat(tgt_group_id, 0)
        tgt_ch = torch.cat(tgt_ch, 0)
        tgt_patch = torch.cat(tgt_patch, 0)

        if tgt_group_id.numel() == 0:
            zero = fused.sum() * 0.0
            return {
                "loss": zero,
                "pred": None,
                "target": None,
                "descriptor": desc,
                "visible": visible,
                "present": present,
            }

        gidx = offsets[tgt_group_id] + tgt_ch  # global channel index of each target

        # embedding of the target channels (recomputed; also feeds the decoder)
        ids_per_ch = torch.cat([c.to(device) for c in channel_ids])
        embed_per_ch = self.frontend.channel_embed(ids_per_ch)
        tgt_channel_embed = embed_per_ch[gidx]

        pred = self.decoder(
            enc_all, enc_all_group_id, B, tgt_group_id, tgt_patch,
            tgt_channel_embed,
        )

        patch_targets = ChannelTokenizer.extract_patches(images, self.patch_size)
        target = patch_targets[gidx, :, tgt_patch]  # [Kt, patch_dim]
        target_cid = ids_per_ch[gidx]

        loss = normalized_patch_mse(pred, target)
        return {
            "loss": loss,
            "pred": pred,
            "target": target,
            "target_channel_id": target_cid,
            "descriptor": desc,
            "visible": visible,
            "present": present,
        }

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def encode(
        self,
        images: Sequence[torch.Tensor],
        channel_ids: Sequence[torch.Tensor],
        pool: str = "attention",
    ):
        """Inference-time representation (no masking).

        Returns ``(memory, pooled_descriptor)`` where ``pooled_descriptor`` is
        ``[B, D]`` -- the fixed-size joint embedding for the whole batch.

        ``pool`` selects how the padding-free encoder outputs are reduced to
        ``[B, D]``:

        * ``"attention"`` (default) -- pooling by multi-head attention (PMA)
          over the content tokens; this is the same descriptor the training
          forward computes and appends to the decoder KV;
        * ``"mean"`` -- mean over the content tokens (ablation).
        """
        visible = [
            torch.ones(im.shape[0], self.n_patches, dtype=torch.bool, device=im.device)
            for im in images
        ]
        cs = self.frontend(images, channel_ids, visible)
        fused, present = self.fusion(cs)
        B, D = fused.shape[0], fused.shape[-1]
        enc_tokens = fused[present]
        enc_group_id = torch.arange(B, device=images[0].device).repeat_interleave(
            present.sum(1)
        )
        mem_out, tok_out = self.encoder(enc_tokens, enc_group_id, B)
        if pool == "attention":
            pooled = self.pooler(tok_out, enc_group_id, B)
        elif pool == "mean":
            pooled = tok_out.new_zeros(B, D)
            pooled.index_add_(0, enc_group_id, tok_out)
            pooled = pooled / present.sum(1).clamp_min(1).unsqueeze(-1)
        else:
            raise ValueError(f"unknown pool '{pool}'")
        return mem_out, pooled
