"""Channel-aware masked autoencoder built on the padding-free fusion path.

Pipeline
--------
``images`` -> DCP masking (token omission) -> channel-tied patch embed +
additive channel embedding -> padding-free channel fusion -> memory-augmented
transformer encoder -> channel-aware shared decoder -> patch-normalized pixel
loss.

Masking omits rows from the flattened token tensor; no mask tokens and no attention masks
are used for masking.  The encoder only ever sees *visible* tokens.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch
import torch.nn as nn

from .container import ChannelSet
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
    ) -> None:
        super().__init__()
        self.patch_size = patch_size

        channel_embedding = ChannelEmbedding(dim, num_channel_ids)
        self.frontend = ChannelTokenizer(image_size, patch_size, dim, channel_embedding)
        n_patches = self.frontend.n_patches
        self.n_patches = n_patches

        if fusion == "attention":
            self.fusion = ChannelFusion(
                dim, n_heads=n_heads, n_queries=n_queries, n_patches=n_patches
            )
        elif fusion == "identity_gated":
            self.fusion = IdentityGatedSum(
                dim, n_queries=n_queries, n_patches=n_patches
            )
        else:
            raise ValueError(f"unknown fusion '{fusion}'")

        out_dim = n_queries * dim
        self.fuse_proj = nn.Identity() if out_dim == dim else nn.Linear(out_dim, dim)

        self.encoder = TransformerEncoder(
            dim, encoder_depth, n_heads, n_memory=n_memory
        )
        self.pooler = MultiheadAttentionPooling(dim, n_heads=n_heads, n_queries=1)
        self.decoder = ChannelAwareDecoder(
            dim, decoder_depth, n_heads, patch_dim=patch_size * patch_size,
            n_patches=n_patches,
        )
        self.masker = DCPVisibilitySampler(
            n_patches, p_patch=p_patch, p_channel=p_channel
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
        fused, present = self.fusion(cs)
        fused = self.fuse_proj(fused)  # [B, N, D]

        # -- padding-free encoder input: only present locations ---------- #
        enc_tokens = fused[present]  # [Kf, D]
        enc_group_id = torch.arange(B, device=device).repeat_interleave(present.sum(1))
        mem_out, tok_out = self.encoder(enc_tokens, enc_group_id, B)
        enc_all = torch.cat([mem_out, tok_out], dim=0)
        mem_group_id = torch.arange(B, device=device).repeat_interleave(
            self.encoder.n_memory
        )
        enc_all_group_id = torch.cat([mem_group_id, enc_group_id], dim=0)

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
            return {"loss": zero, "pred": None, "target": None}

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

        * ``"attention"`` (default) -- pooling by multi-head attention (PMA),
          a learned query cross-attends to the memory + visible tokens;
        * ``"mean"`` -- mean over the visible tokens.
        """
        visible = [
            torch.ones(im.shape[0], self.n_patches, dtype=torch.bool, device=im.device)
            for im in images
        ]
        cs = self.frontend(images, channel_ids, visible)
        fused, present = self.fusion(cs)
        fused = self.fuse_proj(fused)
        B, D = fused.shape[0], fused.shape[-1]
        enc_tokens = fused[present]
        enc_group_id = torch.arange(B, device=images[0].device).repeat_interleave(
            present.sum(1)
        )
        mem_out, tok_out = self.encoder(enc_tokens, enc_group_id, B)
        if pool == "attention":
            mem_group_id = torch.arange(B, device=images[0].device).repeat_interleave(
                self.encoder.n_memory
            )
            pooled = self.pooler(
                torch.cat([mem_out, tok_out], dim=0),
                torch.cat([mem_group_id, enc_group_id], dim=0),
                B,
            )
        elif pool == "mean":
            pooled = tok_out.new_zeros(B, D)
            pooled.index_add_(0, enc_group_id, tok_out)
            pooled = pooled / present.sum(1).clamp_min(1).unsqueeze(-1)
        else:
            raise ValueError(f"unknown pool '{pool}'")
        return mem_out, pooled
