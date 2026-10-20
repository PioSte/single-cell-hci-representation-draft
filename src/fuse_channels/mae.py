"""Channel-aware masked autoencoder built on the padding-free fusion path.

Pipeline
--------
``images`` -> DCP masking (token omission) -> channel-tied patch embed +
additive identity -> padding-free channel fusion -> memory-augmented spatial
transformer -> channel-aware shared decoder -> per-channel normalized pixel
loss.

Masking omits rows from the varlen tensor; no mask tokens and no attention masks
are used for masking.  The encoder only ever sees *visible* tokens.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch
import torch.nn as nn

from .container import ChannelSet
from .decoder import ChannelAwareDecoder
from .encoder import SpatialTransformer
from .fusion import ChannelFusion, HypernetworkWeightedSum, MultiheadAttentionPooling
from .identity import ChannelIdentity
from .losses import per_channel_normalized_mse
from .masking import DCPMasker
from .patch_embed import ChannelFrontend


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
        num_markers: int = 0,
        text_dim: Optional[int] = None,
        use_learnable_identity: bool = True,
        fusion: str = "attention",
        p_patch: float = 0.5,
        p_channel: float = 0.5,
        per_channel_loss: bool = False,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.per_channel_loss = per_channel_loss

        identity = ChannelIdentity(
            dim, num_markers, text_dim, use_learnable=use_learnable_identity
        )
        self.frontend = ChannelFrontend(image_size, patch_size, dim, identity)
        n_patches = self.frontend.n_patches
        self.n_patches = n_patches

        if fusion == "attention":
            self.fusion = ChannelFusion(
                dim, n_heads=n_heads, n_queries=n_queries, n_patches=n_patches
            )
        elif fusion == "hypernetwork":
            self.fusion = HypernetworkWeightedSum(
                dim, n_queries=n_queries, n_patches=n_patches
            )
        else:
            raise ValueError(f"unknown fusion '{fusion}'")

        out_dim = n_queries * dim
        self.fuse_proj = nn.Identity() if out_dim == dim else nn.Linear(out_dim, dim)

        self.encoder = SpatialTransformer(
            dim, encoder_depth, n_heads, n_memory=n_memory
        )
        self.pooler = MultiheadAttentionPooling(dim, n_heads=n_heads, n_queries=1)
        self.decoder = ChannelAwareDecoder(
            dim, decoder_depth, n_heads, patch_dim=patch_size * patch_size,
            n_patches=n_patches,
        )
        self.masker = DCPMasker(n_patches, p_patch=p_patch, p_channel=p_channel)

    # ------------------------------------------------------------------ #
    def forward(
        self,
        images: Sequence[torch.Tensor],
        channel_ids: Sequence[torch.Tensor],
        visible: Optional[Sequence[torch.Tensor]] = None,
        text_embeds: Optional[Sequence[torch.Tensor]] = None,
        generator: Optional[torch.Generator] = None,
    ):
        B = len(images)
        device = images[0].device

        if visible is None:
            visible = self.masker(images, generator=generator)

        cs = self.frontend(images, channel_ids, visible, text_embeds)
        fused, present = self.fusion(cs)
        fused = self.fuse_proj(fused)  # [B, N, D]

        # -- padding-free encoder input: only present locations ---------- #
        enc_tokens = fused[present]  # [Kf, D]
        enc_seg = torch.arange(B, device=device).repeat_interleave(present.sum(1))
        mem_out, tok_out = self.encoder(enc_tokens, enc_seg, B)
        enc_all = torch.cat([mem_out, tok_out], dim=0)
        mem_seg = torch.arange(B, device=device).repeat_interleave(
            self.encoder.n_memory
        )
        enc_all_seg = torch.cat([mem_seg, enc_seg], dim=0)

        # -- decoder targets: the masked channel-patches ----------------- #
        counts = torch.tensor([im.shape[0] for im in images], device=device)
        offsets = torch.zeros(B + 1, dtype=torch.long, device=device)
        offsets[1:] = torch.cumsum(counts, dim=0)

        tgt_seg, tgt_ch, tgt_patch = [], [], []
        for b in range(B):
            m = ~visible[b]
            ci, pi = m.nonzero(as_tuple=True)
            tgt_seg.append(torch.full_like(ci, b))
            tgt_ch.append(ci)
            tgt_patch.append(pi)
        tgt_seg = torch.cat(tgt_seg, 0)
        tgt_ch = torch.cat(tgt_ch, 0)
        tgt_patch = torch.cat(tgt_patch, 0)

        if tgt_seg.numel() == 0:
            zero = fused.sum() * 0.0
            return {"loss": zero, "pred": None, "target": None}

        gidx = offsets[tgt_seg] + tgt_ch  # global channel index of each target

        # identity of the target channels (recomputed; also feeds the decoder)
        ids_per_ch = torch.cat([c.to(device) for c in channel_ids])
        text_per_ch = (
            torch.cat([t.to(device) for t in text_embeds])
            if text_embeds is not None
            else None
        )
        ident_per_ch = self.frontend.identity(ids_per_ch, text_per_ch)
        tgt_identity = ident_per_ch[gidx]

        pred = self.decoder(
            enc_all, enc_all_seg, B, tgt_seg, tgt_patch, tgt_identity
        )

        patch_targets = ChannelFrontend.extract_patches(images, self.patch_size)
        target = patch_targets[gidx, :, tgt_patch]  # [Kt, patch_dim]
        target_cid = ids_per_ch[gidx]

        loss = per_channel_normalized_mse(
            pred, target, target_cid, per_channel=self.per_channel_loss
        )
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
        text_embeds: Optional[Sequence[torch.Tensor]] = None,
        pool: str = "attention",
    ):
        """Inference-time representation (no masking).

        Returns ``(memory, pooled_descriptor)`` where ``pooled_descriptor`` is
        ``[B, D]`` -- the fixed-size joint embedding for the whole batch.

        ``pool`` selects how the padding-free encoder outputs are reduced to
        ``[B, D]``:

        * ``"attention"`` (default) -- pooling by multi-head attention (PMA),
          a learned query cross-attends to the memory + visible tokens;
        * ``"mean"`` -- mean over the visible tokens;
        * ``"memory"`` -- mean over the fixed-count memory tokens.
        """
        visible = [
            torch.ones(im.shape[0], self.n_patches, dtype=torch.bool, device=im.device)
            for im in images
        ]
        cs = self.frontend(images, channel_ids, visible, text_embeds)
        fused, present = self.fusion(cs)
        fused = self.fuse_proj(fused)
        B, D = fused.shape[0], fused.shape[-1]
        enc_tokens = fused[present]
        enc_seg = torch.arange(B, device=images[0].device).repeat_interleave(
            present.sum(1)
        )
        mem_out, tok_out = self.encoder(enc_tokens, enc_seg, B)
        if pool == "attention":
            mem_seg = torch.arange(B, device=images[0].device).repeat_interleave(
                self.encoder.n_memory
            )
            pooled = self.pooler(
                torch.cat([mem_out, tok_out], dim=0),
                torch.cat([mem_seg, enc_seg], dim=0),
                B,
            )
        elif pool == "mean":
            pooled = tok_out.new_zeros(B, D)
            pooled.index_add_(0, enc_seg, tok_out)
            pooled = pooled / present.sum(1).clamp_min(1).unsqueeze(-1)
        elif pool == "memory":  # mean over memory tokens
            pooled = mem_out.reshape(B, self.encoder.n_memory, D).mean(1)
        else:
            raise ValueError(f"unknown pool '{pool}'")
        return mem_out, pooled
