"""Channel-aware masked autoencoder built on the padding-free fusion path.

Pipeline
--------
``images`` -> DCP masking (token omission) -> channel-tied FlexiViT patch embed
+ additive channel embedding -> padding-free channel fusion -> encoder with
2-D axial RoPE -> trained PMA descriptor -> channel-aware decoder with
fractional absolute PE -> reconstruction loss.

The default positional scheme is the FlexiViT target from ``FLEXIVIT.md``:

* encoder: 2-D axial RoPE on normalized ``(row, col)`` coordinates;
* decoder: factorized-fractional absolute PE as a function of
  ``(row, col, grid_h, grid_w)``;
* fusion: no spatial table.

The legacy fixed-grid scheme (``pos_embed="learned"`` or ``"sincos"``) still
selects post-fusion absolute PE plus a decoder lookup table; it cannot change
patch size within a run.

Patch size is selectable per step or per sample.  The patch kernel is
PI-resized from a canonical patch size, masking and target extraction use the
dynamic patch counts, and the decoder head predicts on a canonical output grid
and resizes to the target patch size.

Scale conditioning embeds ``log(p)`` and, when available, ``log(µm/px)`` with
Fourier features and a small MLP.  ``scale_conditioning="add"`` (default) adds
the embedding to encoder and decoder tokens; ``"adaln"`` modulates the
LayerNorms instead; ``"none"`` disables it.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
import torch.nn as nn

from .decoder import ChannelAwareDecoder, PatchReconstructionHead
from .encoder import TransformerEncoder
from .fusion import ChannelFusion, IdentityGatedSum, MultiheadAttentionPooling
from .identity import ChannelEmbedding
from .losses import cha_maevit_recon_loss, normalized_patch_mse
from .masking import DCPVisibilitySampler
from .patch_embed import ChannelTokenizer
from .rope import axial_rope_cache

_LOSSES = {
    "mae": normalized_patch_mse,
    "cha_maevit": cha_maevit_recon_loss,
}

_LEGACY_POS = ("learned", "sincos")


class FourierScaleEmbedding(nn.Module):
    """Fourier features + MLP for ``[log p, log(µm/px), has_µ]``."""

    def __init__(
        self,
        dim: int,
        n_frequencies: int = 8,
        hidden: int | None = None,
    ) -> None:
        super().__init__()
        self.input_dim = 3
        self.n_frequencies = n_frequencies
        freqs = (2.0 ** torch.arange(n_frequencies, dtype=torch.float32)) * math.pi
        self.register_buffer("freqs", freqs)
        hidden = hidden or dim
        in_dim = self.input_dim * 2 * n_frequencies
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, dim)
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.shape[-1] != self.input_dim:
            raise ValueError(
                f"scale features must have last dim {self.input_dim}, got {values.shape[-1]}"
            )
        s = values.unsqueeze(-1) * self.freqs
        feat = torch.cat([s.sin(), s.cos()], dim=-1).flatten(-2)
        return self.mlp(feat)


def _as_hw(size):
    if isinstance(size, int):
        return size, size
    h, w = size
    return int(h), int(w)


def _n_patches_for(im: torch.Tensor, p: int) -> int:
    _, _, H, W = im.shape
    if H % p or W % p:
        raise ValueError(
            f"image {(H, W)} is not divisible by patch size {p}"
        )
    return (H // p) * (W // p)


class ChannelFusionMAE(nn.Module):
    def __init__(
        self,
        image_size,
        patch_size: int | Sequence[int] | None = None,
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
        loss: str = "mae",
        pos_embed: str | None = None,
        encoder_pos_embed: str | None = None,
        decoder_pos_embed: str | None = None,
        fusion_pos_embed: str | None = None,
        patch_sizes: Optional[Sequence[int]] = None,
        flexi_random: bool = True,
        rope_base: float = 100.0,
        rope_ref_extent: float = 32.0,
        rope_shared_extent: bool = False,
        scale_conditioning: str = "add",
        pixel_size=None,
        scale_fourier_frequencies: int = 8,
        fractional_pe_mode: str = "sincos",
        fractional_pe_base: float = 10000.0,
        head_patch_size: int | None = None,
    ) -> None:
        super().__init__()
        if loss not in _LOSSES:
            raise ValueError(
                f"unknown loss '{loss}'; expected one of {sorted(_LOSSES)}"
            )
        if scale_conditioning not in ("none", "add", "adaln"):
            raise ValueError(
                f"unknown scale_conditioning '{scale_conditioning}'; "
                "expected 'none', 'add' or 'adaln'"
            )

        # ---- patch-size / positional-scheme resolution ------------------ #
        if patch_size is None:
            if patch_sizes is None:
                raise ValueError("pass patch_size or patch_sizes")
            patch_size = patch_sizes[0]
        if isinstance(patch_size, (list, tuple)):
            patch_sizes = tuple(patch_size)
            patch_size = patch_size[0]
        self.patch_size = int(patch_size)
        if patch_sizes is None:
            patch_sizes = (self.patch_size,)
        self.patch_sizes = tuple(sorted({int(p) for p in patch_sizes} | {self.patch_size}))
        self.flexi_random = flexi_random

        self.legacy_pos_embed = pos_embed in _LEGACY_POS
        if pos_embed is not None and not self.legacy_pos_embed and pos_embed not in (
            "fractional",
            "rope",
            "none",
        ):
            raise ValueError(
                f"unknown pos_embed '{pos_embed}'; expected learned, sincos, "
                "fractional, rope or none"
            )
        if encoder_pos_embed is None:
            encoder_pos_embed = (
                "none" if self.legacy_pos_embed or pos_embed == "none" else "rope"
            )
        if decoder_pos_embed is None:
            if self.legacy_pos_embed:
                decoder_pos_embed = pos_embed
            elif pos_embed == "none":
                decoder_pos_embed = "none"
            else:
                decoder_pos_embed = "fractional"
        if fusion_pos_embed is None:
            fusion_pos_embed = pos_embed if self.legacy_pos_embed else "none"
        if encoder_pos_embed not in ("rope", "none"):
            raise ValueError(f"unknown encoder_pos_embed '{encoder_pos_embed}'")
        if decoder_pos_embed not in ("fractional", "none", *_LEGACY_POS):
            raise ValueError(f"unknown decoder_pos_embed '{decoder_pos_embed}'")
        if fusion_pos_embed not in ("none", *_LEGACY_POS):
            raise ValueError(f"unknown fusion_pos_embed '{fusion_pos_embed}'")
        if self.legacy_pos_embed and (
            encoder_pos_embed != "none"
            or decoder_pos_embed not in _LEGACY_POS
            or fusion_pos_embed not in _LEGACY_POS
        ):
            raise ValueError(
                "legacy pos_embed='learned'/'sincos' uses table PE on fusion and "
                "decoder; do not combine it with encoder/fusion overrides"
            )
        self.encoder_pos_embed = encoder_pos_embed
        self.decoder_pos_embed = decoder_pos_embed
        self.fusion_pos_embed = fusion_pos_embed
        self.rope_base = rope_base
        self.rope_ref_extent = rope_ref_extent
        self.rope_shared_extent = rope_shared_extent
        self.scale_conditioning = scale_conditioning

        self.loss_name = loss
        self._loss_fn = _LOSSES[loss]
        self.n_queries = n_queries

        channel_embedding = ChannelEmbedding(dim, num_channel_ids)
        self.tokenizer = ChannelTokenizer(
            image_size,
            self.patch_size,
            dim,
            channel_embedding,
            patch_sizes=self.patch_sizes,
        )
        base_n_patches = self.tokenizer.n_patches
        self.n_patches = base_n_patches
        self.image_size = image_size

        fusion_n_patches = (
            base_n_patches if fusion_pos_embed != "none" else None
        )
        if fusion == "attention":
            self.fusion = ChannelFusion(
                dim,
                n_heads=n_heads,
                n_patches=fusion_n_patches,
                dropout=dropout,
                pos_embed=fusion_pos_embed,
            )
        elif fusion == "identity_gated":
            self.fusion = IdentityGatedSum(
                dim,
                n_patches=fusion_n_patches,
                pos_embed=fusion_pos_embed,
            )
        else:
            raise ValueError(f"unknown fusion '{fusion}'")

        self.encoder = TransformerEncoder(
            dim,
            encoder_depth,
            n_heads,
            n_memory=n_memory,
            dropout=dropout,
            adaln=scale_conditioning == "adaln",
        )
        self.pooler = MultiheadAttentionPooling(
            dim,
            n_heads=n_heads,
            n_queries=n_queries,
            dropout=dropout,
            post_sab=n_queries > 1,
            readout="mean",
        )
        canonical_patch_size = int(head_patch_size or max(self.patch_sizes))
        self.decoder = ChannelAwareDecoder(
            dim,
            decoder_depth,
            n_heads,
            patch_dim=canonical_patch_size**2,
            n_patches=base_n_patches,
            dropout=dropout,
            pos_embed=decoder_pos_embed,
            canonical_patch_size=canonical_patch_size,
            fractional_pe_mode=fractional_pe_mode,
            fractional_pe_base=fractional_pe_base,
            adaln=scale_conditioning == "adaln",
        )
        self.masker = DCPVisibilitySampler(
            base_n_patches,
            p_patch=p_patch,
            p_channel=p_channel,
            patch_ratio=patch_ratio,
            mode_sampling=mode_sampling,
            min_visible_channels=min_visible_channels,
            max_masked_channels_frac=max_masked_channels_frac,
        )
        self.pixel_size = pixel_size
        if scale_conditioning != "none":
            self.scale_embed = FourierScaleEmbedding(
                dim, n_frequencies=scale_fourier_frequencies
            )
        else:
            self.scale_embed = None

        head_dim = dim // n_heads
        if encoder_pos_embed == "rope" and head_dim % 4 != 0:
            raise ValueError(
                "axial encoder RoPE needs dim / n_heads divisible by 4; got "
                f"head_dim={head_dim}"
            )

    # ------------------------------------------------------------------ #
    def _resolve_patch_sizes(self, images, patch_sizes, generator=None):
        n = len(images)
        if patch_sizes is not None:
            if isinstance(patch_sizes, int):
                ps = [int(patch_sizes)] * n
            else:
                ps = [int(p) for p in patch_sizes]
                if len(ps) != n:
                    raise ValueError(
                        f"expected {n} patch sizes, got {len(ps)}"
                    )
        elif self.legacy_pos_embed:
            ps = [self.patch_size] * n
        elif len(self.patch_sizes) > 1 and self.flexi_random and self.training:
            gen_device = (
                generator.device if generator is not None else torch.device("cpu")
            )
            choice = int(
                torch.randint(
                    len(self.patch_sizes),
                    (1,),
                    device=gen_device,
                    generator=generator,
                ).item()
            )
            ps = [self.patch_sizes[choice]] * n
        else:
            ps = [self.patch_size] * n
        if self.legacy_pos_embed and any(p != self.patch_size for p in ps):
            raise ValueError(
                "legacy table-based pos_embed cannot change patch size"
            )
        return ps

    def _resolve_pixel_sizes(self, n, pixel_sizes):
        if pixel_sizes is None:
            pixel_sizes = self.pixel_size
        if pixel_sizes is None:
            return [None] * n
        if torch.is_tensor(pixel_sizes) and pixel_sizes.ndim == 0:
            return [float(pixel_sizes)] * n
        if isinstance(pixel_sizes, (int, float)):
            return [float(pixel_sizes)] * n
        ps = [None if p is None else float(p) for p in pixel_sizes]
        if len(ps) != n:
            raise ValueError(
                f"expected {n} pixel sizes, got {len(ps)}"
            )
        return ps

    def _scale_values(self, patch_sizes, pixel_sizes, device):
        p = torch.tensor(patch_sizes, dtype=torch.float32, device=device)
        log_p = torch.log(p)
        has = torch.tensor(
            [0.0 if x is None else 1.0 for x in pixel_sizes],
            dtype=torch.float32,
            device=device,
        )
        log_phys = torch.tensor(
            [0.0 if x is None else math.log(x) for x in pixel_sizes],
            dtype=torch.float32,
            device=device,
        )
        return torch.stack([log_p, log_phys, has], dim=-1)

    def _prepare(self, images, channel_ids, visible, generator, patch_sizes):
        ps = self._resolve_patch_sizes(images, patch_sizes, generator)
        ns = [_n_patches_for(im, p) for im, p in zip(images, ps)]
        if visible is None:
            visible = self.masker(images, generator=generator, n_patches=ns)
        cs = self.tokenizer(
            images, channel_ids, visible=visible, patch_sizes=ps
        )
        return cs, ps, ns, visible

    def _encode(self, cs, ps, scale_emb=None):
        fused, present = self.fusion(cs)  # [B, Nmax, D]
        B, Nmax, D = fused.shape
        device = fused.device
        present_bi = present.nonzero(as_tuple=False)  # [Kf, 2]
        enc_tokens = fused[present]

        # normalized grid coordinates for present locations
        rows = present_bi[:, 1]
        gw = cs.grid_w[present_bi[:, 0]]
        gh = cs.grid_h[present_bi[:, 0]]
        cols = rows % gw
        rows = rows // gw

        rope = None
        if self.encoder_pos_embed == "rope":
            rope = axial_rope_cache(
                rows,
                cols,
                gh,
                gw,
                self.encoder.dim // self.encoder.n_heads,
                base=self.rope_base,
                ref_extent=self.rope_ref_extent,
                shared_extent=self.rope_shared_extent,
            )

        cond_enc = None
        if self.scale_conditioning == "add" and scale_emb is not None:
            enc_tokens = enc_tokens + scale_emb[present_bi[:, 0]]
        elif self.scale_conditioning == "adaln" and scale_emb is not None:
            cond_enc = scale_emb[present_bi[:, 0]]

        enc_group_id = present_bi[:, 0]
        mem_out, tok_out = self.encoder(
            enc_tokens, enc_group_id, B, rope=rope, cond=cond_enc
        )
        return fused, present, mem_out, tok_out, enc_group_id, present_bi

    # ------------------------------------------------------------------ #
    def forward(
        self,
        images: Sequence[torch.Tensor],
        channel_ids: Sequence[torch.Tensor],
        visible: Optional[Sequence[torch.Tensor]] = None,
        generator: Optional[torch.Generator] = None,
        patch_sizes=None,
        pixel_sizes=None,
    ):
        B = len(images)
        device = images[0].device
        cs, ps, ns, visible = self._prepare(
            images, channel_ids, visible, generator, patch_sizes
        )
        scale_values = None
        scale_emb = None
        if self.scale_conditioning != "none":
            scale_values = self._scale_values(
                ps, self._resolve_pixel_sizes(B, pixel_sizes), device
            )
            scale_emb = self.scale_embed(scale_values)
        fused, present, mem_out, tok_out, enc_group_id, present_bi = self._encode(
            cs, ps, scale_emb
        )
        desc = self.pooler(tok_out, enc_group_id, B)  # [B, D]

        enc_all = torch.cat([mem_out, tok_out, desc], dim=0)
        mem_group_id = torch.arange(B, device=device).repeat_interleave(
            self.encoder.n_memory
        )
        desc_group_id = torch.arange(B, device=device)
        enc_all_group_id = torch.cat(
            [mem_group_id, enc_group_id, desc_group_id], dim=0
        )

        # -- decoder targets: the masked channel-patches ------------------ #
        counts = torch.tensor([im.shape[0] for im in images], device=device)
        offsets = torch.zeros(B + 1, dtype=torch.long, device=device)
        offsets[1:] = torch.cumsum(counts, dim=0)

        tgt_group_id, tgt_ch, tgt_patch = [], [], []
        tgt_row, tgt_col, tgt_gh, tgt_gw, tgt_p = [], [], [], [], []
        for b in range(B):
            m = ~visible[b]
            ci, pi = m.nonzero(as_tuple=True)
            p = ps[b]
            gw = ns[b] // (images[b].shape[-2] // p)
            tgt_group_id.append(torch.full_like(ci, b))
            tgt_ch.append(ci)
            tgt_patch.append(pi)
            tgt_row.append(pi // gw)
            tgt_col.append(pi % gw)
            tgt_gh.append(torch.full_like(pi, images[b].shape[-2] // p))
            tgt_gw.append(torch.full_like(pi, gw))
            tgt_p.append(torch.full_like(pi, p))
        if tgt_group_id:
            tgt_group_id = torch.cat(tgt_group_id, 0)
            tgt_ch = torch.cat(tgt_ch, 0)
            tgt_patch = torch.cat(tgt_patch, 0)
            tgt_row = torch.cat(tgt_row, 0)
            tgt_col = torch.cat(tgt_col, 0)
            tgt_gh = torch.cat(tgt_gh, 0)
            tgt_gw = torch.cat(tgt_gw, 0)
            tgt_p = torch.cat(tgt_p, 0)
        else:
            tgt_group_id = torch.empty(0, dtype=torch.long, device=device)
            tgt_ch = tgt_patch = tgt_row = tgt_col = tgt_gh = tgt_gw = tgt_p = tgt_group_id

        if tgt_group_id.numel() == 0:
            zero = fused.sum() * 0.0
            return {
                "loss": zero,
                "pred": None,
                "target": None,
                "descriptor": desc,
                "visible": visible,
                "present": present,
                "patch_sizes": ps,
                "grid_h": cs.grid_h,
                "grid_w": cs.grid_w,
                "n_patches_per_sample": cs.n_patches_per_sample,
                "scale_values": scale_values,
            }

        gidx = offsets[tgt_group_id] + tgt_ch  # global channel index of each target
        ids_per_ch = torch.cat([c.to(device) for c in channel_ids])
        embed_per_ch = self.tokenizer.channel_embed(ids_per_ch)
        tgt_channel_embed = embed_per_ch[gidx]

        cond_dec = None
        if self.scale_conditioning == "add" and scale_emb is not None:
            tgt_channel_embed = tgt_channel_embed + scale_emb[tgt_group_id]
        elif self.scale_conditioning == "adaln" and scale_emb is not None:
            cond_dec = scale_emb[tgt_group_id]

        pred = self.decoder(
            enc_all,
            enc_all_group_id,
            B,
            tgt_group_id,
            tgt_patch,
            tgt_channel_embed,
            target_row=tgt_row,
            target_col=tgt_col,
            target_grid_h=tgt_gh,
            target_grid_w=tgt_gw,
            target_patch_size=tgt_p,
            cond=cond_dec,
        )

        target, pred_mask = self._build_targets(
            images, ps, tgt_group_id, tgt_ch, tgt_patch, tgt_p, pred, gidx
        )
        loss = self._compute_loss(pred, target, tgt_p, pred_mask)

        target_cid = ids_per_ch[gidx]
        out = {
            "loss": loss,
            "pred": pred,
            "target": target,
            "target_channel_id": target_cid,
            "target_patch_size": tgt_p,
            "pred_mask": pred_mask,
            "descriptor": desc,
            "visible": visible,
            "present": present,
            "patch_sizes": ps,
            "grid_h": cs.grid_h,
            "grid_w": cs.grid_w,
            "n_patches_per_sample": cs.n_patches_per_sample,
            "scale_values": scale_values,
        }
        if pred_mask is not None:
            out["pred"] = pred.masked_fill(~pred_mask, 0.0)
        return out

    def _build_targets(
        self, images, ps, tgt_group_id, tgt_ch, tgt_patch, tgt_p, pred, gidx
    ):
        device = pred.device
        if len(set(ps)) == 1:
            p = ps[0]
            patch_targets = ChannelTokenizer.extract_patches(images, p)
            target = patch_targets[gidx, :, tgt_patch]
            return target, None

        max_dim = int((tgt_p * tgt_p).max().item())
        target = pred.new_zeros(pred.shape[0], max_dim)
        patches = ChannelTokenizer.extract_patches_list(images, ps)
        for b in range(len(images)):
            sel = (tgt_group_id == b).nonzero(as_tuple=True)[0]
            if sel.numel() == 0:
                continue
            p2 = int(ps[b]) ** 2
            target[sel, :p2] = patches[b][tgt_ch[sel], :, tgt_patch[sel]]
        return target, PatchReconstructionHead.mask(tgt_p, device=device)

    def _compute_loss(self, pred, target, tgt_p, pred_mask):
        if pred_mask is None:
            return self._loss_fn(pred, target)
        if self.loss_name == "mae":
            return normalized_patch_mse(pred, target, mask=pred_mask)
        # ChA-MAEViT Fourier term is defined per square patch; evaluate each
        # patch-size group separately and weight by row count.
        total = pred.new_zeros(())
        n = pred.shape[0]
        for p in torch.unique(tgt_p).tolist():
            idx = (tgt_p == p).nonzero(as_tuple=True)[0]
            dim = int(p) ** 2
            total = total + self._loss_fn(
                pred[idx, :dim], target[idx, :dim]
            ) * (idx.numel() / n)
        return total

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def encode(
        self,
        images: Sequence[torch.Tensor],
        channel_ids: Sequence[torch.Tensor],
        pool: str = "attention",
        patch_sizes=None,
        pixel_sizes=None,
    ):
        """Inference-time representation (no masking).

        Returns ``(memory, pooled_descriptor)`` where ``pooled_descriptor`` is
        ``[B, D]`` -- the fixed-size joint embedding for the whole batch.
        """
        B = len(images)
        device = images[0].device
        ps = self._resolve_patch_sizes(images, patch_sizes)
        ns = [_n_patches_for(im, p) for im, p in zip(images, ps)]
        visible = [
            torch.ones(im.shape[0], n, dtype=torch.bool, device=im.device)
            for im, n in zip(images, ns)
        ]
        cs = self.tokenizer(images, channel_ids, visible=visible, patch_sizes=ps)
        scale_values = None
        scale_emb = None
        if self.scale_conditioning != "none":
            scale_values = self._scale_values(
                ps, self._resolve_pixel_sizes(B, pixel_sizes), device
            )
            scale_emb = self.scale_embed(scale_values)
        _fused, present, mem_out, tok_out, enc_group_id, _bi = self._encode(
            cs, ps, scale_emb
        )
        D = tok_out.shape[-1]
        if pool == "attention":
            pooled = self.pooler(tok_out, enc_group_id, B)
        elif pool == "mean":
            pooled = tok_out.new_zeros(B, D)
            pooled.index_add_(0, enc_group_id, tok_out)
            pooled = pooled / present.sum(1).clamp_min(1).unsqueeze(-1)
        else:
            raise ValueError(f"unknown pool '{pool}'")
        return mem_out, pooled
