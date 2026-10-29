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
dynamic patch counts, and the decoder keeps one linear reconstruction head per
configured patch size (with a canonical bilinear fallback for unseen sizes).
Patches tile ``floor(H/p) x floor(W/p)``; a border smaller than ``p`` is
discarded rather than rejected.  ``max_tokens`` optionally restricts the random
training sampler to patch sizes whose token count fits the budget; an explicit
``patch_sizes=`` argument overrides that guard.

Scale conditioning embeds ``log(p)`` and, when available, ``log(µm/px)`` with
Fourier features and a small MLP.  ``scale_conditioning="add"`` (default) adds
the embedding to encoder and decoder tokens; ``"adaln"`` modulates the
LayerNorms instead; ``"none"`` disables it.
"""

from __future__ import annotations

import math
import warnings
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
    """Number of patches for one sample.

    Matches FlexiViT's ``floor(H/p) * floor(W/p)`` contract: the border smaller
    than ``p`` is ignored, not padded and not treated as an error.
    """
    _, _, H, W = im.shape
    n = (H // p) * (W // p)
    if n < 1:
        raise ValueError(
            f"image {(H, W)} leaves no tokens at patch size {p}"
        )
    return n


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
        fusion_attn_dim: int | None = None,
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
        pe_space: str = "grid",
        pe_ref_um: float = 1.0,
        loss_reduce: str = "token",
        keep_offcell_as_context: bool = True,
        head_patch_size: int | None = None,
        head_mode: str = "per_p",
        max_tokens: int | None = None,
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
        if head_mode not in ("per_p", "canonical"):
            raise ValueError(
                f"unknown head_mode '{head_mode}'; expected 'per_p' or 'canonical'"
            )
        if max_tokens is not None and int(max_tokens) < 1:
            raise ValueError(f"max_tokens must be >= 1 or None, got {max_tokens}")
        if pe_space not in ("grid", "physical", "auto"):
            raise ValueError(
                f"unknown pe_space '{pe_space}'; expected 'grid', 'physical' or 'auto'"
            )
        if loss_reduce not in ("token", "sample"):
            raise ValueError(
                f"unknown loss_reduce '{loss_reduce}'; expected 'token' or 'sample'"
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
        self.max_tokens = None if max_tokens is None else int(max_tokens)

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
        self.pe_space = pe_space
        self.pe_ref_um = float(pe_ref_um)
        self.loss_reduce = loss_reduce
        self.keep_offcell_as_context = bool(keep_offcell_as_context)
        self._warned_nonmetric = False

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
                attn_dim=fusion_attn_dim,
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
            pe_ref_um=pe_ref_um,
            adaln=scale_conditioning == "adaln",
            patch_sizes=self.patch_sizes,
            head_mode=head_mode,
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
    def _tokens_for(self, im: torch.Tensor, p: int) -> int:
        H, W = im.shape[-2:]
        return (H // p) * (W // p)

    def _within_budget(self, images, p: int) -> bool:
        if self.max_tokens is None:
            return True
        return all(self._tokens_for(im, p) <= self.max_tokens for im in images)

    def _resolve_patch_sizes(self, images, patch_sizes, generator=None):
        n = len(images)
        if patch_sizes is not None:
            # An explicit patch size is an intentional override; max_tokens is
            # a guard on the random training sampler, not an inference cap.
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
            candidates = [p for p in self.patch_sizes if self._within_budget(images, p)]
            if not candidates:
                raise ValueError(
                    "max_tokens excludes every configured patch size; "
                    f"max_tokens={self.max_tokens}, patch_sizes={self.patch_sizes}"
                )
            gen_device = (
                generator.device if generator is not None else torch.device("cpu")
            )
            choice = int(
                torch.randint(
                    len(candidates),
                    (1,),
                    device=gen_device,
                    generator=generator,
                ).item()
            )
            ps = [candidates[choice]] * n
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

    def _prepare(self, images, channel_ids, visible, generator, patch_sizes,
                 pixel_sizes=None, valid=None):
        """Resolve sizes, draw the DCP mask, and apply the off-cell token mask.

        ``valid`` is an optional per-sample ``[N_b]`` boolean mask in full-grid
        row-major order (cell-token mask, *not* an MAE visibility mask).  It
        restores the distinction between three states that ``visible`` alone
        cannot express:

        * ``valid & visible``   -- context in the encoder
        * ``valid & ~visible``  -- MAE reconstruction target
        * ``~valid``            -- neither (off-cell; never attended, never
          reconstructed), unless ``keep_offcell_as_context`` is set, in which
          case it is fed as context only.
        """
        ps = self._resolve_patch_sizes(images, patch_sizes, generator)
        ns = [_n_patches_for(im, p) for im, p in zip(images, ps)]
        if visible is None:
            visible = self.masker(images, generator=generator, n_patches=ns)
        else:
            visible = [v.clone() for v in visible]

        target_valid = None
        if valid is not None:
            target_valid = []
            for b, (v, N) in enumerate(zip(valid, ns)):
                v = v.to(device=images[0].device, dtype=torch.bool)
                if v.numel() != N:
                    raise ValueError(
                        f"valid[{b}] must have {N} entries for patch size "
                        f"{ps[b]}, got {v.numel()}"
                    )
                keep = v[None, :].expand_as(visible[b])
                if self.keep_offcell_as_context:
                    # off-cell patches may be attended, but are never targets
                    visible[b] = visible[b] | (~keep)
                else:
                    visible[b] = visible[b] & keep
                target_valid.append(keep)

        if (
            self.encoder_pos_embed == "rope"
            and self.pe_space == "grid"
            and len(set(ns)) > 1
            and not self._warned_nonmetric
        ):
            warnings.warn(
                "variable patch count across samples with pe_space='grid': "
                "grid-fraction coordinates encode distance relative to the "
                "window extent, so positional phases are not comparable "
                "between differently sized cells. Pass pe_space='physical' "
                "with pixel_sizes for metric coordinates.",
                stacklevel=2,
            )
            self._warned_nonmetric = True

        cs = self.tokenizer(
            images,
            channel_ids,
            visible=visible,
            patch_sizes=ps,
            pixel_sizes=pixel_sizes,
        )
        return cs, ps, ns, visible, target_valid

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
            metric = self._use_metric_pe(cs)
            sample_of = present_bi[:, 0]
            rope = axial_rope_cache(
                rows,
                cols,
                gh,
                gw,
                self.encoder.dim // self.encoder.n_heads,
                base=self.rope_base,
                ref_extent=self.rope_ref_extent,
                shared_extent=self.rope_shared_extent,
                patch_size=cs.per_sample_of(cs.patch_size)[sample_of]
                if metric
                else None,
                pixel_size=cs.per_sample_of(cs.pixel_size)[sample_of]
                if metric
                else None,
                ref_um=self.pe_ref_um,
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

    def _target_step_um(self, cs, ps, ns, tgt_group_id, tgt_p, pixel_sizes):
        """``patch_size * µm_per_px`` per decoder target, or ``None``.

        Must agree with the encoder's coordinate frame: if the encoder RoPE is
        metric, the decoder absolute PE has to be metric with the same
        ``ref_um``, otherwise mask tokens and visible tokens live in different
        frames.
        """
        if not self._use_metric_pe(cs):
            return None
        pix = self._resolve_pixel_sizes(cs.n_samples, pixel_sizes)
        mpp = torch.tensor(
            [1.0 if v is None else float(v) for v in pix],
            device=tgt_p.device,
            dtype=torch.float32,
        )
        return tgt_p.to(torch.float32) * mpp[tgt_group_id]

    def _use_metric_pe(self, cs) -> bool:
        """Whether positional embeddings use physical (µm) coordinates.

        ``pe_space="physical"`` requires a pixel size; ``"auto"`` enables it
        whenever the tokenizer carries one.  Grid-fraction coordinates encode
        distance *relative to the crop extent*, so they are only metric when
        that extent is constant -- with cell-sized crops (variable ``N``) they
        are not, and an additive scale conditioner cannot recover the metric.
        """
        if self.pe_space == "grid":
            return False
        has_mpp = cs.pixel_size is not None
        if self.pe_space == "physical" and not has_mpp:
            raise ValueError(
                "pe_space='physical' needs pixel_sizes (µm/px); pass them to "
                "forward() or use pe_space='auto'/'grid'"
            )
        return has_mpp

    # ------------------------------------------------------------------ #
    def forward(
        self,
        images: Sequence[torch.Tensor],
        channel_ids: Sequence[torch.Tensor],
        visible: Optional[Sequence[torch.Tensor]] = None,
        generator: Optional[torch.Generator] = None,
        patch_sizes=None,
        pixel_sizes=None,
        valid=None,
    ):
        B = len(images)
        device = images[0].device
        cs, ps, ns, visible, target_valid = self._prepare(
            images, channel_ids, visible, generator, patch_sizes,
            pixel_sizes=pixel_sizes, valid=valid,
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
            # target = visible-mask complement, intersected with the cell mask so
            # off-cell patches are never reconstructed (they are only context)
            m = ~visible[b]
            if target_valid is not None:
                m = m & target_valid[b]
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
            target_step_um=self._target_step_um(
                cs, ps, ns, tgt_group_id, tgt_p, pixel_sizes
            ),
            cond=cond_dec,
        )

        target, pred_mask = self._build_targets(
            images, ps, ns, tgt_group_id, tgt_ch, tgt_patch, tgt_p, pred, gidx
        )
        loss = self._compute_loss(pred, target, tgt_p, pred_mask, tgt_group_id)

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
            "pixel_size": cs.pixel_size,
            "pe_metric": self._use_metric_pe(cs),
        }
        if pred_mask is not None:
            out["pred"] = pred.masked_fill(~pred_mask, 0.0)
        return out

    def _build_targets(
        self, images, ps, ns, tgt_group_id, tgt_ch, tgt_patch, tgt_p, pred, gidx
    ):
        device = pred.device
        # The fast path concatenates per-sample unfold results along the channel
        # axis, which requires a uniform patch count.  Cell-sized crops make N
        # vary, so only take it when both p and N are uniform across samples.
        if len(set(ps)) == 1 and len(set(ns)) == 1:
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

    def _loss_on(self, pred, target, tgt_p, pred_mask, sel=None):
        if sel is not None:
            pred = pred[sel]
            target = target[sel]
            tgt_p = tgt_p[sel]
            pred_mask = None if pred_mask is None else pred_mask[sel]
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

    def _compute_loss(self, pred, target, tgt_p, pred_mask, tgt_group_id=None):
        """``loss_reduce="token"`` weights by token count, ``"sample"`` by cell.

        With cell-sized crops the token count is proportional to cell area, so
        token reduction makes large cells dominate the gradient.  ``"sample"``
        gives every cell one vote, which is what a per-cell representation
        wants.
        """
        if (
            self.loss_reduce == "sample"
            and tgt_group_id is not None
            and tgt_group_id.numel() > 0
        ):
            groups = torch.unique(tgt_group_id)
            total = pred.new_zeros(())
            for g in groups.tolist():
                sel = (tgt_group_id == g).nonzero(as_tuple=True)[0]
                total = total + self._loss_on(pred, target, tgt_p, pred_mask, sel)
            return total / groups.numel()
        return self._loss_on(pred, target, tgt_p, pred_mask)

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def encode(
        self,
        images: Sequence[torch.Tensor],
        channel_ids: Sequence[torch.Tensor],
        pool: str = "attention",
        patch_sizes=None,
        pixel_sizes=None,
        valid=None,
    ):
        """Inference-time representation (no masking).

        Returns ``(memory, pooled_descriptor)`` where ``pooled_descriptor`` is
        ``[B, D]`` -- the fixed-size joint embedding for the whole batch.

        ``valid`` (per-sample ``[N_b]`` cell-token mask) drops off-cell patches
        from the encoder exactly as in :meth:`forward`.
        """
        B = len(images)
        device = images[0].device
        ps = self._resolve_patch_sizes(images, patch_sizes)
        ns = [_n_patches_for(im, p) for im, p in zip(images, ps)]
        visible = [
            torch.ones(im.shape[0], n, dtype=torch.bool, device=im.device)
            for im, n in zip(images, ns)
        ]
        if valid is not None:
            for b, v in enumerate(valid):
                v = v.to(device=device, dtype=torch.bool)
                if v.numel() != ns[b]:
                    raise ValueError(
                        f"valid[{b}] must have {ns[b]} entries, got {v.numel()}"
                    )
                if not self.keep_offcell_as_context:
                    visible[b] = visible[b] & v[None, :]
        cs = self.tokenizer(
            images, channel_ids, visible=visible, patch_sizes=ps,
            pixel_sizes=pixel_sizes,
        )
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
