"""Channel-aware shared decoder.

One decoder is shared across *all* channels (per-channel decoders do not scale
past a handful of channels).  Each reconstruction query is conditioned on the
**target channel's embedding** and the **target's fractional spatial
position**, then runs grouped self-attention over the target queries of its
sample (so correlated masked channel-patches can coordinate) before
cross-attending to the encoder's memory + latents.  Those KV components are
selectable per model via ``decoder_kv_components`` (see
``ChannelFusionMAE``); the default uses both, and an empty KV skips
cross-attention entirely.  This turns the additive channel embedding into a
*target specification* so one shared decoder can be used for all channels.

The default decoder positional embedding is the factorized-fractional absolute
PE from ``FLEXIVIT.md``: a function of ``(row, col, grid_h, grid_w)`` with no
grid-sized table, defined for any patch size.  By default it is **metric**: the
caller passes ``target_step_um = p * pixel_size``, so the PE uses the same
realized physical frame as the encoder RoPE (and carries the rounding residual
in ``p * pixel_size``).  ``"learned"`` and ``"sincos"`` remain as legacy
fixed-grid tables/lookup maps.

The reconstruction head has three modes:

* ``head_mode="per_p"`` (default) -- one linear ``dim -> p*p`` head per
  configured patch size.  Exact for configured ``p``; an unseen ``p`` falls
  back to the nearest configured head with ``p_head >= p`` (bilinear
  point-sampling) and refuses to upsample.
* ``head_mode="shared"`` -- **one** ``dim -> p0*p0`` head at a canonical
  ``p0`` whose parameters do not depend on ``p``, resampled to any target
  ``p`` (both directions) by a fixed linear operator.  This is the FlexiViT
  pattern applied to the head instead of the patch kernel: canonical
  parameters + a resolution operator, so a checkpoint is fleet-independent
  and every ``p`` is legal.  ``head_resample`` selects the operator (see
  :class:`PatchReconstructionHead`).
* ``head_mode="canonical"`` -- the legacy single head with bilinear
  point-sampling and no upsampling (kept as an ablation).

The decoder is pretraining-only and discarded after MAE, so head parameters do
not affect deployment.
"""

from __future__ import annotations

import math
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from .losses import patch_offsets
from .ops import grouped_attention, grouped_projected_self_attention
from .pos_embed import FractionalSpatialPE, add_spatial_pos


class DecoderBlock(nn.Module):
    """Pre-norm target self-attention + cross-attention + MLP."""

    def __init__(
        self,
        dim: int,
        n_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        adaln: bool = False,
    ):
        super().__init__()
        assert dim % n_heads == 0, "dim must be divisible by n_heads"
        self.n_heads = n_heads
        self.adaln = adaln
        # target self-attention
        self.norm_self = nn.LayerNorm(dim)
        self.self_q_proj = nn.Linear(dim, dim)
        self.self_k_proj = nn.Linear(dim, dim)
        self.self_v_proj = nn.Linear(dim, dim)
        self.self_out_proj = nn.Linear(dim, dim)
        # cross-attention to the encoder representation
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        # feed-forward
        self.norm_ff = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, dim)
        )
        if adaln:
            self.mod_self = nn.Linear(dim, 2 * dim)
            self.mod_q = nn.Linear(dim, 2 * dim)
            self.mod_ff = nn.Linear(dim, 2 * dim)
            for mod in (self.mod_self, self.mod_q, self.mod_ff):
                nn.init.zeros_(mod.weight)
                nn.init.zeros_(mod.bias)

    def _attention_dropout(self) -> float:
        return self.dropout.p if self.training else 0.0

    def _modulate(self, h, mod, cond):
        if mod is None or cond is None:
            return h
        gamma, beta = mod(cond).chunk(2, dim=-1)
        return h * (1.0 + gamma) + beta

    def forward(
        self,
        q: torch.Tensor,
        q_group_id: torch.Tensor,
        kv: torch.Tensor,
        kv_group_id: torch.Tensor,
        n_groups: int,
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # target queries attend to each other (one group per sample)
        h = self._modulate(self.norm_self(q), self.mod_self if self.adaln else None, cond)
        self_attn = grouped_projected_self_attention(
            self.self_q_proj(h),
            self.self_k_proj(h),
            self.self_v_proj(h),
            q_group_id,
            n_groups,
            self.n_heads,
            dropout_p=self._attention_dropout(),
        )
        q = q + self.dropout(self.self_out_proj(self_attn))

        # then to the encoder representation.  An empty KV (``decoder_kv_components=()``,
        # or only a ``n_memory=0`` memory that contributed nothing) means there
        # is no cross-attention source, so this is skipped: a blind decoder
        # that reconstructs from the mask/position/channel query alone.
        if kv.shape[0] > 0:
            hq = self._modulate(self.norm_q(q), self.mod_q if self.adaln else None, cond)
            hkv = self.norm_kv(kv)
            cross = grouped_attention(
                self.q_proj(hq),
                q_group_id,
                self.k_proj(hkv),
                self.v_proj(hkv),
                kv_group_id,
                n_groups,
                self.n_heads,
                dropout_p=self._attention_dropout(),
            )
            q = q + self.dropout(self.out_proj(cross))
        hff = self._modulate(self.norm_ff(q), self.mod_ff if self.adaln else None, cond)
        q = q + self.dropout(self.mlp(hff))
        return q


def canonical_patch_size_for(patch_sizes) -> int:
    """Suggested ``p0`` for the ``p``-free head: the log-space median.

    ``head_mode="shared"`` resamples in both directions, so the canonical size
    should minimise the worst-case resampling ratio over the fleet rather than
    sit at an edge (``max`` would upsample for every instrument and low-pass
    nothing).  Minimising ``max_p |log(p0 / p)|`` gives the geometric midpoint;
    ties go to the smaller ``p``.  For ``(4, 8, 16)`` this is 8, for
    ``(18, 20, 21)`` it is 20.

    This is a *suggestion*: ``p0`` is a fixed hyperparameter, so pass it
    explicitly as ``head_patch_size`` -- letting it default from
    ``patch_sizes`` would silently change the head's shape when the fleet
    changes.
    """
    ps = sorted({int(p) for p in patch_sizes})
    if not ps:
        raise ValueError("patch_sizes must not be empty")
    if any(p < 1 for p in ps):
        raise ValueError(f"patch sizes must be positive, got {ps}")
    # log ratios are antisymmetric but not bit-exact, so round the key before
    # using the smaller ``p`` as the tie-break.
    return min(
        ps,
        key=lambda q: (round(max(abs(math.log(q / p)) for p in ps), 12), q),
    )


class PatchReconstructionHead(nn.Module):
    """Reconstruction head: per-patch-size heads, or one ``p``-free head.

    ``head_mode="per_p"`` (default) keeps one ``nn.Linear(dim, p*p)`` per
    configured patch size: exact for configured ``p``, and the decoder is
    pretraining-only.  ``head_mode="canonical"`` is the legacy single head with
    bilinear point-sampling that refuses to upsample (ablation).

    ``head_mode="shared"`` is the **parameter-free-in-``p``** head: one
    ``nn.Linear(dim, p0*p0)`` whose output rows are indexed by the canonical
    ``p0 x p0`` patch grid, resampled to the target grid by a fixed linear
    operator.  Since resampling commutes with the linear head, resampling the
    prediction and resampling the head's rows are the same map, so the operator
    can be applied to the ``[K, p**2]`` prediction directly.  ``p0`` is
    ``canonical_patch_size`` (typically the middle of the fleet's ``p`` range).
    Every ``p >= 1`` is legal, in both directions, and the checkpoint carries no
    per-``p`` parameters.

    ``head_resample`` picks that operator (``head_mode="shared"`` only):

    * ``"area"`` (default) -- box-average when shrinking, bilinear when
      growing.  The physically consistent choice under this repo's contract:
      patches hold a constant *physical* size, so a smaller ``p`` is a
      genuinely coarser sampling of the same field and the cross-``p`` relation
      is a low-pass, not a point-sample.
    * ``"bilinear"`` -- point-sampling both ways
      (``align_corners=False``, no antialias): the operator the per-``p``
      fallback uses, kept as the ablation arm.
    * ``"pi"`` -- FlexiViT's PI-resize (``P = (B^T)^+``, ``flexi.py``) applied
      along the output-pixel axis, i.e. the head's rows are treated as a
      ``p0 x p0`` kernel and PI-resized.  Exactly the tokenizer's operator; it
      is the minimum-norm least-squares inverse rather than a low-pass, so it
      keeps detail (and can ring) where ``"area"`` blurs.

    For an unseen ``p`` with ``head_mode="per_p"`` the nearest configured head
    with ``p_head >= p`` is used as the fallback (the canonical head when no
    larger per-p head exists).  That fallback never upsamples: ``p >
    max(patch_sizes)`` raises; use ``"shared"`` to lift that limit.
    """

    RESAMPLE_OPS = ("area", "bilinear", "pi")

    # One of the two is populated by ``__init__`` depending on ``mode``.
    head: nn.Linear | None
    heads: nn.ModuleDict | None

    def __init__(
        self,
        dim: int,
        canonical_patch_size: int,
        patch_sizes=None,
        mode: str = "per_p",
        resample: str = "area",
    ) -> None:
        super().__init__()
        if mode not in ("per_p", "canonical", "shared"):
            raise ValueError(
                f"unknown reconstruction head mode '{mode}'; "
                "expected 'per_p', 'canonical' or 'shared'"
            )
        if resample not in self.RESAMPLE_OPS:
            raise ValueError(
                f"unknown head_resample '{resample}'; expected one of {self.RESAMPLE_OPS}"
            )
        self.canonical_patch_size = int(canonical_patch_size)
        self.mode = mode
        self.resample = resample
        if patch_sizes is None:
            patch_sizes = (self.canonical_patch_size,)
        self.patch_sizes = tuple(
            sorted({int(p) for p in patch_sizes} | {self.canonical_patch_size})
        )
        for p in self.patch_sizes:
            if p < 1:
                raise ValueError(f"patch sizes must be positive, got {p}")
        if mode in ("canonical", "shared"):
            self.heads = None
            self.head = nn.Linear(dim, self.canonical_patch_size**2)
        else:
            self.heads = nn.ModuleDict({str(p): nn.Linear(dim, p * p) for p in self.patch_sizes})
            self.head = None

    @property
    def weight(self):
        """Compatibility alias for the canonical linear head weight."""
        return self._canonical_head().weight

    @property
    def bias(self):
        return self._canonical_head().bias

    def _canonical_head(self) -> nn.Linear:
        if self.heads is None:
            head = self.head
            assert head is not None, "reconstruction head is not initialised"
            return head
        return cast(nn.Linear, self.heads[str(self.canonical_patch_size)])

    def _head_for(self, patch_size: int) -> tuple[nn.Linear, bool, int]:
        """Return ``(head, needs_resize, source_p)`` for one target size."""
        p = int(patch_size)
        head = self.head
        if self.mode == "shared":
            # p-free: one head, resampled in both directions.
            assert head is not None, "reconstruction head is not initialised"
            return head, p != self.canonical_patch_size, self.canonical_patch_size
        heads = self.heads
        if heads is None:
            assert head is not None, "reconstruction head is not initialised"
            if p > self.canonical_patch_size:
                raise ValueError(
                    "canonical reconstruction head cannot upsample: "
                    f"target p={p} > canonical_patch_size={self.canonical_patch_size}"
                )
            return head, p != self.canonical_patch_size, self.canonical_patch_size
        if str(p) in heads:
            return cast(nn.Linear, heads[str(p)]), False, p
        larger = [hp for hp in self.patch_sizes if hp >= p]
        if not larger:
            raise ValueError(
                "per-p reconstruction head has no head for target "
                f"p={p} and cannot upsample; configured p={self.patch_sizes}"
            )
        source = min(larger)
        return cast(nn.Linear, heads[str(source)]), True, source

    def forward(self, x: torch.Tensor, patch_sizes) -> torch.Tensor:
        """Return ``[Σ p**2]`` pixels: row ``i`` is contiguous, ``p_i**2`` long.

        The layout is the one :func:`~fuse_channels.losses.patch_offsets`
        describes, so a mixed-``p`` batch needs no padded ``[K, max(p)**2]``
        block and no validity mask: each patch-size group is predicted in one
        call and scattered into its rows' slots.
        """
        if patch_sizes is None:
            patch_sizes = self.canonical_patch_size
        if isinstance(patch_sizes, int):
            head, resize, source = self._head_for(patch_sizes)
            pred = head(x)
            pred = self._resize(pred, patch_sizes, source) if resize else pred
            return pred.reshape(-1)
        ps = torch.as_tensor(patch_sizes, device=x.device, dtype=torch.long)
        if ps.numel() == 0:
            head, _, _ = self._head_for(self.canonical_patch_size)
            return head(x).reshape(-1)
        uniq = torch.unique(ps)
        if uniq.numel() == 1:
            return self(x, int(uniq.item()))
        offsets = patch_offsets(ps)
        out = x.new_zeros(int(offsets[-1]))
        for p in uniq.tolist():
            idx = (ps == p).nonzero(as_tuple=True)[0]
            head, resize, source = self._head_for(p)
            pred = head(x[idx])
            pred = self._resize(pred, p, source) if resize else pred
            slots = offsets[idx].unsqueeze(1) + torch.arange(p * p, device=x.device)
            out[slots.reshape(-1)] = pred.reshape(-1)
        return out

    @staticmethod
    def pixel_offsets(patch_sizes) -> torch.Tensor:
        """Row boundaries of the flat pixel layout (see :func:`patch_offsets`)."""
        return patch_offsets(patch_sizes)

    def _resize(self, pred: torch.Tensor, patch_size: int, source_patch_size: int) -> torch.Tensor:
        p = int(patch_size)
        c = int(source_patch_size)
        if p == c:
            return pred
        if self.mode == "shared":
            return self._resample_shared(pred, c, p)
        if p > c:
            raise ValueError(f"cannot resize p={c} prediction up to p={p}")
        x = pred.reshape(-1, 1, c, c)
        x = F.interpolate(x, size=(p, p), mode="bilinear", align_corners=False)
        return x.reshape(pred.shape[0], p * p)

    def _resample_shared(self, pred: torch.Tensor, source_p: int, target_p: int) -> torch.Tensor:
        """Resample a ``[K, source_p**2]`` patch prediction to ``target_p``.

        Resampling commutes with the linear head, so applying the operator to
        the prediction is identical to resampling the head's output rows -- and
        much cheaper than rebuilding the head per ``p``.
        """
        if self.resample == "pi":
            from .flexi import _pi_resize_matrix

            # ``_pi_resize_matrix`` is [target_p**2, source_p**2], matching
            # ``pi_resize_patch_embed``'s ``flat @ P.T`` convention.
            op = _pi_resize_matrix(source_p, target_p, pred.dtype).to(pred.device)
            return pred @ op.T
        x = pred.reshape(-1, 1, source_p, source_p)
        if self.resample == "area" and target_p < source_p:
            x = F.interpolate(x, size=(target_p, target_p), mode="area")
        else:
            x = F.interpolate(x, size=(target_p, target_p), mode="bilinear", align_corners=False)
        return x.reshape(pred.shape[0], target_p * target_p)


class ChannelAwareDecoder(nn.Module):
    """Channel-aware shared decoder.

    Each query is ``mask_token + spatial_pe(target) + target_proj(
    channel_embed)``.  The absolute target PE is load-bearing: without it,
    every masked patch of a channel would produce the same query and hence the
    same pixels (``references/mae/paper.md:103``).

    ``pos_embed`` selects ``"fractional"`` (default; a function of fractional
    grid coordinates), or the legacy ``"learned"`` / ``"sincos"`` fixed-grid
    lookup.
    """

    # Registered only by the table-based (``learned``/``sincos``) variants.
    spatial_pos: torch.Tensor

    def __init__(
        self,
        core,
        cfg,
        *,
        n_patches: int | None = None,
        canonical_patch_size: int | None = None,
        patch_sizes=None,
        grid_size=None,
    ) -> None:
        # Shared backbone invariants (core) vs decoder-exclusive knobs (cfg) vs
        # builder-resolved extras (n_patches, canonical_patch_size, patch_sizes).
        # See ``fuse_channels.config`` for why the split is drawn here.
        super().__init__()
        dim, n_heads = core.dim, core.n_heads
        dropout, adaln = core.dropout, core.adaln
        pos_embed = cfg.pos_embed
        if pos_embed is None:
            # ``None`` means "inherit from the parent ModelConfig" (see
            # ``config.ModelConfig.resolve``); a standalone decoder has no
            # parent, so it uses the shipped decoder PE.
            pos_embed = "fractional"
        self.mask_token = nn.Parameter(torch.zeros(dim))
        self.pos_embed = pos_embed
        self.pe_ref_um = float(cfg.pe_ref_um)
        self.fractional_pe = None
        if pos_embed == "fractional":
            self.fractional_pe = FractionalSpatialPE(
                dim, mode=cfg.fractional_pe_mode, base=cfg.fractional_pe_base
            )
        elif pos_embed == "none":
            self.fractional_pe = None
        elif pos_embed in ("learned", "sincos"):
            if n_patches is None:
                raise ValueError("n_patches is required for a table-based decoder PE")
            add_spatial_pos(self, dim, n_patches, pos_embed, grid_size=grid_size)
        else:
            raise ValueError(f"unknown decoder pos_embed '{pos_embed}'")
        self.target_proj = nn.Linear(dim, dim)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.blocks = nn.ModuleList(
            [
                DecoderBlock(dim, n_heads, cfg.mlp_ratio, dropout, adaln=adaln)
                for _ in range(cfg.depth)
            ]
        )
        self.norm = nn.LayerNorm(dim)
        self.adaln = adaln
        if adaln:
            self.mod_out = nn.Linear(dim, 2 * dim)
            nn.init.zeros_(self.mod_out.weight)
            nn.init.zeros_(self.mod_out.bias)
        if canonical_patch_size is None:
            raise ValueError("canonical_patch_size is required for the reconstruction head")
        self.canonical_patch_size = int(canonical_patch_size)
        self.head_mode = cfg.head_mode
        self.head = PatchReconstructionHead(
            dim,
            self.canonical_patch_size,
            patch_sizes=patch_sizes,
            mode=cfg.head_mode,
            resample=cfg.head_resample,
        )

    def _target_pe(
        self,
        target_patch,
        target_row=None,
        target_col=None,
        target_grid_h=None,
        target_grid_w=None,
        target_step_um=None,
    ):
        if self.fractional_pe is not None:
            if target_row is None or target_col is None:
                raise ValueError(
                    "fractional decoder PE needs target_row/target_col and grid extents"
                )
            return self.fractional_pe(
                target_row,
                target_col,
                target_grid_h,
                target_grid_w,
                step_um=target_step_um,
                ref_um=self.pe_ref_um,
            )
        if self.pos_embed == "none":
            return self.mask_token.new_zeros(target_patch.shape[0], self.mask_token.shape[0])
        return self.spatial_pos[target_patch]

    def predict_features(
        self,
        enc_tokens: torch.Tensor,
        enc_group_id: torch.Tensor,
        n_samples: int,
        target_group_id: torch.Tensor,
        target_patch: torch.Tensor,
        *,
        target_row=None,
        target_col=None,
        target_grid_h=None,
        target_grid_w=None,
        target_step_um=None,
        target_condition: torch.Tensor | None = None,
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict a feature vector per target location (JEPA predictor).

        This is the feature-space counterpart of :meth:`forward`: the query is
        ``mask_token + target PE + target_condition``, with no per-channel
        identity term because the JEPA target is the channel-fused location
        representation.  The final LayerNorm makes the prediction comparable to
        the target encoder's final token output; no reconstruction head is
        applied.
        """
        q = self.mask_token + self._target_pe(
            target_patch,
            target_row,
            target_col,
            target_grid_h,
            target_grid_w,
            target_step_um=target_step_um,
        )
        if target_condition is not None:
            q = q + target_condition
        for blk in self.blocks:
            q = blk(q, target_group_id, enc_tokens, enc_group_id, n_samples, cond=cond)
        h = self.norm(q)
        if self.adaln and cond is not None:
            gamma, beta = self.mod_out(cond).chunk(2, dim=-1)
            h = h * (1.0 + gamma) + beta
        return h

    def forward(
        self,
        enc_tokens: torch.Tensor,
        enc_group_id: torch.Tensor,
        n_samples: int,
        target_group_id: torch.Tensor,
        target_patch: torch.Tensor,
        target_channel_embed: torch.Tensor,
        *,
        target_row=None,
        target_col=None,
        target_grid_h=None,
        target_grid_w=None,
        target_patch_size=None,
        target_step_um=None,
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict pixels for the requested (sample, channel, patch) targets.

        ``target_patch_size`` may be a scalar or per-target tensor.  The return
        is the flat ``[Σ p**2]`` layout of
        :func:`~fuse_channels.losses.patch_offsets` for either form.
        """
        q = (
            self.mask_token
            + self._target_pe(
                target_patch,
                target_row,
                target_col,
                target_grid_h,
                target_grid_w,
                target_step_um=target_step_um,
            )
            + self.target_proj(target_channel_embed)
        )
        for blk in self.blocks:
            q = blk(q, target_group_id, enc_tokens, enc_group_id, n_samples, cond=cond)
        h = self.norm(q)
        if self.adaln and cond is not None:
            gamma, beta = self.mod_out(cond).chunk(2, dim=-1)
            h = h * (1.0 + gamma) + beta
        if target_patch_size is None:
            target_patch_size = self.canonical_patch_size
        return self.head(h, target_patch_size)
