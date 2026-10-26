"""Channel-aware shared decoder.

One decoder is shared across *all* channels (per-channel decoders do not scale
past a handful of channels).  Each reconstruction query is conditioned on the
**target channel's embedding** and the **target's fractional spatial
position**, then runs grouped self-attention over the target queries of its
sample (so correlated masked channel-patches can coordinate) before
cross-attending to the encoder's memory + visible tokens + trained descriptor.
This turns the additive channel embedding into a *target specification* so one
shared decoder can be used for all channels.

The default decoder positional embedding is the factorized-fractional absolute
PE from ``FLEXIVIT.md``: a function of ``(row, col, grid_h, grid_w)`` with no
grid-sized table, defined for any patch size.  ``"learned"`` and ``"sincos"``
remain as legacy fixed-grid tables/lookup maps.

The reconstruction head is a single shared linear head on a canonical output
grid, bilinearly resized to the target patch size.  FlexiViT does not define
MAE output-head resampling; using a canonical output grid keeps the head shared
instead of learning a separate head per patch size.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

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
        h = self._modulate(
            self.norm_self(q), self.mod_self if self.adaln else None, cond
        )
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

        # then to the encoder representation
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
        hff = self._modulate(
            self.norm_ff(q), self.mod_ff if self.adaln else None, cond
        )
        q = q + self.dropout(self.mlp(hff))
        return q


class PatchReconstructionHead(nn.Module):
    """Shared linear head on a canonical patch grid, resized to the target ``p``.

    The head produces ``canonical_patch_size**2`` pixels.  For a different
    target patch size the predicted patch is bilinearly resized.  This is a
    deliberate choice: FlexiViT specifies input patch-embedding resampling and
    does not define how to resample an MAE output head.
    """

    def __init__(self, dim: int, canonical_patch_size: int) -> None:
        super().__init__()
        self.canonical_patch_size = int(canonical_patch_size)
        self.head = nn.Linear(dim, self.canonical_patch_size**2)

    @property
    def weight(self):
        """Compatibility alias for the underlying linear head weight."""
        return self.head.weight

    @property
    def bias(self):
        return self.head.bias

    def forward(self, x: torch.Tensor, patch_sizes) -> torch.Tensor:
        """Return ``[K, p**2]`` for uniform ``p`` or padded ``[K, max(p**2)]``.

        In the mixed case the caller also needs a validity mask; :meth:`mask`
        provides it.
        """
        pred = self.head(x)
        if isinstance(patch_sizes, int):
            return self._resize(pred, patch_sizes)
        ps = torch.as_tensor(patch_sizes, device=x.device, dtype=torch.long)
        if ps.numel() == 0:
            return pred
        uniq = torch.unique(ps)
        if uniq.numel() == 1:
            return self._resize(pred, int(uniq.item()))
        max_p = int(ps.max().item())
        out = pred.new_zeros(pred.shape[0], max_p * max_p)
        for p in uniq.tolist():
            idx = (ps == p).nonzero(as_tuple=True)[0]
            out[idx, : p * p] = self._resize(pred[idx], p)
        return out

    def _resize(self, pred: torch.Tensor, patch_size: int) -> torch.Tensor:
        p = int(patch_size)
        c = self.canonical_patch_size
        if p == c:
            return pred
        x = pred.reshape(-1, 1, c, c)
        x = F.interpolate(x, size=(p, p), mode="bilinear", align_corners=False)
        return x.reshape(pred.shape[0], p * p)

    @staticmethod
    def mask(patch_sizes, device=None) -> torch.Tensor | None:
        """Return ``[K, max_p**2]`` validity mask, or ``None`` if all p equal."""
        ps = torch.as_tensor(patch_sizes, device=device, dtype=torch.long)
        if ps.numel() == 0 or torch.unique(ps).numel() == 1:
            return None
        max_p = int(ps.max().item())
        idx = torch.arange(max_p * max_p, device=ps.device)
        return idx[None, :] < (ps * ps)[:, None]


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

    def __init__(
        self,
        dim: int,
        depth: int,
        n_heads: int,
        patch_dim: int | None = None,
        n_patches: int | None = None,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        pos_embed: str = "fractional",
        grid_size=None,
        canonical_patch_size: int | None = None,
        fractional_pe_mode: str = "sincos",
        fractional_pe_base: float = 10000.0,
        adaln: bool = False,
    ) -> None:
        super().__init__()
        self.mask_token = nn.Parameter(torch.zeros(dim))
        self.pos_embed = pos_embed
        self.fractional_pe = None
        if pos_embed == "fractional":
            self.fractional_pe = FractionalSpatialPE(
                dim, mode=fractional_pe_mode, base=fractional_pe_base
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
                DecoderBlock(dim, n_heads, mlp_ratio, dropout, adaln=adaln)
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(dim)
        self.adaln = adaln
        if adaln:
            self.mod_out = nn.Linear(dim, 2 * dim)
            nn.init.zeros_(self.mod_out.weight)
            nn.init.zeros_(self.mod_out.bias)
        if canonical_patch_size is not None:
            self.canonical_patch_size = int(canonical_patch_size)
        elif patch_dim is not None:
            side = math.isqrt(int(patch_dim))
            if side * side != int(patch_dim):
                raise ValueError(
                    f"patch_dim must be a square patch size, got {patch_dim}"
                )
            self.canonical_patch_size = side
        else:
            self.canonical_patch_size = None
        if self.canonical_patch_size is None:
            raise ValueError(
                "canonical_patch_size or patch_dim is required for the reconstruction head"
            )
        self.head = PatchReconstructionHead(dim, self.canonical_patch_size)

    def _target_pe(self, target_patch, target_row=None, target_col=None,
                   target_grid_h=None, target_grid_w=None):
        if self.fractional_pe is not None:
            if target_row is None or target_col is None:
                raise ValueError(
                    "fractional decoder PE needs target_row/target_col and grid extents"
                )
            return self.fractional_pe(
                target_row, target_col, target_grid_h, target_grid_w
            )
        if self.pos_embed == "none":
            return self.mask_token.new_zeros(
                target_patch.shape[0], self.mask_token.shape[0]
            )
        return self.spatial_pos[target_patch]

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
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict pixels for the requested (sample, channel, patch) targets.

        ``target_patch_size`` may be a scalar or per-target tensor.  With mixed
        target patch sizes the returned tensor is padded to ``max(p)**2``;
        :meth:`PatchReconstructionHead.mask` gives the validity mask.
        """
        q = (
            self.mask_token
            + self._target_pe(
                target_patch, target_row, target_col, target_grid_h, target_grid_w
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
