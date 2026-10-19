"""Channel-aware shared decoder.

One decoder is shared across *all* channels (per-channel decoders do not scale
past a handful of channels).  Each reconstruction query is conditioned on the
**target channel's identity embedding** and the **target spatial position**, and
cross-attends to the encoder's memory + visible tokens.  This turns the additive
identity embedding into a *target specification* and gives open-vocabulary
behaviour for free.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .ops import grouped_cross_attention


class DecoderBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.n_heads = n_heads
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.norm_ff = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, dim)
        )
        self.attn_out = nn.Linear(dim, dim)

    def forward(
        self,
        q: torch.Tensor,
        q_seg: torch.Tensor,
        kv: torch.Tensor,
        kv_seg: torch.Tensor,
        n_segments: int,
    ) -> torch.Tensor:
        h = grouped_cross_attention(
            self.norm_q(q), q_seg, self.norm_kv(kv), kv_seg, n_segments, self.n_heads
        )
        q = q + self.dropout(self.attn_out(h))
        q = q + self.dropout(self.mlp(self.norm_ff(q)))
        return q


class ChannelAwareDecoder(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        n_heads: int,
        patch_dim: int,
        n_patches: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.mask_token = nn.Parameter(torch.zeros(dim))
        self.spatial_pos = nn.Parameter(torch.zeros(n_patches, dim))
        self.target_proj = nn.Linear(dim, dim)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.spatial_pos, std=0.02)
        self.blocks = nn.ModuleList(
            [DecoderBlock(dim, n_heads, mlp_ratio, dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, patch_dim)

    def forward(
        self,
        enc_tokens: torch.Tensor,
        enc_seg: torch.Tensor,
        n_samples: int,
        target_seg: torch.Tensor,
        target_patch: torch.Tensor,
        target_identity: torch.Tensor,
    ) -> torch.Tensor:
        """Predict raw pixels for the requested (sample, channel, patch) targets.

        Parameters
        ----------
        enc_tokens, enc_seg:
            encoder output ``[Ke, D]`` and its per-sample segment ids.
        target_seg, target_patch, target_identity:
            ``[Kt]`` sample ids, ``[Kt]`` patch positions and ``[Kt, D]``
            additive identity vectors of the target channels.
        """
        q = (
            self.mask_token
            + self.spatial_pos[target_patch]
            + self.target_proj(target_identity)
        )
        for blk in self.blocks:
            q = blk(q, target_seg, enc_tokens, enc_seg, n_samples)
        return self.head(self.norm(q))
