"""Channel-aware shared decoder.

One decoder is shared across *all* channels (per-channel decoders do not scale
past a handful of channels).  Each reconstruction query is conditioned on the
**target channel's embedding** and the **target spatial position**, then runs
grouped self-attention over the target queries of its sample (so correlated
masked channel-patches can coordinate) before cross-attending to the encoder's
memory + visible tokens + trained descriptor.  This turns the additive channel
embedding into a *target specification* so one shared decoder can be used for
all channels.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .ops import grouped_attention, grouped_projected_self_attention
from .pos_embed import add_spatial_pos


class DecoderBlock(nn.Module):
    """Pre-norm target self-attention + cross-attention + MLP."""

    def __init__(self, dim: int, n_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        assert dim % n_heads == 0, "dim must be divisible by n_heads"
        self.n_heads = n_heads
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

    def _attention_dropout(self) -> float:
        return self.dropout.p if self.training else 0.0

    def forward(
        self,
        q: torch.Tensor,
        q_group_id: torch.Tensor,
        kv: torch.Tensor,
        kv_group_id: torch.Tensor,
        n_groups: int,
    ) -> torch.Tensor:
        # target queries attend to each other (one group per sample)
        h = self.norm_self(q)
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
        hq = self.norm_q(q)
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
        q = q + self.dropout(self.mlp(self.norm_ff(q)))
        return q


class ChannelAwareDecoder(nn.Module):
    """Channel-aware shared decoder.

    Each query is ``mask_token + spatial_pos[target_patch] + target_proj(
    channel_embed)``.  The absolute spatial PE on the target queries is the
    repo's equivalent of MAE's ``decoder_pos_embed``: without it, every masked
    patch of a channel would produce the same query and hence the same pixels
    (``references/mae/paper.md:103``).

    ``pos_embed`` selects the learned ``[n_patches, dim]`` table (default) or the
    fixed 2D sine-cosine map registered as a buffer; see
    :mod:`fuse_channels.pos_embed`.
    """

    def __init__(
        self,
        dim: int,
        depth: int,
        n_heads: int,
        patch_dim: int,
        n_patches: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        pos_embed: str = "learned",
        grid_size=None,
    ) -> None:
        super().__init__()
        self.mask_token = nn.Parameter(torch.zeros(dim))
        # target spatial PE: learned table by default, fixed 2D sincos optional
        add_spatial_pos(self, dim, n_patches, pos_embed, grid_size=grid_size)
        self.target_proj = nn.Linear(dim, dim)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.blocks = nn.ModuleList(
            [DecoderBlock(dim, n_heads, mlp_ratio, dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, patch_dim)

    def forward(
        self,
        enc_tokens: torch.Tensor,
        enc_group_id: torch.Tensor,
        n_samples: int,
        target_group_id: torch.Tensor,
        target_patch: torch.Tensor,
        target_channel_embed: torch.Tensor,
    ) -> torch.Tensor:
        """Predict raw pixels for the requested (sample, channel, patch) targets.

        Parameters
        ----------
        enc_tokens, enc_group_id:
            encoder output ``[Ke, D]`` and its per-sample group ids.
        target_group_id, target_patch, target_channel_embed:
            ``[Kt]`` sample ids, ``[Kt]`` patch positions and ``[Kt, D]``
            additive channel-embedding vectors of the target channels.
        """
        q = (
            self.mask_token
            + self.spatial_pos[target_patch]
            + self.target_proj(target_channel_embed)
        )
        for blk in self.blocks:
            q = blk(q, target_group_id, enc_tokens, enc_group_id, n_samples)
        return self.head(self.norm(q))
