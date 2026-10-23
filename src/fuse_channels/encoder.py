"""Transformer encoder over fused tokens, with fixed-count memory tokens.

Memory / register tokens are carried through the encoder (and handed to the
decoder) as a cardinality-independent long-term memory.  Attention is again
padding-free: the sequence is flattened over real tokens with per-sample group
ids.

Blocks are standard pre-norm transformer blocks: LayerNorm, explicit
Q/K/V projections, scaled dot-product attention, an output projection, then a
residual MLP.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .ops import grouped_projected_self_attention


class TransformerBlock(nn.Module):
    """Standard pre-norm transformer block over grouped ``[T, D]`` tokens."""

    def __init__(self, dim: int, n_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        assert dim % n_heads == 0, "dim must be divisible by n_heads"
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.n_heads = n_heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, dim)
        )

    def _attention_dropout(self) -> float:
        return self.dropout.p if self.training else 0.0

    def forward(self, x: torch.Tensor, group_id: torch.Tensor, n_groups: int) -> torch.Tensor:
        h = self.norm1(x)
        attn = grouped_projected_self_attention(
            self.q_proj(h),
            self.k_proj(h),
            self.v_proj(h),
            group_id,
            n_groups,
            self.n_heads,
            dropout_p=self._attention_dropout(),
        )
        x = x + self.dropout(self.out_proj(attn))
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x


class TransformerEncoder(nn.Module):
    """Pre-norm transformer over ``[T, D]`` real tokens grouped by sample."""

    def __init__(
        self,
        dim: int,
        depth: int,
        n_heads: int,
        n_memory: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.n_memory = n_memory
        self.memory = nn.Parameter(torch.zeros(1, n_memory, dim))
        nn.init.trunc_normal_(self.memory, std=0.02)
        self.blocks = nn.ModuleList(
            [TransformerBlock(dim, n_heads, mlp_ratio, dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(dim)

    def forward(
        self, tokens: torch.Tensor, sample_idx: torch.Tensor, n_samples: int
    ):
        """Return ``(memory_out, tokens_out)``.

        ``memory_out`` is ``[n_samples * n_memory, D]`` and ``tokens_out`` keeps
        the input ordering ``[T, D]``.
        """
        device = tokens.device
        mem = self.memory.expand(n_samples, -1, -1).reshape(-1, self.memory.shape[-1])
        mem_group_id = torch.arange(n_samples, device=device).repeat_interleave(self.n_memory)

        x = torch.cat([mem, tokens], dim=0)
        group_id = torch.cat([mem_group_id, sample_idx.to(device)], dim=0)

        for blk in self.blocks:
            x = blk(x, group_id, n_samples)
        x = self.norm(x)

        n_mem = n_samples * self.n_memory
        return x[:n_mem], x[n_mem:]
