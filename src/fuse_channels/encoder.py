"""Spatial transformer over fused tokens, with fixed-count memory tokens.

Memory / register tokens are carried through the encoder (and handed to the
decoder) as a cardinality-independent long-term memory.  Attention is again
padding-free: the sequence is flattened over real tokens with per-sample
segment ids.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .ops import grouped_self_attention


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, dim)
        )

    def forward(self, x: torch.Tensor, seg: torch.Tensor, n_segments: int) -> torch.Tensor:
        h = grouped_self_attention(
            self.norm1(x), seg, n_segments, self.n_heads, dropout_p=0.0
        )
        x = x + self.dropout(h)
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x


class SpatialTransformer(nn.Module):
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
        mem_seg = torch.arange(n_samples, device=device).repeat_interleave(self.n_memory)

        x = torch.cat([mem, tokens], dim=0)
        seg = torch.cat([mem_seg, sample_idx.to(device)], dim=0)

        for blk in self.blocks:
            x = blk(x, seg, n_samples)
        x = self.norm(x)

        n_mem = n_samples * self.n_memory
        return x[:n_mem], x[n_mem:]
