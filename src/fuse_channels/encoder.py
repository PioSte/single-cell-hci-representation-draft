"""Transformer encoder over fused tokens, with fixed-count memory tokens.

Memory / register tokens are carried through the encoder (and handed to the
decoder) as a cardinality-independent long-term memory.  Attention is again
padding-free: the sequence is flattened over real tokens with per-sample group
ids.

The mechanism follows ChA-MAEViT §3.2
(``references/cha-maevit/paper.md:109-113``): ``n_memory`` learnable embeddings
are prepended to the input sequence, processed by the encoder, and excluded
from the final representation (the descriptor is pooled over patch tokens
only).  ChA-MAEViT, in turn, credits the idea to register tokens (Darcet et
al., *Vision Transformers Need Registers*); that primary paper is not in
``references/``, so the citation here is the second-hand source.

Divergences from ChA-MAEViT, which runs a dense ``C·N`` token layout with
learned absolute positional embeddings:

* Padding-free.  Memory tokens join each sample's *present-only* group
  (``group_id``) instead of a dense ``C·N`` sequence.
* Zero RoPE on memory queries/keys (``identity_rope_cache``).  Memory carries
  no location, so it is not rotated; ChA-MAEViT has no RoPE to begin with.
* Zero scale condition.  Memory is scale-invariant; under ``adaln`` it gets
  zeros as its per-token condition, which is identity because the mod weights
  initialise to zero.
* Returned as a separate tensor.  The encoder returns ``memory_out``
  (``[B·n_memory, D]``) alongside the patch tokens rather than leaving the
  memory in the decoder sequence; the caller (``ChannelFusionMAE``) assembles
  the decoder KV from it.

Blocks are standard pre-norm transformer blocks: LayerNorm, explicit
Q/K/V projections, scaled dot-product attention, an output projection, then a
residual MLP.  Spatial position enters only through optional 2-D axial RoPE
applied to queries and keys; memory tokens get zero rotation.  Optional AdaLN
scale conditioning modulates the LayerNorms with a per-token condition vector.

This module returns the per-token sequence, so its length depends on how many
locations are present.  ``ChannelFusionMAE`` pools that sequence to the fixed
``[B, n_latents, D]`` latent set (spatial pooling to fixed size latent)
before anything downstream sees it, so the
model-level encoder output and the decoder KV do not depend on the patch count;
the cost of this encoder still does.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .ops import grouped_projected_self_attention
from .rope import apply_rope, identity_rope_cache


class TransformerBlock(nn.Module):
    """Standard pre-norm transformer block over grouped ``[T, D]`` tokens."""

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
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.n_heads = n_heads
        self.adaln = adaln
        if adaln:
            self.mod1 = nn.Linear(dim, 2 * dim)
            self.mod2 = nn.Linear(dim, 2 * dim)
            nn.init.zeros_(self.mod1.weight)
            nn.init.zeros_(self.mod1.bias)
            nn.init.zeros_(self.mod2.weight)
            nn.init.zeros_(self.mod2.bias)
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

    def _modulate(self, h: torch.Tensor, mod: nn.Linear, cond: torch.Tensor | None):
        if mod is None or cond is None:
            return h
        gamma, beta = mod(cond).chunk(2, dim=-1)
        return h * (1.0 + gamma) + beta

    def forward(
        self,
        x: torch.Tensor,
        group_id: torch.Tensor,
        n_groups: int,
        rope_cos: torch.Tensor | None = None,
        rope_sin: torch.Tensor | None = None,
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self._modulate(
            self.norm1(x), self.mod1 if self.adaln else None, cond
        )
        q = self.q_proj(h)
        k = self.k_proj(h)
        v = self.v_proj(h)
        if rope_cos is not None:
            T = q.shape[0]
            Dh = q.shape[-1] // self.n_heads
            q = apply_rope(
                q.reshape(T, self.n_heads, Dh), rope_cos, rope_sin
            ).reshape(T, -1)
            k = apply_rope(
                k.reshape(T, self.n_heads, Dh), rope_cos, rope_sin
            ).reshape(T, -1)
        attn = grouped_projected_self_attention(
            q,
            k,
            v,
            group_id,
            n_groups,
            self.n_heads,
            dropout_p=self._attention_dropout(),
        )
        x = x + self.dropout(self.out_proj(attn))
        h2 = self._modulate(
            self.norm2(x), self.mod2 if self.adaln else None, cond
        )
        x = x + self.dropout(self.mlp(h2))
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
        adaln: bool = False,
    ) -> None:
        super().__init__()
        self.n_memory = n_memory
        self.dim = dim
        self.n_heads = n_heads
        self.adaln = adaln
        self.memory = nn.Parameter(torch.zeros(1, n_memory, dim))
        nn.init.trunc_normal_(self.memory, std=0.02)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(dim, n_heads, mlp_ratio, dropout, adaln=adaln)
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(dim)
        if adaln:
            self.mod_out = nn.Linear(dim, 2 * dim)
            nn.init.zeros_(self.mod_out.weight)
            nn.init.zeros_(self.mod_out.bias)

    def forward(
        self,
        tokens: torch.Tensor,
        sample_idx: torch.Tensor,
        n_samples: int,
        rope=None,
        cond: torch.Tensor | None = None,
    ):
        """Return ``(memory_out, tokens_out)``.

        ``memory_out`` is ``[n_samples * n_memory, D]`` and ``tokens_out`` keeps
        the input ordering ``[T, D]``.  The model-level spatial pooling to fixed size
        latent is applied outside this module (``ChannelFusionMAE._encode``), so
        callers that need a patch-count-independent representation should use
        that pooled latent set rather than this raw token sequence.
        """
        device = tokens.device
        mem = self.memory.expand(n_samples, -1, -1).reshape(-1, self.memory.shape[-1])
        mem_group_id = torch.arange(n_samples, device=device).repeat_interleave(self.n_memory)

        x = torch.cat([mem, tokens], dim=0)
        group_id = torch.cat([mem_group_id, sample_idx.to(device)], dim=0)

        n_mem = n_samples * self.n_memory
        rope_cos = rope_sin = None
        if rope is not None:
            cos, sin = rope
            mem_cos, mem_sin = identity_rope_cache(
                n_mem, 2 * cos.shape[-1], device=device, dtype=cos.dtype
            )
            rope_cos = torch.cat([mem_cos, cos.to(device)], dim=0)
            rope_sin = torch.cat([mem_sin, sin.to(device)], dim=0)

        full_cond = None
        if cond is not None:
            # Memory tokens have no per-sample condition supplied; zero is an
            # identity modulation because mod weights start at zero.
            full_cond = torch.cat([cond.new_zeros(n_mem, cond.shape[-1]), cond], dim=0)

        for blk in self.blocks:
            x = blk(
                x,
                group_id,
                n_samples,
                rope_cos=rope_cos,
                rope_sin=rope_sin,
                cond=full_cond,
            )
        if self.adaln and full_cond is not None:
            gamma, beta = self.mod_out(full_cond).chunk(2, dim=-1)
            x = self.norm(x) * (1.0 + gamma) + beta
        else:
            x = self.norm(x)

        return x[:n_mem], x[n_mem:]
