"""Channel-fusion modules: map an arbitrary channel set to a fixed-size joint
embedding, with **no padding tokens**.

Two interchangeable pooling mechanisms are provided:

* :class:`ChannelFusion` -- PMA-style attention pooling over the channel axis
  (the default; ChannelViT / ChAda-ViT / Panopticon pattern).
* :class:`HypernetworkWeightedSum` -- ``O(C)`` hypernetwork-weighted sum, the
  cheap alternative flagged in the handoff for 20-60 channel panels.

Both are set-native: they consume a :class:`~fuse_channels.container.ChannelSet`
built from real tokens only and never create a ``[B, C_max, D]`` tensor.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .container import ChannelSet
from .ops import grouped_pool_attention, segment_softmax


def _unique_segments(seg: torch.Tensor):
    """Return ``(unique_sorted, counts, lookup)`` for a per-token segment id."""
    uniq, counts = torch.unique(seg, return_counts=True)
    order = torch.argsort(seg, stable=True)
    lookup = torch.searchsorted(uniq, seg[order])
    return uniq, counts, order, lookup


class ChannelFusion(nn.Module):
    """Attention pooling of the channel axis at every spatial location.

    Channels are fused *within* each ``(sample, patch)`` location into a fixed
    number of learned queries and the result is scattered back to
    ``[B, N, n_queries*dim]``.  The queries are shared across locations (the
    canonical PMA form): position enters through the shared spatial positional
    embedding on the tokens, not through per-location query parameters.

    Parameters
    ----------
    dim:
        token feature dimension.
    n_heads:
        attention heads (``dim`` must be divisible by it).
    n_queries:
        number of PMA queries per location (usually 1).  Output feature size is
        ``n_queries * dim``.
    n_patches:
        spatial patch count.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int = 8,
        n_queries: int = 1,
        n_patches: int = 0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert dim % n_heads == 0, "dim must be divisible by n_heads"
        assert n_patches > 0, "n_patches is required"
        self.dim = dim
        self.n_heads = n_heads
        self.n_queries = n_queries
        self.n_patches = n_patches
        self.dropout = dropout

        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.proj_out = nn.Linear(dim, dim)

        # one shared query set, applied at every spatial location
        self.queries = nn.Parameter(torch.zeros(1, n_queries, dim))
        nn.init.trunc_normal_(self.queries, std=0.02)

    # ------------------------------------------------------------------ #
    def forward(self, cs: ChannelSet):
        """Return ``(fused, present)``.

        ``fused`` is ``[B, N, n_queries*dim]`` and ``present`` is a boolean mask
        marking locations that had at least one channel.
        """
        B, N, D = cs.n_samples, cs.n_patches, self.dim
        Q, H = self.n_queries, self.n_heads
        Dh = D // H

        seg = cs.location_segments()
        uniq, _, order, lookup = _unique_segments(seg)
        n_seg = uniq.numel()

        tok = self.norm_kv(cs.tokens[order]).view(-1, H, Dh)
        q = self.norm_q(self.queries).reshape(1, Q, H, Dh).expand(n_seg, Q, H, Dh)

        out = grouped_pool_attention(q, tok, tok, lookup, n_seg, self.dropout)
        out = self.proj_out(out.reshape(n_seg, Q, D))  # [n_seg, Q, D]
        out = out.reshape(n_seg, Q * D)

        fused = out.new_zeros(B, N, Q * D)
        sample_of_seg = uniq // N
        patch_of_seg = uniq % N
        fused[sample_of_seg, patch_of_seg] = out

        present = torch.zeros(B, N, dtype=torch.bool, device=cs.tokens.device)
        present[sample_of_seg, patch_of_seg] = True
        return fused, present

    # ------------------------------------------------------------------ #
    def extra_repr(self) -> str:
        return f"dim={self.dim}, n_heads={self.n_heads}, n_queries={self.n_queries}"


class MultiheadAttentionPooling(nn.Module):
    """Pooling by multi-head attention (PMA) over a padding-free token set.

    A learned query (``n_queries`` of them) cross-attends to every real token
    of each segment and emits a fixed-size descriptor per segment.  This is the
    same padding-free grouped-attention primitive as :class:`ChannelFusion`,
    factored out so the inference-time descriptor of
    :class:`~fuse_channels.mae.ChannelFusionMAE` can pool across all of a
    sample's tokens instead of taking a mean.

    Parameters
    ----------
    dim:
        token feature dimension.
    n_heads:
        attention heads (``dim`` must be divisible by it).
    n_queries:
        number of learned pooling queries; output is ``n_queries * dim``.
    dropout:
        attention dropout.

    Inputs
    ------
    x:
        ``[T, D]`` real tokens (no padding).
    seg:
        ``[T]`` segment id of each token, in ``[0, n_segments)``.
    n_segments:
        total number of segments.  Segments without any token pool to zeros.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int = 8,
        n_queries: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert dim % n_heads == 0, "dim must be divisible by n_heads"
        self.dim = dim
        self.n_heads = n_heads
        self.n_queries = n_queries
        self.dropout = dropout

        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.proj_out = nn.Linear(dim, dim)
        self.queries = nn.Parameter(torch.zeros(1, n_queries, dim))
        nn.init.trunc_normal_(self.queries, std=0.02)

    def forward(
        self, x: torch.Tensor, seg: torch.Tensor, n_segments: int
    ) -> torch.Tensor:
        """Return ``[n_segments, n_queries * dim]`` attention-pooled descriptors."""
        H, Dh, Q = self.n_heads, self.dim // self.n_heads, self.n_queries
        out = x.new_zeros(n_segments, Q * self.dim)
        if x.numel() == 0:
            return out

        uniq, _, order, lookup = _unique_segments(seg)
        n_uniq = uniq.numel()
        tok = self.norm_kv(x[order]).view(-1, H, Dh)
        q = self.norm_q(self.queries).reshape(1, Q, H, Dh).expand(n_uniq, Q, H, Dh)

        pooled = grouped_pool_attention(q, tok, tok, lookup, n_uniq, self.dropout)
        pooled = self.proj_out(pooled.reshape(n_uniq, Q, self.dim)).reshape(
            n_uniq, Q * self.dim
        )
        out[uniq] = pooled
        return out

    def extra_repr(self) -> str:
        return f"dim={self.dim}, n_heads={self.n_heads}, n_queries={self.n_queries}"


class HypernetworkWeightedSum(nn.Module):
    """``joint = Σ_c softmax_c(MLP(identity_c)) · V_c`` -- ``O(C)``, no padding.

    Preferred when quadratic channel-attention cost matters (e.g. 60-channel
    multiplexed panels).  Weights are produced from the *channel identity*
    (``ChannelSet.identity``) so routing stays marker-aware.
    """

    def __init__(
        self,
        dim: int,
        n_queries: int = 1,
        n_patches: int = 0,
        hidden: int | None = None,
    ) -> None:
        super().__init__()
        assert n_patches > 0, "n_patches is required"
        self.dim = dim
        self.n_queries = n_queries
        self.n_patches = n_patches

        hidden = hidden or dim
        self.weight_net = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, n_queries)
        )
        self.value_proj = nn.Linear(dim, dim)
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, cs: ChannelSet):
        if cs.identity is not None:
            w_in = cs.identity
        else:  # fall back to token features if identity was not materialised
            w_in = cs.tokens
        logits = self.weight_net(w_in)  # [T, Q]
        values = self.out_norm(self.value_proj(cs.tokens))  # [T, D]

        seg = cs.location_segments()
        uniq, _, _, lookup = _unique_segments(seg)
        n_seg = uniq.numel()

        weights = segment_softmax(logits, lookup, n_seg)  # [T, Q]
        contrib = weights.unsqueeze(-1) * values.unsqueeze(1)  # [T, Q, D]
        out = values.new_zeros(n_seg, self.n_queries, self.dim)
        out.index_add_(0, lookup, contrib)
        out = out.reshape(n_seg, self.n_queries * self.dim)

        N = cs.n_patches
        fused = out.new_zeros(cs.n_samples, N, self.n_queries * self.dim)
        sample_of_seg = uniq // N
        patch_of_seg = uniq % N
        fused[sample_of_seg, patch_of_seg] = out
        present = torch.zeros(
            cs.n_samples, N, dtype=torch.bool, device=cs.tokens.device
        )
        present[sample_of_seg, patch_of_seg] = True
        return fused, present


# ---------------------------------------------------------------------- #
# Masked-padded reference implementation -- used by tests to prove exact
# padding-free equivalence.  NOT used in the model path.
# ---------------------------------------------------------------------- #
def dense_fusion_reference(fusion: ChannelFusion, cs: ChannelSet) -> torch.Tensor:
    """Dense ``[B, C_max, N, D]`` + mask reference for ``ChannelFusion``.

    Implements the exact same math as :meth:`ChannelFusion.forward` but
    materialises the padded channel axis and an attention mask.
    """
    B, N, D = cs.n_samples, cs.n_patches, fusion.dim
    Q, H = fusion.n_queries, fusion.n_heads
    Dh = D // H
    device = cs.tokens.device

    counts = torch.bincount(cs.sample_idx, minlength=B)
    Cmax = int(cs.channel_idx.max().item()) + 1 if cs.channel_idx.numel() else 1

    x = cs.tokens.new_zeros(B, Cmax, N, D)
    mask = torch.zeros(B, Cmax, N, dtype=torch.bool, device=device)
    x[cs.sample_idx, cs.channel_idx, cs.patch_idx] = cs.tokens
    mask[cs.sample_idx, cs.channel_idx, cs.patch_idx] = True
    present = mask.any(dim=1)  # [B, N]

    kv = fusion.norm_kv(x).permute(0, 2, 1, 3).reshape(B * N, Cmax, D)
    q = fusion.norm_q(fusion.queries)  # [1, Q, D]
    q = q.expand(B * N, Q, D)
    m = mask.permute(0, 2, 1).reshape(B * N, Cmax)

    qh = q.view(B * N, Q, H, Dh).permute(0, 2, 1, 3)
    kh = kv.view(B * N, Cmax, H, Dh).permute(0, 2, 1, 3)
    bias = torch.zeros(B * N, 1, 1, Cmax, device=device)
    bias = bias.masked_fill(~m[:, None, None, :], float("-inf"))
    o = F.scaled_dot_product_attention(qh, kh, kh, attn_mask=bias)
    o = o.permute(0, 2, 1, 3).reshape(B * N, Q, D)
    o = fusion.proj_out(o).reshape(B * N, Q * D)
    out = o.view(B, N, Q * D)
    out = out * present.unsqueeze(-1)
    # locations with no channel produced NaNs above; scrub them.
    return torch.nan_to_num(out)
