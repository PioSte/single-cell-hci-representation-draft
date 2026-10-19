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
    """Attention pooling of the channel axis into fixed learned queries.

    Parameters
    ----------
    dim:
        token feature dimension.
    n_heads:
        attention heads (``dim`` must be divisible by it).
    n_queries:
        number of PMA queries per location/sample (usually 1).  Output feature
        size is ``n_queries * dim``.
    n_patches:
        spatial patch count, required for ``mode="per_location"``.
    mode:
        ``"per_location"`` fuses channels *at each spatial location* and returns
        ``[B, N, n_queries*dim]`` (the front-end for a spatial transformer).
        ``"global"`` fuses the whole token set per sample and returns
        ``[B, n_queries*dim]``.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int = 8,
        n_queries: int = 1,
        n_patches: int = 0,
        mode: str = "per_location",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert dim % n_heads == 0, "dim must be divisible by n_heads"
        assert mode in ("per_location", "global")
        self.dim = dim
        self.n_heads = n_heads
        self.n_queries = n_queries
        self.n_patches = n_patches
        self.mode = mode
        self.dropout = dropout

        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.proj_out = nn.Linear(dim, dim)

        if mode == "per_location":
            assert n_patches > 0, "n_patches is required for per_location mode"
            self.queries = nn.Parameter(torch.zeros(n_patches, n_queries, dim))
        else:
            self.queries = nn.Parameter(torch.zeros(1, n_queries, dim))
        nn.init.trunc_normal_(self.queries, std=0.02)

    # ------------------------------------------------------------------ #
    def forward(self, cs: ChannelSet):
        """Return ``(fused, present)``.

        ``fused`` is ``[B, N, n_queries*dim]`` (per_location) or
        ``[B, n_queries*dim]`` (global).  ``present`` is a boolean mask marking
        locations/samples that had at least one channel.
        """
        if self.mode == "per_location":
            return self._forward_per_location(cs)
        return self._forward_global(cs)

    # ------------------------------------------------------------------ #
    def _forward_per_location(self, cs: ChannelSet):
        B, N, D = cs.n_samples, cs.n_patches, self.dim
        Q, H = self.n_queries, self.n_heads
        Dh = D // H

        seg = cs.location_segments()
        uniq, _, order, lookup = _unique_segments(seg)
        n_seg = uniq.numel()

        tok = self.norm_kv(cs.tokens[order]).view(-1, H, Dh)
        patch_of_seg = uniq % N
        q = self.norm_q(self.queries[patch_of_seg]).view(n_seg, Q, H, Dh)

        out = grouped_pool_attention(q, tok, tok, lookup, n_seg, self.dropout)
        out = self.proj_out(out.reshape(n_seg, Q, D))  # [n_seg, Q, D]
        out = out.reshape(n_seg, Q * D)

        fused = out.new_zeros(B, N, Q * D)
        sample_of_seg = uniq // N
        fused[sample_of_seg, patch_of_seg] = out

        present = torch.zeros(B, N, dtype=torch.bool, device=cs.tokens.device)
        present[sample_of_seg, patch_of_seg] = True
        return fused, present

    def _forward_global(self, cs: ChannelSet):
        B, D = cs.n_samples, self.dim
        Q, H = self.n_queries, self.n_heads
        Dh = D // H

        seg = cs.channel_segments()
        _, _, order, lookup = _unique_segments(seg)

        tok = self.norm_kv(cs.tokens[order]).view(-1, H, Dh)
        q = self.norm_q(self.queries).reshape(1, Q, H, Dh).expand(B, Q, H, Dh)

        out = grouped_pool_attention(q, tok, tok, lookup, B, self.dropout)
        out = self.proj_out(out.reshape(B, Q, D)).reshape(B, Q * D)

        present = torch.bincount(seg, minlength=B) > 0
        return out, present

    # ------------------------------------------------------------------ #
    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, n_heads={self.n_heads}, n_queries={self.n_queries}, "
            f"mode={self.mode}"
        )


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
        mode: str = "per_location",
        hidden: int | None = None,
    ) -> None:
        super().__init__()
        assert mode in ("per_location", "global")
        self.dim = dim
        self.n_queries = n_queries
        self.n_patches = n_patches
        self.mode = mode

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

        if self.mode == "per_location":
            seg = cs.location_segments()
            uniq, _, _, lookup = _unique_segments(seg)
            n_seg = uniq.numel()
        else:
            seg = cs.channel_segments()
            uniq, _, _, lookup = _unique_segments(seg)
            n_seg = cs.n_samples

        weights = segment_softmax(logits, lookup, n_seg)  # [T, Q]
        contrib = weights.unsqueeze(-1) * values.unsqueeze(1)  # [T, Q, D]
        out = values.new_zeros(n_seg, self.n_queries, self.dim)
        out.index_add_(0, lookup, contrib)
        out = out.reshape(n_seg, self.n_queries * self.dim)

        if self.mode == "global":
            present = torch.bincount(seg, minlength=cs.n_samples) > 0
            return out, present

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

    Implements the exact same math as :meth:`ChannelFusion._forward_per_location`
    but materialises the padded channel axis and an attention mask.
    """
    assert fusion.mode == "per_location"
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
    q = fusion.norm_q(fusion.queries)  # [N, Q, D]
    q = q.unsqueeze(0).expand(B, N, Q, D).reshape(B * N, Q, D)
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
