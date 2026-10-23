"""Channel-fusion modules: map an arbitrary channel set to a fixed-size joint
embedding, with **no padding tokens**.

Two interchangeable pooling mechanisms are provided:

* :class:`ChannelFusion` -- PMA-style attention pooling over the channel axis
  (the default; the same single-query per-location pattern used by Panopticon).
  It is **deliberately projection-free for K/V** (the Panopticon-style
  simplification: keys and values are the LayerNormed tokens themselves), but
  keeps a learned query and an output projection.
* :class:`IdentityGatedSum` -- ``O(C)`` identity-gated weighted sum, the cheap
  ablation for 20-60 channel panels.  It is *not* a hypernetwork: a small MLP
  emits per-channel mixing weights from the channel embedding; it does not
  generate network weights.

Both emit one ``[B, N, D]`` token per spatial location and add the learned
post-fusion spatial positional embedding before returning.  ChannelViT and
ChAda-ViT are precedents for cross-channel attention, but they use full joint
attention over the ``C·N`` tokens rather than fixed-query pooling.

Both are set-native: they consume a :class:`~fuse_channels.container.ChannelSet`
built from real tokens only and never create a ``[B, C_max, D]`` tensor.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .container import ChannelSet
from .encoder import TransformerBlock
from .ops import grouped_pool_attention, grouped_softmax


def _unique_groups(group_id: torch.Tensor):
    """Return ``(unique_sorted, counts, order, lookup)`` for per-token group ids."""
    uniq, counts = torch.unique(group_id, return_counts=True)
    order = torch.argsort(group_id, stable=True)
    lookup = torch.searchsorted(uniq, group_id[order])
    return uniq, counts, order, lookup


class ChannelFusion(nn.Module):
    """Single-query attention pooling of the channel axis at every location.

    Channels are fused *within* each ``(sample, patch)`` location into one
    shared learned query; the result is scattered back to ``[B, N, D]`` and the
    learned spatial positional embedding is added.

    K/V projections are intentionally omitted: keys = values = the LayerNormed
    channel tokens, which is the Panopticon-style simplification for per-location
    channel pooling.  The learned query and the output projection remain.

    Parameters
    ----------
    dim:
        token feature dimension.
    n_heads:
        attention heads (``dim`` must be divisible by it).
    n_patches:
        spatial patch count (sizes the post-fusion positional embedding).
    dropout:
        attention dropout.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int = 8,
        n_patches: int = 0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert dim % n_heads == 0, "dim must be divisible by n_heads"
        assert n_patches > 0, "n_patches is required"
        self.dim = dim
        self.n_heads = n_heads
        self.n_patches = n_patches
        self.dropout = dropout

        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.proj_out = nn.Linear(dim, dim)

        # one shared query, applied at every spatial location
        self.queries = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.queries, std=0.02)
        # post-fusion spatial positional embedding
        self.spatial_pos = nn.Parameter(torch.zeros(1, n_patches, dim))
        nn.init.trunc_normal_(self.spatial_pos, std=0.02)

    def _attention_dropout(self) -> float:
        """Attention dropout probability, disabled in eval mode."""
        return self.dropout if self.training else 0.0

    # ------------------------------------------------------------------ #
    def forward(self, cs: ChannelSet):
        """Return ``(fused, present)``.

        ``fused`` is ``[B, N, D]`` (post-fusion PE added) and ``present`` is a
        boolean mask marking locations that had at least one channel.
        """
        if cs.n_patches != self.n_patches:
            raise ValueError(
                f"ChannelFusion was built for n_patches={self.n_patches}, "
                f"got ChannelSet.n_patches={cs.n_patches}"
            )
        B, N, D = cs.n_samples, cs.n_patches, self.dim
        H = self.n_heads
        Dh = D // H

        group_id = cs.location_group_ids()
        uniq, _, order, lookup = _unique_groups(group_id)
        n_groups = uniq.numel()

        tok = self.norm_kv(cs.tokens[order]).view(-1, H, Dh)
        q = self.norm_q(self.queries).reshape(1, 1, H, Dh).expand(n_groups, 1, H, Dh)

        out = grouped_pool_attention(
            q, tok, tok, lookup, n_groups, self._attention_dropout()
        )
        out = self.proj_out(out.reshape(n_groups, D))  # [n_groups, D]

        fused = out.new_zeros(B, N, D)
        sample_of_group = uniq // N
        patch_of_group = uniq % N
        fused[sample_of_group, patch_of_group] = out
        fused = fused + self.spatial_pos

        present = torch.zeros(B, N, dtype=torch.bool, device=cs.tokens.device)
        present[sample_of_group, patch_of_group] = True
        return fused, present

    # ------------------------------------------------------------------ #
    def extra_repr(self) -> str:
        return f"dim={self.dim}, n_heads={self.n_heads}"


class MultiheadAttentionPooling(nn.Module):
    """Pooling by multi-head attention (PMA) over a padding-free token set.

    A learned query (``n_queries`` of them) cross-attends to every real token
    of each group through standard Q/K/V/O projections and emits a fixed-size
    descriptor per group.  This module is *trained* through the MAE objective
    (the descriptor is appended to the decoder KV), so it is a real PMA
    readout rather than an inference-only artifact.

    Parameters
    ----------
    dim:
        token feature dimension.
    n_heads:
        attention heads (``dim`` must be divisible by it).
    n_queries:
        number of learned pooling queries.
    dropout:
        attention / SAB dropout.
    post_sab:
        apply a :class:`~fuse_channels.encoder.TransformerBlock` over the
        ``n_queries`` outputs of each sample (one group per sample) before the
        readout.  Only meaningful for ``n_queries > 1``.
    readout:
        ``"flat"`` (default) returns ``[n_groups, n_queries * dim]``;
        ``"mean"`` averages the query outputs and returns
        ``[n_groups, dim]``.

    Inputs
    ------
    x:
        ``[T, D]`` real tokens (no padding).
    group_id:
        ``[T]`` group id of each token, in ``[0, n_groups)``.
    n_groups:
        total number of groups.  Groups without any token pool to zeros.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int = 8,
        n_queries: int = 1,
        dropout: float = 0.0,
        post_sab: bool = False,
        readout: str = "flat",
    ) -> None:
        super().__init__()
        assert dim % n_heads == 0, "dim must be divisible by n_heads"
        if readout not in ("flat", "mean"):
            raise ValueError(f"unknown readout '{readout}'")
        self.dim = dim
        self.n_heads = n_heads
        self.n_queries = n_queries
        self.dropout = dropout
        self.readout = readout

        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.queries = nn.Parameter(torch.zeros(1, n_queries, dim))
        nn.init.trunc_normal_(self.queries, std=0.02)
        self.post_sab = (
            TransformerBlock(dim, n_heads, dropout=dropout) if post_sab else None
        )

    def _attention_dropout(self) -> float:
        """Attention dropout probability, disabled in eval mode."""
        return self.dropout if self.training else 0.0

    def forward(
        self, x: torch.Tensor, group_id: torch.Tensor, n_groups: int
    ) -> torch.Tensor:
        """Return ``[n_groups, n_queries * dim]`` (``flat``) or ``[n_groups, dim]``."""
        H, Dh, Q = self.n_heads, self.dim // self.n_heads, self.n_queries
        out_dim = Q * self.dim if self.readout == "flat" else self.dim
        out = x.new_zeros(n_groups, out_dim)
        if x.numel() == 0:
            return out

        uniq, _, order, lookup = _unique_groups(group_id)
        n_uniq = uniq.numel()
        kv_norm = self.norm_kv(x[order])
        k = self.k_proj(kv_norm).view(-1, H, Dh)
        v = self.v_proj(kv_norm).view(-1, H, Dh)
        q = (
            self.q_proj(self.norm_q(self.queries))
            .reshape(1, Q, H, Dh)
            .expand(n_uniq, Q, H, Dh)
        )

        pooled = grouped_pool_attention(
            q, k, v, lookup, n_uniq, self._attention_dropout()
        )
        pooled = self.out_proj(pooled.reshape(n_uniq, Q, self.dim))

        if self.post_sab is not None:
            pool_group_id = torch.arange(n_uniq, device=x.device).repeat_interleave(Q)
            pooled = self.post_sab(
                pooled.reshape(n_uniq * Q, self.dim), pool_group_id, n_uniq
            ).reshape(n_uniq, Q, self.dim)

        pooled = (
            pooled.reshape(n_uniq, Q * self.dim)
            if self.readout == "flat"
            else pooled.mean(dim=1)
        )
        out[uniq] = pooled
        return out

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, n_heads={self.n_heads}, n_queries={self.n_queries}, "
            f"readout={self.readout}, post_sab={self.post_sab is not None}"
        )


class IdentityGatedSum(nn.Module):
    """``joint = Σ_c softmax_c(MLP(channel_embed_c)) · V_c`` -- ``O(C)``, no padding.

    Preferred when quadratic channel-attention cost matters (e.g. 60-channel
    multiplexed panels).  Weights are produced from the *channel embedding*
    (``ChannelSet.channel_embed``) so routing stays identity-aware; the channel
    MLP is evaluated **once per unique channel** (via
    ``ChannelSet.channel_group_id``) and broadcast to that channel's tokens.

    The output is ``[B, N, D]`` with the learned post-fusion spatial positional
    embedding added, matching :class:`ChannelFusion`.
    """

    def __init__(
        self,
        dim: int,
        n_patches: int = 0,
        hidden: int | None = None,
    ) -> None:
        super().__init__()
        assert n_patches > 0, "n_patches is required"
        self.dim = dim
        self.n_patches = n_patches

        hidden = hidden or dim
        self.weight_net = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, 1)
        )
        self.value_proj = nn.Linear(dim, dim)
        self.out_norm = nn.LayerNorm(dim)
        self.spatial_pos = nn.Parameter(torch.zeros(1, n_patches, dim))
        nn.init.trunc_normal_(self.spatial_pos, std=0.02)

    def forward(self, cs: ChannelSet):
        if cs.n_patches != self.n_patches:
            raise ValueError(
                f"IdentityGatedSum was built for n_patches={self.n_patches}, "
                f"got ChannelSet.n_patches={cs.n_patches}"
            )
        if cs.channel_embed is None:
            raise ValueError("IdentityGatedSum requires ChannelSet.channel_embed")
        if cs.channel_group_id is None:
            raise ValueError("IdentityGatedSum requires ChannelSet.channel_group_id")

        # -- one MLP call for all unique channels, then broadcast ---------- #
        cg = cs.channel_group_id
        _, inverse = torch.unique(cg, return_inverse=True)
        order = torch.argsort(cg, stable=True)
        ranks = inverse[order]
        first = torch.ones_like(ranks, dtype=torch.bool)
        first[1:] = ranks[1:] != ranks[:-1]
        chan_embed = cs.channel_embed[order[first]]  # [C_uniq, D]
        chan_logits = self.weight_net(chan_embed).squeeze(-1)  # [C_uniq]
        logits = chan_logits[inverse]  # [T]
        values = self.out_norm(self.value_proj(cs.tokens))  # [T, D]

        # ``logits``/``values`` are in input-token order, so the group mapping
        # must be too: a sorted-order lookup would scramble locations.
        group_id = cs.location_group_ids()
        uniq, location_inverse = torch.unique(group_id, return_inverse=True)
        n_groups = uniq.numel()

        weights = grouped_softmax(logits, location_inverse, n_groups)  # [T]
        contrib = weights.unsqueeze(-1) * values  # [T, D]
        out = values.new_zeros(n_groups, self.dim)
        out.index_add_(0, location_inverse, contrib)

        N = cs.n_patches
        fused = out.new_zeros(cs.n_samples, N, self.dim)
        sample_of_group = uniq // N
        patch_of_group = uniq % N
        fused[sample_of_group, patch_of_group] = out
        fused = fused + self.spatial_pos
        present = torch.zeros(
            cs.n_samples, N, dtype=torch.bool, device=cs.tokens.device
        )
        present[sample_of_group, patch_of_group] = True
        return fused, present

    def extra_repr(self) -> str:
        return f"dim={self.dim}"


# ---------------------------------------------------------------------- #
# Masked-padded reference implementation -- used by tests to prove exact
# padding-free equivalence.  NOT used in the model path.
# ---------------------------------------------------------------------- #
def _dense_fusion_reference(fusion: ChannelFusion, cs: ChannelSet) -> torch.Tensor:
    """Dense ``[B, C_max, N, D]`` + mask reference for :class:`ChannelFusion`.

    Implements the exact same math as :meth:`ChannelFusion.forward` (including
    the post-fusion positional embedding) but materialises the padded channel
    axis and an attention mask.
    """
    B, N, D = cs.n_samples, cs.n_patches, fusion.dim
    H = fusion.n_heads
    Dh = D // H
    device = cs.tokens.device

    Cmax = int(cs.channel_idx.max().item()) + 1 if cs.channel_idx.numel() else 1

    x = cs.tokens.new_zeros(B, Cmax, N, D)
    mask = torch.zeros(B, Cmax, N, dtype=torch.bool, device=device)
    x[cs.sample_idx, cs.channel_idx, cs.patch_idx] = cs.tokens
    mask[cs.sample_idx, cs.channel_idx, cs.patch_idx] = True
    present = mask.any(dim=1)  # [B, N]

    kv = fusion.norm_kv(x).permute(0, 2, 1, 3).reshape(B * N, Cmax, D)
    q = fusion.norm_q(fusion.queries).expand(B * N, 1, D)
    m = mask.permute(0, 2, 1).reshape(B * N, Cmax)

    qh = q.reshape(B * N, 1, H, Dh).permute(0, 2, 1, 3)
    kh = kv.view(B * N, Cmax, H, Dh).permute(0, 2, 1, 3)
    bias = torch.zeros(B * N, 1, 1, Cmax, device=device)
    bias = bias.masked_fill(~m[:, None, None, :], float("-inf"))
    o = F.scaled_dot_product_attention(qh, kh, kh, attn_mask=bias)
    o = o.permute(0, 2, 1, 3).reshape(B * N, 1, D)
    o = fusion.proj_out(o).reshape(B, N, D)
    out = o * present.unsqueeze(-1)
    # locations with no channel produced NaNs above; scrub them before adding PE.
    return torch.nan_to_num(out) + fusion.spatial_pos
