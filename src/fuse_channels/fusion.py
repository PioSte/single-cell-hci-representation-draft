"""Channel-fusion modules: map an arbitrary channel set to a fixed-size joint
embedding, with **no padding tokens**.

Two interchangeable pooling mechanisms are provided:

* :class:`ChannelFusion` -- PMA-style attention pooling over the channel axis
  (the default; the same single-query per-location pattern used by Panopticon).
  It follows Panopticon's recipe: learned Q/K/V projections into an attention
  dimension ``attn_dim`` (usually wider than ``dim``), a learned query, and a
  final projection back to ``dim``.
* :class:`IdentityGatedSum` -- ``O(C)`` identity-gated weighted sum, the cheap
  ablation for 20-60 channel panels.  It is *not* a hypernetwork: a small MLP
  emits per-channel mixing weights from the channel embedding; it does not
  generate network weights.

Both emit one ``[D]`` vector per *present* spatial location, as a flat
``[K_f, D]`` tensor plus the ``[B, N]`` presence index.  Under the FlexiViT
target architecture they add no positional embedding: the encoder owns layout
through 2-D RoPE and the decoder owns its absolute anchor through fractional
PE.  ``pos_embed="learned"`` / ``"sincos"`` remain as the legacy fixed-grid
ablation; ``"none"`` is the default.  ChannelViT and ChAda-ViT are precedents
for cross-channel attention, but they use full joint attention over the
``C·N`` tokens rather than fixed-query pooling.

Both are set-native: they consume a :class:`~fuse_channels.container.ChannelSet`
built from real tokens only and never create a ``[B, C_max, D]`` tensor.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .container import ChannelSet
from .encoder import TransformerBlock
from .ops import grouped_pool_attention, grouped_softmax
from .pos_embed import add_spatial_pos


def _unique_groups(group_id: torch.Tensor):
    """Return ``(unique_sorted, counts, order, lookup)`` for per-token group ids."""
    uniq, counts = torch.unique(group_id, return_counts=True)
    order = torch.argsort(group_id, stable=True)
    lookup = torch.searchsorted(uniq, group_id[order])
    return uniq, counts, order, lookup


class ChannelFusion(nn.Module):
    """Single-query attention pooling of the channel axis at every location.

    Channels are fused *within* each ``(sample, patch)`` location into one
    shared learned query; the per-location results form ``[K_f, D]`` and the
    spatial positional embedding is added when the legacy ``pos_embed`` table
    is enabled (``"learned"`` or ``"sincos"``; the default is ``"none"``).

    This is Panopticon's channel cross-attention recipe: LayerNormed tokens are
    projected with learned ``q_proj`` / ``k_proj`` / ``v_proj`` into an attention
    dimension ``attn_dim`` (``3 * dim`` by default, matching Panopticon's
    ``D_attn = 3D``), attention runs there, and ``proj_out`` maps back to
    ``dim``.  The output is the flat ``[K_f, D]`` array the pooling already
    produced, so no ``[B, N, D]`` buffer is ever materialised.  Panopticon's
    ablation found ``attn_dim > dim`` important
    (``references/panopticon/paper.md:92,194,219``).  Set ``attn_dim=dim`` for
    the parameter-matched variant.

    Parameters
    ----------
    dim:
        token feature dimension.
    n_heads:
        attention heads (``attn_dim`` must be divisible by it).
    n_patches:
        spatial patch count (sizes the post-fusion positional embedding).
    dropout:
        attention dropout.
    pos_embed:
        ``"learned"`` (default) or ``"sincos"``; see
        :mod:`fuse_channels.pos_embed`.
    grid_size:
        optional ``int | (grid_h, grid_w)`` for the sine-cosine lattice;
        defaults to ``sqrt(n_patches)``.
    attn_dim:
        width of the Q/K/V attention space.  ``None`` (default) uses
        ``3 * dim``, Panopticon's ``D_attn = 3D``; ``dim`` reproduces the
        narrower, parameter-matched ablation.

    Channel identity is decoupled from content: ``cs.tokens`` holds patch
    content only, and the channel embedding enters (i) as a per-head additive
    bias on the pooling logits and (ii) as an attention-pooled skip added
    *after* the content projection.  This preserves ChannelViT's per-channel
    identity routing without ever summing a fixed-scale vector into a
    variable-scale content token before a LayerNorm, so the identity/content
    balance cannot drift with patch size (or any other content-scale change).
    """

    # Registered only by the table-based (``learned``/``sincos``) variants.
    spatial_pos: torch.Tensor

    def __init__(
        self,
        dim: int,
        n_heads: int = 8,
        n_patches: int | None = None,
        dropout: float = 0.0,
        pos_embed: str = "none",
        grid_size=None,
        attn_dim: int | None = None,
    ) -> None:
        super().__init__()
        attn_dim = 3 * dim if attn_dim is None else attn_dim
        assert attn_dim % n_heads == 0, "attn_dim must be divisible by n_heads"
        self.dim = dim
        self.attn_dim = attn_dim
        self.n_heads = n_heads
        self.n_patches = n_patches
        self.dropout = dropout
        self.has_spatial_pos = pos_embed != "none"

        # per-channel, per-head additive routing bias, and a skip that carries
        # the pooled identity into the fused representation (outside the LN).
        self.identity_bias = nn.Linear(dim, n_heads, bias=False)
        self.identity_skip = nn.Linear(dim, dim, bias=False)
        nn.init.trunc_normal_(self.identity_bias.weight, std=0.02)
        nn.init.trunc_normal_(self.identity_skip.weight, std=0.02)

        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.q_proj = nn.Linear(dim, attn_dim)
        self.k_proj = nn.Linear(dim, attn_dim)
        self.v_proj = nn.Linear(dim, attn_dim)
        self.proj_out = nn.Linear(attn_dim, dim)

        # one shared query, applied at every spatial location
        self.queries = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.queries, std=0.02)
        # optional legacy post-fusion positional embedding
        if pos_embed != "none":
            if n_patches is None:
                raise ValueError("n_patches is required for a table-based spatial PE")
            add_spatial_pos(self, dim, n_patches, pos_embed, grid_size=grid_size, batch_dims=1)
        else:
            self.pos_embed = "none"

    def _attention_dropout(self) -> float:
        """Attention dropout probability, disabled in eval mode."""
        return self.dropout if self.training else 0.0

    # ------------------------------------------------------------------ #
    def forward(self, cs: ChannelSet):
        """Return ``(fused, present)``.

        ``fused`` is ``[K_f, D]``: one vector per present ``(sample, patch)``
        location, in the order of ``present.nonzero()``.  Absent locations cost
        nothing -- no ``[B, N, D]`` buffer is allocated -- so ``N`` never enters
        a tensor here.  ``present`` is the ``[B, N]`` index of which locations
        exist; it is also where the per-location grid coordinates come from.
        """
        if self.n_patches is not None and cs.n_patches != self.n_patches and self.has_spatial_pos:
            raise ValueError(
                f"ChannelFusion was built for n_patches={self.n_patches}, "
                f"got ChannelSet.n_patches={cs.n_patches}"
            )
        B, N, D = cs.n_samples, cs.n_patches, self.dim
        H = self.n_heads
        Dh = self.attn_dim // H

        group_id = cs.location_group_ids()
        uniq, _, order, lookup = _unique_groups(group_id)
        n_groups = uniq.numel()

        tok = self.norm_kv(cs.tokens[order])
        k = self.k_proj(tok).view(-1, H, Dh)
        v = self.v_proj(tok).view(-1, H, Dh)
        q = self.q_proj(self.norm_q(self.queries)).reshape(1, 1, H, Dh).expand(n_groups, 1, H, Dh)

        if cs.channel_embed is None:
            raise ValueError("ChannelFusion requires ChannelSet.channel_embed")
        ident = cs.channel_embed[order]  # [T, D]
        logit_bias = self.identity_bias(ident)  # [T, H]
        out, weights = grouped_pool_attention(
            q,
            k,
            v,
            lookup,
            n_groups,
            self._attention_dropout(),
            logit_bias=logit_bias,
            return_weights=True,
        )
        ident_pool = out.new_zeros(n_groups, D)
        ident_pool.index_add_(0, lookup, weights.unsqueeze(-1) * ident)
        out = self.proj_out(out.reshape(n_groups, self.attn_dim))
        out = out + self.identity_skip(ident_pool)  # [n_groups, D]

        sample_of_group = uniq // N
        patch_of_group = uniq % N
        if self.has_spatial_pos:
            out = out + self.spatial_pos[0][patch_of_group]

        present = torch.zeros(B, N, dtype=torch.bool, device=cs.tokens.device)
        present[sample_of_group, patch_of_group] = True
        return out, present

    # ------------------------------------------------------------------ #
    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, attn_dim={self.attn_dim}, n_heads={self.n_heads}, "
            f"pos_embed={self.pos_embed}"
        )


class MultiheadAttentionPooling(nn.Module):
    """Pooling by multi-head attention (PMA) over a padding-free token set.

    A learned query (``n_queries`` of them) cross-attends to every real token
    of each group through standard Q/K/V/O projections and emits a fixed-size
    pooled output per group.  This module is *trained* through the MAE
    objective, so it is a real PMA readout rather than an inference-only
    artifact.

    ``ChannelFusionMAE`` uses it for the spatial pooling to fixed size latent:
    ``n_queries=n_latents``, ``readout="flat"``, reshaped to
    ``[B, n_latents, D]``.  The output is independent of the number of tokens
    in each group.

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
        self.post_sab = TransformerBlock(dim, n_heads, dropout=dropout) if post_sab else None

    def _attention_dropout(self) -> float:
        """Attention dropout probability, disabled in eval mode."""
        return self.dropout if self.training else 0.0

    def forward(self, x: torch.Tensor, group_id: torch.Tensor, n_groups: int) -> torch.Tensor:
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
        q = self.q_proj(self.norm_q(self.queries)).reshape(1, Q, H, Dh).expand(n_uniq, Q, H, Dh)

        pooled = grouped_pool_attention(q, k, v, lookup, n_uniq, self._attention_dropout())
        pooled = self.out_proj(pooled.reshape(n_uniq, Q, self.dim))

        if self.post_sab is not None:
            pool_group_id = torch.arange(n_uniq, device=x.device).repeat_interleave(Q)
            pooled = self.post_sab(
                pooled.reshape(n_uniq * Q, self.dim), pool_group_id, n_uniq
            ).reshape(n_uniq, Q, self.dim)

        pooled = (
            pooled.reshape(n_uniq, Q * self.dim) if self.readout == "flat" else pooled.mean(dim=1)
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

    The output is ``[K_f, D]``, one row per present location, with the
    post-fusion spatial positional embedding added in legacy mode; it matches
    :class:`ChannelFusion`.

    Channel identity is decoupled from content: ``cs.tokens`` holds patch
    content only, the gate still comes from ``channel_embed``, and the identity
    is added back as an attention-weighted skip *after* the content
    normalisation.
    """

    # Registered only by the table-based (``learned``/``sincos``) variants.
    spatial_pos: torch.Tensor

    def __init__(
        self,
        dim: int,
        n_patches: int | None = None,
        hidden: int | None = None,
        pos_embed: str = "none",
        grid_size=None,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.n_patches = n_patches
        self.has_spatial_pos = pos_embed != "none"
        # normalise content before the value projection so its bias cannot
        # re-introduce a content-scale dependence.
        self.content_norm = nn.LayerNorm(dim)
        self.identity_skip = nn.Linear(dim, dim, bias=False)
        nn.init.trunc_normal_(self.identity_skip.weight, std=0.02)
        hidden = hidden or dim
        self.weight_net = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.value_proj = nn.Linear(dim, dim)
        self.out_norm = nn.LayerNorm(dim)
        if pos_embed != "none":
            if n_patches is None:
                raise ValueError("n_patches is required for a table-based spatial PE")
            add_spatial_pos(self, dim, n_patches, pos_embed, grid_size=grid_size, batch_dims=1)
        else:
            self.pos_embed = "none"

    def forward(self, cs: ChannelSet):
        if self.n_patches is not None and cs.n_patches != self.n_patches and self.has_spatial_pos:
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
        values = self.out_norm(self.value_proj(self.content_norm(cs.tokens)))  # [T, D]

        # ``logits``/``values`` are in input-token order, so the group mapping
        # must be too: a sorted-order lookup would scramble locations.
        group_id = cs.location_group_ids()
        uniq, location_inverse = torch.unique(group_id, return_inverse=True)
        n_groups = uniq.numel()

        weights = grouped_softmax(logits, location_inverse, n_groups)  # [T]
        contrib = weights.unsqueeze(-1) * values  # [T, D]
        out = values.new_zeros(n_groups, self.dim)
        out.index_add_(0, location_inverse, contrib)
        ident_pool = out.new_zeros(n_groups, self.dim)
        ident_pool.index_add_(0, location_inverse, weights.unsqueeze(-1) * cs.channel_embed)
        out = out + self.identity_skip(ident_pool)

        N = cs.n_patches
        sample_of_group = uniq // N
        patch_of_group = uniq % N
        if self.has_spatial_pos:
            out = out + self.spatial_pos[0][patch_of_group]
        present = torch.zeros(cs.n_samples, N, dtype=torch.bool, device=cs.tokens.device)
        present[sample_of_group, patch_of_group] = True
        return out, present

    def extra_repr(self) -> str:
        return f"dim={self.dim}, pos_embed={self.pos_embed}"


# ---------------------------------------------------------------------- #
# Masked-padded reference implementation -- used by tests to prove exact
# padding-free equivalence.  NOT used in the model path.
# ---------------------------------------------------------------------- #
def _dense_fusion_reference(fusion: ChannelFusion, cs: ChannelSet) -> torch.Tensor:
    """Dense ``[B, C_max, N, D]`` + mask reference for :class:`ChannelFusion`.

    Implements the exact same math as :meth:`ChannelFusion.forward` (including
    the identity routing bias, the identity skip and the post-fusion positional
    embedding) but materialises the padded channel axis and an attention mask.
    """
    import math

    B, N, D = cs.n_samples, cs.n_patches, fusion.dim
    H = fusion.n_heads
    Dh = fusion.attn_dim // H
    device = cs.tokens.device

    Cmax = int(cs.channel_idx.max().item()) + 1 if cs.channel_idx.numel() else 1

    channel_embed = cs.channel_embed
    assert channel_embed is not None, "ChannelFusion requires ChannelSet.channel_embed"
    x = cs.tokens.new_zeros(B, Cmax, N, D)
    e = cs.tokens.new_zeros(B, Cmax, N, D)
    mask = torch.zeros(B, Cmax, N, dtype=torch.bool, device=device)
    x[cs.sample_idx, cs.channel_idx, cs.patch_idx] = cs.tokens
    e[cs.sample_idx, cs.channel_idx, cs.patch_idx] = channel_embed
    mask[cs.sample_idx, cs.channel_idx, cs.patch_idx] = True
    present = mask.any(dim=1)  # [B, N]

    kv = fusion.norm_kv(x).permute(0, 2, 1, 3).reshape(B * N, Cmax, D)
    q = fusion.q_proj(fusion.norm_q(fusion.queries)).expand(B * N, 1, fusion.attn_dim)
    k = fusion.k_proj(kv)
    v = fusion.v_proj(kv)
    m = mask.permute(0, 2, 1).reshape(B * N, Cmax)

    # identity routing bias, per head
    eb = e.permute(0, 2, 1, 3).reshape(B * N, Cmax, D)
    bias = fusion.identity_bias(eb).permute(0, 2, 1)  # [B*N, H, Cmax]
    bias = bias.masked_fill(~m[:, None, :], float("-inf"))

    qh = q.reshape(B * N, 1, H, Dh).permute(0, 2, 1, 3)
    kh = k.view(B * N, Cmax, H, Dh).permute(0, 2, 1, 3)
    vh = v.view(B * N, Cmax, H, Dh).permute(0, 2, 1, 3)
    scores = (qh @ kh.transpose(-1, -2)) / math.sqrt(Dh) + bias.unsqueeze(2)
    attn = torch.softmax(scores, dim=-1)

    o = (attn @ vh).permute(0, 2, 1, 3).reshape(B * N, 1, fusion.attn_dim)
    content = fusion.proj_out(o).reshape(B, N, D)
    ident = (attn.mean(dim=1) @ eb).reshape(B, N, D)
    out = content + fusion.identity_skip(ident)
    out = out * present.unsqueeze(-1)
    # locations with no channel produced NaNs above; scrub them before adding PE.
    out = torch.nan_to_num(out)
    if fusion.has_spatial_pos:
        out = out + fusion.spatial_pos
    return out
