"""Padding-free attention primitives over flattened token sets.

Everything in this module operates on *flattened, real* tokens plus integer
group ids, so no ``[B, C_max, D]`` padded tensor and no attention mask is ever
materialised.  Attention is always restricted to a single group, which is
exactly what per-sample / per-channel-set attention means.

Layout conventions
------------------
``group_id[i]`` is the group id of token ``i``.  Groups are independent; tokens
from different groups never interact.

A *group* is the set of tokens that share a ``group_id``: encoder, decoder and
pooler groups are samples (one group per sample), while fusion groups are
``(sample, patch)`` locations.  This is grouping for padding-free batching, not
grouped-query attention (there are no head groups and no shared K/V across query
heads).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _lengths_and_offsets(group_id: torch.Tensor, n_groups: int):
    lengths = torch.bincount(group_id, minlength=n_groups)
    offsets = torch.zeros(n_groups + 1, dtype=torch.long, device=group_id.device)
    offsets[1:] = torch.cumsum(lengths, dim=0)
    return lengths, offsets


def grouped_self_attention(
    x: torch.Tensor,
    group_id: torch.Tensor,
    n_groups: int,
    n_heads: int,
    dropout_p: float = 0.0,
) -> torch.Tensor:
    """Full self-attention within each group, no padding.

    Parameters
    ----------
    x:
        ``[T, D]`` token features.
    group_id:
        ``[T]`` group id per token (values in ``[0, n_groups)``).
    """
    T, D = x.shape
    out = torch.zeros_like(x)
    if T == 0:
        return out
    assert D % n_heads == 0
    head_dim = D // n_heads

    lengths, offsets = _lengths_and_offsets(group_id, n_groups)
    order = torch.argsort(group_id, stable=True)
    x_sorted = x[order]
    out_sorted = torch.empty_like(x_sorted)
    ar = None

    for L in torch.unique(lengths).tolist():
        if L == 0:
            continue
        group_ids = torch.nonzero(lengths == L, as_tuple=False).squeeze(1)
        offs = offsets[group_ids]
        if ar is None or ar.numel() < L:
            ar = torch.arange(L, device=x.device)
        idx = offs[:, None] + ar[None, :]  # [M, L]
        xx = x_sorted[idx]  # [M, L, D]
        q = xx.view(-1, L, n_heads, head_dim).permute(0, 2, 1, 3)
        o = F.scaled_dot_product_attention(q, q, q, dropout_p=dropout_p)
        o = o.permute(0, 2, 1, 3).reshape(-1, L, D)
        out_sorted[idx] = o

    out[order] = out_sorted
    return out


def grouped_projected_self_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    group_id: torch.Tensor,
    n_groups: int,
    n_heads: int,
    dropout_p: float = 0.0,
) -> torch.Tensor:
    """Self-attention over projected ``q``/``k``/``v`` sharing one group id.

    ``q``, ``k`` and ``v`` are ``[T, D]`` projections of the *same* token set
    (so every token is both a query and a key/value), and ``group_id[i]``
    identifies the group of all three rows ``i``.  Groups are independent, so
    this is exactly the encoder / target self-attention path: equal-length
    groups are batched into a single SDPA call instead of looping per group.

    Returns ``[T, D]`` in the input token order.
    """
    T, D = q.shape
    out = torch.zeros_like(q)
    if T == 0:
        return out
    assert D % n_heads == 0, "D must be divisible by n_heads"
    head_dim = D // n_heads

    lengths, offsets = _lengths_and_offsets(group_id, n_groups)
    order = torch.argsort(group_id, stable=True)
    q_sorted = q[order]
    k_sorted = k[order]
    v_sorted = v[order]
    out_sorted = torch.empty_like(q_sorted)
    ar = None

    for L in torch.unique(lengths).tolist():
        if L == 0:
            continue
        group_ids = torch.nonzero(lengths == L, as_tuple=False).squeeze(1)
        offs = offsets[group_ids]
        if ar is None or ar.numel() < L:
            ar = torch.arange(L, device=q.device)
        idx = offs[:, None] + ar[None, :]  # [M, L]
        qq = q_sorted[idx].view(-1, L, n_heads, head_dim).permute(0, 2, 1, 3)
        kk = k_sorted[idx].view(-1, L, n_heads, head_dim).permute(0, 2, 1, 3)
        vv = v_sorted[idx].view(-1, L, n_heads, head_dim).permute(0, 2, 1, 3)
        o = F.scaled_dot_product_attention(qq, kk, vv, dropout_p=dropout_p)
        out_sorted[idx] = o.permute(0, 2, 1, 3).reshape(-1, L, D)

    out[order] = out_sorted
    return out


def grouped_pool_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kv_group_id: torch.Tensor,
    n_groups: int,
    dropout_p: float = 0.0,
) -> torch.Tensor:
    """Cross-attention with one query set per group (PMA-style pooling).

    Parameters
    ----------
    q:
        ``[n_groups, n_queries, n_heads, head_dim]`` -- the learned query(s)
        for every group.  Groups with no key tokens produce zeros.
    k, v:
        ``[T, n_heads, head_dim]`` flattened key/value tokens.
    kv_group_id:
        ``[T]`` group id of each key/value token.
    """
    _, n_q, n_heads, head_dim = q.shape
    out = torch.zeros_like(q)
    if kv_group_id.numel() == 0:
        return out

    lengths, offsets = _lengths_and_offsets(kv_group_id, n_groups)
    order = torch.argsort(kv_group_id, stable=True)
    k_sorted = k[order]
    v_sorted = v[order]

    for L in torch.unique(lengths).tolist():
        if L == 0:
            continue
        group_ids = torch.nonzero(lengths == L, as_tuple=False).squeeze(1)
        offs = offsets[group_ids]
        ar = torch.arange(L, device=k.device)
        idx = offs[:, None] + ar[None, :]  # [M, L]
        kk = k_sorted[idx].permute(0, 2, 1, 3)  # [M, H, L, Dh]
        vv = v_sorted[idx].permute(0, 2, 1, 3)
        qq = q[group_ids].permute(0, 2, 1, 3)  # [M, H, n_q, Dh]
        o = F.scaled_dot_product_attention(qq, kk, vv, dropout_p=dropout_p)
        out[group_ids] = o.permute(0, 2, 1, 3)
    return out


def grouped_attention(
    q: torch.Tensor,
    q_group_id: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kv_group_id: torch.Tensor,
    n_groups: int,
    n_heads: int,
    dropout_p: float = 0.0,
) -> torch.Tensor:
    """Arbitrary-query cross-attention within groups (loops over groups).

    The general primitive behind the decoder's cross-attention to the encoder
    output, where the query and key/value lengths differ per group.  For
    self-attention (one shared token set) prefer the bucketed
    :func:`grouped_projected_self_attention`.  Queries, keys and values are
    independent ``[T, D]`` tensors; every group is attended separately, so no
    attention mask is ever materialised.

    Parameters
    ----------
    q:
        ``[Tq, D]`` query features.
    q_group_id:
        ``[Tq]`` group id of each query.
    k, v:
        ``[Tk, D]`` / ``[Tv, D]`` key and value features.
    kv_group_id:
        ``[Tk]`` group id of each key/value token.
    n_groups:
        total number of groups; empty query/key sets produce zeros.
    """
    Tq, D = q.shape
    out = torch.zeros_like(q)
    if Tq == 0 or k.shape[0] == 0:
        return out
    assert D % n_heads == 0, "D must be divisible by n_heads"
    head_dim = D // n_heads
    qh = q.view(Tq, n_heads, head_dim)
    kh = k.view(k.shape[0], n_heads, head_dim)
    vh = v.view(v.shape[0], n_heads, head_dim)

    for g in range(n_groups):
        qi = torch.nonzero(q_group_id == g, as_tuple=False).squeeze(1)
        ki = torch.nonzero(kv_group_id == g, as_tuple=False).squeeze(1)
        if qi.numel() == 0 or ki.numel() == 0:
            continue
        qq = qh[qi].permute(1, 0, 2).unsqueeze(0)  # [1, H, Lq, Dh]
        kk = kh[ki].permute(1, 0, 2).unsqueeze(0)  # [1, H, Lk, Dh]
        vv = vh[ki].permute(1, 0, 2).unsqueeze(0)  # [1, H, Lk, Dh]
        o = F.scaled_dot_product_attention(qq, kk, vv, dropout_p=dropout_p)
        out[qi] = o.squeeze(0).permute(1, 0, 2).reshape(-1, D)
    return out


def grouped_cross_attention(
    q: torch.Tensor,
    q_group_id: torch.Tensor,
    k: torch.Tensor,
    kv_group_id: torch.Tensor,
    n_groups: int,
    n_heads: int,
    dropout_p: float = 0.0,
) -> torch.Tensor:
    """Cross-attention with ``k == v`` (kept as a thin alias of
    :func:`grouped_attention` for existing callers/tests)."""
    return grouped_attention(
        q, q_group_id, k, k, kv_group_id, n_groups, n_heads, dropout_p
    )


def grouped_softmax(
    logits: torch.Tensor, group_id: torch.Tensor, n_groups: int
) -> torch.Tensor:
    """Softmax of ``logits`` within each group.

    ``logits`` may be ``[T]`` or ``[T, Q]`` (query dim last).  Returns the same
    shape.
    """
    squeeze = logits.dim() == 1
    if squeeze:
        logits = logits.unsqueeze(-1)
    T, Q = logits.shape
    out = torch.empty_like(logits)
    for q in range(Q):
        z = logits[:, q]
        maxv = torch.full(
            (n_groups,), float("-inf"), device=z.device, dtype=z.dtype
        ).scatter_reduce(0, group_id, z, reduce="amax", include_self=True)
        maxv = torch.where(torch.isfinite(maxv), maxv, torch.zeros_like(maxv))
        e = torch.exp(z - maxv[group_id])
        denom = torch.zeros(n_groups, device=z.device, dtype=z.dtype).index_add_(
            0, group_id, e
        )
        out[:, q] = e / denom[group_id].clamp_min(1e-12)
    return out.squeeze(-1) if squeeze else out
