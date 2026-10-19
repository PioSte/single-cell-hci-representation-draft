"""Padding-free (varlen) attention primitives.

Everything in this module operates on *flattened, real* tokens plus integer
segment ids, so no ``[B, C_max, D]`` padded tensor and no attention mask is ever
materialised.  Attention is always restricted to a single segment, which is
exactly what per-sample / per-channel-set attention means.

Layout conventions
------------------
``seg[i]`` is the segment id of token ``i``.  Segments are independent; tokens
from different segments never interact.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _lengths_and_offsets(seg: torch.Tensor, n_segments: int):
    lengths = torch.bincount(seg, minlength=n_segments)
    offsets = torch.zeros(n_segments + 1, dtype=torch.long, device=seg.device)
    offsets[1:] = torch.cumsum(lengths, dim=0)
    return lengths, offsets


def grouped_self_attention(
    x: torch.Tensor,
    seg: torch.Tensor,
    n_segments: int,
    n_heads: int,
    dropout_p: float = 0.0,
) -> torch.Tensor:
    """Full self-attention within each segment, no padding.

    Parameters
    ----------
    x:
        ``[T, D]`` token features.
    seg:
        ``[T]`` segment id per token (values in ``[0, n_segments)``).
    """
    T, D = x.shape
    out = torch.zeros_like(x)
    if T == 0:
        return out
    assert D % n_heads == 0
    head_dim = D // n_heads

    lengths, offsets = _lengths_and_offsets(seg, n_segments)
    order = torch.argsort(seg, stable=True)
    x_sorted = x[order]
    out_sorted = torch.empty_like(x_sorted)
    ar = None

    for L in torch.unique(lengths).tolist():
        if L == 0:
            continue
        seg_ids = torch.nonzero(lengths == L, as_tuple=False).squeeze(1)
        offs = offsets[seg_ids]
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


def grouped_pool_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kv_seg: torch.Tensor,
    n_segments: int,
    dropout_p: float = 0.0,
) -> torch.Tensor:
    """Cross-attention with one query *group* per segment (PMA-style pooling).

    Parameters
    ----------
    q:
        ``[n_segments, n_queries, n_heads, head_dim]`` -- the learned query(s)
        for every segment.  Segments with no key tokens produce zeros.
    k, v:
        ``[T, n_heads, head_dim]`` flattened key/value tokens.
    kv_seg:
        ``[T]`` segment id of each key/value token.
    """
    n_seg, n_q, n_heads, head_dim = q.shape
    out = torch.zeros_like(q)
    if kv_seg.numel() == 0:
        return out

    lengths, offsets = _lengths_and_offsets(kv_seg, n_segments)
    order = torch.argsort(kv_seg, stable=True)
    k_sorted = k[order]
    v_sorted = v[order]

    for L in torch.unique(lengths).tolist():
        if L == 0:
            continue
        seg_ids = torch.nonzero(lengths == L, as_tuple=False).squeeze(1)
        offs = offsets[seg_ids]
        ar = torch.arange(L, device=k.device)
        idx = offs[:, None] + ar[None, :]  # [M, L]
        kk = k_sorted[idx].permute(0, 2, 1, 3)  # [M, H, L, Dh]
        vv = v_sorted[idx].permute(0, 2, 1, 3)
        qq = q[seg_ids].permute(0, 2, 1, 3)  # [M, H, n_q, Dh]
        o = F.scaled_dot_product_attention(qq, kk, vv, dropout_p=dropout_p)
        out[seg_ids] = o.permute(0, 2, 1, 3)
    return out


def grouped_cross_attention(
    q: torch.Tensor,
    q_seg: torch.Tensor,
    k: torch.Tensor,
    kv_seg: torch.Tensor,
    n_segments: int,
    n_heads: int,
    dropout_p: float = 0.0,
) -> torch.Tensor:
    """Arbitrary-query cross-attention within segments (loops over segments).

    Used by the decoder, where the number of segments is the batch size (small)
    while both the query and key sets are variable length.
    """
    Tq, D = q.shape
    out = torch.zeros_like(q)
    if Tq == 0 or k.shape[0] == 0:
        return out
    head_dim = D // n_heads
    qh = q.view(Tq, n_heads, head_dim)
    kh = k.view(k.shape[0], n_heads, head_dim)

    for s in range(n_segments):
        qi = torch.nonzero(q_seg == s, as_tuple=False).squeeze(1)
        ki = torch.nonzero(kv_seg == s, as_tuple=False).squeeze(1)
        if qi.numel() == 0 or ki.numel() == 0:
            continue
        qq = qh[qi].permute(1, 0, 2).unsqueeze(0)  # [1, H, Lq, Dh]
        kk = kh[ki].permute(1, 0, 2).unsqueeze(0)  # [1, H, Lk, Dh]
        o = F.scaled_dot_product_attention(qq, kk, kk, dropout_p=dropout_p)
        out[qi] = o.squeeze(0).permute(1, 0, 2).reshape(-1, D)
    return out


def segment_softmax(
    logits: torch.Tensor, seg: torch.Tensor, n_segments: int
) -> torch.Tensor:
    """Softmax of ``logits`` within each segment.

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
            (n_segments,), float("-inf"), device=z.device, dtype=z.dtype
        ).scatter_reduce(0, seg, z, reduce="amax", include_self=True)
        maxv = torch.where(torch.isfinite(maxv), maxv, torch.zeros_like(maxv))
        e = torch.exp(z - maxv[seg])
        denom = torch.zeros(n_segments, device=z.device, dtype=z.dtype).index_add_(
            0, seg, e
        )
        out[:, q] = e / denom[seg].clamp_min(1e-12)
    return out.squeeze(-1) if squeeze else out
