import torch
import torch.nn.functional as F

from fuse_channels import (
    grouped_attention,
    grouped_cross_attention,
    grouped_pool_attention,
    grouped_projected_self_attention,
    grouped_self_attention,
    grouped_softmax,
)


def test_grouped_self_attention_matches_dense():
    torch.manual_seed(0)
    T, H, Dh = 11, 2, 4
    D = H * Dh
    x = torch.randn(T, D)
    group_id = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2, 2, 3, 4])
    n_groups = 5

    out = grouped_self_attention(x, group_id, n_groups, H)

    for g in range(n_groups):
        idx = (group_id == g).nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            continue
        xs = x[idx].view(1, idx.numel(), H, Dh).permute(0, 2, 1, 3)
        ref = F.scaled_dot_product_attention(xs, xs, xs)
        ref = ref.permute(0, 2, 1, 3).reshape(-1, D)
        assert torch.allclose(out[idx], ref, atol=1e-6)


def test_grouped_projected_self_attention_matches_dense():
    """Bucketed projected q/k/v self-attention == per-group dense SDPA."""
    torch.manual_seed(0)
    H, Dh = 2, 4
    D = H * Dh
    # unsorted groups with mixed lengths: 3, 2, 3, 3
    group_id = torch.tensor([1, 0, 2, 0, 2, 1, 1, 3, 3, 3, 2])
    n_groups = 4
    q = torch.randn(len(group_id), D)
    k = torch.randn(len(group_id), D)
    v = torch.randn(len(group_id), D)

    out = grouped_projected_self_attention(q, k, v, group_id, n_groups, H)

    assert out.shape == q.shape
    for g in range(n_groups):
        idx = (group_id == g).nonzero(as_tuple=True)[0]
        qq = q[idx].view(1, idx.numel(), H, Dh).permute(0, 2, 1, 3)
        kk = k[idx].view(1, idx.numel(), H, Dh).permute(0, 2, 1, 3)
        vv = v[idx].view(1, idx.numel(), H, Dh).permute(0, 2, 1, 3)
        ref = F.scaled_dot_product_attention(qq, kk, vv)
        ref = ref.permute(0, 2, 1, 3).reshape(-1, D)
        assert torch.allclose(out[idx], ref, atol=1e-6)


def test_grouped_pool_attention_basic():
    torch.manual_seed(0)
    H, Dh = 2, 4
    T = 6
    k = torch.randn(T, H, Dh)
    group_id = torch.tensor([0, 0, 0, 1, 1, 1])
    q = torch.randn(2, 1, H, Dh)  # one query per group
    out = grouped_pool_attention(q, k, k, group_id, 2)
    assert out.shape == (2, 1, H, Dh)
    assert torch.isfinite(out).all()


def test_grouped_cross_attention_basic():
    torch.manual_seed(0)
    q = torch.randn(5, 8)
    q_group_id = torch.tensor([0, 0, 0, 1, 1])
    k = torch.randn(6, 8)
    kv_group_id = torch.tensor([0, 0, 0, 0, 1, 1])
    out = grouped_cross_attention(q, q_group_id, k, kv_group_id, 2, 2)
    assert out.shape == q.shape
    assert torch.isfinite(out).all()


def test_grouped_attention_arbitrary_qkv_matches_dense():
    """Independent q/k/v tensors attent within their group, like dense SDPA."""
    torch.manual_seed(0)
    H, Dh = 2, 4
    D = H * Dh
    q = torch.randn(4, D)
    q_group_id = torch.tensor([0, 0, 0, 1])
    k = torch.randn(5, D)
    v = torch.randn(5, D)
    kv_group_id = torch.tensor([0, 0, 0, 1, 1])

    out = grouped_attention(q, q_group_id, k, v, kv_group_id, 2, H)

    for g in range(2):
        qi = (q_group_id == g).nonzero(as_tuple=True)[0]
        ki = (kv_group_id == g).nonzero(as_tuple=True)[0]
        qq = q[qi].view(1, qi.numel(), H, Dh).permute(0, 2, 1, 3)
        kk = k[ki].view(1, ki.numel(), H, Dh).permute(0, 2, 1, 3)
        vv = v[ki].view(1, ki.numel(), H, Dh).permute(0, 2, 1, 3)
        ref = F.scaled_dot_product_attention(qq, kk, vv)
        ref = ref.permute(0, 2, 1, 3).reshape(-1, D)
        assert torch.allclose(out[qi], ref, atol=1e-6)


def test_grouped_softmax():
    logits = torch.tensor([1.0, 2.0, 3.0, 10.0, 0.0])
    group_id = torch.tensor([0, 0, 0, 1, 1])
    w = grouped_softmax(logits, group_id, 2)
    assert torch.allclose(w[:3].sum(), torch.tensor(1.0), atol=1e-6)
    assert torch.allclose(w[3:].sum(), torch.tensor(1.0), atol=1e-6)
    # within group 0, ordering preserved
    assert w[0] < w[1] < w[2]
