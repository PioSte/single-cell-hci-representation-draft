import torch
import torch.nn.functional as F

from fuse_channels import (
    grouped_cross_attention,
    grouped_pool_attention,
    grouped_self_attention,
    segment_softmax,
)


def test_grouped_self_attention_matches_dense():
    torch.manual_seed(0)
    T, H, Dh = 11, 2, 4
    D = H * Dh
    x = torch.randn(T, D)
    seg = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2, 2, 3, 4])
    n_seg = 5

    out = grouped_self_attention(x, seg, n_seg, H)

    for s in range(n_seg):
        idx = (seg == s).nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            continue
        xs = x[idx].view(1, idx.numel(), H, Dh).permute(0, 2, 1, 3)
        ref = F.scaled_dot_product_attention(xs, xs, xs)
        ref = ref.permute(0, 2, 1, 3).reshape(-1, D)
        assert torch.allclose(out[idx], ref, atol=1e-6)


def test_grouped_pool_attention_basic():
    torch.manual_seed(0)
    H, Dh = 2, 4
    T = 6
    k = torch.randn(T, H, Dh)
    seg = torch.tensor([0, 0, 0, 1, 1, 1])
    q = torch.randn(2, 1, H, Dh)  # one query per segment
    out = grouped_pool_attention(q, k, k, seg, 2)
    assert out.shape == (2, 1, H, Dh)
    assert torch.isfinite(out).all()


def test_grouped_cross_attention_basic():
    torch.manual_seed(0)
    q = torch.randn(5, 8)
    q_seg = torch.tensor([0, 0, 0, 1, 1])
    k = torch.randn(6, 8)
    kv_seg = torch.tensor([0, 0, 0, 0, 1, 1])
    out = grouped_cross_attention(q, q_seg, k, kv_seg, 2, 2)
    assert out.shape == q.shape
    assert torch.isfinite(out).all()


def test_segment_softmax():
    logits = torch.tensor([1.0, 2.0, 3.0, 10.0, 0.0])
    seg = torch.tensor([0, 0, 0, 1, 1])
    w = segment_softmax(logits, seg, 2)
    assert torch.allclose(w[:3].sum(), torch.tensor(1.0), atol=1e-6)
    assert torch.allclose(w[3:].sum(), torch.tensor(1.0), atol=1e-6)
    # within segment 0, ordering preserved
    assert w[0] < w[1] < w[2]
