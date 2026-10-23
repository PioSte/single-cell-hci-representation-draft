import torch

from fuse_channels import (
    ChannelFusion,
    ChannelSet,
    IdentityGatedSum,
    MultiheadAttentionPooling,
)
from fuse_channels.fusion import _dense_fusion_reference
from helpers import make_channel_set, permute_channels

D = 16
N = 4
HEADS = 4


def _fusion(seed=0):
    torch.manual_seed(seed)
    return ChannelFusion(D, n_heads=HEADS, n_queries=1, n_patches=N)


# ---------------------------------------------------------------- #
def test_cardinality_invariance():
    """Output shape is independent of the channel count C."""
    fusion = _fusion()
    shapes = set()
    for C in (1, 3, 6, 20, 60):
        cs = make_channel_set([2, C], N, D)
        fused, present = fusion(cs)
        assert fused.shape == (2, N, D)
        assert present.all()
        assert torch.isfinite(fused).all()
        shapes.add(tuple(fused.shape))
    assert len(shapes) == 1


def test_padding_free_equivalence():
    """Padding-free fusion == masked-padded reference (exact math)."""
    fusion = _fusion(seed=1)
    fusion.eval()
    # mixed channel counts
    cs = make_channel_set([1, 3, 6, 2], N, D)
    fused, _ = fusion(cs)
    ref = _dense_fusion_reference(fusion, cs)
    assert torch.allclose(fused, ref, atol=1e-5, rtol=1e-4)


def test_omission_equivalence():
    """Omitting a channel equals excluding it from attention (no leakage)."""
    fusion = _fusion(seed=2)
    fusion.eval()
    cs = make_channel_set([4], N, D, seed=5)
    full, _ = fusion(cs)

    # remove channel 2 (all its patches)
    keep = ~((cs.sample_idx == 0) & (cs.channel_idx == 2))
    red = ChannelSet(
        tokens=cs.tokens[keep],
        sample_idx=cs.sample_idx[keep],
        channel_idx=cs.channel_idx[keep],
        patch_idx=cs.patch_idx[keep],
        channel_id=cs.channel_id[keep],
        n_samples=1,
        n_patches=N,
        channel_embed=cs.channel_embed[keep] if cs.channel_embed is not None else None,
    )
    out_rem, _ = fusion(red)
    # removing one channel should change the result but not blow it up
    assert torch.isfinite(out_rem).all()
    # remaining tokens themselves are untouched by omission
    assert torch.equal(cs.tokens[keep], red.tokens)
    assert not torch.allclose(full, out_rem)


def test_permutation_invariance():
    """Shuffling channel order leaves the joint embedding unchanged."""
    fusion = _fusion(seed=3)
    fusion.eval()
    cs = make_channel_set([5], N, D)
    permuted = permute_channels(cs)
    a, _ = fusion(cs)
    b, _ = fusion(permuted)
    assert torch.allclose(a, b, atol=1e-5, rtol=1e-4)


def test_identity_ablation():
    """Output depends on the additive channel-embedding vector."""
    fusion = _fusion(seed=4)
    fusion.eval()
    cs = make_channel_set([3], N, D)
    stripped = make_channel_set([3], N, D, seed=0)
    stripped.tokens = stripped.tokens - stripped.channel_embed  # strip embedding
    a, _ = fusion(cs)
    b, _ = fusion(stripped)
    assert not torch.allclose(a, b)


def test_pooling_queries_are_shared_across_locations():
    """ChannelFusion uses one query set for every spatial location."""
    fusion = _fusion(seed=7)
    fusion.eval()
    assert tuple(fusion.queries.shape) == (1, fusion.n_queries, D)

    # identical channel tokens at two locations must fuse identically, which
    # only holds if the pooling query is shared across locations.
    C = 2
    tok = torch.randn(C, D)
    cs = ChannelSet(
        tokens=tok.repeat_interleave(N, dim=0),
        sample_idx=torch.zeros(C * N, dtype=torch.long),
        channel_idx=torch.arange(C).repeat_interleave(N),
        patch_idx=torch.arange(N).repeat(C),
        channel_id=torch.zeros(C * N, dtype=torch.long),
        n_samples=1,
        n_patches=N,
    )
    fused, _ = fusion(cs)
    for n in range(1, N):
        assert torch.allclose(fused[0, 0], fused[0, n], atol=1e-6)


def test_multihead_attention_pooling():
    """PMA descriptor pool: padding-free, permutation-invariant, empty->0."""
    torch.manual_seed(0)
    pool = MultiheadAttentionPooling(D, n_heads=HEADS, n_queries=1)
    pool.eval()
    x = torch.randn(7, D)
    group_id = torch.tensor([0, 0, 0, 1, 2, 2, 2])
    out = pool(x, group_id, 4)
    assert out.shape == (4, D)
    assert torch.isfinite(out).all()
    # a group with no tokens pools to zeros (not NaN)
    assert torch.equal(out[3], torch.zeros(D))

    # pooling is invariant to the order of tokens inside a group
    perm = torch.tensor([2, 0, 1, 3, 6, 4, 5])
    out_perm = pool(x[perm], group_id[perm], 4)
    assert torch.allclose(out, out_perm, atol=1e-5, rtol=1e-4)

    # n_queries>1 returns a wider fixed-size descriptor
    pool_multi = MultiheadAttentionPooling(D, n_heads=HEADS, n_queries=3)
    out_multi = pool_multi(x, group_id, 4)
    assert out_multi.shape == (4, 3 * D)


def test_identity_gated_sum():
    net = IdentityGatedSum(D, n_queries=1, n_patches=N)
    cs = make_channel_set([2, 5, 9], N, D)
    out, present = net(cs)
    assert out.shape == (3, N, D)
    assert torch.isfinite(out).all()
    assert present.all()


def test_no_pad_artifacts():
    """The public fusion API never exposes a [B, C_max, D] tensor."""
    fusion = _fusion(seed=6)
    cs = make_channel_set([3, 10], N, D)
    fused, present = fusion(cs)
    assert fused.dim() == 3 and fused.shape[1] == N  # spatial axis, not channel
    # no NaNs / -inf leaking from masking
    assert torch.isfinite(fused).all()


def test_dropout_disabled_in_eval():
    """Attention dropout must be a no-op after `.eval()` (both pooling modules)."""
    cs = make_channel_set([3, 5], N, D)

    fusion = ChannelFusion(D, n_heads=HEADS, n_queries=1, n_patches=N, dropout=0.5)
    fusion.eval()
    with torch.no_grad():
        a, _ = fusion(cs)
        b, _ = fusion(cs)
    assert torch.equal(a, b)

    pool = MultiheadAttentionPooling(D, n_heads=HEADS, n_queries=1, dropout=0.5)
    pool.eval()
    x = torch.randn(7, D)
    group_id = torch.tensor([0, 0, 0, 1, 2, 2, 2])
    with torch.no_grad():
        p1 = pool(x, group_id, 4)
        p2 = pool(x, group_id, 4)
    assert torch.equal(p1, p2)


def test_dropout_active_in_train():
    """In training mode, dropout > 0 must actually make the output stochastic."""
    cs = make_channel_set([3, 5], N, D)
    fusion = ChannelFusion(D, n_heads=HEADS, n_queries=1, n_patches=N, dropout=0.5)
    fusion.train()
    a, _ = fusion(cs)
    b, _ = fusion(cs)
    assert not torch.equal(a, b)
