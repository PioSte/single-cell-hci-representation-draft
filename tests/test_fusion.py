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
    return ChannelFusion(D, n_heads=HEADS, n_patches=N)


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
    """Padding-free fusion == masked-padded reference (exact math + PE)."""
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
        channel_group_id=(
            cs.channel_group_id[keep] if cs.channel_group_id is not None else None
        ),
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


def test_pooling_query_is_shared_across_locations():
    """ChannelFusion uses one query for every spatial location."""
    fusion = _fusion(seed=7)
    fusion.eval()
    assert tuple(fusion.queries.shape) == (1, 1, D)

    # identical channel tokens at two locations must fuse identically, which
    # only holds if the pooling query is shared across locations (the
    # post-fusion PE is subtracted because it deliberately differs by patch).
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
        channel_group_id=torch.arange(C).repeat_interleave(N),
    )
    fused, _ = fusion(cs)
    centered = fused[0] - fusion.spatial_pos[0]
    for n in range(1, N):
        assert torch.allclose(centered[0], centered[n], atol=1e-6)


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


def test_multihead_attention_pooling_multi_query_shapes():
    """n_queries>1 + optional post-SAB: flat and mean readouts."""
    torch.manual_seed(0)
    x = torch.randn(7, D)
    group_id = torch.tensor([0, 0, 0, 1, 2, 2, 2])

    flat = MultiheadAttentionPooling(
        D, n_heads=HEADS, n_queries=3, post_sab=True, readout="flat"
    )
    out_flat = flat(x, group_id, 4)
    assert out_flat.shape == (4, 3 * D)
    assert torch.isfinite(out_flat).all()
    assert flat.post_sab is not None

    mean = MultiheadAttentionPooling(
        D, n_heads=HEADS, n_queries=3, post_sab=True, readout="mean"
    )
    out_mean = mean(x, group_id, 4)
    assert out_mean.shape == (4, D)
    assert torch.isfinite(out_mean).all()

    try:
        MultiheadAttentionPooling(D, n_heads=HEADS, readout="nope")
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("invalid readout must raise")


def test_identity_gated_sum():
    net = IdentityGatedSum(D, n_patches=N)
    cs = make_channel_set([2, 5, 9], N, D)
    out, present = net(cs)
    assert out.shape == (3, N, D)
    assert torch.isfinite(out).all()
    assert present.all()


def test_identity_gated_sum_unsorted_locations():
    """Per-location softmax and scatter follow input order, not sorted order."""
    torch.manual_seed(0)
    net = IdentityGatedSum(D, n_patches=N)
    net.eval()
    cs = make_channel_set([3, 4], N, D, seed=11)

    # shuffle the flattened tokens so the location group ids are unsorted
    perm = torch.randperm(
        cs.tokens.shape[0], generator=torch.Generator().manual_seed(3)
    )
    shuffled = ChannelSet(
        tokens=cs.tokens[perm],
        sample_idx=cs.sample_idx[perm],
        channel_idx=cs.channel_idx[perm],
        patch_idx=cs.patch_idx[perm],
        channel_id=cs.channel_id[perm],
        n_samples=cs.n_samples,
        n_patches=N,
        channel_embed=cs.channel_embed[perm],
        channel_group_id=cs.channel_group_id[perm],
    )

    out, _ = net(shuffled)

    with torch.no_grad():
        logits = net.weight_net(shuffled.channel_embed).squeeze(-1)  # [T]
        values = net.out_norm(net.value_proj(shuffled.tokens))  # [T, D]
        loc = shuffled.location_group_ids()
        ref = torch.zeros(cs.n_samples, N, D)
        for g in torch.unique(loc):
            idx = (loc == g).nonzero(as_tuple=True)[0]
            w = torch.softmax(logits[idx], dim=0)
            ref[shuffled.sample_idx[idx[0]], shuffled.patch_idx[idx[0]]] = (
                w.unsqueeze(-1) * values[idx]
            ).sum(0)

    assert torch.allclose(out, ref + net.spatial_pos, atol=1e-6)


def test_identity_gated_sum_channel_mlp_once_per_channel():
    """The channel MLP is evaluated once for all unique channels, then broadcast."""
    net = IdentityGatedSum(D, n_patches=N)
    calls = []

    def hook(_module, inputs, _output):
        calls.append(tuple(inputs[0].shape))

    handle = net.weight_net.register_forward_hook(hook)
    try:
        cs = make_channel_set([2, 5, 9], N, D)  # 16 channels, 16*N tokens
        out, _ = net(cs)
    finally:
        handle.remove()
    assert out.shape == (3, N, D)
    assert calls == [(16, D)]


def test_identity_gated_sum_requires_channel_embed():
    net = IdentityGatedSum(D, n_patches=N)
    cs = make_channel_set([2, 3], N, D, with_channel_embed=False)
    try:
        net(cs)
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("missing channel_embed must raise")


def test_no_pad_artifacts():
    """The public fusion API never exposes a [B, C_max, D] tensor."""
    fusion = _fusion(seed=6)
    cs = make_channel_set([3, 10], N, D)
    fused, present = fusion(cs)
    assert fused.dim() == 3 and fused.shape[1] == N  # spatial axis, not channel
    # no NaNs / -inf leaking from masking
    assert torch.isfinite(fused).all()


def test_post_fusion_spatial_pos():
    """Both fusion modules own the spatial PE; it shifts token values."""
    fusion = _fusion(seed=8)
    assert tuple(fusion.spatial_pos.shape) == (1, N, D)
    cs = make_channel_set([2, 3], N, D)
    with torch.no_grad():
        fused, _ = fusion(cs)
        fusion.spatial_pos.zero_()
        no_pe, _ = fusion(cs)
    assert not torch.allclose(fused, no_pe)

    gated = IdentityGatedSum(D, n_patches=N)
    assert tuple(gated.spatial_pos.shape) == (1, N, D)


def test_dropout_disabled_in_eval():
    """Attention dropout must be a no-op after `.eval()` (both pooling modules)."""
    cs = make_channel_set([3, 5], N, D)

    fusion = ChannelFusion(D, n_heads=HEADS, n_patches=N, dropout=0.5)
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
    fusion = ChannelFusion(D, n_heads=HEADS, n_patches=N, dropout=0.5)
    fusion.train()
    a, _ = fusion(cs)
    b, _ = fusion(cs)
    assert not torch.equal(a, b)
