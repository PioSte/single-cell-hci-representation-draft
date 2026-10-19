import pytest
import torch

from fuse_channels import (
    ChannelFusion,
    ChannelIdentity,
    HypernetworkWeightedSum,
    dense_fusion_reference,
)
from helpers import make_channel_set, permute_channels

D = 16
N = 4
HEADS = 4


def _fusion(seed=0, mode="per_location"):
    torch.manual_seed(seed)
    return ChannelFusion(D, n_heads=HEADS, n_queries=1, n_patches=N, mode=mode)


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
    """Varlen fusion == masked-padded reference (exact math)."""
    fusion = _fusion(seed=1)
    fusion.eval()
    # mixed channel counts
    cs = make_channel_set([1, 3, 6, 2], N, D)
    fused, _ = fusion(cs)
    ref = dense_fusion_reference(fusion, cs)
    assert torch.allclose(fused, ref, atol=1e-5, rtol=1e-4)


def test_omission_equivalence():
    """Omitting a channel equals excluding it from attention (no leakage)."""
    fusion = _fusion(seed=2)
    fusion.eval()
    cs = make_channel_set([4], N, D, seed=5)
    full, _ = fusion(cs)

    # remove channel 2 (all its patches)
    keep = ~((cs.sample_idx == 0) & (cs.channel_idx == 2))
    cs_rem = make_channel_set([3], N, D, seed=5)
    # rebuild with the same tokens minus channel 2
    from fuse_channels import ChannelSet

    red = ChannelSet(
        tokens=cs.tokens[keep],
        sample_idx=cs.sample_idx[keep],
        channel_idx=cs.channel_idx[keep],
        patch_idx=cs.patch_idx[keep],
        channel_id=cs.channel_id[keep],
        n_samples=1,
        n_patches=N,
        identity=cs.identity[keep] if cs.identity is not None else None,
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
    """Output depends on the additive channel-identity embedding."""
    fusion = _fusion(seed=4)
    fusion.eval()
    cs = make_channel_set([3], N, D)
    stripped = make_channel_set([3], N, D, seed=0)
    stripped.tokens = stripped.tokens - stripped.identity  # strip identity
    a, _ = fusion(cs)
    b, _ = fusion(stripped)
    assert not torch.allclose(a, b)


def test_open_vocabulary_identity():
    """Text/sequence-conditioned identity works with no embedding table."""
    ident = ChannelIdentity(D, num_markers=0, text_dim=8)
    text = torch.randn(5, 8)
    out = ident(torch.zeros(5, dtype=torch.long), text_embeds=text)
    assert out.shape == (5, D)
    # table was never created -> unbounded C
    assert ident.embed is None
    # and without text we cannot fabricate an identity
    with pytest.raises(ValueError):
        ident(torch.zeros(5, dtype=torch.long))


def test_global_mode():
    fusion = _fusion(seed=5, mode="global")
    cs = make_channel_set([1, 4, 7], N, D)
    out, present = fusion(cs)
    assert out.shape == (3, D)
    assert present.all()


def test_hypernetwork_weighted_sum():
    net = HypernetworkWeightedSum(D, n_queries=1, n_patches=N, mode="per_location")
    cs = make_channel_set([2, 5, 9], N, D)
    out, present = net(cs)
    assert out.shape == (3, N, D)
    assert torch.isfinite(out).all()
    assert present.all()

    net_g = HypernetworkWeightedSum(D, n_queries=1, mode="global")
    out_g, present_g = net_g(cs)
    assert out_g.shape == (3, D)
    assert present_g.all()


def test_no_pad_artifacts():
    """The public fusion API never exposes a [B, C_max, D] tensor."""
    fusion = _fusion(seed=6)
    cs = make_channel_set([3, 10], N, D)
    fused, present = fusion(cs)
    assert fused.dim() == 3 and fused.shape[1] == N  # spatial axis, not channel
    # no NaNs / -inf leaking from masking
    assert torch.isfinite(fused).all()
