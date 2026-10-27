import inspect
from dataclasses import replace

import pytest
import torch

from fuse_channels import (
    ChannelEmbedding,
    ChannelFusion,
    ChannelSet,
    ChannelTokenizer,
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
    """Omitting a channel is exactly excluding it from per-location attention."""
    fusion = _fusion(seed=2)
    fusion.eval()
    cs = make_channel_set([4], N, D, seed=5)
    full, _ = fusion(cs)

    # remove channel 2 (all its patches)
    omitted = (cs.sample_idx == 0) & (cs.channel_idx == 2)
    keep = ~omitted

    def reduced(source):
        return ChannelSet(
            tokens=source.tokens[keep],
            sample_idx=source.sample_idx[keep],
            channel_idx=source.channel_idx[keep],
            patch_idx=source.patch_idx[keep],
            channel_id=source.channel_id[keep],
            n_samples=1,
            n_patches=N,
            channel_embed=(
                source.channel_embed[keep]
                if source.channel_embed is not None
                else None
            ),
            channel_group_id=(
                source.channel_group_id[keep]
                if source.channel_group_id is not None
                else None
            ),
        )

    red = reduced(cs)
    out_rem, present_rem = fusion(red)
    assert present_rem.all()
    assert not torch.allclose(full, out_rem)

    # Omitting a channel must equal running attention over the reduced set.
    ref = _dense_fusion_reference(fusion, red)
    assert torch.allclose(out_rem, ref, atol=1e-6, rtol=1e-5)


def test_permutation_invariance():
    """Shuffling channel order leaves the joint embedding unchanged."""
    fusion = _fusion(seed=3)
    fusion.eval()
    cs = make_channel_set([5], N, D)
    permuted = permute_channels(cs)
    a, _ = fusion(cs)
    b, _ = fusion(permuted)
    assert torch.allclose(a, b, atol=1e-5, rtol=1e-4)


def test_identity_changes_routing():
    """Channel identity changes the fused output through bias + skip."""
    fusion = _fusion(seed=4)
    fusion.eval()
    cs = make_channel_set([3], N, D)
    stripped = replace(cs, channel_embed=torch.zeros_like(cs.channel_embed))
    a, _ = fusion(cs)
    b, _ = fusion(stripped)
    assert not torch.allclose(a, b)

    # Changing only the learned channel-embedding table (raw image content
    # fixed) changes what ChannelFusion produces.
    embed = ChannelEmbedding(D, 8)
    tokenizer = ChannelTokenizer(
        image_size=4, patch_size=2, dim=D, channel_embedding=embed
    )
    g = torch.Generator().manual_seed(5)
    images = [torch.randn(3, 1, 4, 4, generator=g)]
    ids = [torch.tensor([1, 4, 6])]
    with torch.no_grad():
        embed.embed.weight.zero_()
        cs_zero = tokenizer(images, ids)
        fused_zero, _ = fusion(cs_zero)
        # non-constant so the shift cannot cancel in LayerNorm
        embed.embed.weight.copy_(
            torch.linspace(-1.0, 1.0, D).expand_as(embed.embed.weight)
        )
        cs_shifted = tokenizer(images, ids)
        fused_shifted, _ = fusion(cs_shifted)
    # patch content is identical; only the identity has changed
    assert torch.allclose(cs_zero.tokens, cs_shifted.tokens, atol=1e-6, rtol=0)
    assert not torch.allclose(fused_zero, fused_shifted)


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
        channel_embed=torch.zeros(C * N, D),
        channel_group_id=torch.arange(C).repeat_interleave(N),
    )
    fused, _ = fusion(cs)
    # The default target architecture has no post-fusion PE; when it is
    # enabled the PE is deliberately position-dependent and must be removed
    # before checking that the pooling itself is position-agnostic.
    centered = fused[0] - (fusion.spatial_pos[0] if fusion.has_spatial_pos else 0)
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


def test_channel_fusion_uses_panopticon_qkv_projections():
    """Panopticon recipe: full Q/K/V projections into a wider attn_dim."""
    torch.manual_seed(0)
    fusion = ChannelFusion(D, n_heads=HEADS, n_patches=N)
    # default is Panopticon's D_attn = 3D
    assert fusion.attn_dim == 3 * D
    assert isinstance(fusion.q_proj, torch.nn.Linear)
    assert isinstance(fusion.k_proj, torch.nn.Linear)
    assert isinstance(fusion.v_proj, torch.nn.Linear)
    assert fusion.q_proj.in_features == D
    assert fusion.q_proj.out_features == fusion.attn_dim
    assert fusion.k_proj.out_features == fusion.attn_dim
    assert fusion.v_proj.out_features == fusion.attn_dim
    assert fusion.proj_out.in_features == fusion.attn_dim
    assert fusion.proj_out.out_features == D

    narrow = ChannelFusion(D, n_heads=HEADS, n_patches=N, attn_dim=D)
    assert narrow.attn_dim == D
    cs = make_channel_set([1, 3, 6, 2], N, D)
    with torch.no_grad():
        got, _ = fusion(cs)
        ref = _dense_fusion_reference(fusion, cs)
        got_narrow, _ = narrow(cs)
        ref_narrow = _dense_fusion_reference(narrow, cs)
    assert torch.allclose(got, ref, atol=1e-5, rtol=1e-4)
    assert torch.allclose(got_narrow, ref_narrow, atol=1e-5, rtol=1e-4)

    with pytest.raises(AssertionError, match="divisible"):
        ChannelFusion(D, n_heads=HEADS, attn_dim=D + 1)


def test_channel_fusion_is_single_query():
    """ChannelFusion is one token per location; multi-query lives in the pooler."""
    assert "n_queries" not in inspect.signature(ChannelFusion.__init__).parameters
    fusion = _fusion(seed=12)
    assert not hasattr(fusion, "n_queries")
    cs = make_channel_set([1, 4], N, D)
    fused, present = fusion(cs)
    assert fused.shape == (2, N, D)
    assert present.all()

    # The descriptor pooler still supports the k>1 research ablation.
    torch.manual_seed(0)
    pool = MultiheadAttentionPooling(
        D, n_heads=HEADS, n_queries=3, post_sab=True, readout="flat"
    )
    x = torch.randn(6, D)
    group_id = torch.tensor([0, 0, 0, 1, 1, 1])
    assert pool(x, group_id, 2).shape == (2, 3 * D)


def test_identity_gated_sum():
    net = IdentityGatedSum(D, n_patches=N)
    net.eval()
    cs = make_channel_set([2, 5, 9], N, D)
    out, present = net(cs)
    assert out.shape == (3, N, D)
    assert torch.isfinite(out).all()
    assert present.all()

    # Routing follows channel_embed: identical tokens with a different
    # embedding produce a different weighted sum.
    shifted = replace(cs, channel_embed=cs.channel_embed + 0.5)
    out_shifted, _ = net(shifted)
    assert not torch.allclose(out, out_shifted)


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
        values = net.out_norm(net.value_proj(net.content_norm(shuffled.tokens)))
        loc = shuffled.location_group_ids()
        ref = torch.zeros(cs.n_samples, N, D)
        for g in torch.unique(loc):
            idx = (loc == g).nonzero(as_tuple=True)[0]
            w = torch.softmax(logits[idx], dim=0)
            content = (w.unsqueeze(-1) * values[idx]).sum(0)
            ident = (w.unsqueeze(-1) * shuffled.channel_embed[idx]).sum(0)
            ref[shuffled.sample_idx[idx[0]], shuffled.patch_idx[idx[0]]] = (
                content + net.identity_skip(ident)
            )

    expected = ref + (net.spatial_pos if net.has_spatial_pos else 0)
    assert torch.allclose(out, expected, atol=1e-6)


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


def test_fusion_rejects_patch_count_mismatch():
    """Table-based fusion PE validates the patch count it was built for."""
    cs = make_channel_set([2], N + 1, D)
    fusion = ChannelFusion(D, n_heads=HEADS, n_patches=N, pos_embed="learned")
    with pytest.raises(ValueError, match="n_patches"):
        fusion(cs)
    gated = IdentityGatedSum(D, n_patches=N, pos_embed="learned")
    with pytest.raises(ValueError, match="n_patches"):
        gated(cs)


def test_no_pad_artifacts():
    """Fusion is a real-token [B, N, D] tensor with an exact presence mask."""
    fusion = _fusion(seed=6)
    fusion.eval()
    cs = make_channel_set([3, 10], N, D)

    # Drop every channel at one location so the mask is not trivially all-True.
    keep = ~((cs.sample_idx == 0) & (cs.patch_idx == 2))
    sparse = ChannelSet(
        tokens=cs.tokens[keep],
        sample_idx=cs.sample_idx[keep],
        channel_idx=cs.channel_idx[keep],
        patch_idx=cs.patch_idx[keep],
        channel_id=cs.channel_id[keep],
        n_samples=cs.n_samples,
        n_patches=N,
        channel_embed=cs.channel_embed[keep],
        channel_group_id=cs.channel_group_id[keep],
    )
    fused, present = fusion(sparse)

    assert fused.shape == (2, N, D)  # spatial axis, never a padded channel axis
    assert torch.isfinite(fused).all()
    assert not torch.isneginf(fused).any()
    assert not torch.isnan(fused).any()

    expected = torch.ones(2, N, dtype=torch.bool)
    expected[0, 2] = False
    assert present.dtype == torch.bool
    assert torch.equal(present, expected)


def test_post_fusion_spatial_pos_is_legacy_opt_in():
    """The default has no fusion PE; the learned table remains an ablation."""
    default = _fusion(seed=8)
    assert default.pos_embed == "none"
    assert not default.has_spatial_pos
    assert not hasattr(default, "spatial_pos")
    assert not hasattr(IdentityGatedSum(D), "spatial_pos")

    fusion = ChannelFusion(D, n_heads=HEADS, n_patches=N, pos_embed="learned")
    assert tuple(fusion.spatial_pos.shape) == (1, N, D)
    cs = make_channel_set([2, 3], N, D)
    with torch.no_grad():
        fused, _ = fusion(cs)
        fusion.spatial_pos.zero_()
        no_pe, _ = fusion(cs)
    assert not torch.allclose(fused, no_pe)

    gated = IdentityGatedSum(D, n_patches=N, pos_embed="learned")
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

def test_sincos_pos_embed_is_fixed_and_padding_free_equivalent():
    """pos_embed="sincos": no parameters, deterministic, same math as dense ref."""
    fusion = ChannelFusion(D, n_heads=HEADS, n_patches=N, pos_embed="sincos")
    assert fusion.pos_embed == "sincos"
    assert "spatial_pos" not in dict(fusion.named_parameters())
    assert "spatial_pos" in dict(fusion.named_buffers())

    cs = make_channel_set([1, 3, 6, 2], N, D)
    with torch.no_grad():
        fused, _ = fusion(cs)
        again, _ = fusion(cs)
        ref = _dense_fusion_reference(fusion, cs)
    assert torch.equal(fused, again)
    assert torch.allclose(fused, ref, atol=1e-5, rtol=1e-4)


def test_sincos_pos_embed_keeps_invariants():
    """Cardinality and permutation invariance hold with the fixed sincos map."""
    fusion = ChannelFusion(D, n_heads=HEADS, n_patches=N, pos_embed="sincos")
    fusion.eval()
    cs = make_channel_set([2, 5], N, D)
    with torch.no_grad():
        fused, present = fusion(cs)
        permuted, _ = fusion(permute_channels(cs))
    assert fused.shape == (2, N, D)
    assert present.all()
    assert torch.allclose(fused, permuted, atol=1e-5, rtol=1e-4)


def test_identity_gated_sincos_pos_embed():
    gated = IdentityGatedSum(D, n_patches=N, pos_embed="sincos")
    assert gated.pos_embed == "sincos"
    assert "spatial_pos" not in dict(gated.named_parameters())
    cs = make_channel_set([2, 3], N, D)
    out, _ = gated(cs)
    assert out.shape == (2, N, D)
    assert torch.isfinite(out).all()
    with torch.no_grad():
        before = gated.spatial_pos.clone()
        gated.spatial_pos.zero_()
        no_pe, _ = gated(cs)
    assert not torch.allclose(out, no_pe)


def test_pos_embed_option_validation():
    with pytest.raises(ValueError, match="unknown pos_embed"):
        ChannelFusion(D, n_heads=HEADS, n_patches=N, pos_embed="rope")
    # sincos needs a square lattice unless grid_size is given explicitly
    with pytest.raises(ValueError, match="not a square"):
        ChannelFusion(D, n_heads=HEADS, n_patches=6, pos_embed="sincos")
    nonsquare = ChannelFusion(
        D, n_heads=HEADS, n_patches=6, pos_embed="sincos", grid_size=(2, 3)
    )
    assert tuple(nonsquare.spatial_pos.shape) == (1, 6, D)
