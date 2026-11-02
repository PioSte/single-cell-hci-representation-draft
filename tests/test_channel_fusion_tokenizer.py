import pytest
import torch

from fuse_channels import (
    ChannelEmbedding,
    ChannelFusion,
    ChannelFusionMAE,
    ChannelFusionTokenizer,
    ChannelTokenizer,
    IdentityGatedSum,
)

D = 32
HEADS = 4
IMAGE = 8
P = 4


def _batch(seed=0, C_list=(2, 4, 3), image_size=IMAGE):
    g = torch.Generator().manual_seed(seed)
    images = [torch.randn(C, 1, image_size, image_size, generator=g) for C in C_list]
    ids = [torch.randint(0, 16, (C,), generator=g) for C in C_list]
    return images, ids


def _cft(seed=0, fusion="attention", patch_sizes=(2, 4)):
    torch.manual_seed(seed)
    embedding = ChannelEmbedding(D, num_channel_ids=16)
    return ChannelFusionTokenizer.build(
        IMAGE,
        P,
        D,
        embedding,
        n_heads=HEADS,
        patch_sizes=patch_sizes,
        fusion=fusion,
    )


# ---------------------------------------------------------------- #
def test_one_row_per_present_location():
    """Fixed width D; the row count is the present-location count, never C."""
    cft = _cft()
    images, ids = _batch()
    fused, present, _cs = cft(images, ids)
    assert fused.shape == (present.sum().item(), D)
    assert present.shape == (len(images), (IMAGE // P) ** 2)
    # one row per (sample, patch) location, not per channel-patch token
    assert fused.shape[0] == len(images) * (IMAGE // P) ** 2
    assert torch.isfinite(fused).all()


def test_dropped_rows_are_omitted_not_zeroed():
    """``visible`` drops channel-patches; a location with none is absent."""
    cft = _cft()
    images, ids = _batch()
    N = (IMAGE // P) ** 2
    visible = [torch.zeros(im.shape[0], N, dtype=torch.bool) for im in images]
    visible[0][0, 1] = True  # exactly one channel-patch survives
    fused, present, _ = cft(images, ids, visible=visible)
    assert fused.shape == (1, D)
    assert present.sum().item() == 1
    assert bool(present[0, 1])
    assert not present[1].any() and not present[2].any()


def test_channel_order_does_not_matter():
    cft = _cft()
    cft.eval()
    images, ids = _batch()
    g = torch.Generator().manual_seed(3)
    perms = [torch.randperm(im.shape[0], generator=g) for im in images]
    images_p = [im[pr] for im, pr in zip(images, perms, strict=True)]
    ids_p = [i[pr] for i, pr in zip(ids, perms, strict=True)]
    fused, present, _ = cft(images, ids)
    fused_p, present_p, _ = cft(images_p, ids_p)
    assert torch.equal(present, present_p)
    torch.testing.assert_close(fused, fused_p, atol=1e-6, rtol=0)


def test_variable_patch_and_pixel_sizes_are_forwarded():
    cft = _cft()
    images, ids = _batch()
    ps = [2, 4, 2]
    pixel_sizes = [0.6, 0.65, 0.65]
    fused, _present, cs = cft(images, ids, patch_sizes=ps, pixel_sizes=pixel_sizes)
    expected_rows = sum((IMAGE // p) ** 2 for p in ps)
    assert fused.shape == (expected_rows, D)
    assert cft.patch_sizes == (2, 4)
    assert torch.equal(cs.per_sample_of(cs.patch_size), torch.tensor(ps).float())
    assert torch.equal(cs.per_sample_of(cs.pixel_size), torch.tensor(pixel_sizes).float())


def test_pixel_size_is_omitted_when_not_requested():
    """The stage passes µ/px through; it does not invent one."""
    cft = _cft()
    images, ids = _batch()
    _, _, cs = cft(images, ids)
    assert cs.pixel_size is None


def test_target_stage_has_no_positional_table():
    cft = _cft()
    assert isinstance(cft.fusion, ChannelFusion)
    assert cft.fusion.pos_embed == "none"
    assert not cft.fusion.has_spatial_pos


def test_identity_gated_variant_has_the_same_contract():
    cft = _cft(fusion="identity_gated")
    assert isinstance(cft.fusion, IdentityGatedSum)
    images, ids = _batch()
    fused, present, _ = cft(images, ids)
    assert fused.shape == (present.sum().item(), D)
    assert torch.isfinite(fused).all()


def test_build_matches_manual_wiring():
    """``build`` is only convenience: it must equal composing the modules by hand."""
    torch.manual_seed(0)
    embedding = ChannelEmbedding(D, num_channel_ids=16)
    built = ChannelFusionTokenizer.build(IMAGE, P, D, embedding, n_heads=HEADS, patch_sizes=(2, 4))
    manual = ChannelFusionTokenizer(
        ChannelTokenizer(IMAGE, P, D, embedding, patch_sizes=(2, 4)),
        ChannelFusion(D, n_heads=HEADS),
    )
    manual.load_state_dict(built.state_dict())
    images, ids = _batch()
    fused_b, present_b, _ = built(images, ids)
    fused_m, present_m, _ = manual(images, ids)
    assert torch.equal(present_b, present_m)
    torch.testing.assert_close(fused_b, fused_m)


def test_unknown_fusion_is_rejected():
    torch.manual_seed(0)
    embedding = ChannelEmbedding(D, num_channel_ids=16)
    with pytest.raises(ValueError, match="unknown fusion"):
        ChannelFusionTokenizer.build(IMAGE, P, D, embedding, fusion="nope")


def test_matches_the_model_internal_path():
    """The named stage must not drift from ``ChannelFusionMAE``'s own stages."""
    torch.manual_seed(0)
    model = ChannelFusionMAE(
        image_size=IMAGE,
        patch_size=P,
        dim=D,
        encoder_depth=1,
        decoder_depth=1,
        n_heads=HEADS,
        n_memory=2,
        num_channel_ids=16,
        patch_sizes=(2, 4),
        pixel_size=1.0,
    )
    model.eval()
    cft = ChannelFusionTokenizer(model.tokenizer, model.fusion)

    images, ids = _batch()
    visible = [torch.ones(im.shape[0], (IMAGE // P) ** 2, dtype=torch.bool) for im in images]
    fused, present, _ = cft(images, ids, visible=visible, patch_sizes=P, pixel_sizes=1.0)
    cs, _, _, _, _, _ = model._prepare(images, ids, visible, None, P, pixel_sizes=1.0)
    fused_model, present_model = model.fusion(cs)
    assert torch.equal(present, present_model)
    torch.testing.assert_close(fused, fused_model)
