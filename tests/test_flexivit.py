"""FlexiViT target architecture: RoPE, fractional PE, PI-resize, dynamic p."""

import math

import pytest
import torch
import torch.nn.functional as F

from fuse_channels import (
    ChannelFusionMAE,
    ChannelTokenizer,
    ChannelEmbedding,
    apply_rope,
    axial_rope_cache,
    fractional_1d_sincos,
    get_fractional_sincos_pos_embed,
    identity_rope_cache,
    normalized_grid_coords,
    pi_resize_patch_embed,
    rope_frequencies,
)


def _assert_every_row_is_p_squared(out, p):
    """The flat pixel layout: one ``p**2``-pixel row per target."""
    offsets = out["target_pixel_offset"]
    counts = (offsets[1:] - offsets[:-1]).tolist()
    assert set(counts) == {p * p}
    assert out["pred"].numel() == out["target"].numel() == sum(counts)


# --------------------------------------------------------------------------- #
# encoder: 2-D axial RoPE on normalized coordinates
# --------------------------------------------------------------------------- #
def test_rope_frequencies_require_head_dim_divisible_by_four():
    with pytest.raises(ValueError, match="divisible by 4"):
        rope_frequencies(6)
    assert rope_frequencies(8, base=100.0).shape == (2,)


def test_normalized_rope_coords_are_grid_size_invariant():
    """A given fractional location gets the same rotary phase on any grid."""
    small_rows = torch.arange(4)
    small_cols = torch.arange(4)
    # (row + 0.5) / 4 == (large_row + 0.5) / 12 for large_row = 1, 4, 7, 10
    large_rows = torch.tensor([1, 4, 7, 10])
    large_cols = torch.tensor([1, 4, 7, 10])
    c_small, s_small = axial_rope_cache(
        small_rows, small_cols, 4, 4, head_dim=8, base=100.0, ref_extent=32.0
    )
    c_large, s_large = axial_rope_cache(
        large_rows, large_cols, 12, 12, head_dim=8, base=100.0, ref_extent=32.0
    )
    assert torch.allclose(c_small, c_large, atol=1e-6)
    assert torch.allclose(s_small, s_large, atol=1e-6)

    rn, cn = normalized_grid_coords(
        torch.tensor([0, 1]), torch.tensor([0, 1]), 4, 4, ref_extent=32.0
    )
    assert torch.allclose(rn, torch.tensor([4.0, 12.0]))
    assert torch.allclose(cn, torch.tensor([4.0, 12.0]))


def test_shared_extent_preserves_aspect_ratio():
    rows = torch.tensor([1])
    cols = torch.tensor([3])
    rn, cn = normalized_grid_coords(
        rows, cols, 2, 8, ref_extent=8.0, shared_extent=True
    )
    assert torch.allclose(rn, torch.tensor([1.5]))
    assert torch.allclose(cn, torch.tensor([3.5]))


def test_rope_identity_cache_is_a_noop_and_rotation_preserves_norm():
    x = torch.randn(3, 2, 8)
    cos, sin = identity_rope_cache(3, 8)
    assert torch.allclose(apply_rope(x, cos, sin), x)

    rows = torch.tensor([0, 1, 2])
    cols = torch.tensor([2, 1, 0])
    cos, sin = axial_rope_cache(rows, cols, 4, 4, 8, base=100.0)
    y = apply_rope(x, cos, sin)
    assert torch.allclose(y.norm(dim=-1), x.norm(dim=-1), atol=1e-6)


# --------------------------------------------------------------------------- #
# decoder: factorized-fractional absolute PE
# --------------------------------------------------------------------------- #
def test_fractional_pe_is_grid_size_invariant_and_non_square():
    rows = torch.tensor([0, 1, 2, 3])
    cols = torch.tensor([0, 1, 2, 3])
    pe = get_fractional_sincos_pos_embed(16, rows, cols, 4, 4)
    assert pe.shape == (4, 16)
    # (row + 0.5) / 4 == (large_row + 0.5) / 12
    pe_large = get_fractional_sincos_pos_embed(
        16,
        torch.tensor([1, 4, 7, 10]),
        torch.tensor([1, 4, 7, 10]),
        12,
        12,
    )
    assert torch.allclose(pe, pe_large, atol=1e-6)

    # non-square grid: row fractions from a 12-row grid, column fractions
    # from a 4-column grid
    non_square = get_fractional_sincos_pos_embed(
        16, torch.tensor([1, 4, 7, 10]), cols, 12, 4
    )
    assert non_square.shape == (4, 16)
    assert torch.allclose(non_square, pe, atol=1e-6)


def test_fractional_pe_frequencies_are_finite_and_distinct():
    pe = get_fractional_sincos_pos_embed(32, torch.arange(8), torch.arange(8), 8, 8)
    assert torch.isfinite(pe).all()
    assert torch.unique(pe, dim=0).shape[0] == 8
    with pytest.raises(ValueError, match="even dim"):
        get_fractional_sincos_pos_embed(7, torch.arange(2), torch.arange(2), 2, 2)


# --------------------------------------------------------------------------- #
# FlexiViT patch embedding: PI-resize
# --------------------------------------------------------------------------- #
def test_pi_resize_shape_and_identity_at_same_size():
    w = torch.randn(5, 1, 4, 4)
    assert torch.equal(pi_resize_patch_embed(w, 4), w)
    assert pi_resize_patch_embed(w, 8).shape == (5, 1, 8, 8)
    assert pi_resize_patch_embed(w, 2).shape == (5, 1, 2, 2)


def test_pi_resize_uses_the_pseudoinverse_resize_operator():
    """PI-resize applies ``pinv(B.T)`` to the flattened kernel."""
    from fuse_channels.flexi import _bilinear_resize_matrix, _pi_resize_matrix

    p_in, p_out = 4, 8
    w = torch.randn(1, 1, p_in, p_in)
    B = _bilinear_resize_matrix(p_in, p_out)
    P = _pi_resize_matrix(p_in, p_out)
    assert torch.allclose(P, torch.linalg.pinv(B.T), atol=1e-5)
    w_pi = pi_resize_patch_embed(w, p_out).reshape(-1)
    assert torch.allclose(w_pi, P @ w.reshape(-1), atol=1e-5)
    # PI-resize is not the same operation as bilinear resize.
    bilinear = (B @ w.reshape(-1))
    assert not torch.allclose(w_pi, bilinear)


def test_bilinear_resize_matrix_is_the_real_resize_operator():
    """``B @ vec(x)`` must equal ``F.interpolate(x)``, for both directions."""
    from fuse_channels.flexi import _bilinear_resize_matrix

    for p_in, p_out in [(8, 4), (8, 8), (8, 12), (8, 24)]:
        B = _bilinear_resize_matrix(p_in, p_out)
        for seed in range(3):
            g = torch.Generator().manual_seed(seed)
            x = torch.randn(p_in, p_in, generator=g)
            via_matrix = (B @ x.reshape(-1)).reshape(p_out, p_out)
            via_interp = F.interpolate(
                x[None, None], size=(p_out, p_out), mode="bilinear", align_corners=False
            )[0, 0]
            assert torch.allclose(via_matrix, via_interp, atol=1e-6)


def test_pi_resize_preserves_token_intensity_when_upsampling():
    """For ``p_out >= p_in``, ``<B x, P w> == <x, w>`` for every patch.

    This is FlexiViT Eq. (3): PI-resize compensates the magnitude change that
    bilinear resizing both the patch and the kernel would introduce, with no
    explicit intensity/gain scalar.
    """
    from fuse_channels.flexi import _bilinear_resize_matrix, _pi_resize_matrix

    for p_in, p_out in [(8, 8), (8, 12), (8, 16), (8, 24)]:
        g = torch.Generator().manual_seed(0)
        X = torch.randn(5, p_in * p_in, generator=g)  # canonical patches
        W = torch.randn(3, p_in * p_in, generator=g)  # flattened kernels
        B = _bilinear_resize_matrix(p_in, p_out)
        P = _pi_resize_matrix(p_in, p_out)
        ref: torch.Tensor = X @ W.T  # <x, w>
        got: torch.Tensor = (X @ B.T) @ (W @ P.T).T  # <B x, P w>
        assert torch.allclose(got, ref, atol=1e-4), (p_in, p_out)


def test_pi_resized_kernel_keeps_token_scale_bounded():
    """Regression guard: token RMS must not blow up away from the canonical p.

    The buggy axis order of ``_bilinear_resize_matrix`` produced kernels whose
    norm grew by ~100x at p=24 from a canonical p=16, silently destroying the
    patch embedding (and the relative weight of the additive channel id).
    """
    for p_out in [8, 10, 12, 15, 20, 24, 30, 40]:
        g = torch.Generator().manual_seed(0)
        w = torch.randn(4, 16, 16, generator=g)  # canonical p=16 kernels
        X = torch.randn(64, p_out * p_out, generator=g)  # white-noise patches
        w_pi = pi_resize_patch_embed(w.unsqueeze(1), p_out).reshape(4, -1)
        rms = (X @ w_pi.T).norm(dim=-1).mean()
        canonical = (torch.randn(64, 256, generator=g) @ w.reshape(4, -1).T).norm(dim=-1).mean()
        assert rms < 5.0 * canonical, (p_out, rms.item(), canonical.item())


def test_tokenizer_pi_resizes_and_reports_dynamic_grid():
    emb = ChannelEmbedding(16, 8)
    tok = ChannelTokenizer(16, 8, 16, emb, patch_sizes=(4, 8, 16))
    g = torch.Generator().manual_seed(0)
    images = [
        torch.randn(2, 1, 16, 16, generator=g),
        torch.randn(3, 1, 16, 16, generator=g),
    ]
    ids = [torch.tensor([0, 1]), torch.tensor([2, 3, 4])]
    cs = tok(images, ids, patch_sizes=[4, 16])
    assert cs.grid_h.tolist() == [4, 1]
    assert cs.grid_w.tolist() == [4, 1]
    assert cs.n_patches_per_sample.tolist() == [16, 1]
    assert cs.n_patches == 16
    assert cs.patch_row.max().item() == 3
    assert torch.allclose(cs.patch_size, torch.tensor([4.0] * 32 + [16.0] * 3))


# --------------------------------------------------------------------------- #
# end-to-end dynamic patch size
# --------------------------------------------------------------------------- #
def _batch(image_size=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    images = [
        torch.randn(2, 1, image_size, image_size, generator=g),
        torch.randn(3, 1, image_size, image_size, generator=g),
    ]
    ids = [torch.tensor([0, 1]), torch.tensor([2, 3, 4])]
    return images, ids


def _model(**kw):
    cfg = dict(
        image_size=16,
        patch_size=8,
        patch_sizes=(4, 8),
        dim=32,
        encoder_depth=1,
        decoder_depth=1,
        n_heads=4,
        n_memory=2,
        num_channel_ids=8,
        flexi_random=False,
        pixel_size=1.0,
    )
    cfg.update(kw)
    return ChannelFusionMAE(**cfg)


def test_per_batch_patch_size_changes_token_count_and_head_dim():
    model = _model()
    images, ids = _batch()
    for p, expected_n in ((4, 16), (8, 4)):
        cs = model.tokenizer(images, ids, patch_sizes=p)
        assert cs.n_patches == expected_n
        visible = [
            torch.zeros(2, expected_n, dtype=torch.bool),
            torch.zeros(3, expected_n, dtype=torch.bool),
        ]
        visible[0][0, 0] = True
        visible[1][0, 0] = True
        out = model(images, ids, visible=visible, patch_sizes=p)
        assert out["patch_sizes"] == [p, p]
        _assert_every_row_is_p_squared(out, p)
        assert out["loss"].backward() is None


def test_per_sample_variable_patch_size_keeps_each_rows_own_pixels():
    model = _model(patch_sizes=(4, 8))
    model.eval()
    images, ids = _batch()
    # Force targets in both samples by keeping one channel per sample visible.
    visible = [
        torch.zeros(2, 16, dtype=torch.bool),
        torch.zeros(3, 4, dtype=torch.bool),
    ]
    visible[0][0, 0] = True
    visible[1][0, 0] = True
    with torch.no_grad():
        out = model(images, ids, visible=visible, patch_sizes=[4, 8])
    # mixed p: rows keep exactly their own p**2 pixels, contiguous -- no
    # padded [K, max(p)**2] block and no validity mask
    counts = (
        out["target_pixel_offset"][1:] - out["target_pixel_offset"][:-1]
    ).tolist()
    assert set(counts) == {16, 64}
    assert out["target"].shape == out["pred"].shape
    assert out["pred"].numel() == sum(counts)
    assert torch.isfinite(out["loss"])


def test_single_patch_grid_falls_back_to_channel_masking():
    """FlexiViT can choose p = H; DCP must not require a maskable patch."""
    model = _model(patch_sizes=(4, 16))
    model.train()
    images, ids = _batch()
    out = model(
        images, ids, generator=torch.Generator().manual_seed(4), patch_sizes=16
    )
    assert out["patch_sizes"] == [16, 16]
    assert torch.isfinite(out["loss"])
    out["loss"].backward()


def test_encoder_rope_uses_normalized_layout_not_raw_indices():
    model = _model(patch_sizes=(4, 8, 16))
    assert model.encoder_pos_embed == "rope"
    c1, s1 = axial_rope_cache(
        torch.tensor([0]), torch.tensor([0]), 4, 4, 8,
        base=model.rope_base, ref_extent=model.rope_ref_extent,
    )
    c2, s2 = axial_rope_cache(
        torch.tensor([1]), torch.tensor([1]), 12, 12, 8,
        base=model.rope_base, ref_extent=model.rope_ref_extent,
    )
    assert torch.allclose(c1, c2, atol=1e-6)
    assert torch.allclose(s1, s2, atol=1e-6)


def test_scale_conditioning_add_and_adaln_and_none():
    images, ids = _batch()
    add = _model(scale_conditioning="add", pixel_size=0.5)
    add.eval()
    with torch.no_grad():
        a = add(images, ids, generator=torch.Generator().manual_seed(2), patch_sizes=8)["pred"]
        b = add(images, ids, generator=torch.Generator().manual_seed(2), patch_sizes=8)["pred"]
        c = add(
            images, ids, generator=torch.Generator().manual_seed(2), patch_sizes=8,
            pixel_sizes=[0.5, 1.0],
        )["pred"]
    assert a is not None and torch.allclose(a, b)
    # Physical pixel size is a real conditioner, not just log(p).
    assert not torch.allclose(a, c)

    adaln = _model(scale_conditioning="adaln")
    out = adaln(images, ids, generator=torch.Generator().manual_seed(3), patch_sizes=8)
    assert out["loss"].backward() is None
    assert adaln.encoder.blocks[0].mod1.weight.grad is not None

    none = _model(scale_conditioning="none")
    assert none.scale_embed is None
    assert none(images, ids, patch_sizes=8)["scale_values"] is None


def test_position_can_be_fully_disabled():
    model = _model(pos_embed="none", patch_sizes=(8,))
    assert model.encoder_pos_embed == "none"
    assert model.decoder_pos_embed == "none"
    assert model.fusion_pos_embed == "none"
    assert model.decoder.fractional_pe is None
    images, ids = _batch()
    out = model(images, ids, generator=torch.Generator().manual_seed(1), patch_sizes=8)
    assert torch.isfinite(out["loss"])


def test_fractional_decoder_pe_is_load_bearing_for_mask_tokens():
    model = _model(patch_sizes=(8,))
    model.eval()
    images, ids = _batch(image_size=16)
    images = images[:1]
    ids = ids[:1]
    visible = [torch.zeros(2, 4, dtype=torch.bool)]
    visible[0][1] = True
    with torch.no_grad():
        out = model(images, ids, visible=visible, patch_sizes=8)
    per_patch = out["pred"].reshape(4, -1)
    assert not torch.allclose(per_patch[0], per_patch[1])


# --------------------------------------------------------------------------- #
# reconstruction head: per-p heads, fallback, and token budget
# --------------------------------------------------------------------------- #
def test_per_p_head_is_exact_and_does_not_resize():
    model = _model(patch_sizes=(4, 8), head_mode="per_p")
    model.eval()
    x = torch.randn(3, 32, generator=torch.Generator().manual_seed(0))
    head = model.decoder.head
    assert set(head.heads.keys()) == {"4", "8"}
    # flat layout: the head's output is the per-p block, row-major
    assert torch.equal(head(x, 4), head.heads["4"](x).reshape(-1))
    assert torch.equal(head(x, 8), head.heads["8"](x).reshape(-1))

    images, ids = _batch()
    for p in (4, 8):
        with torch.no_grad():
            out = model(
                images, ids,
                generator=torch.Generator().manual_seed(p), patch_sizes=p,
            )
        _assert_every_row_is_p_squared(out, p)
        assert torch.isfinite(out["loss"])


def test_per_p_head_rejects_unseen_upsampling():
    model = _model(
        patch_sizes=(4, 8),
        head_patch_size=4,
        head_mode="per_p",
    )
    images, ids = _batch()
    with pytest.raises(ValueError, match="cannot upsample"):
        model(
            images, ids,
            generator=torch.Generator().manual_seed(0), patch_sizes=16,
        )


def test_canonical_head_mode_is_still_available():
    model = _model(patch_sizes=(4, 8), head_mode="canonical")
    model.eval()
    images, ids = _batch()
    for p in (4, 8):
        with torch.no_grad():
            out = model(
                images, ids,
                generator=torch.Generator().manual_seed(p), patch_sizes=p,
            )
        _assert_every_row_is_p_squared(out, p)
        assert torch.isfinite(out["loss"])


def test_flexi_random_respects_max_tokens():
    model = _model(
        patch_sizes=(4, 8, 16),
        max_tokens=4,
        flexi_random=True,
    )
    model.train()
    images, ids = _batch(image_size=16)
    seen = set()
    for i in range(40):
        out = model(images, ids, generator=torch.Generator().manual_seed(i))
        p = out["patch_sizes"][0]
        seen.add(p)
        assert (16 // p) * (16 // p) <= 4
    assert seen <= {8, 16}
    assert out["loss"].backward() is None


def test_max_tokens_excludes_every_patch_size():
    model = _model(
        patch_sizes=(4, 8),
        max_tokens=1,
        flexi_random=True,
    )
    model.train()
    images, ids = _batch(image_size=16)
    with pytest.raises(ValueError, match="max_tokens excludes every"):
        model(images, ids, generator=torch.Generator().manual_seed(0))


def test_non_divisible_border_is_cropped_not_rejected():
    model = ChannelFusionMAE(
        image_size=15,
        patch_size=4,
        patch_sizes=(4,),
        dim=32,
        encoder_depth=1,
        decoder_depth=1,
        n_heads=4,
        n_memory=2,
        num_channel_ids=8,
        flexi_random=False,
        pixel_size=1.0,
    )
    g = torch.Generator().manual_seed(0)
    images = [torch.randn(2, 1, 15, 15, generator=g)]
    ids = [torch.tensor([0, 1])]
    cs = model.tokenizer(images, ids, patch_sizes=4)
    assert cs.grid_h.tolist() == [3]
    assert cs.grid_w.tolist() == [3]
    assert cs.n_patches_per_sample.tolist() == [9]

    out = model(
        images, ids, generator=torch.Generator().manual_seed(0), patch_sizes=4
    )
    assert out["grid_h"].tolist() == [3]
    assert out["n_patches_per_sample"].tolist() == [9]
    _assert_every_row_is_p_squared(out, 4)
    assert torch.isfinite(out["loss"])
    out["loss"].backward()


def test_max_tokens_explicit_patch_size_is_an_override():
    model = _model(patch_sizes=(4, 8), max_tokens=1, flexi_random=True)
    model.eval()
    images, ids = _batch(image_size=16)
    with torch.no_grad():
        out = model(
            images, ids,
            generator=torch.Generator().manual_seed(0), patch_sizes=4,
        )
    assert out["patch_sizes"] == [4, 4]
    _assert_every_row_is_p_squared(out, 4)


def test_canonical_head_rejects_unseen_upsampling():
    model = _model(
        patch_sizes=(4, 8),
        head_patch_size=4,
        head_mode="canonical",
    )
    images, ids = _batch()
    with pytest.raises(ValueError, match="cannot upsample"):
        model(
            images, ids,
            generator=torch.Generator().manual_seed(0), patch_sizes=16,
        )


def test_unseen_p_uses_nearest_larger_head_without_upsampling():
    model = _model(patch_sizes=(4, 8, 16), head_mode="per_p")
    head = model.decoder.head
    x = torch.randn(2, 32, generator=torch.Generator().manual_seed(0))
    out = head(x, 6)
    assert out.shape == (2 * 36,)  # two rows of 6**2 pixels, contiguous
    assert torch.isfinite(out).all()
