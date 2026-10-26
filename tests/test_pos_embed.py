"""Tests for ``fuse_channels.pos_embed``.

``pos_embed="learned"`` is an ``nn.Parameter`` table; ``pos_embed="sincos"`` is
the fixed 2D sine-cosine map (MAE/ViT) registered as a non-trainable buffer.
Both expose ``module.spatial_pos`` with the module's expected shape, so the
fusion/decoder forward paths are identical.
"""

import pytest
import torch
import torch.nn as nn

from fuse_channels import (
    ChannelFusion,
    IdentityGatedSum,
    POS_EMBED_KINDS,
    build_spatial_pos,
    get_2d_sincos_pos_embed,
)
from fuse_channels.pos_embed import add_spatial_pos

D = 16
N = 16  # 4x4 grid


# ---------------------------------------------------------------- #
# the raw 2D sine-cosine map
# ---------------------------------------------------------------- #
def _expected_row_col(dim, rows, cols):
    """Reference implementation of MAE's 2D sincos embedding for one position."""

    def one_axis(coord):
        half = (dim // 2) // 2
        omega = 1.0 / (10000.0 ** (torch.arange(half, dtype=torch.float32) / half))
        out = torch.tensor(float(coord)) * omega
        return torch.cat([out.sin(), out.cos()])

    return torch.cat([one_axis(rows), one_axis(cols)])


def test_sincos_shape_range_and_determinism():
    pe = get_2d_sincos_pos_embed(D, 4)
    assert pe.shape == (16, D)
    assert pe.dtype == torch.float32
    assert torch.isfinite(pe).all()
    assert pe.abs().max() <= 1.0
    # deterministic by construction (fixed function of position)
    assert torch.equal(pe, get_2d_sincos_pos_embed(D, 4))
    # distinct positions get distinct rows
    assert torch.unique(pe, dim=0).shape[0] == 16


def test_sincos_is_row_major_over_the_grid():
    """Patch ``p = r * grid_w + c`` must hold, matching Conv2d.flatten(2)."""
    grid = 4
    pe = get_2d_sincos_pos_embed(D, grid)
    for r in range(grid):
        for c in range(grid):
            idx = r * grid + c
            assert torch.allclose(pe[idx], _expected_row_col(D, r, c), atol=1e-6)


def test_sincos_same_position_matches_across_grid_sizes():
    """A module built for grid 4 sees the same vector as grid 8 at that position."""
    small = get_2d_sincos_pos_embed(D, 4)
    large = get_2d_sincos_pos_embed(D, 8)
    for r in range(4):
        for c in range(4):
            assert torch.allclose(
                small[r * 4 + c], large[r * 8 + c], atol=1e-6
            )


def test_sincos_non_square_grid():
    pe = get_2d_sincos_pos_embed(D, (2, 3))
    assert pe.shape == (6, D)
    assert torch.allclose(pe[1], _expected_row_col(D, 0, 1), atol=1e-6)
    assert torch.allclose(pe[3], _expected_row_col(D, 1, 0), atol=1e-6)


def test_sincos_rejects_dims_not_divisible_by_four():
    with pytest.raises(ValueError, match="divisible by 4"):
        get_2d_sincos_pos_embed(6, 4)


# --------------------------------------------------------------------------
# construction / registration
# ---------------------------------------------------------------------------
def test_build_spatial_pos_variants():
    learn = build_spatial_pos(32, 9, "learned")
    assert isinstance(learn, nn.Parameter)
    assert tuple(learn.shape) == (9, 32)
    assert learn.requires_grad

    fixed = build_spatial_pos(32, 9, "sincos")
    assert not isinstance(fixed, nn.Parameter)
    assert tuple(fixed.shape) == (9, 32)
    assert torch.allclose(fixed, get_2d_sincos_pos_embed(32, 3))

    batched = build_spatial_pos(32, 9, "learned", batch_dims=1)
    assert tuple(batched.shape) == (1, 9, 32)


def test_build_spatial_pos_rejects_bad_arguments():
    with pytest.raises(ValueError, match="unknown pos_embed"):
        build_spatial_pos(32, 9, "rope")
    with pytest.raises(ValueError, match="not a square"):
        build_spatial_pos(32, 6, "sincos")  # needs explicit grid_size
    with pytest.raises(ValueError, match="expected n_patches"):
        build_spatial_pos(32, 6, "sincos", grid_size=4)
    with pytest.raises(ValueError, match="n_patches must be positive"):
        build_spatial_pos(32, 0, "learned")


def test_add_spatial_pos_registers_parameter_or_buffer():
    m = nn.Module()

    add_spatial_pos(m, 32, 9, "learned", batch_dims=1)
    assert m.pos_embed == "learned"
    assert "spatial_pos" in dict(m.named_parameters())
    assert "spatial_pos" in m.state_dict()

    m = nn.Module()
    add_spatial_pos(m, 32, 9, "sincos", batch_dims=1)
    assert m.pos_embed == "sincos"
    assert "spatial_pos" not in dict(m.named_parameters())
    assert "spatial_pos" in dict(m.named_buffers())
    assert "spatial_pos" in m.state_dict()


# --------------------------------------------------------------------------
# module wiring
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kind", POS_EMBED_KINDS)
def test_fusion_modules_expose_selected_pe(kind):
    fusion = ChannelFusion(D, n_heads=4, n_patches=N, pos_embed=kind)
    assert fusion.pos_embed == kind
    assert tuple(fusion.spatial_pos.shape) == (1, N, D)
    assert kind in repr(fusion)

    gated = IdentityGatedSum(D, n_patches=N, pos_embed=kind)
    assert gated.pos_embed == kind
    assert tuple(gated.spatial_pos.shape) == (1, N, D)
    assert kind in repr(gated)

    if kind == "learned":
        assert "spatial_pos" in dict(fusion.named_parameters())
        assert "spatial_pos" in dict(gated.named_parameters())
    else:
        assert "spatial_pos" not in dict(fusion.named_parameters())
        assert "spatial_pos" not in dict(gated.named_parameters())
        assert torch.allclose(fusion.spatial_pos[0], get_2d_sincos_pos_embed(D, 4))
