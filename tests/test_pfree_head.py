"""``head_mode="shared"``: a reconstruction head whose parameters are free of ``p``.

The head keeps one canonical ``dim -> p0*p0`` projection and resamples its
output to any target ``p``, so a checkpoint trained on one fleet serves every
``p`` -- including ``p`` values it never saw -- with no per-``p`` parameters and
no "cannot upsample" failure.
"""

import pytest
import torch
import torch.nn.functional as F

from fuse_channels import ChannelFusionMAE
from fuse_channels.decoder import PatchReconstructionHead
from fuse_channels.flexi import _pi_resize_matrix


def _head(mode="shared", p0=8, patch_sizes=(4, 8, 16), resample="area", dim=16):
    return PatchReconstructionHead(dim, p0, patch_sizes=patch_sizes, mode=mode, resample=resample)


# --------------------------------------------------------------------------- #
# the parameters do not depend on p
# --------------------------------------------------------------------------- #
def test_parameter_shapes_are_independent_of_patch_sizes():
    """A different ``patch_sizes`` set yields the identical parameter table."""
    a = _head(patch_sizes=(4, 8, 16))
    b = _head(patch_sizes=(5, 7, 23))
    assert set(a.state_dict()) == set(b.state_dict()) == {"head.weight", "head.bias"}
    assert sum(p.numel() for p in a.parameters()) == sum(p.numel() for p in b.parameters())
    assert a.head.weight.shape == (8 * 8, 16)


def test_per_p_mode_still_grows_with_the_configured_set():
    small = _head(patch_sizes=(8,), mode="per_p")
    large = _head(patch_sizes=(4, 8, 16, 32), mode="per_p")
    assert sum(p.numel() for p in large.parameters()) > sum(p.numel() for p in small.parameters())


# --------------------------------------------------------------------------- #
# every p is legal, in both directions
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("p", [1, 3, 5, 7, 8, 13, 32])
def test_shared_head_serves_any_p(p):
    head = _head()
    x = torch.randn(5, 16, generator=torch.Generator().manual_seed(0))
    out = head(x, p)
    assert out.shape == (5 * p * p,)  # flat: five contiguous p**2 rows
    assert torch.isfinite(out).all()


def test_shared_head_upsamples_where_per_p_refuses():
    """The p-free head removes the ``p > max(patch_sizes)`` hard failure."""
    x = torch.randn(2, 16, generator=torch.Generator().manual_seed(0))
    with pytest.raises(ValueError, match="cannot upsample"):
        _head(patch_sizes=(4, 8), mode="per_p")(x, 16)
    out = _head(patch_sizes=(4, 8))(x, 16)
    assert out.shape == (2 * 256,)


def test_shared_head_is_exact_at_the_canonical_p():
    head = _head()
    x = torch.randn(4, 16, generator=torch.Generator().manual_seed(0))
    assert torch.equal(head(x, 8), head.head(x).reshape(-1))


def test_mixed_patch_sizes_are_concatenated_not_padded():
    head = _head()
    x = torch.randn(3, 16, generator=torch.Generator().manual_seed(0))
    ps = [4, 8, 5]
    out = head(x, ps)
    offsets = head.pixel_offsets(ps)
    assert offsets.tolist() == [0, 16, 80, 105]
    assert out.shape == (105,)  # 16 + 64 + 25: exact, no padded columns
    # each row is its own per-p head block, in row-major order
    for i, p in enumerate(ps):
        block = out[offsets[i] : offsets[i + 1]]
        expected = head._head_for(p)[0](x[i : i + 1])
        expected = (
            head._resize(expected, p, head._head_for(p)[2]) if head._head_for(p)[1] else expected
        )
        assert torch.equal(block, expected.reshape(-1))


# --------------------------------------------------------------------------- #
# the three resample operators
# --------------------------------------------------------------------------- #
def test_area_operator_is_a_box_average_when_shrinking():
    head = _head(resample="area")
    x = torch.randn(2, 64, generator=torch.Generator().manual_seed(0))
    expected = F.interpolate(x.reshape(2, 1, 8, 8), size=(5, 5), mode="area").reshape(2, 25)
    assert torch.allclose(head._resample_shared(x, 8, 5), expected)


def test_area_operator_uses_bilinear_when_growing():
    head = _head(resample="area")
    x = torch.randn(2, 64, generator=torch.Generator().manual_seed(0))
    expected = F.interpolate(
        x.reshape(2, 1, 8, 8), size=(11, 11), mode="bilinear", align_corners=False
    ).reshape(2, 121)
    assert torch.allclose(head._resample_shared(x, 8, 11), expected)


def test_bilinear_operator_is_point_sampling_both_ways():
    head = _head(resample="bilinear")
    x = torch.randn(2, 64, generator=torch.Generator().manual_seed(0))
    down = F.interpolate(
        x.reshape(2, 1, 8, 8), size=(5, 5), mode="bilinear", align_corners=False
    ).reshape(2, 25)
    assert torch.allclose(head._resample_shared(x, 8, 5), down)


def test_pi_operator_is_the_flexivit_resize_on_the_output_axis():
    """``"pi"`` reuses the tokenizer's operator: rows treated as a p0 x p0 kernel."""
    head = _head(resample="pi")
    x = torch.randn(3, 64, generator=torch.Generator().manual_seed(0))
    op = _pi_resize_matrix(8, 5, x.dtype).to(x.device)
    assert torch.allclose(head._resample_shared(x, 8, 5), x @ op.T)


def test_resample_operator_must_be_known():
    with pytest.raises(ValueError, match="unknown head_resample"):
        _head(resample="bicubic")


# --------------------------------------------------------------------------- #
# gradients reach the single head
# --------------------------------------------------------------------------- #
def test_gradients_flow_through_the_resample():
    head = _head()
    x = torch.randn(2, 16, generator=torch.Generator().manual_seed(0))
    head(x, 5).sum().backward()
    assert head.head.weight.grad is not None
    assert torch.isfinite(head.head.weight.grad).all()


# --------------------------------------------------------------------------- #
# model integration
# --------------------------------------------------------------------------- #
def _model(**kw):
    cfg = {
        "image_size": 32,
        "patch_size": 8,
        "patch_sizes": (4, 8),
        "dim": 32,
        "encoder_depth": 1,
        "decoder_depth": 1,
        "n_heads": 4,
        "n_memory": 2,
        "num_channel_ids": 8,
        "flexi_random": False,
        "pixel_size": 1.0,
    }
    cfg.update(kw)
    return ChannelFusionMAE(**cfg)


def _batch(image_size=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    images = [
        torch.randn(2, 1, image_size, image_size, generator=g),
        torch.randn(3, 1, image_size, image_size, generator=g),
    ]
    ids = [torch.tensor([0, 1]), torch.tensor([2, 3, 4])]
    return images, ids


def test_mae_shared_head_runs_at_an_unseen_p_end_to_end():
    model = _model(head_mode="shared", head_patch_size=8)
    images, ids = _batch()
    with torch.no_grad():
        out = model(images, ids, generator=torch.Generator().manual_seed(0), patch_sizes=16)
    offsets = out["target_pixel_offset"]
    assert set((offsets[1:] - offsets[:-1]).tolist()) == {16 * 16}
    assert torch.isfinite(out["loss"])


def test_canonical_p_suggestion_is_the_log_space_median():
    """Edge-canonical p0 would resample maximally, so suggest the geometric middle."""
    from fuse_channels.decoder import canonical_patch_size_for

    for sizes, expected in (
        ((4, 8, 16), 8),
        ((4, 8, 16, 32), 8),
        ((16, 20, 21), 20),
        ((21, 32), 21),
        ((8,), 8),
    ):
        assert canonical_patch_size_for(sizes) == expected


def test_shared_mode_requires_an_explicit_head_patch_size():
    """p0 is a hyperparameter: defaulting it would change shapes with the fleet."""
    with pytest.raises(ValueError, match="needs an explicit head_patch_size"):
        _model(patch_sizes=(4, 8, 16), head_mode="shared")
    model = _model(patch_sizes=(4, 8, 16), head_mode="shared", head_patch_size=8)
    assert model.decoder.head.canonical_patch_size == 8
    # per_p keeps the old largest-head default (it cannot upsample)
    legacy = _model(patch_sizes=(4, 8, 16), head_mode="per_p")
    assert legacy.decoder.head.canonical_patch_size == 16


def test_shared_head_checkpoint_is_fleet_independent():
    """Trained on (4, 8), the same weights load into a (4, 8, 16)-configured model."""
    trained = _model(patch_sizes=(4, 8), head_mode="shared", head_patch_size=8)
    wider = _model(patch_sizes=(4, 8, 16), head_mode="shared", head_patch_size=8)
    wider.load_state_dict(trained.state_dict())  # strict: no missing/unexpected keys
    assert wider.decoder.head.head.weight.shape == trained.decoder.head.head.weight.shape
