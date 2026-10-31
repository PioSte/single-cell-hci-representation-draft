"""The acquisition fleet: discrete pixel sizes, validated at ingest.

The fleet is the set of acquisition instruments and settings the model is
trained and deployed on. Its pixel sizes are a finite, sorted table, so an
in-contract image maps to exactly one trained ``p`` and an unknown pixel size is
a metadata error -- never a silent snap of ``p`` onto a nearest lattice.
"""

import math

import numpy as np
import pytest
import torch

from fuse_channels import ChannelFusionMAE
from fuse_channels.crops import CropGeometry, patch_size_for_pixel_size, plan_patches
from fuse_channels.data import SyntheticChannelDataset
from fuse_channels.fleet import (
    FLEET_PIXEL_SIZES,
    canonical_fleet_pixel_size,
    fleet_patch_sizes,
    sorted_fleet_pixel_sizes,
)


FLEET = (0.598, 0.646, 0.650, 0.652, 0.703)


# --------------------------------------------------------------------------- #
# the table
# --------------------------------------------------------------------------- #
def test_project_fleet_is_sorted_and_discrete():
    assert FLEET_PIXEL_SIZES == FLEET
    assert sorted_fleet_pixel_sizes(FLEET) == FLEET
    assert sorted_fleet_pixel_sizes((0.703, 0.598, 0.703)) == (0.598, 0.703)


def test_fleet_rejects_empty_or_non_positive_entries():
    with pytest.raises(ValueError, match="at least one"):
        sorted_fleet_pixel_sizes(())
    with pytest.raises(ValueError, match="must be positive"):
        sorted_fleet_pixel_sizes((0.5, 0.0))


# --------------------------------------------------------------------------- #
# ingest-time membership
# --------------------------------------------------------------------------- #
def test_every_fleet_member_is_its_own_canonical_value():
    for q in FLEET:
        assert canonical_fleet_pixel_size(q, FLEET) == q


def test_float_noise_maps_to_the_table_value():
    """OME reports 0.59766 or 0.6500000000000001; both are the same instrument."""
    assert canonical_fleet_pixel_size(0.59766, FLEET) == 0.598
    assert canonical_fleet_pixel_size(0.6500000000000001, FLEET) == 0.650
    assert canonical_fleet_pixel_size(0.6519, FLEET) == 0.652


def test_non_member_pixel_size_is_rejected_not_snapped():
    # 0.66 is 1.5% off the nearest member -- outside the tolerance on purpose
    with pytest.raises(ValueError, match="not a member of the acquisition fleet"):
        canonical_fleet_pixel_size(0.66, FLEET)
    with pytest.raises(ValueError, match="untrained patch size"):
        canonical_fleet_pixel_size(0.4, FLEET)


def test_unit_traps_are_called_out():
    with pytest.raises(ValueError, match="nanometres"):
        canonical_fleet_pixel_size(598.0, FLEET)
    with pytest.raises(ValueError, match="pixels per um"):
        canonical_fleet_pixel_size(1.0 / 0.650, FLEET)


def test_non_positive_pixel_size_is_a_value_error():
    with pytest.raises(ValueError, match="must be positive"):
        canonical_fleet_pixel_size(0.0, FLEET)


# --------------------------------------------------------------------------- #
# the derived patch sizes
# --------------------------------------------------------------------------- #
def test_fleet_patch_sizes_are_the_distinct_p_over_the_table():
    assert fleet_patch_sizes(FLEET, target_um=12.8) == (18, 20, 21)
    assert fleet_patch_sizes(FLEET, target_um=8.0) == (11, 12, 13)
    assert fleet_patch_sizes((0.5,), target_um=8.0) == (16,)


def test_three_instruments_share_one_p():
    """0.646/0.650/0.652 all map to p=20, so log(pixel_size) is not redundant."""
    ps = [patch_size_for_pixel_size(q, 12.8) for q in FLEET]
    assert ps == [21, 20, 20, 20, 18]
    assert len(set(ps)) < len(ps)


def test_extent_error_is_bounded_across_the_fleet():
    target = 12.8
    for q in FLEET:
        p = patch_size_for_pixel_size(q, target)
        error = (p * q - target) / target
        assert abs(error) < 0.02  # <= ~1.9% at target_um=12.8


def test_scale_features_encode_the_realized_extent_not_the_target():
    """The conditioner's two inputs recover the actual p*pixel_size.

    Because p is rounded, the realized extent differs from target_um by up to
    ~2%; log(p) + log(pixel_size) is exactly log(extent_um), so that residual is
    an input rather than a silently discarded error.
    """
    ps = fleet_patch_sizes(FLEET, target_um=12.8)
    model = ChannelFusionMAE(
        image_size=48,
        patch_size=ps[-1],
        patch_sizes=ps,
        dim=16,
        encoder_depth=1,
        decoder_depth=1,
        n_heads=4,
        n_memory=2,
        num_channel_ids=8,
        flexi_random=False,
        pixel_size=FLEET[0],
    )
    for q in FLEET:
        geom = CropGeometry.build(q, 12.8, n_tokens=2, fleet_pixel_sizes=FLEET)
        values = model._scale_values(
            [geom.patch_size], [geom.pixel_size], torch.device("cpu")
        )
        realized = float(torch.exp(values[0, 0]) * torch.exp(values[0, 1]))
        assert realized == pytest.approx(geom.extent_um)
        assert realized == pytest.approx(geom.patch_size * geom.pixel_size)


def test_derived_patch_sizes_drive_the_model_config():
    """The fleet, not a hand-written list, is the source of model patch_sizes."""
    ps = fleet_patch_sizes(FLEET, target_um=12.8)
    model = ChannelFusionMAE(
        image_size=48,
        patch_size=ps[-1],
        patch_sizes=ps,
        dim=16,
        encoder_depth=1,
        decoder_depth=1,
        n_heads=4,
        n_memory=2,
        num_channel_ids=8,
        flexi_random=False,
        pixel_size=FLEET[0],
    )
    assert model.patch_sizes == ps


# --------------------------------------------------------------------------- #
# geometry and dataset validation
# --------------------------------------------------------------------------- #
def test_geometry_validates_the_fleet_at_ingest():
    g = CropGeometry.build(0.6500000000000001, 12.8, n_tokens=2, fleet_pixel_sizes=FLEET)
    assert g.pixel_size == 0.650
    assert g.patch_size == 20
    assert g.crop_px == 40
    with pytest.raises(ValueError, match="acquisition fleet"):
        CropGeometry.build(0.66, 12.8, n_tokens=2, fleet_pixel_sizes=FLEET)


def test_geometry_accepts_any_pixel_size_without_a_fleet():
    """The library stays general: another dataset passes its own fleet or none."""
    g = CropGeometry.build(0.37, 12.8, n_tokens=2)
    assert g.patch_size == 35
    assert g.pixel_size == 0.37


def test_plan_patches_validates_the_fleet():
    mask = np.zeros((128, 128), dtype=np.int32)
    mask[40:60, 40:60] = 1
    with pytest.raises(ValueError, match="acquisition fleet"):
        plan_patches(
            mask, (128, 128), pixel_size=0.66, target_um=12.8,
            fleet_pixel_sizes=FLEET, n_tokens=2,
        )
    crops, skipped = plan_patches(
        mask, (128, 128), pixel_size=0.650, target_um=12.8,
        fleet_pixel_sizes=FLEET, n_tokens=2, image_id="im", channel_ids=(0,),
    )
    assert not skipped
    assert crops[0].geometry.patch_size == 20


def test_dataset_draws_discrete_fleet_members():
    ds = SyntheticChannelDataset(
        n_samples=16,
        image_size=16,
        c_min=2,
        c_max=2,
        n_channel_ids=4,
        pixel_size_range=(0.59, 0.71),
        fleet_pixel_sizes=FLEET,
        seed=0,
    )
    values = [ds[i][2] for i in range(16)]
    assert set(values) <= set(FLEET)
    assert len(set(values)) > 1
    assert all(any(math.isclose(v, q) for q in FLEET) for v in values)


def test_dataset_canonicalises_float_noise():
    ds = SyntheticChannelDataset(
        n_samples=2, image_size=8, c_min=2, c_max=2, n_channel_ids=4,
        pixel_size=[0.59766, 0.6500000000000001], fleet_pixel_sizes=FLEET, seed=0,
    )
    assert [ds[i][2] for i in range(2)] == [0.598, 0.650]


def test_dataset_rejects_a_non_member_by_default():
    with pytest.raises(ValueError, match="acquisition fleet"):
        SyntheticChannelDataset(
            n_samples=2, image_size=8, c_min=2, c_max=2, n_channel_ids=4,
            pixel_size=[0.5, 0.5], fleet_pixel_sizes=FLEET, seed=0,
        )


def test_dataset_serves_another_fleet_or_none():
    """Other datasets keep working: their own table, or no validation at all."""
    foreign = SyntheticChannelDataset(
        n_samples=2, image_size=8, c_min=2, c_max=2, n_channel_ids=4,
        pixel_size=[0.5, 0.5], fleet_pixel_sizes=(0.4, 0.5), seed=0,
    )
    assert foreign[0][2] == 0.5
    permissive = SyntheticChannelDataset(
        n_samples=2, image_size=8, c_min=2, c_max=2, n_channel_ids=4,
        pixel_size=[0.37, 0.37], fleet_pixel_sizes=FLEET, strict_fleet=False, seed=0,
    )
    assert permissive[0][2] == 0.37


def test_pixel_size_range_without_a_fleet_stays_continuous():
    ds = SyntheticChannelDataset(
        n_samples=8, image_size=8, c_min=2, c_max=2, n_channel_ids=4,
        pixel_size_range=(0.5, 2.0), seed=0,
    )
    values = [ds[i][2] for i in range(8)]
    assert all(0.5 <= v <= 2.0 for v in values)
    assert not set(values) <= set(FLEET)
