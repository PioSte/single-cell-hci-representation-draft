import torch

from fuse_channels.data import SyntheticChannelDataset, collate_fn


def test_collate_supports_per_sample_pixel_size():
    ds = SyntheticChannelDataset(
        n_samples=4,
        image_size=8,
        c_min=2,
        c_max=2,
        n_channel_ids=4,
        pixel_size=[0.5, 1.0, 2.0, 4.0],
        return_pixel_size=True,
        seed=0,
    )
    batch = [ds[i] for i in range(4)]
    images, ids, pixel_sizes = collate_fn(batch)
    assert len(images) == len(ids) == len(pixel_sizes) == 4
    assert pixel_sizes == [0.5, 1.0, 2.0, 4.0]
    assert all(isinstance(v, float) for v in pixel_sizes)


def test_pixel_size_is_required_metadata():
    import pytest

    with pytest.raises(ValueError, match="pixel size is required"):
        SyntheticChannelDataset(n_samples=2, image_size=8, seed=0)


def test_collate_without_pixel_sizes_is_opt_out_only():
    ds = SyntheticChannelDataset(
        n_samples=2, image_size=8, seed=0, pixel_size=1.0,
        return_pixel_size=False,
    )
    images, ids = collate_fn([ds[0], ds[1]])
    assert len(images) == len(ids) == 2


def test_pixel_size_range_is_per_sample_and_in_bounds():
    ds = SyntheticChannelDataset(
        n_samples=8,
        image_size=8,
        c_min=2,
        c_max=2,
        n_channel_ids=4,
        pixel_size_range=(0.5, 2.0),
        return_pixel_size=True,
        seed=0,
    )
    values = [ds[i][2] for i in range(8)]
    assert len(set(values)) > 1
    assert all(0.5 <= v <= 2.0 for v in values)
