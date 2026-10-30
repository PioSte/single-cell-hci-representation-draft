import numpy as np
import pytest
import torch

from fuse_channels.crops import (
    CellCrop,
    CellCropDataset,
    CropGeometry,
    TiffStack,
    collate_crops,
    patch_size_for_pixel_size,
    place_box,
    plan_patches,
    token_coverage,
    token_coverage_torch,
)


def test_patch_size_holds_physical_extent_across_instruments():
    target = 8.0  # micrometres per token
    # finer sampling -> more pixels per token; extent stays ~8 um
    assert patch_size_for_pixel_size(0.5, target) == 16
    assert patch_size_for_pixel_size(1.0, target) == 8
    assert patch_size_for_pixel_size(2.0, target) == 4
    for pixel_size in (0.5, 1.0, 2.0):
        p = patch_size_for_pixel_size(pixel_size, target)
        assert p * pixel_size == pytest.approx(target)


def test_patch_size_is_a_rounding_not_a_snap():
    """There is no candidate set: p is the rounded quotient, in both directions.

    Rounding half *up* is deliberate -- Python's banker's ``round`` maps 2.5
    to 2, which would make the map non-monotone across instruments.
    """
    assert patch_size_for_pixel_size(2.0, 5.0) == 3   # 2.5 -> 3, not 2
    assert patch_size_for_pixel_size(2.0, 7.0) == 4   # 3.5 -> 4
    # no clamping to a "configured" set: 13.33 is p=13, not 8 or 16
    assert patch_size_for_pixel_size(0.6, 8.0) == 13
    # a very coarse pixel size still yields a usable p, never 0
    assert patch_size_for_pixel_size(100.0, 8.0) == 1
    # and the map is monotone: finer sampling -> larger p
    ps = [
        patch_size_for_pixel_size(q, 12.8)
        for q in (0.703, 0.652, 0.650, 0.646, 0.598)
    ]
    assert ps == sorted(ps) == [18, 20, 20, 20, 21]


def test_crop_px_is_an_exact_multiple_of_patch_size():
    g = CropGeometry.build(pixel_size=0.65, target_um=8.0, n_tokens=4)
    assert g.crop_px == 4 * g.patch_size
    assert g.n_patches == 16
    assert g.crop_px % g.patch_size == 0


def test_place_box_shifts_interior_cell_away_from_border():
    # cell bbox is strictly interior (max 250 < 256) but the 32 px window
    # centred on it would run to 261, so it must snap to the boundary
    bbox = (240, 250, 240, 250)
    top, left = place_box(bbox, 32, (256, 256))
    assert (top, left) == (224, 224)
    assert top + 32 == 256 and left + 32 == 256
    # and the cell is still fully covered by the shifted window
    assert bbox[0] >= top and bbox[1] <= top + 32
    assert bbox[2] >= left and bbox[3] <= left + 32


def test_plan_patches_keeps_interior_near_border_cell_in_bounds_and_covered():
    H = W = 128
    g = CropGeometry.build(pixel_size=1.0, target_um=8.0, n_tokens=2)  # 16 px
    mask = np.zeros((H, W), dtype=np.int64)
    mask[118:126, 118:126] = 7  # interior, 2 px from the border
    crops, skipped = plan_patches(mask, (H, W), g, image_id="im", channel_ids=(0,))
    assert skipped == [] and len(crops) == 1
    c = crops[0]
    assert c.top + g.crop_px <= H and c.left + g.crop_px <= W
    assert c.covers_cell and c.cell_hw == (8, 8)
    assert c.n_valid == int(c.valid.sum()) >= 1


def test_plan_patches_marks_cells_larger_than_the_window():
    g = CropGeometry.build(pixel_size=1.0, target_um=8.0, n_tokens=2)  # 16 px
    mask = np.zeros((128, 128), dtype=np.int64)
    mask[20:80, 20:80] = 1  # 60 px cell vs 16 px window
    crops, _ = plan_patches(mask, (128, 128), g)
    assert len(crops) == 1 and not crops[0].covers_cell
    # and it can be rejected instead of clipped
    crops, skipped = plan_patches(mask, (128, 128), g, require_full_cell=True)
    assert crops == [] and "larger than the" in skipped[0]["reason"]


def test_place_box_clamps_in_bounds_and_centres_when_there_is_room():
    # plenty of room: centred on the bbox
    assert place_box((40, 60, 40, 60), 32, (256, 256)) == (34, 34)
    # near the border: clamped, still the full crop size (no padding needed)
    top, left = place_box((0, 10, 0, 10), 32, (256, 256))
    assert (top, left) == (0, 0)
    top, left = place_box((250, 256, 250, 256), 32, (256, 256))
    assert (top, left) == (224, 224)
    # crop larger than the image cannot be placed
    assert place_box((0, 10, 0, 10), 512, (256, 256)) is None


def test_token_coverage_matches_manual_patch_overlap():
    p = 4
    mask = np.zeros((8, 8), dtype=bool)
    mask[1:3, 1:3] = True  # only touches patch (0, 0)
    cov = token_coverage(mask, p)
    assert cov.shape == (2, 2)
    assert cov[0, 0] and not cov[0, 1] and not cov[1, 0] and not cov[1, 1]

    # 4/16 = 0.25 of the patch, so it only passes a threshold below that
    assert not token_coverage(mask, p, mode="frac", threshold=0.3)[0, 0]
    assert token_coverage(mask, p, mode="frac", threshold=0.2)[0, 0]


def test_token_coverage_torch_agrees_with_numpy():
    rng = np.random.default_rng(0)
    mask = rng.random((12, 12)) > 0.6
    a = token_coverage(mask, 4)
    b = token_coverage_torch(torch.from_numpy(mask), 4)[0, 0].numpy()
    assert np.array_equal(a, b)


class _FakeStack(TiffStack):
    """Plane loader that returns an in-memory tensor channel instead of a TIFF."""

    def __init__(self, stack: torch.Tensor):
        self._stack = stack
        super().__init__(
            paths=[f"ch{i}" for i in range(stack.shape[0])],
            plane_loader=lambda p: stack[int(p[2:]), 0].numpy(),
        )

def _mask_with_three_cells():
    mask = np.zeros((64, 64), dtype=np.int64)
    mask[10:20, 12:22] = 1
    mask[30:42, 22:40] = 2
    # a cell too close to an edge to fit the crop after clamping is still fine,
    # clamping keeps it in-bounds; one tiny cell checks min_valid_tokens
    mask[1:3, 1:3] = 3
    return mask


def test_plan_patches_crops_tile_exactly_and_carry_a_valid_mask():
    g = CropGeometry.build(pixel_size=1.0, target_um=8.0, n_tokens=2)
    assert g.crop_px == 16
    crops, skipped = plan_patches(_mask_with_three_cells(), (64, 64), g, image_id="im0")
    assert skipped == []
    assert {c.cell_id for c in crops} == {1, 2, 3}
    for c in crops:
        assert c.coverage.shape == (2, 2)
        assert c.valid.shape == (4,)
        assert c.n_valid == int(c.valid.sum())
        assert c.top >= 0 and c.left >= 0
        assert c.top + g.crop_px <= 64 and c.left + g.crop_px <= 64


def test_plan_patches_skips_instead_of_padding():
    g = CropGeometry.build(pixel_size=1.0, target_um=8.0, n_tokens=4)  # 32 px
    mask = np.zeros((20, 20), dtype=np.int64)
    mask[5:15, 5:15] = 1
    crops, skipped = plan_patches(mask, (20, 20), g)
    assert crops == []
    assert len(skipped) == 1 and "exceeds image" in skipped[0]["reason"]


def test_plan_patches_respects_min_valid_tokens():
    g = CropGeometry.build(pixel_size=1.0, target_um=8.0, n_tokens=1)  # 8 px
    mask = np.zeros((32, 32), dtype=np.int64)
    mask[10:11, 10:11] = 1  # one pixel, but always inside the single patch
    crops, skipped = plan_patches(mask, (32, 32), g, min_valid_tokens=2)
    assert crops == [] and len(skipped) == 1


def test_dataset_reads_variable_size_crops_and_collates_padding_free():
    # two images with different pixel_size -> different p, hence different crop_px
    im_a = torch.randn(3, 1, 96, 96)
    im_b = torch.randn(2, 1, 96, 96)
    mask = np.zeros((96, 96), dtype=np.int64)
    mask[40:60, 40:60] = 1
    mask[20:32, 60:74] = 2

    g_a = CropGeometry.build(pixel_size=1.0, target_um=8.0, n_tokens=2)
    g_b = CropGeometry.build(pixel_size=0.5, target_um=8.0, n_tokens=2)
    crops_a, _ = plan_patches(mask, (96, 96), g_a, image_id="a", channel_ids=(0, 1, 2))
    crops_b, _ = plan_patches(mask, (96, 96), g_b, image_id="b", channel_ids=(0, 4))
    assert g_a.patch_size != g_b.patch_size

    readers = {"a": _FakeStack(im_a), "b": _FakeStack(im_b)}
    ds = CellCropDataset(crops=crops_a + crops_b, readers=readers)
    batch = collate_crops([ds[i] for i in range(len(ds))])

    # list-shaped, no padding; token count is identical across instruments
    assert isinstance(batch["images"], list)
    assert [im.shape[0] for im in batch["images"]] == [3, 3, 2, 2]
    assert [im.shape[-1] for im in batch["images"]] == [16, 16, 32, 32]
    assert len({im.shape[-1] // b for im, b in zip(batch["images"], batch["patch_sizes"])}) == 1
    assert all(t.shape == (4,) for t in batch["valid"])


def test_cache_roundtrip(tmp_path):
    pytest.importorskip("tifffile")
    from fuse_channels.crops import write_crop_cache
    import json

    g = CropGeometry.build(pixel_size=1.0, target_um=8.0, n_tokens=2)
    crops, _ = plan_patches(
        _mask_with_three_cells(), (64, 64), g, image_id="im0", channel_ids=(0, 1, 2)
    )
    index = []
    for i, c in enumerate(crops):
        write_crop_cache(tmp_path, c, torch.randn(3, 1, g.crop_px, g.crop_px), index)
    (tmp_path / "index.json").write_text(json.dumps(index))

    ds = CellCropDataset(index_path=tmp_path / "index.json")
    assert len(ds) == len(crops)
    item = ds[0]
    assert item["images"].shape == (3, 1, 16, 16)
    assert item["patch_size"] == 8 and item["pixel_size"] == 1.0
    assert item["valid"].dtype == torch.bool and item["valid"].shape == (4,)


class _FakeStack(TiffStack):
    """Plane loader that returns an in-memory tensor channel instead of a TIFF."""

    def __init__(self, stack: torch.Tensor):
        self._stack = stack
        super().__init__(
            paths=[f"ch{i}" for i in range(stack.shape[0])],
            plane_loader=lambda p: stack[int(p[2:]), 0].numpy(),
        )
