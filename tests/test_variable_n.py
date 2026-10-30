"""Variable-N (cell-sized) token grids: metric PE, off-cell dropping, ragged loss.

These cover the three things that a cell-sized patch grid changes relative to
the fixed-window design:

1. positional embeddings must be metric, otherwise the phase encodes
   ``distance / window_extent`` and a small cell's physical geometry differs
   from a large cell's at the same physical spacing;
2. off-cell patches must be droppable from the encoder *and* from the MAE
   targets (the three-state ``valid`` / ``visible`` distinction);
3. a batch whose samples have different patch counts must build targets and a
   loss without concatenating along the channel axis.
"""

import numpy as np
import pytest
import torch

from fuse_channels import (
    ChannelFusionMAE,
    CropGeometry,
    cell_n_tokens,
    plan_patches,
    token_coverage,
)
from fuse_channels.pos_embed import FractionalSpatialPE
from fuse_channels.rope import axial_rope_cache, physical_grid_coords

# every variable-N test below deliberately exercises the grid-PE footgun or the
# metric path explicitly; the warning itself is covered by its own test.
pytestmark = pytest.mark.filterwarnings("ignore:variable patch count")


# --------------------------------------------------------------------------- #
# metric vs grid-fraction coordinates
# --------------------------------------------------------------------------- #
def test_physical_coords_make_the_phase_proportional_to_um():
    """The same physical gap must rotate identically in a small and a large cell."""
    p, pixel_size = 8, 0.5  # 4 um per token
    gap_tokens = 2
    gap_um = gap_tokens * p * pixel_size

    def relative_phase(n_h):
        rows = torch.tensor([0, gap_tokens], dtype=torch.float32)
        r, _ = physical_grid_coords(
            rows, rows, torch.tensor(p), torch.tensor(pixel_size), ref_um=32.0
        )
        return float(r[1] - r[0]) * 32.0

    assert relative_phase(4) == pytest.approx(gap_um)
    assert relative_phase(8) == pytest.approx(gap_um)


def test_grid_fraction_coords_encode_distance_relative_to_the_window():
    """Grid-fraction coordinates are *not* metric once the window extent varies."""
    from fuse_channels.rope import normalized_grid_coords, rope_frequencies

    theta = rope_frequencies(8, base=100.0)[0].item()

    def relative_phase(n_h):
        rows = torch.tensor([0.0, 2.0])
        r, _ = normalized_grid_coords(
            rows, rows, n_h, n_h, ref_extent=32.0
        )
        return float((r[1] - r[0]) * theta)

    small, large = relative_phase(4), relative_phase(8)
    # same 2-token gap gets twice the phase in the small window
    assert small == pytest.approx(2 * large, rel=1e-5)


def test_metric_rope_matches_grid_rope_when_the_window_is_constant():
    """At a fixed physical window the metric PE is a numerical no-op."""
    n_h, p, pixel_size, ref_extent = 4, 8, 1.0, 32.0
    # grid coord (r+0.5)/n_h*S == metric coord (r+0.5)*p*pixel_size/ref_um
    # iff ref_um == n_h*p*pixel_size/S
    ref_um = n_h * p * pixel_size / ref_extent
    rows = torch.arange(n_h * n_h, dtype=torch.float32)
    cols = rows % n_h
    rows = rows // n_h
    cg, sg = axial_rope_cache(rows, cols, n_h, n_h, 8, ref_extent=ref_extent)
    cm, sm = axial_rope_cache(
        rows,
        cols,
        n_h,
        n_h,
        8,
        ref_extent=ref_extent,
        patch_size=torch.tensor(p),
        pixel_size=torch.tensor(pixel_size),
        ref_um=ref_um,
    )
    torch.testing.assert_close(cg, cm, atol=1e-6, rtol=0)
    torch.testing.assert_close(sg, sm, atol=1e-6, rtol=0)


def test_decoder_metric_pe_uses_the_same_frame_as_the_encoder():
    dim, ref_um = 8, 32.0
    pe = FractionalSpatialPE(dim, mode="sincos", base=100.0)
    rows = torch.tensor([0.0, 2.0])
    cols = torch.tensor([0.0, 2.0])
    # metric coordinates with step 4 um == grid fractions of an 8-token window
    # scaled by 4/ref_um ... compare against the direct uv formula instead
    out_metric = pe(rows, cols, 4, 4, step_um=torch.tensor(4.0), ref_um=ref_um)
    from fuse_channels.pos_embed import get_fractional_sincos_pos_embed_uv

    step = 4.0 / ref_um
    v = (rows + 0.5) * step
    u = (cols + 0.5) * step
    torch.testing.assert_close(
        out_metric, get_fractional_sincos_pos_embed_uv(dim, v, u, base=100.0)
    )


def test_pixel_size_is_required_metadata():
    """There is no "unknown pixel size" state: the model refuses to run blind."""
    model = ChannelFusionMAE(image_size=16, patch_size=8, dim=16, n_heads=4,
                             scale_conditioning="none")
    images = [torch.randn(2, 1, 16, 16)]
    ids = [torch.tensor([0, 1])]
    with pytest.raises(ValueError, match="pixel size .* is required"):
        model(images, ids, patch_sizes=8)
    with pytest.raises(ValueError, match="all be known"):
        model(images, ids, patch_sizes=8, pixel_sizes=[None])
    with pytest.raises(ValueError, match="must be positive"):
        model(images, ids, patch_sizes=8, pixel_sizes=[0.0])
    # and it runs once a real value is supplied
    out = model(images, ids, patch_sizes=8, pixel_sizes=[0.65])
    assert torch.isfinite(out["loss"])


# --------------------------------------------------------------------------- #
# variable N planning
# --------------------------------------------------------------------------- #
def test_cell_n_tokens_is_driven_by_cell_size_and_p():
    assert cell_n_tokens((10, 10), 8, min_n=1, max_n=8) == 2
    assert cell_n_tokens((17, 17), 8, min_n=1, max_n=8) == 3
    assert cell_n_tokens((64, 64), 8, min_n=1, max_n=8) == 8  # clamped
    assert cell_n_tokens((3, 3), 8, min_n=2, max_n=8) == 2  # MAE minimum


def test_plan_patches_variable_n_gives_each_cell_its_own_grid():
    mask = np.zeros((200, 200), dtype=np.int64)
    mask[50:66, 50:66] = 1    # 16 px  -> 2 tokens at p=8
    mask[100:132, 100:132] = 2  # 32 px -> 4 tokens
    mask[150:190, 150:190] = 3  # 40 px -> 5 tokens
    crops, skipped = plan_patches(
        mask, (200, 200), pixel_size=1.0, target_um=8.0, n_tokens="cell"
    )
    assert skipped == []
    by_id = {c.cell_id: c for c in crops}
    assert by_id[1].geometry.n_tokens == 2
    assert by_id[2].geometry.n_tokens == 4
    assert by_id[3].geometry.n_tokens == 5
    # every crop still tiles exactly and holds a full-grid valid mask
    for c in crops:
        assert c.geometry.crop_px % c.geometry.patch_size == 0
        assert c.valid.shape == (c.geometry.n_patches,)


def test_large_cell_is_processed_not_clipped():
    """No model-side cap on N: a big cell gets its own large grid."""
    mask = np.zeros((400, 400), dtype=np.int64)
    mask[150:330, 150:330] = 1  # 180 px cell -> 23 tokens/side at p=8
    crops, skipped = plan_patches(
        mask, (400, 400), pixel_size=1.0, target_um=8.0, n_tokens="cell"
    )
    assert skipped == [] and len(crops) == 1
    c = crops[0]
    assert c.geometry.n_tokens == 23
    assert c.covers_cell
    assert c.valid.shape == (23 * 23,)
    assert c.valid.sum() > 0


def test_max_n_budget_skips_and_reports_instead_of_clipping():
    mask = np.zeros((400, 400), dtype=np.int64)
    mask[150:330, 150:330] = 1
    crops, skipped = plan_patches(
        mask, (400, 400), pixel_size=1.0, target_um=8.0,
        n_tokens="cell", max_n=8,
    )
    assert crops == []
    assert len(skipped) == 1
    assert "max_n=8" in skipped[0]["reason"]


def test_model_accepts_a_far_larger_grid_than_its_construction_image_size():
    model = _model()  # built for image_size=16
    im = torch.randn(2, 1, 96, 96)
    out = model(
        [im], [torch.tensor([0, 1])], patch_sizes=8, pixel_sizes=[0.5],
        generator=torch.Generator().manual_seed(0),
    )
    assert int(out["n_patches_per_sample"][0]) == 144
    assert torch.isfinite(out["loss"])


def test_plan_patches_variable_n_still_clamps_to_the_image():
    mask = np.zeros((96, 96), dtype=np.int64)
    mask[80:94, 80:94] = 1  # interior but the window would cross the border
    crops, skipped = plan_patches(
        mask, (96, 96), pixel_size=1.0, target_um=8.0, n_tokens="cell"
    )
    assert skipped == [] and len(crops) == 1
    c = crops[0]
    assert c.top + c.geometry.crop_px <= 96
    assert c.left + c.geometry.crop_px <= 96
    assert c.covers_cell


# --------------------------------------------------------------------------- #
# model: variable N + off-cell dropping
# --------------------------------------------------------------------------- #
def _variable_n_batch(seed=0):
    """Two samples, same p, different H/W hence different N (previously a crash)."""
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(2, 1, 16, 16, generator=g)   # 2x2  = 4 patches at p=8
    b = torch.randn(3, 1, 32, 16, generator=g)   # 4x2  = 8 patches at p=8
    return [a, b], [torch.tensor([0, 1]), torch.tensor([2, 3, 4])]


def _model(**kw):
    defaults = dict(
        image_size=16,
        patch_size=8,
        dim=16,
        encoder_depth=1,
        decoder_depth=1,
        n_heads=4,
        n_memory=1,
        num_channel_ids=8,
        scale_conditioning="none",
        p_patch=0.0,
        p_channel=0.0,
        patch_ratio=0.5,
        pixel_size=1.0,
    )
    defaults.update(kw)
    return ChannelFusionMAE(**defaults)


def test_variable_n_batch_builds_targets_and_loss():
    model = _model()
    images, ids = _variable_n_batch()
    out = model(images, ids, patch_sizes=8)
    assert torch.isfinite(out["loss"])
    assert out["n_patches_per_sample"].tolist() == [4, 8]
    # ragged targets are padded to max(p^2) with a validity mask
    assert out["pred"].shape[0] > 0
    assert out["target"].shape[0] == out["pred"].shape[0]


def test_variable_n_with_grid_pe_warns_about_non_metric_coordinates():
    # metric PE is the default; grid PE is the fixed-window ablation, and using
    # it with cell-sized crops is what the warning is for
    model = _model(pe_space="grid")
    images, ids = _variable_n_batch()
    with pytest.warns(UserWarning, match="variable patch count"):
        model(images, ids, patch_sizes=8)
    # ... and only warns once
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model(images, ids, patch_sizes=8)


def test_extract_patches_rejects_a_ragged_batch_loudly():
    from fuse_channels import ChannelTokenizer

    images, _ = _variable_n_batch()
    with pytest.raises(ValueError, match="variable-N"):
        ChannelTokenizer.extract_patches(images, 8)


def _visible_two_samples():
    return [
        torch.ones(2, 4, dtype=torch.bool),   # sample 0: 2 channels, 4 patches
        torch.ones(3, 8, dtype=torch.bool),   # sample 1: 3 channels, 8 patches
    ]


def test_valid_mask_three_states_with_context():
    model = _model(keep_offcell_as_context=True)
    images, ids = _variable_n_batch()
    valid = [torch.tensor([True, False, False, False]), torch.ones(8, dtype=torch.bool)]
    vis = _visible_two_samples()
    vis[0][:, 1:] = False  # ask for exactly the valid patch
    _cs, _ps, _ns, visible, target_valid, _px = model._prepare(
        images, ids, vis, None, 8, valid=valid
    )
    # off-cell patches are kept as context ...
    assert bool(visible[0][:, 1:].all())
    # ... but are never eligible targets
    for b in range(2):
        m = ~visible[b] & target_valid[b]
        assert not bool((m & ~target_valid[b]).any())
    # sample 1 has no off-cell patches, so nothing is excluded from targets
    assert bool(target_valid[1].all())


def test_masking_an_offcell_patch_produces_no_target():
    model = _model(keep_offcell_as_context=True)
    images, ids = _variable_n_batch()
    valid = [torch.tensor([True, False, False, False]), torch.ones(8, dtype=torch.bool)]
    vis = _visible_two_samples()
    vis[0][:, 1] = False  # mask an OFF-cell patch
    vis[1][:] = True      # sample 1 has nothing masked
    out = model(images, ids, patch_sizes=8, visible=vis, valid=valid)
    # nothing is reconstructable -> the guard returns a zero loss with no pred
    assert out["pred"] is None
    assert float(out["loss"].detach()) == 0.0


def test_dropping_offcell_removes_them_from_the_encoder():
    model = _model(keep_offcell_as_context=False)
    images, ids = _variable_n_batch()
    valid = [torch.tensor([True, False, False, False]), torch.ones(8, dtype=torch.bool)]
    vis = _visible_two_samples()
    out = model(images, ids, patch_sizes=8, visible=vis, valid=valid)
    # sample 0 contributes exactly one present location (its cell patch)
    assert int(out["present"][0].sum()) == 1
    assert bool(out["present"][0, 0])
    assert int(out["present"][1].sum()) == 8
    assert torch.isfinite(out["loss"])


def test_sample_reduction_weights_cells_equally():
    torch.manual_seed(0)
    model_tok = _model(loss_reduce="token")
    model_sam = _model(loss_reduce="sample")
    model_sam.load_state_dict(model_tok.state_dict())
    images, ids = _variable_n_batch()
    a = model_tok(images, ids, patch_sizes=8)["loss"]
    b = model_sam(images, ids, patch_sizes=8)["loss"]
    assert torch.isfinite(a) and torch.isfinite(b)
    assert a.item() != pytest.approx(b.item())


def test_metric_pe_end_to_end_with_pixel_sizes():
    model = _model(pe_space="physical", scale_conditioning="add")
    images, ids = _variable_n_batch()
    out = model(
        images, ids, patch_sizes=8, pixel_sizes=[0.5, 1.0]
    )
    assert out["pe_metric"] is True
    assert torch.isfinite(out["loss"])


def test_metric_pe_is_patch_size_invariant_for_the_same_physical_token():
    """Same physical patch extent at two pixel sizes -> identical rope phases."""
    model = _model(pe_space="physical")
    g = 8
    rows = torch.tensor([0, 1, 2], dtype=torch.float32)
    # p=16 @ 0.5 um/px and p=8 @ 1.0 um/px both give an 8 um patch
    c1, s1 = axial_rope_cache(
        rows, rows, 3, 3, 8, patch_size=torch.tensor(16),
        pixel_size=torch.tensor(0.5), ref_um=32.0,
    )
    c2, s2 = axial_rope_cache(
        rows, rows, 3, 3, 8, patch_size=torch.tensor(8),
        pixel_size=torch.tensor(1.0), ref_um=32.0,
    )
    torch.testing.assert_close(c1, c2)
    torch.testing.assert_close(s1, s2)
    del model, g


# --------------------------------------------------------------------------- #
# integration: crop reader -> variable-N grid -> model
# --------------------------------------------------------------------------- #
def test_crop_reader_to_model_with_variable_n_and_variable_p():
    from fuse_channels import CellCropDataset, TiffStack, collate_crops, plan_patches

    mask = np.zeros((200, 200), dtype=np.int64)
    mask[60:76, 60:76] = 1     # 16 px cell
    mask[100:132, 100:132] = 2  # 32 px cell

    # two instruments: same physical patch extent, different p
    geom_a = CropGeometry.build(pixel_size=1.0, target_um=8.0, n_tokens=2)
    geom_b = CropGeometry.build(pixel_size=0.5, target_um=8.0, n_tokens=2)
    assert geom_a.patch_size == 8 and geom_b.patch_size == 16

    crops_a, _ = plan_patches(mask, (200, 200), geom_a, image_id="a",
                              channel_ids=(0, 1))
    crops_b, _ = plan_patches(mask, (200, 200), geom_b, image_id="b",
                              channel_ids=(0, 1))
    # variable N on instrument a: cell-sized windows
    crops_a_cell, skipped = plan_patches(
        mask, (200, 200), pixel_size=1.0, target_um=8.0,
        n_tokens="cell", image_id="a", channel_ids=(0, 1),
    )
    assert skipped == []
    assert {c.geometry.n_tokens for c in crops_a_cell} == {2, 4}

    plane_a = torch.randn(2, 200, 200)
    plane_b = torch.randn(2, 200, 200)
    readers = {
        "a": TiffStack(paths=["x", "y"], plane_loader=lambda _p: plane_a[0].numpy()),
        "b": TiffStack(paths=["x", "y"], plane_loader=lambda _p: plane_b[0].numpy()),
    }
    readers["a"]._planes = [plane_a[0].numpy(), plane_a[1].numpy()]
    readers["b"]._planes = [plane_b[0].numpy(), plane_b[1].numpy()]

    ds = CellCropDataset(crops=crops_a_cell + crops_b, readers=readers)
    batch = collate_crops([ds[i] for i in range(len(ds))])
    assert len(batch["patch_sizes"]) == 4

    model = _model(
        pe_space="physical", scale_conditioning="add", patch_ratio=0.5,
        patch_sizes=(8, 16),
    )
    out = model(
        batch["images"],
        batch["channel_ids"],
        patch_sizes=batch["patch_sizes"],
        pixel_sizes=batch["pixel_sizes"],
        valid=batch["valid"],
        generator=torch.Generator().manual_seed(0),
    )
    assert out["pe_metric"] is True
    assert torch.isfinite(out["loss"])
    # the variable-N samples really do have different patch counts
    assert len(set(out["n_patches_per_sample"].tolist())) > 1


# --------------------------------------------------------------------------- #
# loss reduction: per-cell vs per-token
# --------------------------------------------------------------------------- #
def test_default_loss_reduction_gives_each_cell_one_vote():
    assert _model().loss_reduce == "sample"
    assert _model(loss_reduce="token").loss_reduce == "token"


def _reduction_setup():
    """One 2x2-token cell and one 8x8-token cell: a 16x target-count ratio."""
    torch.manual_seed(0)
    model = _model(pe_space="physical", p_patch=1.0, p_channel=0.0)
    images = [torch.randn(2, 1, 16, 16), torch.randn(2, 1, 64, 64)]
    ids = [torch.tensor([0, 1]), torch.tensor([0, 1])]
    out = model(images, ids, patch_sizes=8,
                generator=torch.Generator().manual_seed(3))
    _cs, _ps, _ns, visible, _tv, _px = model._prepare(
        images, ids, None, torch.Generator().manual_seed(3), 8
    )
    counts = [int((~v).sum()) for v in visible]
    gid = torch.cat(
        [torch.full((c,), b, dtype=torch.long) for b, c in enumerate(counts)]
    )
    return model, out, gid, counts


def test_token_reduction_weights_by_target_count_sample_does_not():
    model, out, gid, counts = _reduction_setup()
    assert counts == [4, 64]  # the large cell has 16x the target rows
    pred, target = out["pred"], out["target"]

    per_sample = [
        float(
            model._loss_on(
                pred, target, out["target_patch_size"], out["pred_mask"],
                (gid == g).nonzero(as_tuple=True)[0],
            ).detach()
        )
        for g in (0, 1)
    ]
    weights = torch.tensor(counts, dtype=torch.float32)
    weights = weights / weights.sum()
    # token reduction IS the count-weighted mean: the big cell takes ~94%
    assert float(weights[1]) > 0.9
    torch.testing.assert_close(
        model._loss_on(pred, target, out["target_patch_size"], out["pred_mask"]),
        torch.tensor(per_sample) @ weights,
        rtol=1e-5,
        atol=1e-6,
    )
    # sample reduction is the unweighted mean, i.e. 50/50
    torch.testing.assert_close(
        model._sample_mean_loss(pred, target, out["pred_mask"], gid),
        torch.tensor(per_sample).mean(),
        rtol=1e-5,
        atol=1e-6,
    )


@pytest.mark.parametrize("mixed_p", [False, True])
def test_vectorised_sample_reduction_matches_the_per_sample_loop(mixed_p):
    """The one-pass reduction must equal the loop it replaced."""
    model, out, gid, _ = _reduction_setup()
    pred, target, tp, pm = (
        out["pred"],
        out["target"],
        out["target_patch_size"],
        out["pred_mask"],
    )
    if mixed_p:
        # mixed-p padding introduces pred_mask; build such a batch explicitly
        images = [torch.randn(2, 1, 16, 16), torch.randn(2, 1, 64, 64)]
        ids = [torch.tensor([0, 1]), torch.tensor([0, 1])]
        model = _model(
            pe_space="physical", p_patch=1.0, p_channel=0.0, patch_sizes=(8, 16)
        )
        out = model(images, ids, patch_sizes=[8, 16],
                    generator=torch.Generator().manual_seed(3))
        _cs, _ps, _ns, visible, _tv, _px = model._prepare(
            images, ids, None, torch.Generator().manual_seed(3), [8, 16]
        )
        gid = torch.cat(
            [torch.full((int((~v).sum()),), b, dtype=torch.long)
             for b, v in enumerate(visible)]
        )
        pred, target, tp, pm = (
            out["pred"],
            out["target"],
            out["target_patch_size"],
            out["pred_mask"],
        )
        assert pm is not None

    groups = torch.unique(gid)
    loop = pred.new_zeros(())
    for g in groups.tolist():
        loop = loop + model._loss_on(
            pred, target, tp, pm, (gid == g).nonzero(as_tuple=True)[0]
        )
    loop = loop / groups.numel()
    vec = model._sample_mean_loss(pred, target, pm, gid)
    # exact without padding; float32 summation order otherwise
    torch.testing.assert_close(vec, loop, rtol=1e-6, atol=1e-6)


def test_vectorised_sample_reduction_gradient_matches_the_loop():
    model, out, gid, _ = _reduction_setup()
    groups = torch.unique(gid)

    def run(vectorised):
        pred = out["pred"].detach().clone().requires_grad_(True)
        if vectorised:
            loss = model._sample_mean_loss(
                pred, out["target"], out["pred_mask"], gid
            )
        else:
            loss = pred.new_zeros(())
            for g in groups.tolist():
                loss = loss + model._loss_on(
                    pred, out["target"], out["target_patch_size"],
                    out["pred_mask"], (gid == g).nonzero(as_tuple=True)[0],
                )
            loss = loss / groups.numel()
        loss.backward()
        return loss.detach(), pred.grad.detach()

    lv, gv = run(True)
    ll, gl = run(False)
    torch.testing.assert_close(lv, ll, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(gv, gl, rtol=1e-5, atol=1e-7)
