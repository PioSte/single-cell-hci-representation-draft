import math

import pytest
import torch

from fuse_channels import DCPVisibilitySampler, HierarchicalChannelSampler


def _images(C_list, N, seed=0):
    g = torch.Generator().manual_seed(seed)
    # images are [C,1,H,W]; the sampler only uses C and the device, so H=W=1
    return [torch.randn(C, 1, 1, 1, generator=g) for C in C_list]


def _masked_per_row(visible):
    return (~visible).sum(dim=1).tolist()


def test_constructor_validation():
    with pytest.raises(ValueError):
        DCPVisibilitySampler(4, p_patch=-0.1)
    with pytest.raises(ValueError):
        DCPVisibilitySampler(4, p_channel=-0.1)
    with pytest.raises(ValueError):
        DCPVisibilitySampler(4, p_patch=0.6, p_channel=0.5)
    with pytest.raises(ValueError):
        DCPVisibilitySampler(4, patch_ratio=0.0)
    with pytest.raises(ValueError):
        DCPVisibilitySampler(4, patch_ratio=1.0)
    # a used patch region must be able to mask at least one patch
    with pytest.raises(ValueError):
        DCPVisibilitySampler(1, p_patch=1.0, p_channel=0.0)
    with pytest.raises(ValueError):
        DCPVisibilitySampler(1, p_patch=0.0, p_channel=0.0)
    # channel-only N=1 is allowed (no patch region is used)
    DCPVisibilitySampler(1, p_patch=0.0, p_channel=1.0)
    with pytest.raises(ValueError):
        DCPVisibilitySampler(4, mode_sampling="step")
    with pytest.raises(ValueError):
        DCPVisibilitySampler(0)


def test_patch_region_masks_exact_fixed_count_per_channel():
    """p_patch=1: exactly floor(N*patch_ratio) masked patches per channel."""
    N, ratio, C = 8, 0.75, 6
    k = math.floor(N * ratio)
    masker = DCPVisibilitySampler(N, p_patch=1.0, p_channel=0.0, patch_ratio=ratio)

    for seed in range(10):
        g = torch.Generator().manual_seed(seed)
        visible = masker(_images([C], N, seed=seed), generator=g)
        assert masker.last_modes == ["patch"]
        v = visible[0]
        assert v.shape == (C, N)
        assert _masked_per_row(v) == [k] * C
        assert all(bool(row.any()) and bool((~row).any()) for row in v)

    # the spatial mask is drawn per channel, not replicated across channels
    g = torch.Generator().manual_seed(3)
    v = masker(_images([C], N), generator=g)[0]
    assert len({tuple(row.tolist()) for row in v}) > 1


def test_channel_region_respects_min_visible_channels_by_construction():
    """C=5, min_visible_channels=4: HCS draws 4 or 5 whole visible channels."""
    C, N, min_visible = 5, 8, 4
    masker = DCPVisibilitySampler(
        N,
        p_patch=0.0,
        p_channel=1.0,
        min_visible_channels=min_visible,
    )
    counts = set()
    saw_dropped = False

    for seed in range(200):
        g = torch.Generator().manual_seed(seed)
        v = masker(_images([C], N, seed=seed), generator=g)[0]
        assert masker.last_modes == ["channel"]
        # whole channels only: no partial rows, so a dropped channel cannot
        # have been patched back to visibility
        for row in v:
            assert bool(row.all()) or bool((~row).all())
        dropped = ~v.any(dim=1)
        n_visible = C - int(dropped.sum())
        assert min_visible <= n_visible <= C
        counts.add(n_visible)
        saw_dropped |= bool(dropped.any())
        assert not bool(v[dropped].any())

    assert counts == {min_visible, C}
    assert saw_dropped


def test_channel_region_is_whole_channel_hcs():
    """p_channel=1: whole-channel HCS with 1..C visible channels."""
    C, N = 8, 8
    masker = DCPVisibilitySampler(N, p_patch=0.0, p_channel=1.0)
    counts = set()
    saw_dropped = False

    for seed in range(200):
        g = torch.Generator().manual_seed(seed)
        v = masker(_images([C], N, seed=seed), generator=g)[0]
        assert masker.last_modes == ["channel"]
        # whole channels: every row is all-visible or all-masked
        for row in v:
            assert bool(row.all()) or bool((~row).all())
        n_visible = int(v.any(dim=1).sum())
        assert 1 <= n_visible <= C
        counts.add(n_visible)
        saw_dropped |= bool((~v).all(dim=1).any())

    # HCS covers m ~ U{1..C}: every visible count appears
    assert counts == set(range(1, C + 1))
    # dropped channels become fully masked rows (reconstruction targets)
    assert saw_dropped


def test_hierarchical_channel_sampler_api():
    sampler = HierarchicalChannelSampler()
    C = 6
    counts = set()
    for seed in range(200):
        g = torch.Generator().manual_seed(seed)
        visible = sampler(C, generator=g)
        assert visible.shape == (C,)
        assert visible.dtype == torch.bool
        n = int(visible.sum())
        assert 1 <= n <= C
        counts.add(n)
    assert counts == set(range(1, C + 1))

    g = torch.Generator().manual_seed(0)
    assert sampler(1, generator=g).tolist() == [True]

    # min_visible raises the lower bound of the m ~ U{min_visible..C} draw
    min_counts = set()
    for seed in range(100):
        g = torch.Generator().manual_seed(seed)
        visible = sampler(C, generator=g, min_visible=4)
        n = int(visible.sum())
        assert 4 <= n <= C
        min_counts.add(n)
    assert min_counts == {4, 5, 6}

    with pytest.raises(ValueError):
        sampler(C, min_visible=0)
    with pytest.raises(ValueError):
        sampler(C, min_visible=C + 1)


def test_hcs_dropped_channels_are_fully_masked_rows():
    C, N = 7, 8
    masker = DCPVisibilitySampler(N, p_patch=0.0, p_channel=1.0)
    found = False
    for seed in range(20):
        g = torch.Generator().manual_seed(seed)
        v = masker(_images([C], N, seed=seed), generator=g)[0]
        dropped = ~v.any(dim=1)
        if bool(dropped.any()):
            found = True
            assert bool((~v[dropped]).all())
            assert bool(v[~dropped].all())
    assert found


def test_combined_region_unions_patch_and_channel_masks():
    """p_patch=p_channel=0: mask = patch_mask | channel_mask (Eq. 1 region 3)."""
    C, N, ratio = 6, 8, 0.75
    k = math.floor(N * ratio)
    masker = DCPVisibilitySampler(N, p_patch=0.0, p_channel=0.0, patch_ratio=ratio)
    saw_drop, saw_no_drop = False, False

    for seed in range(100):
        g = torch.Generator().manual_seed(seed)
        v = masker(_images([C], N, seed=seed), generator=g)[0]
        assert masker.last_modes == ["combined"]
        assert bool(v.any()) and not bool(v.all())
        per_row = _masked_per_row(v)
        # patch region masks k per channel; channel region masks whole rows
        assert all(x in (k, N) for x in per_row)
        n_dropped = sum(1 for x in per_row if x == N)
        n_visible = C - n_dropped
        assert int((~v).sum()) == n_visible * k + n_dropped * N
        saw_drop |= n_dropped > 0
        saw_no_drop |= n_dropped == 0

    # both HCS outcomes occur: the channel region really drops whole rows,
    # and it can also keep every channel (m == C)
    assert saw_drop and saw_no_drop


def test_guards_never_mask_everything_or_starve_channels():
    configs = [
        dict(p_patch=1.0, p_channel=0.0, patch_ratio=0.875, min_visible_channels=1),
        dict(p_patch=0.0, p_channel=1.0, patch_ratio=0.75, min_visible_channels=2),
        dict(p_patch=0.0, p_channel=0.0, patch_ratio=0.9, min_visible_channels=2),
    ]
    C_list = [1, 2, 3, 5, 8]
    for cfg in configs:
        masker = DCPVisibilitySampler(8, **cfg)
        for seed in range(40):
            g = torch.Generator().manual_seed(seed)
            imgs = _images(C_list, 8, seed=seed)
            for im, v in zip(imgs, masker(imgs, generator=g)):
                C = im.shape[0]
                assert v.shape == (C, 8)
                assert bool(v.any()), "at least one token must stay visible"
                assert (
                    int(v.any(dim=1).sum())
                    >= min(cfg["min_visible_channels"], C)
                )


def test_last_resort_guard_is_unreachable_for_channel_and_combined(monkeypatch):
    """The channel/combined regions satisfy the minimum-content guard by draw."""

    def _fail(*_args, **_kwargs):
        raise AssertionError(
            "last-resort guard must be unreachable for channel/combined"
        )

    configs = [
        dict(p_patch=0.0, p_channel=1.0, min_visible_channels=1),
        dict(p_patch=0.0, p_channel=1.0, min_visible_channels=3),
        dict(p_patch=0.0, p_channel=1.0, min_visible_channels=4),
        dict(p_patch=0.0, p_channel=1.0, min_visible_channels=5),
        dict(p_patch=0.0, p_channel=0.0, min_visible_channels=1),
        dict(p_patch=0.0, p_channel=0.0, min_visible_channels=3),
        dict(p_patch=0.0, p_channel=0.0, min_visible_channels=5),
    ]
    C_list = [1, 2, 3, 5, 8]

    for cfg in configs:
        masker = DCPVisibilitySampler(8, **cfg)
        monkeypatch.setattr(masker, "_force_visible_patch", _fail)
        for seed in range(50):
            g = torch.Generator().manual_seed(seed)
            imgs = _images(C_list, 8, seed=seed)
            for im, v in zip(imgs, masker(imgs, generator=g)):
                C = im.shape[0]
                assert int(v.any(dim=1).sum()) >= min(
                    cfg["min_visible_channels"], C
                )

    # Exact regression case: with the old ``m ~ U{1..C}`` channel draw this
    # seed exhausted the resampling attempts and reached the patch-reveal
    # fallback, leaking encoder evidence from a dropped channel.
    masker = DCPVisibilitySampler(
        8, p_patch=0.0, p_channel=1.0, min_visible_channels=4
    )
    monkeypatch.setattr(masker, "_force_visible_patch", _fail)
    for seed in range(500):
        g = torch.Generator().manual_seed(seed)
        v = masker(_images([5], 8, seed=seed), generator=g)[0]
        assert masker.last_modes == ["channel"]
        assert int(v.any(dim=1).sum()) >= 4


def test_guards_never_reveal_a_whole_channel_row_in_patch_mode():
    N, C = 8, 5
    masker = DCPVisibilitySampler(
        N, p_patch=1.0, p_channel=0.0, patch_ratio=0.875
    )
    for seed in range(50):
        g = torch.Generator().manual_seed(seed)
        v = masker(_images([C], N, seed=seed), generator=g)[0]
        # 7 of 8 patches masked in every row: no guard revealed a full row
        for row in v:
            assert bool(row.any()) and bool((~row).any())


def test_channel_mode_is_not_a_noop_for_c1():
    N = 8
    masker = DCPVisibilitySampler(N, p_patch=0.0, p_channel=1.0, patch_ratio=0.75)
    for seed in range(20):
        g = torch.Generator().manual_seed(seed)
        v = masker(_images([1], N, seed=seed), generator=g)[0]
        assert masker.last_modes == ["patch"]
        assert _masked_per_row(v) == [6]
        assert bool(v.any()) and not bool(v.all())


def test_mode_selection_is_per_batch_or_per_sample():
    imgs = _images([4, 4, 4, 4], 8)
    common = dict(p_patch=0.5, p_channel=0.5, patch_ratio=0.5)

    batch = DCPVisibilitySampler(8, mode_sampling="batch", **common)
    for seed in range(30):
        batch(imgs, generator=torch.Generator().manual_seed(seed))
        assert len(set(batch.last_modes)) == 1
        assert batch.last_modes[0] in ("patch", "channel")

    sample = DCPVisibilitySampler(8, mode_sampling="sample", **common)
    mixed = False
    for seed in range(50):
        sample(imgs, generator=torch.Generator().manual_seed(seed))
        assert len(sample.last_modes) == len(imgs)
        if len(set(sample.last_modes)) > 1:
            mixed = True
            break
    assert mixed, "per-sample selection should let modes differ within a batch"


@pytest.mark.parametrize(
    ("p_patch", "p_channel", "expected"),
    [(1.0, 0.0, "patch"), (0.0, 1.0, "channel"), (0.0, 0.0, "combined")],
)
def test_probability_regions_select_the_expected_mode(p_patch, p_channel, expected):
    masker = DCPVisibilitySampler(
        8, p_patch=p_patch, p_channel=p_channel, patch_ratio=0.5
    )
    imgs = _images([4, 3, 2], 8)
    for seed in range(10):
        masker(imgs, generator=torch.Generator().manual_seed(seed))
        assert masker.last_modes == [expected] * len(imgs)
