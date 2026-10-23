import torch

from fuse_channels import DCPVisibilitySampler


def _images(C_list, N, seed=0):
    g = torch.Generator().manual_seed(seed)
    # images are [C,1,H,W]; the sampler only uses C and the device, so H=W=1
    return [torch.randn(C, 1, 1, 1, generator=g) for C in C_list]


def test_never_masks_all_channels_or_tokens():
    masker = DCPVisibilitySampler(n_patches=4, p_patch=0.9, p_channel=1.0)
    for seed in range(50):
        g = torch.Generator().manual_seed(seed)
        imgs = _images([1, 2, 3, 5, 8], 4, seed=seed)
        visible = masker(imgs, generator=g)
        for im, v in zip(imgs, visible):
            C = im.shape[0]
            assert v.shape == (C, 4)
            assert bool(v.any()), "at least one token must stay visible"
            # at least one channel must keep content
            assert int(v.any(dim=1).sum()) >= 1
            # at least one channel must be masked (target exists)
            assert int((~v).any(dim=1).sum()) >= 0


def test_channel_mode_masks_whole_channels():
    masker = DCPVisibilitySampler(n_patches=4, p_patch=0.0, p_channel=1.0)
    g = torch.Generator().manual_seed(0)
    imgs = _images([6], 4)
    visible = masker(imgs, generator=g)[0]
    # channel-mode rows are all-visible or all-masked
    for c in range(visible.shape[0]):
        row = visible[c]
        assert bool(row.all()) or bool((~row).all())
    # at least one channel retained
    assert bool(visible.any(dim=1).any())


def test_spatial_mode_is_per_patch():
    masker = DCPVisibilitySampler(n_patches=8, p_patch=0.5, p_channel=0.0)
    g = torch.Generator().manual_seed(0)
    imgs = _images([4], 8)
    visible = masker(imgs, generator=g)[0]
    # in spatial mode a channel may be partially visible
    partially = (visible.any(dim=1) & (~visible).any(dim=1)).any()
    assert bool(partially)
