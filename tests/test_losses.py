import math

import pytest
import torch
import torch.nn.functional as F

from fuse_channels import (
    ChannelFusionMAE,
    cha_maevit_recon_loss,
    normalized_patch_mse,
)


def _model(**kw):
    cfg = {
        "image_size": 8,
        "patch_size": 4,
        "dim": 32,
        "encoder_depth": 2,
        "decoder_depth": 1,
        "n_heads": 4,
        "n_memory": 3,
        "num_channel_ids": 16,
        "pixel_size": 1.0,
    }
    cfg.update(kw)
    return ChannelFusionMAE(**cfg)


def _batch(seed=0, C_list=(2, 4, 3), image_size=8):
    g = torch.Generator().manual_seed(seed)
    images = [torch.randn(C, 1, image_size, image_size, generator=g) for C in C_list]
    ids = [torch.randint(0, 16, (C,), generator=g) for C in C_list]
    return images, ids


def _fourier_l1(pred, target, patch_size):
    pred_amp = torch.fft.rfft2(pred.reshape(-1, patch_size, patch_size)).abs()
    target_amp = torch.fft.rfft2(target.reshape(-1, patch_size, patch_size)).abs()
    return F.l1_loss(pred_amp, target_amp)


def test_cha_maevit_loss_zero():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(5, 16, generator=g)
    assert cha_maevit_recon_loss(x, x).item() == 0.0
    # the default lambda must still be zero for identical inputs
    assert cha_maevit_recon_loss(x, x, lambda_f=0.01).item() == 0.0


def test_cha_maevit_loss_lambda_extremes():
    g = torch.Generator().manual_seed(1)
    pred = torch.randn(7, 16, generator=g)
    target = torch.randn(7, 16, generator=g)

    expected_pixel = F.mse_loss(pred, target)
    expected_fourier = _fourier_l1(pred, target, patch_size=4)

    assert torch.allclose(cha_maevit_recon_loss(pred, target, lambda_f=0.0), expected_pixel)
    assert torch.allclose(cha_maevit_recon_loss(pred, target, lambda_f=1.0), expected_fourier)
    # the default interpolates the two terms
    assert torch.allclose(
        cha_maevit_recon_loss(pred, target, lambda_f=0.25),
        0.75 * expected_pixel + 0.25 * expected_fourier,
    )


def test_cha_maevit_loss_requires_square_patch():
    pred = torch.randn(3, 6)
    target = torch.randn(3, 6)
    with pytest.raises(ValueError, match="square patch_dim"):
        cha_maevit_recon_loss(pred, target)
    # a square non-power-of-two patch works
    assert math.isfinite(cha_maevit_recon_loss(torch.randn(3, 9), torch.randn(3, 9)).item())


def test_cha_maevit_loss_shape_mismatch():
    with pytest.raises(ValueError, match="same shape"):
        cha_maevit_recon_loss(torch.randn(3, 16), torch.randn(4, 16))


def test_loss_selector():
    with pytest.raises(ValueError, match="unknown loss"):
        _model(loss="not_a_loss")

    default = _model()
    assert default.loss_name == "mae"

    model = _model(loss="cha_maevit")
    assert model.loss_name == "cha_maevit"
    images, ids = _batch()
    out = model(images, ids, generator=torch.Generator().manual_seed(1))
    assert out["pred"] is not None
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    assert model.decoder.head.weight.grad is not None
    assert torch.isfinite(model.decoder.head.weight.grad).all()


def test_normalized_patch_mse_unchanged():
    g = torch.Generator().manual_seed(2)
    pred = torch.randn(4, 16, generator=g)
    target = torch.randn(4, 16, generator=g)
    norm_target = (target - target.mean(-1, keepdim=True)) / (
        target.std(-1, unbiased=False, keepdim=True) + 1e-6
    )
    assert torch.allclose(normalized_patch_mse(pred, target), F.mse_loss(pred, norm_target))


def test_flat_row_terms_match_the_dense_reference():
    """The padding-free pixel layout reproduces the ``[K, p**2]`` reference.

    This is the loss-side counterpart of the fusion equivalence test: a
    mixed-``p`` batch keeps each row's own ``p**2`` pixels, and the per-row
    terms must equal what the dense square form computes row by row.
    """
    from fuse_channels.losses import patch_offsets, patch_row_terms

    g = torch.Generator().manual_seed(0)
    ps = torch.tensor([4, 8, 4])  # mixed sizes, non-contiguous rows
    offsets = patch_offsets(ps)
    assert offsets.tolist() == [0, 16, 80, 96]
    pred_blocks = [
        torch.randn(1, 16, generator=g),
        torch.randn(1, 64, generator=g),
        torch.randn(1, 16, generator=g),
    ]
    target_blocks = [
        torch.randn(1, 16, generator=g),
        torch.randn(1, 64, generator=g),
        torch.randn(1, 16, generator=g),
    ]
    pred = torch.cat([b.reshape(-1) for b in pred_blocks])
    target = torch.cat([b.reshape(-1) for b in target_blocks])
    assert pred.shape == (int(offsets[-1]),)

    row_err, row_count = patch_row_terms(pred, target, ps)
    for i, (pb, tb) in enumerate(zip(pred_blocks, target_blocks, strict=True)):
        assert row_count[i].item() == ps[i].item() ** 2
        torch.testing.assert_close(
            row_err[i] / row_count[i],
            normalized_patch_mse(pb, tb),
            rtol=1e-5,
            atol=1e-6,
        )


def test_mixed_p_batch_builds_no_padded_pixel_block():
    """A mixed-``p`` batch allocates exactly ``Σ p**2`` pixels per target set."""
    model = _model(patch_sizes=(4, 8))
    images = [torch.randn(2, 1, 16, 16), torch.randn(2, 1, 32, 32)]
    ids = [torch.tensor([0, 1]), torch.tensor([0, 1])]
    with torch.no_grad():
        out = model(
            images,
            ids,
            patch_sizes=[4, 8],
            generator=torch.Generator().manual_seed(0),
        )
    offsets = out["target_pixel_offset"]
    counts = (offsets[1:] - offsets[:-1]).tolist()
    assert set(counts) == {16, 64}
    assert out["pred"].numel() == out["target"].numel() == sum(counts)
    assert out["pred"].shape == out["target"].shape
