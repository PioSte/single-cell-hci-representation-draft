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
    cfg = dict(
        image_size=8,
        patch_size=4,
        dim=32,
        encoder_depth=2,
        decoder_depth=1,
        n_heads=4,
        n_memory=3,
        num_channel_ids=16,
        pixel_size=1.0,
    )
    cfg.update(kw)
    return ChannelFusionMAE(**cfg)


def _batch(seed=0, C_list=(2, 4, 3), image_size=8):
    g = torch.Generator().manual_seed(seed)
    images = [torch.randn(C, 1, image_size, image_size, generator=g) for C in C_list]
    ids = [torch.randint(0, 16, (C,), generator=g) for C in C_list]
    return images, ids


def _fourier_l1(pred, target, patch_size):
    pred_amp = torch.fft.rfft2(
        pred.reshape(-1, patch_size, patch_size)
    ).abs()
    target_amp = torch.fft.rfft2(
        target.reshape(-1, patch_size, patch_size)
    ).abs()
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

    assert torch.allclose(
        cha_maevit_recon_loss(pred, target, lambda_f=0.0), expected_pixel
    )
    assert torch.allclose(
        cha_maevit_recon_loss(pred, target, lambda_f=1.0), expected_fourier
    )
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
    assert math.isfinite(
        cha_maevit_recon_loss(torch.randn(3, 9), torch.randn(3, 9)).item()
    )


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
