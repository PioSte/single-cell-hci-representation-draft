"""Tests for the SIGReg / VICReg anti-collapse regularizers."""

import pytest
import torch

from fuse_channels import (
    JEPAConfig,
    NoReg,
    SIGReg,
    VICReg,
    build_regularizer,
)


def test_build_regularizer_registry():
    assert isinstance(build_regularizer("none"), NoReg)
    assert isinstance(build_regularizer("sigreg"), SIGReg)
    assert isinstance(build_regularizer("vicreg"), VICReg)
    with pytest.raises(ValueError, match="unknown regularizer"):
        build_regularizer("barlow")


def test_sigreg_is_finite_differentiable_and_prefers_gaussian_over_collapse():
    g = torch.Generator().manual_seed(0)
    gaussian = torch.randn(256, 8, generator=g)
    collapsed = torch.zeros(256, 8)
    reg = SIGReg(num_slices=64)
    collapsed_loss = reg(collapsed)
    gaussian_loss = reg(gaussian)
    assert torch.isfinite(collapsed_loss) and torch.isfinite(gaussian_loss)
    assert gaussian_loss < collapsed_loss
    z = torch.randn(32, 8, generator=g, requires_grad=True)
    loss = SIGReg(num_slices=32)(z)
    loss.backward()
    assert z.grad is not None and torch.isfinite(z.grad).all()


def test_vicreg_matches_the_paper_formula():
    z = torch.tensor([[1.0, 0.0], [0.0, 2.0], [3.0, 3.0]])
    reg = VICReg(gamma=1.0, eps=1e-4, var_weight=25.0, cov_weight=1.0)
    centred = z - z.mean(dim=0)
    std = torch.sqrt(centred.var(dim=0) + 1e-4)
    var_loss = torch.relu(1.0 - std).mean()
    cov = (centred.T @ centred) / (z.shape[0] - 1)
    off = cov.flatten()[:-1].view(1, 3)[:, 1:].flatten()
    expected = 25.0 * var_loss + off.pow(2).sum() / z.shape[1]
    assert torch.allclose(reg(z), expected)


def test_vicreg_is_higher_on_a_collapsed_batch():
    g = torch.Generator().manual_seed(1)
    diverse = torch.randn(256, 8, generator=g)
    collapsed = torch.zeros(256, 8)
    reg = VICReg()
    assert reg(collapsed) > reg(diverse)


def test_jepa_config_requires_a_regularizer_without_ema():
    with pytest.raises(ValueError, match="needs a non-zero regularizer"):
        JEPAConfig(use_ema=False, regularizer="none")
    with pytest.raises(ValueError, match="needs a non-zero regularizer"):
        JEPAConfig(use_ema=False, regularizer="sigreg", reg_weight=0.0)
    JEPAConfig(use_ema=False, regularizer="sigreg", reg_weight=1.0)


def _sigreg_at_step(num_slices=64, reduce="mean", step=1):
    """A SIGReg whose next forward uses the random slices of ``step``."""
    reg = SIGReg(num_slices=num_slices, reduce=reduce)
    reg._step.fill_(step - 1)
    return reg


def test_sigreg_mean_is_replication_invariant_and_sum_doubles():
    """The mean-reduced statistic ignores row replication; the sum-reduced one scales."""
    g = torch.Generator().manual_seed(0)
    z = torch.randn(64, 8, generator=g)
    mean_single = _sigreg_at_step(step=1)(z)
    mean_double = _sigreg_at_step(step=1)(torch.cat([z, z]))
    assert torch.allclose(mean_single, mean_double, rtol=1e-4, atol=1e-5)
    sum_single = _sigreg_at_step(reduce="sum", step=1)(z)
    sum_double = _sigreg_at_step(reduce="sum", step=1)(torch.cat([z, z]))
    assert torch.allclose(sum_double, 2 * sum_single, rtol=1e-4, atol=1e-4)


def test_vicreg_covariance_floor_formula():
    assert VICReg.covariance_floor(8, 128) == pytest.approx(127.0 / 7.0)
    assert VICReg.covariance_floor(2, 128) == pytest.approx(127.0)
    assert VICReg.covariance_floor(1, 128) == 0.0


def test_vicreg_bias_correct_removes_the_independent_feature_floor():
    """On iid Gaussian rows the raw covariance term is almost all sampling floor."""
    g = torch.Generator().manual_seed(2)
    for rows in (8, 64, 512):
        z = torch.randn(rows, 128, generator=g)
        raw = VICReg(bias_correct=False)(z)
        corrected = VICReg(bias_correct=True)(z)
        floor = VICReg.covariance_floor(rows, 128)
        # The correction removes between 0 and `floor`, and here most of it.
        assert raw - corrected > 0.5 * floor
        assert raw - corrected <= floor + 1e-2


def test_vicreg_duplication_only_changes_the_unbiased_denominator():
    """VICReg is a per-row mean; duplication scales by ``2(B-1)/(2B-1)``."""
    g = torch.Generator().manual_seed(3)
    B, D = 64, 8
    z = torch.randn(B, D, generator=g)
    # ``gamma`` below every std so the variance hinge contributes nothing and the
    # covariance term (whose denominator is B-1) is isolated.
    single = VICReg(gamma=0.1)(z)
    double = VICReg(gamma=0.1)(torch.cat([z, z]))
    expected = single * (2 * (B - 1) / (2 * B - 1)) ** 2
    assert torch.allclose(double, expected, rtol=1e-5, atol=1e-6)
