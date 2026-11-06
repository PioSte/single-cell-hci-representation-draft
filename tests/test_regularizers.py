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
