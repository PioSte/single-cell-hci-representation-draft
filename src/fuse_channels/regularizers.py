"""Anti-collapse regularizers for the JEPA path.

EMA + stop-gradient is the default collapse-prevention mechanism.  A JEPA can
instead train without an EMA target encoder and regularize the student's
embedding distribution directly; SIGReg (LeJEPA / Epps-Pulley) and VICReg
(variance + covariance) are the two supported ablations.  Both consume one
``[B, D]`` batch of embeddings and return a scalar; the invariance term of
VICReg is the JEPA feature-prediction loss, not part of the regularizer.

References
----------
* SIGReg: Balestriero & LeCun, *LeJEPA*, arXiv:2511.08544v3.  The implementation
  follows Algorithm 1: ``num_slices`` random unit directions, a 17-point
  trapezoid quadrature on ``[-5, 5]``, the standard-normal characteristic
  function ``exp(-t**2/2)`` as both target and Gaussian window, and an
  all-reduced empirical characteristic function for DDP.  The ``O(1/N)``
  minibatch bias is documented in the paper.
* VICReg: Bardes, Ponce & LeCun, arXiv:2105.04906v3.  The variance hinge uses
  the batch standard deviation, not the variance, because the variance's
  gradient vanishes as the embeddings collapse.  Defaults are the paper's
  ``mu=25``, ``nu=1``, ``gamma=1``, ``eps=1e-4``.

The scMorphJEPA reference checkout contains MIT-licensed implementations of the
same regularizers; these are independent adaptations of the published
equations, kept small so the loss scale is explicit and testable.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _world_size() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    return 1


def _all_reduce_mean(x: torch.Tensor) -> torch.Tensor:
    """Average a tensor across the default process group, or return it unchanged."""
    world = _world_size()
    if world == 1:
        return x
    torch.distributed.all_reduce(x, op=torch.distributed.ReduceOp.SUM)
    return x / world


def _off_diagonal(m: torch.Tensor) -> torch.Tensor:
    n, _ = m.shape
    return m.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class NoReg(nn.Module):
    """Zero regularizer, for the EMA-only path and the ablation baseline."""

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z.new_zeros(())


class SIGReg(nn.Module):
    """Sketched Isotropic Gaussian Regularization (Epps-Pulley statistic).

    Parameters
    ----------
    num_slices:
        Number of random unit directions.  LeJEPA's paper default is 256 and
        its recommended starting point is 1024; 256 keeps the cost small.
    knots:
        Number of trapezoid quadrature points on ``[-t_max, t_max]``.
    t_max:
        Integration half-width.  The paper recommends ``[-5, 5]``.

    Input is ``[B, D]``.  The projections are seeded from a module buffer so all
    DDP ranks use the same directions on the same training step; the empirical
    characteristic function is all-reduced across ranks.
    """

    # registered buffers
    t: torch.Tensor
    phi: torch.Tensor
    _step: torch.Tensor

    def __init__(self, num_slices: int = 256, knots: int = 17, t_max: float = 5.0) -> None:
        super().__init__()
        if num_slices < 1:
            raise ValueError(f"num_slices must be >= 1, got {num_slices}")
        if knots < 2:
            raise ValueError(f"knots must be >= 2, got {knots}")
        if t_max <= 0:
            raise ValueError(f"t_max must be positive, got {t_max}")
        self.num_slices = int(num_slices)
        t = torch.linspace(-float(t_max), float(t_max), int(knots), dtype=torch.float32)
        self.register_buffer("t", t)
        self.register_buffer("phi", torch.exp(-0.5 * t.square()))
        # Incremented each forward so ranks draw the same slices without an
        # explicit global_step argument.  Not a learnable parameter.
        self.register_buffer("_step", torch.zeros((), dtype=torch.long))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim != 2:
            raise ValueError(f"SIGReg expects [B, D], got {tuple(z.shape)}")
        if z.shape[0] < 2:
            return z.new_zeros(())
        world = _world_size()
        self._step.add_(1)
        generator = torch.Generator(device=z.device)
        generator.manual_seed(int(self._step.item()))
        slices = torch.randn(z.shape[-1], self.num_slices, device=z.device, generator=generator)
        slices = slices / slices.norm(p=2, dim=0).clamp_min(1e-12)

        work = z.float()
        x_t = (work @ slices).unsqueeze(-1) * self.t
        real = _all_reduce_mean(x_t.cos().mean(dim=0))
        imag = _all_reduce_mean(x_t.sin().mean(dim=0))
        ecf = torch.complex(real, imag)
        err = (ecf - self.phi).abs().square() * self.phi
        n = z.shape[0] * world
        statistic = torch.trapz(err, self.t, dim=-1) * n
        return statistic.mean()


class VICReg(nn.Module):
    """Variance + covariance terms of VICReg (single-view JEPA adaptation).

    The invariance term is the JEPA feature-prediction loss, so the regularizer
    is only ``var_weight * variance_hinge + cov_weight * covariance``.  The
    variance hinge uses the batch standard deviation, as in the paper; using the
    variance makes the gradient vanish near collapse.

    Input is ``[B, D]``.  Defaults follow VICReg: ``gamma=1``, ``eps=1e-4``,
    ``var_weight=25``, ``cov_weight=1``.
    """

    def __init__(
        self,
        gamma: float = 1.0,
        eps: float = 1e-4,
        var_weight: float = 25.0,
        cov_weight: float = 1.0,
    ) -> None:
        super().__init__()
        if gamma <= 0:
            raise ValueError(f"gamma must be positive, got {gamma}")
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}")
        if var_weight < 0 or cov_weight < 0:
            raise ValueError("var_weight and cov_weight must be non-negative")
        self.gamma = float(gamma)
        self.eps = float(eps)
        self.var_weight = float(var_weight)
        self.cov_weight = float(cov_weight)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim != 2:
            raise ValueError(f"VICReg expects [B, D], got {tuple(z.shape)}")
        B, D = z.shape
        if B < 2:
            return z.new_zeros(())
        work = z.float()
        work = work - work.mean(dim=0)
        std = torch.sqrt(work.var(dim=0) + self.eps)
        var_loss = F.relu(self.gamma - std).mean()
        cov = (work.T @ work) / (B - 1)
        cov_loss = _off_diagonal(cov).pow(2).sum() / D
        return self.var_weight * var_loss + self.cov_weight * cov_loss


_REGISTRY = {
    "none": NoReg,
    "sigreg": SIGReg,
    "vicreg": VICReg,
}


def build_regularizer(name: str = "none", **kwargs) -> nn.Module:
    """Construct a regularizer by name: ``none``, ``sigreg`` or ``vicreg``."""
    key = str(name).lower()
    if key not in _REGISTRY:
        raise ValueError(f"unknown regularizer '{name}'; expected one of {sorted(_REGISTRY)}")
    return _REGISTRY[key](**kwargs)
