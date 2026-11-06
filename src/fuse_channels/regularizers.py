"""Anti-collapse regularizers for the JEPA path.

EMA + stop-gradient is the default collapse-prevention mechanism.  A JEPA can
instead train without an EMA target encoder and regularize the student's
embedding distribution directly; SIGReg (LeJEPA / Epps-Pulley) and VICReg
(variance + covariance) are the two supported ablations.  Both consume one
``[rows, D]`` batch of embeddings and return a scalar; the invariance term of
VICReg is the JEPA feature-prediction loss, not part of the regularizer.

**Row convention.**  ``SIGReg`` defaults to ``reduce="mean"``, which drops the
explicit ``* rows`` factor of Algorithm 1, so the value is not multiplied by the
row count; ``reduce="sum"`` reproduces the published statistic verbatim.  VICReg
is already a per-row statistic (not multiplied by the row count), but its
covariance term carries a finite-batch floor (below).  Neither convention makes
``reg_weight`` independent of the batch regime: record ``(rows, D)`` next to any
weight you recommend.

**Regimes matter more than the weight.**  Both statistics are *estimated from a
finite batch*, so their value is signal plus a sampling floor that depends on
``rows`` and ``D``:

* SIGReg's minibatch bias vanishes at rate ``O(1/N)`` (the paper finds it
  negligible from ``N=16``), and the estimator's variance grows as the row count
  falls relative to ``D``.
* VICReg's covariance term has expectation ``~(D-1)/(rows-1)`` even for
  independent features, so it is never zero and its gradient asks for a
  decorrelation the batch cannot support.  ``bias_correct=True`` removes that
  known floor; the paper's own recipe (batch 2048, expander 8192) runs with the
  floor in place, which is why this is opt-in rather than the default.

References
----------
* SIGReg: Balestriero & LeCun, *LeJEPA*, arXiv:2511.08544v3, Algorithm 1:
  ``num_slices`` random unit directions seeded per step, a 17-point trapezoid
  quadrature on ``[-5, 5]``, the standard-normal characteristic function
  ``exp(-t**2/2)`` as both target and Gaussian window, and an all-reduced
  empirical characteristic function for DDP.
* VICReg: Bardes, Ponce & LeCun, arXiv:2105.04906v3.  The variance hinge uses
  the batch standard deviation, not the variance, because the variance's
  gradient vanishes as the embeddings collapse.  Defaults are the paper's
  ``mu=25``, ``nu=1``, ``gamma=1``, ``eps=1e-4``.

The scMorphJEPA reference checkout contains MIT-licensed implementations of the
same regularizers; these are independent adaptations of the published
equations, kept small so the loss scale is explicit and testable.
``reduce`` (SIGReg) and ``bias_correct`` (VICReg) are ours, not from the papers.
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
        Number of random unit directions.  LeJEPA's paper default is 256; 256
        keeps the cost small.
    knots:
        Number of trapezoid quadrature points on ``[-t_max, t_max]``.
    t_max:
        Integration half-width.  The paper uses ``[-5, 5]``.
    reduce:
        ``"mean"`` (default) divides by the row count, so the value is a
        per-row statistic and ``reg_weight`` does not have to be retuned when
        the batch size, the number of target locations or ``D`` changes.
        ``"sum"`` reproduces Algorithm 1 verbatim.

    Input is ``[rows, D]``.  The projections are seeded from a module buffer so
    all DDP ranks use the same directions on the same training step; the
    empirical characteristic function is all-reduced across ranks.
    """

    # registered buffers
    t: torch.Tensor
    phi: torch.Tensor
    _step: torch.Tensor

    def __init__(
        self,
        num_slices: int = 256,
        knots: int = 17,
        t_max: float = 5.0,
        reduce: str = "mean",
    ) -> None:
        super().__init__()
        if num_slices < 1:
            raise ValueError(f"num_slices must be >= 1, got {num_slices}")
        if knots < 2:
            raise ValueError(f"knots must be >= 2, got {knots}")
        if t_max <= 0:
            raise ValueError(f"t_max must be positive, got {t_max}")
        if reduce not in ("mean", "sum"):
            raise ValueError(f"reduce must be 'mean' or 'sum', got {reduce!r}")
        self.reduce = reduce
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
        statistic = torch.trapz(err, self.t, dim=-1).mean()
        if self.reduce == "sum":
            statistic = statistic * (z.shape[0] * world)
        return statistic


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
        bias_correct: bool = False,
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
        self.bias_correct = bool(bias_correct)

    @staticmethod
    def covariance_floor(rows: int, dim: int) -> float:
        """Expected covariance term for ``dim`` *independent* features, ``rows`` rows.

        A sample covariance entry estimated from ``n`` rows has variance
        ``~1/(n-1)`` under independence, and the term is the mean of the squared
        off-diagonal entries, so the floor is ``(dim - 1) / (rows - 1)``.  It is
        the part of the regularizer that carries no information about
        redundancy; ``bias_correct=True`` subtracts it.
        """
        if rows < 2 or dim < 1:
            return 0.0
        return (dim - 1) / (rows - 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim != 2:
            raise ValueError(f"VICReg expects [rows, D], got {tuple(z.shape)}")
        B, D = z.shape
        if B < 2:
            return z.new_zeros(())
        work = z.float()
        work = work - work.mean(dim=0)
        std = torch.sqrt(work.var(dim=0) + self.eps)
        var_loss = F.relu(self.gamma - std).mean()
        cov = (work.T @ work) / (B - 1)
        cov_loss = _off_diagonal(cov).pow(2).sum() / D
        if self.bias_correct:
            cov_loss = (cov_loss - self.covariance_floor(B, D)).clamp_min(0.0)
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
