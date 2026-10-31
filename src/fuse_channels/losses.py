"""Reconstruction losses for the channel-aware MAE.

The default objective is the standard MAE patch-normalized pixel MSE; the model
path uses :func:`patch_row_terms`, which applies it to the flat, padding-free
pixel layout (see :func:`patch_offsets`).  :func:`normalized_patch_mse` is the
``[K, p**2]`` reference form of the same math.  :func:`cha_maevit_recon_loss` implements
ChA-MAEViT Eq. 2 as an explicit ablation: raw-pixel L2 plus an L1 distance
between Fourier amplitudes.  Neither loss applies per-channel normalization.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def patch_offsets(patch_sizes) -> torch.Tensor:
    """Row boundaries of the flat pixel layout: ``[K + 1]`` cumulative offsets.

    Row ``i`` owns ``pixels[offsets[i]:offsets[i + 1]]``, i.e. ``p_i**2``
    contiguous pixels.  Concatenating rows this way covers ``Σ p_i**2`` entries
    with no padding and therefore no validity mask, whatever the mix of patch
    sizes.
    """
    ps = torch.as_tensor(patch_sizes, dtype=torch.long)
    if ps.numel() == 0:
        return torch.zeros(1, dtype=torch.long, device=ps.device)
    offsets = torch.zeros(
        ps.numel() + 1, dtype=torch.long, device=ps.device
    )
    offsets[1:] = torch.cumsum(ps * ps, dim=0)
    return offsets


def patch_row_terms(
    pred: torch.Tensor,
    target: torch.Tensor,
    patch_sizes,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-row ``(squared-error sum, pixel count)`` in the flat pixel layout.

    ``pred``/``target`` follow :func:`patch_offsets`: ``Σ p_i**2`` pixels, row
    ``i`` contiguous.  Each row is normalised by its own mean and standard
    deviation, exactly as :func:`normalized_patch_mse` does on ``[K, p**2]``
    inputs; the row terms come from segment sums, so a mixed-``p`` batch needs
    no padded block and no validity mask.

    Returns ``[K]`` error sums and ``[K]`` counts, so callers implement either
    reduction from the same pass.
    """
    target = target.float()
    offsets = patch_offsets(patch_sizes).to(pred.device)
    counts = (offsets[1:] - offsets[:-1]).to(target.dtype)
    rows = torch.repeat_interleave(
        torch.arange(offsets.numel() - 1, device=pred.device), counts.long()
    )
    denom = counts.clamp_min(1.0)
    mean = target.new_zeros(counts.shape).index_add_(0, rows, target) / denom
    centred = target - mean[rows]
    var = target.new_zeros(counts.shape).index_add_(0, rows, centred * centred)
    var = var / denom
    norm_target = centred / (var.clamp_min(0.0).sqrt() + eps)[rows]
    err = (pred - norm_target) ** 2
    row_err = pred.new_zeros(counts.shape).index_add_(0, rows, err)
    return row_err, counts


def normalized_patch_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """MSE between predicted and per-patch-normalized target pixels.

    Parameters
    ----------
    pred, target:
        ``[K, patch_dim]`` predictions and raw target pixels for ``K`` masked
        channel-patches.  This is the reference form; the model path uses
        :func:`patch_row_terms` on the flat pixel layout.
    """
    target = target.float()
    mean = target.mean(dim=-1, keepdim=True)
    std = target.std(dim=-1, keepdim=True, unbiased=False)
    norm_target = (target - mean) / (std + eps)
    return F.mse_loss(pred, norm_target)


def _fft_input(x: torch.Tensor) -> torch.Tensor:
    """FFT supports float32/float64 only; keep those dtypes unchanged."""
    if x.dtype in (torch.float32, torch.float64):
        return x
    return x.float()


def _fourier_amplitude(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Amplitudes of the 2-D real FFT of ``[K, patch_size * patch_size]`` patches."""
    return torch.fft.rfft2(_fft_input(x).reshape(-1, patch_size, patch_size)).abs()


def cha_maevit_recon_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    lambda_f: float = 0.01,
) -> torch.Tensor:
    """ChA-MAEViT Eq. 2: raw-pixel L2 plus a Fourier-space L1 term.

    ``(1 - lambda_f) * MSE(pred, target) + lambda_f * L1(|rFFT(pred)|, |rFFT(target)|)``,
    where each row of the ``[K, patch_dim]`` inputs is one square patch and
    ``rFFT`` is ``torch.fft.rfft2`` (the real-input half-spectrum).  No
    per-patch normalization is applied.

    Parameters
    ----------
    pred, target:
        ``[K, P*P]`` predictions and raw target pixels for ``K`` masked
        channel-patches.
    lambda_f:
        Fourier weight; ChA-MAEViT uses ``0.01``.
    """
    if pred.shape != target.shape:
        raise ValueError(
            f"pred and target must have the same shape, got {tuple(pred.shape)} "
            f"and {tuple(target.shape)}"
        )
    patch_dim = pred.shape[-1]
    patch_size = math.isqrt(patch_dim)
    if patch_size * patch_size != patch_dim:
        raise ValueError(
            f"cha_maevit_recon_loss requires a square patch_dim, got {patch_dim}"
        )

    target = target.float()
    pixel_loss = F.mse_loss(pred, target)
    if lambda_f == 0.0:
        return pixel_loss

    pred_amp = _fourier_amplitude(pred.float(), patch_size)
    target_amp = _fourier_amplitude(target, patch_size)
    fourier_l1 = F.l1_loss(pred_amp, target_amp)
    if lambda_f == 1.0:
        return fourier_l1
    return (1.0 - lambda_f) * pixel_loss + lambda_f * fourier_l1
