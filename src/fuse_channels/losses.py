"""Reconstruction losses for the channel-aware MAE.

The default objective is :func:`normalized_patch_mse`, the standard MAE
patch-normalized pixel MSE.  :func:`cha_maevit_recon_loss` implements
ChA-MAEViT Eq. 2 as an explicit ablation: raw-pixel L2 plus an L1 distance
between Fourier amplitudes.  Neither loss applies per-channel normalization.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


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
        channel-patches.
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
