"""2-D axial rotary position embedding (RoPE) for vision token grids.

The encoder uses RoPE on **normalized** grid coordinates so that the positional
signal describes layout only: a token at the same fractional location gets the
same phase whether the grid is 4x4 or 32x32.  Physical token size belongs in
the scale conditioner, not in the rotary phase.

The implementation follows the axial 2-D RoPE construction in RoPE-ViT
(``references/rope-vit/paper.md:157-171``) and the original RoPE formulation
(``references/roformer/paper.md:23``): the head dimension is split into a row
half and a column half, and each half is rotated by its own axis coordinate.
"""

from __future__ import annotations

import torch


def _as_tensor(value, *, device, dtype):
    if torch.is_tensor(value):
        return value.to(device=device, dtype=dtype)
    return torch.as_tensor(value, device=device, dtype=dtype)


def rope_frequencies(head_dim: int, base: float = 100.0) -> torch.Tensor:
    """Return the ``head_dim / 4`` inverse frequencies for one axis.

    Axial RoPE needs ``head_dim % 4 == 0``: each axis gets ``head_dim / 2``
    channels, i.e. ``head_dim / 4`` rotation pairs.
    """
    if head_dim % 4 != 0:
        raise ValueError(
            f"axial RoPE needs head_dim divisible by 4 (got {head_dim}); "
            "each axis needs head_dim/4 rotation pairs"
        )
    axis_dim = head_dim // 2
    n_pairs = axis_dim // 2
    i = torch.arange(n_pairs, dtype=torch.float32)
    return base ** (-2.0 * i / axis_dim)


def normalized_grid_coords(
    rows: torch.Tensor,
    cols: torch.Tensor,
    grid_h,
    grid_w,
    *,
    ref_extent: float = 32.0,
    shared_extent: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map integer grid coordinates to normalized coordinates in ``(0, S)``.

    Parameters
    ----------
    rows, cols:
        Per-token integer coordinates ``[T]``.
    grid_h, grid_w:
        Grid extents, either scalars or per-token tensors broadcastable to
        ``[T]``.
    ref_extent:
        Reference extent ``S``.  The default keeps the coordinate range close
        to the vision grids RoPE-ViT tunes its base for.
    shared_extent:
        If ``True``, both axes are divided by ``max(grid_h, grid_w)`` so the
        aspect ratio is preserved.  If ``False`` (default), each axis is
        normalized independently, which removes aspect ratio.
    """
    dtype = torch.float32
    device = rows.device
    gh = _as_tensor(grid_h, device=device, dtype=dtype).expand_as(rows)
    gw = _as_tensor(grid_w, device=device, dtype=dtype).expand_as(cols)
    if bool((gh <= 0).any()) or bool((gw <= 0).any()):
        raise ValueError("grid extents must be positive")

    if shared_extent:
        scale = torch.maximum(gh, gw)
        r_norm = (rows.to(dtype) + 0.5) / scale * ref_extent
        c_norm = (cols.to(dtype) + 0.5) / scale * ref_extent
    else:
        r_norm = (rows.to(dtype) + 0.5) / gh * ref_extent
        c_norm = (cols.to(dtype) + 0.5) / gw * ref_extent
    return r_norm, c_norm


def physical_grid_coords(
    rows: torch.Tensor,
    cols: torch.Tensor,
    patch_size,
    pixel_size,
    *,
    ref_um: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Metric coordinates: token centre in micrometres, divided by ``ref_um``.

    ``coord = (index + 0.5) * patch_size * pixel_size / ref_um``

    Because the coordinate is proportional to physical distance, the RoPE phase
    difference between two tokens is proportional to their separation in µm --
    for *any* grid extent.  This is what makes a cell-sized (variable ``N``)
    grid metric: with :func:`normalized_grid_coords` the phase encodes
    ``distance / window_extent``, so the same physical gap gets a different
    rotation in a small cell than in a large one, and an additive scale
    conditioner cannot undo it (the rotation is applied before attention).

    ``patch_size`` and ``pixel_size`` may be scalars or per-token tensors.

    ``ref_um=1.0`` means one coordinate unit is one micrometre, i.e. the
    coordinates *are* micrometres.  This is a free physical unit, not something
    derived from the crop: a window of ``ref_extent`` micrometres spans
    ``ref_extent`` units, which is the point where the metric and grid-fraction
    coordinate systems coincide (useful for testing, and for keeping the RoPE
    frequency scale comparable).  With cell-sized crops the window extent varies
    and no such fixed mapping exists -- which is exactly why the metric system
    is the correct one there.
    """
    if ref_um <= 0:
        raise ValueError(f"ref_um must be positive, got {ref_um}")
    dtype = torch.float32
    device = rows.device
    p = _as_tensor(patch_size, device=device, dtype=dtype).expand_as(rows)
    pixel_size = _as_tensor(pixel_size, device=device, dtype=dtype).expand_as(rows)
    if bool((p <= 0).any()) or bool((pixel_size <= 0).any()):
        raise ValueError("patch_size and pixel_size must be positive")
    step = p * pixel_size / ref_um
    return (rows.to(dtype) + 0.5) * step, (cols.to(dtype) + 0.5) * step


def axial_rope_cache(
    rows: torch.Tensor,
    cols: torch.Tensor,
    grid_h,
    grid_w,
    head_dim: int,
    *,
    base: float = 100.0,
    ref_extent: float = 32.0,
    shared_extent: bool = False,
    patch_size=None,
    pixel_size=None,
    ref_um: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(cos, sin)`` of shape ``[T, head_dim / 2]`` for axial RoPE.

    Passing ``pixel_size`` (and ``patch_size``) switches from grid-fraction
    coordinates to metric ones via :func:`physical_grid_coords`, so the phase
    encodes physical separation instead of a fraction of the grid extent.
    """
    theta = rope_frequencies(head_dim, base).to(rows.device)
    if pixel_size is not None:
        if patch_size is None:
            raise ValueError("metric RoPE needs patch_size when pixel_size is given")
        r_norm, c_norm = physical_grid_coords(
            rows, cols, patch_size, pixel_size, ref_um=ref_um
        )
    else:
        r_norm, c_norm = normalized_grid_coords(
            rows,
            cols,
            grid_h,
            grid_w,
            ref_extent=ref_extent,
            shared_extent=shared_extent,
        )
    angle = torch.cat(
        [r_norm[:, None] * theta[None, :], c_norm[:, None] * theta[None, :]], dim=-1
    )
    return angle.cos(), angle.sin()


def identity_rope_cache(
    n_tokens: int, head_dim: int, *, device=None, dtype=torch.float32
) -> tuple[torch.Tensor, torch.Tensor]:
    """Zero-angle cache: applying it to a tensor is an exact no-op."""
    n_pairs = head_dim // 2
    cos = torch.ones(n_tokens, n_pairs, device=device, dtype=dtype)
    sin = torch.zeros(n_tokens, n_pairs, device=device, dtype=dtype)
    return cos, sin


def apply_rope(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Rotate ``[T, H, D_head]`` query/key tensors.

    ``cos`` and ``sin`` have shape ``[T, D_head / 2]``.  Pairs are interleaved
    as ``(x[..., 2i], x[..., 2i+1])``; the first half of the pair index range
    is the row axis and the second half is the column axis.
    """
    if x.ndim != 3:
        raise ValueError(f"apply_rope expects [T, H, D_head], got {tuple(x.shape)}")
    T, H, Dh = x.shape
    if Dh != 2 * cos.shape[-1]:
        raise ValueError(
            f"RoPE cache has {2 * cos.shape[-1]} channels, but x has {Dh}"
        )
    if cos.shape[0] != T:
        raise ValueError(
            f"RoPE cache has {cos.shape[0]} positions, but x has {T} tokens"
        )
    pairs = x.reshape(T, H, Dh // 2, 2)
    c = cos.reshape(T, 1, -1).to(dtype=x.dtype)
    s = sin.reshape(T, 1, -1).to(dtype=x.dtype)
    x1, x2 = pairs[..., 0], pairs[..., 1]
    rotated = torch.stack((x1 * c - x2 * s, x1 * s + x2 * c), dim=-1)
    return rotated.reshape(T, H, Dh)
