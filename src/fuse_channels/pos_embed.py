"""Absolute 2D spatial positional embeddings for fusion and decoder.

Two interchangeable parameterizations of the **same** additive signal
``[1, N, D]`` (or ``[N, D]`` for the decoder's per-target lookup):

* ``"learned"`` (default) -- an ``nn.Parameter`` table initialized with
  ``trunc_normal_(std=0.02)``, one row per patch position.  Strictly more
  expressive for a fixed grid, which is what the repo trains on.
* ``"sincos"`` -- ViT/MAE's fixed 2D sine-cosine map, registered as a
  non-trainable buffer.  MAE adds it to *both* encoder and decoder inputs
  (``references/mae/paper.md:546``), using ``get_2d_sincos_pos_embed`` with
  ``requires_grad=False`` in the released code.  It carries no parameters and
  is defined for any grid size, so a module built for one grid can be fed
  positions from another.

The MAE paper motivates the *presence* of an absolute positional signal on the
decoder's token set ("without this, mask tokens would have no information about
their location in the image", ``references/mae/paper.md:103``) but reports no
ablation of learned vs fixed; the fixed sine-cosine choice is inherited from
ViT.  Hence ``"learned"`` stays the default and ``"sincos"`` is an experiment
knob.

Patch indices are **row-major** over the ``(grid_h, grid_w)`` lattice: patch
``p = r * grid_w + c`` sits at row ``r`` and column ``c``, matching
``Conv2d(...).flatten(2)`` in :class:`~fuse_channels.patch_embed.ChannelTiedPatchEmbed`.
"""

from __future__ import annotations

import math
from typing import Tuple, Union

import torch
import torch.nn as nn

POS_EMBED_KINDS = ("learned", "sincos")

GridSize = Union[int, Tuple[int, int]]


def _grid_hw(grid_size: GridSize) -> Tuple[int, int]:
    """Normalize ``int | (grid_h, grid_w)`` to a checked ``(h, w)`` pair."""
    if isinstance(grid_size, int):
        grid_h = grid_w = grid_size
    else:
        try:
            grid_h, grid_w = grid_size
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
            raise ValueError(
                f"grid_size must be an int or (grid_h, grid_w), got {grid_size!r}"
            ) from exc
    if grid_h <= 0 or grid_w <= 0:
        raise ValueError(f"grid_size must be positive, got {(grid_h, grid_w)}")
    return int(grid_h), int(grid_w)


def _grid_from_n_patches(n_patches: int) -> int:
    """Square grid side for a square patch lattice; raise otherwise."""
    root = math.isqrt(n_patches)
    if root * root != n_patches:
        raise ValueError(
            f"n_patches={n_patches} is not a square; pass grid_size=(grid_h, grid_w) "
            "for non-square lattices"
        )
    return root


def _1d_sincos(dim: int, positions: torch.Tensor) -> torch.Tensor:
    """``[P, dim]`` sine-cosine embedding of 1D integer positions (MAE-style)."""
    half = dim // 2
    omega = 1.0 / (10000.0 ** (torch.arange(half, dtype=positions.dtype) / half))
    out = positions.reshape(-1, 1) * omega.reshape(1, -1)
    return torch.cat([out.sin(), out.cos()], dim=1)  # sin first, then cos


def get_2d_sincos_pos_embed(
    dim: int,
    grid_size: GridSize,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Fixed 2D sine-cosine embedding, ``[grid_h * grid_w, dim]``, row-major.

    Half the channels encode the row coordinate and half the columns
    (``dim // 4`` frequencies each for sin and cos), matching MAE's
    ``util/pos_embed.get_2d_sincos_pos_embed``.
    """
    if dim % 4 != 0:
        raise ValueError(
            f"sincos pos_embed needs dim divisible by 4 (got dim={dim}); "
            "use pos_embed='learned' otherwise"
        )
    grid_h, grid_w = _grid_hw(grid_size)

    rows = torch.arange(grid_h, dtype=dtype).reshape(grid_h, 1).expand(grid_h, grid_w)
    cols = torch.arange(grid_w, dtype=dtype).reshape(1, grid_w).expand(grid_h, grid_w)
    emb_h = _1d_sincos(dim // 2, rows.reshape(-1))
    emb_w = _1d_sincos(dim // 2, cols.reshape(-1))
    return torch.cat([emb_h, emb_w], dim=1)


def build_spatial_pos(
    dim: int,
    n_patches: int,
    kind: str = "learned",
    grid_size: GridSize | None = None,
    batch_dims: int = 0,
) -> torch.Tensor | nn.Parameter:
    """Build a positional table of shape ``(1,) * batch_dims + (n_patches, dim)``.

    Returns ``nn.Parameter`` for ``kind="learned"`` and a plain tensor for
    ``kind="sincos"``; :func:`add_spatial_pos` registers it accordingly.
    """
    if kind not in POS_EMBED_KINDS:
        raise ValueError(f"unknown pos_embed '{kind}'; expected one of {POS_EMBED_KINDS}")
    if n_patches <= 0:
        raise ValueError(f"n_patches must be positive, got {n_patches}")

    shape = (1,) * batch_dims + (n_patches, dim)
    if kind == "learned":
        pos = nn.Parameter(torch.zeros(shape))
        nn.init.trunc_normal_(pos, std=0.02)
        return pos

    grid = _grid_from_n_patches(n_patches) if grid_size is None else grid_size
    pos = get_2d_sincos_pos_embed(dim, grid)
    if pos.shape[0] != n_patches:
        raise ValueError(
            f"grid_size={grid} gives {pos.shape[0]} positions, "
            f"expected n_patches={n_patches}"
        )
    return pos.reshape(shape)


def add_spatial_pos(
    module: nn.Module,
    dim: int,
    n_patches: int,
    kind: str = "learned",
    grid_size: GridSize | None = None,
    batch_dims: int = 0,
) -> None:
    """Attach ``module.spatial_pos`` as a parameter or a fixed buffer.

    Also records the choice on ``module.pos_embed`` so it shows up in reprs and
    can be asserted in tests.
    """
    pos = build_spatial_pos(dim, n_patches, kind, grid_size, batch_dims)
    if isinstance(pos, nn.Parameter):
        module.register_parameter("spatial_pos", pos)
    else:
        module.register_buffer("spatial_pos", pos)
    module.pos_embed = kind
