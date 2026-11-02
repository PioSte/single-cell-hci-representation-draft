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

import torch
import torch.nn as nn

POS_EMBED_KINDS = ("learned", "sincos", "none", "fractional")

GridSize = int | tuple[int, int]


def _grid_hw(grid_size: GridSize) -> tuple[int, int]:
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


def fractional_1d_sincos(dim: int, positions: torch.Tensor, base: float = 10000.0) -> torch.Tensor:
    """Sine-cosine embedding of continuous positions in ``[0, 1]``."""
    if dim % 2 != 0:
        raise ValueError(f"fractional sincos needs an even dim, got {dim}")
    half = dim // 2
    omega = 1.0 / (
        base ** (torch.arange(half, dtype=positions.dtype, device=positions.device) / half)
    )
    out = positions.reshape(-1, 1) * omega.reshape(1, -1)
    return torch.cat([out.sin(), out.cos()], dim=1)


def get_fractional_sincos_pos_embed_uv(
    dim: int,
    v: torch.Tensor,
    u: torch.Tensor,
    base: float = 10000.0,
) -> torch.Tensor:
    """Factorized fractional PE from pre-computed row/column coordinates.

    ``v`` and ``u`` are the scalar coordinates fed to ``g_row`` / ``g_col``, so
    callers can supply grid fractions (``(r+0.5)/grid_h``) or metric
    coordinates (``(r+0.5)*p*pixel_size/ref_um``).  The two full-width axis embeddings
    are added.
    """
    if dim % 2 != 0:
        raise ValueError(f"fractional sincos PE needs an even dim, got {dim}")
    row_pe = fractional_1d_sincos(dim, v, base=base)
    col_pe = fractional_1d_sincos(dim, u, base=base)
    return row_pe + col_pe


def get_fractional_sincos_pos_embed(
    dim: int,
    rows: torch.Tensor,
    cols: torch.Tensor,
    grid_h,
    grid_w,
    base: float = 10000.0,
) -> torch.Tensor:
    """Factorized fractional absolute PE for arbitrary dynamic grids.

    For each position ``(r, c)``::

        v = (r + 0.5) / grid_h
        u = (c + 0.5) / grid_w
        PE(r, c) = g_row(v) + g_col(u)

    The two full-width axis embeddings are added.  The result is defined for
    any ``grid_h x grid_w`` and has no lookup table to resize.
    """
    if dim % 2 != 0:
        raise ValueError(f"fractional sincos PE needs an even dim, got {dim}")
    dtype = torch.float32
    device = rows.device
    gh = _as_float(grid_h, device, dtype).expand_as(rows)
    gw = _as_float(grid_w, device, dtype).expand_as(cols)
    v = (rows.to(dtype) + 0.5) / gh
    u = (cols.to(dtype) + 0.5) / gw
    row_pe = fractional_1d_sincos(dim, v, base=base)
    col_pe = fractional_1d_sincos(dim, u, base=base)
    return row_pe + col_pe


def _as_float(value, device, dtype):
    if torch.is_tensor(value):
        return value.to(device=device, dtype=dtype)
    return torch.as_tensor(value, device=device, dtype=dtype)


class FractionalSpatialPE(nn.Module):
    """Decoder absolute PE as a function of fractional grid coordinates.

    ``mode="sincos"`` (default) is a fixed factorized sine-cosine function.
    ``mode="mlp"`` evaluates a small learned MLP on each scalar coordinate and
    sums the two axis embeddings.  Neither variant has a grid-sized table, so
    both work for any grid and need no resampling when the patch size changes.
    """

    def __init__(
        self,
        dim: int,
        mode: str = "sincos",
        base: float = 10000.0,
        hidden: int | None = None,
    ) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"fractional PE needs an even dim, got {dim}")
        if mode not in ("sincos", "mlp"):
            raise ValueError(f"unknown fractional PE mode '{mode}'")
        self.dim = dim
        self.mode = mode
        self.base = base
        if mode == "mlp":
            hidden = hidden or dim
            self.row_mlp = nn.Sequential(nn.Linear(1, hidden), nn.GELU(), nn.Linear(hidden, dim))
            self.col_mlp = nn.Sequential(nn.Linear(1, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(
        self, rows, cols, grid_h, grid_w, *, step_um=None, ref_um: float = 1.0
    ) -> torch.Tensor:
        """``step_um`` (``patch_size * pixel_size`` per target) selects metric coordinates.

        With ``step_um=None`` the coordinates are grid fractions
        ``(r+0.5)/grid_h``, which stretch per sample.  With ``step_um`` given
        they are metric -- ``(r+0.5)*step_um/ref_um`` -- and must use the same
        ``ref_um`` as the encoder RoPE, otherwise mask tokens and visible
        tokens live in different coordinate frames.
        """
        dtype = torch.float32
        device = rows.device
        if step_um is not None:
            step = _as_float(step_um, device, dtype).expand_as(rows) / ref_um
            v = (rows.to(dtype) + 0.5) * step
            u = (cols.to(dtype) + 0.5) * step
        else:
            gh = _as_float(grid_h, device, dtype).expand_as(rows)
            gw = _as_float(grid_w, device, dtype).expand_as(cols)
            v = (rows.to(dtype) + 0.5) / gh
            u = (cols.to(dtype) + 0.5) / gw
        if self.mode == "sincos":
            return get_fractional_sincos_pos_embed_uv(self.dim, v, u, base=self.base)
        row_pe = self.row_mlp(v.reshape(-1, 1))
        col_pe = self.col_mlp(u.reshape(-1, 1))
        return row_pe + col_pe

    def extra_repr(self) -> str:
        return f"dim={self.dim}, mode={self.mode}"


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
    if kind in ("none", "fractional"):
        raise ValueError(
            f"pos_embed '{kind}' is computed from grid coordinates, not a table;"
            " use FractionalSpatialPE or skip the table entirely"
        )
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
            f"grid_size={grid} gives {pos.shape[0]} positions, expected n_patches={n_patches}"
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
    if kind == "none":
        # ``pos_embed`` is a dynamic marker on an arbitrary nn.Module, so the
        # static type checkers cannot see the attribute; setattr keeps ty quiet
        # (the module stubs do not declare ``pos_embed``).
        setattr(module, "pos_embed", "none")  # noqa: B010
        return
    pos = build_spatial_pos(dim, n_patches, kind, grid_size, batch_dims)
    if isinstance(pos, nn.Parameter):
        module.register_parameter("spatial_pos", pos)
    else:
        module.register_buffer("spatial_pos", pos)
    setattr(module, "pos_embed", kind)  # noqa: B010
