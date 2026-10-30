"""The **acquisition fleet**: the discrete per-instrument ``pixel_size`` values.

The fleet is the set of acquisition instruments (microscopes/scanners) and
acquisition settings that the model is trained *and* deployed on. Each
configuration has one physical pixel size, so the fleet is a finite, sorted
table of ``pixel_size`` values -- not an interval. That matters because the
patch size is derived from it:

    p_b = round(target_um / pixel_size_b)        one p per fleet entry

A pixel size outside the table implies a ``p`` the model never trained, so it is
an **ingest-time error**, not something to snap onto a nearest lattice. Snapping
``pixel_size`` to its table entry inside a tolerance is the only rounding kept:
it makes ``p`` reproducible from float-noise metadata (``0.6500000000000001``
and ``0.65`` are the same instrument).

The library stays general: pass ``fleet=None`` to serve a dataset with a
different (or unknown) fleet; pass its own table to validate that one. The
constants below are *this* project's fleet, not a property of the architecture.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

#: The project's acquisition fleet, micrometres per pixel, sorted.
FLEET_PIXEL_SIZES: Tuple[float, ...] = (0.598, 0.646, 0.650, 0.652, 0.703)

#: Relative tolerance for matching metadata against a fleet entry. OME-TIFF
#: reports three to five significant decimals (``0.59766`` for ``0.598``), so
#: this is ~half a unit in the third decimal.
FLEET_RTOL = 1e-3


def sorted_fleet(fleet: Sequence[float]) -> Tuple[float, ...]:
    """Return the fleet sorted and deduplicated, rejecting non-positive sizes."""
    table = tuple(sorted({float(q) for q in fleet}))
    if not table:
        raise ValueError("fleet must contain at least one pixel size")
    if any(q <= 0 for q in table):
        raise ValueError(f"fleet pixel sizes must be positive, got {table}")
    return table


def canonical_fleet_pixel_size(
    pixel_size: float,
    fleet: Optional[Sequence[float]] = None,
    rtol: float = FLEET_RTOL,
) -> float:
    """Map one image's metadata ``pixel_size`` to its fleet entry.

    Returns the **table** value (not the metadata value) so that the derived
    ``p`` is reproducible from noisy floats. Raises when the value is not a
    fleet member: it implies an untrained patch size, so the right fix is the
    metadata or a training run, never a silent snap.

    Unit traps are called out in the error message, because ``598`` (nm),
    ``1/pixel_size`` (px/µm) and ``pixel_size`` (µm/px) are all common.
    """
    if pixel_size <= 0:
        raise ValueError(f"pixel_size must be positive, got {pixel_size}")
    table = sorted_fleet(fleet if fleet is not None else FLEET_PIXEL_SIZES)

    def nearest(value: float) -> Tuple[float, bool]:
        best = min(table, key=lambda q: abs(q - value))
        return best, abs(best - value) <= rtol * best

    best, ok = nearest(float(pixel_size))
    if ok:
        return best

    hints = []
    for label, variant in (
        ("nanometres (÷1000)", pixel_size / 1000.0),
        ("pixels per um (invert)", 1.0 / pixel_size),
        ("millimetres (x1000)", pixel_size * 1000.0),
    ):
        _, variant_ok = nearest(variant)
        if variant_ok:
            hints.append(label)
    hint = (
        f" (looks like {', '.join(hints)}; see FLEXIVIT.md section 0 unit traps)"
        if hints
        else ""
    )
    raise ValueError(
        f"pixel_size={pixel_size} um/px is not a member of the acquisition "
        f"fleet {list(table)}; it implies an untrained patch size -- fix the "
        f"metadata or add the instrument to the fleet and retrain{hint}"
    )


def fleet_patch_sizes(
    fleet: Optional[Sequence[float]] = None, target_um: float = 12.8
) -> Tuple[int, ...]:
    """The distinct ``p`` values implied by a fleet (the model's ``patch_sizes``).

    This is the only place the fleet's concrete numbers should turn into model
    configuration: ``patch_sizes`` must be derived from it, never hand-written,
    or the decoder and the loader can disagree about which instruments are in
    contract.
    """
    from .crops import patch_size_for_pixel_size

    table = sorted_fleet(fleet if fleet is not None else FLEET_PIXEL_SIZES)
    return tuple(sorted({patch_size_for_pixel_size(q, target_um) for q in table}))
