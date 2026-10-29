"""Cell-centred crop reading from raw channel planes + a cell-index mask.

The crop box is chosen so that it is an **exact multiple of the patch size**
implied by the image's pixel size::

    p_px     = round(target_um / pixel_size)      snapped to the trained patch sizes
    crop_px  = n_tokens * p_px             exact, no padding

Read from the raw image rather than from a pre-existing crop, so the box can
always be made to tile exactly; nothing is padded and no border is discarded
(the FlexiViT ``floor(H/p)`` contract never applies).

``n_tokens`` is a **fixed design parameter** -- a physical window of
``n_tokens * target_um`` micrometres per side -- not something derived from the
cell's size. Cells genuinely differ in size; what is held constant is the
observation window, so every sample yields exactly ``n_tokens**2`` tokens on
every instrument while ``p_px`` absorbs the pixel-size variation. The
consequences of a fixed window are deliberate:

* a cell **smaller** than the window is surrounded by background (use
  ``min_valid_tokens`` to reject slivers);
* a cell **larger** than the window is clipped by it. Pass
  ``require_full_cell=True`` to skip such cells instead, or raise
  ``n_tokens``. ``CellCrop.covers_cell`` records which case applies.

Cells that still cannot be placed -- ``crop_px`` larger than the image -- are
skipped and reported, never padded. Cells near (but not touching) a border get
an off-centre box: the window is shifted to start at the boundary so it stays
in-bounds. The caller is expected to have removed edge-touching cells from the
mask upstream.

Everything here is dependency-light: :mod:`tifffile` is imported lazily and only
by the TIFF helpers, so the geometry, planning, coverage and caching logic is
testable (and usable) without it.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch


# --------------------------------------------------------------------------- #
# geometry: pixel size -> patch size -> crop size
# --------------------------------------------------------------------------- #
def patch_size_for_pixel_size(
    pixel_size: float,
    target_um: float,
    allowed: Sequence[int],
    snap: str = "down",
) -> int:
    """Pick the patch size (in pixels) whose physical extent is closest to ``target_um``.

    ``snap="down"`` prefers the largest allowed ``p <= target_um / pixel_size`` (smaller
    physical token, more tokens, never larger than requested); ``"nearest"``
    minimises ``|p * pixel_size - target_um|`` with ties broken towards the smaller ``p``.
    """
    if pixel_size <= 0:
        raise ValueError(f"pixel_size must be positive, got {pixel_size}")
    if target_um <= 0:
        raise ValueError(f"target_um must be positive, got {target_um}")
    cands = sorted({int(p) for p in allowed})
    if not cands:
        raise ValueError("allowed must contain at least one patch size")
    if any(p < 1 for p in cands):
        raise ValueError(f"patch sizes must be positive, got {cands}")
    raw = target_um / pixel_size
    if snap == "down":
        feasible = [p for p in cands if p <= raw]
        return feasible[-1] if feasible else cands[0]
    if snap == "nearest":
        return min(cands, key=lambda p: (abs(p * pixel_size - target_um), p))
    raise ValueError(f"snap must be 'down' or 'nearest', got {snap!r}")


@dataclass(frozen=True)
class CropGeometry:
    """Pixel-domain geometry of one crop.

    ``crop_px == n_tokens * patch_size`` by construction: the crop tiles exactly.
    """

    patch_size: int
    n_tokens: int
    crop_px: int
    pixel_size: float
    target_um: float
    extent_um: float
    n_patches: int

    @classmethod
    def build(
        cls,
        pixel_size: float,
        target_um: float,
        allowed: Sequence[int],
        n_tokens: int,
        snap: str = "down",
    ) -> "CropGeometry":
        if n_tokens < 1:
            raise ValueError(f"n_tokens must be >= 1, got {n_tokens}")
        p = patch_size_for_pixel_size(pixel_size, target_um, allowed, snap=snap)
        return cls(
            patch_size=p,
            n_tokens=int(n_tokens),
            crop_px=int(n_tokens) * p,
            pixel_size=float(pixel_size),
            target_um=float(target_um),
            extent_um=p * float(pixel_size),
            n_patches=int(n_tokens) ** 2,
        )

    @property
    def extent_error_um(self) -> float:
        """Realised token extent minus the requested one; auditable snapping error."""
        return self.extent_um - self.target_um


def place_box(
    bbox: Tuple[int, int, int, int],
    crop_px: int,
    image_hw: Tuple[int, int],
) -> Optional[Tuple[int, int]]:
    """Top-left of a ``crop_px²`` box centred on ``bbox`` and clamped in-bounds.

    ``bbox`` is ``(y0, y1, x0, x1)`` in pixels, half-open. Returns ``None`` when
    the box cannot fit at all (image smaller than the crop in either axis).
    """
    y0, y1, x0, x1 = bbox
    H, W = image_hw
    if crop_px > H or crop_px > W:
        return None
    cy = (y0 + y1) / 2.0
    cx = (x0 + x1) / 2.0
    top = int(round(cy - crop_px / 2.0))
    left = int(round(cx - crop_px / 2.0))
    top = max(0, min(top, H - crop_px))
    left = max(0, min(left, W - crop_px))
    return top, left


# --------------------------------------------------------------------------- #
# mask -> per-token coverage
# --------------------------------------------------------------------------- #
def token_coverage(
    mask_px: np.ndarray,
    patch_size: int,
    mode: str = "any",
    threshold: float = 0.0,
) -> np.ndarray:
    """Per-token coverage of a binary pixel mask, shape ``[n_h, n_w]``.

    ``mode="any"``  token touches the mask (>= 1 pixel).
    ``mode="frac"`` token's masked fraction is ``> threshold``.

    The grid is ``floor(S/patch_size)`` on each axis, matching the patchify
    stride, so token ``(i, j)`` is exactly patch ``(i, j)``.
    """
    if mask_px.ndim != 2:
        raise ValueError(f"mask_px must be 2-D, got {mask_px.shape}")
    if patch_size < 1:
        raise ValueError(f"patch_size must be >= 1, got {patch_size}")
    p = patch_size
    nh, nw = mask_px.shape[0] // p, mask_px.shape[1] // p
    if nh < 1 or nw < 1:
        raise ValueError(
            f"mask {mask_px.shape} smaller than one {p}x{p} patch"
        )
    m = mask_px[: nh * p, : nw * p].astype(np.float32)
    blocks = m.reshape(nh, p, nw, p)
    if mode == "any":
        return blocks.max(axis=(1, 3)) > 0
    if mode == "frac":
        return blocks.mean(axis=(1, 3)) > threshold
    raise ValueError(f"mode must be 'any' or 'frac', got {mode!r}")


def token_coverage_torch(
    mask_px: torch.Tensor,
    patch_size: int,
    mode: str = "any",
    threshold: float = 0.0,
) -> torch.Tensor:
    """Torch twin of :func:`token_coverage`; ``mask_px`` is ``[..., H, W]``."""
    import torch.nn.functional as F

    m = mask_px.float()
    if m.ndim == 2:
        m = m[None, None]
    elif m.ndim == 3:
        m = m[:, None]
    if mode == "any":
        return F.max_pool2d(m, kernel_size=patch_size, stride=patch_size) > 0
    if mode == "frac":
        return F.avg_pool2d(m, kernel_size=patch_size, stride=patch_size) > threshold
    raise ValueError(f"mode must be 'any' or 'frac', got {mode!r}")


# --------------------------------------------------------------------------- #
# planning: one crop per cell id
# --------------------------------------------------------------------------- #
@dataclass
class CellCrop:
    """One planned crop. ``valid`` is the flat ``[n_patches]`` token mask."""

    image_id: str
    cell_id: int
    geometry: CropGeometry
    top: int
    left: int
    channel_ids: Tuple[int, ...]
    valid: Optional[np.ndarray] = None
    coverage: Optional[np.ndarray] = None
    n_valid: int = 0
    path: Optional[str] = None
    cell_hw: Tuple[int, int] = (0, 0)
    covers_cell: bool = False

    @property
    def shape_hw(self) -> Tuple[int, int]:
        return self.geometry.crop_px, self.geometry.crop_px

    def to_dict(self) -> dict:
        d = {
            "image_id": self.image_id,
            "cell_id": int(self.cell_id),
            "geometry": asdict(self.geometry),
            "top": int(self.top),
            "left": int(self.left),
            "channel_ids": [int(c) for c in self.channel_ids],
            "n_valid": int(self.n_valid),
            "cell_hw": [int(v) for v in self.cell_hw],
            "covers_cell": bool(self.covers_cell),
        }
        if self.coverage is not None:
            # row-major bool grid, the same ordering as ChannelSet.patch_idx
            d["coverage"] = np.asarray(self.coverage, dtype=bool).reshape(-1).tolist()
        if self.path is not None:
            d["path"] = self.path
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "CellCrop":
        geom = CropGeometry(**d["geometry"])
        cov = d.get("coverage")
        coverage = (
            np.asarray(cov, dtype=bool).reshape(geom.n_tokens, geom.n_tokens)
            if cov is not None
            else None
        )
        crop = cls(
            image_id=d["image_id"],
            cell_id=int(d["cell_id"]),
            geometry=geom,
            top=int(d["top"]),
            left=int(d["left"]),
            channel_ids=tuple(int(c) for c in d["channel_ids"]),
            coverage=coverage,
            n_valid=int(d.get("n_valid", 0)),
            path=d.get("path"),
            cell_hw=tuple(int(v) for v in d.get("cell_hw", (0, 0))),
            covers_cell=bool(d.get("covers_cell", False)),
        )
        crop.valid = None if coverage is None else coverage.reshape(-1)
        return crop


def cell_n_tokens(
    cell_hw_px: Tuple[int, int],
    patch_size: int,
    min_n: int = 2,
    max_n: Optional[int] = None,
) -> int:
    """Tokens per side needed to cover a cell: ``max(min_n, ceil(max(h, w) / p))``.

    There is **no upper bound**: nothing in the model caps the patch count
    (``N = 10_000`` works), and the dense buffers are per-batch, so a larger cell
    simply costs more. ``max_n`` is an optional *budget*, not a model limit; when
    given, :func:`plan_patches` reports cells that exceed it instead of silently
    clipping them.

    ``min_n`` defaults to 2 because a ``1x1`` grid disables the MAE patch region
    (``floor(1 * patch_ratio) == 0``), so a 1-token cell would be trained with a
    different objective than every other cell. Unlike ``max_n`` this enlarges the
    window rather than cropping content, so it is not a lossy truncation.
    """
    if patch_size < 1:
        raise ValueError(f"patch_size must be >= 1, got {patch_size}")
    if min_n < 1:
        raise ValueError(f"min_n must be >= 1, got {min_n}")
    if max_n is not None and max_n < min_n:
        raise ValueError(f"need max_n >= min_n, got {max_n} < {min_n}")
    h, w = int(cell_hw_px[0]), int(cell_hw_px[1])
    need = max(1, math.ceil(max(h, w) / patch_size))
    n = max(min_n, need)
    return int(n if max_n is None else min(n, max_n))


def plan_patches(
    mask: np.ndarray,
    image_hw: Tuple[int, int],
    geometry: Optional[CropGeometry] = None,
    image_id: str = "image",
    channel_ids: Sequence[int] = (),
    coverage_mode: str = "any",
    coverage_threshold: float = 0.0,
    min_valid_tokens: int = 1,
    require_full_cell: bool = False,
    *,
    pixel_size: Optional[float] = None,
    target_um: Optional[float] = None,
    allowed: Optional[Sequence[int]] = None,
    n_tokens=None,
    min_n: int = 2,
    max_n: Optional[int] = None,
    snap: str = "down",
) -> Tuple[List[CellCrop], List[dict]]:
    """Plan one token window per non-zero id in ``mask``.

    Two modes:

    * pass ``geometry`` -- one fixed window size for every cell;
    * pass ``pixel_size``/``target_um``/``allowed`` and ``n_tokens`` -- the window is
      derived per image, and ``n_tokens="cell"`` additionally sizes it per
      cell (``ceil(cell_side / p)``, at least ``min_n``), i.e. a variable
      patch count driven by cell size and the pixel-size-adjusted ``p``.

    Patch count is not bounded above by the model. ``max_n`` is an optional
    budget: a cell needing more than ``max_n`` tokens per side is skipped and
    reported (never silently clipped), like any other cell that does not fit.

    Either way the window is centred on the cell's bounding box when that fits
    and shifted to the image boundary when the extension would cross one, so an
    interior cell close to a border still gets a full, in-bounds box.

    Returns ``(crops, skipped)``. A cell is skipped (never padded) when the box
    cannot fit inside ``image_hw``, when fewer than ``min_valid_tokens`` patches
    touch the cell, or when ``require_full_cell`` is set and the cell is larger
    than the window.
    """
    if geometry is None:
        if pixel_size is None or target_um is None or allowed is None:
            raise ValueError(
                "pass either geometry=, or pixel_size= with target_um= and allowed="
            )
        if n_tokens is None:
            raise ValueError(
                "pass n_tokens= (an int, or 'cell' to size the window per cell)"
            )
        base_geometry = (
            None
            if n_tokens == "cell"
            else CropGeometry.build(pixel_size, target_um, allowed, int(n_tokens), snap=snap)
        )
    else:
        if n_tokens is not None and n_tokens != "cell":
            raise ValueError("pass geometry= or n_tokens=, not both")
        base_geometry = geometry
        if n_tokens == "cell":
            base_geometry = None
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2-D, got {mask.shape}")
    H, W = int(image_hw[0]), int(image_hw[1])
    crops: List[CellCrop] = []
    skipped: List[dict] = []
    ids = np.unique(mask)
    ids = ids[ids != 0]
    for cid in ids:
        ys, xs = np.nonzero(mask == cid)
        if ys.size == 0:
            continue
        bbox = (int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1)
        geom = base_geometry
        if geom is None:
            # variable-N mode: size this cell's window from its own extent
            p = patch_size_for_pixel_size(pixel_size, target_um, allowed, snap=snap)
            need = cell_n_tokens(
                (bbox[1] - bbox[0], bbox[3] - bbox[2]), p, min_n=min_n
            )
            if max_n is not None and need > max_n:
                skipped.append(
                    {
                        "image_id": image_id,
                        "cell_id": int(cid),
                        "reason": (
                            f"cell needs {need} tokens/side, exceeds max_n={max_n}; "
                            "raise max_n or set it to None to process it"
                        ),
                    }
                )
                continue
            geom = CropGeometry.build(pixel_size, target_um, allowed, need, snap=snap)
        box = place_box(bbox, geom.crop_px, (H, W))
        if box is None:
            skipped.append(
                {
                    "image_id": image_id,
                    "cell_id": int(cid),
                    "reason": f"crop_px={geom.crop_px} exceeds image {H}x{W}",
                }
            )
            continue
        top, left = box
        s = geom.crop_px
        # the whole cell fits only if its bbox is inside the placed window
        covers = (
            bbox[0] >= top
            and bbox[1] <= top + s
            and bbox[2] >= left
            and bbox[3] <= left + s
        )
        if require_full_cell and not covers:
            skipped.append(
                {
                    "image_id": image_id,
                    "cell_id": int(cid),
                    "reason": (
                        f"cell {bbox[1] - bbox[0]}x{bbox[3] - bbox[2]} px larger "
                        f"than the {s} px window"
                    ),
                }
            )
            continue
        sub = mask[top : top + s, left : left + s]
        coverage = token_coverage(
            sub == cid,
            geom.patch_size,
            mode=coverage_mode,
            threshold=coverage_threshold,
        )
        n_valid = int(coverage.sum())
        if n_valid < min_valid_tokens:
            skipped.append(
                {
                    "image_id": image_id,
                    "cell_id": int(cid),
                    "reason": f"only {n_valid} patches touch the cell",
                }
            )
            continue
        crop = CellCrop(
            image_id=image_id,
            cell_id=int(cid),
            geometry=geom,
            top=top,
            left=left,
            channel_ids=tuple(int(c) for c in channel_ids),
            coverage=coverage,
            n_valid=n_valid,
            cell_hw=(bbox[1] - bbox[0], bbox[3] - bbox[2]),
            covers_cell=covers,
        )
        crop.valid = coverage.reshape(-1)
        crops.append(crop)
    return crops, skipped


# --------------------------------------------------------------------------- #
# reading
# --------------------------------------------------------------------------- #
class TiffStack:
    """Region reader over one plane per channel.

    ``tifffile`` is imported lazily. Planes are opened once and sliced, which is
    the fast path for uncompressed or memory-mappable stacks. For compressed or
    pyramidal OME-TIFF, pass a custom ``plane_loader`` that uses
    ``tifffile.aszarr`` + ``zarr`` so only the requested box is decoded; this
    class then simply calls it.
    """

    def __init__(self, paths: Sequence[str], plane_loader: Optional[Callable] = None):
        self.paths = [str(p) for p in paths]
        self._plane_loader = plane_loader
        self._planes: List[Optional[np.ndarray]] = [None] * len(self.paths)

    def _plane(self, i: int) -> np.ndarray:
        if self._planes[i] is None:
            if self._plane_loader is not None:
                self._planes[i] = np.asarray(self._plane_loader(self.paths[i]))
            else:
                import tifffile  # noqa: PLC0415  (optional dependency)

                self._planes[i] = tifffile.imread(self.paths[i])
        return self._planes[i]

    def read_box(self, top: int, left: int, size: int) -> np.ndarray:
        """``[C, size, size]`` float32 crop; singleton leading axes are squeezed."""
        out = []
        for i in range(len(self.paths)):
            plane = np.asarray(self._plane(i))
            while plane.ndim > 2 and plane.shape[0] == 1:
                plane = plane[0]
            if plane.ndim != 2:
                raise ValueError(
                    f"{self.paths[i]} must decode to a 2-D plane, got {plane.shape}"
                )
            out.append(plane[top : top + size, left : left + size])
        return np.stack(out, axis=0).astype(np.float32, copy=False)


def read_crop_tensors(
    stack: TiffStack,
    crop: CellCrop,
) -> torch.Tensor:
    """``[C, 1, S, S]`` tensor for one planned crop."""
    s = crop.geometry.crop_px
    arr = stack.read_box(crop.top, crop.left, s)
    return torch.from_numpy(arr).unsqueeze(1)


# --------------------------------------------------------------------------- #
# caching
# --------------------------------------------------------------------------- #
def write_crop_cache(
    out_dir: str | Path,
    crop: CellCrop,
    images: torch.Tensor | np.ndarray,
    index: List[dict],
) -> str:
    """Write one crop as a channel-stack TIFF plus a JSON sidecar; append to ``index``."""
    import tifffile  # noqa: PLC0415  (optional dependency)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    name = f"{crop.image_id}_cell{int(crop.cell_id):06d}"
    tif_path = out / f"{name}.tif"
    arr = images.detach().cpu().numpy() if isinstance(images, torch.Tensor) else images
    if arr.ndim == 4:
        arr = arr[:, 0]
    # pages = channels; no compression by default so region reads stay cheap
    tifffile.imwrite(tif_path, np.ascontiguousarray(arr), photometric="minisblack")
    crop.path = str(tif_path)
    meta = crop.to_dict()
    meta["pixel_size"] = crop.geometry.pixel_size
    meta["patch_size"] = crop.geometry.patch_size
    meta["n_tokens"] = crop.geometry.n_tokens
    meta["target_um"] = crop.geometry.target_um
    meta["extent_um"] = crop.geometry.extent_um
    (out / f"{name}.json").write_text(json.dumps(meta, indent=2))
    index.append(meta)
    return str(tif_path)


class CellCropDataset(torch.utils.data.Dataset):
    """Serve ``(images, channel_ids, pixel_size, patch_size, valid)`` per cell.

    Two modes:

    * ``crops`` given: items are already planned :class:`CellCrop` objects read
      through a :class:`TiffStack` per image (``readers`` maps image id to stack).
    * ``index_path`` given: items are cached TIFFs plus their JSON sidecars.

    ``valid`` is the flat ``[n_patches]`` boolean token mask derived from the
    cell's own instance mask, in the same row-major order as
    ``ChannelSet.patch_idx``. It is *not* an MAE visibility mask: it says which
    tokens contain the cell at all. AND it into the visibility mask to keep the
    encoder padding-free without reconstructing background.
    """

    def __init__(
        self,
        crops: Optional[Sequence[CellCrop]] = None,
        readers: Optional[Dict[str, TiffStack]] = None,
        index_path: Optional[str | Path] = None,
        return_valid: bool = True,
    ) -> None:
        self.return_valid = bool(return_valid)
        self.readers = readers or {}
        if index_path is not None:
            rows = json.loads(Path(index_path).read_text())
            self.crops = [CellCrop.from_dict(r) for r in rows]
            self.mode = "cache"
        elif crops is not None:
            self.crops = list(crops)
            self.mode = "raw"
        else:
            raise ValueError("pass crops= or index_path=")
        if self.mode == "raw":
            missing = {c.image_id for c in self.crops} - set(self.readers)
            if missing:
                raise ValueError(f"no TiffStack for image ids: {sorted(missing)}")

    def __len__(self) -> int:
        return len(self.crops)

    def __getitem__(self, i: int) -> dict:
        crop = self.crops[i]
        if self.mode == "cache":
            import tifffile  # noqa: PLC0415

            arr = np.asarray(tifffile.imread(crop.path))
            if arr.ndim == 2:
                arr = arr[None]
            images = torch.from_numpy(arr).unsqueeze(1).float()
        else:
            images = read_crop_tensors(self.readers[crop.image_id], crop)
        item = {
            "images": images,
            "channel_ids": torch.tensor(crop.channel_ids, dtype=torch.long),
            "pixel_size": float(crop.geometry.pixel_size),
            "patch_size": int(crop.geometry.patch_size),
            "image_id": crop.image_id,
            "cell_id": int(crop.cell_id),
        }
        if self.return_valid:
            if crop.valid is None:
                raise ValueError(
                    f"crop {crop.image_id}/{crop.cell_id} has no coverage; "
                    "re-plan with plan_patches()"
                )
            item["valid"] = torch.from_numpy(np.asarray(crop.valid, dtype=bool))
        return item


def collate_crops(batch: Sequence[dict]):
    """Padding-free collate: lists for the model, stacked ``valid`` per sample."""
    out = {
        "images": [b["images"] for b in batch],
        "channel_ids": [b["channel_ids"] for b in batch],
        "pixel_sizes": [b["pixel_size"] for b in batch],
        "patch_sizes": [b["patch_size"] for b in batch],
        "image_ids": [b["image_id"] for b in batch],
        "cell_ids": [b["cell_id"] for b in batch],
    }
    if "valid" in batch[0]:
        out["valid"] = [b["valid"] for b in batch]
    return out
