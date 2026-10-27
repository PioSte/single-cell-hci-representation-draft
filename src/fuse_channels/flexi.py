"""FlexiViT patch-embedding resizing.

FlexiViT resizes the patch-embedding kernel with **PI-resize** (pseudoinverse
resize), not plain bilinear resize (``references/flexivit/paper.md:105``,
``:168-171``).  Bilinear resize is only the baseline; PI-resize is the inverse
of bilinear resizing in the patch space, so the inner product with a resized
patch is preserved as closely as possible.

The patch embedding is still one shared ``Conv2d`` parameter at a canonical
patch size.  At every other patch size the kernel is PI-resized on the fly.
"""

from __future__ import annotations

from functools import lru_cache

import torch
import torch.nn.functional as F


@lru_cache(maxsize=32)
def _bilinear_resize_matrix(
    p_in: int, p_out: int, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Bilinear resize operator ``B`` of shape ``[p_out**2, p_in**2]``.

    ``B @ vec(x)`` equals ``F.interpolate(x, (p_out, p_out))`` under row-major
    ``(row, col)`` flattening.  Build it by pushing each input pixel through
    the resize as its own batch element, then transpose the resulting
    ``[p_in**2, p_out**2]`` coefficient table into ``[p_out**2, p_in**2]``.
    """
    eye = torch.eye(p_in * p_in, dtype=dtype).reshape(
        p_in * p_in, 1, p_in, p_in
    )
    out = F.interpolate(
        eye, size=(p_out, p_out), mode="bilinear", align_corners=False
    )
    return out.reshape(p_in * p_in, p_out * p_out).T.contiguous()


@lru_cache(maxsize=32)
def _pi_resize_matrix(
    p_in: int, p_out: int, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """PI-resize operator ``P = pinv(B.T)`` of shape ``[p_out**2, p_in**2]``."""
    B = _bilinear_resize_matrix(p_in, p_out, dtype)
    return torch.linalg.pinv(B.T).contiguous()


def pi_resize_patch_embed(
    weight: torch.Tensor, new_patch_size: int
) -> torch.Tensor:
    """PI-resize a patch-embedding kernel to ``new_patch_size``.

    Parameters
    ----------
    weight:
        ``[out_channels, in_channels, p, p]`` (a ``Conv2d`` weight) or
        ``[out_channels, p, p]``.
    new_patch_size:
        Target square patch size.
    """
    if weight.ndim == 4:
        out_ch, in_ch, p_in, p_w = weight.shape
        if p_in != p_w:
            raise ValueError(f"patch kernel must be square, got {p_in}x{p_w}")
        flat = weight.reshape(out_ch * in_ch, p_in * p_in)
        lead_shape = (out_ch, in_ch)
    elif weight.ndim == 3:
        out_ch, p_in, p_w = weight.shape
        if p_in != p_w:
            raise ValueError(f"patch kernel must be square, got {p_in}x{p_w}")
        flat = weight.reshape(out_ch, p_in * p_in)
        lead_shape = (out_ch,)
    else:
        raise ValueError(f"expected a 3-D or 4-D kernel, got {tuple(weight.shape)}")

    if new_patch_size == p_in:
        return weight
    P = _pi_resize_matrix(p_in, int(new_patch_size), weight.dtype).to(weight.device)
    resized = flat @ P.T
    return resized.reshape(*lead_shape, new_patch_size, new_patch_size)
