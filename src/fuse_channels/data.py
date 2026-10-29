"""Synthetic data for smoke tests / examples.

Real HCI panels are variable-channel; this toy mirrors that structure: every
sample has its own channel count and its own (unordered) channel-id subset.
Each channel image is an independent random field smoothed by a fixed 3x3 blur,
plus a per-channel offset derived from its channel id: there is no shared
across-channel latent, and the only structure tying channels together is the
channel-id offset.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import torch
from torch.utils.data import Dataset


class SyntheticChannelDataset(Dataset):
    def __init__(
        self,
        n_samples: int = 64,
        image_size: int = 16,
        c_min: int = 1,
        c_max: int = 8,
        n_channel_ids: int = 32,
        seed: int = 0,
        pixel_size: Optional[Union[float, Sequence[float]]] = None,
        pixel_size_range: Optional[Sequence[float]] = None,
        return_pixel_size: bool = True,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.n_channel_ids = n_channel_ids
        self.return_pixel_size = bool(return_pixel_size)
        g = torch.Generator().manual_seed(seed)
        self.pixel_sizes = self._resolve_pixel_sizes(
            n_samples, pixel_size, pixel_size_range, g
        )
        if self.pixel_sizes is None:
            raise ValueError(
                "pixel size is required metadata: pass pixel_size=<µm/px> or "
                "pixel_size_range=(lo, hi). There is no 'unknown pixel size'."
            )

        # shared latent basis over channels x pixels
        self.items: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for i in range(n_samples):
            C = int(torch.randint(c_min, c_max + 1, (1,), generator=g).item())
            channel_ids = torch.randperm(n_channel_ids, generator=g)[:C]
            latent = torch.randn(C, 1, image_size, image_size, generator=g)
            # smooth the latent a little so patches are spatially coherent
            k = torch.tensor(
                [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]]
            ).view(1, 1, 3, 3)
            k = k / k.sum()
            import torch.nn.functional as F

            lat = F.conv2d(latent, k, padding=1)
            # per-channel offset derived from channel id (shared structure)
            off = (channel_ids.float() / n_channel_ids).view(C, 1, 1, 1)
            img = lat + off
            self.items.append((img, channel_ids))

    @staticmethod
    def _resolve_pixel_sizes(
        n_samples: int,
        pixel_size,
        pixel_size_range=None,
        generator: Optional[torch.Generator] = None,
    ) -> Optional[List[float]]:
        if pixel_size is not None and pixel_size_range is not None:
            raise ValueError("pass pixel_size or pixel_size_range, not both")
        if pixel_size_range is not None:
            lo, hi = (float(v) for v in pixel_size_range)
            if lo <= 0 or hi < lo:
                raise ValueError(
                    f"pixel_size_range must be positive and ordered, got {pixel_size_range}"
                )
            return torch.empty(n_samples).uniform_(
                lo, hi, generator=generator
            ).tolist()
        if pixel_size is None:
            return None  # caller raises: pixel size is required
        if isinstance(pixel_size, (int, float)):
            return [float(pixel_size)] * n_samples
        values = [float(v) for v in pixel_size]
        if len(values) != n_samples:
            raise ValueError(
                f"pixel_size must be a scalar or {n_samples} values, got {len(values)}"
            )
        return values

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        img, channel_ids = self.items[idx]
        if self.return_pixel_size:
            return img.clone(), channel_ids.clone(), self.pixel_sizes[idx]
        return img.clone(), channel_ids.clone()


def collate_fn(batch):
    """Return a padding-free batch.

    Supports both the legacy ``(image, channel_ids)`` items and
    ``(image, channel_ids, pixel_size)`` items.  The third value is returned as
    a list of per-sample µm/px scalars when present.
    """
    images = [b[0] for b in batch]
    ids = [b[1] for b in batch]
    if batch and len(batch[0]) == 3:
        return images, ids, [b[2] for b in batch]
    return images, ids
