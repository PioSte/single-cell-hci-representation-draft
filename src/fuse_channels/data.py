"""Synthetic data for smoke tests / examples.

Real HCI panels are variable-channel; this toy mirrors that structure: every
sample has its own channel count and its own (unordered) channel-id subset, and
channels are generated from a shared low-rank latent so that cross-channel
structure exists for the MAE to learn.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

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
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.n_channel_ids = n_channel_ids
        g = torch.Generator().manual_seed(seed)

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

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        img, channel_ids = self.items[idx]
        return img.clone(), channel_ids.clone()


def collate_fn(batch: Sequence[Tuple[torch.Tensor, torch.Tensor]]):
    """Return a padding-free batch: a list of images and a list of ids."""
    images = [b[0] for b in batch]
    ids = [b[1] for b in batch]
    return images, ids
