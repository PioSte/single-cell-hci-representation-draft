"""Dynamic Channel-Patch (DCP) masking (ChA-MAEViT, Eq. 1).

DCP combines the two masking axes required for multi-channel-imaging MAE:

1. **Random patch masking** -- per channel, a fixed number
   ``floor(n_patches * patch_ratio)`` of spatial positions is masked with a
   ``randperm`` draw; masks are sampled independently across channels.
2. **Dynamic channel masking** -- :class:`HierarchicalChannelSampler` draws the
   number of *visible* channels ``m ~ U{min_visible..C}`` (``min_visible=1``
   by default) and then a uniform ``m``-subset, so the channel mask always
   covers whole channels and can satisfy a minimum-content guard by
   construction.

A scalar ``s ~ U(0, 1)`` selects the region of Eq. 1::

    mask = patch_mask                  if s < p_patch
           channel_mask                if s < p_patch + p_channel
           patch_mask | channel_mask   otherwise

``mode_sampling="batch"`` (default) draws ``s`` once per call, i.e. once per
training step, matching ChA-MAEViT's per-iteration alternation;
``mode_sampling="sample"`` draws one ``s`` per sample.

The returned boolean mask marks *visible* tokens; masked tokens are omitted
from the padding-free tensor, never zeroed.  Channels dropped by the channel
mask stay in ``images`` and are reconstructed as MAE targets.

Configurations of interest: ``p_patch=0, p_channel=1`` is HCS-only,
``p_patch=1, p_channel=0`` is patch-only, and ``p_patch=p_channel=0`` is the
combined region (ChA-MAEViT's default).
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence

import torch


class HierarchicalChannelSampler:
    """Hierarchical Channel Sampling (ChannelViT).

    For a sample with ``C`` channels, draw the number of visible channels
    ``m ~ U{min_visible, ..., C}`` (``min_visible=1`` by default) and then a
    uniform ``m``-subset of channels.  The returned ``[C]`` boolean mask marks
    *visible* channels: every patch of a visible channel is visible and every
    dropped channel is fully masked.  Requiring ``min_visible > 1`` lets
    callers satisfy a minimum-content guard by construction instead of
    post-hoc repairs.

    Dropped channels are not removed from the data; they remain in ``images``
    and become reconstruction targets in the MAE.  This is the single
    channel-subset sampler -- DCP's channel region reuses it, so there is no
    second, independent channel-dropping mechanism.
    """

    def __call__(
        self,
        n_channels: int,
        device: Optional[torch.device] = None,
        generator: Optional[torch.Generator] = None,
        min_visible: int = 1,
    ) -> torch.Tensor:
        """Return a ``[n_channels]`` boolean visible-channel mask.

        ``min_visible`` is the smallest number of channels kept visible;
        ``m ~ U{min_visible..n_channels}``.  It defaults to ``1``, preserving
        the standalone ChannelViT draw ``m ~ U{1..C}``.
        """
        if n_channels < 1:
            raise ValueError(f"n_channels must be >= 1, got {n_channels}")
        if min_visible < 1 or min_visible > n_channels:
            raise ValueError(
                f"min_visible must be in [1, {n_channels}], got {min_visible}"
            )
        m = int(
            torch.randint(
                min_visible,
                n_channels + 1,
                (1,),
                device=device,
                generator=generator,
            ).item()
        )
        visible = torch.zeros(n_channels, dtype=torch.bool, device=device)
        perm = torch.randperm(n_channels, device=device, generator=generator)
        visible[perm[:m]] = True
        return visible


class DCPVisibilitySampler:
    """ChA-MAEViT Eq. 1 dynamic channel-patch visibility sampler.

    Parameters
    ----------
    n_patches:
        Number of spatial patches per channel (``N``).
    p_patch, p_channel:
        Probabilities of the patch-only and channel-only regions.  They must
        be non-negative and sum to at most ``1``; the remainder selects the
        combined region.
    patch_ratio:
        Fixed fraction of patches masked *per channel* in the patch region
        (``r_p`` in the paper, ``0.75`` by default).
    min_visible_channels:
        Minimum number of channels that must keep at least one visible patch.
        The channel and combined regions satisfy it by construction: HCS draws
        ``m ~ U{min(min_visible_channels, C)..C}`` visible channels, so a
        dropped channel is never patched back to visibility.
    max_masked_channels_frac:
        Optional cap on the number of whole channels the channel mask may
        drop, as a fraction of ``C``.  The default ``1.0`` means no cap,
        matching ``m ~ U{1..C}``.
    mode_sampling:
        ``"batch"`` (default) draws the Eq. 1 selection value ``s`` once per
        call, so every sample in a batch shares one selected region (the
        channel -> patch fallback for very small samples can still change the
        *applied* mode); ``"sample"`` draws one ``s`` per sample.

    After each call, ``last_modes`` records the mode actually applied to each
    sample (``"patch"``, ``"channel"`` or ``"combined"``), including the
    channel -> patch fallback for ``C <= min_visible_channels``.
    """

    _MAX_ATTEMPTS = 10

    def __init__(
        self,
        n_patches: int,
        p_patch: float = 0.5,
        p_channel: float = 0.5,
        patch_ratio: float = 0.75,
        min_visible_channels: int = 1,
        max_masked_channels_frac: float = 1.0,
        mode_sampling: str = "batch",
    ) -> None:
        if n_patches < 1:
            raise ValueError(f"n_patches must be >= 1, got {n_patches}")
        if p_patch < 0 or p_channel < 0:
            raise ValueError(
                f"p_patch and p_channel must be non-negative, "
                f"got {p_patch} and {p_channel}"
            )
        if p_patch + p_channel > 1:
            raise ValueError(
                f"p_patch + p_channel must be <= 1, got {p_patch + p_channel}"
            )
        if not 0.0 < patch_ratio < 1.0:
            raise ValueError(f"patch_ratio must be in (0, 1), got {patch_ratio}")
        patch_region_used = p_patch > 0 or p_patch + p_channel < 1
        if patch_region_used and math.floor(n_patches * patch_ratio) < 1:
            raise ValueError(
                "n_patches * patch_ratio must be >= 1 when the patch region "
                "is used; got "
                f"n_patches={n_patches}, patch_ratio={patch_ratio}"
            )
        if min_visible_channels < 1:
            raise ValueError(
                f"min_visible_channels must be >= 1, got {min_visible_channels}"
            )
        if not 0.0 <= max_masked_channels_frac <= 1.0:
            raise ValueError(
                "max_masked_channels_frac must be in [0, 1], got "
                f"{max_masked_channels_frac}"
            )
        if mode_sampling not in ("batch", "sample"):
            raise ValueError(
                f"mode_sampling must be 'batch' or 'sample', got {mode_sampling!r}"
            )

        self.n_patches = n_patches
        self.p_patch = p_patch
        self.p_channel = p_channel
        self.patch_ratio = patch_ratio
        self.min_visible_channels = min_visible_channels
        self.max_masked_channels_frac = max_masked_channels_frac
        self.mode_sampling = mode_sampling
        self.channel_sampler = HierarchicalChannelSampler()
        self.last_modes: List[str] = []

    # ------------------------------------------------------------------ #
    def _uniform(self, device, generator) -> float:
        return float(
            torch.rand(
                (), device=device, generator=generator, dtype=torch.float32
            ).item()
        )

    def _randint(self, high: int, device, generator) -> int:
        return int(
            torch.randint(0, high, (1,), device=device, generator=generator).item()
        )

    def _mode_from_s(self, s: float) -> str:
        if s < self.p_patch:
            return "patch"
        if s < self.p_patch + self.p_channel:
            return "channel"
        return "combined"

    def _select_modes(self, n_samples: int, device, generator) -> List[str]:
        if self.mode_sampling == "batch":
            s = self._uniform(device, generator)
            return [self._mode_from_s(s)] * n_samples
        return [
            self._mode_from_s(self._uniform(device, generator))
            for _ in range(n_samples)
        ]

    # ------------------------------------------------------------------ #
    def _draw_visible(
        self, n_channels: int, device, generator, mode: str
    ) -> torch.Tensor:
        """Draw a ``[C, N]`` boolean visible mask for one sample."""
        visible = torch.ones(
            n_channels, self.n_patches, dtype=torch.bool, device=device
        )

        if mode in ("patch", "combined"):
            n_masked = int(math.floor(self.n_patches * self.patch_ratio))
            for c in range(n_channels):
                if n_masked:
                    idx = torch.randperm(
                        self.n_patches, device=device, generator=generator
                    )[:n_masked]
                    visible[c, idx] = False

        if mode in ("channel", "combined"):
            channel_visible = self.channel_sampler(
                n_channels,
                device=device,
                generator=generator,
                min_visible=self._min_content_channels(n_channels),
            )
            dropped = int((~channel_visible).sum().item())
            cap = int(math.floor(n_channels * self.max_masked_channels_frac))
            if dropped > cap:
                # Optional cap: uniformly restore enough channels instead of
                # dropping more than requested.
                masked_idx = torch.nonzero(~channel_visible, as_tuple=False).squeeze(1)
                restore = masked_idx[
                    torch.randperm(masked_idx.numel(), device=device, generator=generator)[
                        : dropped - cap
                    ]
                ]
                channel_visible[restore] = True
            # Equivalently, ``visible &= ~channel_mask`` where the channel
            # mask is constant down the dropped channels' rows.
            visible[~channel_visible] = False

        return visible

    # ------------------------------------------------------------------ #
    def _min_content_channels(self, n_channels: int) -> int:
        return min(self.min_visible_channels, n_channels)

    def _satisfies_guards(self, visible: torch.Tensor) -> bool:
        if not bool(visible.any()):
            return False
        n_with_content = int(visible.any(dim=1).sum().item())
        return n_with_content >= self._min_content_channels(visible.shape[0])

    def _force_visible_patch(self, visible: torch.Tensor, generator) -> torch.Tensor:
        """Last-resort guard: reveal one patch in every empty channel.

        Never reveals a whole channel row, so a channel dropped by the channel
        mask cannot leak into the encoder as a fully visible channel.

        This is unreachable for every valid configuration: patch mode leaves
        ``N - floor(N * patch_ratio) >= 1`` visible patch in each row, and the
        channel/combined regions draw ``m >= min_visible_channels`` visible
        channels from :class:`HierarchicalChannelSampler` by construction.  It
        exists only to keep the guard invariant explicit if a future sampling
        scheme can violate it.
        """
        missing = torch.nonzero(~visible.any(dim=1), as_tuple=False).squeeze(1)
        if missing.numel() == 0:
            # Cannot happen for C >= 1, but keep the invariant explicit.
            visible[0, self._randint(self.n_patches, visible.device, generator)] = True
            return visible
        for c in missing.tolist():
            pi = self._randint(self.n_patches, visible.device, generator)
            visible[c, pi] = True
        return visible

    # ------------------------------------------------------------------ #
    def __call__(
        self,
        images: Sequence[torch.Tensor],
        generator: Optional[torch.Generator] = None,
    ) -> List[torch.Tensor]:
        """Return a list of ``[C_b, n_patches]`` boolean *visible* masks."""
        self.last_modes = []
        if len(images) == 0:
            return []

        device = images[0].device
        modes = self._select_modes(len(images), device, generator)
        out: List[torch.Tensor] = []

        for im, mode in zip(images, modes):
            C = int(im.shape[0])
            applied = mode
            if applied == "channel" and C <= self.min_visible_channels:
                # Channel-only masking would be a no-op (HCS keeps >= 1
                # channel visible and C <= min_visible); use patch masking.
                applied = "patch"

            visible = self._draw_visible(C, device, generator, applied)
            for _ in range(self._MAX_ATTEMPTS):
                if self._satisfies_guards(visible):
                    break
                visible = self._draw_visible(C, device, generator, applied)
            if not self._satisfies_guards(visible):
                visible = self._force_visible_patch(visible, generator)

            self.last_modes.append(applied)
            out.append(visible)

        return out
