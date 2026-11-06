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
from collections.abc import Sequence

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
        device: torch.device | None = None,
        generator: torch.Generator | None = None,
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
            raise ValueError(f"min_visible must be in [1, {n_channels}], got {min_visible}")
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
        n_patches: int | None = None,
        p_patch: float = 0.5,
        p_channel: float = 0.5,
        patch_ratio: float = 0.75,
        min_visible_channels: int = 1,
        max_masked_channels_frac: float = 1.0,
        mode_sampling: str = "batch",
    ) -> None:
        if n_patches is not None and n_patches < 1:
            raise ValueError(f"n_patches must be >= 1, got {n_patches}")
        if p_patch < 0 or p_channel < 0:
            raise ValueError(
                f"p_patch and p_channel must be non-negative, got {p_patch} and {p_channel}"
            )
        if p_patch + p_channel > 1:
            raise ValueError(f"p_patch + p_channel must be <= 1, got {p_patch + p_channel}")
        if not 0.0 < patch_ratio < 1.0:
            raise ValueError(f"patch_ratio must be in (0, 1), got {patch_ratio}")
        patch_region_used = p_patch > 0 or p_patch + p_channel < 1
        if n_patches is not None and patch_region_used and math.floor(n_patches * patch_ratio) < 1:
            raise ValueError(
                "n_patches * patch_ratio must be >= 1 when the patch region "
                "is used; got "
                f"n_patches={n_patches}, patch_ratio={patch_ratio}"
            )
        if min_visible_channels < 1:
            raise ValueError(f"min_visible_channels must be >= 1, got {min_visible_channels}")
        if not 0.0 <= max_masked_channels_frac <= 1.0:
            raise ValueError(
                f"max_masked_channels_frac must be in [0, 1], got {max_masked_channels_frac}"
            )
        if mode_sampling not in ("batch", "sample"):
            raise ValueError(f"mode_sampling must be 'batch' or 'sample', got {mode_sampling!r}")

        self.n_patches = n_patches
        self.p_patch = p_patch
        self.p_channel = p_channel
        self.patch_ratio = patch_ratio
        self.min_visible_channels = min_visible_channels
        self.max_masked_channels_frac = max_masked_channels_frac
        self.mode_sampling = mode_sampling
        self.channel_sampler = HierarchicalChannelSampler()
        self.last_modes: list[str] = []

    # ------------------------------------------------------------------ #
    def _uniform(self, device, generator) -> float:
        return float(torch.rand((), device=device, generator=generator, dtype=torch.float32).item())

    def _randint(self, high: int, device, generator) -> int:
        return int(torch.randint(0, high, (1,), device=device, generator=generator).item())

    def _mode_from_s(self, s: float) -> str:
        if s < self.p_patch:
            return "patch"
        if s < self.p_patch + self.p_channel:
            return "channel"
        return "combined"

    def _select_modes(self, n_samples: int, device, generator) -> list[str]:
        if self.mode_sampling == "batch":
            s = self._uniform(device, generator)
            return [self._mode_from_s(s)] * n_samples
        return [self._mode_from_s(self._uniform(device, generator)) for _ in range(n_samples)]

    # ------------------------------------------------------------------ #
    def _draw_visible(
        self, n_channels: int, n_patches: int, device, generator, mode: str
    ) -> torch.Tensor:
        """Draw a ``[C, N]`` boolean visible mask for one sample."""
        visible = torch.ones(n_channels, n_patches, dtype=torch.bool, device=device)

        if mode in ("patch", "combined"):
            n_masked = math.floor(n_patches * self.patch_ratio)
            for c in range(n_channels):
                if n_masked:
                    idx = torch.randperm(n_patches, device=device, generator=generator)[:n_masked]
                    visible[c, idx] = False

        if mode in ("channel", "combined"):
            channel_visible = self.channel_sampler(
                n_channels,
                device=device,
                generator=generator,
                min_visible=self._min_content_channels(n_channels),
            )
            dropped = int((~channel_visible).sum().item())
            cap = math.floor(n_channels * self.max_masked_channels_frac)
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

    def _force_visible_patch(
        self, visible: torch.Tensor, n_patches: int, generator
    ) -> torch.Tensor:
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
            visible[0, self._randint(n_patches, visible.device, generator)] = True
            return visible
        for c in missing.tolist():
            pi = self._randint(n_patches, visible.device, generator)
            visible[c, pi] = True
        return visible

    # ------------------------------------------------------------------ #
    def _resolve_n_patches(self, n_samples: int, n_patches) -> list[int]:
        if n_patches is None:
            if self.n_patches is None:
                raise ValueError(
                    "DCPVisibilitySampler needs n_patches at construction or call time"
                )
            return [self.n_patches] * n_samples
        if isinstance(n_patches, int):
            return [n_patches] * n_samples
        ns = [int(n) for n in n_patches]
        if len(ns) != n_samples:
            raise ValueError(f"expected {n_samples} n_patches values, got {len(ns)}")
        return ns

    def __call__(
        self,
        images: Sequence[torch.Tensor],
        generator: torch.Generator | None = None,
        n_patches=None,
    ) -> list[torch.Tensor]:
        """Return a list of ``[C_b, N_b]`` boolean *visible* masks.

        ``n_patches`` overrides the constructor value and may be a scalar or
        one value per sample, which is what FlexiViT needs when the patch size
        changes between steps or samples.
        """
        self.last_modes = []
        if len(images) == 0:
            return []

        device = images[0].device
        ns = self._resolve_n_patches(len(images), n_patches)
        modes = self._select_modes(len(images), device, generator)
        out: list[torch.Tensor] = []

        for im, mode, N in zip(images, modes, ns, strict=True):
            if N < 1:
                raise ValueError(f"n_patches must be >= 1, got {N}")
            patch_region_used = self.p_patch > 0 or self.p_patch + self.p_channel < 1
            patch_ok = not patch_region_used or math.floor(N * self.patch_ratio) >= 1
            C = int(im.shape[0])
            applied = mode
            if not patch_ok and applied in ("patch", "combined"):
                # A 1-patch grid cannot use the patch region.  Fall back to
                # whole-channel masking, or to an unmasked sample when even
                # channel masking would be a no-op.
                applied = "channel" if self.min_visible_channels < C else "none"
            if applied == "channel" and self.min_visible_channels >= C:
                # Channel-only masking would be a no-op (HCS keeps >= 1
                # channel visible and C <= min_visible); use patch masking.
                applied = "patch" if patch_ok else "none"

            visible = self._draw_visible(C, N, device, generator, applied)
            for _ in range(self._MAX_ATTEMPTS):
                if self._satisfies_guards(visible):
                    break
                visible = self._draw_visible(C, N, device, generator, applied)
            if not self._satisfies_guards(visible):
                visible = self._force_visible_patch(visible, N, generator)

            self.last_modes.append(applied)
            out.append(visible)

        return out


class MultiBlockTargetSampler:
    """I-JEPA/V-JEPA-style multi-block target and context locations.

    A JEPA needs **two** masks with different distributions, and they answer
    different questions:

    ``target``
        what the predictor must infer.  A union of ``n_blocks`` rectangles with
        area fraction from ``target_scale`` and aspect ratio from
        ``target_aspect``, following I-JEPA's finding that a few large semantic
        blocks beat random patch masking (Table 6: 54.2 vs 17.6 top-1 at 1%
        labels).  Removed for *all* channels, so no partially visible token is
        left at a target position.

    ``context``
        what the student encoder may see.  One rectangle with area fraction from
        ``context_scale`` (unit aspect) minus every target location.  This is
        I-JEPA's context block (0.85-1.0 there); its size is a real axis —
        I-JEPA Table 9: 31.2 at (0.40, 1.0), 54.2 at (0.85, 1.0).

    Unlike :class:`DCPVisibilitySampler`, both masks are over **locations**, not
    per-channel channel-patches: after channel fusion there is one feature per
    location, so a per-channel mask would define no target set and would leave
    the fused token visible.

    ``context_scale=(1.0, 1.0)`` (the default) makes the context the exact
    complement of the targets, which is what a target-only masker produces and
    what I-JEPA's best context row reports.  Any lower value additionally drops
    a randomly placed border strip that is neither target nor context, which
    shrinks the student's token count (encoder cost) and restricts its field of
    view.  Channel selection is deliberately *not* part of this sampler:
    :class:`~fuse_channels.jepa.ChannelFusionJEPA` applies optional whole-channel
    dropout to the context with :class:`HierarchicalChannelSampler`.

    Both masks are guarded non-empty: an empty target reveals one location, and
    an empty context reveals one non-target location (or, on a grid fully
    covered by targets, releases one target).
    """

    def __init__(
        self,
        n_blocks: int = 4,
        target_scale: tuple[float, float] = (0.15, 0.2),
        target_aspect: tuple[float, float] = (0.75, 1.5),
        context_scale: tuple[float, float] = (1.0, 1.0),
    ) -> None:
        if n_blocks < 1:
            raise ValueError(f"n_blocks must be >= 1, got {n_blocks}")
        for name, (lo, hi) in (
            ("target_scale", target_scale),
            ("context_scale", context_scale),
        ):
            if not 0.0 < lo <= hi <= 1.0:
                raise ValueError(f"{name} must satisfy 0 < lo <= hi <= 1, got {(lo, hi)}")
        alo, ahi = target_aspect
        if not 0.0 < alo <= ahi:
            raise ValueError(f"target_aspect must satisfy 0 < lo <= hi, got {target_aspect}")
        self.n_blocks = int(n_blocks)
        self.target_scale = (float(target_scale[0]), float(target_scale[1]))
        self.target_aspect = (float(target_aspect[0]), float(target_aspect[1]))
        self.context_scale = (float(context_scale[0]), float(context_scale[1]))

    def _uniform(self, lo: float, hi: float, device, generator) -> float:
        u = float(torch.rand((), device=device, generator=generator).item())
        return lo + (hi - lo) * u

    def _rect(self, gh: int, gw: int, scale: float, aspect: float, device, generator):
        """Return integer ``(r0, r1, c0, c1)`` for one rectangle."""
        h = max(1, min(gh, round(math.sqrt(scale / aspect) * gh)))
        w = max(1, min(gw, round(math.sqrt(scale * aspect) * gw)))
        r0 = int(torch.randint(0, gh - h + 1, (1,), device=device, generator=generator).item())
        c0 = int(torch.randint(0, gw - w + 1, (1,), device=device, generator=generator).item())
        return r0, r0 + h, c0, c0 + w

    @property
    def _whole_grid_context(self) -> bool:
        """True when the context is the target complement (no rectangle drawn)."""
        return self.context_scale == (1.0, 1.0)

    def _pick(self, free: torch.Tensor, device, generator) -> tuple[int, int]:
        index = int(torch.randint(free.shape[0], (1,), device=device, generator=generator).item())
        row, col = free[index].tolist()
        return int(row), int(col)

    def sample_targets(self, grid_hw: Sequence[tuple[int, int]], generator=None):
        """Return one ``[N_b]`` boolean target-location mask per sample."""
        device = generator.device if generator is not None else torch.device("cpu")
        out: list[torch.Tensor] = []
        for gh, gw in grid_hw:
            gh, gw = int(gh), int(gw)
            if gh * gw < 1:
                raise ValueError(f"grid {(gh, gw)} leaves no locations")
            target = torch.zeros(gh, gw, dtype=torch.bool, device=device)
            for _ in range(self.n_blocks):
                scale = self._uniform(*self.target_scale, device, generator)
                aspect = self._uniform(*self.target_aspect, device, generator)
                r0, r1, c0, c1 = self._rect(gh, gw, scale, aspect, device, generator)
                target[r0:r1, c0:c1] = True
            if not bool(target.any()):
                target.view(-1)[0] = True
            out.append(target.reshape(-1))
        return out

    def context_masks(self, target, grid_hw: Sequence[tuple[int, int]], generator=None):
        """Return ``(context, target)`` location masks for one sample-group.

        ``target`` is one ``[N_b]`` mask per sample, either from
        :meth:`sample_targets` or supplied by the caller; it is returned
        alongside the context because a grid fully covered by targets forces one
        target location to be released (there would be no context at all).  The
        context is disjoint from every target and non-empty.  At
        ``context_scale=(1.0, 1.0)`` this draws no random numbers, so changing
        any other mask knob leaves the sampled targets bit-identical; below it,
        one context rectangle is drawn per sample.
        """
        device = generator.device if generator is not None else torch.device("cpu")
        out: list[torch.Tensor] = []
        released: list[torch.Tensor] = []
        for t, (gh, gw) in zip(target, grid_hw, strict=True):
            gh, gw = int(gh), int(gw)
            t = t.reshape(gh, gw).to(device=device, dtype=torch.bool)
            if self._whole_grid_context:
                context = ~t
            else:
                scale = self._uniform(*self.context_scale, device, generator)
                r0, r1, c0, c1 = self._rect(gh, gw, scale, 1.0, device, generator)
                context = torch.zeros(gh, gw, dtype=torch.bool, device=device)
                context[r0:r1, c0:c1] = True
                context &= ~t
            if not bool(context.any()):
                free = (~t).nonzero(as_tuple=False)
                if free.numel() == 0:
                    # Grid fully covered by targets: release one target location.
                    t = t.clone()
                    t.view(-1)[-1] = False
                    free = (~t).nonzero(as_tuple=False)
                context[self._pick(free, device, generator)] = True
            out.append(context.reshape(-1))
            released.append(t.reshape(-1))
        return out, released

    def __call__(self, grid_hw: Sequence[tuple[int, int]], generator=None):
        """Return ``(target, context)`` lists of ``[N_b]`` boolean location masks.

        ``grid_hw`` is one ``(grid_h, grid_w)`` pair per sample.  ``target``
        marks locations to predict; ``context`` marks locations the student
        encoder may see.  The two are disjoint by construction.
        """
        target = self.sample_targets(grid_hw, generator=generator)
        context, target = self.context_masks(target, grid_hw, generator=generator)
        return target, context
