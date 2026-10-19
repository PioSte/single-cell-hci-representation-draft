"""Dynamic Channel-Patch (DCP) masking.

Combines the two masking axes required for multi-channel-imaging MAE:

1. **Spatial patch masking** sampled *independently per channel* (not a single
   mask replicated across channels).
2. **Dynamic channel masking** -- sample how many channels to mask, then which
   ones (Hierarchical Channel Sampling repurposed from a regularizer into a
   supervision target).

Per sample, one of the two modes is chosen (``p_patch`` / ``p_channel``).  The
returned boolean mask marks *visible* tokens; masked tokens are omitted from the
padding-free tensor, never zeroed.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch


class DCPMasker:
    def __init__(
        self,
        n_patches: int,
        p_patch: float = 0.5,
        p_channel: float = 0.5,
        min_visible_channels: int = 1,
        max_masked_channels_frac: float = 1.0,
    ) -> None:
        self.n_patches = n_patches
        self.p_patch = p_patch
        self.p_channel = p_channel
        self.min_visible_channels = min_visible_channels
        self.max_masked_channels_frac = max_masked_channels_frac

    # ------------------------------------------------------------------ #
    def _rand(self, size, device, generator, dtype=torch.float32):
        return torch.rand(size, device=device, generator=generator, dtype=dtype)

    def _randint(self, high, device, generator):
        if high <= 1:
            return 0
        return int(torch.randint(0, high, (1,), device=device, generator=generator).item())

    def __call__(
        self,
        images: Sequence[torch.Tensor],
        generator: Optional[torch.Generator] = None,
    ) -> List[torch.Tensor]:
        """Return a list of ``[C_b, n_patches]`` boolean *visible* masks."""
        out: List[torch.Tensor] = []
        for im in images:
            C = im.shape[0]
            N = self.n_patches
            device = im.device

            channel_mode = (
                self._rand((1,), device, generator).item() < self.p_channel
            )

            if channel_mode:
                upper = max(1, int(round(C * self.max_masked_channels_frac)))
                # k in [1, C - min_visible_channels], sample k, then which k.
                k_hi = max(1, C - self.min_visible_channels)
                k = self._randint(k_hi + 1, device, generator)  # 0..k_hi
                k = min(k, upper)
                channel_visible = torch.ones(C, dtype=torch.bool, device=device)
                if k > 0:
                    perm = torch.randperm(C, device=device, generator=generator)
                    channel_visible[perm[:k]] = False
                visible = channel_visible[:, None].expand(C, N).clone()
            else:
                r = self._rand((C, N), device, generator)
                visible = r >= self.p_patch

            # -- safety fix-ups: never mask everything -------------------- #
            if not bool(visible.any()):
                ci = self._randint(C, device, generator)
                pi = self._randint(N, device, generator)
                visible[ci, pi] = True
            # keep at least `min_visible_channels` channels with some content
            has_content = visible.any(dim=1)
            n_with_content = int(has_content.sum().item())
            if n_with_content < self.min_visible_channels:
                need = self.min_visible_channels - n_with_content
                candidates = torch.nonzero(~has_content, as_tuple=False).squeeze(1)
                for c in candidates[:need].tolist():
                    visible[c] = True
            out.append(visible)
        return out


def visible_to_masked(
    images: Sequence[torch.Tensor], visible: Sequence[torch.Tensor]
) -> List[torch.Tensor]:
    """Return the per-sample masked (``~visible``) boolean masks."""
    return [~v for v in visible]
