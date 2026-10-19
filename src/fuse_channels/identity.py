"""Channel-identity embeddings.

Channel identity is non-negotiable for this architecture: removing the additive
identity embedding destroys performance in the ChannelViT / ChAda-ViT ablations.
Two paths are supported:

* **learnable** per-marker embeddings for a closed/fixed panel;
* **text/sequence conditioned** embeddings (PhenoPLIFT-style text prompts or
  ESM-2 protein embeddings) for open / multiplexed panels.  These never need a
  fixed lookup table, so ``C`` can grow without bound and there is no pad slot.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class ChannelIdentity(nn.Module):
    """Map marker ids (and/or external text/sequence vectors) to ``[N, dim]``."""

    def __init__(
        self,
        dim: int,
        num_markers: int = 0,
        text_dim: Optional[int] = None,
        use_learnable: bool = True,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.use_learnable = use_learnable and num_markers > 0
        self.embed: Optional[nn.Embedding] = None
        if self.use_learnable:
            self.embed = nn.Embedding(num_markers, dim)
            nn.init.trunc_normal_(self.embed.weight, std=0.02)

        self.text_proj: Optional[nn.Module] = None
        if text_dim is not None:
            self.text_proj = nn.Sequential(
                nn.Linear(text_dim, dim),
                nn.GELU(),
                nn.Linear(dim, dim),
            )

    def forward(
        self,
        channel_ids: torch.Tensor,
        text_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return ``[N, dim]`` identity vectors.

        Parameters
        ----------
        channel_ids:
            ``[N]`` marker ids.  Ignored when ``text_embeds`` is given.
        text_embeds:
            ``[N, text_dim]`` external identity vectors (prompt / protein).
            When provided, the text/sequence projection path is used and the
            learnable table is *not* resized, so novel markers work zero-shot.
        """
        if text_embeds is not None:
            if self.text_proj is None:
                raise ValueError(
                    "text_embeds were passed but this ChannelIdentity was built "
                    "without `text_dim`."
                )
            return self.text_proj(text_embeds)
        if self.embed is None:
            raise ValueError(
                "No learnable embedding table and no text_embeds were given; "
                "cannot produce channel identities."
            )
        return self.embed(channel_ids)
