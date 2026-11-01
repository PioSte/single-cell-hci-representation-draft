"""Structured configuration for ``ChannelFusionMAE`` and its ablations.

This module realises the "shared core, exclusive subconfigs" design
(option *b*): a parameter shared by several modules is **not** copied into each
module's config, and it is **not** only a flat keyword on the model.  Instead it
lives once in :class:`SharedCore`, which every consuming module receives
explicitly::

    TransformerEncoder(core: SharedCore, cfg: EncoderConfig)
    ChannelAwareDecoder(core: SharedCore, cfg: DecoderConfig, *, <derived>)

The three categories of parameter, and why they are separated, are:

* **core** (:class:`SharedCore`) -- invariants the encoder and decoder *must*
  agree on: ``dim``, ``n_heads``, ``dropout`` and the ``adaln`` wiring flag.
  They are properties of the whole backbone, not of any one module, so a
  mismatch is a bug, not a knob.  ``adaln`` is derived from
  ``scale_conditioning`` and is therefore not part of the flat API.
* **cfg** (:class:`EncoderConfig`, :class:`DecoderConfig`) -- knobs exclusive to
  one module (depth, ``n_memory``, the decoder PE/head choice, ...).  These are
  what you tweak to ablate that module without touching the other.
* **builder-resolved extras** -- facts only the composition root knows: the
  decoder's ``n_patches`` (from the tokenizer), the resolved
  ``canonical_patch_size`` (from ``head_patch_size or max(patch_sizes)``) and
  the fleet ``patch_sizes``.  They are passed to the decoder constructor as
  keyword extras, not stored in its config, because they are derived, not
  authored.

:class:`ModelConfig` is the authoring layer: it groups the model-level knobs,
holds one :class:`SharedCore`, one :class:`EncoderConfig` and one
:class:`DecoderConfig`, and flattens to the ``ChannelFusionMAE`` constructor via
:meth:`ModelConfig.to_kwargs`.  :func:`build_model` adds an ablation registry:
each ablation is a small transform on a :class:`ModelConfig`, so it is a
*reviewable diff* naming exactly the subconfig it changes, and it tracks
default changes instead of baking values.

Ablation is a first-class concern here, so the registry is the recommended way
to spell one.  Examples::

    build_model("no_memory")        # cfg.encoder.n_memory = 0
    build_model("blind_decoder")    # no memory and no KV -> self-attention only
    build_model("no_scale_cond")    # scale_conditioning = "none"

Prefer :func:`build_model` / :meth:`ModelConfig.to_kwargs` over hand-writing a
flat constructor call when you want to record *how* a run differs from the
default.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable, Optional, Sequence


@dataclass(frozen=True)
class SharedCore:
    """Backbone invariants that the encoder and decoder must agree on.

    Passed to every composite module explicitly.  ``adaln`` is derived from
    ``scale_conditioning`` by the model builder; do not set it independently of
    the scale-conditioning mode.
    """

    dim: int = 192
    n_heads: int = 6
    dropout: float = 0.0
    adaln: bool = False

    def __post_init__(self) -> None:
        if self.dim % self.n_heads != 0:
            raise ValueError(
                f"dim ({self.dim}) must be divisible by n_heads ({self.n_heads})"
            )


@dataclass(frozen=True)
class EncoderConfig:
    """Encoder-exclusive knobs (width/regularization live in :class:`SharedCore`)."""

    depth: int = 4
    n_memory: int = 4
    mlp_ratio: float = 4.0


@dataclass(frozen=True)
class DecoderConfig:
    """Decoder-exclusive knobs.

    ``head_patch_size`` is the canonical ``p0`` requested for the
    reconstruction head; the builder resolves it (``head_patch_size or
    max(patch_sizes)``) into the ``canonical_patch_size`` it passes as a
    derived extra.  ``mlp_ratio`` is not part of the flat model API (it is baked
    at 4.0); it is here for callers that build a decoder directly.
    """

    depth: int = 2
    mlp_ratio: float = 4.0
    pos_embed: str = "fractional"
    fractional_pe_mode: str = "sincos"
    fractional_pe_base: float = 10000.0
    pe_ref_um: float = 1.0
    head_mode: str = "per_p"
    head_resample: str = "area"
    head_patch_size: Optional[int] = None


@dataclass(frozen=True)
class ModelConfig:
    """Authoring configuration for :class:`ChannelFusionMAE`.

    Field grouping mirrors the parameter-scope tree: the shared backbone lives
    in :attr:`core`, per-module knobs in :attr:`encoder` / :attr:`decoder`, and
    the data/frame/loss facts that the whole model shares sit at the top.
    :meth:`to_kwargs` flattens exactly to the ``ChannelFusionMAE`` constructor.
    """

    image_size: int = 16
    core: SharedCore = field(default_factory=SharedCore)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)

    # --- geometry / patch-pixel frame (shared by tokenizer + encoder + decoder)
    patch_size: int | Sequence[int] | None = 8
    patch_sizes: Optional[Sequence[int]] = None
    flexi_random: bool = True
    max_tokens: Optional[int] = None
    pixel_size: Optional[float] = 1.0
    pe_space: str = "physical"

    # --- positional-embedding variants (decoder's lives in DecoderConfig) -----
    pos_embed: Optional[str] = None
    encoder_pos_embed: Optional[str] = None
    fusion_pos_embed: Optional[str] = None
    rope_base: float = 100.0
    rope_ref_extent: float = 32.0
    rope_shared_extent: bool = False

    # --- fusion ---------------------------------------------------------------
    fusion: str = "attention"
    fusion_attn_dim: Optional[int] = None

    # --- pooling / decoder-KV wiring -----------------------------------------
    n_latents: int = 16
    n_queries: int = 1
    decoder_kv: tuple[str, ...] = ("memory", "latents", "descriptor")

    # --- masking (DCP / HCS) --------------------------------------------------
    p_patch: float = 0.5
    p_channel: float = 0.5
    patch_ratio: float = 0.75
    mode_sampling: str = "batch"
    min_visible_channels: int = 1
    max_masked_channels_frac: float = 1.0
    keep_offcell_as_context: bool = True

    # --- scale conditioning ---------------------------------------------------
    scale_conditioning: str = "add"
    scale_fourier_frequencies: int = 8

    # --- identity / loss ------------------------------------------------------
    num_channel_ids: int = 16
    loss: str = "mae"
    loss_reduce: str = "sample"

    def to_kwargs(self) -> dict:
        """Flatten to ``ChannelFusionMAE.__init__`` keyword names.

        ``mlp_ratio`` on the encoder/decoder is intentionally not emitted: the
        flat constructor bakes it at 4.0.  Keep this in sync with the
        constructor; ``tests/test_config.py`` asserts the key sets match.
        """
        return dict(
            image_size=self.image_size,
            patch_size=self.patch_size,
            patch_sizes=self.patch_sizes,
            flexi_random=self.flexi_random,
            max_tokens=self.max_tokens,
            pixel_size=self.pixel_size,
            pe_space=self.pe_space,
            # core (adaln is derived from scale_conditioning, not a flat kwarg)
            dim=self.core.dim,
            n_heads=self.core.n_heads,
            dropout=self.core.dropout,
            # encoder
            encoder_depth=self.encoder.depth,
            n_memory=self.encoder.n_memory,
            # decoder
            decoder_depth=self.decoder.depth,
            decoder_pos_embed=self.decoder.pos_embed,
            fractional_pe_mode=self.decoder.fractional_pe_mode,
            fractional_pe_base=self.decoder.fractional_pe_base,
            pe_ref_um=self.decoder.pe_ref_um,
            head_mode=self.decoder.head_mode,
            head_resample=self.decoder.head_resample,
            head_patch_size=self.decoder.head_patch_size,
            # top-level
            pos_embed=self.pos_embed,
            encoder_pos_embed=self.encoder_pos_embed,
            fusion_pos_embed=self.fusion_pos_embed,
            rope_base=self.rope_base,
            rope_ref_extent=self.rope_ref_extent,
            rope_shared_extent=self.rope_shared_extent,
            fusion=self.fusion,
            fusion_attn_dim=self.fusion_attn_dim,
            n_latents=self.n_latents,
            n_queries=self.n_queries,
            decoder_kv=self.decoder_kv,
            p_patch=self.p_patch,
            p_channel=self.p_channel,
            patch_ratio=self.patch_ratio,
            mode_sampling=self.mode_sampling,
            min_visible_channels=self.min_visible_channels,
            max_masked_channels_frac=self.max_masked_channels_frac,
            keep_offcell_as_context=self.keep_offcell_as_context,
            scale_conditioning=self.scale_conditioning,
            scale_fourier_frequencies=self.scale_fourier_frequencies,
            num_channel_ids=self.num_channel_ids,
            loss=self.loss,
            loss_reduce=self.loss_reduce,
        )


# --------------------------------------------------------------------------- #
# Ablation registry: each entry is a transform on a ModelConfig, so it reads
# as a diff naming the exact subconfig it changes and stays valid as defaults
# drift.  Add new ablations here rather than editing ``forward`` by hand.
# --------------------------------------------------------------------------- #
def _ablate_no_memory(cfg: ModelConfig) -> ModelConfig:
    """Drop the memory tokens from the encoder (they leave the KV too)."""
    return replace(cfg, encoder=replace(cfg.encoder, n_memory=0))


def _ablate_blind_decoder(cfg: ModelConfig) -> ModelConfig:
    """No memory and an empty decoder KV -> decoder is self-attention only."""
    return replace(_ablate_no_memory(cfg), decoder_kv=())


def _ablate_no_scale_cond(cfg: ModelConfig) -> ModelConfig:
    return replace(cfg, scale_conditioning="none")


def _ablate_grid_pe(cfg: ModelConfig) -> ModelConfig:
    return replace(cfg, pe_space="grid")


#: Named ablations: a name maps to the ``ModelConfig -> ModelConfig`` transform
#: that produces it from the current default.  ``"full"`` is the shipped model.
ABLATIONS: dict[str, Callable[[ModelConfig], ModelConfig]] = {
    "full": lambda cfg: cfg,
    "no_memory": _ablate_no_memory,
    "blind_decoder": _ablate_blind_decoder,
    "no_scale_cond": _ablate_no_scale_cond,
    "grid_pe": _ablate_grid_pe,
}


def ablated_config(
    name: str = "full", base: ModelConfig | None = None
) -> ModelConfig:
    """Return the :class:`ModelConfig` for a named ablation."""
    if name not in ABLATIONS:
        raise KeyError(
            f"unknown ablation '{name}'; expected one of {sorted(ABLATIONS)}"
        )
    return ABLATIONS[name](base or ModelConfig())


def build_model(name: str = "full", base: ModelConfig | None = None, **overrides):
    """Build a :class:`ChannelFusionMAE` for a named ablation.

    ``overrides`` are flat ``ChannelFusionMAE.__init__`` keywords applied after
    the ablation, so an experiment can still tweak one knob on top of a named
    arm.
    """
    from .mae import ChannelFusionMAE

    kwargs = ablated_config(name, base).to_kwargs()
    kwargs.update(overrides)
    return ChannelFusionMAE(**kwargs)
