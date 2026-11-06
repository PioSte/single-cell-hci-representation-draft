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
:meth:`ModelConfig.to_kwargs`.  :meth:`ModelConfig.resolve` is the one place
the canonical patch size, the ``pos_embed`` fan-out and ``core.adaln`` are
derived, and :meth:`ModelConfig.from_kwargs` is the inverse of ``to_kwargs``.
The flat constructor sources its defaults from :data:`DEFAULT_CONFIG` and
routes through both, so the flat and structured entry points share one default
set and one resolution path.  :func:`build_model` adds an ablation registry:
each ablation is a small transform on a :class:`ModelConfig`, so it is a
*reviewable diff* naming exactly the config field(s) it changes, and it tracks
default changes instead of baking values.

Ablation is a first-class concern here, so the registry is the recommended way
to spell one.  Each transform documents its diff and the question it isolates,
next to the code that defines it; :func:`describe_ablations` renders that
documentation as the Markdown table in ``README.md`` (a test keeps the two in
sync), so the registry is the only hand-maintained inventory::

    build_model("blind_decoder")    # no memory and no KV -> self-attention only

Prefer :func:`build_model` / :meth:`ModelConfig.to_kwargs` over hand-writing a
flat constructor call.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

#: ``pos_embed`` kinds that select the legacy fixed-grid table scheme.  The
#: fan-out to ``encoder_pos_embed`` / ``decoder_pos_embed`` /
#: ``fusion_pos_embed`` lives in :meth:`ModelConfig.resolve`.
LEGACY_POS_EMBEDS = ("learned", "sincos")

#: Shipped canonical patch size, used only when the caller supplies neither
#: ``patch_size`` nor ``patch_sizes``.  Keeping the config fields at ``None``
#: lets an explicit ``patch_sizes=`` still choose its first entry as ``p0``.
DEFAULT_PATCH_SIZE: int = 8


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
            raise ValueError(f"dim ({self.dim}) must be divisible by n_heads ({self.n_heads})")


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

    ``pos_embed=None`` means "inherit from ``ModelConfig.pos_embed``";
    :meth:`ModelConfig.resolve` fills it in.  A decoder built standalone uses
    the shipped ``"fractional"`` because it has no parent to inherit from.
    """

    depth: int = 2
    mlp_ratio: float = 4.0
    pos_embed: str | None = None
    fractional_pe_mode: str = "sincos"
    fractional_pe_base: float = 10000.0
    pe_ref_um: float = 1.0
    head_mode: str = "per_p"
    head_resample: str = "area"
    head_patch_size: int | None = None


@dataclass(frozen=True)
class JEPAConfig:
    """Objective-specific knobs for :class:`~fuse_channels.jepa.ChannelFusionJEPA`.

    These deliberately do not live in :class:`ModelConfig`: they parameterise
    the JEPA predictor/target pair and its multi-block mask, not the shared
    channel-fusion backbone.  The backbone is still authored by
    :class:`ModelConfig`; :func:`fuse_channels.jepa.build_jepa` combines the
    two.  ``target_scale`` is a *fraction of the grid*, not a physical extent,
    so it is independent of the realized patch size and pixel size.

    ``use_ema=False`` removes the target-encoder copy and requires a non-zero
    ``reg_weight`` for ``sigreg`` or ``vicreg``; ``reg_kwargs`` is forwarded to
    the regularizer constructor.  With ``use_ema=True`` the regularizer is an
    optional auxiliary term on the pooled latents.
    """

    loss: str = "l1"
    use_ema: bool = True
    ema_momentum: float = 0.996
    regularizer: str = "none"
    reg_weight: float = 0.0
    reg_kwargs: dict[str, Any] = field(default_factory=dict)
    target_blocks: int = 4
    target_scale: tuple[float, float] = (0.15, 0.2)
    target_aspect: tuple[float, float] = (0.75, 1.5)
    context_scale: tuple[float, float] = (0.85, 1.0)
    channel_dropout: bool = False
    min_visible_channels: int = 1

    def __post_init__(self) -> None:
        if self.loss not in ("l1", "l2"):
            raise ValueError(f"unknown JEPA loss '{self.loss}'; expected 'l1' or 'l2'")
        if not 0.0 <= self.ema_momentum < 1.0:
            raise ValueError(f"ema_momentum must be in [0, 1), got {self.ema_momentum}")
        if self.regularizer not in ("none", "sigreg", "vicreg"):
            raise ValueError(
                f"unknown regularizer '{self.regularizer}'; expected 'none', 'sigreg' or 'vicreg'"
            )
        if not math.isfinite(self.reg_weight) or self.reg_weight < 0:
            raise ValueError(f"reg_weight must be finite and non-negative, got {self.reg_weight}")
        if not self.use_ema and (self.regularizer == "none" or self.reg_weight <= 0):
            raise ValueError(
                "use_ema=False needs a non-zero regularizer weight "
                "('sigreg' or 'vicreg'); EMA is otherwise the only "
                "collapse-prevention mechanism"
            )
        if self.target_blocks < 1:
            raise ValueError(f"target_blocks must be >= 1, got {self.target_blocks}")
        for name, (lo, hi) in (
            ("target_scale", self.target_scale),
            ("context_scale", self.context_scale),
        ):
            if not 0.0 < lo <= hi <= 1.0:
                raise ValueError(f"{name} must satisfy 0 < lo <= hi <= 1, got {(lo, hi)}")
        alo, ahi = self.target_aspect
        if not 0.0 < alo <= ahi:
            raise ValueError(f"target_aspect must satisfy 0 < lo <= hi, got {self.target_aspect}")
        if self.min_visible_channels < 1:
            raise ValueError(f"min_visible_channels must be >= 1, got {self.min_visible_channels}")


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
    # ``patch_size=None`` means "derive from patch_sizes[0]"; ``pixel_size=None"
    # means "no model-level default, pass pixel_sizes= per call".  Both "auto"
    # meanings are resolved by :meth:`resolve`, not duplicated in the flat API.
    patch_size: int | Sequence[int] | None = None
    patch_sizes: Sequence[int] | None = None
    flexi_random: bool = True
    max_tokens: int | None = None
    pixel_size: float | None = None
    pe_space: str = "physical"

    # --- positional-embedding variants (decoder's lives in DecoderConfig) -----
    pos_embed: str | None = None
    encoder_pos_embed: str | None = None
    fusion_pos_embed: str | None = None
    rope_base: float = 100.0
    rope_ref_extent: float = 32.0
    rope_shared_extent: bool = False

    # --- fusion ---------------------------------------------------------------
    fusion: str = "attention"
    fusion_attn_dim: int | None = None

    # --- pooling / decoder-KV wiring -----------------------------------------
    n_latents: int = 16
    decoder_kv_components: tuple[str, ...] = ("memory", "latents")

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

    def resolve(self) -> ModelConfig:
        """Return a copy with every derived field made explicit.

        This is the single implementation of the derivation the flat
        ``ChannelFusionMAE`` constructor used to carry:

        * the canonical ``patch_size`` (``patch_sizes[0]`` when unset) and the
          sorted union ``patch_sizes`` that always contains it;
        * the ``pos_embed -> encoder/decoder/fusion`` fan-out, including the
          legacy fixed-grid scheme;
        * ``core.adaln`` from ``scale_conditioning``.

        ``ModelConfig().resolve()`` is the shipped recipe, and is idempotent.
        """
        patch_size = self.patch_size
        patch_sizes = self.patch_sizes
        if patch_size is None:
            patch_size = patch_sizes[0] if patch_sizes else DEFAULT_PATCH_SIZE
        if isinstance(patch_size, Sequence) and not isinstance(patch_size, (str, bytes)):
            patch_sizes = tuple(patch_size)
            patch_size = patch_size[0]
        assert isinstance(patch_size, int), "patch_size must resolve to a single int"
        patch_size = int(patch_size)
        if patch_sizes is None:
            patch_sizes = (patch_size,)
        patch_sizes = tuple(sorted({int(p) for p in patch_sizes} | {patch_size}))

        pos_embed = self.pos_embed
        legacy = pos_embed in LEGACY_POS_EMBEDS
        if (
            pos_embed is not None
            and not legacy
            and pos_embed
            not in (
                "fractional",
                "rope",
                "none",
            )
        ):
            raise ValueError(
                f"unknown pos_embed '{pos_embed}'; expected learned, sincos, "
                "fractional, rope or none"
            )
        encoder_pos_embed = self.encoder_pos_embed
        if encoder_pos_embed is None:
            encoder_pos_embed = "none" if legacy or pos_embed == "none" else "rope"
        decoder_pos_embed = self.decoder.pos_embed
        if decoder_pos_embed is None:
            if legacy:
                decoder_pos_embed = pos_embed
            elif pos_embed == "none":
                decoder_pos_embed = "none"
            else:
                decoder_pos_embed = "fractional"
        fusion_pos_embed = self.fusion_pos_embed
        if fusion_pos_embed is None:
            fusion_pos_embed = pos_embed if legacy else "none"
        if encoder_pos_embed not in ("rope", "none"):
            raise ValueError(f"unknown encoder_pos_embed '{encoder_pos_embed}'")
        if decoder_pos_embed not in ("fractional", "none", *LEGACY_POS_EMBEDS):
            raise ValueError(f"unknown decoder_pos_embed '{decoder_pos_embed}'")
        if fusion_pos_embed not in ("none", *LEGACY_POS_EMBEDS):
            raise ValueError(f"unknown fusion_pos_embed '{fusion_pos_embed}'")
        if legacy and (
            encoder_pos_embed != "none"
            or decoder_pos_embed not in LEGACY_POS_EMBEDS
            or fusion_pos_embed not in LEGACY_POS_EMBEDS
        ):
            raise ValueError(
                "legacy pos_embed='learned'/'sincos' uses table PE on fusion and "
                "decoder; do not combine it with encoder/fusion overrides"
            )

        return replace(
            self,
            core=replace(self.core, adaln=self.scale_conditioning == "adaln"),
            decoder=replace(self.decoder, pos_embed=decoder_pos_embed),
            patch_size=patch_size,
            patch_sizes=patch_sizes,
            encoder_pos_embed=encoder_pos_embed,
            fusion_pos_embed=fusion_pos_embed,
        )

    @classmethod
    def from_kwargs(cls, **kw: Any) -> ModelConfig:
        """Inverse of :meth:`to_kwargs`: build from flat constructor keywords.

        The flat ``ChannelFusionMAE`` constructor routes through this so its
        defaults (sourced from :data:`DEFAULT_CONFIG`) and the structured
        config cannot diverge.
        """
        unknown = set(kw) - _FLAT_KEYS
        missing = _FLAT_KEYS - set(kw)
        if unknown or missing:
            raise TypeError(
                f"flat config keywords mismatch; unknown={sorted(unknown)} "
                f"missing={sorted(missing)}"
            )
        return cls(
            core=SharedCore(dim=kw["dim"], n_heads=kw["n_heads"], dropout=kw["dropout"]),
            encoder=EncoderConfig(depth=kw["encoder_depth"], n_memory=kw["n_memory"]),
            decoder=DecoderConfig(
                depth=kw["decoder_depth"],
                pos_embed=kw["decoder_pos_embed"],
                fractional_pe_mode=kw["fractional_pe_mode"],
                fractional_pe_base=kw["fractional_pe_base"],
                pe_ref_um=kw["pe_ref_um"],
                head_mode=kw["head_mode"],
                head_resample=kw["head_resample"],
                head_patch_size=kw["head_patch_size"],
            ),
            **{k: kw[k] for k in _TOP_LEVEL_FLAT},
        )

    def to_kwargs(self) -> dict:
        """Flatten to ``ChannelFusionMAE.__init__`` keyword names.

        ``mlp_ratio`` on the encoder/decoder is intentionally not emitted: the
        flat constructor bakes it at 4.0.  Keep this in sync with the
        constructor; ``tests/test_config.py`` asserts the key sets match.
        """
        return {
            "image_size": self.image_size,
            "patch_size": self.patch_size,
            "patch_sizes": self.patch_sizes,
            "flexi_random": self.flexi_random,
            "max_tokens": self.max_tokens,
            "pixel_size": self.pixel_size,
            "pe_space": self.pe_space,
            # core (adaln is derived from scale_conditioning, not a flat kwarg)
            "dim": self.core.dim,
            "n_heads": self.core.n_heads,
            "dropout": self.core.dropout,
            # encoder
            "encoder_depth": self.encoder.depth,
            "n_memory": self.encoder.n_memory,
            # decoder
            "decoder_depth": self.decoder.depth,
            "decoder_pos_embed": self.decoder.pos_embed,
            "fractional_pe_mode": self.decoder.fractional_pe_mode,
            "fractional_pe_base": self.decoder.fractional_pe_base,
            "pe_ref_um": self.decoder.pe_ref_um,
            "head_mode": self.decoder.head_mode,
            "head_resample": self.decoder.head_resample,
            "head_patch_size": self.decoder.head_patch_size,
            # top-level
            "pos_embed": self.pos_embed,
            "encoder_pos_embed": self.encoder_pos_embed,
            "fusion_pos_embed": self.fusion_pos_embed,
            "rope_base": self.rope_base,
            "rope_ref_extent": self.rope_ref_extent,
            "rope_shared_extent": self.rope_shared_extent,
            "fusion": self.fusion,
            "fusion_attn_dim": self.fusion_attn_dim,
            "n_latents": self.n_latents,
            "decoder_kv_components": self.decoder_kv_components,
            "p_patch": self.p_patch,
            "p_channel": self.p_channel,
            "patch_ratio": self.patch_ratio,
            "mode_sampling": self.mode_sampling,
            "min_visible_channels": self.min_visible_channels,
            "max_masked_channels_frac": self.max_masked_channels_frac,
            "keep_offcell_as_context": self.keep_offcell_as_context,
            "scale_conditioning": self.scale_conditioning,
            "scale_fourier_frequencies": self.scale_fourier_frequencies,
            "num_channel_ids": self.num_channel_ids,
            "loss": self.loss,
            "loss_reduce": self.loss_reduce,
        }


# --------------------------------------------------------------------------- #
# Ablation registry: each entry is a transform on a ModelConfig, so it reads
# as a diff naming the exact config field(s) it changes and stays valid as
# defaults drift.  Add new ablations here rather than editing ``forward`` by
# hand, and give the transform a one-line docstring (the README table is
# rendered from it).
# --------------------------------------------------------------------------- #
def _ablate_full(cfg: ModelConfig) -> ModelConfig:
    """The shipped model; the identity transform."""
    return cfg


def _ablate_no_memory(cfg: ModelConfig) -> ModelConfig:
    """Drop encoder memory tokens: `encoder.n_memory = 0`.

    Isolates what the memory/register tokens add to the representation (they
    leave the decoder KV too).
    """
    return replace(cfg, encoder=replace(cfg.encoder, n_memory=0))


def _ablate_blind_decoder(cfg: ModelConfig) -> ModelConfig:
    """`no_memory` plus an empty decoder KV: `decoder_kv_components = ()`.

    The decoder cross-attends to nothing, so it is self-attention over the
    reconstruction targets only.
    """
    return replace(_ablate_no_memory(cfg), decoder_kv_components=())


def _ablate_no_scale_cond(cfg: ModelConfig) -> ModelConfig:
    """Drop scale conditioning: `scale_conditioning = "none"`."""
    return replace(cfg, scale_conditioning="none")


def _ablate_grid_pe(cfg: ModelConfig) -> ModelConfig:
    """Use normalized grid coordinates: `pe_space = "grid"`."""
    return replace(cfg, pe_space="grid")


#: Named ablations: a name maps to the ``ModelConfig -> ModelConfig`` transform
#: that produces it from the current default.  ``"full"`` is the shipped model.
#: Each transform's docstring is the single source for the README table.
ABLATIONS: dict[str, Callable[[ModelConfig], ModelConfig]] = {
    "full": _ablate_full,
    "no_memory": _ablate_no_memory,
    "blind_decoder": _ablate_blind_decoder,
    "no_scale_cond": _ablate_no_scale_cond,
    "grid_pe": _ablate_grid_pe,
}


def describe_ablations() -> str:
    """Render :data:`ABLATIONS` as a Markdown table.

    The first docstring line of each transform is its ``diff`` cell, so the
    registry is the only hand-maintained copy of the inventory; the README
    embeds the output and a test keeps the two in sync.
    """
    rows = ["| name | diff |", "| --- | --- |"]
    for name, transform in ABLATIONS.items():
        summary = (transform.__doc__ or "").strip().splitlines()[0]
        rows.append(f"| `{name}` | {summary} |")
    return "\n".join(rows)


#: Flat constructor keywords that map 1:1 onto a top-level :class:`ModelConfig`
#: field (everything except the core/encoder/decoder subconfig fields).
_TOP_LEVEL_FLAT = (
    "image_size",
    "patch_size",
    "patch_sizes",
    "flexi_random",
    "max_tokens",
    "pixel_size",
    "pe_space",
    "pos_embed",
    "encoder_pos_embed",
    "fusion_pos_embed",
    "rope_base",
    "rope_ref_extent",
    "rope_shared_extent",
    "fusion",
    "fusion_attn_dim",
    "n_latents",
    "decoder_kv_components",
    "p_patch",
    "p_channel",
    "patch_ratio",
    "mode_sampling",
    "min_visible_channels",
    "max_masked_channels_frac",
    "keep_offcell_as_context",
    "scale_conditioning",
    "scale_fourier_frequencies",
    "num_channel_ids",
    "loss",
    "loss_reduce",
)

#: Flat keywords that live in :class:`SharedCore` / :class:`EncoderConfig` /
#: :class:`DecoderConfig` rather than at the top level.
_SUBCONFIG_FLAT = (
    "dim",
    "n_heads",
    "dropout",
    "encoder_depth",
    "n_memory",
    "decoder_depth",
    "decoder_pos_embed",
    "fractional_pe_mode",
    "fractional_pe_base",
    "pe_ref_um",
    "head_mode",
    "head_resample",
    "head_patch_size",
)

_FLAT_KEYS = frozenset(_TOP_LEVEL_FLAT + _SUBCONFIG_FLAT)

#: The authoring defaults: one instance shared by the ablation registry's base
#: and by the flat ``ChannelFusionMAE.__init__`` signature defaults, so the two
#: entry points cannot drift.
DEFAULT_CONFIG: ModelConfig = ModelConfig()


def ablated_config(name: str = "full", base: ModelConfig | None = None) -> ModelConfig:
    """Return the :class:`ModelConfig` for a named ablation."""
    if name not in ABLATIONS:
        raise KeyError(f"unknown ablation '{name}'; expected one of {sorted(ABLATIONS)}")
    return ABLATIONS[name](base or DEFAULT_CONFIG)


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
