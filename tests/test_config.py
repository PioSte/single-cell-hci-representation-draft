"""Tests for the structured config, the shared-core split, and the ablation registry."""

import inspect
from dataclasses import replace
from pathlib import Path

import torch

from fuse_channels import (
    ABLATIONS,
    DEFAULT_CONFIG,
    ChannelAwareDecoder,
    ChannelFusionMAE,
    DecoderConfig,
    EncoderConfig,
    ModelConfig,
    SharedCore,
    TransformerEncoder,
    ablated_config,
    build_model,
    describe_ablations,
)


def _tiny_base():
    return ModelConfig(
        image_size=8,
        patch_size=4,
        pixel_size=1.0,
        core=SharedCore(dim=32, n_heads=4),
        encoder=EncoderConfig(depth=1),
        decoder=DecoderConfig(depth=1),
    )


def _batch():
    g = torch.Generator().manual_seed(0)
    images = [torch.randn(C, 1, 8, 8, generator=g) for C in (2, 3)]
    ids = [torch.randint(0, 16, (c,), generator=g) for c in (2, 3)]
    return images, ids


def test_config_matches_constructor():
    """to_kwargs() must cover exactly the ChannelFusionMAE keyword surface."""
    init = set(inspect.signature(ChannelFusionMAE.__init__).parameters) - {"self"}
    kw = set(ModelConfig().to_kwargs())
    assert kw == init, f"missing={sorted(init - kw)} extra={sorted(kw - init)}"


def test_flat_defaults_come_from_the_config():
    """The flat constructor and ModelConfig share one set of defaults."""
    sig = inspect.signature(ChannelFusionMAE.__init__)
    for key, value in DEFAULT_CONFIG.to_kwargs().items():
        assert sig.parameters[key].default == value, key


def test_from_kwargs_is_the_inverse_of_to_kwargs():
    cfg = DEFAULT_CONFIG
    assert ModelConfig.from_kwargs(**cfg.to_kwargs()) == cfg


def test_resolve_is_idempotent_and_fills_the_shipped_recipe():
    resolved = DEFAULT_CONFIG.resolve()
    assert resolved.patch_size == 8 and resolved.patch_sizes == (8,)
    assert resolved.decoder.pos_embed == "fractional"
    assert resolved.encoder_pos_embed == "rope"
    assert resolved.fusion_pos_embed == "none"
    assert resolved.core.adaln is False
    assert resolved.resolve() == resolved


def test_legacy_pos_embed_is_expressible_through_the_config():
    """Deriving the decoder PE from pos_embed must work via ModelConfig too."""
    cfg = replace(_tiny_base(), pos_embed="learned")
    assert cfg.decoder.pos_embed is None  # inherit, not a hardcoded value
    resolved = cfg.resolve()
    assert resolved.decoder.pos_embed == "learned"
    model = ChannelFusionMAE.from_config(cfg)
    assert model.legacy_pos_embed and model.decoder_pos_embed == "learned"


def test_from_config_matches_flat_constructor():
    cfg = _tiny_base()
    a = ChannelFusionMAE.from_config(cfg).eval()
    b = ChannelFusionMAE(**cfg.to_kwargs()).eval()
    assert a.encoder.n_memory == b.encoder.n_memory
    assert len(a.encoder.blocks) == len(b.encoder.blocks)


def test_shared_core_is_one_object():
    """Encoder and decoder share the same core, so invariants cannot drift."""
    model = build_model("full", base=_tiny_base())
    assert model.core.dim == 32 and model.core.n_heads == 4
    # the modules read their width from that core, not from a private copy
    assert model.encoder.dim == model.core.dim
    assert model.encoder.n_heads == model.core.n_heads


def test_submodules_construct_from_core_and_cfg():
    """Option (b): composite modules take (core, cfg), usable standalone."""
    core = SharedCore(dim=32, n_heads=4, dropout=0.0, adaln=False)
    enc = TransformerEncoder(core, EncoderConfig(depth=2, n_memory=0))
    assert enc.n_memory == 0 and len(enc.blocks) == 2 and enc.dim == 32

    dec = ChannelAwareDecoder(
        core,
        DecoderConfig(depth=3, head_mode="per_p"),
        n_patches=4,
        canonical_patch_size=4,
        patch_sizes=(4,),
    )
    assert len(dec.blocks) == 3 and dec.canonical_patch_size == 4


def test_default_carries_no_memory_tokens():
    """The shipped model has no encoder memory tokens and a latents-only KV."""
    assert DEFAULT_CONFIG.encoder.n_memory == 0
    assert DEFAULT_CONFIG.decoder_kv_components == ("latents",)
    model = build_model(pixel_size=1.0).eval()
    assert model.encoder.memory is None
    images, ids = _batch()
    mem, latents = model.encode(images, ids)
    assert mem.shape == (0, model.encoder.dim)
    assert latents.shape == (len(images), model.n_latents, model.encoder.dim)


def test_ablated_config_changes_the_named_subconfig():
    base = _tiny_base()
    assert base.encoder.n_memory == 0  # default
    with_mem = ablated_config("memory", base)
    assert with_mem.encoder.n_memory > 0
    assert with_mem.decoder_kv_components == ("memory", "latents")

    blind = ablated_config("blind_decoder", base)
    assert blind.decoder_kv_components == ()


def test_every_ablation_builds_and_runs():
    images, ids = _batch()
    for name in ABLATIONS:
        model = build_model(name, base=_tiny_base()).eval()
        out = model(images, ids, generator=torch.Generator().manual_seed(1))
        assert torch.isfinite(out["loss"]), name


def test_every_ablation_documents_its_diff():
    """The registry is the only hand-maintained inventory; docs are rendered."""
    for name, transform in ABLATIONS.items():
        assert (transform.__doc__ or "").strip(), f"undocumented ablation: {name}"


def test_readme_table_matches_registry():
    """README carries the one ablation table, rendered from the registry."""
    readme = Path(__file__).resolve().parents[1] / "README.md"
    assert describe_ablations() in readme.read_text(encoding="utf-8")


def test_unknown_ablation_rejected():
    try:
        ablated_config("does_not_exist")
    except KeyError:
        pass
    else:  # pragma: no cover
        raise AssertionError("unknown ablation name should raise")
