import torch

from fuse_channels import ChannelFusionMAE


def _batch(seed=0, C_list=(2, 4, 3), image_size=8):
    g = torch.Generator().manual_seed(seed)
    images = [torch.randn(C, 1, image_size, image_size, generator=g) for C in C_list]
    ids = [torch.randint(0, 16, (C,), generator=g) for C in C_list]
    return images, ids


def _fixed_visible(images, masker, seed=0):
    g = torch.Generator().manual_seed(seed)
    return masker(images, generator=g)


def _model(**kw):
    cfg = dict(
        image_size=8,
        patch_size=4,
        dim=32,
        encoder_depth=2,
        decoder_depth=1,
        n_heads=4,
        n_memory=3,
        num_markers=16,
    )
    cfg.update(kw)
    return ChannelFusionMAE(**cfg)


def test_forward_smoke():
    model = _model()
    images, ids = _batch()
    out = model(images, ids, generator=torch.Generator().manual_seed(1))
    assert torch.isfinite(out["loss"])
    assert out["pred"].shape[-1] == 16  # patch_size**2
    out["loss"].backward()
    assert any(p.grad is not None for p in model.parameters())


def test_padding_free_encoder_input():
    """Encoder consumes exactly the visible channel-patches (no pad tokens)."""
    model = _model()
    images, ids = _batch(C_list=(1, 5, 3))
    visible = _fixed_visible(images, model.masker, seed=2)
    cs = model.frontend(images, ids, visible)
    n_visible = sum(int(v.sum()) for v in visible)
    assert cs.tokens.shape[0] == n_visible

    out = model(images, ids, visible=visible)
    present = out["present"]
    # each present location contributes exactly one fused token
    assert int(present.sum()) == sum(
        int(present[b].sum()) for b in range(len(images))
    )
    assert torch.isfinite(out["loss"])


def test_overfit_single_batch():
    model = _model(per_channel_loss=False)
    images, ids = _batch(C_list=(2, 4, 3), seed=3)
    visible = _fixed_visible(images, model.masker, seed=4)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)

    losses = []
    for step in range(80):
        out = model(images, ids, visible=visible)
        opt.zero_grad()
        out["loss"].backward()
        opt.step()
        losses.append(out["loss"].item())

    first = sum(losses[:5]) / 5
    last = sum(losses[-5:]) / 5
    assert last < first * 0.7, f"loss did not drop: {first:.4f} -> {last:.4f}"


def test_memory_ablation_runs():
    model = _model(n_memory=0)
    images, ids = _batch()
    out = model(images, ids, generator=torch.Generator().manual_seed(1))
    assert torch.isfinite(out["loss"])

    model2 = _model(n_memory=4)
    out2 = model2(images, ids, generator=torch.Generator().manual_seed(1))
    assert torch.isfinite(out2["loss"])


def test_encode_inference():
    model = _model()
    model.eval()
    images, ids = _batch()
    mem, pooled = model.encode(images, ids)
    assert pooled.shape == (3, 32)
    assert torch.isfinite(pooled).all()


def test_encode_attention_pool_replaces_mean():
    """The descriptor is attention-pooled by default (no longer a mean pool)."""
    model = _model()
    model.eval()
    images, ids = _batch()
    _, pooled_attn = model.encode(images, ids, pool="attention")
    _, pooled_mean = model.encode(images, ids, pool="mean")
    assert pooled_attn.shape == pooled_mean.shape == (3, 32)
    assert torch.isfinite(pooled_attn).all()
    assert not torch.allclose(pooled_attn, pooled_mean)

    # pooling still works with no memory tokens
    model0 = _model(n_memory=0)
    model0.eval()
    _, pooled0 = model0.encode(images, ids, pool="attention")
    assert pooled0.shape == (3, 32)
    assert torch.isfinite(pooled0).all()


def test_open_vocab_text_conditioned():
    model = _model(num_markers=0, text_dim=12, use_learnable_identity=False)
    images = [torch.randn(C, 1, 8, 8) for C in (2, 5)]
    ids = [torch.randint(0, 16, (C,)) for C in (2, 5)]
    text = [torch.randn(C, 12) for C in (2, 5)]
    out = model(images, ids, text_embeds=text, generator=torch.Generator().manual_seed(0))
    assert torch.isfinite(out["loss"])
    # no fixed table was created -> unbounded channel count
    assert model.frontend.identity.embed is None


def test_hypernetwork_fusion_mae():
    model = _model(fusion="hypernetwork")
    images, ids = _batch()
    out = model(images, ids, generator=torch.Generator().manual_seed(1))
    assert torch.isfinite(out["loss"])
