import torch

from fuse_channels import ChannelFusionMAE, DecoderBlock, TransformerBlock


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
        num_channel_ids=16,
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


def test_encoder_and_decoder_use_standard_projections():
    """Encoder blocks and decoder blocks carry Q/K/V/O projections."""
    enc = TransformerBlock(32, 4)
    enc_params = set(dict(enc.named_parameters()))
    for name in ("q_proj.weight", "k_proj.weight", "v_proj.weight", "out_proj.weight"):
        assert name in enc_params, name

    dec = DecoderBlock(32, 4)
    dec_params = set(dict(dec.named_parameters()))
    # target self-attention
    for name in (
        "self_q_proj.weight",
        "self_k_proj.weight",
        "self_v_proj.weight",
        "self_out_proj.weight",
    ):
        assert name in dec_params, name
    # cross-attention
    for name in ("q_proj.weight", "k_proj.weight", "v_proj.weight", "out_proj.weight"):
        assert name in dec_params, name


def test_decoder_target_self_attention_mixes_queries():
    """A target query's prediction changes when a sibling target query changes."""
    torch.manual_seed(0)
    dec = DecoderBlock(32, 4)
    dec.eval()
    kv = torch.randn(5, 32)
    kv_group_id = torch.zeros(5, dtype=torch.long)
    q_group_id = torch.zeros(2, dtype=torch.long)
    qa = torch.randn(2, 32)
    qb = qa.clone()
    qb[1] += torch.randn(32)  # only the second target query changes
    with torch.no_grad():
        a = dec(qa, q_group_id, kv, kv_group_id, 1)
        b = dec(qb, q_group_id, kv, kv_group_id, 1)
    # self-attention makes the sibling change propagate to query 0
    assert not torch.allclose(a[0], b[0])


def test_encoder_input_length_is_present_locations():
    """The encoder gets one token per present location, not per visible patch."""
    model = _model()
    images, ids = _batch(C_list=(1, 5, 3))
    # Construct the mask explicitly so n_present < n_visible by construction:
    # sample 0 and 2 are fully visible; in sample 1 each of the 5 channels is
    # visible at one patch, so 5 tokens occupy only 4 locations.
    visible = [
        torch.ones(1, model.n_patches, dtype=torch.bool),
        torch.zeros(5, model.n_patches, dtype=torch.bool),
        torch.ones(3, model.n_patches, dtype=torch.bool),
    ]
    for c in range(5):
        visible[1][c, c % model.n_patches] = True
    cs = model.tokenizer(images, ids, visible)
    n_visible = sum(int(v.sum()) for v in visible)
    assert cs.tokens.shape[0] == n_visible

    encoder_lengths = []

    def capture_length(_module, args):
        encoder_lengths.append(args[0].shape[0])

    handle = model.encoder.register_forward_pre_hook(capture_length)
    try:
        out = model(images, ids, visible=visible)
    finally:
        handle.remove()

    n_present = int(out["present"].sum())
    assert encoder_lengths == [n_present]
    # A location may lose every channel, so present locations < visible tokens.
    assert n_present < n_visible
    assert torch.isfinite(out["loss"])


def test_overfit_single_batch():
    model = _model()
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


def test_memory_tokens_optional():
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


def test_descriptor_is_trained_by_the_mae_loss():
    """Every pooler parameter receives a gradient from the reconstruction loss."""
    model = _model()
    images, ids = _batch()
    out = model(images, ids, generator=torch.Generator().manual_seed(1))
    out["loss"].backward()
    missing = [
        name for name, p in model.pooler.named_parameters() if p.grad is None
    ]
    assert not missing, f"pooler parameters without gradients: {missing}"


def test_decoder_consumes_the_descriptor():
    """The trained descriptor is a decoder KV token and affects predictions."""
    model = _model()
    model.eval()
    images, ids = _batch()
    visible = _fixed_visible(images, model.masker, seed=2)
    B = len(images)
    captured = {}

    handles = [
        model.pooler.register_forward_hook(
            lambda _m, _i, out: captured.__setitem__("desc", out.detach())
        ),
        model.decoder.register_forward_pre_hook(
            lambda _m, args: captured.__setitem__("kv", args[0].detach())
        ),
    ]
    try:
        with torch.no_grad():
            out = model(images, ids, visible=visible)
    finally:
        for handle in handles:
            handle.remove()

    assert out["pred"] is not None
    # The decoder KV ends with the per-sample descriptor token.
    assert torch.allclose(captured["kv"][-B:], captured["desc"])
    assert torch.allclose(out["descriptor"], captured["desc"], atol=1e-6)

    # Perturbing only the pooler output must change the reconstruction.  The
    # delta is non-constant because a constant shift cancels in LayerNorm.
    delta = torch.linspace(-1.0, 1.0, captured["desc"].shape[-1])
    handle = model.pooler.register_forward_hook(lambda _m, _i, out: out + delta)
    try:
        with torch.no_grad():
            out_perturbed = model(images, ids, visible=visible)
    finally:
        handle.remove()
    assert not torch.allclose(out["pred"], out_perturbed["pred"])


def test_forward_and_encode_share_the_descriptor():
    """encode(pool='attention') returns the descriptor trained in forward."""
    model = _model()
    model.eval()
    images, ids = _batch()
    visible = [
        torch.ones(im.shape[0], model.n_patches, dtype=torch.bool) for im in images
    ]
    with torch.no_grad():
        out = model(images, ids, visible=visible)
        _, pooled = model.encode(images, ids, pool="attention")
    assert out["pred"] is None  # nothing masked
    assert out["descriptor"].shape == pooled.shape == (3, 32)
    assert torch.allclose(out["descriptor"], pooled, atol=1e-6)


def test_post_fusion_pe_and_no_patch_embed_pe():
    model = _model()
    assert tuple(model.fusion.spatial_pos.shape) == (1, model.n_patches, 32)
    assert not hasattr(model.tokenizer.patch_embed, "spatial_pos")


def test_multi_query_descriptor_smoke():
    model = _model(n_queries=3)
    images, ids = _batch()
    out = model(images, ids, generator=torch.Generator().manual_seed(1))
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    model.eval()
    _, pooled = model.encode(images, ids, pool="attention")
    assert pooled.shape == (3, 32)
    assert torch.isfinite(pooled).all()


def test_identity_gated_fusion_mae():
    model = _model(fusion="identity_gated")
    images, ids = _batch()
    out = model(images, ids, generator=torch.Generator().manual_seed(1))
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    assert model.fusion.weight_net[0].weight.grad is not None


def test_dcp_dropped_channel_is_a_reconstruction_target():
    """Fully dropped channels are reconstructed; absent ids never are."""
    model = _model(p_patch=0.0, p_channel=1.0)
    g = torch.Generator().manual_seed(7)
    images = [torch.randn(3, 1, 8, 8, generator=g)]
    ids = [torch.tensor([3, 5, 7])]
    absent_id = 0  # valid embedding id, but not in this sample's panel

    for seed in range(50):
        visible = model.masker(
            images, generator=torch.Generator().manual_seed(seed)
        )
        dropped = ~visible[0].any(dim=1)
        if bool(dropped.any()):
            break
    assert bool(dropped.any())
    assert not bool(visible[0][dropped].any())

    out = model(images, ids, visible=visible)
    target_ids = out["target_channel_id"]
    assert out["pred"] is not None
    assert out["pred"].shape[0] == target_ids.numel()

    # Every patch of a dropped channel is a target, and nothing else is.
    for c in dropped.nonzero(as_tuple=True)[0].tolist():
        assert int((target_ids == ids[0][c]).sum()) == model.n_patches
    for c in (~dropped).nonzero(as_tuple=True)[0].tolist():
        assert not bool((target_ids == ids[0][c]).any())
    assert set(target_ids.tolist()) <= set(ids[0].tolist())
    assert absent_id not in set(target_ids.tolist())


def test_attention_primitives_route_self_and_cross_attention(monkeypatch):
    """Encoder/decoder self-attention use the bucketed projected primitive;
    decoder cross-attention uses the general grouped primitive."""
    import fuse_channels.decoder as decoder_module
    import fuse_channels.encoder as encoder_module

    calls = {"encoder_self": 0, "decoder_self": 0, "decoder_cross": 0}
    encoder_self = encoder_module.grouped_projected_self_attention
    decoder_self = decoder_module.grouped_projected_self_attention
    decoder_cross = decoder_module.grouped_attention

    def count_encoder_self(*args, **kwargs):
        calls["encoder_self"] += 1
        return encoder_self(*args, **kwargs)

    def count_decoder_self(*args, **kwargs):
        calls["decoder_self"] += 1
        return decoder_self(*args, **kwargs)

    def count_decoder_cross(*args, **kwargs):
        calls["decoder_cross"] += 1
        return decoder_cross(*args, **kwargs)

    monkeypatch.setattr(
        encoder_module, "grouped_projected_self_attention", count_encoder_self
    )
    monkeypatch.setattr(
        decoder_module, "grouped_projected_self_attention", count_decoder_self
    )
    monkeypatch.setattr(decoder_module, "grouped_attention", count_decoder_cross)

    model = _model()
    images, ids = _batch()
    visible = _fixed_visible(images, model.masker, seed=2)
    out = model(images, ids, visible=visible)

    assert out["pred"] is not None
    assert calls["encoder_self"] == len(model.encoder.blocks)
    assert calls["decoder_self"] == len(model.decoder.blocks)
    assert calls["decoder_cross"] == len(model.decoder.blocks)


def test_sincos_pos_embed_model():
    """pos_embed="sincos" swaps both PE tables for fixed buffers; training still runs."""
    model = _model(pos_embed="sincos")
    assert model.fusion.pos_embed == "sincos"
    assert model.decoder.pos_embed == "sincos"
    assert "spatial_pos" not in dict(model.named_parameters())
    assert tuple(model.fusion.spatial_pos.shape) == (1, model.n_patches, 32)
    assert tuple(model.decoder.spatial_pos.shape) == (model.n_patches, 32)
    assert not model.decoder.spatial_pos.requires_grad

    images, ids = _batch()
    fusion_before = model.fusion.spatial_pos.clone()
    before = model.decoder.spatial_pos.clone()
    out = model(images, ids, generator=torch.Generator().manual_seed(1))
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    assert any(p.grad is not None for p in model.parameters())
    # the fixed maps are never optimized
    assert torch.equal(model.decoder.spatial_pos, before)
    assert torch.equal(model.fusion.spatial_pos, fusion_before)
    assert model.decoder.spatial_pos.grad is None


def test_sincos_pos_embed_with_identity_gated_fusion():
    model = _model(pos_embed="sincos", fusion="identity_gated")
    assert model.fusion.pos_embed == "sincos"
    assert "spatial_pos" not in dict(model.fusion.named_parameters())
    images, ids = _batch()
    out = model(images, ids, generator=torch.Generator().manual_seed(1))
    assert torch.isfinite(out["loss"])
    out["loss"].backward()


def test_decoder_target_position_changes_predictions():
    """Absolute target PE is load-bearing: same channel, different patch differs."""
    for kind in ("learned", "sincos"):
        model = _model(pos_embed=kind)
        model.eval()
        images, ids = _batch(C_list=(2,))
        # mask one channel entirely: every target query is mask token + PE + identity
        visible = [torch.zeros(2, model.n_patches, dtype=torch.bool)]
        visible[0][1] = True  # keep channel 1 fully visible
        with torch.no_grad():
            out = model(images, ids, visible=visible)
        pred = out["pred"]
        # channel 0 is fully masked: its n_patches target queries share the
        # mask token and channel embedding, so only the spatial PE separates them
        per_patch = pred.reshape(model.n_patches, -1)
        assert not torch.allclose(per_patch[0], per_patch[1])
        assert not torch.allclose(per_patch[0], per_patch[-1])
