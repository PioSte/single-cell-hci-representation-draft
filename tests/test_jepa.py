"""Tests for the JEPA path: target encoder, feature predictor, and MAE coexistence."""

import torch
import torch.nn.functional as F

from fuse_channels import (
    ChannelFusionJEPA,
    ChannelFusionMAE,
    DecoderConfig,
    EncoderConfig,
    JEPAConfig,
    ModelConfig,
    SharedCore,
    build_jepa,
    feature_prediction_loss,
)


def _batch(seed=0, c_list=(2, 3), image_size=8):
    g = torch.Generator().manual_seed(seed)
    images = [torch.randn(C, 1, image_size, image_size, generator=g) for C in c_list]
    ids = [torch.randint(0, 16, (C,), generator=g) for C in c_list]
    return images, ids


def _model(**kw):
    cfg = {
        "image_size": 8,
        "patch_size": 4,
        "dim": 32,
        "encoder_depth": 2,
        "decoder_depth": 1,
        "n_heads": 4,
        "n_memory": 3,
        "n_latents": 4,
        "num_channel_ids": 16,
        "pixel_size": 1.0,
        "jepa": JEPAConfig(
            target_blocks=1,
            target_scale=(0.5, 0.5),
            target_aspect=(1.0, 1.0),
            context_scale=(1.0, 1.0),
        ),
    }
    cfg.update(kw)
    return ChannelFusionJEPA(**cfg)


def _targets(n_patches=4):
    return [torch.tensor([True, False, False, False]) for _ in range(2)]


def test_jepa_forward_smoke_and_backward():
    model = _model()
    images, ids = _batch()
    out = model(
        images, ids, target_locations=_targets(), generator=torch.Generator().manual_seed(1)
    )
    assert torch.isfinite(out["loss"])
    assert out["pred"].shape == out["target"].shape
    assert out["pred"].shape[1] == 32
    out["loss"].backward()
    assert any(p.grad is not None for p in model.encoder.parameters())
    assert any(p.grad is not None for p in model.latent_pooler.parameters())
    assert all(p.grad is None for p in model.target_encoder.parameters())


def test_jepa_samples_targets_when_omitted():
    model = _model(
        jepa=JEPAConfig(target_blocks=1, target_scale=(0.25, 0.25), context_scale=(1.0, 1.0))
    )
    images, ids = _batch()
    out = model(images, ids, generator=torch.Generator().manual_seed(2))
    assert torch.isfinite(out["loss"])
    assert all(bool(t.any()) for t in out["target_locations"])


def test_build_jepa_uses_the_shared_backbone_config():
    cfg = ModelConfig(
        image_size=8,
        patch_size=4,
        pixel_size=1.0,
        core=SharedCore(dim=32, n_heads=4),
        encoder=EncoderConfig(depth=1),
        decoder=DecoderConfig(depth=1),
    )
    model = build_jepa(
        base=cfg,
        jepa=JEPAConfig(target_blocks=1, target_scale=(0.5, 0.5), context_scale=(1.0, 1.0)),
        n_memory=2,
        n_latents=4,
    )
    assert isinstance(model, ChannelFusionJEPA)
    assert model.n_latents == 4
    images, ids = _batch()
    out = model(images, ids, target_locations=_targets())
    assert torch.isfinite(out["loss"])


def test_mae_path_still_trains_alongside_jepa():
    mae = ChannelFusionMAE(
        image_size=8,
        patch_size=4,
        dim=32,
        encoder_depth=2,
        decoder_depth=1,
        n_heads=4,
        n_memory=3,
        num_channel_ids=16,
        pixel_size=1.0,
    )
    images, ids = _batch()
    visible = [torch.tensor([[True, False, True, True], [True, True, False, True]])]
    visible.append(
        torch.tensor(
            [[True, True, True, False], [False, True, True, True], [True, False, True, True]]
        )
    )
    mae_out = mae(images, ids, visible=visible)
    assert torch.isfinite(mae_out["loss"])
    assert mae_out["pred"] is not None
    mae_out["loss"].backward()
    assert any(p.grad is not None for p in mae.decoder.parameters())


def test_jepa_target_encoder_sees_every_location():
    model = _model()
    images, ids = _batch()
    lengths = []

    def capture(_module, args):
        lengths.append(args[0].shape[0])

    handle = model.target_encoder.register_forward_pre_hook(capture)
    try:
        out = model(images, ids, target_locations=_targets())
    finally:
        handle.remove()
    # One fused token per location, with all channels visible to the teacher.
    n_patches = (8 // 4) ** 2
    assert lengths == [len(images) * n_patches]
    assert bool(out["present"].all())


def test_jepa_target_is_stop_grad_and_channel_permutation_invariant():
    model = _model()
    model.eval()
    images, ids = _batch()
    targets = _targets()
    with torch.no_grad():
        out = model(images, ids, target_locations=targets)
        orders = [torch.arange(im.shape[0] - 1, -1, -1) for im in images]
        permuted = model(
            [im[order] for im, order in zip(images, orders, strict=True)],
            [i[order] for i, order in zip(ids, orders, strict=True)],
            target_locations=targets,
        )
    assert not out["target"].requires_grad
    # Fusion is permutation invariant, so the target representation at a
    # location must not depend on channel order.
    assert torch.allclose(out["target"], permuted["target"], atol=1e-6)
    assert torch.allclose(out["pred"], permuted["pred"], atol=1e-6)


def test_jepa_target_count_is_independent_of_channel_count():
    model = _model()
    model.eval()
    images, ids = _batch(c_list=(1, 3))
    images2, ids2 = _batch(c_list=(2, 5))
    with torch.no_grad():
        a = model(images, ids, target_locations=_targets())
        b = model(images2, ids2, target_locations=_targets())
    assert a["pred"].shape == b["pred"].shape == (2, 32)


def test_jepa_ema_update_uses_student_weights():
    model = _model()
    old = [p.detach().clone() for p in model.target_encoder.parameters()]
    with torch.no_grad():
        for p in model.encoder.parameters():
            p.add_(1.0)
    model.update_target_encoder(momentum=0.5)
    for before, target, student in zip(
        old, model.target_encoder.parameters(), model.encoder.parameters(), strict=True
    ):
        assert torch.allclose(target, 0.5 * before + 0.5 * student)


def test_jepa_respects_the_cell_valid_mask():
    model = _model()
    images, ids = _batch()
    valid = [
        torch.ones(4, dtype=torch.bool),
        torch.tensor([True, False, True, True]),
    ]
    targets = [torch.tensor([True, True, False, False]) for _ in range(2)]
    out = model(images, ids, target_locations=targets, valid=valid)
    assert torch.isfinite(out["loss"])
    # The invalid location in sample 1 is never a target.
    sample1 = out["target_group_id"] == 1
    assert 1 not in set(out["target_patch"][sample1].tolist())


def test_jepa_channel_dropout_smoke():
    model = _model(
        jepa=JEPAConfig(
            target_blocks=1,
            target_scale=(0.5, 0.5),
            target_aspect=(1.0, 1.0),
            context_scale=(1.0, 1.0),
            channel_dropout=True,
        )
    )
    images, ids = _batch()
    out = model(
        images, ids, target_locations=_targets(), generator=torch.Generator().manual_seed(3)
    )
    assert torch.isfinite(out["loss"])
    out["loss"].backward()


def test_feature_prediction_loss_reduction():
    pred = torch.tensor([[1.0, 0.0], [0.0, 2.0], [3.0, 3.0]])
    target = torch.zeros_like(pred)
    groups = torch.tensor([0, 0, 1])
    expected_l1 = torch.tensor([0.75, 3.0]).mean()
    assert torch.allclose(
        feature_prediction_loss(pred, target, groups, kind="l1", reduce="sample"),
        expected_l1,
    )
    expected_l2 = torch.tensor([1.25, 9.0]).mean()
    assert torch.allclose(
        feature_prediction_loss(pred, target, groups, kind="l2", reduce="sample"),
        expected_l2,
    )
    assert torch.allclose(
        feature_prediction_loss(pred, target, groups, kind="l1", reduce="token"),
        F.l1_loss(pred, target),
    )


def test_jepa_no_ema_uses_detached_student_target_and_a_regularizer():
    model = _model(
        jepa=JEPAConfig(
            target_blocks=1,
            target_scale=(0.5, 0.5),
            target_aspect=(1.0, 1.0),
            context_scale=(1.0, 1.0),
            use_ema=False,
            regularizer="sigreg",
            reg_weight=0.5,
        )
    )
    assert model.target_encoder is None
    assert isinstance(model.regularizer, torch.nn.Module)
    images, ids = _batch()
    out = model(images, ids, target_locations=_targets())
    assert torch.isfinite(out["loss"])
    assert out["reg_loss"] != 0
    assert torch.allclose(out["loss"], out["jepa_loss"] + 0.5 * out["reg_loss"])
    out["loss"].backward()
    assert any(p.grad is not None for p in model.encoder.parameters())
    # The no-EMA path has no target update to perform.
    model.update_target_encoder()


def test_jepa_ema_with_vicreg_regularizer():
    model = _model(
        jepa=JEPAConfig(
            target_blocks=1,
            target_scale=(0.5, 0.5),
            target_aspect=(1.0, 1.0),
            context_scale=(1.0, 1.0),
            regularizer="vicreg",
            reg_weight=0.1,
        )
    )
    assert model.target_encoder is not None
    images, ids = _batch()
    out = model(images, ids, target_locations=_targets())
    assert torch.isfinite(out["loss"])
    assert out["reg_loss"] >= 0
    assert torch.allclose(out["loss"], out["jepa_loss"] + 0.1 * out["reg_loss"])
    out["loss"].backward()
    before = [p.detach().clone() for p in model.target_encoder.parameters()]
    with torch.no_grad():
        for p in model.encoder.parameters():
            p.add_(0.5)
    model.update_target_encoder()
    assert any(
        not torch.equal(before_i, after_i)
        for before_i, after_i in zip(before, model.target_encoder.parameters(), strict=True)
    )


def test_jepa_overfits_a_fixed_batch():
    model = _model(
        jepa=JEPAConfig(
            target_blocks=1,
            target_scale=(0.5, 0.5),
            target_aspect=(1.0, 1.0),
            context_scale=(1.0, 1.0),
            ema_momentum=0.9,
        )
    )
    images, ids = _batch()
    targets = _targets()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    losses = []
    for _ in range(20):
        out = model(images, ids, target_locations=targets)
        opt.zero_grad()
        out["loss"].backward()
        opt.step()
        model.update_target_encoder()
        losses.append(float(out["loss"].detach()))
    assert losses[-1] < losses[0], losses
