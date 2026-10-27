"""Identity decoupling: channel identity as routing bias + output skip.

The channel embedding is never added to the patch token before a LayerNorm.
Instead it enters the fusion twice:

    logits_c = q · LN(t_c) + bias(e_c)
    fused    = proj(Σ a_c LN(t_c)) + skip(Σ a_c e_c)

so (i) identity still routes the per-location channel attention (ChannelViT /
Panopticon), (ii) identity still reaches the encoder representation (ChannelViT
Table 10), and (iii) the fused token is invariant to a positive rescaling of
content, so a patch-size-induced change of token magnitude cannot move the
identity/content balance.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch
import torch.nn.functional as F

from fuse_channels import (
    ChannelEmbedding,
    ChannelFusion,
    ChannelFusionMAE,
    ChannelTokenizer,
    IdentityGatedSum,
)
from fuse_channels.container import ChannelSet
from fuse_channels.fusion import _dense_fusion_reference


DIM = 192
N_HEADS = 6
IMAGE = 240


# --------------------------------------------------------------------------- #
# fixtures / helpers
# --------------------------------------------------------------------------- #
def _smooth_image(seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(1, 1, IMAGE, IMAGE, generator=g)
    spec = torch.fft.fftshift(torch.fft.fft2(x))
    yy, xx = torch.meshgrid(
        torch.arange(IMAGE) - IMAGE / 2,
        torch.arange(IMAGE) - IMAGE / 2,
        indexing="ij",
    )
    spec = spec / (yy**2 + xx**2).sqrt().clamp_min(1)
    x = torch.real(torch.fft.ifft2(torch.fft.ifftshift(spec)))
    return (x - x.mean()) / x.std()


def _punctate_image(n_puncta: int = 120, sigma: float = 1.2, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    x = torch.zeros(1, 1, IMAGE, IMAGE)
    x.view(-1)[torch.randperm(IMAGE * IMAGE, generator=g)[:n_puncta]] = (
        torch.rand(n_puncta, generator=g) * 8 + 4
    )
    k = torch.exp(-(torch.arange(-4, 5).float() ** 2) / (2 * sigma**2))
    k = k / k.sum()
    kernel = (k[None, None] * k[None, :, None]).expand(1, 1, -1, -1).contiguous()
    x = F.conv2d(x, kernel, padding=4)
    return (x - x.mean()) / x.std()


def _fusion(kind: str):
    if kind == "attention":
        return ChannelFusion(DIM, n_heads=N_HEADS, pos_embed="none")
    return IdentityGatedSum(DIM, pos_embed="none")


def _cs(
    tokens: torch.Tensor,
    channel_embed: torch.Tensor,
    n_patches: int,
    n_channels: int,
) -> ChannelSet:
    """Build a small hand-made ChannelSet for fusion-only tests."""
    n_tokens = tokens.shape[0]
    assert n_tokens == n_channels * n_patches
    channel_idx = torch.arange(n_channels).repeat_interleave(n_patches)
    patch_idx = torch.arange(n_patches).repeat(n_channels)
    return ChannelSet(
        tokens=tokens,
        sample_idx=torch.zeros(n_tokens, dtype=torch.long),
        channel_idx=channel_idx,
        patch_idx=patch_idx,
        channel_id=channel_idx.clone(),
        n_samples=1,
        n_patches=n_patches,
        channel_embed=channel_embed,
        channel_group_id=channel_idx.clone(),
    )


def _sensitivity(fusion, cs_for_id) -> float:
    f0 = fusion(cs_for_id(0))[0]
    f1 = fusion(cs_for_id(1))[0]
    return ((f0 - f1).norm() / f0.norm()).item()


# --------------------------------------------------------------------------- #
# tokenizer contract
# --------------------------------------------------------------------------- #
def test_tokenizer_returns_content_only_tokens():
    emb = ChannelEmbedding(DIM, 8)
    img = torch.randn(1, 1, 32, 32, generator=torch.Generator().manual_seed(0))
    ids = [torch.tensor([0, 1])]
    tok = ChannelTokenizer(32, 8, DIM, emb)
    cs = tok([img], ids, patch_sizes=8)
    raw, _ = tok.patch_embed([img], ids, patch_sizes=8)
    assert torch.equal(cs.tokens, raw)
    assert cs.channel_embed is not None
    assert not torch.allclose(cs.tokens, cs.channel_embed)


# --------------------------------------------------------------------------- #
# core invariants
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kind", ["attention", "identity_gated"])
def test_fusion_is_content_scale_invariant(kind):
    """Rescaling content must not move the fused token."""
    g = torch.Generator().manual_seed(0)
    n_ch, n_patch = 3, 5
    tokens = torch.randn(n_ch * n_patch, DIM, generator=g)
    e_ids = torch.randn(n_ch, DIM, generator=g) * 0.3
    channel_embed = e_ids.repeat_interleave(n_patch, dim=0)
    fusion = _fusion(kind)

    base = fusion(_cs(tokens, channel_embed, n_patch, n_ch))[0]
    for scale in (0.2, 5.0):
        out = fusion(_cs(scale * tokens, channel_embed, n_patch, n_ch))[0]
        assert torch.allclose(out, base, atol=5e-4), (kind, scale)


def test_zero_content_is_identity_only():
    """An empty patch carries identity, not amplified noise."""
    n_ch, n_patch = 1, 3
    channel_embed = (
        torch.randn(n_ch, DIM, generator=torch.Generator().manual_seed(0)) * 0.3
    )
    zeros = torch.zeros(n_ch * n_patch, DIM)
    fusion = ChannelFusion(DIM, n_heads=N_HEADS, pos_embed="none")
    fused = fusion(
        _cs(zeros, channel_embed.repeat_interleave(n_patch, dim=0), n_patch, n_ch)
    )[0]
    # one channel per location: content attention collapses to v_proj(0), so the
    # only content that survives is the value-projection bias.
    v0 = fusion.v_proj(torch.zeros(1, DIM))
    expected = fusion.proj_out(v0) + fusion.identity_skip(channel_embed[0])
    assert torch.allclose(fused, expected.expand_as(fused), atol=1e-5)


@pytest.mark.parametrize("kind", ["attention", "identity_gated"])
def test_routing_still_uses_identity(kind):
    fusion = _fusion(kind)
    n_ch, n_patch = 2, 4
    tokens = torch.randn(
        n_ch * n_patch, DIM, generator=torch.Generator().manual_seed(0)
    )
    e_ids = torch.randn(n_ch, DIM, generator=torch.Generator().manual_seed(1)) * 0.3
    channel_embed = e_ids.repeat_interleave(n_patch, dim=0)
    cs = _cs(tokens, channel_embed, n_patch, n_ch)
    zero = _cs(tokens, torch.zeros_like(channel_embed), n_patch, n_ch)
    assert not torch.allclose(fusion(cs)[0], fusion(zero)[0])


# --------------------------------------------------------------------------- #
# the regression: identity routing across patch sizes
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kind", ["attention", "identity_gated"])
@pytest.mark.parametrize("content", ["smooth", "punctate"])
def test_identity_routing_is_patch_size_invariant(kind, content):
    img = _smooth_image() if content == "smooth" else _punctate_image()
    emb = ChannelEmbedding(DIM, 8)
    p0 = 40
    patch_sizes = [8, 16, 40]

    tok = ChannelTokenizer(IMAGE, p0, DIM, emb)
    fusion = _fusion(kind)

    sens = [
        _sensitivity(
            fusion,
            lambda c, p=p: tok([img], [torch.tensor([c])], patch_sizes=p),
        )
        for p in patch_sizes
    ]
    ratio = max(sens) / max(min(sens), 1e-9)
    assert ratio < 1.15, (kind, content, sens)
    assert min(sens) > 0.0


# --------------------------------------------------------------------------- #
# end-to-end plumbing
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("fusion_name", ["attention", "identity_gated"])
def test_mae_end_to_end(fusion_name):
    g = torch.Generator().manual_seed(0)
    model = ChannelFusionMAE(
        image_size=32,
        patch_size=16,
        dim=48,
        encoder_depth=1,
        decoder_depth=1,
        n_heads=4,
        n_memory=2,
        num_channel_ids=8,
        fusion=fusion_name,
        patch_sizes=(8, 16),
        scale_conditioning="adaln",
        pixel_size=0.5,
        p_patch=0.5,
        p_channel=0.0,  # keep multi-channel locations so the routing bias gets gradient
    )
    images = [
        torch.randn(3, 1, 32, 32, generator=g),
        torch.randn(5, 1, 32, 32, generator=g),
    ]
    ids = [torch.tensor([0, 1, 2]), torch.tensor([2, 3, 4, 5, 6])]
    out = model(images, ids, patch_sizes=[8, 16])
    assert torch.isfinite(out["loss"])
    out["loss"].backward()

    assert model.tokenizer.channel_embed.embed.weight.grad is not None
    assert model.tokenizer.channel_embed.embed.weight.grad.norm() > 0
    if fusion_name == "attention":
        assert model.fusion.identity_bias.weight.grad is not None
        assert model.fusion.identity_bias.weight.grad.norm() > 0
    assert model.fusion.identity_skip.weight.grad is not None
    assert model.fusion.identity_skip.weight.grad.norm() > 0

    _, descriptor = model.encode(images, ids, patch_sizes=[8, 16])
    assert descriptor.shape[0] == 2
    assert torch.isfinite(descriptor).all()


def test_fusion_requires_channel_embed():
    fusion = ChannelFusion(DIM, n_heads=N_HEADS, pos_embed="none")
    tokens = torch.randn(2, DIM)
    cs = ChannelSet(
        tokens=tokens,
        sample_idx=torch.zeros(2, dtype=torch.long),
        channel_idx=torch.tensor([0, 1]),
        patch_idx=torch.zeros(2, dtype=torch.long),
        channel_id=torch.tensor([0, 1]),
        n_samples=1,
        n_patches=1,
    )
    with pytest.raises(ValueError, match="channel_embed"):
        fusion(cs)


# --------------------------------------------------------------------------- #
# padding-free equivalence for the decoupled path
# --------------------------------------------------------------------------- #
def test_padding_free_matches_dense_reference():
    fusion = ChannelFusion(DIM, n_heads=N_HEADS, pos_embed="none")
    N = 4
    g = torch.Generator().manual_seed(0)
    tokens = torch.randn(2 * N + 3 * N, DIM, generator=g)
    e_ids = torch.randn(5, DIM, generator=g) * 0.3
    channel_embed = e_ids.repeat_interleave(N, dim=0)
    cs = ChannelSet(
        tokens=tokens,
        sample_idx=torch.tensor([0] * (2 * N) + [1] * (3 * N)),
        channel_idx=torch.tensor([0] * N + [1] * N + [0] * N + [1] * N + [2] * N),
        patch_idx=torch.tensor([i % N for i in range(5 * N)]),
        channel_id=torch.tensor([0] * N + [1] * N + [2] * N + [3] * N + [4] * N),
        n_samples=2,
        n_patches=N,
        channel_embed=channel_embed,
        channel_group_id=torch.tensor([0] * N + [1] * N + [2] * N + [3] * N + [4] * N),
    )
    fused, present = fusion(cs)
    dense = _dense_fusion_reference(fusion, cs)
    assert torch.allclose(fused, dense, atol=1e-5)
    assert present.shape == (2, N)
