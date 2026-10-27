"""Cost check: cross-attention fusion vs. identity-gated sum at C=60.

    uv run python examples/benchmark_fusion.py
"""

from __future__ import annotations

import time

import torch

from fuse_channels import ChannelFusion, ChannelSet, IdentityGatedSum


def make_set(B, C, N, D, seed=0):
    g = torch.Generator().manual_seed(seed)
    T = B * C * N
    tok = torch.randn(T, D, generator=g)
    si = torch.arange(B).repeat_interleave(C * N)
    ci = torch.arange(C).repeat_interleave(N).repeat(B)
    pi = torch.arange(N).repeat(C).repeat(B)
    embed = torch.randn(T, D, generator=g)
    return ChannelSet(
        tokens=tok + embed,
        sample_idx=si,
        channel_idx=ci,
        patch_idx=pi,
        channel_id=torch.zeros(T, dtype=torch.long),
        n_samples=B,
        n_patches=N,
        channel_embed=embed,
        channel_group_id=torch.arange(B * C).repeat_interleave(N),
    )


@torch.no_grad()
def bench(module, cs, iters=20):
    module.eval()
    for _ in range(3):
        module(cs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        module(cs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.time() - t0) / iters


def main() -> None:
    B, C, N, D = 8, 60, 16, 128
    cs = make_set(B, C, N, D)
    attn = ChannelFusion(D, n_heads=4, n_patches=N)
    gated = IdentityGatedSum(D, n_patches=N)

    t_attn = bench(attn, cs)
    t_gated = bench(gated, cs)
    print(f"tokens: B={B} C={C} N={N} D={D}  -> T={B*C*N}")
    print(
        f"attention fusion       : {t_attn*1e3:8.2f} ms / step "
        f"(attn_dim={attn.attn_dim})"
    )
    print(f"identity-gated sum     : {t_gated*1e3:8.2f} ms / step")
    print(f"speedup (gated/attn)   : {t_attn/max(t_gated,1e-9):.2f}x")


if __name__ == "__main__":
    main()
