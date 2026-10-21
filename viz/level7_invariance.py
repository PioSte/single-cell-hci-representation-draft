"""Level 7 -- Invariance & cost (measured values from a real model).

Unlike the other levels this runs the **actual** model and plots measured evidence
for the four claimed properties:

1. Cardinality invariance — output shape/size is independent of ``C``.
2. Permutation invariance — shuffling channel order does not change the result.
3. Padding-free ≡ masked-padded — exact numeric equivalence (modulo tolerance).
4. Cost — ``O(C²)`` vs ``O(C)`` fusion measured runtime or analytic FLOPs.

This keeps the total runtime modest (< 60s) by using a tiny model (16×16, 2 spatial
patches, 1-layer encoder/decoder) and only a handful of cardinality samples.

Usage::

    cd /home/piotr/Repos/fuse-channels
    uv run --extra viz python viz/level7_invariance.py
    uv run --extra viz python viz/level7_invariance.py --formats pdf png
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

# Add viz/ to path for figure_style import
sys.path.insert(0, str(Path(__file__).resolve().parent))

from figure_style import (
    BLUE,
    GREEN,
    INK,
    MUTED,
    ORANGE,
    RED,
    VIOLET,
    install_rcparams,
    save_figure,
)

# Import the model components
from fuse_channels.fusion import ChannelFusion, HypernetworkWeightedSum, dense_fusion_reference
from fuse_channels.mae import ChannelFusionMAE

OUT = Path(__file__).resolve().parent / "out"


def set_seed(seed: int = 42):
    """Set torch random seed for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def measure_cardinality_invariance(model, device, Cs=(1, 3, 6, 20, 60)):
    """Measure that descriptor shape is independent of channel count C."""
    results = []
    print("\n=== Cardinality Invariance ===")
    num_markers = 16
    for C in Cs:
        images = [torch.randn(C, 1, 16, 16, device=device)]
        # Use modulo to keep channel_ids within valid range
        channel_ids = [torch.arange(C, device=device) % num_markers]
        _, descriptor = model.encode(images, channel_ids, pool="mean")
        shape = list(descriptor.shape)
        norm = descriptor.norm(dim=-1).item()
        results.append({"C": C, "shape": shape, "norm": norm})
        print(f"C={C:2d} → shape={shape}, L2 norm={norm:.4f}")
    return results


def measure_permutation_invariance(model, device, C=10):
    """Measure that shuffling channel order does not change the result."""
    print("\n=== Permutation Invariance ===")
    num_markers = 16
    images = [torch.randn(C, 1, 16, 16, device=device)]
    channel_ids = [torch.arange(C, device=device) % num_markers]
    
    _, desc1 = model.encode(images, channel_ids, pool="mean")
    
    # Shuffle the channel order
    perm = torch.randperm(C, device=device)
    images_perm = [images[0][perm]]
    channel_ids_perm = [channel_ids[0][perm]]
    
    _, desc2 = model.encode(images_perm, channel_ids_perm, pool="mean")
    
    max_diff = (desc1 - desc2).abs().max().item()
    print(f"C={C}, shuffled → max|diff|={max_diff:.2e} (should be ~0)")
    return max_diff


def measure_padding_free_equiv(model, device):
    """Measure that padding-free and masked-padded implementations match."""
    print("\n=== Padding-free ≡ Masked-padded ===")
    # Mixed-C batch
    images = [
        torch.randn(3, 1, 16, 16, device=device),
        torch.randn(5, 1, 16, 16, device=device),
    ]
    channel_ids = [torch.arange(3, device=device), torch.arange(5, device=device)]
    visible = [
        torch.ones(3, model.n_patches, dtype=torch.bool, device=device),
        torch.ones(5, model.n_patches, dtype=torch.bool, device=device),
    ]
    
    # Get the ChannelSet via frontend
    cs = model.frontend(images, channel_ids, visible)
    
    # Compute fusion via padding-free and dense reference
    fused_padless, _ = model.fusion(cs)
    fused_dense = dense_fusion_reference(model.fusion, cs)
    
    max_diff = (fused_padless - fused_dense).abs().max().item()
    print(f"max|padding-free - dense|={max_diff:.2e} (should be ~1e-6)")
    return max_diff


def measure_cost(model, device, Cs=(1, 3, 5, 10, 15, 20, 30, 40)):
    """Measure runtime vs C for O(C²) and O(C) fusion."""
    print("\n=== Cost O(C²) vs O(C) ===")
    num_markers = 16
    
    # Build a hypernetwork version of the same model for comparison
    model_hyper = ChannelFusionMAE(
        image_size=16,
        patch_size=8,
        dim=32,
        encoder_depth=1,
        decoder_depth=1,
        n_heads=4,
        n_memory=2,
        num_markers=16,
        fusion="hypernetwork",
    ).to(device).eval()
    
    results_attn = []
    results_hyper = []
    
    warmup = 3
    runs = 5
    
    for C in Cs:
        images = [torch.randn(C, 1, 16, 16, device=device)]
        channel_ids = [torch.arange(C, device=device) % num_markers]
        
        # Warm up + measure attention fusion (O(C²))
        for _ in range(warmup):
            _ = model.encode(images, channel_ids, pool="mean")
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(runs):
            _ = model.encode(images, channel_ids, pool="mean")
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_attn = (time.perf_counter() - t0) / runs * 1000  # ms
        
        # Warm up + measure hypernetwork fusion (O(C))
        for _ in range(warmup):
            _ = model_hyper.encode(images, channel_ids, pool="mean")
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(runs):
            _ = model_hyper.encode(images, channel_ids, pool="mean")
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_hyper = (time.perf_counter() - t0) / runs * 1000  # ms
        
        results_attn.append({"C": C, "time_ms": t_attn})
        results_hyper.append({"C": C, "time_ms": t_hyper})
        print(f"C={C:2d}: Attention={t_attn:.3f}ms, Hypernetwork={t_hyper:.3f}ms")
    
    return results_attn, results_hyper


def header(ax, letter, text):
    """Panel header as a left-aligned axes title."""
    ax.set_title(f"{letter}   {text}", loc="left", fontsize=11.5,
                 fontweight="bold", color=INK)


def build_figure(card_results, perm_diff, padded_diff, cost_attn, cost_hyper):
    """Build the 4-panel figure."""
    fig = plt.figure(figsize=(14, 8.6))
    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.26,
                          left=0.07, right=0.98, top=0.91, bottom=0.09)

    # Panel A: Cardinality invariance
    ax_a = fig.add_subplot(gs[0, 0])
    header(ax_a, "A", "Cardinality invariance \u2014 shape [1, D] for all C")
    
    Cs = [r["C"] for r in card_results]
    norms = [r["norm"] for r in card_results]
    ax_a.bar(range(len(Cs)), norms, color=BLUE, alpha=0.7, edgecolor=BLUE, linewidth=1.5)
    ax_a.set_xticks(range(len(Cs)))
    ax_a.set_xticklabels([f"{C}" for C in Cs])
    ax_a.set_xlabel("Channel count $C$", fontsize=10, fontweight="bold")
    ax_a.set_ylabel("Descriptor L2 norm", fontsize=10, fontweight="bold")
    ax_a.grid(axis="y", alpha=0.3, linestyle="--")
    ax_a.spines["top"].set_visible(False)
    ax_a.spines["right"].set_visible(False)
    
    # Annotate the constant shape
    shape_str = str(card_results[0]["shape"])
    ax_a.text(0.98, 0.95, f"shape = {shape_str}", transform=ax_a.transAxes,
             ha="right", va="top", fontsize=9, color=INK, 
             bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor=GREEN, linewidth=1.5))
    
    # Panel B: Permutation invariance
    ax_b = fig.add_subplot(gs[0, 1])
    header(ax_b, "B", "Permutation invariance \u2014 max|\u0394| \u2248 0")
    
    ax_b.axhline(1.19e-7, color=MUTED, ls="--", lw=1.2, label="float32 eps")
    ax_b.plot([0], [max(perm_diff, 1e-9)], "o", color=ORANGE, markersize=11,
              label="shuffled channels")
    ax_b.set_xlim([-0.5, 0.5])
    ax_b.set_xticks([])
    ax_b.set_ylabel("max|descriptor - descriptor$_{\\mathrm{perm}}$|", fontsize=10, fontweight="bold")
    ax_b.set_ylim([1e-8, 1e-5])
    ax_b.set_yscale("log")
    ax_b.legend(loc="lower right", frameon=True, fontsize=8.5)
    ax_b.grid(axis="y", alpha=0.3, linestyle="--", which="both")
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)
    
    # Annotate the tolerance
    ax_b.text(0.98, 0.95, f"diff = {perm_diff:.2e}", transform=ax_b.transAxes,
             ha="right", va="top", fontsize=9, color=INK,
             bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor=ORANGE, linewidth=1.5))
    
    # Panel C: Padding-free == masked-padded
    ax_c = fig.add_subplot(gs[1, 0])
    header(ax_c, "C", "Padding-free \u2261 masked-padded \u2014 max|\u0394| ~ 1e-7")
    
    ax_c.axhline(1.19e-7, color=MUTED, ls="--", lw=1.2, label="float32 eps")
    ax_c.plot([0], [max(padded_diff, 1e-9)], "o", color=VIOLET, markersize=11,
              label="padding-free vs dense")
    ax_c.set_xlim([-0.5, 0.5])
    ax_c.set_xticks([])
    ax_c.set_ylabel("max|padding-free - dense|", fontsize=10, fontweight="bold")
    ax_c.set_ylim([1e-8, 1e-5])
    ax_c.set_yscale("log")
    ax_c.legend(loc="lower right", frameon=True, fontsize=8.5)
    ax_c.grid(axis="y", alpha=0.3, linestyle="--", which="both")
    ax_c.spines["top"].set_visible(False)
    ax_c.spines["right"].set_visible(False)
    
    # Annotate the tolerance
    ax_c.text(0.98, 0.95, f"diff = {padded_diff:.2e}", transform=ax_c.transAxes,
             ha="right", va="top", fontsize=9, color=INK,
             bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor=VIOLET, linewidth=1.5))
    
    # Panel D: Cost comparison
    ax_d = fig.add_subplot(gs[1, 1])
    header(ax_d, "D", "Cost: O(C\u00b2) attention vs O(C) hypernetwork")
    
    Cs_cost = [r["C"] for r in cost_attn]
    times_attn = [r["time_ms"] for r in cost_attn]
    times_hyper = [r["time_ms"] for r in cost_hyper]
    
    ax_d.plot(Cs_cost, times_attn, "o-", color=RED, linewidth=2, markersize=6, 
             label="ChannelFusion O(C²)", alpha=0.8)
    ax_d.plot(Cs_cost, times_hyper, "s-", color=GREEN, linewidth=2, markersize=6,
             label="HypernetworkWeightedSum O(C)", alpha=0.8)
    ax_d.set_xlabel("Channel count $C$", fontsize=10, fontweight="bold")
    ax_d.set_ylabel("Time (ms)", fontsize=10, fontweight="bold")
    ax_d.legend(loc="upper left", frameon=True, fontsize=8.5)
    ax_d.grid(alpha=0.3, linestyle="--")
    ax_d.spines["top"].set_visible(False)
    ax_d.spines["right"].set_visible(False)
    
    return fig


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--formats", nargs="+", default=["pdf", "svg", "png"])
    ap.add_argument("--stem", default="level7_invariance")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Build a small model
    set_seed(42)
    model = ChannelFusionMAE(
        image_size=16,
        patch_size=8,
        dim=32,
        encoder_depth=1,
        decoder_depth=1,
        n_heads=4,
        n_memory=2,
        num_markers=16,
        fusion="attention",
    ).to(device).eval()
    
    print(f"Model: {model.n_patches} patches, dim={model.fusion.dim}")
    
    # Run measurements
    with torch.no_grad():
        card_results = measure_cardinality_invariance(model, device)
        perm_diff = measure_permutation_invariance(model, device)
        padded_diff = measure_padding_free_equiv(model, device)
        cost_attn, cost_hyper = measure_cost(model, device)
    
    # Build and save figure
    install_rcparams()
    fig = build_figure(card_results, perm_diff, padded_diff, cost_attn, cost_hyper)
    save_figure(fig, OUT, args.stem, args.formats)
    plt.close(fig)
    
    print("\n✓ All measurements complete.")


if __name__ == "__main__":
    main()
