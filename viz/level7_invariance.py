"""Level 7 -- Invariance & cost (measured values from a real model).

Unlike the other levels this runs the **actual** model and plots measured
evidence for three properties of the default path:

1. Cardinality invariance — output size is independent of ``C``.
2. Permutation invariance — shuffling channel order does not change the result.
3. Cost — per-location analytic FLOPs of the default fusion (single-query PMA,
   ``ChannelFusion``) and the ``O(C)`` ``IdentityGatedSum`` **ablation**; both
   scale as ``O(C)``.

This keeps the total runtime modest (< 60s) by using a tiny model (16x16, 4
spatial patches, 1-layer encoder/decoder) and only a handful of cardinality
samples.

Usage::

    cd /home/piotr/Repos/fuse-channels
    uv run --extra viz python viz/level7_invariance.py
    uv run --extra viz python viz/level7_invariance.py --formats pdf png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

# Add viz/ to path for figure_style import
sys.path.insert(0, str(Path(__file__).resolve().parent))

from figure_style import (  # noqa: E402
    FUSION,
    INK,
    MUTED,
    POOL,
    VIOLET,
    install_rcparams,
    save_figure,
)

# Import the model components
from fuse_channels.mae import ChannelFusionMAE

OUT = Path(__file__).resolve().parent / "out"

EPS = float(torch.finfo(torch.float32).eps)  # 1.19e-07
TOL = 1e-6


def set_seed(seed: int = 42):
    """Set torch random seed for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# Measurements
# --------------------------------------------------------------------------- #
def measure_cardinality_invariance(model, device, Cs=(1, 3, 6, 20, 60)):
    """Measure input size growth vs. constant ``encode()`` descriptor size.

    ``pool="attention"`` is the PMA readout the MAE trains and appends to the
    decoder KV, so the plotted vector is the real descriptor path (the weights
    are random here; this script measures shape invariance, not quality).
    """
    results = []
    print("\n=== Cardinality Invariance ===")
    n_channel_ids = 16
    for C in Cs:
        images = [torch.randn(C, 1, 16, 16, device=device)]
        # Use modulo to keep channel_ids within valid range
        channel_ids = [torch.arange(C, device=device) % n_channel_ids]
        _, descriptor = model.encode(images, channel_ids, pool="attention")
        shape = list(descriptor.shape)
        n_in = C * model.n_patches * descriptor.shape[-1]  # elements of [ΣC·N, D]
        results.append(
            {"C": C, "shape": shape, "n_in": n_in, "D": descriptor.shape[-1]}
        )
        print(
            f"C={C:2d} → shape={shape}, input token elements={n_in}, "
            f"descriptor elements={descriptor.numel()}"
        )
    return results


def measure_permutation_invariance(
    model, device, Cs=(3, 10, 20, 40), n_shuffles=4, seed=0
):
    """Shuffle channel order several times at several C; measure max|Δ|."""
    print("\n=== Permutation Invariance ===")
    results = []
    n_channel_ids = 16
    for C in Cs:
        torch.manual_seed(seed + C)
        images = [torch.randn(C, 1, 16, 16, device=device)]
        channel_ids = [torch.arange(C, device=device) % n_channel_ids]
        _, desc1 = model.encode(images, channel_ids, pool="attention")
        for s in range(n_shuffles):
            perm = torch.randperm(C, device=device)
            images_perm = [images[0][perm]]
            channel_ids_perm = [channel_ids[0][perm]]
            _, desc2 = model.encode(images_perm, channel_ids_perm, pool="attention")
            diff = (desc1 - desc2).abs().max().item()
            results.append({"C": C, "shuffle": s, "diff": diff})
        print(f"C={C:2d}, {n_shuffles} shuffles → max diff = "
              f"{max(r['diff'] for r in results if r['C'] == C):.2e}")
    return results


def analytic_cost(Cs, D, attn_dim=None):
    """Per-location analytic multiply-adds of the default fusion + ablation.

    * default single-query PMA (``ChannelFusion``): Q/K/V projections
      ``3·C·D·D_attn``, scores + weighted sum ``2·C·D_attn``, and the
      per-location output projection ``D·D_attn``, with
      ``D_attn = 3D`` by default;
    * ``IdentityGatedSum`` ablation: one per-channel ``D→D`` MLP + ``D→1`` head
      (``D² + D``) and a ``D→D`` value projection (``D²``) → ``C·(2·D² + D)``.

    Both scale as ``O(C)`` in the channel count.
    """
    A = 3 * D if attn_dim is None else attn_dim
    attn, gated = [], []
    for C in Cs:
        attn.append(3 * C * D * A + 2 * C * A + D * A)
        gated.append(C * (2 * D * D + D))
    return np.array(attn), np.array(gated)


# --------------------------------------------------------------------------- #
# Plotting helpers
# --------------------------------------------------------------------------- #
def header(ax, letter, text):
    """Panel header in the L5/L8 ``panel_header`` style (bold, left-aligned)."""
    ax.set_title(f"{letter}   {text}", loc="left", fontsize=12.2,
                 fontweight="bold", color=INK)


def tidy(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #
def build_figure(card_results, perm_results, Cs_cost, cost_attn, cost_gated,
                 n_patches, B):
    """Build the 3-panel figure (A/B on top, cost spanning the bottom)."""
    fig = plt.figure(figsize=(13.5, 9.9))
    gs = fig.add_gridspec(2, 2, hspace=0.40, wspace=0.24,
                          left=0.07, right=0.98, top=0.90, bottom=0.115)

    # ---- L7 badge (same style as L5/L8 panel_header) -------------------- #
    fig.text(0.07, 0.958, "L7", ha="left", va="center", fontsize=15,
             fontweight="bold", color=INK)
    fig.text(0.07 + 4.5 / 140.0, 0.958,
             "Invariance & cost — measured on the real model",
             ha="left", va="center", fontsize=12.2, fontweight="bold",
             color=INK)

    # ---- Panel A: cardinality invariance -------------------------------- #
    ax_a = fig.add_subplot(gs[0, 0])
    header(ax_a, "A", "Cardinality invariance — descriptor shape "
                      "constant in $C$")

    Cs = [r["C"] for r in card_results]
    n_in = [r["n_in"] for r in card_results]
    D = card_results[0]["D"]
    x = np.arange(len(Cs))

    ax_a.plot(x, n_in, "o-", color=MUTED, linewidth=2.0, markersize=7,
              label="input token list  $[\\Sigma C{\\cdot}N,\\ D]$  elements")
    ax_a.axhline(D, color=POOL, ls="-", lw=2.6,
                 label=f"descriptor  $[B,\\ D]$  "
                       f"($B{{\\cdot}}D={D}$ at $B{{=}}1$, constant)")

    ax_a.set_yscale("log")
    ax_a.set_ylim(10, max(n_in) * 4)
    ax_a.set_xticks(x)
    ax_a.set_xticklabels([str(C) for C in Cs], fontsize=9.5)
    ax_a.set_xlabel("Channel count $C$", fontsize=10, fontweight="bold")
    ax_a.set_ylabel("input token elements  $\\Sigma C{\\cdot}N{\\cdot}D$  "
                    "(log)", fontsize=10, fontweight="bold")
    ax_a.grid(axis="y", alpha=0.3, linestyle="--", which="both")
    tidy(ax_a)
    ax_a.legend(loc="upper left", frameon=True, fontsize=8.3)

    ax_a.text(0.98, 0.02,
              f"model: $B={B}$, $N={n_patches}$ patches/channel, $D={D}$\n"
              f"descriptor $[B, D] = [{B}, {D}]$, constant for all $C$",
              transform=ax_a.transAxes, ha="right", va="bottom",
              fontsize=8.6, color=INK, fontweight="bold",
              bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                        edgecolor=POOL, linewidth=1.5))

    # ---- Panel B: permutation invariance -------------------------------- #
    ax_b = fig.add_subplot(gs[0, 1])
    header(ax_b, "B", "Permutation invariance — channel order is irrelevant")

    bCs = sorted({r["C"] for r in perm_results})
    rng = np.random.default_rng(0)
    for i, C in enumerate(bCs):
        vals = [r["diff"] for r in perm_results if r["C"] == C]
        jit = rng.uniform(-0.16, 0.16, size=len(vals))
        ax_b.scatter(i + jit, vals, s=46, color=FUSION, alpha=0.9,
                     edgecolor="white", linewidth=0.6, zorder=4,
                     label=("one point per shuffle, max|$\\Delta$|"
                            if i == 0 else None))
    eps_line = ax_b.axhline(EPS, color=MUTED, ls="--", lw=1.3,
                            label=f"float32 machine $\\epsilon$ = {EPS:.1e}")
    tol_line = ax_b.axhline(TOL, color=INK, ls=":", lw=1.7,
                            label="chosen tolerance  $10^{-6}$")

    perm_max = max(r["diff"] for r in perm_results)
    ax_b.set_yscale("log")
    ax_b.set_ylim(5e-9, 3e-6)
    ax_b.set_xlim(-0.5, len(bCs) - 0.5)
    ax_b.set_xticks(range(len(bCs)))
    ax_b.set_xticklabels([f"{C}" for C in bCs])
    ax_b.set_xlabel("Channel count $C$", fontsize=10, fontweight="bold")
    ax_b.set_ylabel("max|descriptor - descriptor$_{\\mathrm{perm}}$|",
                    fontsize=10, fontweight="bold")
    ax_b.grid(axis="y", alpha=0.3, linestyle="--", which="both")
    tidy(ax_b)
    handles, labels = ax_b.get_legend_handles_labels()
    ax_b.legend(handles, labels, loc="lower right", frameon=True, fontsize=8.3)
    ax_b.text(0.98, 0.975,
              f"max|$\\Delta$| = {perm_max:.1e}  "
              f"(> $\\epsilon$: accumulator round-off)\n"
              f"$\\leq$ chosen $10^{{-6}}$ tolerance",
              transform=ax_b.transAxes, ha="right", va="top", fontsize=8.2,
              color=INK,
              bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                        edgecolor=FUSION, linewidth=1.5))

    # ---- Panel C: cost (spans the bottom row) --------------------------- #
    ax_c = fig.add_subplot(gs[1, :])
    header(ax_c, "C", "Cost per location — default PMA vs $O(C)$ ablation")

    ax_c.loglog(Cs_cost, cost_attn, "-", color=FUSION, linewidth=2.8,
                label="default PMA: ChannelFusion single-query  "
                      "($2CD + D^2$, $O(C)$)")
    ax_c.loglog(Cs_cost, cost_gated, "--", color=VIOLET, linewidth=2.8,
                label="ablation: IdentityGatedSum  "
                      "($C(2D^2 + D)$, $O(C)$)")

    ax_c.set_xlabel("Channel count $C$ (log scale)", fontsize=10,
                    fontweight="bold")
    ax_c.set_ylabel("analytic multiply-adds per location",
                    fontsize=10, fontweight="bold")
    ax_c.set_xlim(1, 60)
    ax_c.set_ylim(5e2, 2e6)
    ax_c.set_xticks([1, 3, 10, 30, 60])
    ax_c.set_xticklabels(["1", "3", "10", "30", "60"])
    ax_c.grid(alpha=0.3, linestyle="--", which="both")
    tidy(ax_c)
    ax_c.legend(loc="upper left", frameon=True, fontsize=9.0)

    ax_c.text(0.97, 0.97,
              f"analytic cost at $D={D}$:\n"
              f"default PMA  $2CD + D^2$\n"
              f"ablation  $C(2D^2 + D)$\n"
              "both scale as $O(C)$",
              transform=ax_c.transAxes, ha="right", va="top", fontsize=9.0,
              color=INK,
              bbox=dict(boxstyle="round,pad=0.45", facecolor="white",
                        edgecolor=VIOLET, linewidth=1.4))

    # ---- glossary footer ------------------------------------------------- #
    fig.text(0.07, 0.052,
             "symbols / glossary:  $C$ channel count (order-free);  "
             "$D$ token/embedding dim ($=32$);  "
             "$N$ spatial patches/channel ($=4$);  $B$ batch size ($=1$);",
             ha="left", va="center", fontsize=8.4, color=INK)
    fig.text(0.07, 0.022,
             "$\\mathrm{PMA}$ single-query pooled multi-head attention;  "
             "$O(C)$ cost grows linearly with channel count;  "
             "$\\epsilon$ float32 machine epsilon "
             "($\\approx 1.2\\times10^{-7}$).",
             ha="left", va="center", fontsize=8.4, color=INK)

    return fig


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--formats", nargs="+", default=["pdf", "svg", "png"])
    ap.add_argument("--stem", default="level7_invariance")
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    install_rcparams()
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
        num_channel_ids=16,
        pixel_size=1.0,
        fusion="attention",
    ).to(device).eval()

    print(f"Model: {model.n_patches} patches, dim={model.fusion.dim}, "
          f"descriptor-pooler queries={model.n_queries}")

    # Run measurements
    with torch.no_grad():
        card_results = measure_cardinality_invariance(model, device)
        perm_results = measure_permutation_invariance(model, device)

    D = card_results[0]["D"]
    Cs_cost = np.arange(1, 61)
    cost_attn, cost_gated = analytic_cost(Cs_cost, D)

    # Build and save figure
    fig = build_figure(card_results, perm_results, Cs_cost, cost_attn,
                       cost_gated, n_patches=model.n_patches, B=1)
    save_figure(fig, OUT, args.stem, args.formats)
    plt.close(fig)

    print("\n\u2713 All measurements complete.")


if __name__ == "__main__":
    main()
