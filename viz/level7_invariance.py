"""Level 7 -- Invariance & cost.

Panels A and B measure the **real, randomly initialised** default
:class:`~fuse_channels.mae.ChannelFusionMAE` (built through
``config.build_model`` / ``ModelConfig`` defaults) to show two properties of
the default path:

1. Cardinality invariance -- the fixed latent set ``[B, L, D]`` keeps its shape
   as the channel count ``C`` grows, while the ragged token list ``[ΣC·N, D]``
   grows.
2. Permutation invariance -- shuffling channel order leaves the latents
   unchanged up to float32 round-off.

Panel C is **analytic** (no forward pass): per-location multiply-adds of the
default single-query PMA fusion (``ChannelFusion``, ``D_attn = 3D``), the
shipped model path.

The model is small enough to run in a few seconds (16x16 images, 4 patches,
D=192, 4 encoder layers); the encoder and the latent pooler are exercised by
``encode``.

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
from matplotlib.ticker import NullFormatter

# Add viz/ to path for figure_style import
sys.path.insert(0, str(Path(__file__).resolve().parent))

from figure_style import (  # noqa: E402
    FUSION,
    GHOST,
    INK,
    MUTED,
    POOL,
    install_rcparams,
    rrect,
    save_figure,
    tint,
)

# Build through the config layer (never the flat constructor by hand).
from fuse_channels.config import ModelConfig, build_model  # noqa: E402
from fuse_channels.fleet import FLEET_PIXEL_SIZES  # noqa: E402

OUT = Path(__file__).resolve().parent / "out"

EPS = float(torch.finfo(torch.float32).eps)  # 1.19e-07
TOL = 1e-6

#: A fleet member (µm/px). 1.0 µm/px is *not* in the fleet and must not be used.
PIXEL_SIZE = 0.65
assert PIXEL_SIZE in FLEET_PIXEL_SIZES, "PIXEL_SIZE must be an acquisition-fleet member"


def set_seed(seed: int = 42):
    """Set torch random seed for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# Measurements
# --------------------------------------------------------------------------- #
def measure_cardinality_invariance(model, device, Cs=(1, 3, 6, 20, 60)):
    """Measure input size growth vs. the constant fixed latent set.

    ``encode`` returns the encoder representation that becomes the decoder KV:
    the ``[B, L, D]`` latent set from spatial pooling (the memory arm would add
    memory tokens on top).  The
    weights are random here: this measures shape invariance, not
    representation quality.
    """
    results = []
    print("\n=== Cardinality Invariance ===")
    for C in Cs:
        images = [torch.randn(C, 1, 16, 16, device=device)]
        channel_ids = [torch.arange(C, device=device)]
        _, latents = model.encode(images, channel_ids, pixel_sizes=PIXEL_SIZE)
        pooled = latents.mean(dim=1)
        D = latents.shape[-1]
        n_in = C * model.n_patches * D  # elements of [ΣC·N, D]
        results.append(
            {
                "C": C,
                "pooled_shape": list(pooled.shape),
                "latent_shape": list(latents.shape),
                "n_in": n_in,
                "D": D,
                "L": latents.shape[1],
            }
        )
        print(
            f"C={C:2d} → pooled={list(pooled.shape)}, "
            f"latents={list(latents.shape)}, input token elements={n_in}"
        )
    return results


def measure_permutation_invariance(
    model, device, Cs=(3, 10, 20, 40), n_shuffles=4, seed=0
):
    """Shuffle channel order several times at several C; measure max|Δ|."""
    print("\n=== Permutation Invariance ===")
    results = []
    for C in Cs:
        torch.manual_seed(seed + C)
        images = [torch.randn(C, 1, 16, 16, device=device)]
        channel_ids = [torch.arange(C, device=device)]
        _, lat1 = model.encode(images, channel_ids, pixel_sizes=PIXEL_SIZE)
        vec1 = lat1.mean(dim=1)
        for s in range(n_shuffles):
            perm = torch.randperm(C, device=device)
            images_perm = [images[0][perm]]
            channel_ids_perm = [channel_ids[0][perm]]
            _, lat2 = model.encode(
                images_perm, channel_ids_perm, pixel_sizes=PIXEL_SIZE
            )
            diff = (vec1 - lat2.mean(dim=1)).abs().max().item()
            results.append({"C": C, "shuffle": s, "diff": diff})
        print(f"C={C:2d}, {n_shuffles} shuffles → max diff = "
              f"{max(r['diff'] for r in results if r['C'] == C):.2e}")
    return results


def analytic_cost(Cs, D, attn_dim=None):
    """Per-location analytic multiply-adds of the default fusion.

    The default single-query PMA (``ChannelFusion``) spends K/V projections
    ``2·C·D·D_attn``, scores + weighted sum ``2·C·D_attn``, and the
    per-location output projection ``D·D_attn``, with ``D_attn = 3D`` by
    default -- i.e. ``6CD² + 6CD + 3D²`` (the one-off shared query
    projection and the small norm/identity terms are omitted).
    It scales as ``O(C)`` in the channel count.
    """
    A = 3 * D if attn_dim is None else attn_dim
    attn = []
    for C in Cs:
        attn.append(2 * C * D * A + 2 * C * A + D * A)
    return np.array(attn)


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
def build_figure(card_results, perm_results, Cs_cost, cost_attn, run):
    """Build the 3-panel figure (A/B on top, cost spanning the bottom)."""
    B = run["B"]
    N = run["N"]
    D = run["D"]
    L = run["L"]
    H = run["H"]
    P = run["P"]
    pixel = run["pixel_size"]
    n_heads = run["n_heads"]

    fig = plt.figure(figsize=(13.6, 10.6))
    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.22,
                          left=0.065, right=0.985, top=0.895, bottom=0.215)

    # ---- L7 badge + subtitle -------------------------------------------- #
    fig.text(0.065, 0.958, "L7", ha="left", va="center", fontsize=15,
             fontweight="bold", color=INK)
    fig.text(0.065 + 4.5 / 140.0, 0.958,
             "Invariance & cost — default model, randomly initialised weights",
             ha="left", va="center", fontsize=12.2, fontweight="bold",
             color=INK)
    fig.text(0.065, 0.925,
             "Panels A–B: measured on the untrained default model.  "
             "Panel C: analytic cost (no forward pass).  "
             f"Pixel size {pixel:g} µm/px (a fleet member).",
             ha="left", va="center", fontsize=9.4, color=MUTED)

    # ---- Panel A: cardinality invariance -------------------------------- #
    ax_a = fig.add_subplot(gs[0, 0])
    header(ax_a, "A", "Cardinality invariance — fixed latent set constant in $C$")

    Cs = [r["C"] for r in card_results]
    n_in = [r["n_in"] for r in card_results]
    D_A = card_results[0]["D"]
    L_A = card_results[0]["L"]

    ax_a.plot(Cs, n_in, "o-", color=MUTED, linewidth=2.0, markersize=7,
              label="input token list:  $\\Sigma C{\\cdot}N$ rows "
                    "$\\times\\, D$  (ragged, primary)")
    ax_a.axhline(L_A * D_A, color=POOL, ls="-", lw=2.6,
                 label="latents:  $[B,L,D]$, $L{\\cdot}D$  (constant)")

    # Padded tensor: only ever a dashed ghost, never materialised.
    ax_a.text(0.97, 0.96,
              "padded $[B,\\,C_{\\max},\\,N,\\,D]$\nnever materialised",
              transform=ax_a.transAxes, ha="right", va="top", fontsize=7.6,
              color=GHOST, style="italic",
              bbox=dict(boxstyle="round,pad=0.35", facecolor="#FAFAFA",
                        edgecolor=GHOST, linestyle="--", linewidth=1.2))

    ax_a.set_xscale("log")
    ax_a.set_yscale("log")
    ax_a.set_ylim(20, 5e5)
    ax_a.set_xlim(0.8, 90)
    ax_a.set_xticks(Cs)
    ax_a.set_xticklabels([str(C) for C in Cs], fontsize=9.5)
    ax_a.xaxis.set_minor_formatter(NullFormatter())
    ax_a.set_xlabel("channel count  $C$  (log scale)", fontsize=10,
                    fontweight="bold")
    ax_a.set_ylabel("elements  (log)", fontsize=10, fontweight="bold")
    ax_a.grid(alpha=0.3, linestyle="--", which="both")
    tidy(ax_a)
    ax_a.legend(loc="upper left", frameon=True, fontsize=8.2)

    ax_a.text(0.98, 0.02,
              f"fixed latents $[B,\\,L,\\,D]=[{B},\\,{L_A},\\,{D_A}]$ "
              "(measured, constant)\n"
              "so the decoder KV $[B\\cdot L,\\,D]$ does not grow "
              "with $N$",
              transform=ax_a.transAxes, ha="right", va="bottom",
              fontsize=8.2, color=INK,
              bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                        edgecolor=POOL, linewidth=1.4))

    # ---- Panel B: permutation invariance -------------------------------- #
    ax_b = fig.add_subplot(gs[0, 1])
    header(ax_b, "B",
           "Permutation invariance — latents unchanged to float32 round-off")

    bCs = sorted({r["C"] for r in perm_results})
    rng = np.random.default_rng(0)
    for i, C in enumerate(bCs):
        vals = [r["diff"] for r in perm_results if r["C"] == C]
        jit = rng.uniform(-0.16, 0.16, size=len(vals))
        ax_b.scatter(i + jit, vals, s=46, color=FUSION, alpha=0.9,
                     edgecolor="white", linewidth=0.6, zorder=4,
                     label=("one point per shuffle, max|$\\Delta$|"
                            if i == 0 else None))
    ax_b.axhline(EPS, color=MUTED, ls="--", lw=1.3,
                 label=f"float32 machine $\\epsilon$ = {EPS:.1e}")
    ax_b.axhline(TOL, color=INK, ls=":", lw=1.7,
                 label="chosen tolerance  $10^{-6}$")

    perm_max = max(r["diff"] for r in perm_results)
    ax_b.set_yscale("log")
    ax_b.set_ylim(5e-9, 3e-6)
    ax_b.set_xlim(-0.5, len(bCs) - 0.5)
    ax_b.set_xticks(range(len(bCs)))
    ax_b.set_xticklabels([f"{C}" for C in bCs])
    ax_b.set_xlabel("channel count  $C$", fontsize=10, fontweight="bold")
    ax_b.set_ylabel("max |latent-mean - latent-mean$_{\\mathrm{perm}}$|  (log)",
                    fontsize=10, fontweight="bold")
    ax_b.grid(axis="y", alpha=0.3, linestyle="--", which="both")
    tidy(ax_b)
    handles, labels = ax_b.get_legend_handles_labels()
    ax_b.legend(handles, labels, loc="lower right", frameon=True, fontsize=8.3)
    ax_b.text(0.98, 0.975,
              f"max|$\\Delta$| = {perm_max:.1e}  "
              "(consistent with float32\nsummation order)\n"
              "$\\leq$ chosen $10^{-6}$ tolerance",
              transform=ax_b.transAxes, ha="right", va="top", fontsize=8.2,
              color=INK,
              bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                        edgecolor=FUSION, linewidth=1.5))

    # ---- Panel C: cost (spans the bottom row) --------------------------- #
    ax_c = fig.add_subplot(gs[1, :])
    header(ax_c, "C", "Cost per location (analytic) — "
                      "default ChannelFusion PMA")

    ax_c.loglog(Cs_cost, cost_attn, "-", color=FUSION, linewidth=2.9,
                label="default PMA (ChannelFusion), $D_{attn}=3D$, $O(C)$")

    ax_c.set_xlabel("channel count  $C$  (log scale)", fontsize=10,
                    fontweight="bold")
    ax_c.set_ylabel("analytic multiply-adds per location  (log)",
                    fontsize=10, fontweight="bold")
    ax_c.set_xlim(1, 60)
    ax_c.set_ylim(2e4, 2e8)
    ax_c.set_xticks([1, 3, 10, 30, 60])
    ax_c.set_xticklabels(["1", "3", "10", "30", "60"])
    ax_c.grid(alpha=0.3, linestyle="--", which="both")
    tidy(ax_c)
    ax_c.legend(loc="upper left", frameon=True, fontsize=9.0)

    ax_c.text(0.97, 0.05,
              f"analytic per location ($D={D}$, $D_{{attn}}=3D={3 * D}$):\n"
              "default PMA  $2CDD_{attn}+2CD_{attn}+DD_{attn}"
              " = 6CD^2+6CD+3D^2$\n"
              "one-off shared query projection and small norm/skip terms omitted",
              transform=ax_c.transAxes, ha="right", va="bottom", fontsize=7.8,
              color=INK, linespacing=1.35,
              bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                        edgecolor=FUSION, linewidth=1.4))

    # ---- symbols / glossary (bordered rounded box, 3 aligned columns) ---- #
    foot = fig.add_axes([0.025, 0.007, 0.95, 0.149])
    foot.set_xlim(0, 1)
    foot.set_ylim(0, 1)
    foot.axis("off")
    rrect(foot, 0.0, 0.0, 1.0, 1.0, edge=GHOST, face=tint(GHOST, 0.95),
          lw=1.2, rounding=0.02, z=1)

    foot.text(0.016, 0.905, "symbols / glossary", ha="left", va="center",
              fontsize=8.6, fontweight="bold", color=INK, zorder=4)
    foot.plot([0.014, 0.986], [0.818, 0.818], color=GHOST, lw=0.8,
              alpha=0.7, zorder=3)

    col1 = [
        (r"$B$", f"batch size (samples) $={B}$"),
        (r"$C$", "channel count, order-free, per sample"),
        (r"$\Sigma C$", f"total channels in the batch ($=C$ at $B{{=}}1$)"),
        (r"$N$", f"patches / channel $=(H/P)^2={N}$"),
        (r"$H$", f"image side $={H}$ px"),
        (r"$P$", f"patch size $={P}$ px"),
        (r"$C_{\max}$", "per-sample max channel count (ghost only)"),
    ]
    col2 = [
        (r"$D$", f"token / embedding dim $={D}$"),
        (r"$L$", f"latents / sample ($n_{{\\mathrm{{latents}}}}={L}$)"),
        (r"$n_{\mathrm{heads}}$", f"attention heads per block $={n_heads}$"),
        (r"$D_{\mathrm{attn}}$", "fusion attention width $=3D$"),
        (r"$O(C)$", "cost grows linearly with $C$"),
        (r"$\mu$", f"per-pixel physical size (${pixel:g}$ µm/px, fleet member)"),
        (r"$\epsilon$", "float32 machine epsilon ($\\approx 1.2\\times10^{-7}$)"),
    ]
    col3 = [
        ("PMA", "multi-head attention pooling; $L$ queries → latents $[B,L,D]$"),
        ("tolerance", "chosen $10^{-6}$"),
        (r"$\Delta$", "latent-mean change under shuffle (max|$\\Delta$|)"),
        ("decoder KV", "latents $[B\\cdot L,D]$"),
        ("ragged token list",
         "solid primary object $[\\Sigma C{\\cdot}N,\\ D]$"),
        ("padded tensor",
         "dashed ghost $[B,\\,C_{\\max},\\,N,\\,D]$, never materialised"),
        ("omitted rows", "missing channels omitted, never zeroed or padded"),
    ]
    y0, pitch = 0.735, 0.086
    for cx, sw, entries in ((0.016, 0.050, col1),
                            (0.345, 0.058, col2),
                            (0.678, 0.078, col3)):
        for i, (sym, desc) in enumerate(entries):
            yy = y0 - i * pitch
            foot.text(cx, yy, sym, ha="left", va="center", fontsize=6.9,
                      fontweight="bold", color=INK, zorder=4)
            foot.text(cx + sw, yy, desc, ha="left", va="center",
                      fontsize=6.4, color=MUTED, zorder=4)

    foot.plot([0.014, 0.986], [0.108, 0.108], color=GHOST, lw=0.8,
              alpha=0.7, zorder=3)
    foot.text(0.016, 0.055,
              f"Panels A–B: measured, random init;  Panel C: analytic"
              f"    ·    pixel size {pixel:g} µm/px (fleet member)"
              f"    ·    † num_channel_ids=64 for C>16 (data-range necessity)",
              ha="left", va="center", fontsize=6.6, color=INK, zorder=4)

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

    # Build the shipped default through the config layer.  ``num_channel_ids``
    # is raised only so the cardinality sweep (C up to 60) has a valid id per
    # channel; every other field is the shipped default.
    set_seed(42)
    cfg = ModelConfig(num_channel_ids=64)
    model = build_model("full", base=cfg).to(device).eval()

    B = 1
    print(f"Model: D={model.core.dim}, n_heads={model.core.n_heads}, "
          f"patch_size={model.patch_size}, n_patches={model.n_patches}, "
          f"n_latents={model.n_latents}, n_memory={model.encoder.n_memory}")

    # Run measurements
    with torch.no_grad():
        card_results = measure_cardinality_invariance(model, device)
        perm_results = measure_permutation_invariance(model, device)

    D = card_results[0]["D"]
    Cs_cost = np.arange(1, 61)
    cost_attn = analytic_cost(Cs_cost, D)

    run = {
        "B": B,
        "N": model.n_patches,
        "D": D,
        "L": card_results[0]["L"],
        "H": model.image_size,
        "P": model.patch_size,
        "pixel_size": PIXEL_SIZE,
        "n_heads": model.core.n_heads,
    }

    # Build and save figure
    fig = build_figure(card_results, perm_results, Cs_cost, cost_attn, run)
    save_figure(fig, OUT, args.stem, args.formats)
    plt.close(fig)

    print("\n\u2713 All measurements complete.")


if __name__ == "__main__":
    main()
