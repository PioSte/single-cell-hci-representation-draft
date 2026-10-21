"""L6 -- MAE objective: encoder-decoder asymmetry and channel-aware reconstruction.

Shows the two-branch objective with:
* Encoder branch: sees only visible tokens [Kf, D] + memory [M, D] → enc_all
* Decoder branch: target queries conditioned on channel identity → predictions
* One shared decoder for all channels (virtual staining capability)
* Asymmetry: Kf (visible) vs Kt (masked), Kf + Kt = ΣC·N

Usage::

    uv run --extra viz python viz/level6_objective.py
    uv run --extra viz python viz/level6_objective.py --formats pdf svg png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from figure_style import (  # noqa: E402
    BLUE, GREEN, ORANGE, RED, PURPLE, VIOLET, INK, MUTED, CHANNEL_COLORS,
    tint, rrect, box, arrow, edge_label, sq_bracket, token_rows,
    install_rcparams, save_figure,
)

OUT = Path(__file__).resolve().parent / "out"


# --------------------------------------------------------------------------- #
# Array stacks
# --------------------------------------------------------------------------- #
def draw_encoder_input(ax, x=8.0, y0=48.0):
    """Visible tokens [Kf, D] after DCP omission."""
    w, h, pitch = 11.0, 2.9, 3.95
    specs = [
        (BLUE, False),
        (BLUE, False),
        (GREEN, False),
        (ORANGE, True),  # omitted channel-patch
        (ORANGE, False),
        (PURPLE, False),
    ]
    top = token_rows(ax, x, y0, w, h, pitch, specs)
    sq_bracket(ax, x + w + 1.0, y0, top)
    ax.text(x + w + 1.6, (y0 + top) / 2, "$K_f$ visible", ha="left", va="center",
            fontsize=7.2, color=BLUE, rotation=90)
    ax.text(x + w / 2, top + 2.4, "$[K_f, D]$", ha="center", va="center",
            fontsize=7.8, color=INK, fontweight="bold")
    # Return both top and the omitted token y position
    omitted_y = y0 + 3 * pitch + h / 2
    return top, omitted_y


def draw_memory(ax, x=8.0, y0=28.0):
    """Memory tokens [B·M, D]."""
    w, h, pitch = 11.0, 2.9, 3.95
    specs = [(ORANGE, False), (ORANGE, False)]
    top = token_rows(ax, x, y0, w, h, pitch, specs)
    sq_bracket(ax, x + w + 1.0, y0, top)
    ax.text(x + w + 1.6, (y0 + top) / 2, "$B{\\cdot}M$", ha="left", va="center",
            fontsize=7.2, color=ORANGE, rotation=90)
    return top


def draw_enc_all(ax, x=52.0, y0=28.0):
    """Encoder output [B·M + Kf, D]."""
    w, h, pitch = 11.0, 2.9, 3.95
    specs = [
        (ORANGE, False),
        (ORANGE, False),
        (BLUE, False),
        (BLUE, False),
        (BLUE, False),
        (GREEN, False),
        (PURPLE, False),
        (ORANGE, False),
    ]
    top = token_rows(ax, x, y0, w, h, pitch, specs)
    sq_bracket(ax, x + w + 1.0, y0, top)
    ax.text(x + w / 2, top + 2.4, "$[B{\\cdot}M + K_f, D]$", ha="center",
            va="center", fontsize=7.8, color=INK, fontweight="bold")
    ax.text(x + w / 2, y0 - 3.0, "encoder memory", ha="center", va="center",
            fontsize=7.0, color=INK)
    # annotate the two sections
    ax.text(x - 1.0, 32.0, "mem", ha="right", va="center", fontsize=6.4,
            color=ORANGE)
    ax.text(x - 1.0, 47.0, "vis", ha="right", va="center", fontsize=6.4,
            color=BLUE)
    return top


def draw_target_queries(ax, x=90.0, y0=48.0):
    """Decoder target queries [Kt, D]."""
    w, h, pitch = 12.0, 2.9, 3.95
    specs = [
        (ORANGE, False),
        (VIOLET, False),
        (RED, False),
    ]
    top = token_rows(ax, x, y0, w, h, pitch, specs)
    sq_bracket(ax, x + w + 1.0, y0, top)
    ax.text(x + w + 1.6, (y0 + top) / 2, "$K_t$ target", ha="left", va="center",
            fontsize=7.2, color=VIOLET, rotation=90)
    ax.text(x + w / 2, top + 2.4, "$[K_t, D]$ queries", ha="center", va="center",
            fontsize=7.8, color=INK, fontweight="bold")
    # annotation: q = mask_token + spatial_pos + proj(z_c)
    ax.text(x + w / 2, y0 - 2.5,
            r"$q = \mathrm{mask} + \mathrm{pos}_n + W z_c$",
            ha="center", va="center", fontsize=6.8, color=MUTED, style="italic")
    return top


def draw_predictions(ax, x=130.0, y0=48.0):
    """Decoder predictions [Kt, P²]."""
    w, h, pitch = 14.0, 2.9, 3.95
    specs = [
        (ORANGE, False),
        (VIOLET, False),
        (RED, False),
    ]
    top = token_rows(ax, x, y0, w, h, pitch, specs)
    sq_bracket(ax, x + w + 1.0, y0, top)
    ax.text(x + w / 2, top + 2.4, "$[K_t, P^2]$ predictions", ha="center", va="center",
            fontsize=7.8, color=INK, fontweight="bold")
    return top


def draw_targets(ax, x=130.0, y0=28.0):
    """Target pixels [Kt, P²]."""
    w, h, pitch = 14.0, 2.9, 3.95
    specs = [
        (ORANGE, False),
        (VIOLET, False),
        (RED, False),
    ]
    top = token_rows(ax, x, y0, w, h, pitch, specs)
    ax.text(x + w / 2, y0 - 2.0, "$[K_t, P^2]$ targets", ha="center", va="center",
            fontsize=7.8, color=INK, fontweight="bold")
    ax.text(x + w / 2, y0 - 4.5, "(raw masked pixels)", ha="center", va="center",
            fontsize=6.4, color=MUTED, style="italic")
    return top


# --------------------------------------------------------------------------- #
# The figure
# --------------------------------------------------------------------------- #
def build() -> plt.Figure:
    fig = plt.figure(figsize=(13.0, 8.0))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_xlim(0, 156)
    ax.set_ylim(0, 88)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    # ---- title ------------------------------------------------------------ #
    ax.text(2, 85.2, "L6", ha="left", va="center", fontsize=15,
            fontweight="bold", color=INK)
    ax.text(6.5, 85.2,
            "MAE objective: encoder-decoder asymmetry & channel-aware reconstruction",
            ha="left", va="center", fontsize=12.2, fontweight="bold", color=INK)

    # ---- two-branch structure --------------------------------------------- #
    # ENCODER BRANCH (top half)
    rrect(ax, 4.0, 42.0, 62.0, 40.0, edge=BLUE, face=tint(BLUE, 0.96),
          lw=1.5, dashed=True, rounding=1.0, z=1)
    ax.text(5.0, 80.5, "Encoder branch (sees only visible tokens)",
            ha="left", va="center", fontsize=9.5, fontweight="bold", color=BLUE)
    
    # DECODER BRANCH (overlapping, slightly lower)
    rrect(ax, 84.0, 42.0, 68.0, 40.0, edge=VIOLET, face=tint(VIOLET, 0.96),
          lw=1.5, dashed=True, rounding=1.0, z=1)
    ax.text(85.0, 80.5, "Decoder branch (channel-conditioned reconstruction)",
            ha="left", va="center", fontsize=9.5, fontweight="bold", color=VIOLET)

    # ---- encoder pathway -------------------------------------------------- #
    enc_top, omitted_y = draw_encoder_input(ax, x=8.0, y0=48.0)
    mem_top = draw_memory(ax, x=8.0, y0=28.0)
    
    # DCP masker annotation - moved left to avoid overlap with bracket
    ax.text(3.5, omitted_y, "DCP\nomission", ha="center", va="center",
            fontsize=6.8, color=GREEN, fontweight="bold", linespacing=1.1)
    ax.text(3.5, omitted_y - 2.5, "(never\nmasked)", ha="center",
            va="center", fontsize=6.0, color=GREEN, style="italic", linespacing=1.0)

    # Encoder module
    box(ax, 26.0, 50.0, 20.0, 18.0, "Spatial encoder",
        "grouped self-attn\n$\\times L$ blocks", edge=PURPLE, sub_size=6.8)
    
    # Arrows: visible tokens and memory both feed encoder
    arrow(ax, (20.0, 58.0), (26.0, 58.0), color=BLUE, lw=1.5)
    arrow(ax, (20.0, 32.0), (26.0, 32.0), color=ORANGE, lw=1.5)
    
    # Merge point - both inputs converge
    ax.plot([24.0, 24.0, 26.0], [32.0, 58.0, 58.0], color=PURPLE, lw=1.5, zorder=5)
    ax.text(22.5, 45.0, "concat", ha="center", va="center", fontsize=6.0,
            color=PURPLE, style="italic", rotation=90)
    
    draw_enc_all(ax, x=52.0, y0=28.0)
    arrow(ax, (46.0, 55.0), (52.0, 55.0), color=PURPLE, lw=1.5)

    # ---- decoder pathway -------------------------------------------------- #
    tgt_top = draw_target_queries(ax, x=90.0, y0=48.0)
    
    # Decoder module
    box(ax, 108.0, 50.0, 16.0, 18.0, "Decoder",
        "cross-attn\n$\\times D$ blocks", edge=VIOLET, sub_size=6.8)
    
    # queries → decoder
    arrow(ax, (103.0, 57.0), (108.0, 57.0), color=VIOLET, lw=1.5)
    
    # enc_all → decoder (cross-attention)
    arrow(ax, (63.0, 50.0), (63.0, 46.0), color=PURPLE, lw=1.1, dashed=True)
    ax.plot([63.0, 116.0, 116.0], [46.0, 46.0, 50.0], color=PURPLE, lw=1.1,
            linestyle="--", zorder=4)
    edge_label(ax, 88.0, 43.2, "encoder memory (keys, values)", color=PURPLE,
               size=6.4)
    
    # decoder head
    box(ax, 129.0, 63.0, 17.0, 8.0, "Head",
        "$\\mathrm{Linear}(D, P^2)$", edge=VIOLET, title_size=7.6, sub_size=6.4)
    arrow(ax, (116.0, 67.0), (129.0, 67.0), color=VIOLET, lw=1.5)
    
    pred_top = draw_predictions(ax, x=130.0, y0=48.0)
    arrow(ax, (137.0, 63.0), (137.0, pred_top), color=VIOLET, lw=1.5)
    
    # ---- loss ------------------------------------------------------------ #
    targ_top = draw_targets(ax, x=130.0, y0=28.0)
    
    # Target path from the bottom
    ax.plot([4.0, 4.0, 125.0, 125.0], [82.0, 18.0, 18.0, 28.0],
            color=MUTED, lw=1.2, linestyle="--", zorder=2)
    ax.text(64.0, 20.0, "target = raw pixels of masked channel-patches",
            ha="center", va="center", fontsize=7.2, color=MUTED, style="italic")
    
    # Loss box
    box(ax, 129.0, 7.0, 17.0, 9.0, "Loss",
        "normalized patch MSE", edge=RED, title_size=7.6, sub_size=6.4)
    
    # Clean routing: predictions route right of the targets stack, targets left.
    ax.plot([144.0, 149.0, 149.0], [53.0, 53.0, 18.0], color=VIOLET, lw=1.3,
            zorder=4)
    ax.text(150.2, 34.0, "pred", ha="left", va="center", fontsize=6.6,
            color=VIOLET)

    ax.plot([130.0, 125.0, 125.0], [29.0, 29.0, 18.0], color=MUTED, lw=1.3,
            linestyle="--", zorder=4)
    ax.text(123.5, 24.0, "tgt", ha="right", va="center", fontsize=6.6,
            color=MUTED)

    # Both converge below the Loss box and enter it from the bottom-centre.
    ax.plot([125.0, 149.0], [18.0, 18.0], color=RED, lw=1.3, zorder=4)
    arrow(ax, (137.0, 18.0), (137.0, 16.0), color=RED, lw=1.5)
    
    # ---- key points ------------------------------------------------------- #
    # Asymmetry
    rrect(ax, 6.0, 2.0, 58.0, 5.0, edge=BLUE, face=tint(BLUE, 0.88), lw=1.3,
          rounding=0.5, z=2)
    ax.text(35.0, 4.5,
            r"Asymmetry: encoder sees $K_f$ (visible), decoder reconstructs $K_t$ (masked); $K_f + K_t = \Sigma C \cdot N$",
            ha="center", va="center", fontsize=7.6, color=INK, fontweight="bold")
    
    # One shared decoder
    rrect(ax, 68.0, 2.0, 84.0, 5.0, edge=VIOLET, face=tint(VIOLET, 0.88),
          lw=1.3, rounding=0.5, z=2)
    ax.text(110.0, 4.5,
            r"One shared decoder for ALL channels — conditioned on target identity $z_c$ (virtual staining)",
            ha="center", va="center", fontsize=7.6, color=INK, fontweight="bold")
    
    # Loss detail
    ax.text(78.0, 10.0,
            "Loss: per-patch standardize target, then MSE (no per-channel norm)",
            ha="center", va="center", fontsize=6.8, color=RED, style="italic")

    return fig


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--formats", nargs="+", default=["pdf", "svg", "png"])
    ap.add_argument("--stem", default="level6_objective")
    args = ap.parse_args()

    install_rcparams()
    fig = build()
    save_figure(fig, OUT, args.stem, args.formats)
    plt.close(fig)


if __name__ == "__main__":
    main()
