"""L0 -- positioning / comparison: padding-free vs. padded approaches.

Three-column comparison in the ChAda-ViT Fig 13 idiom:
(a) Padded dense ViT -- stack channels into [B, C_max, N, D], explicit pad rows + -inf mask
(b) Dense cross-channel attention (ChannelViT / ChAda-ViT) -- padded channel axis, additive channel embedding
(c) Ours - padding-free channel fusion -- ragged [ΣC·N, D] -> per-(b,n) fusion -> [B,N,D]; NO pad/mask

Each column has an operation equation and a one-line takeaway at the bottom.

Usage::

    uv run --extra viz python viz/level0_positioning.py
    uv run --extra viz python viz/level0_positioning.py --formats pdf svg
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from figure_style import (  # noqa: E402
    BLUE, GREEN, ORANGE, RED, PURPLE, VIOLET, INK, MUTED, CHANNEL_COLORS,
    tint, rrect, box, arrow, edge_label, sq_bracket, panel_header,
    token_rows, install_rcparams, save_figure,
)

OUT = Path(__file__).resolve().parent / "out"


# --------------------------------------------------------------------------- #
# Shared input at the top
# --------------------------------------------------------------------------- #
def draw_shared_input(ax, cx):
    """Multi-marker input set (ragged)."""
    cw, pitch = 4.8, 4.2
    n_markers = 3
    x0 = cx - (n_markers * cw + (n_markers - 1) * 1.0) / 2
    for i in range(n_markers):
        c = CHANNEL_COLORS[i]
        xx = x0 + i * (cw + 1.0)
        rrect(ax, xx, 61.0, cw, cw, edge=c, face=tint(c, 0.72), lw=1.4)
        # faint image squiggle
        ax.plot([xx + 0.8, xx + cw - 0.8], [61.8, 64.2], color=c, lw=0.9, alpha=0.65, zorder=4)
        ax.plot([xx + 0.8, xx + cw - 0.8], [64.2, 61.8], color=c, lw=0.9, alpha=0.4, zorder=4)
    ax.text(cx, 68.0, "$C_b{\\in}[1..5]$ markers · order-free", ha="center", va="center",
            fontsize=7.8, color=MUTED)


# --------------------------------------------------------------------------- #
# Column (a): Padded dense ViT
# --------------------------------------------------------------------------- #
def draw_column_a(ax, x0=8.0):
    """Padded dense ViT -- explicit padding + -inf mask."""
    panel_header(ax, x0, 72.2, "(a)", "Padded dense ViT", color=INK)
    
    # Input arrows
    draw_shared_input(ax, x0 + 15.0)
    arrow(ax, (x0 + 15.0, 61.0), (x0 + 15.0, 56.0), color=MUTED)
    edge_label(ax, x0 + 8.5, 58.5, "stack & pad")
    
    # Padded tensor [B, C_max, N, D]
    cx, cy = x0 + 15.0, 46.0
    cell, pitch = 2.4, 2.7
    C_max, N_cols = 5, 3
    # Draw C_max rows, N columns
    for c in range(C_max):
        col = CHANNEL_COLORS[c] if c < 3 else MUTED
        omitted = c >= 3  # pad rows
        for n in range(N_cols):
            xx = cx - (N_cols * pitch) / 2 + n * pitch
            yy = cy + c * pitch
            rrect(ax, xx, yy, cell, cell, edge=col,
                  face=("#FFFFFF" if omitted else tint(col, 0.8)),
                  lw=1.0, dashed=omitted, rounding=0.25, z=3)
    
    # Labels
    ax.text(cx - N_cols * pitch / 2 - 1.8, cy + C_max * pitch / 2,
            "$C_{\\max}$", ha="right", va="center", fontsize=7.2, color=MUTED)
    ax.text(cx + 2.5, cy + C_max * pitch + 1.2, "$N$ patches", ha="left",
            va="center", fontsize=6.9, color=MUTED)
    ax.text(x0 + 25.5, cy - 6.0 + 1.75, "$[B, C_{\\max}, N, D]$", ha="left",
            va="center", fontsize=7.0, color=INK)
    
    # Pad row indicator
    ax.text(cx + N_cols * pitch / 2 + 1.8, cy + 4.0 * pitch, "pad rows",
            ha="left", va="center", fontsize=6.2, color=RED, style="italic")
    
    # Attention mask
    ymask = cy - 6.0
    box(ax, x0 + 6.0, ymask, 18.0, 3.5, "-inf mask on pad", "$O(C_{\\max}^2 N^2)$ memory",
        edge=RED, title_size=7.0, sub_size=6.0)
    arrow(ax, (cx, cy - 0.2), (cx, ymask + 3.5), color=RED, lw=1.2)
    
    # Transformer
    ytrans = ymask - 6.0
    box(ax, x0 + 6.0, ytrans, 18.0, 4.8, "Transformer", "all tokens", edge=PURPLE,
        title_size=7.6, sub_size=6.4)
    arrow(ax, (cx, ymask), (cx, ytrans + 4.8), lw=1.3)
    
    # Operation equation
    eq_y = ytrans - 3.5
    ax.text(cx, eq_y, r"$\mathrm{softmax}\left(\frac{QK^T}{\sqrt{d}} + M\right)V$",
            ha="center", va="center", fontsize=7.4, color=INK)
    
    # Takeaway
    takeaway_y = eq_y - 4.0
    rrect(ax, x0 + 2.0, takeaway_y - 2.0, 26.0, 4.5, edge=MUTED, face=tint(MUTED, 0.92),
          lw=1.0, rounding=0.5, z=2)
    ax.text(cx, takeaway_y, "Waste: $O(B \\cdot C_{\\max} \\cdot N \\cdot D)$ memory\n"
            "for pad tokens + explicit mask",
            ha="center", va="center", fontsize=6.6, color=MUTED, linespacing=1.3)


# --------------------------------------------------------------------------- #
# Column (b): Dense cross-channel attention
# --------------------------------------------------------------------------- #
def draw_column_b(ax, x0=50.0):
    """Dense cross-channel attention (ChannelViT / ChAda-ViT)."""
    panel_header(ax, x0, 72.2, "(b)", "Dense cross-channel attn", color=INK)
    
    # Input arrows
    draw_shared_input(ax, x0 + 15.0)
    arrow(ax, (x0 + 15.0, 61.0), (x0 + 15.0, 56.0), color=MUTED)
    edge_label(ax, x0 + 8.5, 58.5, "pad + embed $z_c$")
    
    # Padded tensor with channel embedding
    cx, cy = x0 + 15.0, 46.0
    cell, pitch = 2.4, 2.7
    C_max, N_cols = 5, 3
    for c in range(C_max):
        col = CHANNEL_COLORS[c] if c < 3 else MUTED
        omitted = c >= 3
        for n in range(N_cols):
            xx = cx - (N_cols * pitch) / 2 + n * pitch
            yy = cy + c * pitch
            rrect(ax, xx, yy, cell, cell, edge=col,
                  face=("#FFFFFF" if omitted else tint(col, 0.8)),
                  lw=1.0, dashed=omitted, rounding=0.25, z=3)
    
    # Labels
    ax.text(cx - N_cols * pitch / 2 - 1.8, cy + C_max * pitch / 2,
            "$C_{\\max}$", ha="right", va="center", fontsize=7.2, color=MUTED)
    ax.text(cx + 2.5, cy + C_max * pitch + 1.2, "$N$", ha="left",
            va="center", fontsize=6.9, color=MUTED)
    ax.text(x0 + 25.5, cy - 6.0 + 1.75, "$[B, C_{\\max}, N, D]$", ha="left",
            va="center", fontsize=7.0, color=INK)
    ax.text(cx + N_cols * pitch / 2 + 1.5, cy + 4.0 * pitch, "pad + $z_c$",
            ha="left", va="center", fontsize=6.2, color=RED, style="italic")
    
    # Channel-aware attention
    ymask = cy - 6.0
    box(ax, x0 + 6.0, ymask, 18.0, 3.5, "Channel mask", "additive $z_c$ embedding",
        edge=ORANGE, title_size=7.0, sub_size=6.0)
    arrow(ax, (cx, cy - 0.2), (cx, ymask + 3.5), color=ORANGE, lw=1.2)
    
    # Transformer
    ytrans = ymask - 6.0
    box(ax, x0 + 6.0, ytrans, 18.0, 4.8, "Channel transformer", "inter-channel attn",
        edge=PURPLE, title_size=7.6, sub_size=6.4)
    arrow(ax, (cx, ymask), (cx, ytrans + 4.8), lw=1.3)
    
    # Operation equation
    eq_y = ytrans - 3.5
    ax.text(cx, eq_y, r"$\mathrm{Attn}(Q, K + z_c, V)$",
            ha="center", va="center", fontsize=7.4, color=INK)
    
    # Takeaway
    takeaway_y = eq_y - 4.0
    rrect(ax, x0 + 2.0, takeaway_y - 2.0, 26.0, 4.5, edge=MUTED, face=tint(MUTED, 0.92),
          lw=1.0, rounding=0.5, z=2)
    ax.text(cx, takeaway_y, "Still $O(C_{\\max})$ overhead;\n"
            "additive identity, explicit mask",
            ha="center", va="center", fontsize=6.6, color=MUTED, linespacing=1.3)


# --------------------------------------------------------------------------- #
# Column (c): Ours - padding-free channel fusion
# --------------------------------------------------------------------------- #
def draw_column_c(ax, x0=92.0):
    """Ours - padding-free channel fusion (highlighted)."""
    # Accent box around entire column
    rrect(ax, x0 - 1.5, 5.0, 32.0, 69.0, edge=GREEN, face=tint(GREEN, 0.97),
          lw=2.2, rounding=1.0, z=0)
    
    panel_header(ax, x0, 72.2, "(c)", "Ours: padding-free fusion", color=GREEN)
    
    # Input arrows
    draw_shared_input(ax, x0 + 15.0)
    arrow(ax, (x0 + 15.0, 61.0), (x0 + 15.0, 55.5), color=GREEN, lw=1.4)
    edge_label(ax, x0 + 7.5, 58.2, "ragged list", color=GREEN, ha="right")
    
    # Ragged token list [ΣC·N, D] - moved down to avoid input overlap
    cx = x0 + 15.0
    x_rag, w_rag, h_rag, pitch_rag, y0_rag = cx - 5.5, 11.0, 2.2, 2.8, 42.5
    rows = 6
    for i in range(rows):
        yy = y0_rag + i * pitch_rag
        c = CHANNEL_COLORS[i % 3]
        rrect(ax, x_rag, yy, w_rag, h_rag, edge=c, face=tint(c, 0.8),
              lw=1.1, rounding=0.25, z=3)
    top_rag = y0_rag + (rows - 1) * pitch_rag + h_rag
    sq_bracket(ax, x_rag + w_rag + 0.6, y0_rag, top_rag)
    ax.text(x_rag + w_rag + 1.2, (y0_rag + top_rag) / 2,
            "$[\\Sigma C{\\cdot}N, D]$", ha="left", va="center", fontsize=7.2, color=INK)
    
    # Channel fusion - moved down to maintain spacing
    yfuse = 33.5
    box(ax, x0 + 3.0, yfuse, 24.0, 5.5, "Channel fusion",
        "per-$(b,n)$ pool · $O(1)$ output size", edge=RED, title_size=7.6, sub_size=6.2)
    arrow(ax, (cx, y0_rag - 0.5), (cx, yfuse + 5.5), color=RED, lw=1.4)
    
    # Joint embeddings [B, N, D]
    yjoint = 24.5
    cell_j, pitch_j = 2.2, 2.5
    B_cols, N_rows = 3, 3
    x_j0 = cx - (B_cols * pitch_j) / 2
    for b in range(B_cols):
        for n in range(N_rows):
            rrect(ax, x_j0 + b * pitch_j, yjoint + n * pitch_j, cell_j, cell_j,
                  edge=BLUE, face=tint(BLUE, 0.75), lw=1.0, rounding=0.25, z=3)
    ax.text(cx, yjoint + N_rows * pitch_j + 1.0, "$B$", ha="center", va="center",
            fontsize=6.8, color=MUTED)
    ax.text(x_j0 - 1.2, yjoint + N_rows * pitch_j / 2, "$N$", ha="right",
            va="center", fontsize=6.8, color=MUTED)
    ax.text(cx, yjoint - 2.0, "$[B, N, D]$", ha="center", va="center",
            fontsize=7.6, fontweight="bold", color=BLUE)

    # Ghost padded tensor -- beside the fused grid; it is never materialised.
    gx, gy, gw, gh = cx + 6.0, yjoint - 0.8, 9.0, 8.0
    rrect(ax, gx, gy, gw, gh, edge=MUTED, face="#FCFCFC", lw=1.3,
          dashed=True, rounding=0.5, z=2)
    ax.plot([gx + 1.0, gx + gw - 1.0], [gy + 1.0, gy + gh - 1.0],
            color=RED, lw=1.8, alpha=0.8, zorder=3)
    ax.text(gx + gw / 2, gy + gh - 1.0, "$[B, C_{\\max}, N, D]$",
            ha="center", va="center", fontsize=5.4, color=MUTED, style="italic",
            zorder=4)
    ax.text(gx + gw / 2, gy + 1.3, "never materialised",
            ha="center", va="center", fontsize=5.4, color=MUTED, style="italic",
            zorder=4)
    arrow(ax, (cx, yfuse), (cx, yjoint + N_rows * pitch_j + 1.2), lw=1.3)
    
    # Spatial transformer
    ytrans = 13.5
    box(ax, x0 + 3.0, ytrans, 24.0, 4.8, "Spatial transformer",
        "no padding, no mask", edge=PURPLE, title_size=7.6, sub_size=6.4)
    arrow(ax, (cx, yjoint - 2.0), (cx, ytrans + 4.8), lw=1.3)
    
    # Operation equation
    eq_y = ytrans - 3.5
    ax.text(cx, eq_y, r"$\mathrm{GroupedPool}(q, \{x_c\}_{c \in C_b})$",
            ha="center", va="center", fontsize=7.4, color=INK)
    
    # Takeaway
    takeaway_y = eq_y - 3.5
    rrect(ax, x0 + 0.5, takeaway_y - 2.0, 29.0, 4.5, edge=GREEN, face=tint(GREEN, 0.88),
          lw=1.5, rounding=0.5, z=2)
    ax.text(cx, takeaway_y, "No pad/mask overhead;\n"
            "$O(B{\\cdot}N{\\cdot}D)$ — independent of $C$",
            ha="center", va="center", fontsize=6.6, color=INK, fontweight="bold",
            linespacing=1.3)


# --------------------------------------------------------------------------- #
# The figure
# --------------------------------------------------------------------------- #
def build() -> plt.Figure:
    fig = plt.figure(figsize=(13, 8))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_xlim(0, 128)
    ax.set_ylim(0, 76)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    
    # Title
    ax.text(2.0, 74.0, "L0", ha="left", va="center", fontsize=15,
            fontweight="bold", color=INK)
    ax.text(6.5, 74.0,
            "Positioning: padding-free channel fusion vs. padded approaches",
            ha="left", va="center", fontsize=11.5, fontweight="bold", color=INK)
    
    # Three columns
    draw_column_a(ax, x0=8.0)
    draw_column_b(ax, x0=50.0)
    draw_column_c(ax, x0=92.0)
    
    return fig


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--formats", nargs="+", default=["pdf", "svg", "png"])
    ap.add_argument("--stem", default="level0_positioning")
    args = ap.parse_args()
    
    install_rcparams()
    
    fig = build()
    save_figure(fig, OUT, args.stem, args.formats)
    plt.close(fig)


if __name__ == "__main__":
    main()
