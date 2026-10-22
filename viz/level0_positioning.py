"""L0 -- positioning / comparison: padding-free vs. padded approaches.

Three-column comparison in the ChAda-ViT Fig 13 idiom:
(a) Padded dense ViT -- stack channels into [B, C_max, N, D], explicit pad rows + -inf mask
(b) Dense cross-channel attention (ChannelViT / ChAda-ViT) -- padded channel axis, additive channel embedding
(c) Ours - padding-free channel fusion -- ragged [ΣC·N, D] -> per-(b,n) fusion -> [B,N,D]; NO pad/mask

All three columns share the same scale and width; the stages are aligned at
identical y so the panels read across.  Panel (c) is emphasised only through
the green accent frame/tint, never through size.

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
    install_rcparams, save_figure,
)

OUT = Path(__file__).resolve().parent / "out"


# --------------------------------------------------------------------------- #
# Shared column geometry -- every panel uses the same y for every stage
# --------------------------------------------------------------------------- #
COL_W = 30.0                 # width of a column / of every module box
HEADER_Y = 72.0
INPUT_LABEL_Y = 67.6
INPUT_TILE_Y = 61.0          # bottom edge of the input tiles
TENSOR_TOP = 57.6            # top of the padded grid / ragged list
TENSOR_BOT = 46.2            # bottom of the padded grid / ragged list
CELL = 2.2                   # padded-grid cell size
PITCH = 2.3                  # padded-grid row pitch
C_MAX = 5
N_COLS = 3
FUSE_BOX_Y = 39.5            # mask / channel-fusion box
FUSE_BOX_H = 4.8
TRANS_BOX_Y = 23.5           # transformer box
TRANS_BOX_H = 5.0
EQ_Y = 19.5                  # operation equation
TAKEAWAY_Y = 12.0            # takeaway box
TAKEAWAY_H = 5.0

GRID_W = (N_COLS - 1) * PITCH + CELL          # padded-grid width
GRID_H = (C_MAX - 1) * PITCH + CELL            # padded-grid height


# --------------------------------------------------------------------------- #
# Shared input at the top
# --------------------------------------------------------------------------- #
def draw_shared_input(ax, cx):
    """Multi-marker input set (ragged)."""
    cw, pitch = 4.8, 5.8
    n_markers = 3
    x0 = cx - (n_markers * cw + (n_markers - 1) * 1.0) / 2
    for i in range(n_markers):
        c = CHANNEL_COLORS[i]
        xx = x0 + i * (cw + 1.0)
        rrect(ax, xx, INPUT_TILE_Y, cw, cw, edge=c, face=tint(c, 0.72), lw=1.4)
        # faint image squiggle
        ax.plot([xx + 0.8, xx + cw - 0.8], [INPUT_TILE_Y + 0.8, INPUT_TILE_Y + 4.0],
                color=c, lw=0.9, alpha=0.65, zorder=4)
        ax.plot([xx + 0.8, xx + cw - 0.8], [INPUT_TILE_Y + 4.0, INPUT_TILE_Y + 0.8],
                color=c, lw=0.9, alpha=0.4, zorder=4)
    ax.text(cx, INPUT_LABEL_Y, "$C_b{\\in}[1..5]$ markers · order-free",
            ha="center", va="center", fontsize=7.6, color=MUTED)


def draw_padded_grid(ax, cx):
    """The [B, C_max, N, D] padded tensor: real cells + dashed pad rows."""
    x_g0 = cx - GRID_W / 2
    y_g0 = TENSOR_BOT
    for c in range(C_MAX):
        col = CHANNEL_COLORS[c] if c < 3 else MUTED
        omitted = c >= 3  # pad rows -- absent rows, never zeroed
        for n in range(N_COLS):
            rrect(ax, x_g0 + n * PITCH, y_g0 + c * PITCH, CELL, CELL, edge=col,
                  face=("#FFFFFF" if omitted else tint(col, 0.8)),
                  lw=1.0, dashed=omitted, rounding=0.25, z=3)

    # C_max on the left edge
    ax.text(x_g0 - 1.6, y_g0 + (C_MAX - 1) * PITCH / 2,
            "$C_{\\max}$", ha="right", va="center", fontsize=7.2, color=MUTED)

    # N / shape / pad rows, all hanging off the right edge of the grid
    xr = x_g0 + GRID_W + 1.4
    ax.text(xr, TENSOR_TOP + 0.4, "$N$ patches", ha="left", va="center",
            fontsize=6.8, color=MUTED)
    ax.text(xr, y_g0 + (C_MAX - 1) * PITCH + CELL / 2, "[$B, C_{\\max}, N, D$]",
            ha="left", va="center", fontsize=6.8, color=INK)
    return x_g0, xr


# --------------------------------------------------------------------------- #
# Column (a): Padded dense ViT
# --------------------------------------------------------------------------- #
def draw_column_a(ax, x0):
    """Padded dense ViT -- explicit padding + -inf mask."""
    cx = x0 + COL_W / 2
    panel_header(ax, x0, HEADER_Y, "(a)", "Padded dense ViT", color=INK)

    draw_shared_input(ax, cx)
    arrow(ax, (cx, INPUT_TILE_Y), (cx, TENSOR_TOP + 0.3), color=MUTED)
    edge_label(ax, cx - 2.6, 59.6, "stack & pad", ha="right")

    _, xr = draw_padded_grid(ax, cx)
    # pad-row tag attached to the dashed rows, on the right
    ax.text(xr, TENSOR_BOT + 3.5 * PITCH, "pad rows", ha="left", va="center",
            fontsize=6.2, color=RED, style="italic")

    # Attention mask
    box(ax, x0, FUSE_BOX_Y, COL_W, FUSE_BOX_H, "-inf mask on pad",
        "$O(C_{\\max}^2 N^2)$ memory", edge=RED, title_size=7.0, sub_size=6.0)
    arrow(ax, (cx, TENSOR_BOT - 0.2), (cx, FUSE_BOX_Y + FUSE_BOX_H),
          color=RED, lw=1.2)

    # Transformer
    box(ax, x0, TRANS_BOX_Y, COL_W, TRANS_BOX_H, "Transformer", "all tokens",
        edge=PURPLE, title_size=7.6, sub_size=6.4)
    arrow(ax, (cx, FUSE_BOX_Y), (cx, TRANS_BOX_Y + TRANS_BOX_H), lw=1.3)

    # Operation equation
    ax.text(cx, EQ_Y, r"$\mathrm{softmax}\left(\frac{QK^T}{\sqrt{d}} + M\right)V$",
            ha="center", va="center", fontsize=7.4, color=INK)

    # Takeaway
    rrect(ax, x0, TAKEAWAY_Y, COL_W, TAKEAWAY_H, edge=MUTED,
          face=tint(MUTED, 0.92), lw=1.0, rounding=0.5, z=2)
    ax.text(cx, TAKEAWAY_Y + TAKEAWAY_H / 2,
            "Waste: $O(B \\cdot C_{\\max} \\cdot N \\cdot D)$ memory\n"
            "for pad tokens + explicit mask",
            ha="center", va="center", fontsize=6.6, color=MUTED, linespacing=1.3)


# --------------------------------------------------------------------------- #
# Column (b): Dense cross-channel attention
# --------------------------------------------------------------------------- #
def draw_column_b(ax, x0):
    """Dense cross-channel attention (ChannelViT / ChAda-ViT)."""
    cx = x0 + COL_W / 2
    panel_header(ax, x0, HEADER_Y, "(b)", "Dense cross-channel attn", color=INK)

    draw_shared_input(ax, cx)
    arrow(ax, (cx, INPUT_TILE_Y), (cx, TENSOR_TOP + 0.3), color=MUTED)
    edge_label(ax, cx - 2.6, 59.6, "pad + embed $z_c$", ha="right")

    _, xr = draw_padded_grid(ax, cx)
    # pad + z_c tag attached to the dashed pad rows, exactly as in (a)
    ax.text(xr, TENSOR_BOT + 3.5 * PITCH, "pad + $z_c$", ha="left", va="center",
            fontsize=6.2, color=RED, style="italic")

    # Channel-aware attention
    box(ax, x0, FUSE_BOX_Y, COL_W, FUSE_BOX_H, "Channel mask",
        "additive $z_c$ embedding", edge=ORANGE, title_size=7.0, sub_size=6.0)
    arrow(ax, (cx, TENSOR_BOT - 0.2), (cx, FUSE_BOX_Y + FUSE_BOX_H),
          color=ORANGE, lw=1.2)

    # Transformer
    box(ax, x0, TRANS_BOX_Y, COL_W, TRANS_BOX_H, "Channel transformer",
        "inter-channel attn", edge=PURPLE, title_size=7.6, sub_size=6.4)
    arrow(ax, (cx, FUSE_BOX_Y), (cx, TRANS_BOX_Y + TRANS_BOX_H), lw=1.3)

    # Operation equation
    ax.text(cx, EQ_Y, r"$\mathrm{Attn}(Q, K + z_c, V)$",
            ha="center", va="center", fontsize=7.4, color=INK)

    # Takeaway
    rrect(ax, x0, TAKEAWAY_Y, COL_W, TAKEAWAY_H, edge=MUTED,
          face=tint(MUTED, 0.92), lw=1.0, rounding=0.5, z=2)
    ax.text(cx, TAKEAWAY_Y + TAKEAWAY_H / 2,
            "Still $O(C_{\\max})$ overhead;\n"
            "additive identity, explicit mask",
            ha="center", va="center", fontsize=6.6, color=MUTED, linespacing=1.3)


# --------------------------------------------------------------------------- #
# Column (c): Ours - padding-free channel fusion
# --------------------------------------------------------------------------- #
def draw_column_c(ax, x0):
    """Ours - padding-free channel fusion (highlighted by accent only)."""
    cx = x0 + COL_W / 2
    # Green accent frame + tint: the only thing that sets (c) apart
    rrect(ax, x0 - 1.5, 10.5, COL_W + 3.0, 63.0, edge=GREEN,
          face=tint(GREEN, 0.97), lw=2.2, rounding=1.0, z=0)

    panel_header(ax, x0, HEADER_Y, "(c)", "Ours: padding-free fusion", color=GREEN)

    draw_shared_input(ax, cx)
    arrow(ax, (cx, INPUT_TILE_Y), (cx, TENSOR_TOP + 0.3), color=GREEN, lw=1.4)
    edge_label(ax, cx - 2.6, 59.6, "ragged list", color=GREEN, ha="right")

    # Ragged token list [Sigma C . N, D]: one block of 2 patch rows per channel
    w_rag = 9.5
    x_rag = cx - w_rag / 2
    h_row, pitch_rag, block_gap = 1.6, 1.8, 0.4
    yy = TENSOR_BOT
    for c in range(3):
        col = CHANNEL_COLORS[c]
        for _ in range(2):
            rrect(ax, x_rag, yy, w_rag, h_row, edge=col, face=tint(col, 0.8),
                  lw=1.1, rounding=0.25, z=3)
            yy += pitch_rag
        yy += block_gap
    top_rag = yy - pitch_rag - block_gap + h_row
    sq_bracket(ax, x_rag + w_rag + 0.6, TENSOR_BOT, top_rag)
    ax.text(x_rag + w_rag + 1.2, (TENSOR_BOT + top_rag) / 2,
            "$[\\Sigma C{\\cdot}N, D]$", ha="left", va="center",
            fontsize=7.0, color=INK)

    # Channel fusion
    box(ax, x0, FUSE_BOX_Y, COL_W, FUSE_BOX_H, "Channel fusion",
        "per-$(b,n)$ pool · $O(1)$ output size", edge=RED,
        title_size=7.6, sub_size=6.2)
    arrow(ax, (cx, TENSOR_BOT - 0.2), (cx, FUSE_BOX_Y + FUSE_BOX_H),
          color=RED, lw=1.4)

    # Joint embeddings [B, N, D] -- fill the gap between fusion and transformer
    cell_j, pitch_j = 1.8, 2.0
    B_cols, N_rows = 3, 3
    jw = (B_cols - 1) * pitch_j + cell_j
    y_j0 = 31.1
    x_j0 = cx - jw / 2
    for b in range(B_cols):
        for n in range(N_rows):
            rrect(ax, x_j0 + b * pitch_j, y_j0 + n * pitch_j, cell_j, cell_j,
                  edge=BLUE, face=tint(BLUE, 0.75), lw=1.0, rounding=0.25, z=3)
    ax.text(x_j0 + jw + 0.7, y_j0 + (N_rows - 1) * pitch_j + cell_j,
            "$B$", ha="left", va="center", fontsize=6.8, color=MUTED)
    ax.text(x_j0 - 0.7, y_j0 + ((N_rows - 1) * pitch_j + cell_j) / 2, "$N$",
            ha="right", va="center", fontsize=6.8, color=MUTED)
    ax.text(x_j0 - 0.7, y_j0 - 1.5, "$[B, N, D]$", ha="right",
            va="center", fontsize=7.4, fontweight="bold", color=BLUE)
    arrow(ax, (cx, FUSE_BOX_Y), (cx, y_j0 + (N_rows - 1) * pitch_j + cell_j + 0.3),
          lw=1.3)
    arrow(ax, (cx, y_j0 - 0.2), (cx, TRANS_BOX_Y + TRANS_BOX_H), lw=1.3)

    # Ghost padded tensor beside the fused grid -- never materialised
    gx, gy, gw, gh = x_j0 + jw + 1.8, 31.0, 7.4, 5.8
    rrect(ax, gx, gy, gw, gh, edge=MUTED, face="#FCFCFC", lw=1.3,
          dashed=True, rounding=0.5, z=2)
    ax.plot([gx + 2.0, gx + gw - 2.0], [gy + 1.4, gy + gh - 1.4],
            color=RED, lw=1.6, alpha=0.8, zorder=3)
    ax.text(gx + gw / 2, gy + gh - 0.9, "$[B, C_{\\max}, N, D]$",
            ha="center", va="center", fontsize=5.2, color=MUTED,
            style="italic", zorder=4)
    ax.text(gx + gw / 2, gy + 0.8, "never materialised",
            ha="center", va="center", fontsize=5.2, color=MUTED,
            style="italic", zorder=4)

    # Spatial transformer
    box(ax, x0, TRANS_BOX_Y, COL_W, TRANS_BOX_H, "Spatial transformer",
        "no padding, no mask", edge=PURPLE, title_size=7.6, sub_size=6.4)

    # Operation equation
    ax.text(cx, EQ_Y, r"$\mathrm{GroupedPool}(q, \{x_c\}_{c \in C_b})$",
            ha="center", va="center", fontsize=7.4, color=INK)

    # Takeaway
    rrect(ax, x0, TAKEAWAY_Y, COL_W, TAKEAWAY_H, edge=GREEN,
          face=tint(GREEN, 0.88), lw=1.5, rounding=0.5, z=2)
    ax.text(cx, TAKEAWAY_Y + TAKEAWAY_H / 2,
            "No pad/mask overhead;\n"
            "$O(B{\\cdot}N{\\cdot}D)$ — independent of $C$",
            ha="center", va="center", fontsize=6.6, color=INK,
            fontweight="bold", linespacing=1.3)


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
    ax.text(2.0, 74.2, "L0", ha="left", va="center", fontsize=15,
            fontweight="bold", color=INK)
    ax.text(6.5, 74.2,
            "Positioning: padding-free channel fusion vs. padded approaches",
            ha="left", va="center", fontsize=11.5, fontweight="bold", color=INK)

    # Three columns -- same width, same stage y
    draw_column_a(ax, x0=5.0)
    draw_column_b(ax, x0=47.0)
    draw_column_c(ax, x0=89.0)

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
