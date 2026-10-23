"""L0 -- positioning / comparison: padding-free vs. padded approaches.

Three-column comparison in the ChAda-ViT Fig 13 idiom:
(a) Padded dense ViT -- stack channels into [B, C_max, N, D], explicit pad rows + -inf mask
(b) Dense cross-channel attention (ChannelViT / ChAda-ViT) -- padded channel axis, additive channel embedding
(c) Ours - padding-free channel fusion -- ragged [Sigma C*N, D] -> per-(b,n) fusion -> [B,N,D]; NO pad/mask

All three columns share the same scale and width; the stages are aligned at
identical y so the panels read across.  Panel (c) is emphasised only through
the green accent frame/tint, never through size.

Colour is semantic (see ``figure_style`` role palette): front-end green,
fusion orange, spatial transformer pink, fused/joint tokens blue, loss/waste
red, never-materialised padded tensor a dashed ghost.

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
    GREEN, INK, MUTED, CHANNEL_COLORS,
    FRONTEND, FUSION, ENCODER, DECODER, MEMORY, LOSS, FUSED,
    tint, rrect, box, arrow, edge_label, sq_bracket, panel_header,
    legend, role_legend, channel_legend, padded_ghost,
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
    n_channel_ids = 3
    x0 = cx - (n_channel_ids * cw + (n_channel_ids - 1) * 1.0) / 2
    for i in range(n_channel_ids):
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

    # N / D / shape / pad rows, all hanging off the right edge of the grid
    xr = x_g0 + GRID_W + 1.4
    ax.text(xr, TENSOR_TOP + 0.4, "$N$ patches · $D$ dim", ha="left",
            va="center", fontsize=6.8, color=MUTED)
    ax.text(xr, y_g0 + (C_MAX - 1) * PITCH + CELL / 2, "[$B, C_{\\max}, N, D$]",
            ha="left", va="center", fontsize=6.8, color=INK)
    return x_g0, xr


def dashed_module(ax, x, y, w, h, title, sub="", edge=MUTED,
                  title_size=7.4, sub_size=6.2):
    """A muted dashed module box -- the visual verb for "rejected baseline"."""
    rrect(ax, x, y, w, h, edge=edge, face=tint(edge, 0.9), lw=1.4,
          dashed=True, rounding=0.7, z=3)
    cx = x + w / 2
    if sub:
        ax.text(cx, y + h * 0.66, title, ha="center", va="center",
                fontsize=title_size, fontweight="bold", color=MUTED, zorder=4)
        ax.text(cx, y + h * 0.30, sub, ha="center", va="center",
                fontsize=sub_size, color=MUTED, zorder=4)
    else:
        ax.text(cx, y + h / 2, title, ha="center", va="center",
                fontsize=title_size, fontweight="bold", color=MUTED, zorder=4)


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
    # pad-row tag attached to the dashed rows, on the right (LOSS = waste)
    ax.text(xr, TENSOR_BOT + 3.5 * PITCH, "pad rows", ha="left", va="center",
            fontsize=6.2, color=LOSS, style="italic")

    # Attention mask -- waste, stays LOSS red
    box(ax, x0, FUSE_BOX_Y, COL_W, FUSE_BOX_H, "-inf mask on pad",
        "$O(C_{\\max}^2 N^2)$ memory", edge=LOSS, title_size=7.0, sub_size=6.0)
    arrow(ax, (cx, TENSOR_BOT - 0.2), (cx, FUSE_BOX_Y + FUSE_BOX_H),
          color=LOSS, lw=1.2)

    # Spatial transformer -- ENCODER role
    box(ax, x0, TRANS_BOX_Y, COL_W, TRANS_BOX_H, "Transformer", "all tokens",
        edge=ENCODER, title_size=7.6, sub_size=6.4)
    arrow(ax, (cx, FUSE_BOX_Y), (cx, TRANS_BOX_Y + TRANS_BOX_H), lw=1.3)

    # Operation equation
    ax.text(cx, EQ_Y, r"$\mathrm{softmax}\left(\frac{QK^T}{\sqrt{d}} + M\right)V$",
            ha="center", va="center", fontsize=7.4, color=INK)

    # Takeaway -- the waste is the point, keep it LOSS red
    rrect(ax, x0, TAKEAWAY_Y, COL_W, TAKEAWAY_H, edge=LOSS,
          face=tint(LOSS, 0.9), lw=1.1, rounding=0.5, z=2)
    ax.text(cx, TAKEAWAY_Y + TAKEAWAY_H / 2,
            "Waste: $O(B \\cdot C_{\\max} \\cdot N \\cdot D)$ memory\n"
            "for pad tokens + explicit mask",
            ha="center", va="center", fontsize=6.6, color=LOSS, linespacing=1.3)


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
            fontsize=6.2, color=LOSS, style="italic")

    # Additive channel identity -- FRONTEND role (green)
    box(ax, x0, FUSE_BOX_Y, COL_W, FUSE_BOX_H, "Additive $z_c$ identity",
        "marker vector · not omitted", edge=FRONTEND,
        title_size=6.8, sub_size=6.0)
    arrow(ax, (cx, TENSOR_BOT - 0.2), (cx, FUSE_BOX_Y + FUSE_BOX_H),
          color=FRONTEND, lw=1.2)

    # Dense C x C channel attention -- the rejected baseline (muted + dashed)
    dashed_module(ax, x0, TRANS_BOX_Y, COL_W, TRANS_BOX_H,
                  "Dense channel attn", "$O(C_{\\max}^2)$ · rejected",
                  edge=MUTED, title_size=7.4, sub_size=6.2)
    arrow(ax, (cx, FUSE_BOX_Y), (cx, TRANS_BOX_Y + TRANS_BOX_H), lw=1.3,
          color=MUTED)
    # a small red cross on the rejected baseline
    ax.text(x0 + COL_W - 2.6, TRANS_BOX_Y + TRANS_BOX_H - 0.85, "✗",
            ha="center", va="center", fontsize=11, color=LOSS, zorder=6)

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

    panel_header(ax, x0, HEADER_Y, "(c)", "Ours: padding-free fusion",
                 color=GREEN)

    draw_shared_input(ax, cx)
    arrow(ax, (cx, INPUT_TILE_Y), (cx, TENSOR_TOP + 0.3), color=GREEN, lw=1.4)
    edge_label(ax, cx - 2.6, 59.6, "ragged list", color=GREEN, ha="right")

    # Ragged token list [Sigma C . N, D]: the solid FUSED primary object, with a
    # per-channel colour tab so channel identity is still visible.
    w_rag = 9.5
    x_rag = cx - w_rag / 2
    h_row, pitch_rag, block_gap = 1.6, 1.8, 0.4
    bar_w = 0.8
    y_block = TENSOR_BOT
    for c in range(3):
        col = CHANNEL_COLORS[c]
        rrect(ax, x_rag, y_block, bar_w, pitch_rag + h_row, edge=col, face=col,
              lw=0.8, rounding=0.15, z=4)
        for _ in range(2):
            rrect(ax, x_rag + bar_w, y_block, w_rag - bar_w, h_row, edge=FUSED,
                  face=tint(FUSED, 0.8), lw=1.0, rounding=0.2, z=3)
            y_block += pitch_rag
        y_block += block_gap
    top_rag = y_block - pitch_rag - block_gap + h_row
    sq_bracket(ax, x_rag + w_rag + 0.6, TENSOR_BOT, top_rag)
    ax.text(x_rag + w_rag + 1.2, (TENSOR_BOT + top_rag) / 2,
            "$[\\Sigma C{\\cdot}N, D]$", ha="left", va="center",
            fontsize=7.0, color=FUSED, fontweight="bold")

    # Channel fusion -- FUSION role
    box(ax, x0, FUSE_BOX_Y, COL_W, FUSE_BOX_H, "Channel fusion",
        "per-$(b,n)$ pool · $O(1)$ output size", edge=FUSION,
        title_size=7.6, sub_size=6.2)
    arrow(ax, (cx, TENSOR_BOT - 0.2), (cx, FUSE_BOX_Y + FUSE_BOX_H),
          color=FUSION, lw=1.4)

    # Joint embeddings [B, N, D] -- FUSED blue, filling fusion -> transformer
    cell_j, pitch_j = 1.8, 2.0
    B_cols, N_rows = 3, 3
    jw = (B_cols - 1) * pitch_j + cell_j
    y_j0 = 31.1
    x_j0 = cx - jw / 2
    for b in range(B_cols):
        for n in range(N_rows):
            rrect(ax, x_j0 + b * pitch_j, y_j0 + n * pitch_j, cell_j, cell_j,
                  edge=FUSED, face=tint(FUSED, 0.75), lw=1.0, rounding=0.25, z=3)
    ax.text(x_j0 + jw + 0.7, y_j0 + (N_rows - 1) * pitch_j + cell_j,
            "$B$", ha="left", va="center", fontsize=6.8, color=MUTED)
    ax.text(x_j0 - 0.7, y_j0 + ((N_rows - 1) * pitch_j + cell_j) / 2, "$N$",
            ha="right", va="center", fontsize=6.8, color=MUTED)
    ax.text(x_j0 - 0.7, y_j0 - 1.5, "$[B, N, D]$", ha="right",
            va="center", fontsize=7.4, fontweight="bold", color=FUSED)
    arrow(ax, (cx, FUSE_BOX_Y), (cx, y_j0 + (N_rows - 1) * pitch_j + cell_j + 0.3),
          color=FUSION, lw=1.3)
    arrow(ax, (cx, y_j0 - 0.2), (cx, TRANS_BOX_Y + TRANS_BOX_H), lw=1.3)

    # Ghost padded tensor beside the fused grid -- never materialised
    gx, gy, gw, gh = x_j0 + jw + 2.4, 30.6, 9.5, 5.6
    padded_ghost(ax, gx, gy, gw, gh)

    # Spatial transformer -- ENCODER role
    box(ax, x0, TRANS_BOX_Y, COL_W, TRANS_BOX_H, "Spatial transformer",
        "no padding, no mask", edge=ENCODER, title_size=7.6, sub_size=6.4)

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
# Compact legends in the whitespace under the columns
# --------------------------------------------------------------------------- #
def draw_legends(ax):
    role_legend(
        ax, 4.0, 8.5,
        [
            ("front-end: patch embed / tied conv / spatial + $z_c$", FRONTEND),
            ("channel fusion (ChannelFusion / weighted sum)", FUSION),
            ("spatial transformer", ENCODER),
            ("channel-aware decoder", DECODER),
            ("memory / register rows", MEMORY),
            ("fused / joint tokens $[B,N,D]$", FUSED),
            ("loss / waste / $-\\infty$ mask", LOSS),
        ],
        title="module roles", pitch=2.3, fontsize=6.2, sw=1.3,
    )
    legend(
        ax, 52.0, 8.5,
        [
            ("solid cell = kept token (primary object)", FUSED, "fill"),
            ("dashed white = omitted / absent row", MUTED, "omit"),
            ("dashed pale = never materialised tensor", MUTED, "ghost"),
            ("muted / dashed = rejected dense baseline", MUTED, "dash"),
        ],
        title="visual key", pitch=2.3, fontsize=6.2, sw=1.3,
    )
    channel_legend(ax, 96.0, 8.5, n=3, title="channel identity",
                   pitch=2.3, fontsize=6.2, sw=1.3)


# --------------------------------------------------------------------------- #
# The figure
# --------------------------------------------------------------------------- #
def build() -> plt.Figure:
    fig = plt.figure(figsize=(13, 8))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_xlim(0, 128)
    ax.set_ylim(-9.5, 76)
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

    # Legends underneath, in the empty band
    draw_legends(ax)

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
