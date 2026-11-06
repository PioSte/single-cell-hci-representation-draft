"""L0 -- positioning: padding-free channel fusion.

A single hero panel for the *shipped* padding-free path:

    multi-channel input -> tied Conv2d patch embed (canonical kernel,
      PI-resized per runtime patch size p) -> content tokens [V, D]
      together with a parallel channel-identity list z_c [V, D]
      (solid primary object; rows are real
         tokens only, never padded or zeroed; V <= Sum C . N)
      -> grouped_pool_attention at each (b, n): one shared learned query,
         Q/K/V projections to D_attn = 3D, proj_out; fusion adds no
         positional table
      -> one fused [D] vector per present (b, n) -> encoder with
         2-D axial RoPE and scale conditioning.

Only this shipped solution is drawn.  The padded tensor [B, C_max, N, D]
appears merely as a small dashed ghost tagged "never allocated"; no dense /
padded baseline and no alternative fusion is shown.  The decoder (not drawn)
uses fractional absolute PE.

Colour is semantic (see ``figure_style`` role palette): channel identity
green, channel fusion orange, encoder pink, fused tokens blue,
never-materialised padded tensor a dashed ghost.

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
    GREEN, INK, MUTED, CHANNEL_COLORS, GHOST, LOSS,
    TOKENIZER, FUSION, ENCODER, FUSED,
    tint, rrect, box, arrow, edge_label, sq_bracket_mirror,
    token_rows, grid_cells, legend, channel_legend, padded_ghost,
    install_rcparams, save_figure,
)

OUT = Path(__file__).resolve().parent / "out"


# --------------------------------------------------------------------------- #
# Single-panel geometry -- canvas is 128 wide, panel centred on x = 64
# --------------------------------------------------------------------------- #
CX = 64.0
COL_W = 34.0
X0 = CX - COL_W / 2

N_PATCH = 3                  # same N for the token list and the fused grid

# input tiles
INPUT_TILE_Y = 64.0          # bottom edge
INPUT_LABEL_Y = 70.5
TILE_SIDE = 5.0

# flat token list [V, D] (visible rows only; V <= Sum C . N)
LIST_BOT = 42.0
LIST_H_ROW = 1.35
LIST_PITCH = 1.6
LIST_GAP = 0.35
LIST_W = 10.0
LIST_BAR_W = 0.8
_LIST_TAB_H = (N_PATCH - 1) * LIST_PITCH + LIST_H_ROW
LIST_TOP = LIST_BOT + 3 * _LIST_TAB_H + 2 * LIST_GAP

# channel fusion (grouped_pool_attention) + its primitive annotation
FUSE_BOX_Y = 33.8
FUSE_BOX_H = 6.2
ANNOT_Y = 32.4

# fused vectors, one per present (b, n)
FUSED_BOT = 24.6
FUSED_CELL = 1.9
FUSED_PITCH = 2.1
FUSED_B_COLS = 3

# encoder (TransformerEncoder)
TRANS_BOX_Y = 16.8
TRANS_BOX_H = 5.6
DECODER_NOTE_Y = 15.3

# takeaway
TAKEAWAY_Y = 10.4
TAKEAWAY_H = 3.8

# hero frame
FRAME_X = 44.0
FRAME_W = 40.0
FRAME_Y = 9.5
FRAME_H = 64.2


# --------------------------------------------------------------------------- #
# Shared input
# --------------------------------------------------------------------------- #
def draw_input(ax):
    """Multi-channel input set (variable C_b, order-free)."""
    cw, gap = TILE_SIDE, 1.5
    n_ids = 3
    x0 = CX - (n_ids * cw + (n_ids - 1) * gap) / 2
    for i in range(n_ids):
        c = CHANNEL_COLORS[i]
        xx = x0 + i * (cw + gap)
        rrect(ax, xx, INPUT_TILE_Y, cw, cw, edge=c, face=tint(c, 0.72),
              lw=1.4, rounding=0.4)
        ax.plot([xx + 0.8, xx + cw - 0.8],
                [INPUT_TILE_Y + 0.8, INPUT_TILE_Y + cw - 1.0],
                color=c, lw=0.9, alpha=0.65, zorder=4)
        ax.plot([xx + 0.8, xx + cw - 0.8],
                [INPUT_TILE_Y + cw - 1.0, INPUT_TILE_Y + 0.8],
                color=c, lw=0.9, alpha=0.4, zorder=4)
    rrect(ax, 54.5, 62.6, 19.0, 6.8, edge=GHOST, face="none", lw=1.0,
          dashed=True, rounding=0.4, z=2)
    ax.text(53.8, 66.5, "input set $[\\Sigma C, 1, H, W]$", ha="right",
            va="center", fontsize=5.8, color=MUTED)
    ax.text(CX, INPUT_LABEL_Y, "$C_b$ channels · order-free",
            ha="center", va="center", fontsize=7.8, color=MUTED)


# --------------------------------------------------------------------------- #
# Token list [V, D] -- the solid primary object
# --------------------------------------------------------------------------- #
def draw_token_list(ax):
    """[V, D] token list with per-channel colour tabs."""
    x_list = CX - LIST_W / 2
    y_block = LIST_BOT
    for c in range(3):
        col = CHANNEL_COLORS[c]
        rrect(ax, x_list, y_block, LIST_BAR_W, _LIST_TAB_H, edge=col, face=col,
              lw=0.8, rounding=0.15, z=4)
        specs = [(FUSED, False)] * N_PATCH
        if c == 2:
            specs = [(FUSED, False), (FUSED, False), (LOSS, True)]
        token_rows(ax, x_list + LIST_BAR_W, y_block, LIST_W - LIST_BAR_W,
                   LIST_H_ROW, LIST_PITCH, specs,
                   rounding=0.2, z=3)
        y_block += _LIST_TAB_H + LIST_GAP

    # dashed top row = omitted (masked) token; name it explicitly
    omitted_y = (LIST_BOT + 2 * (_LIST_TAB_H + LIST_GAP)
                 + 2 * LIST_PITCH + LIST_H_ROW / 2)
    ax.text(x_list - 0.6, omitted_y, "omitted", ha="right", va="center",
            fontsize=5.2, color=LOSS, style="italic", zorder=5)

    # bracket to the RIGHT of the stack: ticks point left to embrace it
    sq_bracket_mirror(ax, x_list + LIST_W + 0.6, LIST_BOT, LIST_TOP)
    ax.text(x_list + LIST_W + 1.2, (LIST_BOT + LIST_TOP) / 2 + 1.0,
            "$[V, D]$", ha="left", va="center",
            fontsize=7.2, color=FUSED, fontweight="bold")
    ax.text(x_list + LIST_W + 1.2, (LIST_BOT + LIST_TOP) / 2 - 1.4,
            "visible rows only\n$V \\leq \\Sigma_b C_b{\\cdot}N_b$",
            ha="left", va="center", fontsize=5.8, color=MUTED,
            linespacing=1.3)


# --------------------------------------------------------------------------- #
# Fused vectors + never-materialised ghost
# --------------------------------------------------------------------------- #
def draw_fused(ax):
    """The [B, N] location grid of fused vectors, plus the ghost."""
    jw = (FUSED_B_COLS - 1) * FUSED_PITCH + FUSED_CELL
    jh = (N_PATCH - 1) * FUSED_PITCH + FUSED_CELL
    x_j0 = CX - jw / 2
    grid_cells(ax, x_j0, FUSED_BOT, N_PATCH, FUSED_B_COLS, FUSED_CELL,
               FUSED_PITCH, edge=FUSED, face=tint(FUSED, 0.75), lw=1.0,
               rounding=0.25, z=3)

    ax.text(x_j0 - 0.9, FUSED_BOT + jh / 2, "$[K_f, D]$", ha="right",
            va="center", fontsize=7.4, fontweight="bold", color=FUSED)
    # dashed padded tensor, nudged clear of the fused grid and frame edge
    gx, gy, gw, gh = x_j0 + jw + 3.0, FUSED_BOT + 0.6, 10.0, 5.0
    padded_ghost(ax, gx, gy, gw, gh)


# --------------------------------------------------------------------------- #
# Hero panel
# --------------------------------------------------------------------------- #
def draw_panel(ax):
    """The single padding-free hero panel."""
    rrect(ax, FRAME_X, FRAME_Y, FRAME_W, FRAME_H, edge=GHOST,
          face=tint(GHOST, 0.96), lw=2.2, rounding=1.0, z=0)
    ax.text(CX, 72.0, "ChannelFusionTokenizer", ha="center", va="center",
            fontsize=7.8, fontweight="bold", color=INK)

    draw_input(ax)

    # tokenizer path: tied conv (PI-resized kernel); identity carried separately
    arrow(ax, (CX, 63.0), (CX, LIST_TOP + 0.3), color=TOKENIZER, lw=1.5)
    ax.text(CX + 2.2, 59.0,
            "tied conv · kernel PI-resized\n"
            "when $p \\neq P$ (identity at the shipped $P$)\n"
            "content token; carry $z_c$",
            ha="left", va="center", fontsize=6.0, color=TOKENIZER,
            linespacing=1.3)

    draw_token_list(ax)

    # channel fusion -- the O(C) primitive lives here
    box(ax, X0, FUSE_BOX_Y, COL_W, FUSE_BOX_H, "ChannelFusion",
        "one shared query · $W_Q/W_K/W_V$ to $D_{attn}\\!=\\!3D$ · proj_out\n"
        "identity routing bias + output skip · no fusion PE\n"
        "$O(C{\\cdot}D{\\cdot}D_{attn}) + O(D{\\cdot}D_{attn})$ per location",
        edge=FUSION,
        title_size=7.8, sub_size=5.9)
    ax.text(CX - 8.0, ANNOT_Y, "grouped_pool_attention", ha="center", va="center",
            fontsize=7.0, color=INK, family="monospace", fontweight="bold",
            zorder=7,
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                      edgecolor="none", alpha=0.92))
    arrow(ax, (CX, LIST_BOT - 0.2), (CX, FUSE_BOX_Y + FUSE_BOX_H),
          color=FUSION, lw=1.5)

    draw_fused(ax)
    arrow(ax, (CX, FUSE_BOX_Y), (CX, FUSED_BOT + (N_PATCH - 1) * FUSED_PITCH
                                 + FUSED_CELL + 0.3), color=FUSION, lw=1.4)

    # encoder (TransformerEncoder)
    box(ax, X0, TRANS_BOX_Y, COL_W, TRANS_BOX_H,
        "encoder (TransformerEncoder)",
        "grouped self-attention\n"
        "2-D axial RoPE · scale conditioning", edge=ENCODER,
        title_size=7.4, sub_size=5.9)
    arrow(ax, (CX, FUSED_BOT - 0.2), (CX, TRANS_BOX_Y + TRANS_BOX_H), lw=1.4)

    ax.text(CX, DECODER_NOTE_Y,
            "decoder (not drawn): fractional absolute PE", ha="center",
            va="center", fontsize=5.8, color=MUTED)

    # takeaway
    rrect(ax, X0, TAKEAWAY_Y, COL_W, TAKEAWAY_H, edge=GHOST,
          face=tint(GHOST, 0.90), lw=1.5, rounding=0.5, z=2)
    ax.text(CX, TAKEAWAY_Y + TAKEAWAY_H / 2,
            "One fused token per $(b,n)$ location;\n"
            "$O(B{\\cdot}N{\\cdot}D)$ memory — independent of $C$",
            ha="center", va="center", fontsize=6.4, color=INK,
            fontweight="bold", linespacing=1.3)


# --------------------------------------------------------------------------- #
# Glossary beside the hero panel
# --------------------------------------------------------------------------- #
_GLOSSARY = [
    ("$B$", "batch size"),
    ("$C_b$", "channels in sample $b$ (order-free)"),
    ("$\\Sigma C$", "total channels in the batch"),
    ("$H, W$", "image side (square)"),
    ("$N$", "patches per channel"),
    ("$D$", "token / embedding dim"),
    ("$P$", "canonical patch size"),
    ("$p$", "runtime patch size"),
    ("$V$", "visible channel-patches ($\\leq \\sum_b C_b{\\cdot}N_b$)"),
    ("$K_f$", "present $(b,n)$ locations ($\\leq B{\\cdot}N$)"),
    ("$z_c$", "channel identity (routing bias + skip)"),
    ("PE", "positional embedding"),
    ("$D_{attn}$", "attention dimension (= 3D)"),
    ("RoPE", "2-D axial rotary position on $(row,col)$"),
]


def draw_glossary(ax):
    """Compact symbol glossary in the lower-left band."""
    gx, gy, gw, gh = 3.0, 9.0, 38.0, 32.0
    rrect(ax, gx, gy, gw, gh, edge=MUTED, face="#FFFFFF", lw=1.0,
          rounding=0.6, z=1)
    ax.text(gx + 2.0, gy + gh - 1.6, "symbols / glossary", ha="left", va="center",
            fontsize=6.8, fontweight="bold", color=INK)
    y = gy + gh - 3.4
    for term, definition in _GLOSSARY:
        ax.text(gx + 2.0, y, term, ha="left", va="center", fontsize=5.9,
                fontweight="bold", color=INK, zorder=4)
        ax.text(gx + 8.5, y, definition, ha="left", va="center", fontsize=5.7,
                color=MUTED, zorder=4)
        y -= 1.9


# --------------------------------------------------------------------------- #
# Legends beside the hero panel
# --------------------------------------------------------------------------- #
def draw_legends(ax):
    legend(
        ax, 3.0, 70.5,
        [
            ("tokenizer: tied conv; carry $z_c$", TOKENIZER, "fill"),
            ("visible channel-patch tokens $[V,D]$", FUSED, "cross"),
            ("channel fusion (grouped_pool_attention)", FUSION, "fill"),
            ("encoder: TransformerEncoder + RoPE", ENCODER, "fill"),
            ("fused vectors $[K_f,D]$", FUSED, "cross"),
        ],
        title="module roles", pitch=2.4, fontsize=6.3, sw=1.3,
    )
    channel_legend(ax, 3.0, 50.0, n=3, title="token colour = channel identity",
                   pitch=2.4, fontsize=6.3, sw=1.3)
    legend(
        ax, 88.0, 70.5,
        [
            ("crossed cell = token row $[V,D]$ (channel colour)\n"
             "or fused row $[K_f,D]$ (blue)", FUSED, "cross"),
            ("dashed row = omitted (masked) token;\n"
             "never zeroed / padded", LOSS, "omit"),
            ("dashed ghost $[B,C_{\\max},N,D]$ = NEVER ALLOCATED;\n"
             "no dense tensor is built", MUTED, "ghost"),
        ],
        title="visual key", pitch=2.9, fontsize=6.3, sw=1.3,
    )

    # short notes fill the side bands with shipped-code facts only
    rrect(ax, 88.0, 50.2, 28.0, 6.4, edge=MUTED, face="#FFFFFF", lw=1.0,
          rounding=0.6, z=1)
    ax.text(102.0, 55.0, "masking convention", ha="center", va="center",
            fontsize=6.6, fontweight="bold", color=INK)
    ax.text(102.0, 52.2,
            "masked rows are omitted\n(never zeroed or padded)",
            ha="center", va="center", fontsize=6.0, color=MUTED,
            linespacing=1.4)

    rrect(ax, 88.0, 30.4, 28.0, 7.6, edge=GHOST, face=tint(GHOST, 0.95),
          lw=1.0, rounding=0.6, z=1)
    ax.text(102.0, 36.5, "padding-free by construction", ha="center",
            va="center", fontsize=6.6, fontweight="bold", color=INK)
    ax.text(102.0, 33.1,
            "fusion consumes the token list directly:\n"
            "$O(C{\\cdot}D{\\cdot}D_{attn})+O(D{\\cdot}D_{attn})$ per $(b,n)$ —\n"
            "$O(C)$ scaling, not dense $C{\\times}C$; output indep. of $C$",
            ha="center", va="center", fontsize=6.0, color=MUTED,
            linespacing=1.4)


# --------------------------------------------------------------------------- #
# The figure
# --------------------------------------------------------------------------- #
def build() -> plt.Figure:
    fig = plt.figure(figsize=(13, 8))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_xlim(0, 128)
    ax.set_ylim(7.0, 77.5)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    # Title
    ax.text(2.0, 75.3, "L0", ha="left", va="center", fontsize=15,
            fontweight="bold", color=INK)
    ax.text(6.5, 75.3, "Padding-free channel fusion",
            ha="left", va="center", fontsize=12.2, fontweight="bold",
            color=INK)

    draw_glossary(ax)
    draw_panel(ax)
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
