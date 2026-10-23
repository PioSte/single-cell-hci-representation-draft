"""L0 -- positioning: padding-free channel fusion.

A single hero panel for the *shipped* padding-free path:

    multi-channel input -> tied patch embed + additive channel identity z_c
      -> ragged token list [Sum C . N, D]   (solid primary object; rows are
         real tokens only, never padded or zeroed)
      -> grouped_pool_attention at each (b, n): one shared learned query,
         keys = values = norm_kv(tokens), proj_out, + learned post-fusion PE
      -> fused [B, N, D] joint tokens -> spatial transformer with no pad/mask.

Only this shipped solution is drawn.  The padded tensor [B, C_max, N, D]
appears merely as a small dashed ghost tagged "never materialised"; no dense /
padded baseline and no alternative fusion is shown.

Colour is semantic (see ``figure_style`` role palette): additive channel
identity green, channel fusion orange, spatial transformer pink, fused tokens
blue, never-materialised padded tensor a dashed ghost.

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
    TOKENIZER, FUSION, ENCODER, FUSED,
    tint, rrect, box, arrow, edge_label, sq_bracket,
    legend, role_legend, channel_legend, padded_ghost,
    install_rcparams, save_figure,
)

OUT = Path(__file__).resolve().parent / "out"


# --------------------------------------------------------------------------- #
# Single-panel geometry -- canvas is 128 wide, panel centred on x = 64
# --------------------------------------------------------------------------- #
CX = 64.0
COL_W = 34.0
X0 = CX - COL_W / 2

N_PATCH = 3                  # same N for the ragged list and the fused grid

# input tiles
INPUT_TILE_Y = 65.8          # bottom edge
INPUT_LABEL_Y = 72.2
TILE_SIDE = 5.0

# ragged token list [Sum C . N, D]
RAG_BOT = 43.6
RAG_H_ROW = 1.35
RAG_PITCH = 1.6
RAG_GAP = 0.35
RAG_W = 10.0
RAG_BAR_W = 0.8
_RAG_TAB_H = (N_PATCH - 1) * RAG_PITCH + RAG_H_ROW
RAG_TOP = RAG_BOT + 3 * _RAG_TAB_H + 2 * RAG_GAP

# channel fusion
FUSE_BOX_Y = 36.5
FUSE_BOX_H = 5.0

# fused joint tokens [B, N, D]
FUSED_BOT = 28.6
FUSED_CELL = 1.9
FUSED_PITCH = 2.1
FUSED_B_COLS = 3

# spatial transformer
TRANS_BOX_Y = 21.0
TRANS_BOX_H = 5.0

# equation + takeaway
EQ_TOP_Y = 18.8
EQ_SUB_Y = 16.9
TAKEAWAY_Y = 11.8
TAKEAWAY_H = 4.2

# hero frame
FRAME_X = 44.0
FRAME_W = 40.0
FRAME_Y = 9.8
FRAME_H = 63.7


# --------------------------------------------------------------------------- #
# Shared input
# --------------------------------------------------------------------------- #
def draw_input(ax):
    """Ragged multi-channel input set (variable C_b, order-free)."""
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
    ax.text(CX, INPUT_LABEL_Y, "$C_b$ channels · order-free",
            ha="center", va="center", fontsize=7.8, color=MUTED)


# --------------------------------------------------------------------------- #
# Ragged token list -- the solid primary object
# --------------------------------------------------------------------------- #
def draw_ragged(ax):
    """[Sum C . N, D] ragged token list with per-channel colour tabs."""
    x_rag = CX - RAG_W / 2
    y_block = RAG_BOT
    for c in range(3):
        col = CHANNEL_COLORS[c]
        rrect(ax, x_rag, y_block, RAG_BAR_W, _RAG_TAB_H, edge=col, face=col,
              lw=0.8, rounding=0.15, z=4)
        for r in range(N_PATCH):
            rrect(ax, x_rag + RAG_BAR_W, y_block + r * RAG_PITCH,
                  RAG_W - RAG_BAR_W, RAG_H_ROW, edge=FUSED,
                  face=tint(FUSED, 0.8), lw=1.0, rounding=0.2, z=3)
        y_block += _RAG_TAB_H + RAG_GAP

    sq_bracket(ax, x_rag + RAG_W + 0.6, RAG_BOT, RAG_TOP)
    ax.text(x_rag + RAG_W + 1.2, (RAG_BOT + RAG_TOP) / 2,
            "$[\\Sigma C{\\cdot}N, D]$", ha="left", va="center",
            fontsize=7.2, color=FUSED, fontweight="bold")


# --------------------------------------------------------------------------- #
# Fused joint tokens [B, N, D] + never-materialised ghost
# --------------------------------------------------------------------------- #
def draw_fused(ax):
    """Fused [B, N, D] grid and the dashed padded-tensor ghost."""
    jw = (FUSED_B_COLS - 1) * FUSED_PITCH + FUSED_CELL
    jh = (N_PATCH - 1) * FUSED_PITCH + FUSED_CELL
    x_j0 = CX - jw / 2
    for b in range(FUSED_B_COLS):
        for n in range(N_PATCH):
            rrect(ax, x_j0 + b * FUSED_PITCH, FUSED_BOT + n * FUSED_PITCH,
                  FUSED_CELL, FUSED_CELL, edge=FUSED, face=tint(FUSED, 0.75),
                  lw=1.0, rounding=0.25, z=3)

    ax.text(x_j0 - 0.9, FUSED_BOT + jh / 2, "$[B, N, D]$", ha="right",
            va="center", fontsize=7.4, fontweight="bold", color=FUSED)
    ax.text(x_j0 + jw + 0.8, FUSED_BOT + (N_PATCH - 1) * FUSED_PITCH
            + FUSED_CELL / 2, "$B$", ha="left", va="center", fontsize=6.8,
            color=MUTED)
    # dashed padded tensor, nudged clear of the frame edge
    gx, gy, gw, gh = x_j0 + jw + 2.2, FUSED_BOT + 0.6, 10.0, 5.2
    padded_ghost(ax, gx, gy, gw, gh)


# --------------------------------------------------------------------------- #
# Hero panel
# --------------------------------------------------------------------------- #
def draw_panel(ax):
    """The single padding-free hero panel."""
    rrect(ax, FRAME_X, FRAME_Y, FRAME_W, FRAME_H, edge=GREEN,
          face=tint(GREEN, 0.965), lw=2.2, rounding=1.0, z=0)

    draw_input(ax)

    # tokenizer path: tied embed + additive channel identity
    arrow(ax, (CX, INPUT_TILE_Y), (CX, RAG_TOP + 0.3), color=TOKENIZER, lw=1.5)
    edge_label(ax, CX + 2.0, (INPUT_TILE_Y + RAG_TOP) / 2,
               "tied conv + $z_c$", color=TOKENIZER, ha="left", size=6.6)

    draw_ragged(ax)

    # channel fusion
    box(ax, X0, FUSE_BOX_Y, COL_W, FUSE_BOX_H, "Channel fusion",
        "one shared query · $O(C)$ per location", edge=FUSION,
        title_size=7.8, sub_size=6.2)
    arrow(ax, (CX, RAG_BOT - 0.2), (CX, FUSE_BOX_Y + FUSE_BOX_H),
          color=FUSION, lw=1.5)

    draw_fused(ax)
    arrow(ax, (CX, FUSE_BOX_Y), (CX, FUSED_BOT + (N_PATCH - 1) * FUSED_PITCH
                                 + FUSED_CELL + 0.3), color=FUSION, lw=1.4)

    # spatial transformer
    box(ax, X0, TRANS_BOX_Y, COL_W, TRANS_BOX_H, "Spatial transformer",
        "grouped self-attention + $M$ memory", edge=ENCODER, title_size=7.8, sub_size=6.4)
    arrow(ax, (CX, FUSED_BOT - 0.2), (CX, TRANS_BOX_Y + TRANS_BOX_H), lw=1.4)

    # real primitive (shipped), not an invented op
    ax.text(CX, EQ_TOP_Y, "grouped_pool_attention", ha="center", va="center",
            fontsize=7.4, color=INK, family="monospace", fontweight="bold")
    ax.text(CX, EQ_SUB_Y,
            "one shared query · K=V=norm_kv(tokens) · "
            "proj_out · + post-fusion PE",
            ha="center", va="center", fontsize=6.2, color=MUTED)

    # takeaway
    rrect(ax, X0, TAKEAWAY_Y, COL_W, TAKEAWAY_H, edge=GREEN,
          face=tint(GREEN, 0.88), lw=1.5, rounding=0.5, z=2)
    ax.text(CX, TAKEAWAY_Y + TAKEAWAY_H / 2,
            "One fused token per $(b,n)$ location;\n"
            "$O(B{\\cdot}N{\\cdot}D)$ memory — independent of $C$",
            ha="center", va="center", fontsize=6.6, color=INK,
            fontweight="bold", linespacing=1.3)


# --------------------------------------------------------------------------- #
# Legends beside the hero panel
# --------------------------------------------------------------------------- #
def draw_legends(ax):
    role_legend(
        ax, 4.0, 68.0,
        [
            ("tokenizer: tied conv / additive $z_c$", TOKENIZER),
            ("channel fusion (grouped_pool_attention)", FUSION),
            ("spatial transformer", ENCODER),
            ("fused / joint tokens $[B,N,D]$", FUSED),
        ],
        title="module roles", pitch=2.4, fontsize=6.3, sw=1.3,
    )
    channel_legend(ax, 4.0, 48.0, n=3, title="channel identity",
                   pitch=2.4, fontsize=6.3, sw=1.3)
    legend(
        ax, 88.0, 68.0,
        [
            ("solid cell = kept token (primary object)", FUSED, "fill"),
            ("dashed pale = never materialised tensor", MUTED, "ghost"),
        ],
        title="visual key", pitch=2.4, fontsize=6.3, sw=1.3,
    )

    # short notes fill the lower side bands with shipped-code facts only
    rrect(ax, 4.0, 17.5, 36.0, 13.5, edge=MUTED, face="#FFFFFF", lw=1.0,
          rounding=0.6, z=1)
    ax.text(22.0, 28.7, "masking convention", ha="center", va="center",
            fontsize=6.8, fontweight="bold", color=INK)
    ax.text(22.0, 22.3,
            "missing / masked channel-patches\nare absent from the ragged list;\n"
            "only visible rows are used",
            ha="center", va="center", fontsize=6.2, color=MUTED,
            linespacing=1.4)

    rrect(ax, 88.0, 31.5, 36.0, 13.5, edge=GREEN, face=tint(GREEN, 0.95),
          lw=1.0, rounding=0.6, z=1)
    ax.text(106.0, 42.6, "padding-free by construction", ha="center",
            va="center", fontsize=6.8, fontweight="bold", color=INK)
    ax.text(106.0, 36.3,
            "fusion consumes the ragged token list\ndirectly: $O(C)$ per $(b,n)$ and the\n"
            "output is independent of $C$",
            ha="center", va="center", fontsize=6.2, color=MUTED,
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
            ha="left", va="center", fontsize=12.5, fontweight="bold",
            color=INK)

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
