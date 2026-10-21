"""L1 -- system-level block diagram of the padding-free channel-fusion MAE.

A single hand-authored, vector-first figure (PDF/SVG/PNG) in the idiom of the
reference architecture figures in ``references/`` (Panopticon Fig. 1,
Perceiver-IO Fig. 2): literal array stacks with axis labels, module boxes,
dashed stage groups / shared-weight brackets, and shapes written on the edges.

Shows the *training* path (left to right) plus the inference ``encode()``
branch, the ghost ``[B, C_max, N, D]`` tensor that is never materialised, and
the MAE asymmetry (encoder sees only visible tokens; the channel-aware decoder
reconstructs omitted channel-patches).

Usage::

    uv run python viz/level1_system.py
    uv run python viz/level1_system.py --formats pdf svg
    uv run python viz/level1_system.py --stem level1_system
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from figure_style import (  # noqa: E402  (sibling module, run from viz/)
    BLUE, GREEN, ORANGE, RED, PURPLE, VIOLET, INK, MUTED, CHANNEL_COLORS,
    tint, rrect, box, arrow, edge_label, sq_bracket, install_rcparams,
    save_figure,
)

OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Array stacks
# --------------------------------------------------------------------------- #
def draw_input_set(ax):
    """Ragged set of single-channel images -- the variable axis is the height."""
    cw, pitch, x = 7.6, 6.4, 3.0
    for i, c in enumerate(CHANNEL_COLORS):
        yy = 40.0 + (len(CHANNEL_COLORS) - 1 - i) * pitch
        rrect(ax, x, yy, cw, cw, edge=c, face=tint(c, 0.72), lw=1.5)
        # a faint "image" squiggle so the tiles read as pictures
        ax.plot([x + 1.2, x + cw - 1.2], [yy + 2.2, yy + cw - 2.6],
                color=c, lw=1.0, alpha=0.7, zorder=4)
        ax.plot([x + 1.2, x + cw - 1.2], [yy + cw - 2.6, yy + 2.2],
                color=c, lw=1.0, alpha=0.4, zorder=4)
    ax.text(x + cw / 2, 76.4, "input set", ha="center", va="center",
            fontsize=8.4, fontweight="bold", color=INK)
    ax.text(x + cw / 2, 36.6, "$C_b$ markers\norder-free", ha="center",
            va="center", fontsize=6.6, color=MUTED, linespacing=1.35)
    ax.text(x - 0.9, 58.0, "$1\\times H\\times W$", ha="right", va="center",
            fontsize=6.8, color=MUTED, rotation=90)
    # "variable length" hint
    ax.text(x + cw + 0.6, 42.2, "$\\vdots$", ha="left", va="center",
            fontsize=10, color=MUTED)
    ax.text(x + cw + 0.6, 68.0, "$\\vdots$", ha="left", va="center",
            fontsize=10, color=MUTED)


def draw_ragged_tokens(ax):
    """Flattened ``[sum C . N, D]`` token set -- one row per channel-patch."""
    x, w, h, pitch, y0 = 46.5, 11.0, 2.9, 3.95, 40.0
    rows = 8
    for i in range(rows):
        yy = y0 + i * pitch
        c = CHANNEL_COLORS[(i // 2) % len(CHANNEL_COLORS)]
        omitted = i == 5
        rrect(ax, x, yy, w, h, edge=c, face=("#FFFFFF" if omitted else tint(c, 0.8)),
              lw=1.2, dashed=omitted)
        if omitted:
            ax.text(x + w / 2, yy + h / 2, "omitted row (DCP)", ha="center",
                    va="center", fontsize=6.2, color=RED, style="italic", zorder=4)
    top = y0 + (rows - 1) * pitch + h
    sq_bracket(ax, x + w + 1.0, y0, top)
    ax.text(x + w + 1.6, top + 1.3, "$[\\,\\Sigma C\\cdot N,\\; D\\,]$",
            ha="left", va="center", fontsize=8.0, color=INK)
    ax.text(x + w + 1.6, y0 - 1.5, "DCP $\\to K_f$",
            ha="left", va="center", fontsize=6.8, color=GREEN)
    ax.text(x + w / 2, 76.4, "ragged token set", ha="center", va="center",
            fontsize=8.4, fontweight="bold", color=INK)


def draw_joint_grid(ax, x0=86.0):
    """``[B, N, D]`` joint per-patch embeddings after channel fusion."""
    cell, pitch, y0 = 3.1, 3.55, 51.0
    B, N = 3, 4
    for b in range(B):
        for n in range(N):
            rrect(ax, x0 + n * pitch, y0 + b * pitch, cell, cell,
                  edge=BLUE, face=tint(BLUE, 0.72), lw=1.1, rounding=0.3)
    ax.text(x0 + N * pitch / 2 - 0.2, y0 + B * pitch + 1.0, "$N$ patches",
            ha="center", va="center", fontsize=6.8, color=MUTED)
    ax.text(x0 - 1.0, y0 + B * pitch / 2, "$B$", ha="right", va="center",
            fontsize=6.8, color=MUTED)
    ax.text(x0 + N * pitch / 2 - 0.2, 76.4, "joint embeddings", ha="center",
            va="center", fontsize=8.4, fontweight="bold", color=INK)


def draw_encoded(ax, x=134.0):
    """``[B . M + K_f, D]`` -- fixed-count memory + visible tokens."""
    w, h, pitch, y0 = 11.0, 2.9, 3.7, 42.0
    rows = [("mem", ORANGE)] * 2 + [("tok", BLUE)] * 5
    for i, (kind, c) in enumerate(rows):
        yy = y0 + i * pitch
        rrect(ax, x, yy, w, h, edge=c, face=tint(c, 0.8), lw=1.2)
    top = y0 + (len(rows) - 1) * pitch + h
    sq_bracket(ax, x + w + 1.0, y0, top)
    ax.text(x + w / 2, 76.4, "encoder memory", ha="center", va="center",
            fontsize=8.4, fontweight="bold", color=INK)
    ax.text(x + w / 2, 73.6, "$[\\,B\\cdot M + K_f,\\; D\\,]$", ha="center",
            va="center", fontsize=7.4, color=INK)
    ax.text(x - 0.9, 65.0, "$M$\nmem", ha="right", va="center", fontsize=6.2,
            color=ORANGE, linespacing=1.2)
    ax.text(x - 0.9, 50.0, "$K_f$\nvis", ha="right", va="center", fontsize=6.2,
            color=BLUE, linespacing=1.2)


def draw_output(ax, x, y0=30.0, w=22.0):
    """``[K_t, P^2]`` predicted pixels for the omitted channel-patches."""
    h, pitch, rows = 3.0, 4.0, 4
    for i in range(rows):
        rrect(ax, x, y0 + i * pitch, w, h, edge=VIOLET, face=tint(VIOLET, 0.82),
              lw=1.2)
    top = y0 + (rows - 1) * pitch + h
    ax.text(x + w / 2 + 3.0, top + 2.6, "$[\\,K_t,\\; P^2\\,]$ predictions",
            ha="center", va="center", fontsize=7.0, color=INK)
    return top


# --------------------------------------------------------------------------- #
# The figure
# --------------------------------------------------------------------------- #
def build() -> plt.Figure:
    fig = plt.figure(figsize=(16.5, 8.4))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_xlim(0, 176)
    ax.set_ylim(0, 86)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    # ---- title ------------------------------------------------------------ #
    ax.text(2, 83.2, "L1", ha="left", va="center", fontsize=15,
            fontweight="bold", color=INK)
    ax.text(6.5, 83.2, "System: padding-free channel-fusion MAE (training path + inference)",
            ha="left", va="center", fontsize=12.2, fontweight="bold", color=INK)

    # ---- arrays ----------------------------------------------------------- #
    draw_input_set(ax)
    draw_ragged_tokens(ax)
    draw_joint_grid(ax)
    draw_encoded(ax)

    # ---- module boxes ----------------------------------------------------- #
    box(ax, 18, 51, 24, 15, "Channel-tied patch embed",
        "shared Conv $1{\\to}D$, stride $P$\n$+$ shared spatial pos $+\\, z_c$",
        edge=GREEN)
    box(ax, 62, 51, 20, 15, "Channel fusion",
        "pool over channel axis\nper $(b,n)$ location\nPMA $O(C^2)$ · hypernet $O(C)$",
        edge=RED, sub_size=6.3)
    box(ax, 104, 51, 26, 15, "Spatial transformer",
        "grouped self-attention over real tokens\n$+M$ fixed-count memory tokens",
        edge=PURPLE)
    box(ax, 150, 51, 22, 15, "Channel-aware decoder",
        "one shared decoder\n$q=\\mathrm{mask} + \\mathrm{pos}_n + Wz_c$\n"
        "cross-attn $\\to$ pixels", edge=VIOLET, sub_size=6.1)

    # ---- arrows along the main track ------------------------------------- #
    ymid = 58.5
    arrow(ax, (11.0, ymid), (18.0, ymid))
    edge_label(ax, 14.5, ymid + 1.6, "$[\\Sigma C,1,H,W]$")
    arrow(ax, (42.0, ymid), (46.5, ymid))
    arrow(ax, (58.5, ymid), (62.0, ymid))
    arrow(ax, (82.0, ymid), (86.0, ymid))
    edge_label(ax, 84.0, ymid + 9.0, "$[B,N,D]$", color=BLUE)
    arrow(ax, (100.2, ymid), (104.0, ymid))
    arrow(ax, (130.0, ymid), (134.0, ymid))
    arrow(ax, (146.2, ymid), (150.0, ymid))

    # ---- DCP masker chip above the front-end ------------------------------ #
    box(ax, 18, 69.0, 24, 6.2, "DCP masker", "token omission (never zero)",
        edge=GREEN, title_size=7.6, sub_size=6.2)
    arrow(ax, (30.0, 69.0), (30.0, 66.2), color=GREEN, lw=1.3)

    # ---- shared-weight bracket over the transformer ----------------------- #
    rrect(ax, 101, 48.5, 33.2, 19.8, edge=PURPLE, face="none", lw=1.3,
          dashed=True, rounding=0.9, z=1)
    ax.text(117.6, 69.9, "$\\times L$ blocks, weights shared", ha="center",
            va="center", fontsize=6.9, color=PURPLE)

    # ---- ghost padded tensor --------------------------------------------- #
    rrect(ax, 3.0, 19.5, 55.0, 14.0, edge=MUTED, face="#F4F4F4", lw=1.3,
          dashed=True, rounding=0.7, z=1)
    ax.plot([3.6, 57.4], [20.0, 33.0], color=RED, lw=2.4, alpha=0.85, zorder=5)
    ax.text(30.5, 26.5, "the padded tensor $[B,\\,C_{\\max},\\,N,\\,D]$ is never materialised",
            ha="center", va="center", fontsize=7.6, color=MUTED, style="italic",
            fontweight="bold", zorder=6)

    # ---- inference branch ------------------------------------------------- #
    rrect(ax, 80.0, 6.0, 26.0, 34.0, edge=BLUE, face=tint(BLUE, 0.94), lw=1.3,
          dashed=True, rounding=0.9, z=1)
    ax.text(81.0, 39.2, "encode() — inference (no masking)", ha="left",
            va="center", fontsize=6.9, color=BLUE)
    arrow(ax, (93.0, 51.0), (93.0, 35.8), color=BLUE, lw=1.4, dashed=True)
    box(ax, 82.5, 26.0, 21.0, 9.0, "PMA pooler", "learned query",
        edge=BLUE, title_size=7.6, sub_size=6.2)
    arrow(ax, (93.0, 26.0), (93.0, 21.0), color=BLUE, lw=1.4)
    rrect(ax, 84.5, 8.5, 17.0, 11.0, edge=BLUE, face=tint(BLUE, 0.72), lw=1.5)
    ax.text(93.0, 14.0, "$[\\,B,\\; D\\,]$", ha="center", va="center",
            fontsize=9.0, fontweight="bold", color=INK)
    ax.text(93.0, 10.2, "fixed-size descriptor", ha="center", va="center",
            fontsize=6.4, color=MUTED)

    # ---- decoder output -> loss ------------------------------------------ #
    otop = draw_output(ax, 150.0)
    arrow(ax, (153.0, 51.0), (153.0, otop), color=VIOLET, lw=1.5)
    box(ax, 148.0, 16.0, 26.0, 10.0, "patch-normalised MSE",
        "channel-masked reconstruction", edge=RED, title_size=7.6, sub_size=6.2)
    arrow(ax, (161.0, 30.0), (161.0, 26.0), color=RED, lw=1.5)

    # ---- target pixels, routed along the bottom --------------------------- #
    ax.plot([1.6, 1.6, 161.0, 161.0], [47.6, 1.3, 1.3, 16.0],
            color=MUTED, lw=1.2, linestyle="--", zorder=2)
    arrow(ax, (161.0, 3.0), (161.0, 16.0), color=MUTED, lw=1.2, dashed=True,
          ms=10)
    ax.text(80.0, 3.1, "target = raw pixels of the masked channel-patches",
            ha="center", va="bottom", fontsize=7.0, color=MUTED, style="italic")

    # ---- footnote --------------------------------------------------------- #
    ax.text(2.0, 78.8,
            "The encoder only ever sees visible tokens; the decoder is conditioned on "
            "the target channel's identity — one shared decoder, no padding, no mask tokens.",
            ha="left", va="center", fontsize=7.2, color=MUTED, style="italic")

    return fig


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--formats", nargs="+", default=["pdf", "svg", "png"])
    ap.add_argument("--stem", default="level1_system")
    args = ap.parse_args()

    install_rcparams()

    fig = build()
    save_figure(fig, OUT, args.stem, args.formats)
    plt.close(fig)


if __name__ == "__main__":
    main()
