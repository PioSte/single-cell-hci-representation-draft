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

# Vertical centre of the whole training track: every array stack is centred here
# and every main-flow arrow runs along this line.
YMID = 46.0


# --------------------------------------------------------------------------- #
# Array stacks  (all vertically centred on YMID)
# --------------------------------------------------------------------------- #
def draw_input_set(ax):
    """Ragged set of single-channel images -- the variable axis is the height."""
    cw, pitch, x = 6.8, 5.6, 3.0
    n = len(CHANNEL_COLORS)
    y0 = YMID - (n - 1) * pitch / 2 - cw / 2
    for i, c in enumerate(CHANNEL_COLORS):
        yy = y0 + (n - 1 - i) * pitch
        rrect(ax, x, yy, cw, cw, edge=c, face=tint(c, 0.72), lw=1.5)
        # a faint "image" squiggle so the tiles read as pictures
        ax.plot([x + 1.0, x + cw - 1.0], [yy + 1.9, yy + cw - 2.2],
                color=c, lw=1.0, alpha=0.7, zorder=4)
        ax.plot([x + 1.0, x + cw - 1.0], [yy + cw - 2.2, yy + 1.9],
                color=c, lw=1.0, alpha=0.4, zorder=4)
    top = y0 + (n - 1) * pitch + cw
    ax.text(x + cw / 2, top + 2.6, "input set", ha="center", va="center",
            fontsize=8.4, fontweight="bold", color=INK)
    ax.text(x + cw / 2, y0 - 2.6, "$C_b$ markers\norder-free", ha="center",
            va="center", fontsize=6.6, color=MUTED, linespacing=1.35)
    ax.text(x - 1.0, YMID, "$1\\times H\\times W$", ha="right", va="center",
            fontsize=6.8, color=MUTED, rotation=90)
    ax.text(x + cw + 0.7, y0 + 1.8, "$\\vdots$", ha="left", va="center",
            fontsize=10, color=MUTED)
    ax.text(x + cw + 0.7, top - 2.0, "$\\vdots$", ha="left", va="center",
            fontsize=10, color=MUTED)


def draw_ragged_tokens(ax):
    """Flattened ``[sum C . N, D]`` token set -- one row per channel-patch."""
    x, w, h, pitch, rows = 41.0, 10.5, 2.6, 3.4, 8
    total = (rows - 1) * pitch + h
    y0 = YMID - total / 2
    for i in range(rows):
        yy = y0 + i * pitch
        c = CHANNEL_COLORS[(i // 2) % len(CHANNEL_COLORS)]
        omitted = i == 5
        rrect(ax, x, yy, w, h, edge=c,
              face=("#FFFFFF" if omitted else tint(c, 0.8)),
              lw=1.2, dashed=omitted)
        if omitted:
            ax.text(x + w / 2, yy + h / 2, "omitted row (DCP)", ha="center",
                    va="center", fontsize=6.0, color=RED, style="italic",
                    zorder=4)
    top = y0 + total
    sq_bracket(ax, x + w + 1.0, y0, top)
    ax.text(x + w + 1.6, top, "$[\\,\\Sigma C\\cdot N,\\; D\\,]$",
            ha="left", va="center", fontsize=8.0, color=INK)
    ax.text(x + w / 2, top + 2.6, "ragged token set", ha="center", va="center",
            fontsize=8.4, fontweight="bold", color=INK)


def draw_joint_grid(ax, x0=79.0):
    """``[B, N, D]`` joint per-patch embeddings after channel fusion."""
    cell, pitch = 2.9, 3.3
    B, N = 3, 4
    gh = (B - 1) * pitch + cell
    y0 = YMID - gh / 2
    for b in range(B):
        for n in range(N):
            rrect(ax, x0 + n * pitch, y0 + b * pitch, cell, cell,
                  edge=BLUE, face=tint(BLUE, 0.72), lw=1.1, rounding=0.3)
    top = y0 + gh
    ax.text(x0 + (N - 1) * pitch / 2 + cell / 2, top + 1.0, "$N$ patches",
            ha="center", va="center", fontsize=6.6, color=MUTED)
    ax.text(x0 - 0.7, YMID - 3.2, "$B$", ha="right", va="center",
            fontsize=6.6, color=MUTED)
    ax.text(x0 + (N - 1) * pitch / 2 + cell / 2, top + 5.4, "joint embeddings",
            ha="center", va="center", fontsize=8.4, fontweight="bold", color=INK)
    ax.text(x0 + (N - 1) * pitch / 2 + cell / 2, top + 3.1, "$[\\,B,\\,N,\\,D\\,]$",
            ha="center", va="center", fontsize=6.4, color=BLUE)
    return x0 + (N - 1) * pitch + cell


def draw_encoded(ax, x=124.0):
    """``[B . M + K_f, D]`` -- fixed-count memory + visible tokens."""
    w, h, pitch = 10.5, 2.6, 3.4
    rows = [("tok", BLUE)] * 5 + [("mem", ORANGE)] * 2
    total = (len(rows) - 1) * pitch + h
    y0 = YMID - total / 2
    for i, (kind, c) in enumerate(rows):
        yy = y0 + i * pitch
        rrect(ax, x, yy, w, h, edge=c, face=tint(c, 0.8), lw=1.2)
    top = y0 + total
    sq_bracket(ax, x + w + 1.0, y0, top)
    ax.text(x + w / 2, top + 2.0, "$[\\,B\\cdot M + K_f,\\; D\\,]$",
            ha="center", va="center", fontsize=7.4, color=INK)
    ax.text(x + w / 2, top + 4.6, "encoder memory", ha="center", va="center",
            fontsize=8.4, fontweight="bold", color=INK)
    ax.text(x - 0.8, y0 + 5.8 * pitch + h / 2, "$M$", ha="right", va="center",
            fontsize=6.2, color=ORANGE)
    ax.text(x - 0.8, y0 + 2 * pitch + h / 2, "$K_f$", ha="right", va="center",
            fontsize=6.2, color=BLUE)


def draw_output(ax, x, w=21.0):
    """``[K_t, P^2]`` predicted pixels for the omitted channel-patches."""
    h, pitch, rows = 2.6, 3.5, 4
    total = (rows - 1) * pitch + h
    y0 = 31.0 - total          # top of the stack sits at y = 31
    for i in range(rows):
        rrect(ax, x, y0 + i * pitch, w, h, edge=VIOLET,
              face=tint(VIOLET, 0.82), lw=1.2)
    top = y0 + total
    return top, y0


# --------------------------------------------------------------------------- #
# The figure
# --------------------------------------------------------------------------- #
def build() -> plt.Figure:
    fig = plt.figure(figsize=(16.0, 7.5))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_xlim(0, 164)
    ax.set_ylim(0, 76)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    # ---- title ------------------------------------------------------------ #
    ax.text(2, 72.4, "L1", ha="left", va="center", fontsize=15,
            fontweight="bold", color=INK)
    ax.text(6.5, 72.4, "System: padding-free channel-fusion MAE (training path + inference)",
            ha="left", va="center", fontsize=12.2, fontweight="bold", color=INK)
    ax.text(2.0, 68.2,
            "The encoder only ever sees visible tokens; the decoder is conditioned on "
            "the target channel's identity — one shared decoder, no padding, no mask tokens.",
            ha="left", va="center", fontsize=7.2, color=MUTED, style="italic")

    # ---- arrays ----------------------------------------------------------- #
    draw_input_set(ax)
    draw_ragged_tokens(ax)
    grid_right = draw_joint_grid(ax)
    draw_encoded(ax)

    # ---- module boxes (all centred on YMID) ------------------------------- #
    box(ax, 15, YMID - 7.5, 22, 15, "Channel-tied patch embed",
        "shared Conv $1{\\to}D$, stride $P$\n$+$ shared spatial pos $+\\, z_c$",
        edge=GREEN, sub_size=6.4)
    box(ax, 56, YMID - 7.5, 19, 15, "Channel fusion",
        "pool over channel axis\nper $(b,n)$ location\nfixed-query PMA $O(C)$ · hypernet $O(C)$",
        edge=RED, sub_size=5.9)
    box(ax, 95, YMID - 7.5, 24, 15, "Spatial transformer",
        "grouped self-attention\nover real tokens\n$+M$ fixed memory tokens",
        edge=PURPLE, sub_size=6.1)
    box(ax, 138, YMID - 7.5, 21, 15, "Channel-aware decoder",
        "one shared decoder\n$q=\\mathrm{mask} + \\mathrm{pos}_n + Wz_c$\n"
        "cross-attn $\\to$ pixels", edge=VIOLET, sub_size=5.7)

    # ---- arrows along the main track ------------------------------------- #
    arrow(ax, (9.8, YMID), (15.0, YMID))
    edge_label(ax, 12.0, YMID + 1.7, "$[\\Sigma C,1,H,W]$")
    arrow(ax, (37.0, YMID), (41.0, YMID))
    arrow(ax, (52.5, YMID), (56.0, YMID))
    arrow(ax, (75.0, YMID), (79.0, YMID))
    arrow(ax, (grid_right, YMID), (95.0, YMID))
    arrow(ax, (119.0, YMID), (124.0, YMID))
    arrow(ax, (134.5, YMID), (138.0, YMID))

    # ---- DCP masker chip feeds the patch embed ---------------------------- #
    box(ax, 15, 57.0, 22, 6.2, "DCP masker", "token omission (never zero)",
        edge=GREEN, title_size=7.6, sub_size=6.2)
    arrow(ax, (26.0, 57.0), (26.0, 53.5), color=GREEN, lw=1.3)

    # ---- shared-weight bracket over the transformer ----------------------- #
    rrect(ax, 93.0, YMID - 10.2, 28.0, 19.4, edge=PURPLE, face="none", lw=1.3,
          dashed=True, rounding=0.9, z=1)
    ax.text(107.0, YMID + 10.4, "$\\times L$ blocks, weights shared",
            ha="center", va="center", fontsize=6.9, color=PURPLE)

    # ---- ghost padded tensor (never materialised) ------------------------- #
    rrect(ax, 3.0, 6.0, 54.0, 13.0, edge=MUTED, face="#F4F4F4", lw=1.3,
          dashed=True, rounding=0.7, z=1)
    ax.plot([3.7, 56.3], [6.5, 18.5], color=RED, lw=2.4, alpha=0.85, zorder=5)
    ax.text(30.0, 12.6,
            "the padded tensor $[B,\\,C_{\\max},\\,N,\\,D]$ is never materialised",
            ha="center", va="center", fontsize=7.4, color=MUTED, style="italic",
            fontweight="bold", zorder=6)

    # ---- inference branch: encode() taps the encoder output --------------- #
    rrect(ax, 62.0, 6.0, 42.0, 24.0, edge=BLUE, face=tint(BLUE, 0.94), lw=1.3,
          dashed=True, rounding=0.9, z=1)
    ax.text(83.0, 28.5, "encode(): no masking, then pool", ha="center",
            va="center", fontsize=6.9, color=BLUE)
    # dashed elbow from the encoder-memory stack into the branch
    ax.plot([129.25, 129.25, 96.0], [34.5, 24.0, 24.0], color=BLUE, lw=1.3,
            linestyle="--", zorder=2)
    arrow(ax, (97.0, 24.0), (94.0, 24.0), color=BLUE, lw=1.4, dashed=True)
    box(ax, 72.0, 20.0, 22.0, 8.0, "PMA pooler", "learned query",
        edge=BLUE, title_size=7.6, sub_size=6.2)
    arrow(ax, (83.0, 20.0), (83.0, 16.0), color=BLUE, lw=1.4)
    rrect(ax, 74.0, 7.0, 18.0, 9.0, edge=BLUE, face=tint(BLUE, 0.72), lw=1.5)
    ax.text(83.0, 12.1, "$[\\,B,\\; D\\,]$", ha="center", va="center",
            fontsize=9.0, fontweight="bold", color=INK)
    ax.text(83.0, 8.6, "fixed-size descriptor", ha="center", va="center",
            fontsize=6.2, color=MUTED)

    # ---- decoder output -> loss ------------------------------------------ #
    otop, _obot = draw_output(ax, 138.0)
    arrow(ax, (148.5, YMID - 7.5), (148.5, otop), color=VIOLET, lw=1.5)
    edge_label(ax, 146.0, 33.6, "$[\\,K_t,\\; P^2\\,]$ predictions",
               color=INK, size=6.7, ha="right")
    box(ax, 138.0, 3.0, 21.0, 9.0, "patch-normalised MSE",
        "channel-masked reconstruction", edge=RED, title_size=7.4, sub_size=6.1)
    arrow(ax, (148.5, 17.9), (148.5, 12.0), color=RED, lw=1.5)

    # ---- target pixels, routed along the bottom --------------------------- #
    ax.plot([1.5, 1.5, 148.5, 148.5], [31.4, 1.0, 1.0, 3.0],
            color=MUTED, lw=1.2, linestyle="--", zorder=2)
    arrow(ax, (148.5, 1.6), (148.5, 3.0), color=MUTED, lw=1.2, dashed=True,
          ms=9)
    ax.text(74.0, 1.8, "target = raw pixels of the masked channel-patches",
            ha="center", va="bottom", fontsize=6.9, color=MUTED, style="italic")

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
