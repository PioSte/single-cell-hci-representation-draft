"""Step-by-step, spatial explanation of channel-count independence.

Companion to ``paper_figure.py``.  It shows the *same* computation as a
filmstrip of consecutive steps and makes the cardinality argument spatial:

* the channel axis is a variable-height axis that grows with ``C``;
* channel fusion consumes that axis **once per spatial location**, producing a
  fixed ``[B, N, D]`` grid;
* everything downstream therefore has a shape and cost independent of ``C``.

Outputs ``viz/out/cardinality.{pdf,svg,png}``.

Usage::

    uv run python viz/cardinality_figure.py
    uv run python viz/cardinality_figure.py --formats pdf png
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from paper_figure import (
    BLUE,
    GREEN,
    INK,
    MUTED,
    ORANGE,
    RED,
    SKY,
    VIOLET,
    OUT,
    arrow,
    node,
    tint,
)

CH = [BLUE, GREEN, ORANGE, RED, VIOLET, SKY]


def cell(ax, x, y, s, fc, ec, ls="solid", lw=1.0, r=0.35, z=2):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), s, s,
            boxstyle=f"round,pad=0.0,rounding_size={r}",
            linewidth=lw, edgecolor=ec, facecolor=fc, linestyle=ls, zorder=z,
        )
    )


def label(ax, x, y, s, size=7.5, color=MUTED, weight="normal",
          ha="center", va="center", style="normal", z=3):
    ax.text(x, y, s, fontsize=size, color=color, fontweight=weight,
            ha=ha, va=va, style=style, zorder=z)


def frame(ax, x, y, w, h, title, subtitle):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.0,rounding_size=0.9",
            linewidth=1.2, edgecolor="#D5D5D5", facecolor="#FBFBFB", zorder=1,
        )
    )
    label(ax, x + w / 2, y + h - 2.2, title, size=10.0,
          color=INK, weight="bold", z=3)
    label(ax, x + w / 2, y + h - 5.2, subtitle, size=7.2,
          color=MUTED, z=3)


# --------------------------------------------------------------------------- #
# Row A -- the filmstrip
# --------------------------------------------------------------------------- #
def step1(ax, x0):
    for i in range(5):
        y = 35.2 + (4 - i) * 4.6
        cell(ax, x0 + 9.0, y, 4.4, tint(CH[i], 0.75), CH[i], r=0.5)
        label(ax, x0 + 14.0, y + 2.2, f"$c_{{{i+1}}}$", size=7.2,
              color=CH[i], weight="bold", ha="left")
    label(ax, x0 + 13.5, 33.4,
          "$\\vdots$  input set $[C,\\,1,\\,H,\\,W]$ · $C=1\\dots60{+}$",
          size=7.6, color=INK)


def step2(ax, x0):
    rows, cols = 5, 4
    s, pitch = 3.6, 4.0
    gx = x0 + (27 - cols * pitch) / 2
    gy = 38.0
    for r in range(rows):
        for c in range(cols):
            cell(ax, gx + c * pitch, gy + (rows - 1 - r) * pitch, s,
                 tint(CH[r], 0.8), CH[r])
        label(ax, gx - 1.3, gy + (rows - 1 - r) * pitch + s / 2, f"$c_{{{r+1}}}$",
              size=6.6, color=CH[r], ha="right")
    label(ax, x0 + 13.5, 35.6,
          "rows $=C$ channels · cols $=N$ spatial patches", size=6.9, color=MUTED)
    label(ax, x0 + 13.5, 33.4, "$[\\,\\Sigma C\\cdot N,\\; D\\,]$ token set",
          size=7.6, color=INK)


def step3(ax, x0):
    rows, cols = 5, 4
    s, pitch = 3.0, 3.4
    gx = x0 + (27 - cols * pitch) / 2
    gy = 44.0
    for r in range(rows):
        for c in range(cols):
            cell(ax, gx + c * pitch, gy + (rows - 1 - r) * pitch, s,
                 tint(CH[r], 0.8), CH[r], lw=0.9)
    for c in range(cols):
        cx = gx + c * pitch + s / 2
        arrow(ax, (cx, gy - 0.9), (cx, 39.4), color=RED, lw=1.1, ms=8)
        cell(ax, gx + c * pitch, 36.0, s, tint(RED, 0.82), RED)
    label(ax, x0 + 13.5, 34.2,
          "$C$ tokens $\\to$ 1 per $(b,n)$ · $O(C)$ per location",
          size=7.0, color=RED, weight="bold")


def step4(ax, x0):
    s, pitch = 5.0, 5.7
    fx, fy = x0 + 3.5, 49.0
    for r in range(2):
        for c in range(2):
            cell(ax, fx + c * pitch, fy + (1 - r) * pitch, s,
                 tint(RED, 0.82), RED, r=0.6)
    label(ax, fx + pitch, fy - 1.8, "fused $[B,N,D]$", size=7.4,
          color=RED, weight="bold")
    for m in range(3):
        cell(ax, fx + m * 3.0, 42.0, 2.4, tint(SKY, 0.75), SKY, r=0.4)
    label(ax, fx + 4.1, 40.0, "$+M$ memory", size=6.8, color=SKY, weight="bold")
    arrow(ax, (fx + 2 * pitch + 0.5, fy + pitch), (x0 + 22.6, fy + pitch),
          color=INK, lw=1.5)
    ax.add_patch(
        FancyBboxPatch(
            (x0 + 23.0, fy + 0.2), 3.0, 9.6,
            boxstyle="round,pad=0.0,rounding_size=0.5",
            linewidth=1.4, edgecolor=VIOLET, facecolor=tint(VIOLET, 0.8), zorder=2,
        )
    )
    label(ax, x0 + 24.5, fy - 1.8, "$[B,D]$", size=7.4, color=VIOLET, weight="bold")
    label(ax, x0 + 13.5, 34.2, "shape & size independent of $C$",
          size=8.2, color=GREEN, weight="bold")


def draw_filmstrip(ax):
    label(ax, 1, 70.0, "A", size=15, color=INK, weight="bold", ha="left")
    label(ax, 3.6, 70.0,
          "Consecutive steps: the channel axis is consumed once per location",
          size=12.0, color=INK, weight="bold", ha="left")
    xs = [1.0 + i * 29.5 for i in range(4)]
    frames = [
        ("1 · input set", "one single-channel image per marker", step1),
        ("2 · tied embed $+$ identity", "shared $1{\\to}D$ conv $+\\,z_c$", step2),
        ("3 · channel fusion", "pool over channels per location", step3),
        ("4 · spatial path", "transformer $+$ descriptor", step4),
    ]
    for x0, (title, sub, fn) in zip(xs, frames):
        frame(ax, x0, 32.0, 27.0, 35.0, title, sub)
        fn(ax, x0)
    for i in range(3):
        arrow(ax, (xs[i] + 27.2, 48.0), (xs[i + 1] - 0.4, 48.0),
              color=INK, lw=2.0, ms=12)


# --------------------------------------------------------------------------- #
# Row B -- invariance across cardinalities
# --------------------------------------------------------------------------- #
def draw_invariance(ax):
    label(ax, 1, 29.5, "B", size=15, color=INK, weight="bold", ha="left")
    label(ax, 3.6, 29.5,
          "Cardinality invariance: different $C$, identical output",
          size=12.0, color=INK, weight="bold", ha="left")
    label(ax, 1.0, 26.4,
          "the only place $C$ appears is the $\\Sigma C\\cdot N$ input length; "
          "it is summed away, never padded",
          size=8.0, color=INK, style="italic", ha="left")

    s, pitch = 3.4, 3.8
    x0 = 9.0
    rows = [(1, 21.0, BLUE), (3, 14.0, GREEN), (5, 7.0, ORANGE)]
    bus_x = 48.0
    for c, yc, color in rows:
        for i in range(c):
            cell(ax, x0 + i * pitch, yc - s / 2, s, tint(color, 0.78), color, r=0.4)
        label(ax, x0 - 1.6, yc, f"$C={c}$", size=8.0, color=INK,
              weight="bold", ha="right")
        row_right = x0 + (c - 1) * pitch + s
        label(ax, row_right + 1.2, yc + 2.6, "$\\Sigma C\\cdot N$",
              size=6.6, color=MUTED, ha="left")
        arrow(ax, (row_right + 0.6, yc), (bus_x - 0.4, yc),
              color=MUTED, lw=1.3, ms=9)
    ax.plot([bus_x, bus_x], [7.0, 21.0], color=MUTED, lw=1.3, zorder=4)
    arrow(ax, (bus_x, 14.0), (52.0, 14.0), color=INK, lw=1.8, ms=11)

    node(ax, 52.5, 8.0, 30.0, 12.0,
         "channel fusion", "PMA / identity-gated $\\to$ $[B,N,D]$",
         edge=RED, title_size=9.0, sub_size=7.2)
    arrow(ax, (82.7, 14.0), (86.0, 14.0), color=INK, lw=1.8, ms=11)
    node(ax, 86.5, 8.5, 23.0, 11.0,
         "descriptor", "$[B,D]$", edge=VIOLET, title_size=9.0, sub_size=7.6)

    label(ax, 73.0, 4.2,
          "identical for every channel count — no $C_{\\max}$, no padding",
          size=7.8, color=GREEN, weight="bold")


# --------------------------------------------------------------------------- #
def build():
    fig = plt.figure(figsize=(15.0, 9.0))
    ax = fig.add_axes([0.005, 0.005, 0.99, 0.99])
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 72)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    draw_filmstrip(ax)
    draw_invariance(ax)
    return fig


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--formats", nargs="+", default=["pdf", "svg", "png"])
    ap.add_argument("--stem", default="cardinality")
    args = ap.parse_args()

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "mathtext.fontset": "dejavusans",
        }
    )

    fig = build()
    for fmt in args.formats:
        path = OUT / f"{args.stem}.{fmt}"
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"wrote {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
