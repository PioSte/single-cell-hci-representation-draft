"""Step-by-step, spatial explanation of channel-count independence.

Companion to ``paper_figure.py``.  It shows the *same* computation as a
filmstrip of consecutive steps and makes the cardinality argument spatial:

* the channel axis is a variable-height axis that grows with ``C``;
* channel fusion consumes that axis **once per spatial location**, producing a
  single ``[D]`` vector per present location;
* the encoder mixes those location tokens with the fixed memory set;
* spatial pooling to fixed size latent then produces a fixed ``[B, L, D]``
  latent set, independent of both ``C`` and the patch count ``N``.

Outputs ``viz/out/cardinality.{pdf,svg,png}``.

Usage::

    uv run python viz/cardinality_figure.py
    uv run python viz/cardinality_figure.py --formats pdf png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from figure_style import (  # noqa: E402
    INK, MUTED, GREEN, CHANNEL_COLORS, channel_color,
    TOKENIZER, FUSION, ENCODER, MEMORY, POOL, FUSED,
    tint, rrect, arrow, panel_header, padded_ghost,
    install_rcparams, save_figure, legend,
)

OUT = Path(__file__).resolve().parent / "out"

# --------------------------------------------------------------------------- #
# Local helpers (thin wrappers over the shared primitives)
# --------------------------------------------------------------------------- #
def cell(ax, x, y, s, edge, face=None, crossed=False, lw=1.0, r=0.35,
         z=2, dashed=False):
    rrect(ax, x, y, s, s, edge=edge,
          face=face if face is not None else tint(edge, 0.8),
          lw=lw, rounding=r, z=z, crossed=crossed, dashed=dashed)


def label(ax, x, y, s, size=6.6, color=MUTED, weight="normal",
          ha="center", va="center", style="normal", z=5):
    ax.text(x, y, s, fontsize=size, color=color, fontweight=weight,
            ha=ha, va=va, style=style, zorder=z)


def frame(ax, x, y, w, h, title, subtitle):
    rrect(ax, x, y, w, h, edge="#D5D5D5", face="#FBFBFB", lw=1.2,
          rounding=0.9, z=1)
    label(ax, x + w / 2, y + h - 2.4, title, size=9.2,
          color=INK, weight="bold", z=3)
    label(ax, x + w / 2, y + h - 5.2, subtitle, size=6.8,
          color=MUTED, z=3)


# --------------------------------------------------------------------------- #
# Row A -- the filmstrip
# --------------------------------------------------------------------------- #
FRAME_W = 21.0
FRAME_XS = [1.0, 24.0, 47.0, 70.0, 93.0]
FRAME_Y0, FRAME_H = 39.0, 37.0


def step1(ax, x0):
    s = 4.2
    cx = x0 + 4.2
    for i in range(5):
        y = 63.5 - i * 4.6
        col = channel_color(i)
        cell(ax, cx, y, s, col, face=tint(col, 0.8), r=0.5)
        label(ax, cx + s + 0.7, y + s / 2, f"$c_{{{i+1}}}$", size=6.4,
              color=col, weight="bold", ha="left")
    label(ax, x0 + 10.5, 43.2, "input set $[C_b,\\,1,\\,H,\\,W]$",
          size=6.8, color=INK, weight="bold")
    label(ax, x0 + 10.5, 40.8, "one image per channel $\\cdot$ variable $C$",
          size=6.0, color=MUTED)


def _token_grid(ax, x0, rows, cols, s, pitch, top, labels=False):
    gx = x0 + (FRAME_W - ((cols - 1) * pitch + s)) / 2
    for r in range(rows):
        y = top - r * pitch
        col = channel_color(r)
        for c in range(cols):
            cell(ax, gx + c * pitch, y, s, col, crossed=True, lw=0.9)
        if labels:
            label(ax, gx - 0.9, y + s / 2, f"$c_{{{r+1}}}$", size=6.0,
                  color=col, weight="bold", ha="right")
    return gx


def step2(ax, x0):
    _token_grid(ax, x0, rows=5, cols=4, s=2.5, pitch=2.9, top=65.0,
                labels=True)
    label(ax, x0 + 10.5, 43.2, "token set $[\\Sigma C\\cdot N,\\; D]$",
          size=6.8, color=TOKENIZER, weight="bold")
    label(ax, x0 + 10.5, 40.8,
          "shared $1{\\to}D$ conv $\\cdot$ carry $z_c$",
          size=6.0, color=MUTED)


def step3(ax, x0):
    s, pitch, top = 2.5, 2.9, 65.0
    gx = _token_grid(ax, x0, rows=5, cols=4, s=s, pitch=pitch, top=top)
    fs, fpitch, fy = 2.6, 3.0, 48.0
    fx = x0 + (FRAME_W - ((3 * fpitch) + fs)) / 2
    for c in range(4):
        cx = gx + c * pitch + s / 2
        arrow(ax, (cx, top - 4 * pitch - 0.2),
              (fx + c * fpitch + fs / 2, fy + fs + 0.3),
              color=FUSION, lw=1.0, ms=7)
    for c in range(4):
        cell(ax, fx + c * fpitch, fy, fs, FUSED, crossed=True, lw=1.0)
    label(ax, x0 + 10.5, 45.6,
          "fused $[K_f,D]$ $\\cdot$ 1 per present location",
          size=6.2, color=FUSED, weight="bold")
    label(ax, x0 + 10.5, 43.0, "$C$ summed away $\\cdot$ $O(C)$ per location",
          size=6.2, color=FUSION, weight="bold")
    label(ax, x0 + 10.5, 40.8,
          "$z_c$ routes as bias $+$ skip, never added",
          size=6.0, color=MUTED)


def step4(ax, x0):
    s, pitch, n = 2.4, 2.8, 4
    w = (n - 1) * pitch + s
    mx = x0 + (FRAME_W - w) / 2
    my, ty = 64.5, 60.3
    for i in range(n):
        cell(ax, mx + i * pitch, my, s, MEMORY, crossed=True, lw=0.9)
    label(ax, x0 + 10.5, my + s + 1.3,
          "memory $[B,M,D]$ $\\cdot$ $M{=}4$",
          size=6.0, color=MEMORY, weight="bold")
    for i in range(n):
        cell(ax, mx + i * pitch, ty, s, FUSED, crossed=True, lw=0.9)
    label(ax, x0 + 10.5, ty - 1.5,
          "$\\mathrm{tok}\\;[K_f,D]$ content", size=6.0, color=FUSED,
          weight="bold")
    # brace: memory + tokens are concatenated before the encoder
    xb = x0 + FRAME_W - 2.6
    ax.plot([xb, xb], [my + s / 2, ty + s / 2], color=ENCODER, lw=1.2, zorder=4)
    for yy in (my + s / 2, ty + s / 2):
        ax.plot([mx + w, xb], [yy, yy], color=ENCODER, lw=1.2, zorder=4)
    # encoder box
    bx, bw, by, bh = x0 + 1.5, FRAME_W - 3.0, 45.0, 8.0
    rrect(ax, bx, by, bw, bh, edge=ENCODER, face=tint(ENCODER), lw=1.5,
          rounding=0.7, z=3)
    label(ax, bx + bw / 2, by + bh * 0.68, "Transformer encoder",
          size=7.4, color=INK, weight="bold")
    label(ax, bx + bw / 2, by + bh * 0.30,
          "cat$(\\mathrm{mem},\\mathrm{tok})$ $[B\\cdot M{+}K_f,D]$ $\\cdot$ RoPE",
          size=5.3, color=MUTED)
    arrow(ax, (xb, (my + ty) / 2 + s / 2), (xb, by + bh + 0.3),
          color=ENCODER, lw=1.3, ms=9)
    label(ax, x0 + 10.5, 43.2, "$\\mathrm{mem}_{out}$ $+$ $\\mathrm{tok}_{out}$",
          size=6.2, color=ENCODER, weight="bold")
    label(ax, x0 + 10.5, 40.8,
          "$\\mathrm{tok}_{out}\\notin$ decoder KV", size=6.0, color=MUTED)


def step5(ax, x0):
    s, pitch, n = 2.4, 2.8, 4
    w = (n - 1) * pitch + s
    tx = x0 + (FRAME_W - w) / 2
    ty = 63.8
    for i in range(n):
        cell(ax, tx + i * pitch, ty, s, FUSED, crossed=True, lw=0.9)
    label(ax, x0 + 10.5, ty + s + 1.3,
          "$\\mathrm{tok}_{out}\\,[K_f,D]$ content only",
          size=6.0, color=FUSED, weight="bold")
    # one PMA pooler: the learned latent queries
    py, ph = 52.0, 6.6
    bx_, bw_ = x0 + 1.2, 18.6
    rrect(ax, bx_, py, bw_, ph, edge=POOL, face=tint(POOL), lw=1.4,
          rounding=0.6, z=3)
    label(ax, bx_ + bw_ / 2, py + ph * 0.66, "PMA", size=7.6,
          color=INK, weight="bold")
    label(ax, bx_ + bw_ / 2, py + ph * 0.28, "$n_q{=}L{=}16$", size=6.0,
          color=MUTED)
    arrow(ax, (tx + w / 2, ty - 0.1), (bx_ + bw_ / 2, py + ph + 0.3),
          color=POOL, lw=1.2, ms=8)
    # output: the fixed latent set
    oy, os_, opitch = 45.4, 2.2, 2.4
    lx2 = x0 + 2.4
    for i in range(4):
        cell(ax, lx2 + i * opitch, oy, os_, FUSED, crossed=True, lw=0.8)
    label(ax, x0 + 5.9, oy - 1.6, "$[B,L,D]$ latents", size=6.2,
          color=FUSED, weight="bold")
    label(ax, x0 + 10.5, 42.4,
          "decoder KV $=$ $\\mathrm{mem}_{out}{+}$latents",
          size=5.4, color=MUTED)
    label(ax, x0 + 10.5, 40.4,
          "fixed size $\\cdot$ independent of $C$ and $N$",
          size=6.2, color=GREEN, weight="bold")


def draw_filmstrip(ax):
    panel_header(ax, 1.0, 78.0, "A",
                 "Consecutive steps: the channel axis is consumed once per location")
    frames = [
        ("1 $\\cdot$ input set", "one single-channel image per channel", step1),
        ("2 $\\cdot$ tied embed $\\cdot$ identity",
         "shared $1{\\to}D$ conv; carry $z_c$", step2),
        ("3 $\\cdot$ channel fusion", "pool over channels per location", step3),
        ("4 $\\cdot$ encoder $+$ memory",
         "memory $+$ locations through attention", step4),
        ("5 $\\cdot$ spatial pooling", "fixed latent set", step5),
    ]
    for x0, (title, sub, fn) in zip(FRAME_XS, frames):
        frame(ax, x0, FRAME_Y0, FRAME_W, FRAME_H, title, sub)
        fn(ax, x0)
    for i in range(4):
        arrow(ax, (FRAME_XS[i] + FRAME_W + 0.2, 57.0),
              (FRAME_XS[i + 1] - 0.2, 57.0), color=INK, lw=2.0, ms=11)


# --------------------------------------------------------------------------- #
# Row B -- invariance across cardinalities
# --------------------------------------------------------------------------- #
def draw_invariance(ax):
    panel_header(ax, 1.0, 35.2, "B",
                 "Cardinality invariance: different $C$, identical output")
    label(ax, 1.0, 32.2,
          "the only place $C$ appears is the $\\Sigma C\\cdot N$ input length; "
          "it is summed away, never padded",
          size=7.2, color=INK, style="italic", ha="left")

    s, pitch, x0 = 2.8, 3.0, 6.0
    rows = [(1, 24.5, channel_color(0)),
            (3, 18.0, channel_color(1)),
            (5, 11.5, channel_color(2))]
    bus_x = 26.0
    for c, yc, color in rows:
        for i in range(c):
            cell(ax, x0 + i * pitch, yc, s, color, face=tint(color, 0.8))
        label(ax, x0 - 1.4, yc + s / 2, f"$C_b{{=}}{c}$", size=7.0,
              color=INK, weight="bold", ha="right")
        row_right = x0 + (c - 1) * pitch + s
        label(ax, row_right + 0.5, yc + s + 0.5, "$\\Sigma C\\cdot N$",
              size=6.0, color=MUTED, ha="left")
        arrow(ax, (row_right + 0.3, yc + s / 2), (bus_x - 0.3, yc + s / 2),
              color=MUTED, lw=1.2, ms=8)
    mid_y = 19.4
    ax.plot([bus_x, bus_x], [11.5 + s / 2, 24.5 + s / 2],
            color=MUTED, lw=1.2, zorder=4)
    arrow(ax, (bus_x, mid_y), (30.0 - 0.3, mid_y), color=INK, lw=1.6, ms=10)

    fy, fh = 13.0, 11.0
    fx, fw = 30.0, 19.0
    rrect(ax, fx, fy, fw, fh, edge=FUSION, face=tint(FUSION), lw=1.5,
          rounding=0.7, z=3)
    label(ax, fx + fw / 2, fy + fh * 0.68, "channel fusion", size=8.6,
          color=INK, weight="bold")
    label(ax, fx + fw / 2, fy + fh * 0.30,
          "PMA over channels $\\to [K_f,D]$", size=6.3, color=MUTED)
    arrow(ax, (fx + fw + 0.2, fy + fh / 2), (53.0 - 0.3, fy + fh / 2),
          color=INK, lw=1.6, ms=10)

    ex, ew = 53.0, 21.0
    rrect(ax, ex, fy, ew, fh, edge=POOL, face=tint(POOL), lw=1.5,
          rounding=0.7, z=3)
    label(ax, ex + ew / 2, fy + fh * 0.68, "encoder $+$ spatial pool",
          size=8.2, color=INK, weight="bold")
    label(ax, ex + ew / 2, fy + fh * 0.30,
          "$\\to [B,L,D]$ latents", size=6.3, color=MUTED)

    padded_ghost(ax, 40.0, 27.2, 21.0, 4.2,
                 shape="$[B,\\,C_{\\max},\\,N,\\,D]$",
                 note="never materialised")
    label(ax, 38.0, 9.5,
          "identical for every channel count $\\cdot$ no $C_{\\max}$ token, "
          "no padding", size=7.2, color=GREEN, weight="bold")


# --------------------------------------------------------------------------- #
# Glossary -- roles + every symbol
# --------------------------------------------------------------------------- #
def _glossary(ax, x, y_top, title, entries, sym_w=6.2, pitch=2.15):
    label(ax, x, y_top, title, size=7.4, color=INK, weight="bold", ha="left")
    y = y_top - 2.3
    for sym, meaning in entries:
        label(ax, x, y, sym, size=6.4, color=INK, weight="bold", ha="left")
        label(ax, x + sym_w, y, meaning, size=5.9, color=MUTED, ha="left")
        y -= pitch
    return y


def draw_glossary(ax):
    legend(ax, 79.0, 35.0,
           entries=[
               ("TOKENIZER  patch embed / $z_c$", TOKENIZER, "fill"),
               ("FUSION  channel fusion", FUSION, "fill"),
               ("ENCODER  transformer encoder", ENCODER, "fill"),
               ("MEMORY  memory token set", MEMORY, "fill"),
               ("POOL  spatial pooling", POOL, "fill"),
               ("FUSED  fused $[D]$ vectors", FUSED, "fill"),
               ("CHANNEL  $c_i$ image / rows", channel_color(0), "cross"),
           ],
           title="module / data roles", pitch=2.6, sw=1.6, fontsize=6.0,
           title_size=7.4)
    _glossary(ax, 98.5, 35.0, "symbols", [
        ("$B$", "batch size"),
        ("$C$", "channel count (variable)"),
        ("$C_b$", "channels of sample $b$"),
        ("$H,W$", "image side (pixels)"),
        ("$\\Sigma C$", "total channels, batch"),
        ("$N$", "max patches / channel"),
        ("$D$", "token / embedding dim"),
        ("$K_f$", "present $(b,n)$ locations"),
        ("$M$", "memory tokens / sample"),
        ("$L$", "latents / sample"),
        ("$z_c$", "channel identity vector"),
        ("PMA", "attention pooling readout"),
    ])


# --------------------------------------------------------------------------- #
def build():
    fig = plt.figure(figsize=(15.0, 10.0))
    ax = fig.add_axes([0.005, 0.005, 0.99, 0.99])
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 80)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    draw_filmstrip(ax)
    draw_invariance(ax)
    draw_glossary(ax)
    return fig


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--formats", nargs="+", default=["pdf", "svg", "png"])
    ap.add_argument("--stem", default="cardinality")
    args = ap.parse_args()

    install_rcparams()
    fig = build()
    save_figure(fig, OUT, args.stem, tuple(args.formats))
    plt.close(fig)


if __name__ == "__main__":
    main()
