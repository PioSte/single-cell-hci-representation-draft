"""Shared design system for the fuse-channels architecture figures (L0-L8).

One place for the palette, module-box / arrow / array-stack primitives and the
channel colour table, so every figure reads as one authored set -- the same
grammar the reference architecture figures in ``references/`` use:

* literal array stacks with axis labels,
* rounded module boxes with a title + shape/role annotation,
* dashed stage / shared-weight frames,
* shapes written on the edges,
* the token list as the primary object and the padded tensor as a ghost.

Import from a figure script::

    from figure_style import (BLUE, GREEN, ORANGE, RED, VIOLET, INK, MUTED,
                              CHANNEL_COLORS, TOKENIZER, FUSION, ENCODER,
                              DECODER, MEMORY, LOSS, FUSED, POOL,
                              tint, rrect, box, arrow,
                              edge_label, sq_bracket, panel_header,
                              token_rows, grid_cells, install_rcparams, save_figure,
                              legend, role_legend, channel_legend, padded_ghost)

Okabe-Ito palette (colour-blind safe).

Role palette -- the same *role* always gets the same colour in every figure,
so the L0-L8 set reads as one authored system:

    TOKENIZER green   patch embed / tied Conv / spatial pos / channel identity z_c
    FUSION    orange  channel fusion (ChannelFusion + IdentityGatedSum)
    ENCODER   pink    transformer encoder + its memory-augmented blocks
    DECODER   violet  channel-aware decoder + reconstruction head
    MEMORY    gold    fixed memory / register token rows
    LOSS      red     objective / waste / -inf mask / omitted-row accent
    FUSED     blue    fused vectors, one [D] row per present location
    POOL      sky     pooling / descriptor reduction
    GHOST     muted   geometry, brackets, never-allocated padded tensor

Per-channel colours live in ``CHANNEL_COLORS`` and are used ONLY for
the channel axis (image tiles and channel-patch token rows).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.colors as mcolors
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

BLUE = "#0072B2"
GREEN = "#009E73"
ORANGE = "#E69F00"
SKY = "#56B4E9"
RED = "#D55E00"
PURPLE = "#CC79A7"
VIOLET = "#6A4C93"
INK = "#1A1A1A"
MUTED = "#5C5C5C"

# One colour per channel; reused for image borders, token rows and cards.
CHANNEL_COLORS = [BLUE, GREEN, ORANGE, PURPLE, VIOLET]

# --- semantic role palette (see module docstring) ------------------------- #
TOKENIZER = GREEN
FUSION = ORANGE
ENCODER = PURPLE
DECODER = VIOLET
MEMORY = "#B8860B"          # legible gold for the fixed memory / register rows
LOSS = RED
FUSED = BLUE
POOL = SKY
GHOST = MUTED


# --------------------------------------------------------------------------- #
# Primitives
# --------------------------------------------------------------------------- #
def channel_color(i: int) -> str:
    """Colour of channel ``i`` (cycles through the channel table)."""
    return CHANNEL_COLORS[i % len(CHANNEL_COLORS)]


def tint(color: str, amount: float = 0.84):
    """Mix ``color`` toward white by ``amount`` (fills for node bodies)."""
    r, g, b = mcolors.to_rgb(color)
    return (r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount)


def rrect(ax, x, y, w, h, edge, face="white", lw=1.1, dashed=False,
          rounding=0.4, z=3, crossed=False, cross_alpha=0.55):
    """Rounded rectangle; the workhorse cell / frame.

    ``crossed=True`` overlays the two diagonals used throughout the level set
    for **data** cells (tokens, embeddings, arrays).  Building-block/module
    frames stay uncrossed.
    """
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle=f"round,pad=0.0,rounding_size={rounding}",
            linewidth=lw, edgecolor=edge, facecolor=face,
            linestyle="--" if dashed else "solid", zorder=z,
        )
    )
    if crossed:
        inset = min(w, h) * 0.18
        ax.plot([x + inset, x + w - inset], [y + inset, y + h - inset],
                color=edge, lw=max(0.8, lw * 0.7), alpha=cross_alpha,
                zorder=z + 1, solid_capstyle="round")
        ax.plot([x + inset, x + w - inset], [y + h - inset, y + inset],
                color=edge, lw=max(0.8, lw * 0.7), alpha=cross_alpha * 0.8,
                zorder=z + 1, solid_capstyle="round")


def box(ax, x, y, w, h, title, sub="", edge=BLUE, title_size=8.4, sub_size=6.7):
    """Rounded module box with a bold title and a smaller shape/role note."""
    rrect(ax, x, y, w, h, edge=edge, face=tint(edge), lw=1.5, rounding=0.7)
    cx = x + w / 2
    if sub:
        ax.text(cx, y + h * 0.66, title, ha="center", va="center",
                fontsize=title_size, fontweight="bold", color=INK, zorder=4)
        ax.text(cx, y + h * 0.30, sub, ha="center", va="center",
                fontsize=sub_size, color=MUTED, linespacing=1.4, zorder=4)
    else:
        ax.text(cx, y + h / 2, title, ha="center", va="center",
                fontsize=title_size, fontweight="bold", color=INK, zorder=4)


def arrow(ax, p0, p1, color=INK, lw=1.7, style="-|>", ms=11, dashed=False,
          rad=0.0):
    ax.add_patch(
        FancyArrowPatch(
            p0, p1, arrowstyle=style, mutation_scale=ms, linewidth=lw,
            color=color, connectionstyle=f"arc3,rad={rad}",
            linestyle="--" if dashed else "solid",
            shrinkA=1.5, shrinkB=1.5, zorder=6,
        )
    )


def edge_label(ax, x, y, text, color=MUTED, size=6.6, ha="center"):
    """Small shape annotation sitting on an arrow."""
    ax.text(x, y, text, ha=ha, va="center", fontsize=size, color=color, zorder=7)


def sq_bracket(ax, x, y0, y1, color=MUTED, lw=1.2, tick=0.8):
    """Square bracket whose ticks point **right**, toward a stack on its left."""
    ax.plot([x, x], [y0, y1], color=color, lw=lw, zorder=3)
    ax.plot([x, x + tick], [y0, y0], color=color, lw=lw, zorder=3)
    ax.plot([x, x + tick], [y1, y1], color=color, lw=lw, zorder=3)


def sq_bracket_mirror(ax, x, y0, y1, color=MUTED, lw=1.2, tick=0.8):
    """Square bracket whose ticks point **left**, toward a stack on its right.

    Use this for a bracket drawn to the right of a token stack, so the bracket
    embraces the stack instead of pointing away from it.
    """
    ax.plot([x, x], [y0, y1], color=color, lw=lw, zorder=3)
    ax.plot([x, x - tick], [y0, y0], color=color, lw=lw, zorder=3)
    ax.plot([x, x - tick], [y1, y1], color=color, lw=lw, zorder=3)


def panel_header(ax, x, y, letter, text, color=INK):
    ax.text(x, y, letter, ha="left", va="center", fontsize=15,
            fontweight="bold", color=color)
    ax.text(x + 4.5, y, text, ha="left", va="center", fontsize=12.2,
            fontweight="bold", color=color)


# --------------------------------------------------------------------------- #
# Legends
# --------------------------------------------------------------------------- #
def legend(ax, x, y_top, entries, title=None, sw=1.5, pitch=2.5,
           fontsize=6.6, title_size=7.0, label_gap=1.0, label_color=INK):
    """Compact vertical legend drawn downward from ``(x, y_top)``.

    ``entries`` is a list of ``(label, color, style)``; ``style`` is one of

    * ``"fill"``  -- plain rounded cell (module / building block),
    * ``"cross"`` -- crossed rounded cell (data / token tensor),
    * ``"omit"``  -- dashed white cell (masked / absent row, never zero),
    * ``"ghost"`` -- dashed pale cell (never-allocated tensor),
    * ``"line"``  -- a short solid line swatch (curve / edge),
    * ``"dash"``  -- a short dashed line swatch.

    Returns the y of the last row's centre.
    """
    y = y_top
    if title:
        ax.text(x, y, title, ha="left", va="center", fontsize=title_size,
                fontweight="bold", color=INK, zorder=5)
        y -= pitch
    last = y
    for label, color, style in entries:
        cy = y - sw / 2
        last = cy
        if style in ("omit", "ghost"):
            rrect(ax, x, y - sw, sw, sw, edge=color,
                  face="#FFFFFF" if style == "omit" else tint(color, 0.94),
                  lw=1.1, dashed=True, rounding=0.2, z=4)
        elif style == "line":
            ax.plot([x, x + sw], [cy, cy], color=color, lw=2.2, zorder=4,
                    solid_capstyle="round")
        elif style == "dash":
            ax.plot([x, x + sw], [cy, cy], color=color, lw=2.2, ls="--",
                    zorder=4)
        elif style == "cross":
            rrect(ax, x, y - sw, sw, sw, edge=color, face=tint(color, 0.8),
                  lw=1.1, rounding=0.2, z=4, crossed=True)
        else:  # fill
            rrect(ax, x, y - sw, sw, sw, edge=color, face=tint(color, 0.8),
                  lw=1.1, rounding=0.2, z=4)
        ax.text(x + sw + label_gap, cy, label, ha="left", va="center",
                fontsize=fontsize, color=label_color, zorder=5)
        y -= pitch
    return last


def role_legend(ax, x, y_top, roles, title="module roles", **kw):
    """Legend for module roles: ``roles`` is ``[(label, color), ...]``."""
    entries = [(label, color, "fill") for label, color in roles]
    return legend(ax, x, y_top, entries, title=title, **kw)


def channel_legend(ax, x, y_top, n=None, title="channel identity",
                   labels=None, **kw):
    """Legend for the channel colour table (image tiles, token rows)."""
    n = n or len(CHANNEL_COLORS)
    entries = [
        (labels[i] if labels else f"channel {i}", channel_color(i), "cross")
        for i in range(n)
    ]
    return legend(ax, x, y_top, entries, title=title, **kw)


def padded_ghost(ax, x, y, w, h, shape=r"$[B,\,C_{\max},\,N,\,D]$",
                 note="never allocated", color=GHOST, strike=True, z=1):
    """Dashed padded tensor that is never allocated/built (with a strike)."""
    rrect(ax, x, y, w, h, edge=color, face="#FAFAFA", lw=1.3, dashed=True,
          rounding=0.6, z=z)
    if strike:
        ax.plot([x + 0.3, x + w - 0.3], [y + 0.3, y + h - 0.3],
                color=LOSS, lw=2.2, alpha=0.85, zorder=z + 4)
    # white backing boxes keep the shape and the note legible over the strike
    bbox = dict(boxstyle="round,pad=0.18", facecolor="white",
                edgecolor="none", alpha=0.9)
    ax.text(x + w / 2, y + h / 2 + 0.8, shape, ha="center", va="center",
            fontsize=7.0, color=color, zorder=z + 5, fontweight="bold",
            bbox=bbox)
    ax.text(x + w - 0.6, y + 0.6, note, ha="right", va="bottom",
            fontsize=6.2, color=LOSS, style="italic", zorder=z + 5, bbox=bbox)


# --------------------------------------------------------------------------- #
# Array-stack helpers
# --------------------------------------------------------------------------- #
def token_rows(ax, x, y0, w, h, pitch, specs, rounding=0.4, z=3,
               crossed=True):
    """Draw a vertical column of **data** token cells (crossed).

    ``specs`` is a list of ``(edge_color, omitted)`` tuples.  An *omitted* row is
    drawn dashed with a white face and is **not** crossed -- the visual verb for
    "this token does not exist", never "this token is zero".  Real rows are
    crossed to mark data tensors.  Returns the top ``y`` of the stack.
    """
    for i, (edge, omitted) in enumerate(specs):
        yy = y0 + i * pitch
        rrect(ax, x, yy, w, h, edge=edge,
              face=("#FFFFFF" if omitted else tint(edge, 0.8)), lw=1.2,
              dashed=omitted, rounding=rounding, z=z,
              crossed=crossed and not omitted)
    return y0 + (len(specs) - 1) * pitch + h


def grid_cells(ax, x0, y0, nrows, ncols, cw, pitch, edge=BLUE, face=None,
               lw=1.1, rounding=0.3, z=3, crossed=True):
    """An ``nrows x ncols`` grid of **data** cells (crossed by default).

    Returns ``(top, right)``.
    """
    face = face if face is not None else tint(edge, 0.72)
    for r in range(nrows):
        for c in range(ncols):
            rrect(ax, x0 + c * pitch, y0 + r * pitch, cw, cw, edge=edge,
                  face=face, lw=lw, rounding=rounding, z=z, crossed=crossed)
    return y0 + (nrows - 1) * pitch + cw, x0 + (ncols - 1) * pitch + cw


# --------------------------------------------------------------------------- #
# Figure lifecycle
# --------------------------------------------------------------------------- #
def install_rcparams() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "mathtext.fontset": "dejavusans",
    })


def save_figure(fig, outdir, stem, formats=("pdf", "svg", "png")):
    path = Path(outdir)
    path.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        out = path / f"{stem}.{fmt}"
        fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"wrote {out}")
