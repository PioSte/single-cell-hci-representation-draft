"""L5 -- Dynamic Channel-Patch (DCP) masking: token omission, not zero-padding.

Shows the two masking modes (channel mode: whole channels masked via
count-then-subset; patch mode: independent spatial mask per channel), the
safety guards (never mask every token, keep >= min_visible_channels with
content), and the ragged token list before/after with dashed-frame omitted rows.

Makes explicit: ``omitted row == masked-padded row (present, but -inf attention)``,
but the code uses **only omission** -- never zeroed, never padded, no -inf mask.

Usage::

    cd /home/piotr/Repos/fuse-channels
    uv run --extra viz python viz/level5_masking.py
    uv run --extra viz python viz/level5_masking.py --formats pdf svg png
    uv run --extra viz python viz/level5_masking.py --stem level5_masking
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from figure_style import (  # noqa: E402
    BLUE, GREEN, ORANGE, INK, MUTED, CHANNEL_COLORS,
    FRONTEND, LOSS, GHOST,
    tint, rrect, arrow, edge_label, sq_bracket, panel_header,
    install_rcparams, save_figure,
    legend, channel_legend, channel_color,
)

OUT = Path(__file__).resolve().parent / "out"

# Shared geometry so the two mode grids line up row-for-row / column-for-column.
GRID_C = 5
GRID_N = 6
CELL = 2.8
PITCH = 3.1
GRID_Y0 = 57.8


# --------------------------------------------------------------------------- #
# Small local primitive: a horizontal extent bracket (the D axis).
# --------------------------------------------------------------------------- #
def dim_bracket(ax, x0, x1, y, label, color=MUTED, tick=0.8, lw=1.0,
                below=False):
    """Horizontal extent bracket with its label, above or below the row."""
    ax.plot([x0, x1], [y, y], color=color, lw=lw, zorder=3)
    if below:
        ax.plot([x0, x0], [y, y - tick], color=color, lw=lw, zorder=3)
        ax.plot([x1, x1], [y, y - tick], color=color, lw=lw, zorder=3)
        ax.text((x0 + x1) / 2, y - tick - 0.9, label,
                ha="center", va="top", fontsize=6.8, color=color)
    else:
        ax.plot([x0, x0], [y, y + tick], color=color, lw=lw, zorder=3)
        ax.plot([x1, x1], [y, y + tick], color=color, lw=lw, zorder=3)
        ax.text((x0 + x1) / 2, y + tick + 0.9, label,
                ha="center", va="bottom", fontsize=6.8, color=color)


# --------------------------------------------------------------------------- #
# Mask grid visualization
# --------------------------------------------------------------------------- #
def draw_mask_grid(ax, x0, y0, C, N, mask, title, subtitle=""):
    """Draw a [C, N] boolean mask grid with per-channel coloured cells.

    ``mask[c][n]`` True = visible (kept), False = masked (omitted).  Both grids
    are drawn with the same C, N, cell size and pitch so their rows and columns
    line up exactly.
    """
    for c in range(C):
        for n in range(N):
            visible = mask[c][n]
            edge = channel_color(c)
            # visible: tinted face, solid border.  masked: white face, dashed.
            rrect(ax, x0 + n * PITCH, y0 - c * PITCH, CELL, CELL,
                  edge=edge,
                  face=("#FFFFFF" if not visible else tint(edge, 0.78)),
                  lw=1.5,
                  dashed=(not visible),
                  rounding=0.3)

    cx = x0 + (N - 1) * PITCH / 2 + CELL / 2
    top = y0 + CELL
    ax.text(cx, top + 1.6, "$N$ patches",
            ha="center", va="bottom", fontsize=6.8, color=MUTED)
    ax.text(x0 - 1.8, y0 - (C - 1) * PITCH / 2 + CELL / 2, "$C_b$",
            ha="right", va="center", fontsize=6.8, color=MUTED)

    ax.text(cx, top + 7.2, title,
            ha="center", va="center", fontsize=8.0, fontweight="bold", color=INK)
    if subtitle:
        ax.text(cx, top + 4.4, subtitle,
                ha="center", va="center", fontsize=6.6, color=MUTED,
                linespacing=1.35)

    return y0 - (C - 1) * PITCH  # bottom edge of the grid


# --------------------------------------------------------------------------- #
# Ragged token stack with dashed omitted rows
# --------------------------------------------------------------------------- #
def draw_ragged_tokens(ax, x0, y0, specs, label="", width=9.0):
    """Draw a ragged token list; omitted rows are dashed white (never zeroed).

    ``specs``: list of ``(channel_color, visible)`` tuples.
    """
    h, pitch = 2.7, 3.5
    for i, (color, visible) in enumerate(specs):
        yy = y0 + i * pitch
        rrect(ax, x0, yy, width, h,
              edge=color,
              face=("#FFFFFF" if not visible else tint(color, 0.8)),
              lw=1.2,
              dashed=(not visible),
              rounding=0.3)
        if not visible:
            ax.text(x0 + width / 2, yy + h / 2, "omitted",
                    ha="center", va="center", fontsize=6.0,
                    color=LOSS, style="italic")

    top = y0 + (len(specs) - 1) * pitch + h
    sq_bracket(ax, x0 + width + 0.8, y0, top)
    if label:
        ax.text(x0 + width + 1.4, (y0 + top) / 2, label,
                ha="left", va="center", fontsize=7.0, color=INK)
    return top


# --------------------------------------------------------------------------- #
# The figure
# --------------------------------------------------------------------------- #
def build() -> plt.Figure:
    fig = plt.figure(figsize=(14.0, 8.6))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_xlim(0, 140)
    ax.set_ylim(0, 86)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    COLS_RIGHT = 53.0        # guards / equivalence panel left edge
    PANEL_RIGHT = 112.0      # shared right edge of the two panels
    LEG_X = 114.5            # right-hand legend column

    # ---- title ------------------------------------------------------------ #
    panel_header(ax, 2, 82, "L5", "Dynamic Channel-Patch masking (token omission)")

    # ---- two modes section ------------------------------------------------ #
    ax.text(2.0, 73.5,
            "Two masking modes (per sample, chosen with $p_{\\mathrm{channel}}$ / $p_{\\mathrm{patch}}$):",
            ha="left", va="center", fontsize=8.8, fontweight="bold", color=INK)

    # Channel mode: whole channels masked -> constant down the channel rows.
    mask_channel = [[True] * GRID_N for _ in range(GRID_C)]
    mask_channel[1] = [False] * GRID_N
    mask_channel[3] = [False] * GRID_N

    x_ch = 4.0
    bot_ch = draw_mask_grid(ax, x_ch, GRID_Y0, GRID_C, GRID_N, mask_channel,
                            "Channel mode",
                            "mask constant down\neach channel's row")
    ax.text(x_ch + GRID_N * PITCH / 2 - 0.3, bot_ch - 2.8,
            "sample $k$, then which $k$ via randperm",
            ha="center", va="center", fontsize=6.8, color=GREEN,
            style="italic")

    # Patch mode: independent spatial mask per channel.
    mask_patch = [
        [True, True, False, True, False, True],
        [False, True, True, False, True, True],
        [True, False, True, True, False, False],
        [True, True, True, False, True, False],
        [False, False, True, True, True, True],
    ]
    x_pt = 29.0
    bot_pt = draw_mask_grid(ax, x_pt, GRID_Y0, GRID_C, GRID_N, mask_patch,
                            "Patch mode",
                            "independent spatial mask\nper channel")
    ax.text(x_pt + GRID_N * PITCH / 2 - 0.3, bot_pt - 2.8,
            "$\\mathrm{rand}(C_b, N) \\geq p_{\\mathrm{patch}}$",
            ha="center", va="center", fontsize=6.8, color=ORANGE,
            style="italic")

    # ---- guards section (LOSS red: safety / never-zero warnings) ---------- #
    g_top, g_bot = 70.5, 44.0
    rrect(ax, COLS_RIGHT, g_bot, PANEL_RIGHT - COLS_RIGHT, g_top - g_bot,
          edge=LOSS, face=tint(LOSS, 0.94), lw=1.3, dashed=False, rounding=0.7)
    gx = (COLS_RIGHT + PANEL_RIGHT) / 2
    ax.text(gx, g_top - 3.2, "Safety guards",
            ha="center", va="center", fontsize=8.4, fontweight="bold", color=INK)
    ax.text(gx, g_top - 9.0, "\u2022 Never mask every token",
            ha="center", va="center", fontsize=7.4, color=INK)
    ax.text(gx, g_top - 12.5, "(keep \u2265 1 visible token)",
            ha="center", va="center", fontsize=6.4, color=MUTED, style="italic")
    ax.text(gx, g_top - 17.5, "\u2022 Keep \u2265 min_visible_channels",
            ha="center", va="center", fontsize=7.4, color=INK)
    ax.text(gx, g_top - 21.0, "channels with some content",
            ha="center", va="center", fontsize=6.4, color=MUTED, style="italic")

    # ---- token omission visualization ------------------------------------- #
    y_tokens = 6.0
    stack_top = y_tokens + 7 * 3.5 + 2.7  # top row of an 8-row stack

    ax.text(2.0, 40.0,
            "Token omission (masking \u2192 omitted rows):",
            ha="left", va="center", fontsize=8.8, fontweight="bold", color=INK)

    # Before masking: all tokens present.
    specs_before = [
        (CHANNEL_COLORS[0], True), (CHANNEL_COLORS[0], True),
        (CHANNEL_COLORS[1], True), (CHANNEL_COLORS[1], True),
        (CHANNEL_COLORS[2], True), (CHANNEL_COLORS[2], True),
        (CHANNEL_COLORS[3], True), (CHANNEL_COLORS[3], True),
    ]
    x_before, width = 3.0, 9.0
    draw_ragged_tokens(ax, x_before, y_tokens, specs_before,
                       label="$[\\Sigma C \\cdot N, D]$\nall tokens", width=width)
    ax.text(x_before + width / 2, stack_top + 3.6, "Before DCP",
            ha="center", va="center", fontsize=7.4, fontweight="bold", color=INK)
    dim_bracket(ax, x_before, x_before + width, y_tokens - 1.4, "$D$",
                below=True)

    # After masking: some tokens omitted (dashed white, never zeroed).
    specs_after = [
        (CHANNEL_COLORS[0], True), (CHANNEL_COLORS[0], True),
        (CHANNEL_COLORS[1], False), (CHANNEL_COLORS[1], False),
        (CHANNEL_COLORS[2], True), (CHANNEL_COLORS[2], False),
        (CHANNEL_COLORS[3], True), (CHANNEL_COLORS[3], True),
    ]
    x_after = 28.0
    draw_ragged_tokens(ax, x_after, y_tokens, specs_after,
                       label="$[V, D]$\nvisible only", width=width)
    ax.text(x_after + width / 2, stack_top + 3.6, "After DCP",
            ha="center", va="center", fontsize=7.4, fontweight="bold", color=INK)
    dim_bracket(ax, x_after, x_after + width, y_tokens - 1.4, "$D$",
                below=True)

    # Arrow between the two stacks (above the shape label, clear of it).
    arrow(ax, (13.5, 25.0), (27.0, 25.0), color=FRONTEND, lw=2.0)
    edge_label(ax, 20.25, 27.2, "apply mask", color=FRONTEND)

    # ---- equivalence panel (neutral / MUTED) ------------------------------ #
    e_top, e_bot = stack_top, y_tokens
    rrect(ax, COLS_RIGHT, e_bot, PANEL_RIGHT - COLS_RIGHT, e_top - e_bot,
          edge=GHOST, face=tint(GHOST, 0.94), lw=1.4, dashed=False, rounding=0.7)
    ex = (COLS_RIGHT + PANEL_RIGHT) / 2
    ax.text(ex, e_top - 3.2, "Equivalence (but the code uses only omission):",
            ha="center", va="center", fontsize=8.2, fontweight="bold", color=INK)

    # Omission vs masked-padded, using the 'omit' and 'ghost' legend styles.
    eq_x = COLS_RIGHT + 4.0
    y1 = legend(ax, eq_x, e_top - 7.0,
                [("omitted row", LOSS, "omit")], fontsize=7.4, sw=1.6)
    ax.text(eq_x + 0.8, y1 - 1.8, "$\\equiv$",
            ha="left", va="center", fontsize=11.0, color=INK)
    legend(ax, eq_x, y1 - 3.4,
           [("masked-padded row (present, but $-\\infty$ attention)", GHOST, "ghost")],
           fontsize=7.4, sw=1.6)

    ax.text(ex, e_bot + 8.5, "Implementation:",
            ha="center", va="center", fontsize=7.8, fontweight="bold", color=INK)
    ax.text(ex, e_bot + 5.0,
            "\u2022 never zeroed      \u2022 never padded      \u2022 no $-\\infty$ attention mask",
            ha="center", va="center", fontsize=7.2, color=LOSS)
    ax.text(ex, e_bot + 1.8,
            "the $[C_b, N]$ boolean mask decides which rows exist",
            ha="center", va="center", fontsize=6.6, color=MUTED, style="italic")

    # ---- legends (right-hand column) -------------------------------------- #
    legend(ax, LEG_X, 31.5,
           [("visible (kept)", BLUE, "fill"),
            ("masked (omitted)", LOSS, "omit")],
           title="token rows", pitch=2.6, sw=1.5, fontsize=6.6, title_size=7.0)
    channel_legend(ax, LEG_X, 23.0, n=GRID_C, title="channel identity",
                   pitch=2.5, sw=1.5, fontsize=6.6, title_size=7.0)

    # ---- footnote --------------------------------------------------------- #
    ax.text(2.0, 1.5,
            "DCP = Dynamic Channel-Patch masking. The $[C_b, N]$ boolean mask determines which tokens exist; "
            "the ragged list never materializes a padded tensor.",
            ha="left", va="center", fontsize=7.2, color=MUTED, style="italic")

    return fig


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--formats", nargs="+", default=["pdf", "svg", "png"])
    ap.add_argument("--stem", default="level5_masking")
    args = ap.parse_args()

    install_rcparams()

    fig = build()
    save_figure(fig, OUT, args.stem, args.formats)
    plt.close(fig)


if __name__ == "__main__":
    main()
