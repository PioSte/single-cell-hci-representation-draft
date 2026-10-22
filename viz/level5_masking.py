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
    BLUE, GREEN, ORANGE, RED, PURPLE, VIOLET, INK, MUTED, CHANNEL_COLORS,
    tint, rrect, box, arrow, edge_label, sq_bracket, panel_header,
    token_rows, install_rcparams, save_figure,
)

OUT = Path(__file__).resolve().parent / "out"


# --------------------------------------------------------------------------- #
# Mask grid visualizations
# --------------------------------------------------------------------------- #
def draw_mask_grid(ax, x0, y0, C, N, mask, title, subtitle="", cell=2.8, pitch=3.1):
    """Draw a [C, N] boolean mask grid with colored channels.

    mask: [C, N] where True = visible, False = masked/invisible.
    """
    for c in range(C):
        for n in range(N):
            visible = mask[c][n]
            edge = CHANNEL_COLORS[c % len(CHANNEL_COLORS)]
            # masked cells: white face with dashed border
            # visible cells: tinted face with solid border
            rrect(ax, x0 + n * pitch, y0 - c * pitch, cell, cell,
                  edge=edge,
                  face=("#FFFFFF" if not visible else tint(edge, 0.78)),
                  lw=1.5,
                  dashed=(not visible),
                  rounding=0.3)

    # Axis labels live *above* / *left of* the grid so they never sit on cells.
    cx = x0 + (N - 1) * pitch / 2 + cell / 2
    top = y0 + cell
    ax.text(cx, top + 1.6, "$N$ patches",
            ha="center", va="bottom", fontsize=6.8, color=MUTED)
    ax.text(x0 - 1.8, y0 - (C - 1) * pitch / 2 + cell / 2, "$C_b$",
            ha="right", va="center", fontsize=6.8, color=MUTED)

    # Title / subtitle, stacked and separated from the axis label.
    ax.text(cx, top + 7.2, title,
            ha="center", va="center", fontsize=8.0, fontweight="bold", color=INK)
    if subtitle:
        ax.text(cx, top + 4.4, subtitle,
                ha="center", va="center", fontsize=6.6, color=MUTED,
                linespacing=1.35)

    return y0 - (C - 1) * pitch  # bottom edge of the grid


def draw_ragged_tokens_with_omission(ax, x0, y0, specs, label="", width=10.0):
    """Draw a ragged token list with dashed rows for omitted tokens.

    specs: list of (channel_color, visible) tuples.
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
            # label the omitted row
            ax.text(x0 + width / 2, yy + h / 2, "omitted",
                    ha="center", va="center", fontsize=6.0,
                    color=RED, style="italic")

    top = y0 + (len(specs) - 1) * pitch + h

    # bracket and label
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

    # Shared horizontal guides --------------------------------------------- #
    COLS_LEFT = 3.0          # before-token stack
    COLS_RIGHT = 54.0        # guards box / equivalence panel left edge
    PANEL_RIGHT = 110.0      # shared right edge

    # ---- title ------------------------------------------------------------ #
    panel_header(ax, 2, 82, "L5", "Dynamic Channel-Patch masking (token omission)")

    # ---- two modes section ------------------------------------------------ #
    ax.text(2.0, 74.0,
            "Two masking modes (per sample, chosen with $p_{\\mathrm{channel}}$ / $p_{\\mathrm{patch}}$):",
            ha="left", va="center", fontsize=8.8, fontweight="bold", color=INK)

    # Channel mode
    C_ch, N_ch = 5, 6
    mask_channel = [[True] * N_ch for _ in range(C_ch)]
    mask_channel[1] = [False] * N_ch
    mask_channel[3] = [False] * N_ch

    x_ch, y_ch = 5.0, 56.8
    bot_ch = draw_mask_grid(ax, x_ch, y_ch, C_ch, N_ch, mask_channel,
                            "Channel mode", "mask constant down\neach channel's row")
    ax.text(x_ch + N_ch * 3.1 / 2 - 0.3, bot_ch - 2.8,
            "sample $k$, then which $k$ via randperm",
            ha="center", va="center", fontsize=6.8, color=GREEN,
            style="italic")

    # Patch mode
    C_pt, N_pt = 5, 6
    mask_patch = [
        [True, True, False, True, False, True],
        [False, True, True, False, True, True],
        [True, False, True, True, False, False],
        [True, True, True, False, True, False],
        [False, False, True, True, True, True],
    ]

    x_pt, y_pt = 32.0, 56.8
    bot_pt = draw_mask_grid(ax, x_pt, y_pt, C_pt, N_pt, mask_patch,
                            "Patch mode", "independent spatial mask\nper channel")
    ax.text(x_pt + N_pt * 3.1 / 2 - 0.3, bot_pt - 2.8,
            "$\\mathrm{rand}(C_b, N) \\geq p_{\\mathrm{patch}}$",
            ha="center", va="center", fontsize=6.8, color=ORANGE,
            style="italic")

    # ---- guards section --------------------------------------------------- #
    # Top/bottom aligned with the grid title row and the grid bottom.
    g_top, g_bot = 68.0, 44.0
    rrect(ax, COLS_RIGHT, g_bot, PANEL_RIGHT - COLS_RIGHT, g_top - g_bot,
          edge=RED, face=tint(RED, 0.94), lw=1.3, dashed=False, rounding=0.7)
    gx = (COLS_RIGHT + PANEL_RIGHT) / 2
    ax.text(gx, g_top - 3.0, "Safety guards",
            ha="center", va="center", fontsize=8.4, fontweight="bold", color=INK)
    ax.text(gx, g_top - 8.5, "• Never mask every token",
            ha="center", va="center", fontsize=7.4, color=INK)
    ax.text(gx, g_top - 12.0, "(keep ≥ 1 visible token)",
            ha="center", va="center", fontsize=6.4, color=MUTED, style="italic")
    ax.text(gx, g_top - 17.0, "• Keep ≥ min_visible_channels",
            ha="center", va="center", fontsize=7.4, color=INK)
    ax.text(gx, g_top - 20.5, "channels with some content",
            ha="center", va="center", fontsize=6.4, color=MUTED, style="italic")

    # ---- token omission visualization ------------------------------------- #
    y_tokens = 5.8  # bottom of the ragged stacks
    stack_top = y_tokens + 7 * 3.5 + 2.7  # top row of an 8-row stack

    ax.text(2.0, stack_top + 4.5,
            "Token omission (masking → omitted rows):",
            ha="left", va="center", fontsize=8.8, fontweight="bold", color=INK)

    # Before masking: all tokens present
    specs_before = [
        (CHANNEL_COLORS[0], True), (CHANNEL_COLORS[0], True),
        (CHANNEL_COLORS[1], True), (CHANNEL_COLORS[1], True),
        (CHANNEL_COLORS[2], True), (CHANNEL_COLORS[2], True),
        (CHANNEL_COLORS[3], True), (CHANNEL_COLORS[3], True),
    ]
    x_before = COLS_LEFT
    draw_ragged_tokens_with_omission(
        ax, x_before, y_tokens, specs_before,
        label="$[\\Sigma C \\cdot N, D]$\nall tokens", width=9.0)
    ax.text(x_before + 4.5, stack_top + 2.4, "Before DCP",
            ha="center", va="center", fontsize=7.4, fontweight="bold", color=INK)

    # Arrow between the two stacks (mid-height, inside the clear gap)
    arrow(ax, (21.0, y_tokens + 13.6), (26.0, y_tokens + 13.6),
          color=GREEN, lw=2.0)
    edge_label(ax, 23.5, y_tokens + 15.4, "apply mask", color=GREEN)

    # After masking: some tokens omitted (dashed)
    specs_after = [
        (CHANNEL_COLORS[0], True), (CHANNEL_COLORS[0], True),
        (CHANNEL_COLORS[1], False), (CHANNEL_COLORS[1], False),
        (CHANNEL_COLORS[2], True), (CHANNEL_COLORS[2], False),
        (CHANNEL_COLORS[3], True), (CHANNEL_COLORS[3], True),
    ]
    x_after = 28.0
    draw_ragged_tokens_with_omission(
        ax, x_after, y_tokens, specs_after,
        label="$[V, D]$\nvisible only", width=9.0)
    ax.text(x_after + 4.5, stack_top + 2.4, "After DCP",
            ha="center", va="center", fontsize=7.4, fontweight="bold", color=INK)

    # ---- equivalence panel ------------------------------------------------ #
    e_top, e_bot = stack_top, y_tokens
    rrect(ax, COLS_RIGHT, e_bot, PANEL_RIGHT - COLS_RIGHT, e_top - e_bot,
          edge=VIOLET, face=tint(VIOLET, 0.94), lw=1.5, dashed=False, rounding=0.7)
    ex = (COLS_RIGHT + PANEL_RIGHT) / 2
    ax.text(ex, e_top - 3.0, "Equivalence (but code uses only omission):",
            ha="center", va="center", fontsize=8.4, fontweight="bold", color=INK)

    ax.text(ex, e_top - 8.5, "omitted row",
            ha="center", va="center", fontsize=8.5, color=RED, fontweight="bold")
    ax.text(ex, e_top - 12.0, "$\\equiv$",
            ha="center", va="center", fontsize=11.0, color=INK)
    ax.text(ex, e_top - 15.5, "masked-padded row (present, but $-\\infty$ attention)",
            ha="center", va="center", fontsize=8.5, color=MUTED, fontweight="bold")

    ax.text(ex, e_top - 20.5, "Implementation:",
            ha="center", va="center", fontsize=7.8, fontweight="bold", color=INK)
    ax.text(ex, e_top - 24.0,
            "• never zeroed      • never padded      • no $-\\infty$ attention mask",
            ha="center", va="center", fontsize=7.2, color=GREEN)

    # ---- legend (adjacent to the omission / equivalence story) ------------ #
    legend_x, legend_y = PANEL_RIGHT + 3.0, e_top - 11.0

    rrect(ax, legend_x, legend_y - 1.0, 2.2, 2.2, edge=BLUE,
          face=tint(BLUE, 0.78), lw=1.2, rounding=0.25)
    ax.text(legend_x + 3.0, legend_y, "visible (kept)",
            ha="left", va="center", fontsize=6.8, color=INK)

    rrect(ax, legend_x, legend_y - 6.0, 2.2, 2.2, edge=BLUE, face="#FFFFFF",
          lw=1.2, dashed=True, rounding=0.25)
    ax.text(legend_x + 3.0, legend_y - 5.0, "masked (omitted)",
            ha="left", va="center", fontsize=6.8, color=INK)

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
