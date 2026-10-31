"""L5 -- Dynamic Channel-Patch (DCP) masking as token omission.

Shows the three DCP regions of ChA-MAEViT Eq. 1 -- patch (a fixed count
``floor(N_b * patch_ratio)`` masked per channel via randperm), channel (whole
channels via HCS = Hierarchical Channel Sampling: the number of *visible*
channels is ``m ~ U{min(min_visible_channels, C)..C}``, then a uniform
``m``-subset), and their union -- the safety guards (never mask every token,
keep >= min(min_visible_channels, C) channels with content, optional
``max_masked_channels_frac`` cap, resample rather than ever reveal a whole
channel row), and the token list before/after.  The combined region
requires ``p_patch + p_channel < 1``; at the defaults ``p_patch = p_channel =
0.5`` the union is never sampled.

Masked tokens are **omitted**: the dashed rows do not exist in the
``[V, D]`` token list.  Omitted rows are absent from that list, and no ``-inf``
attention mask is built.  The ``[C_b, N_b]`` boolean mask decides which rows
exist.

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
    BLUE, GREEN, ORANGE, RED, INK, MUTED, CHANNEL_COLORS,
    TOKENIZER, LOSS, GHOST,
    tint, rrect, arrow, edge_label, sq_bracket_mirror, panel_header,
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
        ax.text((x0 + x1) / 2, y - tick - 0.4, label,
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
            # visible (data) cells are crossed; masked cells stay dashed
            # channel-coloured outlines so the omission still reads clearly.
            rrect(ax, x0 + n * PITCH, y0 - c * PITCH, CELL, CELL,
                  edge=edge,
                  face=("#FFFFFF" if not visible else tint(edge, 0.78)),
                  lw=1.5,
                  dashed=(not visible),
                  rounding=0.3,
                  crossed=visible)

    cx = x0 + (N - 1) * PITCH / 2 + CELL / 2
    top = y0 + CELL
    ax.text(cx, top + 1.6, "$N_b$ patches",
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
# Token stack with dashed omitted rows
# --------------------------------------------------------------------------- #
def draw_token_list(ax, x0, y0, specs, label="", width=9.0):
    """Draw a token list; omitted rows are dashed white (never zeroed).

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
              rounding=0.3,
              crossed=visible)
        if not visible:
            ax.text(x0 + width / 2, yy + h / 2, "omitted",
                    ha="center", va="center", fontsize=6.0,
                    color=LOSS, style="italic")

    top = y0 + (len(specs) - 1) * pitch + h
    # Bracket sits to the right of the stack, so mirror its ticks left.
    sq_bracket_mirror(ax, x0 + width + 0.8, y0, top)
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

    COLS_RIGHT = 53.0        # guards / implementation panel left edge
    PANEL_RIGHT = 112.0      # shared right edge of the two panels
    LEG_X = 114.5            # right-hand legend column

    # ---- title ------------------------------------------------------------ #
    panel_header(ax, 2, 82, "L5", "Dynamic Channel-Patch masking (token omission)")

    # ---- notation (top-right, above the guards panel) --------------------- #
    nx = 76.0
    ax.text(nx, 79.2, "symbols / glossary", ha="left", va="center",
            fontsize=7.4, fontweight="bold", color=INK)
    for i, line in enumerate([
        "$D$ token dim  ·  $V$ visible tokens",
        "$N$ / $N_b$ patches per channel (batch / sample $b$)",
        "$C_b$ channels of sample $b$  ·  $r_p$ = patch_ratio",
        "HCS = Hierarchical Channel Sampling",
    ]):
        ax.text(nx, 77.0 - i * 2.0, line, ha="left", va="center",
                fontsize=6.5, color=MUTED)

    # ---- three regions section -------------------------------------------- #
    ax.text(2.0, 73.5,
            "DCP Eq. 1: patch, channel, or combined (= union) — one region per batch by default ($s \\sim \\mathcal{U}(0,1)$):",
            ha="left", va="center", fontsize=8.6, fontweight="bold", color=INK)
    ax.text(2.0, 70.8,
            "Defaults $p_{patch}=p_{channel}=0.5$: $s<1$ selects patch/channel; "
            "combined needs $p_{patch}+p_{channel}<1$ (e.g. both $0$).",
            ha="left", va="center", fontsize=7.0, color=LOSS, style="italic")

    # Channel mode: whole channels masked -> constant down the channel rows.
    mask_channel = [[True] * GRID_N for _ in range(GRID_C)]
    mask_channel[1] = [False] * GRID_N
    mask_channel[3] = [False] * GRID_N

    x_ch = 4.0
    bot_ch = draw_mask_grid(ax, x_ch, GRID_Y0, GRID_C, GRID_N, mask_channel,
                            "Channel mode",
                            "HCS = Hierarchical Channel Sampling\n"
                            "$m$ visible, $m \\sim \\mathcal{U}\\{\\min(min\\_visible\\_channels,C)..C\\}$\n"
                            "then a uniform $m$-subset")

    # Patch mode: independent spatial mask per channel.
    # Default patch_ratio=0.75, N=6 -> floor(6*0.75) = 4 masked per row
    # (2 visible) via randperm.
    mask_patch = [
        [True, False, False, True, False, False],
        [False, True, False, False, True, False],
        [False, False, True, False, False, True],
        [True, False, False, False, True, False],
        [False, True, False, True, False, False],
    ]
    x_pt = 29.0
    bot_pt = draw_mask_grid(ax, x_pt, GRID_Y0, GRID_C, GRID_N, mask_patch,
                            "Patch mode",
                            "independent spatial mask\nper channel")
    ax.text(x_pt + GRID_N * PITCH / 2 - 0.3, bot_pt - 2.0,
            "$r_p$ = patch_ratio = 0.75 (default)\n"
            "$\\lfloor N_b \\cdot r_p \\rfloor$ masked per channel via randperm",
            ha="center", va="center", fontsize=6.8, color=ORANGE,
            style="italic", linespacing=1.4)

    # Combined region: patch mask OR channel mask (Eq. 1's union).
    mask_combined = [
        [mask_patch[c][n] and mask_channel[c][n] for n in range(GRID_N)]
        for c in range(GRID_C)
    ]
    x_cb = 54.0
    bot_cb = draw_mask_grid(ax, x_cb, GRID_Y0, GRID_C, GRID_N, mask_combined,
                            "Combined (union)",
                            "patch mask $\\cup$\nchannel mask")
    ax.text(x_cb + GRID_N * PITCH / 2 - 0.3, bot_cb - 2.8,
            "masked channel-patches are decoder targets\n"
            "combined requires $p_{patch}+p_{channel}<1$\n"
            "not selected at defaults ($0.5+0.5=1$)",
            ha="center", va="center", fontsize=6.8, color=RED,
            style="italic", linespacing=1.4)

    # ---- guards section (LOSS red: safety / never-zero warnings) ---------- #
    GUARD_X0 = 75.0
    g_top, g_bot = 70.0, 41.5
    rrect(ax, GUARD_X0, g_bot, PANEL_RIGHT - GUARD_X0, g_top - g_bot,
          edge=LOSS, face=tint(LOSS, 0.94), lw=1.3, dashed=False, rounding=0.7)
    gx = (GUARD_X0 + PANEL_RIGHT) / 2
    ax.text(gx, g_top - 3.2, "Safety guards",
            ha="center", va="center", fontsize=8.4, fontweight="bold", color=INK)
    ax.text(gx, g_top - 8.4, "\u2022 Never mask every token",
            ha="center", va="center", fontsize=7.4, color=INK)
    ax.text(gx, g_top - 11.5, "(keep \u2265 1 visible token)",
            ha="center", va="center", fontsize=6.4, color=MUTED, style="italic")
    ax.text(gx, g_top - 16.0,
            "\u2022 Keep \u2265 min(min_visible_channels, C)",
            ha="center", va="center", fontsize=7.4, color=INK)
    ax.text(gx, g_top - 19.1, "channels; cap max_masked_channels_frac",
            ha="center", va="center", fontsize=6.4, color=MUTED, style="italic")
    ax.text(gx, g_top - 23.4, "\u2022 Resample \u2264 10\u00d7; never reveal",
            ha="center", va="center", fontsize=6.8, color=INK)
    ax.text(gx, g_top - 26.4, "a whole channel row",
            ha="center", va="center", fontsize=6.4, color=MUTED, style="italic")

    # ---- token omission visualization ------------------------------------- #
    y_tokens = 7.0
    stack_top = y_tokens + 7 * 3.5 + 2.7  # top row of an 8-row stack

    ax.text(2.0, 40.0,
            "Token omission (rows top\u2192bottom = channel $0 \\to C_b{-}1$, matching the grids):",
            ha="left", va="center", fontsize=8.8, fontweight="bold", color=INK)

    # Before masking: all tokens present.
    # Channel 0 is the TOP row (matching the grids), so list order is reversed
    # relative to the draw order (index 0 = bottom).
    specs_before = [
        (CHANNEL_COLORS[3], True), (CHANNEL_COLORS[3], True),
        (CHANNEL_COLORS[2], True), (CHANNEL_COLORS[2], True),
        (CHANNEL_COLORS[1], True), (CHANNEL_COLORS[1], True),
        (CHANNEL_COLORS[0], True), (CHANNEL_COLORS[0], True),
    ]
    x_before, width = 3.0, 9.0
    draw_token_list(ax, x_before, y_tokens, specs_before,
                       label="$[\\Sigma C \\cdot N, D]$\nall tokens", width=width)
    ax.text(x_before + width / 2, stack_top + 3.6, "Before DCP",
            ha="center", va="center", fontsize=7.4, fontweight="bold", color=INK)
    dim_bracket(ax, x_before, x_before + width, y_tokens - 1.4, "$D$",
                below=True)

    # After masking: some tokens omitted (dashed white, never zeroed).
    specs_after = [
        (CHANNEL_COLORS[3], True), (CHANNEL_COLORS[3], True),
        (CHANNEL_COLORS[2], False), (CHANNEL_COLORS[2], True),
        (CHANNEL_COLORS[1], False), (CHANNEL_COLORS[1], False),
        (CHANNEL_COLORS[0], True), (CHANNEL_COLORS[0], True),
    ]
    x_after = 28.0
    draw_token_list(ax, x_after, y_tokens, specs_after,
                       label="$[V, D]$\nvisible only", width=width)
    ax.text(x_after + width / 2, stack_top + 3.6, "After DCP",
            ha="center", va="center", fontsize=7.4, fontweight="bold", color=INK)
    dim_bracket(ax, x_after, x_after + width, y_tokens - 1.4, "$D$",
                below=True)

    # Arrow between the two stacks (above the shape label, clear of it).
    arrow(ax, (13.5, 25.0), (27.0, 25.0), color=INK, lw=2.0)
    edge_label(ax, 20.25, 27.2, "apply mask", color=INK)

    # ---- shipped implementation panel (omission only) --------------------- #
    e_top, e_bot = stack_top, y_tokens
    rrect(ax, COLS_RIGHT, e_bot, PANEL_RIGHT - COLS_RIGHT, e_top - e_bot,
          edge=GHOST, face=tint(GHOST, 0.94), lw=1.4, dashed=False, rounding=0.7)
    ex = (COLS_RIGHT + PANEL_RIGHT) / 2
    ax.text(ex, e_top - 3.4, "Shipped code: omission only",
            ha="center", va="center", fontsize=8.2, fontweight="bold", color=INK)

    bx = COLS_RIGHT + 4.0
    ax.text(bx, e_top - 8.6,
            "\u2022 masked tokens are omitted from the token list",
            ha="left", va="center", fontsize=7.0, color=INK)
    ax.text(bx, e_top - 12.6,
            "\u2022 omitted rows are absent; visible rows only are passed on",
            ha="left", va="center", fontsize=7.0, color=LOSS)
    ax.text(bx, e_top - 16.6,
            "\u2022 the $[C_b, N_b]$ boolean mask",
            ha="left", va="center", fontsize=7.0, color=INK)
    ax.text(bx + 1.6, e_top - 20.0,
            "decides which rows exist",
            ha="left", va="center", fontsize=7.0, color=MUTED, style="italic")
    legend(ax, bx, e_top - 22.8,
           [("omitted row (dashed)", LOSS, "omit")],
           fontsize=6.8, sw=1.5)

    # ---- legends (right-hand column) -------------------------------------- #
    legend(ax, LEG_X, 31.5,
           [("visible: tinted solid fill", channel_color(0), "cross"),
            ("masked: dashed channel outline", channel_color(0), "omit")],
           title="channel-patch rows", pitch=2.6, sw=1.5, fontsize=6.6,
           title_size=7.0)
    channel_legend(ax, LEG_X, 21.5, n=GRID_C, title="channel identity",
                   pitch=2.5, sw=1.5, fontsize=6.6, title_size=7.0)

    # ---- footnote --------------------------------------------------------- #
    ax.text(2.0, 2.4,
            "Code mode fallbacks (patch \u2194 channel \u2194 none for tiny $C$/$N$) and "
            "last_modes record the region actually applied.",
            ha="left", va="bottom", fontsize=7.2, color=MUTED, style="italic")
    ax.text(2.0, 0.8,
            "DCP = Dynamic Channel-Patch masking. The $[C_b, N_b]$ boolean mask determines which tokens exist; "
            "masked tokens are omitted, never zeroed; the token list is the only token container.",
            ha="left", va="bottom", fontsize=7.2, color=MUTED, style="italic")

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
