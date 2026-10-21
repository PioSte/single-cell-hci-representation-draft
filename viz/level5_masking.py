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
    
    # axis labels
    ax.text(x0 + N * pitch / 2 - 0.5, y0 + 2.2, "$N$ patches",
            ha="center", va="center", fontsize=6.8, color=MUTED)
    ax.text(x0 - 1.8, y0 - (C - 1) * pitch / 2, "$C_b$",
            ha="right", va="center", fontsize=6.8, color=MUTED)
    
    # title
    top = y0 + 5.0
    ax.text(x0 + N * pitch / 2 - 0.5, top, title,
            ha="center", va="center", fontsize=8.0, fontweight="bold", color=INK)
    if subtitle:
        ax.text(x0 + N * pitch / 2 - 0.5, top - 2.0, subtitle,
                ha="center", va="center", fontsize=6.6, color=MUTED,
                linespacing=1.35)
    
    return y0 - (C - 1) * pitch - cell  # bottom y


def draw_ragged_tokens_with_omission(ax, x0, y0, specs, label="", width=12.0):
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
    fig = plt.figure(figsize=(15.0, 9.0))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_xlim(0, 150)
    ax.set_ylim(0, 90)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    # ---- title ------------------------------------------------------------ #
    panel_header(ax, 2, 86, "L5", "Dynamic Channel-Patch masking (token omission)")

    # ---- two modes section ------------------------------------------------ #
    ax.text(2.0, 78.0, "Two masking modes (per sample, chosen with $p_{\\mathrm{channel}}$ / $p_{\\mathrm{patch}}$):",
            ha="left", va="center", fontsize=8.8, fontweight="bold", color=INK)
    
    # Channel mode
    C_ch, N_ch = 5, 6
    # Create channel mode mask: channels 1 and 3 fully masked
    mask_channel = [[True] * N_ch for _ in range(C_ch)]
    mask_channel[1] = [False] * N_ch
    mask_channel[3] = [False] * N_ch
    
    x_ch, y_ch = 5.0, 62.0
    bot_ch = draw_mask_grid(ax, x_ch, y_ch, C_ch, N_ch, mask_channel,
                            "Channel mode", "mask constant down\neach channel's row")
    
    # annotations
    ax.text(x_ch + N_ch * 3.1 / 2 - 0.5, bot_ch - 2.5,
            "sample $k$, then which $k$ via randperm",
            ha="center", va="center", fontsize=6.8, color=GREEN,
            style="italic")
    
    # Patch mode
    C_pt, N_pt = 5, 6
    # Create patch mode mask: independent spatial mask per channel
    mask_patch = [
        [True, True, False, True, False, True],
        [False, True, True, False, True, True],
        [True, False, True, True, False, False],
        [True, True, True, False, True, False],
        [False, False, True, True, True, True],
    ]
    
    x_pt, y_pt = 42.0, 62.0
    bot_pt = draw_mask_grid(ax, x_pt, y_pt, C_pt, N_pt, mask_patch,
                            "Patch mode", "independent spatial mask\nper channel")
    
    ax.text(x_pt + N_pt * 3.1 / 2 - 0.5, bot_pt - 2.5,
            "$\\mathrm{rand}(C_b, N) \\geq p_{\\mathrm{patch}}$",
            ha="center", va="center", fontsize=6.8, color=ORANGE,
            style="italic")
    
    # ---- guards section --------------------------------------------------- #
    y_guard = 50.0
    rrect(ax, 78.0, y_guard - 8.5, 36.0, 20.0, edge=RED, face=tint(RED, 0.94),
          lw=1.3, dashed=False, rounding=0.7)
    ax.text(96.0, y_guard + 10.2, "Safety guards",
            ha="center", va="center", fontsize=8.4, fontweight="bold", color=INK)
    
    ax.text(96.0, y_guard + 6.5, "• Never mask every token",
            ha="center", va="center", fontsize=7.2, color=INK)
    ax.text(96.0, y_guard + 3.5, "(keep ≥ 1 visible token)",
            ha="center", va="center", fontsize=6.4, color=MUTED, style="italic")
    
    ax.text(96.0, y_guard - 0.5, "• Keep ≥ min_visible_channels",
            ha="center", va="center", fontsize=7.2, color=INK)
    ax.text(96.0, y_guard - 3.5, "channels with some content",
            ha="center", va="center", fontsize=6.4, color=MUTED, style="italic")
    
    # ---- token omission visualization ------------------------------------- #
    y_tokens = 32.0
    ax.text(2.0, y_tokens + 6.0, "Token omission (masking → omitted rows):",
            ha="left", va="center", fontsize=8.8, fontweight="bold", color=INK)
    
    # Before masking: all tokens present
    specs_before = [
        (CHANNEL_COLORS[0], True),
        (CHANNEL_COLORS[0], True),
        (CHANNEL_COLORS[1], True),
        (CHANNEL_COLORS[1], True),
        (CHANNEL_COLORS[2], True),
        (CHANNEL_COLORS[2], True),
        (CHANNEL_COLORS[3], True),
        (CHANNEL_COLORS[3], True),
    ]
    
    x_before = 5.0
    top_before = draw_ragged_tokens_with_omission(
        ax, x_before, y_tokens - 22.0, specs_before,
        label="$[\\Sigma C \\cdot N, D]$\nall tokens", width=11.0
    )
    
    ax.text(x_before + 5.5, top_before + 2.5, "Before DCP",
            ha="center", va="center", fontsize=7.4, fontweight="bold", color=INK)
    
    # Arrow
    arrow(ax, (x_before + 12.5, y_tokens), (x_before + 20.0, y_tokens),
          color=GREEN, lw=2.0)
    edge_label(ax, x_before + 16.2, y_tokens + 1.8, "apply mask", color=GREEN)
    
    # After masking: some tokens omitted (dashed)
    specs_after = [
        (CHANNEL_COLORS[0], True),
        (CHANNEL_COLORS[0], True),
        (CHANNEL_COLORS[1], False),  # omitted
        (CHANNEL_COLORS[1], False),  # omitted
        (CHANNEL_COLORS[2], True),
        (CHANNEL_COLORS[2], False),  # omitted
        (CHANNEL_COLORS[3], True),
        (CHANNEL_COLORS[3], True),
    ]
    
    x_after = 24.0
    top_after = draw_ragged_tokens_with_omission(
        ax, x_after, y_tokens - 22.0, specs_after,
        label="$[V, D]$\nvisible only", width=11.0
    )
    
    ax.text(x_after + 5.5, top_after + 2.5, "After DCP",
            ha="center", va="center", fontsize=7.4, fontweight="bold", color=INK)
    
    # ---- equivalence panel ------------------------------------------------ #
    y_equiv = 2.0
    rrect(ax, 44.0, y_equiv, 64.0, 32.0, edge=VIOLET, face=tint(VIOLET, 0.94),
          lw=1.5, dashed=False, rounding=0.7)
    
    ax.text(76.0, y_equiv + 28.5, "Equivalence (but code uses only omission):",
            ha="center", va="center", fontsize=8.4, fontweight="bold", color=INK)
    
    ax.text(76.0, y_equiv + 23.5, "omitted row",
            ha="center", va="center", fontsize=8.5, color=RED, fontweight="bold")
    
    ax.text(76.0, y_equiv + 20.0, "$\\equiv$",
            ha="center", va="center", fontsize=11.0, color=INK)
    
    ax.text(76.0, y_equiv + 16.5, "masked-padded row (present, but $-\\infty$ attention)",
            ha="center", va="center", fontsize=8.5, color=MUTED, fontweight="bold")
    
    ax.text(76.0, y_equiv + 11.5, "Implementation:",
            ha="center", va="center", fontsize=7.8, fontweight="bold", color=INK)
    
    ax.text(76.0, y_equiv + 8.5, "• Never zeroed",
            ha="center", va="center", fontsize=7.2, color=GREEN)
    ax.text(76.0, y_equiv + 5.5, "• Never padded",
            ha="center", va="center", fontsize=7.2, color=GREEN)
    ax.text(76.0, y_equiv + 2.5, "• No $-\\infty$ attention mask",
            ha="center", va="center", fontsize=7.2, color=GREEN)
    
    # ---- legend ----------------------------------------------------------- #
    legend_y = y_equiv + 22.5
    legend_x = 114.0
    
    # solid cell
    rrect(ax, legend_x, legend_y - 1.0, 2.2, 2.2, edge=BLUE, face=tint(BLUE, 0.78),
          lw=1.2, rounding=0.25)
    ax.text(legend_x + 3.0, legend_y, "visible (kept)",
            ha="left", va="center", fontsize=6.8, color=INK)
    
    # dashed cell
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
