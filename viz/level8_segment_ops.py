"""L8 -- padding-free segmented-attention primitive (methods-box figure).

Shows how ``grouped_self_attention`` / ``grouped_pool_attention`` /
``segment_softmax`` work: the flattened token list ``[T, D]`` + ``seg: [T]``
layout, the sort-by-segment bucketing, batched SDPA over groups of equal length,
and scatter-back.  Draws the equivalence to a dense ``[B, T_max, D]`` tensor +
``-inf`` attention mask reference, and why the segmented form is exactly that
without materialising padding (cf. FlashAttention varlen / cu_seqlens).

Usage::

    uv run --extra viz python viz/level8_segment_ops.py
    uv run --extra viz python viz/level8_segment_ops.py --formats pdf svg png
    uv run --extra viz python viz/level8_segment_ops.py --stem level8_segment_ops
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
    install_rcparams, save_figure,
)

OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Token array stacks
# --------------------------------------------------------------------------- #
def draw_input_tokens(ax, x, y0):
    """[T, D] flattened token tensor plus seg: [T] segment ids."""
    w, h, pitch = 14.0, 2.5, 3.2
    specs = [
        (BLUE, 0),
        (BLUE, 1),
        (BLUE, 2),
        (GREEN, 0),
        (GREEN, 1),
        (ORANGE, 0),
        (ORANGE, 1),
        (ORANGE, 2),
    ]
    for i, (c, seg_id) in enumerate(specs):
        yy = y0 + i * pitch
        rrect(ax, x, yy, w, h, edge=c, face=tint(c, 0.8), lw=1.2)
        # segment id annotation
        ax.text(x - 1.2, yy + h / 2, f"$s_{seg_id}$", ha="right", va="center",
                fontsize=6.8, color=c, fontweight="bold")
    
    top = y0 + (len(specs) - 1) * pitch + h
    sq_bracket(ax, x + w + 0.9, y0, top)
    ax.text(x + w + 1.6, (y0 + top) / 2, "$[T, D]$", ha="left", va="center",
            fontsize=8.5, color=INK)
    
    # seg ids column - moved further right to avoid overlap
    sx = x + w + 9.5
    sw = 3.5
    ax.text(sx + sw / 2, top + 2.0, "$\\mathtt{seg}$", ha="center", va="center",
            fontsize=8.0, fontweight="bold", color=MUTED)
    for i, (c, seg_id) in enumerate(specs):
        yy = y0 + i * pitch
        rrect(ax, sx, yy, sw, h, edge=MUTED, face="#F4F4F4", lw=1.0)
        ax.text(sx + sw / 2, yy + h / 2, str(seg_id), ha="center", va="center",
                fontsize=7.2, color=INK)
    
    return top, x + w + 9.5 + sw


def draw_sorted_tokens(ax, x, y0):
    """[T, D] after argsort(seg, stable=True), grouped by segment."""
    w, h, pitch = 14.0, 2.5, 3.2
    # Now grouped: seg 0 (blue), seg 1 (green), seg 2 (orange)
    specs = [
        (BLUE, 0, 0),   # t0
        (BLUE, 0, 1),   # t3
        (BLUE, 0, 2),   # t6
        (GREEN, 1, 0),  # t1
        (GREEN, 1, 1),  # t4
        (ORANGE, 2, 0), # t2
        (ORANGE, 2, 1), # t5
        (ORANGE, 2, 2), # t7
    ]
    
    # Draw group separators
    group_tops = []
    for i, (c, seg_id, _) in enumerate(specs):
        yy = y0 + i * pitch
        rrect(ax, x, yy, w, h, edge=c, face=tint(c, 0.8), lw=1.2)
        # Check if next token is different segment
        if i < len(specs) - 1 and specs[i][1] != specs[i+1][1]:
            group_tops.append(yy + h)
            ax.plot([x - 0.5, x + w + 0.5], [yy + h, yy + h],
                    color=MUTED, lw=1.0, linestyle=":", zorder=2)
    
    top = y0 + (len(specs) - 1) * pitch + h
    sq_bracket(ax, x + w + 0.9, y0, top)
    ax.text(x + w + 1.6, (y0 + top) / 2, "$[T, D]$", ha="left", va="center",
            fontsize=8.5, color=INK)
    
    # Segment length annotations - moved further left for clarity
    lx = x - 9.5
    # seg 0: 3 tokens (blue)
    y_mid0 = y0 + pitch
    ax.text(lx, y_mid0, "$L_0{=}3$", ha="center", va="center",
            fontsize=6.8, color=BLUE, fontweight="bold")
    # seg 1: 2 tokens (green)
    y_mid1 = y0 + 3.5 * pitch
    ax.text(lx, y_mid1, "$L_1{=}2$", ha="center", va="center",
            fontsize=6.8, color=GREEN, fontweight="bold")
    # seg 2: 3 tokens (orange)
    y_mid2 = y0 + 6.5 * pitch
    ax.text(lx, y_mid2, "$L_2{=}3$", ha="center", va="center",
            fontsize=6.8, color=ORANGE, fontweight="bold")
    
    return top


def draw_batched_groups(ax, x, y0):
    """Groups of equal length batched for SDPA."""
    # Show two groups: L=3 (blue+orange), L=2 (green)
    cell = 2.8
    pitch = 3.3
    
    # Group 1: L=3 (two samples: seg 0 blue, seg 2 orange)
    # Move text higher to avoid overlap with cells
    ax.text(x + 7.0, y0 + 14.5, "length $L{=}3$", ha="center", va="center",
            fontsize=7.5, fontweight="bold", color=INK)
    ax.text(x + 7.0, y0 + 12.7, "batch $M{=}2$ segs", ha="center", va="center",
            fontsize=6.2, color=MUTED)
    
    # Blue segment (3 tokens)
    for i in range(3):
        rrect(ax, x + i * pitch, y0 + 4.5, cell, cell, edge=BLUE,
              face=tint(BLUE, 0.75), lw=1.1, rounding=0.3)
    ax.text(x - 2.5, y0 + 4.5 + cell/2, "$s_0$", ha="right", va="center",
            fontsize=6.5, color=BLUE, fontweight="bold")
    
    # Orange segment (3 tokens)
    for i in range(3):
        rrect(ax, x + i * pitch, y0 + 1.0, cell, cell, edge=ORANGE,
              face=tint(ORANGE, 0.75), lw=1.1, rounding=0.3)
    ax.text(x - 2.5, y0 + 1.0 + cell/2, "$s_2$", ha="right", va="center",
            fontsize=6.5, color=ORANGE, fontweight="bold")
    
    # Group 2: L=2 (one sample: seg 1 green)
    gy0 = y0 - 5.5
    # Move text higher and further left to avoid overlap
    ax.text(x + 4.5, gy0 + 5.5, "length $L{=}2$", ha="center", va="center",
            fontsize=7.5, fontweight="bold", color=INK)
    ax.text(x + 4.5, gy0 + 3.7, "batch $M{=}1$ seg", ha="center", va="center",
            fontsize=6.2, color=MUTED)
    
    # Green segment (2 tokens)
    for i in range(2):
        rrect(ax, x + i * pitch, gy0, cell, cell, edge=GREEN,
              face=tint(GREEN, 0.75), lw=1.1, rounding=0.3)
    ax.text(x - 2.5, gy0 + cell/2, "$s_1$", ha="right", va="center",
            fontsize=6.5, color=GREEN, fontweight="bold")


def draw_output_tokens(ax, x, y0):
    """[T, D] output after scatter back."""
    w, h, pitch = 14.0, 2.5, 3.2
    specs = [
        (BLUE, 0),
        (BLUE, 1),
        (BLUE, 2),
        (GREEN, 0),
        (GREEN, 1),
        (ORANGE, 0),
        (ORANGE, 1),
        (ORANGE, 2),
    ]
    for i, (c, seg_id) in enumerate(specs):
        yy = y0 + i * pitch
        rrect(ax, x, yy, w, h, edge=c, face=tint(c, 0.85), lw=1.2)
    
    top = y0 + (len(specs) - 1) * pitch + h
    sq_bracket(ax, x + w + 0.9, y0, top)
    ax.text(x + w + 1.6, (y0 + top) / 2, "$[T, D]$", ha="left", va="center",
            fontsize=8.5, color=INK)
    
    return top


def draw_padded_reference(ax, x, y0):
    """The ghost padded [B, T_max, D] + -inf mask that is never materialized."""
    cell = 2.5
    pitch = 2.9
    B, T_max = 3, 3
    
    # Dashed frame
    fw, fh = T_max * pitch + 1.5, B * pitch + 1.5
    rrect(ax, x, y0, fw, fh, edge=MUTED, face="#F8F8F8", lw=1.5,
          dashed=True, rounding=0.6, z=1)
    
    ox = x + 0.5
    oy = y0 + 0.5
    
    # Blue segment: 3 tokens (full row)
    for j in range(3):
        rrect(ax, ox + j * pitch, oy + 2 * pitch, cell, cell, edge=BLUE,
              face=tint(BLUE, 0.75), lw=1.0, rounding=0.25)
    
    # Green segment: 2 tokens + 1 padding
    for j in range(2):
        rrect(ax, ox + j * pitch, oy + 1 * pitch, cell, cell, edge=GREEN,
              face=tint(GREEN, 0.75), lw=1.0, rounding=0.25)
    # Padding cell (crossed out)
    rrect(ax, ox + 2 * pitch, oy + 1 * pitch, cell, cell, edge=RED,
          face="#FFFFFF", lw=1.0, rounding=0.25)
    ax.plot([ox + 2 * pitch + 0.3, ox + 2 * pitch + cell - 0.3],
            [oy + 1 * pitch + 0.3, oy + 1 * pitch + cell - 0.3],
            color=RED, lw=1.8, alpha=0.8, zorder=4)
    
    # Orange segment: 3 tokens (full row)
    for j in range(3):
        rrect(ax, ox + j * pitch, oy, cell, cell, edge=ORANGE,
              face=tint(ORANGE, 0.75), lw=1.0, rounding=0.25)
    
    # Labels
    ax.text(x - 1.2, oy + 2 * pitch + cell/2, "$b_0$", ha="right", va="center",
            fontsize=6.5, color=BLUE)
    ax.text(x - 1.2, oy + 1 * pitch + cell/2, "$b_1$", ha="right", va="center",
            fontsize=6.5, color=GREEN)
    ax.text(x - 1.2, oy + cell/2, "$b_2$", ha="right", va="center",
            fontsize=6.5, color=ORANGE)
    
    # Move label above the frame to avoid overlap
    ax.text(x + fw / 2, y0 + fh + 2.2, "$[B, T_{\\max}, D]$ ghost",
            ha="center", va="center", fontsize=7.5, color=MUTED,
            fontweight="bold", style="italic")
    # Move below text lower to avoid the red slash
    ax.text(x + fw / 2, y0 - 4.0, "never materialized", ha="center", va="center",
            fontsize=6.6, color=RED, style="italic")


# --------------------------------------------------------------------------- #
# The figure
# --------------------------------------------------------------------------- #
def build() -> plt.Figure:
    fig = plt.figure(figsize=(13, 7))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_xlim(0, 130)
    ax.set_ylim(0, 70)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    # ---- title ------------------------------------------------------------ #
    panel_header(ax, 2, 67, "L8",
                 "Segmented-attention primitive: padding-free batched SDPA")

    # ---- main flow stages ------------------------------------------------- #
    y0 = 37.0
    
    # Stage 1: Input tokens
    ax.text(10.0, y0 + 28.5, "① flattened tokens", ha="center", va="center",
            fontsize=8.0, fontweight="bold", color=INK)
    top1, seg_right = draw_input_tokens(ax, 3.5, y0)
    
    # Stage 2: Sort by segment
    ax.text(35.0, y0 + 28.5, "② sort by segment", ha="center", va="center",
            fontsize=8.0, fontweight="bold", color=INK)
    ax.text(35.0, y0 + 26.0, "$\\mathtt{order} = \\mathtt{argsort(seg)}$",
            ha="center", va="center", fontsize=6.8, color=MUTED)
    top2 = draw_sorted_tokens(ax, 28.0, y0)
    
    # Arrow 1->2
    arrow(ax, (seg_right + 1.0, y0 + 13.0), (27.0, y0 + 13.0))
    
    # Stage 3: Batched SDPA
    ax.text(60.0, y0 + 28.5, "③ batched SDPA", ha="center", va="center",
            fontsize=8.0, fontweight="bold", color=INK)
    ax.text(60.0, y0 + 26.0, "by equal length $L$", ha="center", va="center",
            fontsize=6.8, color=MUTED)
    draw_batched_groups(ax, 49.0, y0 + 8.0)
    
    # Arrow 2->3
    arrow(ax, (43.5, y0 + 13.0), (47.0, y0 + 13.0))
    
    # Stage 4: Scatter back
    ax.text(87.0, y0 + 28.5, "④ scatter back", ha="center", va="center",
            fontsize=8.0, fontweight="bold", color=INK)
    ax.text(87.0, y0 + 26.0, "$\\mathtt{out[order]} = \\mathtt{out\\_sorted}$",
            ha="center", va="center", fontsize=6.8, color=MUTED)
    top4 = draw_output_tokens(ax, 78.0, y0)
    
    # Arrow 3->4
    arrow(ax, (70.5, y0 + 13.0), (77.0, y0 + 13.0))
    
    # ---- equation box ----------------------------------------------------- #
    box(ax, 98.5, y0 + 17.0, 28.0, 9.5,
        "$\\mathrm{softmax}(QK^T/\\sqrt{d})V$",
        "keys $=$ values for pooling", edge=PURPLE, title_size=11,
        sub_size=6.5)
    
    # ---- key insight box -------------------------------------------------- #
    rrect(ax, 98.5, y0 + 4.0, 28.0, 11.0, edge=RED, face=tint(RED, 0.92),
          lw=1.5, rounding=0.6)
    ax.text(112.5, y0 + 12.0, "Tokens from different", ha="center", va="center",
            fontsize=7.2, fontweight="bold", color=INK)
    ax.text(112.5, y0 + 9.3, "segments NEVER interact", ha="center", va="center",
            fontsize=7.2, fontweight="bold", color=INK)
    ax.text(112.5, y0 + 5.8, "⇒ exactly per-sample /", ha="center", va="center",
            fontsize=6.5, color=MUTED, style="italic")
    ax.text(112.5, y0 + 4.2, "per-location attention", ha="center", va="center",
            fontsize=6.5, color=MUTED, style="italic")
    
    # ---- equivalence to padded reference ---------------------------------- #
    ax.text(65.0, 20.0, "Equivalence (cf. FlashAttention $\\mathtt{cu\\_seqlens}$)",
            ha="center", va="center", fontsize=8.5, fontweight="bold", color=INK)
    
    # Segmented form on left - moved text upward to avoid overlap
    seg_x = 40.0
    seg_y = 5.0
    ax.text(seg_x + 7.0, 27.0, "segmented form", ha="center", va="center",
            fontsize=7.0, fontweight="bold", color=GREEN)
    ax.text(seg_x + 7.0, 25.4, "(what we compute)", ha="center", va="center",
            fontsize=6.2, color=MUTED, style="italic")
    
    # Mini token stack
    w, h, pitch = 8.0, 1.8, 2.4
    for i, c in enumerate([BLUE, BLUE, BLUE, GREEN, GREEN, ORANGE, ORANGE, ORANGE]):
        rrect(ax, seg_x, seg_y + i * pitch, w, h, edge=c, face=tint(c, 0.8),
              lw=1.0)
    ax.text(seg_x + w + 0.6, seg_y + 3.5 * pitch, "$[T, D]$", ha="left", va="center",
            fontsize=7.0, color=INK)
    
    # Check mark
    ax.text(seg_x + w / 2, seg_y - 1.5, "✓", ha="center", va="center",
            fontsize=18, color=GREEN, fontweight="bold")
    
    # Padded form on right - moved text upward
    pad_x = 95.0
    ax.text(pad_x + 6.0, 27.0, "padded $+$ mask", ha="center", va="center",
            fontsize=7.0, fontweight="bold", color=MUTED)
    ax.text(pad_x + 6.0, 25.4, "(never materialized)", ha="center", va="center",
            fontsize=6.2, color=RED, style="italic")
    draw_padded_reference(ax, pad_x, seg_y)
    
    # Cross mark
    ax.text(pad_x + 5.0, seg_y - 1.5, "✗", ha="center", va="center",
            fontsize=18, color=RED, fontweight="bold")
    
    # Equivalence arrow
    arrow(ax, (seg_x + w + 12.0, 10.0), (pad_x - 2.0, 10.0), color=MUTED,
          lw=1.8, dashed=True, ms=13)
    ax.text(70.0, 11.8, "same semantics", ha="center", va="center",
            fontsize=6.8, color=MUTED)
    
    # ---- footer note ------------------------------------------------------ #
    ax.text(2.0, 1.5,
            "FlashAttention varlen / cu_seqlens idiom: no padding tokens, "
            "no $-\\infty$ mask, $O(T)$ memory.",
            ha="left", va="center", fontsize=7.0, color=MUTED, style="italic")
    
    return fig


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--formats", nargs="+", default=["pdf", "svg", "png"])
    ap.add_argument("--stem", default="level8_segment_ops")
    args = ap.parse_args()

    install_rcparams()

    fig = build()
    save_figure(fig, OUT, args.stem, args.formats)
    plt.close(fig)


if __name__ == "__main__":
    main()
