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
    BLUE, GREEN, ORANGE, RED, FUSED, GHOST, LOSS, INK, MUTED,
    tint, rrect, box, arrow, sq_bracket, panel_header, legend,
    padded_ghost, install_rcparams, save_figure,
)

OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Shared row grid -- every token stack uses these, so a token can be tracked
# row-for-row across stages (stage-1 seg column, [T,D] brackets, stage-2 group
# separators and the stage-4 scatter-back all line up).
# --------------------------------------------------------------------------- #
W_TOK, H_TOK, PITCH = 14.0, 2.5, 3.2
N_TOK = 8
Y0 = 45.0
STACK_TOP = Y0 + (N_TOK - 1) * PITCH + H_TOK
ARROW_Y = Y0 + 15.4  # aligned with the L=3 batch row in stage 3

# Flattened input order: segments are interleaved (s0 blue, s1 green,
# s2 orange), so the stable argsort visibly regroups them.
IN_SPECS = [
    (BLUE, 0), (GREEN, 1), (ORANGE, 2),
    (BLUE, 0), (GREEN, 1), (ORANGE, 2),
    (BLUE, 0), (ORANGE, 2),
]
# After ``order = argsort(seg, stable=True)``: grouped by segment id.
SORTED_SPECS = [
    (BLUE, 0), (BLUE, 0), (BLUE, 0),
    (GREEN, 1), (GREEN, 1),
    (ORANGE, 2), (ORANGE, 2), (ORANGE, 2),
]
# (first_row, last_row, segment_id, colour, length) for the sorted stack.
GROUPS = [
    (0, 2, 0, BLUE, 3),
    (3, 4, 1, GREEN, 2),
    (5, 7, 2, ORANGE, 3),
]


# --------------------------------------------------------------------------- #
# Array-stack drawing
# --------------------------------------------------------------------------- #
def draw_token_column(ax, x, specs, face=0.80):
    """A vertical [T, D] token column; returns the top y of the stack."""
    for i, (c, _seg) in enumerate(specs):
        rrect(ax, x, Y0 + i * PITCH, W_TOK, H_TOK, edge=c,
              face=tint(c, face), lw=1.2)
    return Y0 + (len(specs) - 1) * PITCH + H_TOK


def bracket_and_shape(ax, x, top):
    """[T, D] bracket + label to the right of a token column."""
    sq_bracket(ax, x + W_TOK + 0.9, Y0, top)
    ax.text(x + W_TOK + 1.6, (Y0 + top) / 2, "$[T, D]$",
            ha="left", va="center", fontsize=8.5, color=INK)


def draw_input_tokens(ax, x):
    """[T, D] flattened tokens + the ``seg: [T]`` id column."""
    for i, (c, seg_id) in enumerate(IN_SPECS):
        ax.text(x - 1.0, Y0 + i * PITCH + H_TOK / 2, f"$s_{seg_id}$",
                ha="right", va="center", fontsize=6.8, color=c,
                fontweight="bold")

    top = draw_token_column(ax, x, IN_SPECS)
    bracket_and_shape(ax, x, top)

    # seg: [T] integer id column, to the right of the [T, D] bracket.
    sx = x + W_TOK + 7.0
    sw = 3.5
    ax.text(sx + sw / 2, top + 2.0, "$\\mathtt{seg}$", ha="center",
            va="center", fontsize=8.0, fontweight="bold", color=MUTED)
    for i, (_c, seg_id) in enumerate(IN_SPECS):
        rrect(ax, sx, Y0 + i * PITCH, sw, H_TOK, edge=MUTED,
              face="#F4F4F4", lw=1.0)
        ax.text(sx + sw / 2, Y0 + i * PITCH + H_TOK / 2, str(seg_id),
                ha="center", va="center", fontsize=7.2, color=INK)
    return top, sx + sw


def draw_sorted_tokens(ax, x):
    """[T, D] after ``argsort(seg, stable=True)``, grouped by segment."""
    top = draw_token_column(ax, x, SORTED_SPECS)

    for (i0, i1, seg_id, c, length) in GROUPS:
        if i0 > 0:  # dotted separator between adjacent groups
            yy = Y0 + i0 * PITCH
            ax.plot([x - 0.7, x + W_TOK + 0.7], [yy, yy],
                    color=MUTED, lw=1.0, linestyle=":", zorder=2)
        yc = Y0 + (i0 + i1) / 2 * PITCH + H_TOK / 2
        # length label clearly attached to this group, left of the column
        ax.text(x - 1.4, yc, f"$L_{{{seg_id}}}{{=}}{length}$",
                ha="right", va="center", fontsize=6.8, color=c,
                fontweight="bold")

    bracket_and_shape(ax, x, top)
    return top


def draw_batched_groups(ax, x):
    """Batched SDPA: equal-length segments grouped into one call each."""
    cell, cpitch = 2.8, 3.2

    def row(y, n, c):
        for i in range(n):
            rrect(ax, x + i * cpitch, y, cell, cell, edge=c,
                  face=tint(c, 0.75), lw=1.1, rounding=0.3)

    # --- L = 3, batch M = 2 (segments s0 blue and s2 orange) --------------- #
    blue_y, orange_y = Y0 + 14.0, Y0 + 10.5
    row(blue_y, 3, BLUE)
    row(orange_y, 3, ORANGE)
    ax.text(x + 3.0, Y0 + 20.4, "length $L{=}3$", ha="center", va="center",
            fontsize=7.5, fontweight="bold", color=INK)
    ax.text(x + 3.0, Y0 + 18.7, "batch $M{=}2$: $s_0 + s_2$", ha="center",
            va="center", fontsize=6.2, color=MUTED)

    # --- L = 2, batch M = 1 (segment s1 green) ----------------------------- #
    green_y = Y0 + 3.0
    row(green_y, 2, GREEN)
    ax.text(x + 3.0, Y0 + 9.0, "length $L{=}2$", ha="center", va="center",
            fontsize=7.5, fontweight="bold", color=INK)
    ax.text(x + 3.0, Y0 + 7.3, "batch $M{=}1$: $s_1$", ha="center",
            va="center", fontsize=6.2, color=MUTED)


def draw_output_tokens(ax, x):
    """[T, D] output after the scatter-back ``out[order] = out_sorted``."""
    top = draw_token_column(ax, x, IN_SPECS, face=0.88)
    bracket_and_shape(ax, x, top)
    return top


# --------------------------------------------------------------------------- #
# Ghost padded reference (never materialised)
# --------------------------------------------------------------------------- #
def draw_padded_reference(ax, x, y0):
    """Dashed ghost of the dense [B, T_max, D] + -inf mask reference.

    Uses the shared :func:`padded_ghost` primitive so the never-materialised
    padded tensor reads the same way as in every other level.
    """
    w, h = 16.0, 12.0
    padded_ghost(ax, x, y0, w, h,
                 shape=r"$[B, T_{\max}, D]$",
                 note="never materialized", color=GHOST, strike=False, z=1)
    return y0 + h, w


# --------------------------------------------------------------------------- #
# The figure
# --------------------------------------------------------------------------- #
def build() -> plt.Figure:
    fig = plt.figure(figsize=(14, 8.2))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_xlim(0, 140)
    ax.set_ylim(0, 82)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    # ---- title ------------------------------------------------------------ #
    panel_header(ax, 2, 79, "L8",
                 "Segmented-attention primitive: padding-free batched SDPA")

    # ---- stage headings --------------------------------------------------- #
    head_y, sub_y = STACK_TOP + 4.5, STACK_TOP + 2.2

    # Stage 1: flattened tokens + seg ids
    ax.text(10.0, head_y, "① flattened tokens", ha="center", va="center",
            fontsize=8.0, fontweight="bold", color=INK)
    _top1, seg_right = draw_input_tokens(ax, 3.5)

    # Stage 2: sort by segment
    ax.text(41.0, head_y, "② sort by segment", ha="center", va="center",
            fontsize=8.0, fontweight="bold", color=INK)
    ax.text(41.0, sub_y, "$\\mathtt{order} = \\mathtt{argsort(seg, stable)}$",
            ha="center", va="center", fontsize=6.8, color=MUTED)
    draw_sorted_tokens(ax, 34.0)

    # Stage 3: batched SDPA by equal length
    ax.text(64.5, head_y, "③ batched SDPA", ha="center", va="center",
            fontsize=8.0, fontweight="bold", color=INK)
    ax.text(64.5, sub_y, "one call per length $L$, batch $M$", ha="center",
            va="center", fontsize=6.8, color=MUTED)
    draw_batched_groups(ax, 58.0)

    # Stage 4: scatter back
    ax.text(91.0, head_y, "④ scatter back", ha="center", va="center",
            fontsize=8.0, fontweight="bold", color=INK)
    ax.text(91.0, sub_y, "$\\mathtt{out[order]} = \\mathtt{out\\_sorted}$",
            ha="center", va="center", fontsize=6.8, color=MUTED)
    draw_output_tokens(ax, 84.0)

    # ---- inter-stage arrows (same y so the flow reads straight) ----------- #
    arrow(ax, (seg_right + 0.6, ARROW_Y), (33.0, ARROW_Y))
    arrow(ax, (54.5, ARROW_Y), (57.0, ARROW_Y))
    arrow(ax, (68.0, ARROW_Y), (83.0, ARROW_Y))

    # ---- right-hand side notes -------------------------------------------- #
    box(ax, 104.0, Y0 + 17.0, 33.0, 9.5,
        "SDPA primitive",
        "$\\mathrm{softmax}(QK^T/\\sqrt{d})V$\n"
        "keys $=$ values for pooling",
        edge=FUSED, title_size=9.5, sub_size=6.5)

    rrect(ax, 104.0, Y0 + 4.0, 33.0, 11.0, edge=LOSS, face=tint(LOSS, 0.92),
          lw=1.5, rounding=0.6)
    ax.text(120.5, Y0 + 13.5, "HARD CONSTRAINT", ha="center",
            va="center", fontsize=6.2, fontweight="bold", color=LOSS)
    ax.text(120.5, Y0 + 11.2, "Tokens from different", ha="center",
            va="center", fontsize=7.2, fontweight="bold", color=INK)
    ax.text(120.5, Y0 + 8.9, "segments NEVER interact", ha="center",
            va="center", fontsize=7.2, fontweight="bold", color=INK)
    ax.text(120.5, Y0 + 6.4, "⇒ exactly per-sample /", ha="center",
            va="center", fontsize=6.5, color=MUTED, style="italic")
    ax.text(120.5, Y0 + 5.1, "per-location attention", ha="center",
            va="center", fontsize=6.5, color=MUTED, style="italic")

    # ---- legends (segment ids are NOT channel identity) ------------------- #
    legend(ax, 2.0, 43.0,
           [("$s_0$", BLUE, "fill"), ("$s_1$", GREEN, "fill"),
            ("$s_2$", ORANGE, "fill")],
           title="segment IDs (not channels)", pitch=2.3, fontsize=6.6)
    legend(ax, 52.0, 43.0,
           [("SDPA primitive", FUSED, "fill"),
            ("hard constraint", LOSS, "fill"),
            ("never materialized", GHOST, "ghost")],
           title="roles", pitch=2.3, fontsize=6.6)

    # ---- equivalence to the padded reference (aligned bottom row) --------- #
    bot_c = 17.0
    eq_y = 32.0
    ax.text(63.0, eq_y,
            "Equivalence (cf. FlashAttention $\\mathtt{cu\\_seqlens}$)",
            ha="center", va="center", fontsize=8.5, fontweight="bold",
            color=INK)

    # segmented form (left) -- what we actually compute
    seg_x, seg_w = 14.0, 8.0
    seg_h, seg_pitch = 1.8, 2.4
    seg_bottom = bot_c - (7 * seg_pitch + seg_h) / 2
    ax.text(seg_x + seg_w / 2, eq_y - 3.5, "segmented form", ha="center",
            va="center", fontsize=7.0, fontweight="bold", color=GREEN)
    ax.text(seg_x + seg_w / 2, eq_y - 5.0, "(what we compute)", ha="center",
            va="center", fontsize=6.2, color=MUTED, style="italic")
    for i, c in enumerate([BLUE, BLUE, BLUE, GREEN, GREEN,
                           ORANGE, ORANGE, ORANGE]):
        rrect(ax, seg_x, seg_bottom + i * seg_pitch, seg_w, seg_h, edge=c,
              face=tint(c, 0.8), lw=1.0)
    ax.text(seg_x + seg_w + 0.7, seg_bottom + 3.5 * seg_pitch, "$[T, D]$",
            ha="left", va="center", fontsize=7.0, color=INK)
    ax.text(seg_x + seg_w / 2, seg_bottom - 2.2, "✓", ha="center",
            va="center", fontsize=18, color=GREEN, fontweight="bold")

    # padded + mask ghost (right) -- never materialised
    pad_x = 96.0
    pad_y0 = bot_c - 12.0 / 2
    _pad_top, pad_w = draw_padded_reference(ax, pad_x, pad_y0)
    ax.text(pad_x + pad_w / 2, eq_y - 3.5, "padded $+$ $-\\infty$ mask",
            ha="center", va="center", fontsize=7.0, fontweight="bold",
            color=MUTED)
    ax.text(pad_x + pad_w / 2, eq_y - 5.0, "(dense reference)",
            ha="center", va="center", fontsize=6.2, color=MUTED,
            style="italic")
    ax.text(pad_x + pad_w / 2, pad_y0 - 2.4, "✗", ha="center", va="center",
            fontsize=18, color=RED, fontweight="bold")

    # equivalence arrow between the two forms
    arrow(ax, (27.0, bot_c), (pad_x - 2.0, bot_c), color=MUTED, lw=1.8,
          dashed=True, ms=13)
    ax.text(60.0, bot_c + 2.0, "same semantics", ha="center", va="center",
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
