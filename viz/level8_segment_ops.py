"""L8 -- padding-free grouped-attention primitive (methods-box figure).

Shows how the shipped grouped-attention primitive works: the flattened token
list ``[T, D]`` + ``group_id: [T]`` layout, the sort-by-group bucketing, batched
SDPA over groups of equal length, and scatter-back.

* ``grouped_projected_self_attention`` — the bucketed projected primitive used
  by the encoder blocks and the decoder's target self-attention.
* ``grouped_attention`` — the general grouped cross-attention primitive used
  by the decoder; it loops per group, so query and key/value tokens may have
  different lengths per group.
* ``grouped_pool_attention`` — one learned query set per group, used by channel
  fusion and the descriptor pooler.
* ``grouped_softmax`` — the per-group softmax used by ``IdentityGatedSum``.

Groups are samples for the encoder, decoder, and pooler, and ``(sample, patch)``
locations for channel fusion. This is padding-free batching, **not** grouped-query
attention: there are no head groups and no shared K/V across query heads.

Tokens from different groups never interact, so the primitive needs no padding
tokens and no ``-inf`` attention mask.

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
    BLUE, GREEN, ORANGE, FUSED, LOSS, INK, MUTED,
    tint, rrect, box, arrow, sq_bracket_mirror, panel_header, legend,
    install_rcparams, save_figure,
)

OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Shared row grid -- every token stack uses these, so a token can be tracked
# row-for-row across stages (stage-1 group_id column, [T,D] brackets, stage-2 group
# separators and the stage-4 scatter-back all line up).
# --------------------------------------------------------------------------- #
W_TOK, H_TOK, PITCH = 14.0, 2.5, 3.2
N_TOK = 8
Y0 = 45.0
STACK_TOP = Y0 + (N_TOK - 1) * PITCH + H_TOK
ARROW_Y = Y0 + 15.4  # aligned with the L=3 batch row in stage 3

# Flattened input order: groups are interleaved (g0 blue, g1 green,
# g2 orange), so the stable argsort visibly regroups them.
IN_SPECS = [
    (BLUE, 0), (GREEN, 1), (ORANGE, 2),
    (BLUE, 0), (GREEN, 1), (ORANGE, 2),
    (BLUE, 0), (ORANGE, 2),
]
# After ``order = argsort(group_id, stable=True)``: grouped by group id.
SORTED_SPECS = [
    (BLUE, 0), (BLUE, 0), (BLUE, 0),
    (GREEN, 1), (GREEN, 1),
    (ORANGE, 2), (ORANGE, 2), (ORANGE, 2),
]
# (first_row, last_row, group_id, colour, length) for the sorted stack.
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
    for i, (c, _group) in enumerate(specs):
        rrect(ax, x, Y0 + i * PITCH, W_TOK, H_TOK, edge=c,
              face=tint(c, face), lw=1.2, crossed=True)
    return Y0 + (len(specs) - 1) * PITCH + H_TOK


def bracket_and_shape(ax, x, top):
    """[T, D] bracket + label to the right of a token column.

    The bracket sits to the **right** of the stack, so its ticks are mirrored
    to point back toward the tokens it embraces.
    """
    sq_bracket_mirror(ax, x + W_TOK + 0.9, Y0, top)
    ax.text(x + W_TOK + 1.6, (Y0 + top) / 2, "$[T, D]$",
            ha="left", va="center", fontsize=8.5, color=INK)


def draw_input_tokens(ax, x):
    """[T, D] flattened tokens + the ``group_id: [T]`` id column."""
    for i, (c, group_id) in enumerate(IN_SPECS):
        ax.text(x - 1.0, Y0 + i * PITCH + H_TOK / 2, f"$g_{group_id}$",
                ha="right", va="center", fontsize=6.8, color=c,
                fontweight="bold")

    top = draw_token_column(ax, x, IN_SPECS)
    bracket_and_shape(ax, x, top)

    # group_id: [T] integer id column, to the right of the [T, D] bracket.
    sx = x + W_TOK + 7.0
    sw = 3.5
    ax.text(sx + sw / 2, top + 2.0, "$\\mathtt{group\\_id}$", ha="center",
            va="center", fontsize=8.0, fontweight="bold", color=MUTED)
    for i, (_c, group_id) in enumerate(IN_SPECS):
        rrect(ax, sx, Y0 + i * PITCH, sw, H_TOK, edge=MUTED,
              face="#F4F4F4", lw=1.0)
        ax.text(sx + sw / 2, Y0 + i * PITCH + H_TOK / 2, str(group_id),
                ha="center", va="center", fontsize=7.2, color=INK)
    return top, sx + sw


def draw_sorted_tokens(ax, x):
    """[T, D] after ``argsort(group_id, stable=True)``, grouped by group."""
    top = draw_token_column(ax, x, SORTED_SPECS)

    for (i0, i1, group_id, c, length) in GROUPS:
        if i0 > 0:  # dotted separator between adjacent groups
            yy = Y0 + i0 * PITCH
            ax.plot([x - 0.7, x + W_TOK + 0.7], [yy, yy],
                    color=MUTED, lw=1.0, linestyle=":", zorder=2)
        yc = Y0 + (i0 + i1) / 2 * PITCH + H_TOK / 2
        # length label clearly attached to this group, left of the column
        ax.text(x - 1.4, yc, f"$L_{{{group_id}}}{{=}}{length}$",
                ha="right", va="center", fontsize=6.8, color=c,
                fontweight="bold")

    bracket_and_shape(ax, x, top)
    return top


def draw_batched_groups(ax, x):
    """Batched SDPA: equal-length groups batched into one call each.

    The L=3 bucket keeps stage-2's vertical order for g0/g2 (g2 orange above
    g0 blue); the L=2 bucket sits below with its heading immediately above it.
    """
    cell, cpitch = 2.8, 3.2

    def row(y, n, c):
        for i in range(n):
            rrect(ax, x + i * cpitch, y, cell, cell, edge=c,
                  face=tint(c, 0.75), lw=1.1, rounding=0.3, crossed=True)
        return x + (n - 1) * cpitch + cell

    # --- L = 3, batch M = 2 (groups g2 orange, then g0 blue) --------------- #
    orange_y, blue_y = Y0 + 14.0, Y0 + 10.5
    right3 = row(orange_y, 3, ORANGE)
    row(blue_y, 3, BLUE)
    ax.text(x + 4.6, Y0 + 20.0, "length $L{=}3$", ha="center", va="center",
            fontsize=7.5, fontweight="bold", color=INK)
    ax.text(x + 4.6, Y0 + 18.3, "batch $M{=}2$: $g_2 + g_0$", ha="center",
            va="center", fontsize=6.2, color=MUTED)
    ax.text(right3 + 0.7, orange_y + cell / 2, "$g_2$", ha="left",
            va="center", fontsize=6.4, color=ORANGE, fontweight="bold")
    ax.text(right3 + 0.7, blue_y + cell / 2, "$g_0$", ha="left",
            va="center", fontsize=6.4, color=BLUE, fontweight="bold")

    # --- L = 2, batch M = 1 (group g1 green) ------------------------------- #
    green_y = Y0 + 3.0
    right2 = row(green_y, 2, GREEN)
    # Keep the L=2 heading tight above its row so it reads as the label for
    # this bucket, not as a continuation of the L=3 batch above it.
    ax.text(x + 3.0, Y0 + 8.2, "length $L{=}2$", ha="center", va="center",
            fontsize=7.5, fontweight="bold", color=INK)
    ax.text(x + 3.0, Y0 + 6.5, "batch $M{=}1$: $g_1$", ha="center",
            va="center", fontsize=6.2, color=MUTED)
    ax.text(right2 + 0.7, green_y + cell / 2, "$g_1$", ha="left",
            va="center", fontsize=6.4, color=GREEN, fontweight="bold")


def draw_output_tokens(ax, x):
    """[T, D] output after the scatter-back ``out[order] = out_sorted``."""
    top = draw_token_column(ax, x, IN_SPECS, face=0.88)
    bracket_and_shape(ax, x, top)
    return top


# --------------------------------------------------------------------------- #
# The figure
# --------------------------------------------------------------------------- #
def build() -> plt.Figure:
    fig = plt.figure(figsize=(14, 5.2))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_xlim(0, 140)
    ax.set_ylim(30, 82)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    # ---- title ------------------------------------------------------------ #
    panel_header(ax, 2, 79, "L8",
                 "Grouped-attention primitive: padding-free batched SDPA")

    # ---- stage headings --------------------------------------------------- #
    head_y, sub_y = STACK_TOP + 4.5, STACK_TOP + 2.2

    # Stage 1: flattened tokens + group ids
    ax.text(10.0, head_y, "① flattened tokens", ha="center", va="center",
            fontsize=8.0, fontweight="bold", color=INK)
    _top1, group_right = draw_input_tokens(ax, 3.5)

    # Stage 2: sort by group
    ax.text(41.0, head_y, "② sort by group", ha="center", va="center",
            fontsize=8.0, fontweight="bold", color=INK)
    ax.text(41.0, sub_y, "$\\mathtt{order} = \\mathtt{argsort(group\\_id, stable)}$",
            ha="center", va="center", fontsize=6.8, color=MUTED)
    draw_sorted_tokens(ax, 34.0)

    # Stage 3: batched SDPA by equal length
    ax.text(64.5, head_y, "③ batched SDPA", ha="center", va="center",
            fontsize=8.0, fontweight="bold", color=INK)
    ax.text(64.5, sub_y,
            "bucketed primitives: one call per length $L$, batch $M$",
            ha="center", va="center", fontsize=6.2, color=MUTED)
    ax.text(64.5, sub_y - 1.7, "general cross-attn: loops per group",
            ha="center", va="center", fontsize=6.2, color=MUTED)
    draw_batched_groups(ax, 58.0)

    # Stage 4: scatter back
    ax.text(91.0, head_y, "④ scatter back", ha="center", va="center",
            fontsize=8.0, fontweight="bold", color=INK)
    ax.text(91.0, sub_y, "$\\mathtt{out[order]} = \\mathtt{out\\_sorted}$",
            ha="center", va="center", fontsize=6.8, color=MUTED)
    draw_output_tokens(ax, 84.0)

    # ---- inter-stage arrows (same y so the flow reads straight) ----------- #
    arrow(ax, (group_right + 0.6, ARROW_Y), (33.0, ARROW_Y))
    arrow(ax, (54.5, ARROW_Y), (57.0, ARROW_Y))
    arrow(ax, (71.0, ARROW_Y), (83.0, ARROW_Y))

    # ---- right-hand side notes -------------------------------------------- #
    box(ax, 104.0, Y0 + 16.2, 33.0, 10.4,
        "Grouped SDPA primitives",
        "projected self-attn (bucketed)\n"
        "general cross-attn (per-group loop)\n"
        "channel fusion and descriptor pooler: both Q/K/V projected",
        edge=FUSED, title_size=8.8, sub_size=5.9)

    rrect(ax, 104.0, Y0 + 4.0, 33.0, 11.0, edge=LOSS, face=tint(LOSS, 0.92),
          lw=1.5, rounding=0.6)
    ax.text(120.5, Y0 + 13.5, "HARD CONSTRAINT", ha="center",
            va="center", fontsize=6.2, fontweight="bold", color=LOSS)
    ax.text(120.5, Y0 + 11.2, "Tokens from different", ha="center",
            va="center", fontsize=7.2, fontweight="bold", color=INK)
    ax.text(120.5, Y0 + 8.9, "groups NEVER interact", ha="center",
            va="center", fontsize=7.2, fontweight="bold", color=INK)
    ax.text(120.5, Y0 + 6.4, "encoder/decoder/pooler: groups = samples", ha="center",
            va="center", fontsize=5.9, color=MUTED, style="italic")
    ax.text(120.5, Y0 + 5.1, "fusion: groups = (sample, patch)", ha="center",
            va="center", fontsize=5.9, color=MUTED, style="italic")

    # ---- glossary (compact notation + RoPE scope) ------------------------- #
    rrect(ax, 104.0, 31.2, 33.0, 16.6, edge=MUTED, face="#FAFAFA",
          lw=1.0, rounding=0.5)
    ax.text(105.2, 46.6, "symbols / glossary", ha="left", va="center",
            fontsize=7.0, fontweight="bold", color=INK)
    glossary = [
        "$T$ total real tokens ($=V$ elsewhere);  $D$ token dim",
        "group $g_i$ = one independent attention set",
        "$L$ group length;  $M$ batch of equal-length groups",
        "encoder/decoder/pooler: 1 group per sample",
        "fusion: 1 group per $(sample, patch)$ location",
        "SDPA scaled dot-product attention;  $K/V$ keys/values",
        "$\\mathtt{group\\_id}$ per-token id;  $\\mathtt{order}$ stable argsort",
        "group colours = group IDs, not channel identity",
        "default encoder: 2-D axial RoPE on q/k before grouping",
        "(the primitive itself is unchanged)",
    ]
    gy = 44.5
    for line in glossary:
        ax.text(105.2, gy, line, ha="left", va="center",
                fontsize=5.2, color=MUTED)
        gy -= 1.35

    # ---- legends (group colours are NOT channel identity) ----------------- #
    legend(ax, 2.0, 43.0,
           [("$g_0$", BLUE, "cross"), ("$g_1$", GREEN, "cross"),
            ("$g_2$", ORANGE, "cross")],
           title="group IDs (not channel identity)", pitch=2.3, fontsize=6.6)
    legend(ax, 52.0, 43.0,
           [("SDPA primitive", FUSED, "fill"),
            ("hard constraint", LOSS, "fill")],
           title="roles", pitch=2.3, fontsize=6.6)

    # answer "what are the groups?" right beside the stage-1 group legend
    note_x, note_y = 22.0, 42.6
    ax.text(note_x, note_y, "what is a group?", ha="left", va="center",
            fontsize=6.4, fontweight="bold", color=INK)
    note_lines = [
        "group = one independent attention set;",
        "here $g_0, g_1, g_2$ are toy groups (not channels).",
        "encoder/decoder/pooler: one group per sample;",
        "fusion: one group per $(sample, patch)$ location.",
    ]
    ny = note_y - 2.1
    for line in note_lines:
        ax.text(note_x, ny, line, ha="left", va="center", fontsize=6.0,
                color=MUTED)
        ny -= 1.95

    # ---- footer note ------------------------------------------------------ #
    ax.text(2.0, 31.3,
            "Grouped attention over ragged tokens: one equal-length batch at a time, "
            "$O(T \\cdot D)$ working memory.",
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
