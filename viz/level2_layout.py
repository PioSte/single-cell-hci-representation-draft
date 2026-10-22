"""L2 -- tensor/axis layout of the padding-free channel-fusion architecture.

The hero figure: a left-to-right diagram of literal, axis-labelled array stacks
in the Perceiver-IO Fig 2 / Perceiver Fig 1 idiom, showing where the variable
channel count ``C`` appears (input ragged list length only) and how it is
summed away by fusion to produce a fixed-size ``[B, N, D]`` joint embedding.

Shows:

1. input set ``[ΣC, 1, H, W]`` (ragged: height = C_b, colour per marker)
2. ragged token list ``[ΣC·N, D]`` with the ``(b, c, n)`` index triple annotated
3. the per-``(b,n)`` attention grouping highlighted (box round C tokens)
4. fused ``[B, N, D]`` grid after channel fusion
5. encoder memory ``[B·M, D]`` (fixed) + visible tokens ``[Kf, D]``
6. decoder target queries ``[Kt, D]`` -> predicted pixels ``[Kt, P²]``
7. inference branch: pooled descriptor ``[B, D]``

Plus the ghost ``[B, C_max, N, D]`` tensor (dashed, slashed red) that is never
materialised, with a note that C is fused away.

Usage::

    uv run --extra viz python viz/level2_layout.py
    uv run --extra viz python viz/level2_layout.py --formats pdf svg png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from figure_style import (  # noqa: E402
    BLUE, ORANGE, RED, VIOLET, INK, MUTED, CHANNEL_COLORS,
    tint, rrect, arrow, edge_label, sq_bracket, token_rows, grid_cells,
    install_rcparams, save_figure,
)

OUT = Path(__file__).resolve().parent / "out"

# The single main dataflow line every stack is vertically centred on.
YMID = 30.5


# --------------------------------------------------------------------------- #
# Small local helper: horizontal bracket (for the D / N axis of a stack)
# --------------------------------------------------------------------------- #
def h_bracket(ax, x0, x1, y, color=MUTED, lw=1.1, tick=0.55, down=True):
    """Horizontal square bracket spanning ``x0..x1`` at height ``y``."""
    ax.plot([x0, x1], [y, y], color=color, lw=lw, zorder=3)
    d = -tick if down else tick
    ax.plot([x0, x0], [y, y + d], color=color, lw=lw, zorder=3)
    ax.plot([x1, x1], [y, y + d], color=color, lw=lw, zorder=3)


def sq_bracket_mirror(ax, x, y0, y1, color=MUTED, lw=1.1, tick=0.55):
    """Right-hand square bracket (ticks point left, toward the stack)."""
    ax.plot([x, x], [y0, y1], color=color, lw=lw, zorder=3)
    ax.plot([x, x - tick], [y0, y0], color=color, lw=lw, zorder=3)
    ax.plot([x, x - tick], [y1, y1], color=color, lw=lw, zorder=3)


def axis_text(ax, x, y, text, color=INK, size=7.5, rotation=0, ha="center",
              va="center"):
    ax.text(x, y, text, ha=ha, va=va, fontsize=size, color=color,
            rotation=rotation, zorder=7)


# --------------------------------------------------------------------------- #
# Array stacks
# --------------------------------------------------------------------------- #
def draw_input_images(ax):
    """Ragged set of single-channel images [ΣC, 1, H, W]."""
    x, cw, pitch = 2.2, 4.2, 4.4
    n = 5
    y0 = YMID - ((n - 1) * pitch + cw) / 2.0
    for i in range(n):
        c = CHANNEL_COLORS[i]
        yy = y0 + i * pitch
        rrect(ax, x, yy, cw, cw, edge=c, face=tint(c, 0.72), lw=1.5, z=3)
        ax.plot([x + 0.8, x + cw - 0.8], [yy + 0.9, yy + cw - 0.9],
                color=c, lw=0.9, alpha=0.6, zorder=4)
        ax.plot([x + 0.8, x + cw - 0.8], [yy + cw - 0.9, yy + 0.9],
                color=c, lw=0.9, alpha=0.4, zorder=4)

    top = y0 + (n - 1) * pitch + cw
    sq_bracket(ax, x - 0.7, y0, top, color=MUTED, lw=1.1, tick=0.55)
    axis_text(ax, x - 1.7, (y0 + top) / 2, "$\\Sigma C$", rotation=90)
    h_bracket(ax, x, x + cw, top + 0.6, color=MUTED, down=True)
    axis_text(ax, x + cw / 2, top + 1.6, "$1{\\times}H{\\times}W$", size=7.5)
    ax.text(x + cw / 2, y0 - 1.1, "input", ha="center", va="center",
            fontsize=7.8, fontweight="bold", color=INK)


def _token_column(ax, x, w, h, pitch, n_rows, specs):
    y0 = YMID - ((n_rows - 1) * pitch + h) / 2.0
    top = token_rows(ax, x, y0, w, h, pitch, specs, rounding=0.35, z=3)
    return y0, top


def draw_ragged_tokens(ax):
    """Ragged token list [ΣC·N, D] with (b,c,n) annotation."""
    x, w, h, pitch, n = 11.5, 9.0, 2.0, 2.3, 9
    specs = []
    for i in range(n):
        if i == 6:
            specs.append((MUTED, True))  # omitted row -> masked by DCP
        else:
            specs.append((CHANNEL_COLORS[(i // 3) % len(CHANNEL_COLORS)], False))
    y0, top = _token_column(ax, x, w, h, pitch, n, specs)

    # (b, c, n) index-triple annotation just right of the stack (not on cells)
    anno_y = y0 + 1 * pitch + h / 2
    ax.text(x + w + 1.0, anno_y, "$(b,\\,c,\\,n)$", ha="left", va="center",
            fontsize=6.5, color=INK, style="italic", zorder=5)

    sq_bracket(ax, x - 0.7, y0, top, color=MUTED, lw=1.1, tick=0.55)
    axis_text(ax, x - 1.7, y0 + 0.30 * (top - y0), "$\\Sigma C{\\cdot}N$", rotation=90)
    h_bracket(ax, x, x + w, top + 0.6, color=MUTED, down=True)
    axis_text(ax, x + w / 2, top + 1.6, "$D$", size=7.5)
    ax.text(x + w / 2, y0 - 1.1, "ragged tokens", ha="center", va="center",
            fontsize=7.8, fontweight="bold", color=INK)


def draw_channel_grouping(ax):
    """Highlight the per-(b,n) grouping with a box around C tokens."""
    x, w, h, pitch, n = 25.5, 9.0, 2.0, 2.3, 9
    specs = []
    for i in range(n):
        if i == 6:
            specs.append((MUTED, True))
        else:
            specs.append((CHANNEL_COLORS[(i // 3) % len(CHANNEL_COLORS)], False))
    y0, top = _token_column(ax, x, w, h, pitch, n, specs)

    # Highlight box around 3 consecutive tokens (one location's channels)
    group_start, group_n = 3, 3
    gy0 = y0 + group_start * pitch - 0.3
    gh = group_n * pitch + 0.6
    rrect(ax, x - 0.4, gy0, w + 0.8, gh, edge=RED, face="none", lw=2.0,
          dashed=False, rounding=0.5, z=4)

    sq_bracket(ax, x - 0.7, y0, top, color=MUTED, lw=1.1, tick=0.55)
    axis_text(ax, x - 1.7, y0 + 0.30 * (top - y0), "$\\Sigma C{\\cdot}N$", rotation=90)
    h_bracket(ax, x, x + w, top + 0.6, color=MUTED, down=True)
    axis_text(ax, x + w / 2, top + 1.6, "$D$", size=7.5)
    ax.text(x + w / 2, y0 - 1.1, "grouped by location", ha="center", va="center",
            fontsize=7.8, fontweight="bold", color=INK)
    ax.text(x + w / 2, y0 - 3.1, "pool over $C$ channels at $(b,n)$",
            ha="center", va="center", fontsize=6.3, color=RED)


def draw_fused_grid(ax):
    """Fused [B, N, D] grid."""
    x0, cw, pitch = 42.5, 2.4, 2.7
    B, N = 4, 5
    gh = (B - 1) * pitch + cw
    y0 = YMID - gh / 2.0
    top, right = grid_cells(ax, x0, y0, B, N, cw, pitch, edge=BLUE,
                            face=tint(BLUE, 0.76), lw=1.1, rounding=0.25, z=3)

    h_bracket(ax, x0, right, top + 0.6, color=MUTED, down=True)
    axis_text(ax, (x0 + right) / 2, top + 1.6, "$N$", size=7.5)
    sq_bracket(ax, x0 - 0.7, y0, top, color=MUTED, lw=1.1, tick=0.55)
    axis_text(ax, x0 - 1.7, (y0 + top) / 2, "$B$", rotation=90)
    axis_text(ax, right + 0.9, top - 0.6, "$D$", ha="left", size=7.5)
    ax.text((x0 + right) / 2, y0 - 1.1, "fused", ha="center", va="center",
            fontsize=7.8, fontweight="bold", color=INK)


def draw_encoder_memory(ax):
    """Encoder: [B·M, D] memory + [Kf, D] visible tokens."""
    x, w, h, pitch = 62.5, 9.0, 1.9, 2.2
    n_mem, n_vis = 3, 4
    n = n_mem + n_vis
    y0 = YMID - ((n - 1) * pitch + h) / 2.0

    for i in range(n_mem):
        rrect(ax, x, y0 + i * pitch, w, h, edge=ORANGE, face=tint(ORANGE, 0.80),
              lw=1.2, rounding=0.3, z=3)
    for i in range(n_vis):
        rrect(ax, x, y0 + (n_mem + i) * pitch, w, h, edge=BLUE,
              face=tint(BLUE, 0.80), lw=1.2, rounding=0.3, z=3)
    top = y0 + (n - 1) * pitch + h

    h_bracket(ax, x, x + w, top + 0.6, color=MUTED, down=True)
    axis_text(ax, x + w / 2, top + 1.6, "$D$", size=7.5)

    mem_top = y0 + (n_mem - 1) * pitch + h
    sq_bracket(ax, x - 0.7, y0, mem_top, color=ORANGE, lw=1.1, tick=0.55)
    axis_text(ax, x - 1.9, (y0 + mem_top) / 2, "$B{\\cdot}M$", color=ORANGE,
              rotation=90, size=7.0)

    vis_y0 = y0 + n_mem * pitch
    sq_bracket(ax, x - 0.7, vis_y0, top, color=BLUE, lw=1.1, tick=0.55)
    axis_text(ax, x - 1.9, (vis_y0 + top) / 2, "$K_f$", color=BLUE,
              rotation=90, size=7.0)

    ax.text(x + w / 2, y0 - 1.1, "encoder", ha="center", va="center",
            fontsize=7.8, fontweight="bold", color=INK)


def draw_decoder_targets(ax):
    """Decoder: [Kt, D] queries -> [Kt, P²] predictions."""
    x, w, h, pitch = 78.0, 7.0, 1.9, 2.2
    n = 4
    y0 = YMID - ((n - 1) * pitch + h) / 2.0
    top = y0 + (n - 1) * pitch + h

    # -- target queries ----------------------------------------------------- #
    for i in range(n):
        rrect(ax, x, y0 + i * pitch, w, h, edge=VIOLET, face=tint(VIOLET, 0.82),
              lw=1.2, rounding=0.3, z=3)
    sq_bracket(ax, x - 0.7, y0, top, color=MUTED, lw=1.1, tick=0.55)
    # row-axis label sits low on the bracket, clear of the incoming dec arrow
    axis_text(ax, x - 1.7, y0 + 0.22 * (top - y0), "$K_t$", rotation=90)
    h_bracket(ax, x, x + w, top + 0.6, color=MUTED, down=True)
    axis_text(ax, x + w / 2, top + 1.6, "$D$", size=7.5)
    ax.text(x + w / 2, y0 - 1.1, "targets", ha="center", va="center",
            fontsize=7.8, fontweight="bold", color=INK)

    # -- arrow queries -> predictions --------------------------------------- #
    px, pw = 90.5, 7.0
    arrow(ax, (x + w + 0.4, YMID), (px - 0.4, YMID), color=VIOLET, lw=1.4)

    # -- predictions -------------------------------------------------------- #
    for i in range(n):
        rrect(ax, px, y0 + i * pitch, pw, h, edge=VIOLET,
              face=tint(VIOLET, 0.82), lw=1.2, rounding=0.3, z=3)
    sq_bracket_mirror(ax, px + pw + 0.7, y0, top, color=MUTED, lw=1.1, tick=0.55)
    axis_text(ax, px + pw + 1.9, (y0 + top) / 2, "$K_t$", rotation=90)
    h_bracket(ax, px, px + pw, top + 0.6, color=MUTED, down=True)
    axis_text(ax, px + pw / 2, top + 1.6, "$P^2$", size=7.5)
    ax.text(px + pw / 2, y0 - 1.1, "predictions", ha="center", va="center",
            fontsize=7.8, fontweight="bold", color=INK)


def draw_inference_pooled(ax):
    """Inference: pooled descriptor [B, D] branching off the encoder."""
    x, w, h, pitch = 63.0, 8.0, 2.0, 2.5
    B = 4
    y0 = 6.0
    for i in range(B):
        rrect(ax, x, y0 + i * pitch, w, h, edge=BLUE, face=tint(BLUE, 0.76),
              lw=1.1, rounding=0.25, z=3)
    top = y0 + (B - 1) * pitch + h

    sq_bracket(ax, x - 0.7, y0, top, color=MUTED, lw=1.1, tick=0.55)
    axis_text(ax, x - 1.7, (y0 + top) / 2, "$B$", rotation=90)
    h_bracket(ax, x, x + w, top + 0.6, color=MUTED, down=True)
    axis_text(ax, x + w / 2, top + 1.6, "$D$", size=7.5)
    ax.text(x + w / 2, y0 - 1.1, "pooled", ha="center", va="center",
            fontsize=7.8, fontweight="bold", color=INK)

    # dashed inference branch: encoder bottom -> pooled top-left
    arrow(ax, (63.5, 22.95), (64.8, top + 0.4), color=BLUE, lw=1.3,
          dashed=True, rad=-0.10)
    ax.text(x + 1.2, (22.95 + top) / 2 + 0.2, "inference\npool", ha="right",
            va="center", fontsize=6.3, color=BLUE, style="italic",
            linespacing=1.2)


def draw_ghost_padded(ax):
    """The ghost [B, C_max, N, D] tensor that is never materialised."""
    x, y0, w, h = 2.5, 3.5, 21.5, 12.0

    rrect(ax, x, y0, w, h, edge=MUTED, face="#F8F8F8", lw=1.8, dashed=True,
          rounding=0.7, z=1)

    # red slash (bottom-left -> top-right); labels placed clear of it
    ax.plot([x + 1.5, x + w - 1.5], [y0 + 1.5, y0 + h - 1.5],
            color=RED, lw=2.8, alpha=0.9, zorder=5)

    ax.text(x + 0.6, y0 + h - 1.7, "$[B, C_{\\max}, N, D]$", ha="left",
            va="center", fontsize=8.3, color=MUTED, fontweight="bold", zorder=6)
    ax.text(x + w - 0.6, y0 + 1.7, "never materialised", ha="right",
            va="center", fontsize=6.8, color=RED, style="italic",
            fontweight="bold", zorder=6)


# --------------------------------------------------------------------------- #
# Main figure
# --------------------------------------------------------------------------- #
def build() -> plt.Figure:
    fig = plt.figure(figsize=(15.9, 7.5))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_xlim(0, 106)
    ax.set_ylim(0, 50)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    # ---- title ------------------------------------------------------------ #
    ax.text(1, 47.8, "L2", ha="left", va="center", fontsize=15,
            fontweight="bold", color=INK)
    ax.text(5.5, 47.8,
            "Tensor layout: literal array stacks with axis labels (Perceiver-IO Fig 2 idiom)",
            ha="left", va="center", fontsize=11.5, fontweight="bold", color=INK)
    ax.text(1, 45.0,
            "The output shape is independent of the channel count $C$. "
            "Omitted rows = masked by DCP (dashed); the padded tensor is never created.",
            ha="left", va="center", fontsize=7.0, color=MUTED, style="italic")

    # ---- array stacks ----------------------------------------------------- #
    draw_input_images(ax)
    draw_ragged_tokens(ax)
    draw_channel_grouping(ax)
    draw_fused_grid(ax)
    draw_encoder_memory(ax)
    draw_decoder_targets(ax)
    draw_inference_pooled(ax)
    draw_ghost_padded(ax)

    # ---- main dataflow arrows (all on the shared centre line) ------------- #
    arrow(ax, (6.9, YMID), (11.1, YMID), lw=1.5)
    edge_label(ax, 9.0, YMID + 1.4, "embed", size=6.5)

    arrow(ax, (21.0, YMID), (25.1, YMID), lw=1.5)

    arrow(ax, (35.0, YMID), (42.1, YMID), lw=1.5, color=RED)
    edge_label(ax, 38.5, YMID + 1.4, "channel\nfusion", size=6.5, color=RED)

    arrow(ax, (56.2, YMID), (62.1, YMID), lw=1.5)
    edge_label(ax, 59.1, YMID + 1.4, "enc", size=6.5)

    arrow(ax, (71.9, YMID), (77.6, YMID), lw=1.5, color=VIOLET)
    edge_label(ax, 74.7, YMID + 1.4, "dec", size=6.5, color=VIOLET)

    # ---- key note --------------------------------------------------------- #
    ax.text(45.0, 11.5,
            "Note: $C$ appears only in the input set length $\\Sigma C{\\cdot}N$; channel\n"
            "fusion sums it away at each $(b,n)$ so the output is fixed-size $[B,N,D]$.",
            ha="center", va="center", fontsize=7.3, color=MUTED, style="italic",
            bbox=dict(boxstyle="round,pad=0.55", facecolor="white",
                      edgecolor=MUTED, linewidth=1.0))

    return fig


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--formats", nargs="+", default=["pdf", "svg", "png"])
    ap.add_argument("--stem", default="level2_layout")
    args = ap.parse_args()

    install_rcparams()

    fig = build()
    save_figure(fig, OUT, args.stem, args.formats)
    plt.close(fig)


if __name__ == "__main__":
    main()
