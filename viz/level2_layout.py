"""L2 -- tensor/axis layout of the padding-free channel-fusion architecture.

The hero figure: a left-to-right diagram of literal, axis-labelled array stacks
in the Perceiver-IO Fig 2 / Perceiver Fig 1 idiom, showing where the variable
channel count ``C`` appears (input ragged list length only) and how it is
summed away by fusion to produce a fixed-size ``[B, N, D]`` joint embedding.

Shows:

1. input set ``[ΣC, 1, H, W]`` (ragged: height = C_b, colour per channel)
2. ragged token list ``[ΣC·N, D]`` with the ``(b, c, n)`` index triple annotated
3. the per-``(b,n)`` attention grouping highlighted (box round C tokens)
4. fused ``[B, N, D]`` grid after channel fusion
5. encoder module ``[B·M, D]`` memory (gold) + ``[K_f, D]`` visible tokens
6. decoder target queries ``[K_t, D]`` -> predicted pixels ``[K_t, P²]``
7. inference branch: trained content-only descriptor ``[B, D]``
8. the ghost ``[B, C_max, N, D]`` tensor (dashed, struck) that is never
   materialised, plus role / channel legends and a faithful note.

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
    GREEN, FUSION, ENCODER, MEMORY, LOSS, FUSED, VIOLET, INK, MUTED,
    CHANNEL_COLORS,
    tint, rrect, arrow, edge_label, sq_bracket, token_rows, grid_cells,
    role_legend, channel_legend, padded_ghost,
    install_rcparams, save_figure,
)

OUT = Path(__file__).resolve().parent / "out"

# The single main dataflow line every stack is vertically centred on.
YMID = 30.5


# --------------------------------------------------------------------------- #
# Small local helpers: vertical / horizontal brackets and axis text
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
    """Ragged set of single-channel images [ΣC, 1, H, W].

    Tiles use ``channel_color(i)`` with the level-1 vertical order (channel 0 at
    the top); the variable axis is the stack height.
    """
    x, cw, pitch = 2.2, 4.2, 4.4
    n = len(CHANNEL_COLORS)
    y0 = YMID - ((n - 1) * pitch + cw) / 2.0
    for i in range(n):
        c = CHANNEL_COLORS[i]
        yy = y0 + (n - 1 - i) * pitch  # channel 0 at the top, like L1
        rrect(ax, x, yy, cw, cw, edge=c, face=tint(c, 0.72), lw=1.5, z=3)
        ax.plot([x + 0.8, x + cw - 0.8], [yy + 0.9, yy + cw - 0.9],
                color=c, lw=0.9, alpha=0.6, zorder=4)
        ax.plot([x + 0.8, x + cw - 0.8], [yy + cw - 0.9, yy + 0.9],
                color=c, lw=0.9, alpha=0.4, zorder=4)

    top = y0 + (n - 1) * pitch + cw
    ax.text(x + cw + 0.7, top - 1.6, "$\\vdots$", ha="left", va="center",
            fontsize=10, color=MUTED)
    ax.text(x + cw + 0.7, y0 + 1.6, "$\\vdots$", ha="left", va="center",
            fontsize=10, color=MUTED)

    sq_bracket(ax, x - 0.7, y0, top, color=MUTED, lw=1.1, tick=0.55)
    axis_text(ax, x - 1.7, (y0 + top) / 2, "$\\Sigma C$", rotation=90)
    h_bracket(ax, x, x + cw, top + 0.6, color=MUTED, down=True)
    axis_text(ax, x + cw / 2, top + 1.6, "$1{\\times}H{\\times}W$", size=7.5)
    ax.text(x + cw / 2, y0 - 1.1, "input set", ha="center", va="center",
            fontsize=7.8, fontweight="bold", color=INK)
    ax.text(x + cw / 2, y0 - 2.7, "$P{\\times}P$ patches $\\to N$", ha="center",
            va="center", fontsize=6.2, color=MUTED)


def _token_column(ax, x, w, h, pitch, n_rows, specs):
    y0 = YMID - ((n_rows - 1) * pitch + h) / 2.0
    top = token_rows(ax, x, y0, w, h, pitch, specs, rounding=0.35, z=3)
    return y0, top


def _ragged_specs(n):
    specs = []
    for i in range(n):
        if i == 6:
            specs.append((MUTED, True))  # omitted row -> masked by DCP
        else:
            specs.append((CHANNEL_COLORS[(i // 3) % len(CHANNEL_COLORS)], False))
    return specs


def draw_ragged_tokens(ax):
    """Ragged token list [ΣC·N, D] with (b,c,n) annotation."""
    x, w, h, pitch, n = 11.5, 9.0, 2.0, 2.3, 9
    y0, top = _token_column(ax, x, w, h, pitch, n, _ragged_specs(n))

    # (b, c, n) index-triple annotation just right of the stack (not on cells)
    anno_y = y0 + 1 * pitch + h / 2
    ax.text(x + w + 1.0, anno_y, "$(b,\\,c,\\,n)$", ha="left", va="center",
            fontsize=6.5, color=INK, style="italic", zorder=5)

    sq_bracket(ax, x - 0.7, y0, top, color=MUTED, lw=1.1, tick=0.55)
    axis_text(ax, x - 1.7, y0 + 0.30 * (top - y0), "$\\Sigma C{\\cdot}N$",
              rotation=90)
    h_bracket(ax, x, x + w, top + 0.6, color=MUTED, down=True)
    axis_text(ax, x + w / 2, top + 1.6, "$D$", size=7.5)
    ax.text(x + w / 2, y0 - 1.1, "ragged tokens", ha="center", va="center",
            fontsize=7.8, fontweight="bold", color=INK)
    ax.text(x + w / 2, y0 - 2.7, "solid primary object", ha="center",
            va="center", fontsize=6.2, color=MUTED, style="italic")


def draw_channel_grouping(ax):
    """Highlight the per-(b,n) grouping with a box around C tokens."""
    x, w, h, pitch, n = 25.5, 9.0, 2.0, 2.3, 9
    y0, top = _token_column(ax, x, w, h, pitch, n, _ragged_specs(n))

    # FUSION-tinted box around 3 consecutive tokens (one location's channels)
    group_start, group_n = 3, 3
    gy0 = y0 + group_start * pitch - 0.3
    gh = group_n * pitch + 0.6
    rrect(ax, x - 0.4, gy0, w + 0.8, gh, edge=FUSION, face="none", lw=2.0,
          dashed=False, rounding=0.5, z=4)

    sq_bracket(ax, x - 0.7, y0, top, color=MUTED, lw=1.1, tick=0.55)
    axis_text(ax, x - 1.7, y0 + 0.30 * (top - y0), "$\\Sigma C{\\cdot}N$",
              rotation=90)
    h_bracket(ax, x, x + w, top + 0.6, color=MUTED, down=True)
    axis_text(ax, x + w / 2, top + 1.6, "$D$", size=7.5)
    ax.text(x + w / 2, y0 - 1.1, "grouped by location", ha="center",
            va="center", fontsize=7.8, fontweight="bold", color=INK)
    ax.text(x + w / 2, y0 - 2.55, "pool over $C$ channels at $(b,n)$",
            ha="center", va="center", fontsize=6.3, color=FUSION)


def draw_fused_grid(ax):
    """Fused [B, N, D] grid (FUSED blue)."""
    x0, cw, pitch = 42.5, 2.4, 2.7
    B, N = 4, 5
    gh = (B - 1) * pitch + cw
    y0 = YMID - gh / 2.0
    top, right = grid_cells(ax, x0, y0, B, N, cw, pitch, edge=FUSED,
                            face=tint(FUSED, 0.76), lw=1.1, rounding=0.25, z=3)

    h_bracket(ax, x0, right, top + 0.6, color=MUTED, down=True)
    axis_text(ax, (x0 + right) / 2, top + 1.6, "$N$", size=7.5)
    sq_bracket(ax, x0 - 0.7, y0, top, color=MUTED, lw=1.1, tick=0.55)
    axis_text(ax, x0 - 1.7, (y0 + top) / 2, "$B$", rotation=90)
    axis_text(ax, right + 0.9, top - 0.5, "$D$", ha="left", size=7.5)
    ax.text(right + 0.9, top - 2.0, "1 query\n$+$ post PE", ha="left",
            fontsize=5.6, color=MUTED, linespacing=1.3)
    ax.text((x0 + right) / 2, y0 - 1.1, "fused $[B,N,D]$", ha="center",
            va="center", fontsize=7.8, fontweight="bold", color=INK)


def draw_encoder_memory(ax):
    """Encoder module: [B·M, D] gold memory + [K_f, D] blue visible tokens."""
    x, w, h, pitch = 62.5, 9.0, 1.9, 2.2
    n_mem, n_vis = 3, 4
    n = n_mem + n_vis
    y0 = YMID - ((n - 1) * pitch + h) / 2.0

    for i in range(n_mem):
        rrect(ax, x, y0 + i * pitch, w, h, edge=MEMORY, face=tint(MEMORY, 0.80),
              lw=1.2, rounding=0.3, z=3)
    for i in range(n_vis):
        rrect(ax, x, y0 + (n_mem + i) * pitch, w, h, edge=FUSED,
              face=tint(FUSED, 0.80), lw=1.2, rounding=0.3, z=3)
    top = y0 + (n - 1) * pitch + h

    # ENCODER-tinted dashed module frame so the role colour reads.
    rrect(ax, x - 0.4, y0 - 0.6, w + 0.9, (top - y0) + 1.2, edge=ENCODER,
          face="none", lw=1.4, dashed=True, rounding=0.6, z=2)

    h_bracket(ax, x, x + w, top + 1.3, color=MUTED, down=True)
    axis_text(ax, x + w / 2, top + 2.2, "$D$", size=7.5)

    # module title / role sits above the frame, clear of the flow arrows
    ax.text(x + w / 2, top + 4.6, "encoder $\\cdot$ TransformerEncoder",
            ha="center", va="center", fontsize=7.8, fontweight="bold",
            color=INK)
    ax.text(x + w / 2, top + 3.35, "$+M$ memory tokens $\\cdot\\ \\times L$ blocks",
            ha="center", va="center", fontsize=6.3, color=ENCODER)

    mem_top = y0 + (n_mem - 1) * pitch + h
    sq_bracket(ax, x - 1.0, y0, mem_top, color=MEMORY, lw=1.1, tick=0.55)
    axis_text(ax, x - 2.1, (y0 + mem_top) / 2, "$B{\\cdot}M$", color=MEMORY,
              rotation=90, size=7.0)
    ax.text(x + w + 1.2, (y0 + mem_top) / 2, "$M$", ha="left", va="center",
            fontsize=6.8, color=MEMORY)

    vis_y0 = y0 + n_mem * pitch
    sq_bracket(ax, x - 1.0, vis_y0, top, color=FUSED, lw=1.1, tick=0.55)
    axis_text(ax, x - 2.1, (vis_y0 + top) / 2, "$K_f$", color=FUSED,
              rotation=90, size=7.0)


def draw_decoder_targets(ax):
    """Decoder: [Kt, D] queries -> [Kt, P²] predictions (DECODER violet)."""
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
    """Inference branch: trained content-only descriptor [B, D] below the encoder."""
    x, w, h, pitch = 63.0, 8.0, 2.0, 2.5
    B = 4
    y0 = 5.6
    for i in range(B):
        rrect(ax, x, y0 + i * pitch, w, h, edge=FUSED, face=tint(FUSED, 0.76),
              lw=1.1, rounding=0.25, z=3)
    top = y0 + (B - 1) * pitch + h

    sq_bracket(ax, x - 0.7, y0, top, color=MUTED, lw=1.1, tick=0.55)
    axis_text(ax, x - 1.7, (y0 + top) / 2, "$B$", rotation=90)
    h_bracket(ax, x, x + w, top + 0.6, color=MUTED, down=True)
    axis_text(ax, x + w / 2, top + 1.6, "$D$", size=7.5)
    ax.text(x + w / 2, y0 - 1.1, "trained descriptor", ha="center", va="center",
            fontsize=7.8, fontweight="bold", color=INK)

    # dashed inference branch: encoder bottom -> pooled top-left (clear of the
    # centred $D$ label)
    cx = x + w / 2
    arrow(ax, (cx, 22.3), (x + 1.0, top + 0.55), color=FUSED, lw=1.3,
          dashed=True, rad=-0.15)
    ax.text(cx + 0.8, (22.3 + top) / 2, "encode()\npool", ha="left",
            va="center", fontsize=6.3, color=FUSED, style="italic",
            linespacing=1.2)


def draw_ghost_padded(ax):
    """The ghost [B, C_max, N, D] tensor that is never materialised."""
    padded_ghost(ax, 2.5, 3.4, 21.5, 11.2,
                 shape="$[B,\\,C_{\\max},\\,N,\\,D]$",
                 note="never materialised", color=MUTED, z=1)


def draw_legends(ax):
    """Fill the empty bottom band with role + channel legends and a note."""
    role_legend(
        ax, 27.0, 16.1,
        [("tokenizer patch embed", GREEN),
         ("channel fusion", FUSION),
         ("spatial transformer", ENCODER),
         ("memory / register tokens", MEMORY),
         ("fused tokens $[B,N,D]$", FUSED),
         ("decoder / targets", VIOLET),
         ("omitted row (masked)", LOSS)],
        title="module roles", pitch=1.78, sw=1.15, fontsize=6.2,
        title_size=6.8,
    )
    channel_legend(
        ax, 47.0, 16.1, n=len(CHANNEL_COLORS), title="channel identity",
        labels=[f"channel {i}" for i in range(len(CHANNEL_COLORS))],
        pitch=1.78, sw=1.15, fontsize=6.2, title_size=6.8,
    )

    ax.text(48.5, 20.7,
            "Channel count $C$ lives only in the input length $\\Sigma C{\\cdot}N$.\n"
            "Fusion pools the $C$ channel tokens at each $(b,n)$ with one learned\n"
            "query: $O(C{\\cdot}D)$ per location, then adds the post-fusion PE.\n"
            "The output is fixed-size $[B,N,D]$; omitted rows are masked, never zeroed.",
            ha="center", va="center", fontsize=6.5, color=MUTED, style="italic",
            linespacing=1.45,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                      edgecolor=MUTED, linewidth=0.9))


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
    draw_legends(ax)

    # ---- main dataflow arrows (all on the shared centre line) ------------- #
    arrow(ax, (6.9, YMID), (11.1, YMID), lw=1.5)
    edge_label(ax, 9.0, YMID + 1.4, "embed", size=6.5)

    arrow(ax, (21.0, YMID), (25.1, YMID), lw=1.5)

    arrow(ax, (35.0, YMID), (42.1, YMID), lw=1.5, color=FUSION)
    edge_label(ax, 38.5, YMID + 1.4, "channel\nfusion", size=6.5, color=FUSION)

    arrow(ax, (56.2, YMID), (62.0, YMID), lw=1.5)
    edge_label(ax, 58.8, YMID + 1.4, "enc", size=6.5)

    arrow(ax, (72.4, YMID), (77.6, YMID), lw=1.5, color=VIOLET)
    edge_label(ax, 75.0, YMID + 1.4, "dec", size=6.5, color=VIOLET)

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
