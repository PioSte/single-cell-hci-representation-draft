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
5. encoder memory ``[M, D]`` (fixed) + visible tokens ``[Kf, D]`` = ``[B·M + Kf, D]``
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
    BLUE, GREEN, ORANGE, RED, PURPLE, VIOLET, INK, MUTED, CHANNEL_COLORS,
    tint, rrect, box, arrow, edge_label, sq_bracket, panel_header,
    token_rows, grid_cells, install_rcparams, save_figure,
)

OUT = Path(__file__).resolve().parent / "out"


# --------------------------------------------------------------------------- #
# Array stacks
# --------------------------------------------------------------------------- #
def draw_input_images(ax):
    """Ragged set of single-channel images [ΣC, 1, H, W]."""
    x, y0, cw, pitch = 2.0, 32.0, 4.5, 5.0
    n_channels = 5
    for i in range(n_channels):
        c = CHANNEL_COLORS[i]
        yy = y0 + i * pitch
        rrect(ax, x, yy, cw, cw, edge=c, face=tint(c, 0.72), lw=1.5, z=3)
        # image squiggles
        ax.plot([x + 0.8, x + cw - 0.8], [yy + 1.0, yy + cw - 1.0],
                color=c, lw=0.9, alpha=0.6, zorder=4)
        ax.plot([x + 0.8, x + cw - 0.8], [yy + cw - 1.0, yy + 1.0],
                color=c, lw=0.9, alpha=0.4, zorder=4)
    
    # Axis labels
    top = y0 + (n_channels - 1) * pitch + cw
    sq_bracket(ax, x - 0.8, y0, top, color=MUTED, lw=1.1, tick=0.6)
    ax.text(x - 1.2, (y0 + top) / 2, "$\\Sigma C$", ha="right", va="center",
            fontsize=7.5, color=INK, rotation=90)
    ax.text(x + cw / 2, top + 1.2, "$1{\\times}H{\\times}W$", ha="center",
            va="bottom", fontsize=7.5, color=INK)
    ax.text(x + cw / 2, y0 - 0.9, "input", ha="center", va="top",
            fontsize=7.8, fontweight="bold", color=INK)


def draw_ragged_tokens(ax):
    """Ragged token list [ΣC·N, D] with (b,c,n) annotation."""
    x, y0, w, h, pitch = 13.0, 32.0, 9.0, 2.2, 2.6
    n_rows = 9
    
    # Draw token cells with one omitted row
    specs = []
    for i in range(n_rows):
        if i == 6:
            specs.append((MUTED, True))  # omitted
        else:
            c_idx = (i // 3) % len(CHANNEL_COLORS)
            specs.append((CHANNEL_COLORS[c_idx], False))
    
    top = token_rows(ax, x, y0, w, h, pitch, specs, rounding=0.35, z=3)
    
    # Index triple annotation on one cell
    anno_row = 1
    anno_y = y0 + anno_row * pitch + h / 2
    ax.text(x + w / 2, anno_y, "$(b, c, n)$", ha="center", va="center",
            fontsize=6.5, color=INK, style="italic", zorder=5)
    
    # Axis labels
    sq_bracket(ax, x + w + 0.7, y0, top, color=MUTED, lw=1.1, tick=0.6)
    ax.text(x + w / 2, top + 1.5, "$D$", ha="center", va="bottom",
            fontsize=7.5, color=INK)
    ax.text(x - 1.2, (y0 + top) / 2, "$\\Sigma C{\\cdot}N$", ha="right",
            va="center", fontsize=7.5, color=INK, rotation=90)
    ax.text(x + w / 2, y0 - 0.9, "ragged tokens", ha="center", va="top",
            fontsize=7.8, fontweight="bold", color=INK)


def draw_channel_grouping(ax):
    """Highlight the per-(b,n) grouping with a box around C tokens."""
    x, y0, w, h, pitch = 29.0, 32.0, 9.0, 2.2, 2.6
    n_rows = 9
    
    # Draw token cells
    specs = []
    for i in range(n_rows):
        if i == 6:
            specs.append((MUTED, True))  # omitted
        else:
            c_idx = (i // 3) % len(CHANNEL_COLORS)
            specs.append((CHANNEL_COLORS[c_idx], False))
    
    top = token_rows(ax, x, y0, w, h, pitch, specs, rounding=0.35, z=3)
    
    # Highlight box around 3 consecutive tokens (one location's channels)
    group_start = 3
    group_n = 3
    gy0 = y0 + group_start * pitch - 0.3
    gh = group_n * pitch + 0.6
    rrect(ax, x - 0.4, gy0, w + 0.8, gh, edge=RED, face="none", lw=2.0,
          dashed=False, rounding=0.5, z=4)
    ax.text(x + w + 3.0, gy0 + gh + 2.4, "pool over $C$ channels at $(b,n)$",
            ha="left", va="bottom", fontsize=6.2, color=RED)
    
    # Axis labels
    sq_bracket(ax, x + w + 0.7, y0, top, color=MUTED, lw=1.1, tick=0.6)
    ax.text(x + w / 2, top + 1.5, "$D$", ha="center", va="bottom",
            fontsize=7.5, color=INK)
    ax.text(x - 1.2, (y0 + top) / 2, "$\\Sigma C{\\cdot}N$", ha="right",
            va="center", fontsize=7.5, color=INK, rotation=90)
    ax.text(x + w / 2, y0 - 0.9, "grouped by location", ha="center", va="top",
            fontsize=7.8, fontweight="bold", color=INK)


def draw_fused_grid(ax):
    """Fused [B, N, D] grid."""
    x0, y0, cw, pitch = 49.0, 38.0, 2.4, 2.8
    B, N = 4, 5
    
    top, right = grid_cells(ax, x0, y0, B, N, cw, pitch, edge=BLUE,
                            face=tint(BLUE, 0.76), lw=1.1, rounding=0.25, z=3)
    
    # Axis labels
    ax.text(x0 + (N - 1) * pitch / 2 + cw / 2, top + 0.9, "$N$", ha="center",
            va="bottom", fontsize=7.5, color=INK)
    ax.text(right + 0.8, top + 0.9, "$D$", ha="left", va="bottom",
            fontsize=7.5, color=INK)
    ax.text(x0 - 0.8, y0 + (B - 1) * pitch / 2 + cw / 2, "$B$", ha="right",
            va="center", fontsize=7.5, color=INK)
    ax.text(x0 + (N - 1) * pitch / 2 + cw / 2, y0 - 0.9, "fused",
            ha="center", va="top", fontsize=7.8, fontweight="bold", color=INK)


def draw_encoder_memory(ax):
    """Encoder: [B·M, D] memory + [Kf, D] visible tokens."""
    x, y0, w, h, pitch = 70.0, 32.0, 8.0, 2.0, 2.4
    
    # Draw M memory tokens
    n_mem = 3
    for i in range(n_mem):
        yy = y0 + i * pitch
        rrect(ax, x, yy, w, h, edge=ORANGE, face=tint(ORANGE, 0.80), lw=1.2,
              rounding=0.3, z=3)
    
    # Draw Kf visible tokens
    n_vis = 4
    for i in range(n_vis):
        yy = y0 + (n_mem + i) * pitch
        rrect(ax, x, yy, w, h, edge=BLUE, face=tint(BLUE, 0.80), lw=1.2,
              rounding=0.3, z=3)
    
    top = y0 + (n_mem + n_vis - 1) * pitch + h
    
    # Axis labels and brackets
    sq_bracket(ax, x + w + 0.7, y0, top, color=MUTED, lw=1.1, tick=0.6)
    ax.text(x + w / 2, top + 1.2, "$D$", ha="center", va="bottom",
            fontsize=7.5, color=INK)
    
    # Separate brackets for M and Kf sections
    mem_top = y0 + (n_mem - 1) * pitch + h
    ax.text(x - 1.4, (y0 + mem_top) / 2, "$B{\\cdot}M$", ha="right",
            va="center", fontsize=6.5, color=ORANGE, rotation=90)
    vis_y0 = y0 + n_mem * pitch
    ax.text(x - 1.4, (vis_y0 + top) / 2, "$K_f$", ha="right", va="center",
            fontsize=6.5, color=BLUE, rotation=90)
    
    ax.text(x + w / 2, y0 - 0.9, "encoder", ha="center", va="top",
            fontsize=7.8, fontweight="bold", color=INK)


def draw_decoder_targets(ax):
    """Decoder: [Kt, D] queries -> [Kt, P²] predictions."""
    x, y0, w, h, pitch = 87.0, 36.0, 8.0, 2.0, 2.4
    
    # Queries
    n_targets = 4
    for i in range(n_targets):
        yy = y0 + i * pitch
        rrect(ax, x, yy, w, h, edge=VIOLET, face=tint(VIOLET, 0.82), lw=1.2,
              rounding=0.3, z=3)
    
    top = y0 + (n_targets - 1) * pitch + h
    
    # Axis labels
    sq_bracket(ax, x + w + 0.7, y0, top, color=MUTED, lw=1.1, tick=0.6)
    ax.text(x + w / 2, top + 1.2, "$D$", ha="center", va="bottom",
            fontsize=7.5, color=INK)
    ax.text(x - 0.8, (y0 + top) / 2, "$K_t$", ha="right", va="center",
            fontsize=7.5, color=INK, rotation=90)
    ax.text(x + w / 2, y0 - 0.9, "targets", ha="center", va="top",
            fontsize=7.8, fontweight="bold", color=INK)
    
    # Arrow to predictions
    arrow(ax, (x + w, y0 + (top - y0) / 2), (x + w + 5.5, y0 + (top - y0) / 2),
          color=VIOLET, lw=1.4)
    
    # Predictions
    px = x + w + 6.0
    pw = 6.0
    for i in range(n_targets):
        yy = y0 + i * pitch
        rrect(ax, px, yy, pw, h, edge=VIOLET, face=tint(VIOLET, 0.82), lw=1.2,
              rounding=0.3, z=3)
    
    # Predictions axis labels
    sq_bracket(ax, px + pw + 0.7, y0, top, color=MUTED, lw=1.1, tick=0.6)
    ax.text(px + pw / 2, top + 1.2, "$P^2$", ha="center", va="bottom",
            fontsize=7.5, color=INK)
    ax.text(px - 0.8, (y0 + top) / 2, "$K_t$", ha="right", va="center",
            fontsize=7.5, color=INK, rotation=90)
    ax.text(px + pw / 2, y0 - 0.9, "predictions", ha="center", va="top",
            fontsize=7.8, fontweight="bold", color=INK)


def draw_inference_pooled(ax):
    """Inference: pooled descriptor [B, D]."""
    x, y0, cw, pitch = 70.0, 15.0, 2.4, 2.8
    B = 4
    
    # Single column of B cells
    for i in range(B):
        yy = y0 + i * pitch
        rrect(ax, x, yy, cw, cw, edge=BLUE, face=tint(BLUE, 0.76), lw=1.1,
              rounding=0.25, z=3)
    
    top = y0 + (B - 1) * pitch + cw
    
    # Axis labels
    ax.text(x + cw / 2, top + 0.9, "$D$", ha="center", va="bottom",
            fontsize=7.5, color=INK)
    ax.text(x - 0.8, (y0 + top) / 2, "$B$", ha="right", va="center",
            fontsize=7.5, color=INK, rotation=90)
    ax.text(x + cw / 2, y0 - 0.9, "pooled", ha="center", va="top",
            fontsize=7.8, fontweight="bold", color=INK)
    
    # Arrow from encoder
    arrow(ax, (78.0 - 3.0, 44.0), (x + cw / 2, top), color=BLUE, lw=1.3,
          dashed=True, rad=0.15)
    ax.text(77.0, 31.5, "inference pool", ha="left", va="center",
            fontsize=6.2, color=BLUE, style="italic")


def draw_ghost_padded(ax):
    """The ghost [B, C_max, N, D] tensor that is never materialised."""
    x, y0, w, h = 28.0, 9.0, 20.0, 10.0
    
    rrect(ax, x, y0, w, h, edge=MUTED, face="#F8F8F8", lw=1.8, dashed=True,
          rounding=0.7, z=1)
    
    # Red slash
    ax.plot([x + 1.5, x + w - 1.5], [y0 + 1.5, y0 + h - 1.5],
            color=RED, lw=2.8, alpha=0.9, zorder=5)
    
    # Text
    ax.text(x + w / 2, y0 + h / 2 + 2.2, "$[B, C_{\\max}, N, D]$",
            ha="center", va="center", fontsize=8.5, color=MUTED,
            fontweight="bold", zorder=6)
    ax.text(x + w / 2, y0 + h / 2 - 2.2, "never materialised",
            ha="center", va="center", fontsize=7.0, color=RED,
            style="italic", fontweight="bold", zorder=6)


# --------------------------------------------------------------------------- #
# Main figure
# --------------------------------------------------------------------------- #
def build() -> plt.Figure:
    fig = plt.figure(figsize=(15.5, 7.8))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_xlim(0, 106)
    ax.set_ylim(0, 66)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    # ---- title ------------------------------------------------------------ #
    ax.text(1, 63.0, "L2", ha="left", va="center", fontsize=15,
            fontweight="bold", color=INK)
    ax.text(5.5, 63.0,
            "Tensor layout: literal array stacks with axis labels (Perceiver-IO Fig 2 idiom)",
            ha="left", va="center", fontsize=11.5, fontweight="bold", color=INK)

    # ---- array stacks ----------------------------------------------------- #
    draw_input_images(ax)
    draw_ragged_tokens(ax)
    draw_channel_grouping(ax)
    draw_fused_grid(ax)
    draw_encoder_memory(ax)
    draw_decoder_targets(ax)
    draw_inference_pooled(ax)
    draw_ghost_padded(ax)

    # ---- main dataflow arrows --------------------------------------------- #
    ymid = 44.0
    arrow(ax, (6.5, ymid), (13.0, ymid), lw=1.5)
    edge_label(ax, 9.7, ymid + 1.3, "embed", size=6.5)
    
    arrow(ax, (22.0, ymid), (29.0, ymid), lw=1.5)
    
    arrow(ax, (38.0, ymid), (49.0, ymid), lw=1.5, color=RED)
    edge_label(ax, 43.5, ymid + 1.3, "channel\nfusion", size=6.5, color=RED)
    
    arrow(ax, (62.5, ymid), (70.0, ymid), lw=1.5)
    edge_label(ax, 66.2, ymid + 1.3, "enc", size=6.5)
    
    arrow(ax, (78.0, 44.0), (87.0, 44.0), lw=1.5, color=VIOLET)
    edge_label(ax, 82.5, 44.0 + 1.3, "dec", size=6.5, color=VIOLET)

    # ---- key note --------------------------------------------------------- #
    ax.text(53.0, 6.0,
            "Note: $C$ appears only in the input set length $\\Sigma C{\\cdot}N$. "
            "Channel fusion sums it away at each $(b,n)$ to produce fixed-size $[B, N, D]$.",
            ha="center", va="center", fontsize=7.5, color=MUTED,
            style="italic", bbox=dict(boxstyle="round,pad=0.6", facecolor="white",
                                      edgecolor=MUTED, linewidth=1.0))

    # ---- subtitle --------------------------------------------------------- #
    ax.text(1, 59.8,
            "The output shape is independent of the channel count $C$. "
            "Omitted tokens = masked by DCP (dashed). "
            "The padded tensor is never created.",
            ha="left", va="center", fontsize=7.0, color=MUTED, style="italic")

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
