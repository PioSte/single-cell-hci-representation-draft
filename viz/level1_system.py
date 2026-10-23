"""L1 -- system-level block diagram of the padding-free channel-fusion MAE.

A single hand-authored, vector-first figure (PDF/SVG/PNG) in the idiom of the
reference architecture figures in ``references/`` (Panopticon Fig. 1,
Perceiver-IO Fig. 2): literal array stacks with axis labels, module boxes,
dashed stage groups / shared-weight brackets, and shapes written on the edges.

Shows the *training* path (left to right) plus the inference ``encode()``
branch, the ghost ``[B, C_max, N, D]`` tensor that is never materialised, and
the MAE asymmetry (encoder sees only visible tokens; the channel-aware decoder
reconstructs omitted channel-patches).

Usage::

    uv run python viz/level1_system.py
    uv run python viz/level1_system.py --formats pdf svg
    uv run python viz/level1_system.py --stem level1_system
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from figure_style import (  # noqa: E402  (sibling module, run from viz/)
    TOKENIZER, FUSION, ENCODER, DECODER, MEMORY, LOSS, FUSED, POOL,
    INK, MUTED, CHANNEL_COLORS,
    tint, rrect, box, arrow, edge_label, sq_bracket, install_rcparams,
    save_figure, role_legend, channel_legend, padded_ghost,
)

OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(parents=True, exist_ok=True)

# Vertical centre of the whole training track: every array stack is centred here
# and every main-flow arrow runs along this line.
YMID = 46.0


# --------------------------------------------------------------------------- #
# Small local helper
# --------------------------------------------------------------------------- #
def dim_extent(ax, x, y, w, text="$D$", color=MUTED):
    """Horizontal double arrow marking the embedding width ``D`` of a stack."""
    ax.annotate(
        "", xy=(x, y), xytext=(x + w, y),
        arrowprops=dict(arrowstyle="<->", color=color, lw=0.9,
                        shrinkA=0.0, shrinkB=0.0),
        zorder=5,
    )
    ax.text(x + w / 2, y - 1.4, text, ha="center", va="center",
            fontsize=6.4, color=color, zorder=6)


# --------------------------------------------------------------------------- #
# Array stacks  (all vertically centred on YMID)
# --------------------------------------------------------------------------- #
def draw_input_set(ax):
    """Ragged set of single-channel images -- the variable axis is the height."""
    cw, pitch, x = 6.8, 5.6, 3.0
    n = len(CHANNEL_COLORS)
    y0 = YMID - (n - 1) * pitch / 2 - cw / 2
    for i, c in enumerate(CHANNEL_COLORS):
        yy = y0 + (n - 1 - i) * pitch
        rrect(ax, x, yy, cw, cw, edge=c, face=tint(c, 0.72), lw=1.5)
        # a faint "image" squiggle so the tiles read as pictures
        ax.plot([x + 1.0, x + cw - 1.0], [yy + 1.9, yy + cw - 2.2],
                color=c, lw=1.0, alpha=0.7, zorder=4)
        ax.plot([x + 1.0, x + cw - 1.0], [yy + cw - 2.2, yy + 1.9],
                color=c, lw=1.0, alpha=0.4, zorder=4)
    top = y0 + (n - 1) * pitch + cw
    ax.text(x + cw / 2, top + 2.6, "input set", ha="center", va="center",
            fontsize=8.4, fontweight="bold", color=INK)
    ax.text(x + cw / 2, y0 - 2.6, "$C_b$ channels\norder-free", ha="center",
            va="center", fontsize=6.6, color=MUTED, linespacing=1.35)
    ax.text(x - 1.0, YMID, "$1\\times H\\times W$", ha="right", va="center",
            fontsize=6.8, color=MUTED, rotation=90)
    ax.text(x + cw + 0.7, y0 + 1.8, "$\\vdots$", ha="left", va="center",
            fontsize=10, color=MUTED)
    ax.text(x + cw + 0.7, top - 2.0, "$\\vdots$", ha="left", va="center",
            fontsize=10, color=MUTED)


def draw_ragged_tokens(ax):
    """Flattened ``[sum C . N, D]`` token set -- one row per channel-patch.

    Row colours follow the input channel colours; the DCP-omitted row is drawn
    as a dashed white row with the LOSS red edge (absent, never zeroed).
    """
    x, w, h, pitch, rows = 41.0, 10.5, 2.4, 3.0, 10
    total = (rows - 1) * pitch + h
    y0 = YMID - total / 2
    omit = 2
    for i in range(rows):
        yy = y0 + i * pitch
        c = CHANNEL_COLORS[((rows - 1 - i) // 2) % len(CHANNEL_COLORS)]
        omitted = i == omit
        rrect(ax, x, yy, w, h, edge=(LOSS if omitted else c),
              face=("#FFFFFF" if omitted else tint(c, 0.8)),
              lw=1.4 if omitted else 1.2, dashed=omitted)
        if omitted:
            ax.text(x + w / 2, yy + h / 2, "omitted row (DCP)", ha="center",
                    va="center", fontsize=6.0, color=LOSS, style="italic",
                    zorder=4)
    top = y0 + total
    sq_bracket(ax, x + w + 1.0, y0, top)
    ax.text(x + w + 1.6, top, "$[\\,\\Sigma C\\cdot N,\\; D\\,]$",
            ha="left", va="center", fontsize=8.0, color=INK)
    ax.text(x + w / 2, top + 2.6, "ragged token set", ha="center", va="center",
            fontsize=8.4, fontweight="bold", color=INK)
    dim_extent(ax, x, y0 - 1.8, w)


def draw_joint_grid(ax, x0=79.0):
    """``[B, N, D]`` joint per-patch embeddings after channel fusion."""
    cell, pitch = 2.9, 3.3
    B, N = 3, 4
    gh = (B - 1) * pitch + cell
    y0 = YMID - gh / 2
    for b in range(B):
        for n in range(N):
            rrect(ax, x0 + n * pitch, y0 + b * pitch, cell, cell,
                  edge=FUSED, face=tint(FUSED, 0.72), lw=1.1, rounding=0.3)
    top = y0 + gh
    cx = x0 + (N - 1) * pitch / 2 + cell / 2
    ax.text(cx, top + 1.0, "$N$ patches", ha="center", va="center",
            fontsize=6.6, color=MUTED)
    ax.text(x0 - 0.6, top + 1.0, "$B$", ha="right", va="center",
            fontsize=6.6, color=MUTED)
    ax.text(cx, top + 5.4, "joint embeddings", ha="center", va="center",
            fontsize=8.4, fontweight="bold", color=INK)
    ax.text(cx, top + 3.1, "$[\\,B,\\,N,\\,D\\,]$", ha="center", va="center",
            fontsize=6.4, color=FUSED)
    dim_extent(ax, x0, y0 - 1.8, (N - 1) * pitch + cell)
    return x0 + (N - 1) * pitch + cell


def draw_encoded(ax, x=124.0):
    """``[B . M + K_f + B, D]`` -- the decoder KV: fixed memory rows, visible
    fused tokens, and the trained per-sample descriptor ``desc``.

    ``mae.py`` builds ``enc_all = cat([mem_out, tok_out, desc])``, so the
    descriptor row sits here as well as in the inference branch.
    """
    w, h, pitch, gap = 10.5, 2.4, 3.0, 1.1
    n_tok, n_mem = 5, 2
    total = 2 * h + gap + (n_tok + n_mem - 1) * pitch
    y0 = YMID - total / 2
    # desc: one extra per-sample token appended to the decoder KV
    rrect(ax, x, y0, w, h, edge=POOL, face=tint(POOL, 0.8), lw=1.2)
    ax.text(x + w / 2, y0 + h / 2, "desc  $[B, D]$", ha="center", va="center",
            fontsize=5.8, color=INK, zorder=4)
    y_tok0 = y0 + h + gap
    for i in range(n_tok):
        rrect(ax, x, y_tok0 + i * pitch, w, h, edge=FUSED,
              face=tint(FUSED, 0.8), lw=1.2)
    y_mem0 = y_tok0 + n_tok * pitch
    for j in range(n_mem):
        rrect(ax, x, y_mem0 + j * pitch, w, h, edge=MEMORY,
              face=tint(MEMORY, 0.8), lw=1.2)
    top = y_mem0 + (n_mem - 1) * pitch + h
    ax.text(x + w / 2, top + 2.1, "$[\\,B\\cdot M + K_f + B,\\; D\\,]$",
            ha="center", va="center", fontsize=7.4, color=INK)
    ax.text(x + w / 2, top + 4.7, "encoder tokens = decoder KV", ha="center",
            va="center", fontsize=8.4, fontweight="bold", color=INK)
    ax.text(x - 0.8, y_mem0 + (n_mem - 1) * pitch / 2 + h / 2, "$M$",
            ha="right", va="center", fontsize=6.2, color=MEMORY)
    ax.text(x - 0.8, y_tok0 + (n_tok - 1) * pitch / 2 + h / 2, "$K_f$",
            ha="right", va="center", fontsize=6.2, color=FUSED)
    dim_extent(ax, x, y0 - 1.8, w)
    # tap point for encode(): the bottom token row (clear of the $K_f$ label)
    return x, y0, top, y0 + h / 2, y_tok0 + h / 2


def draw_output(ax, x, w=21.0):
    """``[K_t, P^2]`` predicted pixels for the omitted channel-patches."""
    h, pitch, rows = 2.6, 3.5, 4
    total = (rows - 1) * pitch + h
    top = 31.0
    y0 = top - total
    for i in range(rows):
        rrect(ax, x, y0 + i * pitch, w, h, edge=DECODER,
              face=tint(DECODER, 0.82), lw=1.2)
    return top, y0


# --------------------------------------------------------------------------- #
# The figure
# --------------------------------------------------------------------------- #
def build() -> plt.Figure:
    fig = plt.figure(figsize=(16.0, 7.5))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_xlim(0, 164)
    ax.set_ylim(0, 76)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    # ---- title ------------------------------------------------------------ #
    ax.text(2, 72.4, "L1", ha="left", va="center", fontsize=15,
            fontweight="bold", color=INK)
    ax.text(6.5, 72.4, "System: padding-free channel-fusion MAE (training path + inference)",
            ha="left", va="center", fontsize=12.2, fontweight="bold", color=INK)
    ax.text(2.0, 68.2,
            "The encoder only ever sees visible tokens; the decoder is conditioned on "
            "the target channel's identity — one shared decoder; tokens stay "
            "ragged, with no inserted mask tokens.",
            ha="left", va="center", fontsize=7.2, color=MUTED, style="italic")

    # ---- arrays ----------------------------------------------------------- #
    draw_input_set(ax)
    draw_ragged_tokens(ax)
    grid_right = draw_joint_grid(ax)
    enc_x, enc_y0, enc_top, desc_y, tok_y = draw_encoded(ax)

    # ---- module boxes (all centred on YMID) ------------------------------- #
    box(ax, 15, YMID - 7.5, 22, 15, "Channel-tied patch embed",
        "shared Conv $1{\\to}D$, stride $P$\n$+\\,z_c$; spatial PE post-fusion",
        edge=TOKENIZER, sub_size=6.2)
    box(ax, 56, YMID - 7.5, 19, 15, "Channel fusion",
        "pool over channel axis\nper $(b,n)$ location\nsingle-query PMA $O(C)$",
        edge=FUSION, sub_size=5.7)
    box(ax, 95, YMID - 7.5, 24, 15, "Spatial transformer",
        "grouped self-attention\nover real tokens\n$+M$ fixed memory tokens",
        edge=ENCODER, sub_size=6.1)
    box(ax, 138, YMID - 7.5, 21, 15, "Channel-aware decoder",
        "target self-attn\n$q=\\mathrm{mask} + \\mathrm{pos}_n + Wz_c$\n"
        "cross-attn to mem+tokens+desc", edge=DECODER, sub_size=5.5)

    # ---- arrows along the main track ------------------------------------- #
    arrow(ax, (9.8, YMID), (15.0, YMID))
    # stacked two-line shape label, kept clear of the tiles and the box edge
    edge_label(ax, 12.4, YMID + 3.6, "$[\\Sigma C,$\n$1,H,W]$", size=6.3)
    arrow(ax, (37.0, YMID), (41.0, YMID))
    arrow(ax, (52.5, YMID), (56.0, YMID))
    arrow(ax, (75.0, YMID), (79.0, YMID))
    arrow(ax, (grid_right, YMID), (95.0, YMID))
    arrow(ax, (119.0, YMID), (124.0, YMID))
    arrow(ax, (134.5, YMID), (138.0, YMID))

    # ---- DCP masker chip feeds the patch embed ---------------------------- #
    box(ax, 15, 57.0, 22, 6.2, "DCP visibility sampler", "Eq. 1 · token omission (never zero)",
        edge=TOKENIZER, title_size=7.2, sub_size=6.0)
    arrow(ax, (26.0, 57.0), (26.0, 53.5), color=TOKENIZER, lw=1.3)

    # ---- shared-weight bracket over the transformer ----------------------- #
    rrect(ax, 93.0, YMID - 10.2, 28.0, 19.4, edge=ENCODER, face="none", lw=1.3,
          dashed=True, rounding=0.9, z=1)
    ax.text(107.0, YMID + 10.4, "$\\times L$ blocks",
            ha="center", va="center", fontsize=6.9, color=ENCODER)

    # ---- ghost padded tensor (never materialised) ------------------------- #
    padded_ghost(ax, 3.0, 2.0, 57.0, 9.0)

    # ---- legends (large empty lower-left area) ---------------------------- #
    role_legend(
        ax, 13.0, 32.0,
        [("tokenizer", TOKENIZER), ("fusion", FUSION), ("encoder", ENCODER),
         ("decoder", DECODER), ("memory rows", MEMORY), ("loss / omitted", LOSS),
         ("fused tokens", FUSED), ("pooling", POOL)],
        title="module roles", pitch=2.2, fontsize=6.4, title_size=6.9,
    )
    channel_legend(ax, 31.0, 32.0, n=len(CHANNEL_COLORS),
                   title="channel identity", pitch=2.2, fontsize=6.4,
                   title_size=6.9)

    # ---- inference branch: encode() taps the encoder output --------------- #
    rrect(ax, 62.0, 6.0, 42.0, 24.0, edge=POOL, face=tint(POOL, 0.94), lw=1.3,
          dashed=True, rounding=0.9, z=1)
    ax.text(83.0, 29.2, "encode(): full token set, then pool", ha="center",
            va="center", fontsize=6.9, color=POOL)
    # dashed elbow: content tokens -> pooler (encode() branch)
    ax.plot([enc_x, 117.0, 117.0, 97.0], [tok_y, tok_y, 24.0, 24.0],
            color=FUSED, lw=1.3, linestyle="--", zorder=2)
    arrow(ax, (97.0, 24.0), (94.0, 24.0), color=FUSED, lw=1.4, dashed=True)
    # dashed sky elbow: pooler desc -> the desc row of the decoder KV
    ax.plot([94.0, 122.2, 122.2], [21.0, 21.0, desc_y], color=POOL, lw=1.3,
            linestyle="--", zorder=2)
    arrow(ax, (122.2, desc_y), (enc_x, desc_y), color=POOL, lw=1.4,
          dashed=True)
    box(ax, 72.0, 20.0, 22.0, 8.0, "PMA pooler", "content tokens · trained with MAE",
        edge=POOL, title_size=7.4, sub_size=5.8)
    arrow(ax, (83.0, 20.0), (83.0, 16.0), color=POOL, lw=1.4)
    rrect(ax, 74.0, 7.0, 18.0, 9.0, edge=FUSED, face=tint(FUSED, 0.72), lw=1.5)
    ax.text(83.0, 12.1, "$[\\,B,\\; D\\,]$", ha="center", va="center",
            fontsize=9.0, fontweight="bold", color=INK)
    ax.text(83.0, 8.6, "trained content-only descriptor", ha="center", va="center",
            fontsize=5.8, color=MUTED)

    # ---- decoder output -> loss ------------------------------------------ #
    otop, obot = draw_output(ax, 138.0)
    arrow(ax, (148.5, YMID - 7.5), (148.5, otop), color=DECODER, lw=1.5)
    ax.text(137.0, (otop + obot) / 2 + 0.9, "$[\\,K_t,\\; P^2\\,]$",
            ha="right", va="center", fontsize=8.4, fontweight="bold", color=INK)
    ax.text(137.0, (otop + obot) / 2 - 1.7, "predicted pixels", ha="right",
            va="center", fontsize=6.2, color=MUTED)
    box(ax, 138.0, 3.0, 21.0, 9.0, "reconstruction loss",
        "default: per-patch-standardised MSE",
        edge=LOSS, title_size=7.2, sub_size=6.0)
    arrow(ax, (148.5, obot), (148.5, 12.0), color=LOSS, lw=1.5)

    # ---- target pixels, routed along the bottom --------------------------- #
    ax.plot([1.5, 1.5, 148.5, 148.5], [31.4, 1.0, 1.0, 3.0],
            color=MUTED, lw=1.2, linestyle="--", zorder=2)
    arrow(ax, (148.5, 1.6), (148.5, 3.0), color=MUTED, lw=1.2, dashed=True,
          ms=9)
    ax.text(74.0, 1.8, "target = raw pixels of the masked channel-patches",
            ha="center", va="bottom", fontsize=6.9, color=MUTED, style="italic")

    return fig


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--formats", nargs="+", default=["pdf", "svg", "png"])
    ap.add_argument("--stem", default="level1_system")
    args = ap.parse_args()

    install_rcparams()

    fig = build()
    save_figure(fig, OUT, args.stem, args.formats)
    plt.close(fig)


if __name__ == "__main__":
    main()
