"""L1 -- system-level block diagram of the padding-free channel-fusion MAE.

A single hand-authored, vector-first figure (PDF/SVG/PNG) in the idiom of the
reference architecture figures in ``references/`` (Panopticon Fig. 1,
Perceiver-IO Fig. 2): literal array stacks with axis labels, module boxes,
dashed stage groups / shared-weight brackets, and shapes written on the edges.

Shows the *shipped default* training path (left to right) plus the inference
``encode()`` branch, the ghost ``[B, C_max, N, D]`` tensor that is never
materialised, the MAE asymmetry (encoder sees only visible tokens; the
channel-aware decoder reconstructs omitted channel-patches), and a glossary.

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
    TOKENIZER, FUSION, ENCODER, DECODER, LOSS, FUSED, POOL,
    INK, MUTED, CHANNEL_COLORS,
    tint, rrect, box, arrow, edge_label, sq_bracket, sq_bracket_mirror,
    install_rcparams, save_figure, legend, channel_legend, padded_ghost, GHOST,
)

OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(parents=True, exist_ok=True)

# Vertical centre of the whole training track: every array stack is centred here
# and every main-flow arrow runs along this line.
YMID = 46.0


# --------------------------------------------------------------------------- #
# Small local helpers
# --------------------------------------------------------------------------- #
def dim_extent(ax, x, y, w, text="$D$", color=MUTED):
    """Horizontal double arrow marking a tensor's row width (columns)."""
    ax.annotate(
        "", xy=(x, y), xytext=(x + w, y),
        arrowprops=dict(arrowstyle="<->", color=color, lw=0.9,
                        shrinkA=0.0, shrinkB=0.0),
        zorder=5,
    )
    ax.text(x + w / 2, y - 1.3, text, ha="center", va="center",
            fontsize=6.4, color=color, zorder=6)


def h_bracket(ax, x0, x1, y, color=MUTED, lw=1.1, tick=0.55, down=True):
    """Horizontal square bracket spanning ``x0..x1`` at height ``y``."""
    ax.plot([x0, x1], [y, y], color=color, lw=lw, zorder=3)
    d = -tick if down else tick
    ax.plot([x0, x0], [y, y + d], color=color, lw=lw, zorder=3)
    ax.plot([x1, x1], [y, y + d], color=color, lw=lw, zorder=3)


# --------------------------------------------------------------------------- #
# Array stacks  (all vertically centred on YMID)
# --------------------------------------------------------------------------- #
def draw_input_set(ax):
    """Set of single-channel images -- the variable axis is the height."""
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
    ax.text(x, y0 - 1.9, "$C_b$ channels $\\cdot$ order-free",
            ha="left", va="top", fontsize=6.6, color=MUTED)
    ax.text(x - 1.0, YMID, "$1\\times H\\times W$", ha="right", va="center",
            fontsize=6.8, color=MUTED, rotation=90)
    ax.text(x + cw + 0.2, y0 + 1.8, "$\\vdots$", ha="left", va="center",
            fontsize=10, color=MUTED)
    ax.text(x + cw + 0.2, top - 2.0, "$\\vdots$", ha="left", va="center",
            fontsize=10, color=MUTED)


def draw_token_list(ax):
    """Flattened ``[V, D]`` visible token set -- one row per visible ch-patch."""
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
              lw=1.4 if omitted else 1.2, dashed=omitted,
              crossed=not omitted)
        if omitted:
            ax.text(x + w / 2, yy + h / 2, "omitted row (DCP)", ha="center",
                    va="center", fontsize=6.0, color=LOSS, style="italic",
                    zorder=4)
    top = y0 + total
    sq_bracket_mirror(ax, x + w + 1.0, y0, top)
    ax.text(x + w + 1.6, top, "$[\\,V,\\; D\\,]$",
            ha="left", va="center", fontsize=8.0, color=INK)
    ax.text(x + w + 1.6, top - 2.1, "$V \\leq \\Sigma_b\\, C_b{\\cdot}N_b$",
            ha="left", va="center", fontsize=6.0, color=MUTED, style="italic")
    ax.text(x + w / 2, top + 2.6, "visible token set", ha="center", va="center",
            fontsize=8.4, fontweight="bold", color=INK)
    dim_extent(ax, x, y0 - 1.8, w)


def draw_fused_grid(ax, x0=79.0):
    """Fused embeddings: one ``[D]`` vector per present ``(b, n)`` location.

    The object is ``[Kf, D]`` -- one row per present location, so ``Kf`` never
    appears as its own axis; ``B`` (rows) and ``N`` (columns) are the location
    grid axes.  The bottom double arrow spans the ``N`` axis, not ``D``.
    """
    cell, pitch = 2.9, 3.3
    B, N = 3, 4
    gh = (B - 1) * pitch + cell
    y0 = YMID - gh / 2
    for b in range(B):
        for n in range(N):
            rrect(ax, x0 + n * pitch, y0 + b * pitch, cell, cell,
                  edge=FUSED, face=tint(FUSED, 0.72), lw=1.1, rounding=0.3,
                  crossed=True)
    top = y0 + gh
    right = x0 + (N - 1) * pitch + cell
    cx = (x0 + right) / 2

    # location-grid axes: B down the rows, N across the columns
    sq_bracket(ax, x0 - 0.9, y0, top)
    ax.text(x0 - 2.0, y0 + 0.22 * (top - y0), "$B$", ha="right", va="center",
            fontsize=6.6, color=MUTED)
    dim_extent(ax, x0, y0 - 1.8, (N - 1) * pitch + cell, text="$N$ patches")

    # enclosing bracket for the whole fused object (rows are Kf, width is D)
    h_bracket(ax, x0, right, top + 0.7)
    ax.text(cx, top + 1.9, "$[\\,K_f,\\; D\\,]$", ha="center", va="center",
            fontsize=7.4, color=INK)
    ax.text(cx, top + 3.2, "one row per present $(b,n)$", ha="center",
            va="center", fontsize=5.6, color=MUTED, style="italic")
    ax.text(cx, top + 5.2, "fused embeddings", ha="center", va="center",
            fontsize=8.4, fontweight="bold", color=INK)
    return right


def draw_decoder_kv(ax, x=124.0):
    """``[B . L, D]`` decoder KV (the memory arm would prepend ``mem_out``)."""
    w, h, pitch = 10.5, 2.0, 2.2
    n_lat = 9
    total = (n_lat - 1) * pitch + h
    y0 = YMID - total / 2
    top = y0 + total

    lat_y0 = y0
    for i in range(n_lat):
        rrect(ax, x, lat_y0 + i * pitch, w, h, edge=POOL,
              face=tint(POOL, 0.8), lw=1.2, crossed=True)
    lat_top = lat_y0 + (n_lat - 1) * pitch + h

    sq_bracket_mirror(ax, x + w + 1.0, y0, top)

    ax.text(x - 0.7, (lat_y0 + lat_top) / 2, "$L$", ha="right", va="center",
            fontsize=6.2, color=POOL, zorder=5)

    ax.text(x + w / 2, top + 1.9,
            "$[\\,B\\cdot L,\\; D\\,]$", ha="center", va="center",
            fontsize=7.4, color=INK)
    ax.text(x + w / 2, top + 3.3, "rows illustrative $\\cdot$ default $L{=}16$",
            ha="center", va="center", fontsize=5.7, color=MUTED, style="italic")
    ax.text(x + w / 2, top + 5.1, "decoder KV ($enc_{all}$)", ha="center",
            va="center", fontsize=8.4, fontweight="bold", color=INK)
    return x, y0, top, lat_y0


def draw_output(ax, x, top, w=21.0):
    """``[K_t, p^2]`` predicted pixels for the omitted channel-patches."""
    h, pitch, rows = 2.6, 3.5, 4
    total = (rows - 1) * pitch + h
    y0 = top - total
    for i in range(rows):
        rrect(ax, x, y0 + i * pitch, w, h, edge=DECODER,
              face=tint(DECODER, 0.82), lw=1.2, crossed=True)
    return top, y0


def draw_glossary(ax):
    """Concise symbol glossary strip along the bottom of the figure."""
    rrect(ax, 3.0, -14.0, 158.0, 12.0, edge=MUTED, face="#FAFAFA", lw=1.0,
          dashed=True, rounding=0.6, z=1)
    ax.text(5.0, -3.4, "symbols / glossary", ha="left", va="center", fontsize=7.4,
            fontweight="bold", color=INK, zorder=5)
    cols = [
        (6.0, [
            ("$B$", "batch size"),
            ("$C_b$", "channels of sample $b$"),
            ("$\\Sigma C$", "channels in batch"),
            ("$H,\\,W$", "image side (square)"),
            ("$N$", "patches / channel (max$_b\\,N_b$)"),
            ("$D$", "token / embedding dim"),
        ]),
        (42.0, [
            ("$M$", "memory tokens (memory arm)"),
            ("$L$", "latents (default 16)"),
            ("$V$", "visible tokens ($\\leq\\sum_b C_b N_b$)"),
            ("$K_f$", "present $(b,n)$ locations"),
            ("$K_t$", "masked ch-patches (targets)"),
            ("$P$", "canonical patch size"),
        ]),
        (82.0, [
            ("$p_b$", "runtime patch size"),
            ("pixel_size", "$\\mu$m/px (fleet table)"),
            ("target_um", "target patch extent"),
            ("$z_c$", "channel identity (bias+skip)"),
            ("$enc_{all}$", "decoder KV $[\\,B\\cdot L, D\\,]$"),
        ]),
        (120.0, [
            ("PMA", "pooled multi-head attention"),
            ("RoPE", "rotary position embedding"),
            ("PI-resize", "FlexiViT kernel resizing"),
            ("DCP", "dynamic channel-patch masking"),
            ("scale", "$\\log p,\\ \\log\\mu$m/px cond."),
            ("ghost", "padded $[B,C_{\\max},N,D]$, never allocated"),
        ]),
    ]
    y0, pitch, off = -4.6, 1.35, 8.0
    for x, entries in cols:
        for i, (term, desc) in enumerate(entries):
            yy = y0 - i * pitch
            ax.text(x, yy, term, ha="left", va="center", fontsize=5.6,
                    fontweight="bold", color=INK, zorder=5)
            ax.text(x + off, yy, desc, ha="left", va="center", fontsize=5.6,
                    color=MUTED, zorder=5)
    ax.text(6.0, -12.9,
            "$p_b = \\max(1,\\ \\lfloor\\, \\mathrm{target\\_um}"
            " / \\mathrm{pixel\\_size}_b + 0.5 \\rfloor)$",
            ha="left", va="center", fontsize=5.9, color=INK, zorder=5)


# --------------------------------------------------------------------------- #
# The figure
# --------------------------------------------------------------------------- #
def build() -> plt.Figure:
    fig = plt.figure(figsize=(16.0, 8.9))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_xlim(0, 164)
    ax.set_ylim(-15, 76)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    # ---- title ------------------------------------------------------------ #
    ax.text(2, 72.4, "L1", ha="left", va="center", fontsize=15,
            fontweight="bold", color=INK)
    ax.text(6.5, 72.4, "System: padding-free channel-fusion MAE (training path + inference)",
            ha="left", va="center", fontsize=12.2, fontweight="bold", color=INK)
    ax.text(2.0, 68.2,
            "The encoder only ever sees visible tokens; the decoder is conditioned on "
            "the target channel's identity and fractional position — one shared decoder; "
            "mask tokens never enter the encoder token set.",
            ha="left", va="center", fontsize=7.2, color=MUTED, style="italic")

    # ---- arrays ----------------------------------------------------------- #
    draw_input_set(ax)
    draw_token_list(ax)
    grid_right = draw_fused_grid(ax)
    kv_x, _kv_y0, kv_top, kv_lat_y0 = draw_decoder_kv(ax)

    # ---- module boxes (all centred on YMID) ------------------------------- #
    box(ax, 15, YMID - 7.5, 22, 15, "Channel-tied patch embed",
        "canonical Conv $1{\\to}D$ (kernel $P$)\n"
        "PI-resized $+$ stride $p_b$ per sample\n"
        "carry $z_c$ (channel identity)",
        edge=TOKENIZER, sub_size=5.7)
    box(ax, 56, YMID - 7.5, 19, 15, "Channel fusion",
        "pool over channel axis per $(b,n)$\n"
        "single-query PMA, no spatial table\n"
        "$O(C{\\cdot}D{\\cdot}D_{attn})+O(D{\\cdot}D_{attn})$ per loc",
        edge=FUSION, sub_size=5.4)
    box(ax, 95, YMID - 7.5, 24, 15, "TransformerEncoder",
        "grouped self-attention $\\cdot$ 2-D axial RoPE\n"
        "RoPE on metric $(r,c)$ $\\cdot$ no memory tokens\n"
        "depth blocks (default 4)\n"
        "scale cond (add): $\\log p,\\ \\log\\mu$m/px",
        edge=ENCODER, sub_size=5.5)
    box(ax, 138, YMID - 7.5, 21, 15, "Channel-aware decoder",
        "target self-attn\n"
        "$q=\\mathrm{mask}+\\mathrm{fracPE}(r,c)$\n"
        "$\\quad+\\,\\mathrm{target\\_proj}(z_c+\\mathrm{scale})$\n"
        "cross-attn: latents $[B,L,D]$", edge=DECODER, sub_size=5.2)

    # ---- arrows along the main track ------------------------------------- #
    arrow(ax, (9.8, YMID), (15.0, YMID))
    # stacked two-line shape label, kept clear of the tiles and the box edge
    edge_label(ax, 13.3, YMID - 4.0, "$[\\Sigma C,$\n$1,H,W]$", size=6.3)
    arrow(ax, (37.0, YMID), (41.0, YMID))
    arrow(ax, (52.5, YMID), (56.0, YMID))
    arrow(ax, (75.0, YMID), (79.0, YMID))
    arrow(ax, (grid_right, YMID), (95.0, YMID))
    arrow(ax, (kv_x + 11.5, YMID), (138.0, YMID))

    # ---- DCP masker chip feeds the visible token list --------------------- #
    box(ax, 15, 56.0, 22, 6.2, "DCP visibility sampler", "Eq. 1 · token omission (never zero)",
        edge=TOKENIZER, title_size=7.2, sub_size=6.0)
    # route the elbow left of the embed box, then below it into the omitted row,
    # so it never crosses the embed -> token-list arrow
    ax.plot([15.0, 11.1, 11.1, 40.0, 40.0],
            [59.1, 59.1, 37.0, 37.0, 38.5], color=TOKENIZER, lw=1.3, zorder=5)
    arrow(ax, (40.0, 38.5), (41.0, 38.5), color=TOKENIZER, lw=1.3)

    # ---- shared-weight bracket over the transformer ----------------------- #
    rrect(ax, 92.5, YMID - 9.7, 28.0, 19.4, edge=ENCODER, face="none", lw=1.3,
          dashed=True, rounding=0.9, z=1)
    ax.text(106.5, YMID + 10.3, "depth blocks (default 4)",
            ha="center", va="center", fontsize=6.9, color=ENCODER)

    # ---- pooling stage: tok_out -> the fixed latent set ------------------- #
    box(ax, 86, 15.0, 21, 9.0, "spatial pooling (PMA)",
        "$n_q = L$; PMA$(\\mathrm{tok}_{out})$\n$\\to$ latents $[B,L,D]$",
        edge=POOL, title_size=6.8, sub_size=5.5)

    # tok_out content tokens feed the pooler; routed *below* it so the
    # pooler -> decoder-KV line never crosses this bus
    ax.plot([98.0, 98.0, 85.0, 85.0, 96.5],
            [38.5, 36.0, 36.0, 11.5, 11.5], color=FUSED, lw=1.4, zorder=2)
    arrow(ax, (96.5, 11.5), (96.5, 15.0), color=FUSED, lw=1.4)
    ax.text(108.0, 12.7, "tok_out $[K_f,D]$ (not in decoder KV)",
            ha="center", va="center", fontsize=6.0, color=FUSED, zorder=7,
            bbox=dict(boxstyle="round,pad=0.14", facecolor="white",
                      edgecolor="none", alpha=0.95))
    ax.text(106.0, 9.0,
            "training path: this pooler builds the decoder KV;\n"
            "encode() returns these latents",
            ha="center", va="center", fontsize=6.0, color=POOL, zorder=7,
            linespacing=1.35,
            bbox=dict(boxstyle="round,pad=0.16", facecolor="white",
                      edgecolor="none", alpha=0.95))

    # spatial pooling -> latents rows of the decoder KV
    ax.plot([96.5, 96.5, 121.5, 121.5], [24.0, 32.0, 32.0, YMID - 4.0],
            color=POOL, lw=1.5, zorder=2)
    arrow(ax, (121.5, YMID - 4.0), (kv_x, YMID - 4.0), color=POOL, lw=1.5)
    # ---- ghost padded tensor (never materialised) ------------------------- #
    padded_ghost(ax, 3.0, 2.0, 41.0, 7.0, note="never materialised")

    # ---- legends (empty lower-left band) ---------------------------------- #
    legend(
        ax, 3.0, 27.0,
        [("tokenizer", TOKENIZER, "fill"),
         ("fusion", FUSION, "fill"),
         ("encoder", ENCODER, "fill"),
         ("decoder", DECODER, "fill"),
         ("fused tokens $[K_f,D]$", FUSED, "cross"),
         ("latents $[B,L,D]$", POOL, "cross"),
         ("omitted row / loss", LOSS, "omit"),
         ("ghost padded tensor", GHOST, "ghost")],
        title="roles & data", pitch=1.45, sw=1.2, fontsize=5.8,
        title_size=6.5, label_gap=0.9,
    )
    channel_legend(ax, 29.0, 27.0, n=len(CHANNEL_COLORS),
                   title="channel identity", pitch=1.45, sw=1.2,
                   fontsize=5.8, title_size=6.5, label_gap=0.9)

    # ---- decoder -> shared head -> output -> loss ------------------------- #
    box(ax, 138.0, 29.5, 21.0, 7.5, "Reconstruction head",
        "one Linear $D{\\to}p^2$ per configured $p$\n"
        "nearest configured head with $p_{head}\\geq p$,\n"
        "bilinear, never upsamples", edge=DECODER, title_size=6.8,
        sub_size=5.1)
    arrow(ax, (148.5, YMID - 8.0), (148.5, 37.0), color=DECODER, lw=1.5)
    otop, obot = draw_output(ax, 138.0, 28.0)
    arrow(ax, (148.5, 29.5), (148.5, otop), color=DECODER, lw=1.5)
    ax.text(147.0, 13.7,
            "$[\\Sigma\\,p^2]$ predicted pixels",
            ha="right", va="center", fontsize=7.0, fontweight="bold", color=INK)
    ax.text(147.0, 12.6,
            "flat: row $i$ contiguous $p_i^2$; no padding",
            ha="right", va="center", fontsize=5.9, color=MUTED)
    box(ax, 138.0, 3.0, 21.0, 8.0, "reconstruction loss",
        "standardised-target MSE,\n"
        "mean per sample (loss_reduce=sample)",
        edge=LOSS, title_size=6.8, sub_size=5.4)
    arrow(ax, (148.5, obot), (148.5, 11.0), color=LOSS, lw=1.5)
    ax.text(136.5, 7.0,
            "flat $[\\Sigma p^2]$ layout:\n"
            "no padded block / validity mask",
            ha="right", va="center", fontsize=5.8, color=MUTED, linespacing=1.3)

    # ---- target pixels, routed along the bottom --------------------------- #
    ax.plot([1.5, 1.5, 148.5, 148.5], [31.4, 1.0, 1.0, 3.0],
            color=MUTED, lw=1.2, linestyle="--", zorder=2)
    arrow(ax, (148.5, 1.6), (148.5, 3.0), color=MUTED, lw=1.2, dashed=True,
          ms=9)
    ax.text(74.0, 1.8, "target = raw pixels of the masked channel-patches",
            ha="center", va="bottom", fontsize=6.9, color=MUTED, style="italic")

    # ---- symbol glossary -------------------------------------------------- #
    draw_glossary(ax)

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
