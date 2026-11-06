"""Paper-grade figure of the ``fuse-channels`` core architecture.

One hand-authored, vector-first figure (PDF/SVG/PNG) with two panels:

* **A** -- the end-to-end padding-free channel-fusion MAE: channel-tied
  tokenizer -> ChannelFusion -> transformer encoder -> spatial pooling to a
  fixed size latent -> channel-aware decoder + head.  The decoder KV is drawn
  as the pooled ``latents``; ``tok_out`` is deliberately absent.
* **B** -- the set-native token layout, the **default** ``ChannelFusion``
  (single-query PMA with identity bias + skip), DCP masking, the ragged/padded
  visual rules, and a symbol/role legend.

Only the shipped default path is drawn.  Ablations (``IdentityGatedSum``,
learned/sincos post-fusion PE, ...) are intentionally omitted;
``viz/figure_facts.md`` is the shared source of truth for the shapes.

Usage::

    uv run python viz/paper_figure.py                 # pdf + svg + png
    uv run python viz/paper_figure.py --formats pdf png
    uv run python viz/paper_figure.py --stem fig1     # custom output stem
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from figure_style import (  # noqa: E402
    INK, MUTED, CHANNEL_COLORS,
    TOKENIZER, FUSION, ENCODER, DECODER, LOSS, FUSED, POOL, GHOST,
    tint, rrect, box, arrow, token_rows, sq_bracket, panel_header,
    legend, padded_ghost, install_rcparams, save_figure,
)

OUT = Path(__file__).resolve().parent / "out"

# NOTE: role colours come from ``figure_style`` and are the L0-L9 palette:
#   TOKENIZER green, FUSION orange, ENCODER pink, DECODER violet,
#   LOSS red, FUSED blue, POOL sky, GHOST muted.


# --------------------------------------------------------------------------- #
# Local primitives
# --------------------------------------------------------------------------- #
def data_cell(ax, x, y, w, h, edge, title, sub=None, tsize=7.6, ssize=5.3):
    """Crossed data cell (token / embedding / array), optional sub-line."""
    rrect(ax, x, y, w, h, edge=edge, face=tint(edge, 0.82), lw=1.3,
          rounding=0.5, z=3, crossed=True)
    cx = x + w / 2
    bbox = dict(boxstyle="round,pad=0.12", facecolor="white",
                edgecolor="none", alpha=0.85)
    if sub:
        ax.text(cx, y + h * 0.64, title, ha="center", va="center",
                fontsize=tsize, color=INK, fontweight="bold", zorder=5,
                bbox=bbox)
        ax.text(cx, y + h * 0.28, sub, ha="center", va="center",
                fontsize=ssize, color=MUTED, zorder=5, bbox=bbox)
    else:
        ax.text(cx, y + h / 2, title, ha="center", va="center",
                fontsize=tsize, color=INK, fontweight="bold", zorder=5,
                bbox=bbox)


def route(ax, pts, color=INK, lw=1.5, ms=9):
    """Orthogonal poly-line with an arrowhead on the final segment."""
    for a, b in zip(pts[:-2], pts[1:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=lw,
                solid_capstyle="round", zorder=6)
    arrow(ax, pts[-2], pts[-1], color=color, lw=lw, ms=ms)


# --------------------------------------------------------------------------- #
# Panel A -- end-to-end pipeline
# --------------------------------------------------------------------------- #
def _draw_kv(ax, x, y, w, h):
    """Decoder KV ``enc_all = latents`` (fixed length)."""
    rows = [("latents", POOL)] * 4
    rh, pitch = 2.2, 3.25
    y0 = y + h / 2 - (len(rows) - 1) * pitch / 2 - rh / 2
    for i, (name, color) in enumerate(rows):
        yy = y0 + (len(rows) - 1 - i) * pitch
        rrect(ax, x, yy, w, rh, edge=color, face=tint(color, 0.78), lw=1.2,
              rounding=0.3, z=3, crossed=True)
        if i == 0:  # one label for the whole latent stack
            ax.text(x + w / 2, yy + rh / 2, name, ha="center", va="center",
                    fontsize=6.0, color=INK, fontweight="bold", zorder=5)
    top = y0 + (len(rows) - 1) * pitch + rh
    sq_bracket(ax, x - 1.0, y0, top, color=MUTED, lw=1.1, tick=0.55)
    # Left-aligned so the routed encoder/pool arrow (at the stack centre)
    # drops past the text instead of through it.
    lx = x - 4.5
    ax.text(lx, top + 3.3,
            "$enc\\_all=latents$", ha="left",
            va="center", fontsize=5.7, color=INK, fontweight="bold")
    ax.text(lx, top + 1.6,
            "$[B{\\cdot}L,D]$ fixed, independent of $C,N$;\n"
            "raw $tok\\_out\\notin$ KV",
            ha="left", va="center", fontsize=5.1, color=MUTED,
            linespacing=1.3)
    return top


def draw_pipeline(ax):
    ax.set_xlim(0, 140)
    ax.set_ylim(0, 56)
    ax.axis("off")
    panel_header(ax, 1.0, 53.0, "A",
                 "End-to-end padding-free channel-fusion MAE")

    # ---- row 1: tokenizer -> fusion -> encoder -> pool -------------------- #
    y1, h1, w1 = 30.0, 12.5, 19.0
    xs = [3.0 + i * 23.0 for i in range(6)]
    stages = [
        ("Variable channel set",
         "$\\{I_c\\in\\mathbb{R}^{1\\times H\\times W}\\}$\n"
         "$C_b$ variable, order-free\n$C_b \\leq C_{\\max}$", GHOST),
        ("Channel-tied embed",
         "shared $1{\\to}D$ conv (kernel $P$)\nPI-resized to $p_b$\n"
         "$+$ identity $z_c$ (carried)", TOKENIZER),
        ("Padding-free tokens",
         "ragged $[\\Sigma C\\cdot N,\\ D]$\n"
         "$(b,c,n)$ index; no pad row", TOKENIZER),
        ("ChannelFusion",
         "single-query PMA at each $(b,n)$\nidentity bias $+$ skip, no PE\n"
         "$\\to[K_f,\\ D]$", FUSION),
        ("Transformer encoder",
         "no memory tokens, depth 4\n2-D axial RoPE, metric $p_b\\mu$\n"
         "$\\to tok\\_out[K_f,D]$", ENCODER),
        ("Spatial pool (PMA)",
         "$L$ learned queries pool $tok\\_out$:\n"
         "$\\to latents\\ [B,L,D]$\n"
         "loss-trained, independent of $C,N$", POOL),
    ]
    for x, (title, sub, edge) in zip(xs, stages):
        box(ax, x, y1, w1, h1, title, sub, edge=edge,
            title_size=7.2, sub_size=5.3)
    for i in range(5):
        arrow(ax, (xs[i] + w1, y1 + h1 / 2), (xs[i + 1], y1 + h1 / 2),
              lw=1.6)

    # the padded tensor never exists: attach the headline to a small dashed
    # ghost so the claim is not carried by a leader into empty space.
    gx, gw, gh, gy = 76.0, 22.0, 3.4, 43.6
    ax.text(gx, 50.3, "no $[B,\\,C_{\\max},\\,N,\\,D]$ tensor anywhere",
            ha="center", va="center", fontsize=8.4, fontweight="bold",
            color=LOSS)
    arrow(ax, (gx, 49.6), (gx, gy + gh + 0.2), color=LOSS, lw=1.1, ms=7,
          dashed=True)
    padded_ghost(ax, gx - gw / 2, gy, gw, gh,
                 shape="$[B,\\,C_{\\max},\\,N,\\,D]$",
                 note="never materialised")
    ax.text(xs[2] + w1 / 2, 27.4,
            "solid ragged token list: the only token container",
            ha="center", va="center", fontsize=5.8, color=MUTED,
            style="italic")

    # ---- row 2: KV -> decoder -> head ------------------------------------- #
    y2, h2 = 4.5, 12.5
    kv_x, kv_w = 6.0, 18.0
    kv_top = _draw_kv(ax, kv_x, y2, kv_w, h2)

    box(ax, 40.0, y2, 52.0, h2, "Channel-aware decoder",
        "target self-attn  $\\cdot$  cross-attn to KV\n"
        "fractional metric PE, depth 2", edge=DECODER,
        title_size=7.2, sub_size=5.3)
    box(ax, 104.0, y2, 32.0, h2, "Reconstruction head",
        "$\\mathrm{Linear}(D,p^2)$ per configured $p$\n"
        "mixed $p$: flat $[\\Sigma p^2]$ pixels\n"
        "no padded block, no validity mask",
        edge=DECODER, title_size=7.2, sub_size=4.9)

    arrow(ax, (kv_x + kv_w, y2 + h2 / 2), (40.0, y2 + h2 / 2), lw=1.6,
          color=DECODER)
    arrow(ax, (92.0, y2 + h2 / 2), (104.0, y2 + h2 / 2), lw=1.6,
          color=DECODER)

    # encoder / pool -> KV wrap (routed through the clear corridor)
    route(ax, [(xs[5] + w1 / 2, y1), (xs[5] + w1 / 2, 23.4),
               (kv_x + kv_w / 2, 23.4), (kv_x + kv_w / 2, kv_top + 0.35)],
          color=POOL, lw=1.4, ms=9)
    ax.text(70.0, 24.6,
            "spatial pool to fixed size latent  $\\to$  decoder KV",
            ha="center", va="center", fontsize=5.8, color=POOL,
            style="italic")


# --------------------------------------------------------------------------- #
# Panel B -- token layout, ChannelFusion detail, masking, legends
# --------------------------------------------------------------------------- #
def draw_token_grid(ax):
    cell, pitch = 4.6, 5.3
    y_top = 56.0  # y of the bottom edge of the top channel row

    def sample(x0, n_ch, label, missing=None, group_col=None):
        rows = []
        for c in range(n_ch):
            yy = y_top - c * pitch
            rows.append(yy)
            for n in range(4):
                xx = x0 + n * pitch
                is_missing = missing == (c, n)
                rrect(ax, xx, yy, cell, cell,
                      edge=(LOSS if is_missing else FUSED),
                      face=("#FFFFFF" if is_missing else tint(FUSED, 0.80)),
                      lw=1.1, rounding=0.35, z=3, dashed=is_missing,
                      crossed=not is_missing)
                if is_missing:
                    ax.text(xx + cell / 2, yy + cell / 2, "omitted",
                            ha="center", va="center", fontsize=5.3,
                            color=LOSS, style="italic", zorder=5)
            ax.text(x0 - 1.1, yy + cell / 2, f"$c_{c}$", ha="right",
                    va="center", fontsize=7.0, color=MUTED)
        if group_col is not None:
            gx = x0 + group_col * pitch
            rrect(ax, gx - 0.35, rows[-1] - 0.35, cell + 0.7,
                  (rows[0] - rows[-1]) + cell + 0.7, edge=FUSION,
                  face="none", lw=2.0, rounding=0.5, z=4)
        ax.text(x0 + 1.5 * pitch, 38.3, label, ha="center", va="center",
                fontsize=7.4, color=INK, fontweight="bold")

    for n in range(4):
        ax.text(6.0 + n * pitch + cell / 2, y_top + cell + 1.5, f"$n_{n}$",
                ha="center", va="center", fontsize=7.4, color=MUTED)

    sample(6.0, 4, "sample A  ($C_b{=}4$)", group_col=1)
    sample(34.0, 3, "sample B  ($C_b{=}3$)", missing=(1, 2), group_col=2)

    # leader arrow lands in the clear gap between the n_1 and n_2 columns,
    # never on a column label.
    gap_x = 6.0 + 1.5 * pitch + cell / 2
    ax.text(gap_x, 64.0, "attention group $(b,n)$", ha="center", va="center",
            fontsize=7.2, fontweight="bold", color=FUSION)
    arrow(ax, (gap_x, 63.2), (gap_x, y_top + cell + 0.6), color=FUSION,
          lw=1.2, ms=9)
    ax.text(34.0 + 2.5 * pitch, 64.0, "same rule, other sample",
            ha="center", va="center", fontsize=7.2, color=FUSION)
    ax.text(30.0, 34.9,
            "each row $=$ a channel's $N_b$ patches; $V$ visible tokens "
            "in all.\na masked token is an absent (dashed) row, never a "
            "zeroed one.",
            ha="center", va="center", fontsize=5.6, color=MUTED,
            style="italic", linespacing=1.4)


def draw_fusion_detail(ax):
    x, y, w, h = 64.0, 38.0, 74.0, 26.0
    rrect(ax, x, y, w, h, edge=FUSION, face=tint(FUSION, 0.92), lw=1.5,
          rounding=0.7, z=2)
    ax.text(x + w / 2, y + h - 2.1, "ChannelFusion (default)", ha="center",
            va="center", fontsize=9.0, fontweight="bold", color=INK, zorder=5)
    ax.text(x + w / 2, y + h - 4.4,
            "single-query PMA over the channel axis at each present $(b,n)$",
            ha="center", va="center", fontsize=6.3, color=MUTED, zorder=5)

    lines = [
        ("$q=W_Q(\\mathrm{norm}_q(query))$; $k,v$ from "
         "$\\mathrm{norm}_{kv}(tokens)$", FUSION),
        ("$\\alpha_{c,h}=\\mathrm{softmax}_c\\!\\left("
         "q_h k_{c,h}^{\\top}/\\sqrt{D_h}+b_{c,h}\\right)$"
         "   identity bias", FUSION),
        ("$y_j=\\mathrm{proj\\_out}\\!\\left(\\sum_c \\alpha_{c,h}"
         " v_{c,h}\\right)+\\mathrm{identity\\_skip}\\!\\left("
         "\\sum_c \\bar{w}_c z_c\\right)$", FUSION),
        ("$\\to y_j\\in[K_f,D]$: one row per present location; "
         "an absent $(b,n)$ owns no row", FUSED),
        ("no positional table; $z_c$ is a routing bias $+$ output skip, "
         "never added to tokens", MUTED),
        ("$O(C\\cdot D\\cdot D_{attn})+O(D\\cdot D_{attn})$,  "
         "$D_{attn}=3D$", MUTED),
    ]
    yy = y + h - 6.8
    for text, color in lines:
        ax.text(x + 1.7, yy, text, ha="left", va="center", fontsize=6.2,
                color=color, zorder=5)
        yy -= 3.2


def draw_dcp(ax):
    x, y, w, h = 64.0, 20.0, 74.0, 15.0
    rrect(ax, x, y, w, h, edge=LOSS, face=tint(LOSS, 0.93), lw=1.5,
          rounding=0.7, z=2)
    ax.text(x + w / 2, y + h - 1.8, "DCP masking $=$ token omission",
            ha="center", va="center", fontsize=9.0, fontweight="bold",
            color=INK, zorder=5)
    bullets = [
        "Eq. 1 region: patch, channel, or combined",
        "patch: fixed $\\lfloor N\\cdot r_p\\rfloor$ per channel",
        "channel: unified HCS $m\\sim U\\{1..C\\}$, then subset",
        "masked channel-patch $=$ dropped row ($K_t$ target); never a "
        "zero/pad row",
    ]
    for i, b in enumerate(bullets):
        yy = y + h - 4.4 - i * 2.7
        ax.text(x + 1.7, yy, "$\\bullet$", ha="left", va="center",
                fontsize=7.0, color=LOSS, zorder=5)
        ax.text(x + 3.0, yy, b, ha="left", va="center", fontsize=6.2,
                color=INK, zorder=5)


def draw_ragged_and_ghost(ax):
    # the solid ragged token list -- the primary object
    x, w, h, pitch = 4.0, 10.0, 1.8, 2.0
    specs = [
        (CHANNEL_COLORS[0], False),
        (LOSS, True),   # masked -> absent (dashed) row, never zeroed
        (CHANNEL_COLORS[1], False),
        (CHANNEL_COLORS[2], False),
        (CHANNEL_COLORS[3], False),
    ]
    top = token_rows(ax, x, 21.5, w, h, pitch, specs)
    ax.text(x + w / 2, top + 1.4, "$[\\Sigma C\\cdot N,\\ D]$", ha="center",
            va="center", fontsize=6.4, fontweight="bold", color=INK)
    ax.text(x + w / 2, top + 2.9, "solid ragged list", ha="center",
            va="center", fontsize=5.7, color=FUSED)
    ax.text(x + w / 2, top + 4.2, "the only token container", ha="center",
            va="center", fontsize=5.4, color=MUTED, style="italic")

    # the padded tensor that is never materialised -- dashed ghost only
    padded_ghost(ax, 22.0, 21.5, 34.0, 8.5,
                 shape="$[B,\\,C_{\\max},\\,N,\\,D]$",
                 note="never materialised")


def draw_legends(ax):
    legend(
        ax, 3.0, 17.0,
        [("tokenizer: patch embed $+$ $z_c$", TOKENIZER, "fill"),
         ("channel fusion", FUSION, "fill"),
         ("encoder (TransformerEncoder)", ENCODER, "fill"),
         ("decoder $+$ head", DECODER, "fill"),
         ("fused $[K_f,D]$", FUSED, "cross"),
         ("latent pool", POOL, "cross"),
         ("masked row (omitted)", LOSS, "omit"),
         ("padded ghost (never built)", GHOST, "ghost")],
        title="module roles / data", pitch=1.6, sw=1.2, fontsize=5.8,
        title_size=6.6, label_gap=0.9,
    )

    ax.text(46.0, 17.8, "symbols / glossary", ha="left", va="center",
            fontsize=6.6, fontweight="bold", color=INK)
    cols = [
        (46.0, [
            ("$B$", "batch size"),
            ("$C_b$", "channels of sample $b$ ($C$ generic)"),
            ("$C_{\\max}$", "$\\max_b C_b$"),
            ("$\\Sigma C$", "channels across batch"),
            ("$N_b$", "patches/channel, sample $b$"),
            ("$N$", "$\\max_b N_b$ (fused grid)"),
            ("$H,W$", "image side (pixels)"),
        ]),
        (70.0, [
            ("$P$", "canonical patch (kernel)"),
            ("$p$", "runtime patch (pixels)"),
            ("$p_b$", "runtime patch, sample $b$"),
            ("$\\mu$", "pixel size (µm/px)"),
            ("$D$", "token / embedding dim"),
            ("$V$", "visible tokens"),
            ("$K_f$", "present $(b,n)$ locations"),
        ]),
        (94.0, [
            ("$K_t$", "decoder targets (masked)"),
            ("$M$", "memory tokens (memory arm)"),
            ("$L$", "latents / sample"),
            ("$n_q$", "PMA query count"),
            ("$z_c$", "channel identity vector"),
            ("$\\bar{w}_c$", "head-mean attn weight"),
            ("$\\alpha_{c,h}$", "channel attn weight"),
        ]),
        (114.0, [
            ("$y_j$", "fused row (per location)"),
            ("PMA", "attention pooling"),
            ("RoPE", "2-D axial rotary PE"),
            ("DCP", "row-omitting mask sampler"),
            ("$r_p$", "DCP patch-mask ratio"),
            ("ghost", "never-built padded tensor"),
            ("$\\Sigma p^2$", "flat head output"),
        ]),
    ]
    for x0, entries in cols:
        yy = 15.7
        for sym, meaning in entries:
            ax.text(x0, yy, sym, ha="left", va="center", fontsize=5.4,
                    color=INK, fontweight="bold")
            ax.text(x0 + 5.0, yy, meaning, ha="left", va="center",
                    fontsize=5.2, color=MUTED)
            yy -= 1.9


def draw_token_panel(ax):
    ax.set_xlim(0, 140)
    ax.set_ylim(0, 68)
    ax.axis("off")
    panel_header(ax, 1.0, 66.0, "B",
                 "Set-native token layout, ChannelFusion, masking, symbols")
    draw_token_grid(ax)
    draw_fusion_detail(ax)
    draw_dcp(ax)
    draw_ragged_and_ghost(ax)
    draw_legends(ax)


# --------------------------------------------------------------------------- #
def build() -> plt.Figure:
    fig = plt.figure(figsize=(15.0, 13.6))
    gs = fig.add_gridspec(
        2, 1, height_ratios=[56, 68], hspace=0.03,
        left=0.005, right=0.995, top=0.995, bottom=0.005,
    )
    ax_a = fig.add_subplot(gs[0])
    ax_a.set_aspect("equal", adjustable="box")
    draw_pipeline(ax_a)

    ax_b = fig.add_subplot(gs[1])
    ax_b.set_aspect("equal", adjustable="box")
    draw_token_panel(ax_b)
    return fig


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--formats", nargs="+", default=["pdf", "svg", "png"])
    ap.add_argument("--stem", default="architecture")
    args = ap.parse_args()

    install_rcparams()

    fig = build()
    save_figure(fig, OUT, args.stem, args.formats)
    plt.close(fig)


if __name__ == "__main__":
    main()
