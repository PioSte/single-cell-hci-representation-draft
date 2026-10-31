"""L4 -- ChannelFusion: single-query PMA pooling over the channel axis.

Shows :class:`ChannelFusion`, the **default** fusion.  It is a single-query
PMA-style pooling over the channel axis at each spatial location.  It follows
Panopticon's recipe: LayerNormed tokens pass through learned ``q_proj`` /
``k_proj`` / ``v_proj`` into an attention width ``D_attn`` (3·D by default),
attention runs there, and ``proj_out`` maps back to ``D``.  One learned query
is shared by every location, and there is no ``n_queries`` axis: the fused
tensor is ``[K_f, D]``, one vector per present ``(b, n)`` location.

It adds **no positional table**: the encoder owns layout through 2-D axial
RoPE, so the fused ``[K_f, D]`` carries content only.  The module consumes the
visible tokens only (``ChannelSet``) and returns ``fused: [K_f, D]`` plus
``present: [B, N]``, which is both the index of those rows and the source of
their grid coordinates.
``C`` lives only in the input token count ``sum C * N``.

Usage::

    uv run --extra viz python viz/level4_fusion.py
    uv run --extra viz python viz/level4_fusion.py --formats pdf svg
    uv run --extra viz python viz/level4_fusion.py --stem level4_fusion
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from figure_style import (  # noqa: E402
    INK, MUTED, CHANNEL_COLORS, TOKENIZER, FUSION, FUSED, GHOST,
    tint, rrect, arrow, install_rcparams, save_figure,
    legend, channel_legend, edge_label,
)

OUT = Path(__file__).resolve().parent / "out"

# Short-hand role colours (the figure is built entirely from role colours).
FUS = FUSION          # orange: every ChannelFusion operation / module box
OUT_C = FUSED         # blue: the fused output vector, one per location
TOK = TOKENIZER       # green: the data tokens (flat input, keys/values)


# --------------------------------------------------------------------------- #
# Local helpers
# --------------------------------------------------------------------------- #
def stage(ax, x, y, w, h, title, sub=None, edge=FUS, tcolor=INK,
          tsize=7.6, ssize=6.2, face=None, lw=1.5, rounding=0.6, z=3):
    """Rounded **operation/module** box (never crossed).

    Title at 63 % height, muted sub-line at 27 %.  Operation boxes stay plain:
    only data tensors (tokens, embeddings, arrays) carry the cross.
    """
    rrect(ax, x, y, w, h, edge=edge, face=face if face else tint(edge, 0.84),
          lw=lw, rounding=rounding, z=z)
    cx = x + w / 2
    if sub:
        ax.text(cx, y + h * 0.64, title, ha="center", va="center",
                fontsize=tsize, color=tcolor, fontweight="bold", zorder=4)
        ax.text(cx, y + h * 0.25, sub, ha="center", va="center",
                fontsize=ssize, color=MUTED, zorder=4)
    else:
        ax.text(cx, y + h / 2, title, ha="center", va="center",
                fontsize=tsize, color=tcolor, fontweight="bold", zorder=4)
    return x + w / 2


def data_cell(ax, x, y, w, h, edge, title, sub=None, tsize=8.4,
              rounding=0.6, lw=1.5, z=3):
    """Crossed **data** cell (token / embedding / array), optional sub-line."""
    rrect(ax, x, y, w, h, edge=edge, face=tint(edge, 0.82), lw=lw,
          rounding=rounding, z=z, crossed=True)
    cx = x + w / 2
    bbox = dict(boxstyle="round,pad=0.12", facecolor="white",
                edgecolor="none", alpha=0.82)
    if sub:
        ax.text(cx, y + h * 0.64, title, ha="center", va="center",
                fontsize=tsize, color=edge, fontweight="bold", zorder=5,
                bbox=bbox)
        ax.text(cx, y + h * 0.26, sub, ha="center", va="center",
                fontsize=5.8, color=MUTED, zorder=5, bbox=bbox)
    else:
        ax.text(cx, y + h / 2, title, ha="center", va="center",
                fontsize=tsize, color=edge, fontweight="bold", zorder=5,
                bbox=bbox)


# --------------------------------------------------------------------------- #
# The single ChannelFusion panel
# --------------------------------------------------------------------------- #
def draw_fusion(ax):
    """ChannelFusion: the per-location channel pool, then the collapse."""
    # dashed alignment band around the whole panel
    rrect(ax, 2.0, 28.0, 146.0, 52.0, edge=GHOST, face="none", lw=1.0,
          dashed=True, rounding=1.0, z=0)

    # ---- geometry --------------------------------------------------------- #
    GX0, GX1 = 16.0, 58.0             # channel-group extent
    CX = [16.0, 31.0, 46.0]           # left edge of each channel column
    CW = 12.0
    CTR = [cx + CW / 2 for cx in CX]  # column centres
    GC = (GX0 + GX1) / 2

    # ---- token list: the solid primary object ----------------------------- #
    rrect(ax, GX0, 76.6, GX1 - GX0, 3.0, edge=TOK, face=tint(TOK, 0.86),
          lw=1.5, rounding=0.5, crossed=True)
    ax.text(GC, 78.1, "$[V, D]$ token list", ha="center", va="center",
            fontsize=7.0, color=INK, fontweight="bold", zorder=5,
            bbox=dict(boxstyle="round,pad=0.14", facecolor="white",
                      edgecolor="none", alpha=0.85))
    ax.text(60.0, 78.1, "$V \\leq \\Sigma C\\cdot N$ visible rows",
            ha="left", va="center", fontsize=6.0, color=MUTED,
            style="italic")

    # ---- stage 1: channel tokens at this (b, n) location ------------------ #
    y_tok, h_tok = 69.2, 6.0
    for i, cx in enumerate(CX):
        col = CHANNEL_COLORS[i]
        data_cell(ax, cx, y_tok, CW, h_tok, col, f"$x_{i+1}$",
                  sub=f"$c_{i+1}$", tsize=9.0)
    ax.text(60.0, 74.8, "$C$ present channels, order-free $\\cdot$ one "
            "$(b,n)$", ha="left", va="center", fontsize=6.2, color=INK)

    # ---- stage 2: LayerNorm -> learned Q/K/V projections ------------------ #
    y_ln, h_ln = 62.0, 3.0
    stage(ax, GX0, y_ln, GX1 - GX0, h_ln,
          "LayerNorm $\\mathrm{norm}_{kv}$ $\\to$ Q/K/V projections",
          "$W_Q, W_K, W_V: D \\to D_{attn}$ (here $D_{attn}=3D$)",
          tcolor=FUS, tsize=6.6)

    y_kv, h_kv = 53.6, 6.0
    for i, cx in enumerate(CX):
        col = CHANNEL_COLORS[i]
        data_cell(ax, cx, y_kv, CW, h_kv, col, f"$k_{i+1}$",
                  sub="from $W_K$", tsize=8.4)

    # ---- stage 3: scores q_h . k_c,h / sqrt(D_h) -------------------------- #
    y_s, h_s = 44.0, 6.0
    stage(ax, GX0, y_s, GX1 - GX0, h_s,
          "$s_{c,h} = q_h \\cdot k_{c,h} / \\sqrt{D_h}$",
          "$H$ heads  $\\cdot$  $D_h = D_{attn}/H$", tcolor=FUS, tsize=8.0)

    # ---- stage 4: softmax over the channel axis --------------------------- #
    y_a, h_a = 33.6, 6.0
    stage(ax, GX0, y_a, GX1 - GX0, h_a,
          "$\\alpha_{c} = \\mathrm{softmax}_c\\,(s_{c})$",
          "softmax over the channel ($C$) axis", tcolor=FUS, tsize=7.8)

    # ---- connectors: tokens -> LayerNorm -> Q/K/V -> scores -> alpha ------ #
    for cx in CTR:
        arrow(ax, (cx, 76.6), (cx, y_tok + h_tok), lw=1.0, ms=7, color=TOK)
        arrow(ax, (cx, y_tok), (cx, y_ln + h_ln), lw=1.1, ms=7)
        arrow(ax, (cx, y_ln), (cx, y_kv + h_kv), lw=1.1, ms=7)
        arrow(ax, (cx, y_kv), (cx, y_s + h_s), lw=1.1, ms=7)
        arrow(ax, (cx, y_s), (cx, y_a + h_a), lw=1.1, ms=7, color=FUS)

    # ---- the learned query q: a parameter, so its box stays plain --------- #
    qx, qw, qh = 2.5, 11.0, 6.0
    qy = y_s + (h_s - qh) / 2
    rrect(ax, qx, qy, qw, qh, edge=FUS, face=tint(FUS, 0.74), lw=1.7,
          rounding=0.6)
    ax.text(qx + qw / 2, qy + qh / 2 + 0.8, "$q$", ha="center",
            va="center", fontsize=10.0, color=FUS, fontweight="bold")
    ax.text(qx + qw / 2, qy + qh / 2 - 1.5, "$W_Q\\,\\mathrm{norm}_q$",
            ha="center", va="center", fontsize=5.2, color=MUTED,
            style="italic")
    ax.text(2.5, 41.6, "one learned query $q$", ha="left", va="center",
            fontsize=6.0, color=FUS)
    ax.text(2.5, 40.1, "shared by all $(b,n)$", ha="left", va="center",
            fontsize=5.8, color=MUTED, style="italic")
    arrow(ax, (qx + qw, qy + qh / 2), (GX0, y_s + h_s / 2), lw=1.5,
          color=FUS, ms=9)

    # ---- collapse: weighted sum -> proj_out -> y_j ------------------------ #
    y_c = y_a + h_a / 2               # centre of the alpha row (36.6)
    sx, sw, sh = 64.0, 38.0, 11.0
    sy = y_c - sh / 2
    stage(ax, sx, sy, sw, sh, "$\\sum_c \\alpha_c\\, v_c$",
          "per head $h$", tcolor=FUS, tsize=10.0, rounding=0.7)

    px, pw, ph = 108.0, 17.0, 6.0
    py = y_c - ph / 2
    stage(ax, px, py, pw, ph, "proj_out", "concat heads", tcolor=FUS,
          tsize=7.4)

    yx, yw, yh = 131.0, 16.0, 7.0
    yy = y_c - yh / 2
    data_cell(ax, yx, yy, yw, yh, OUT_C, "$y_j$", sub="fused", tsize=10.5)

    arrow(ax, (GX1, y_c), (sx, y_c), lw=1.5, color=FUS, ms=9)
    arrow(ax, (sx + sw, y_c), (px, y_c), lw=1.5, color=FUS, ms=9)
    arrow(ax, (px + pw, y_c), (yx, y_c), lw=1.5, color=OUT_C, ms=9)
    edge_label(ax, (GX1 + sx) / 2, y_c + 1.6, "$\\alpha_c$", color=FUS)

    # ---- values v_c tap into the weighted sum ----------------------------- #
    vx, vw, vh = 62.0, 18.0, 6.0
    vy = y_kv
    data_cell(ax, vx, vy, vw, vh, TOK, "$v_c$", sub="values", tsize=8.6)
    arrow(ax, (GX1, y_kv + h_kv / 2), (vx, vy + vh / 2), lw=1.3, ms=8)
    arrow(ax, (vx + vw / 2, vy), (vx + vw / 2, sy + sh), lw=1.3, ms=8)
    edge_label(ax, vx + vw / 2 + 1.6, (vy + sy + sh) / 2, "$v_c$")

    # ---- explanatory notes: fill the right half of the panel -------------- #
    n_kw = dict(ha="left", va="center", fontsize=6.0, color=MUTED)
    ax.text(60.0, 72.0, "$k_c=W_K\\,\\mathrm{norm}_{kv}(x_c)$, "
            "$v_c=W_V\\,\\mathrm{norm}_{kv}(x_c)$", **n_kw)
    ax.text(60.0, 69.5, "one learned query $q=W_Q\\,\\mathrm{norm}_q$, "
            "shared by every $(b,n)$", **n_kw)
    ax.text(60.0, 67.0, "attention over the $C$ present channels at $(b,n)$",
            **n_kw)
    ax.text(60.0, 64.5, "$O(C{\\cdot}D{\\cdot}D_{attn})$ per location; "
            "$D_{attn}=3D$ by default", **n_kw)
    ax.text(60.0, 62.0, "no positional table; encoder owns layout via 2-D "
            "axial RoPE", ha="left", va="center", fontsize=6.0, color=OUT_C)
    ax.text(84.0, 56.6, "one vector per present $(b,n)$: $[K_f, D]$",
            ha="left", va="center", fontsize=6.0,
            color=OUT_C, fontweight="bold")
    ax.text(84.0, 54.0, "$\\alpha_c$ weights each channel's value",
            **n_kw)
    ax.text(84.0, 51.5, "then $\\mathrm{proj\\_out}: D_{attn}\\to D$ "
            "merges the heads", **n_kw)


# --------------------------------------------------------------------------- #
# Bottom legends + dimension key
# --------------------------------------------------------------------------- #
def draw_legends(ax):
    # ---- legend box: roles, channels and the geometry convention ---------- #
    rrect(ax, 2.0, 3.0, 66.0, 24.0, edge=INK, face=tint(INK, 0.975),
          lw=1.2, rounding=0.6, z=1)

    legend(ax, 4.0, 25.0,
           [("channel token / value", TOKENIZER, "cross"),
            ("channel fusion op", FUSION, "fill"),
            ("fused output $y_j$", FUSED, "cross")],
           title="module roles / data", pitch=2.5, sw=1.4, fontsize=6.1,
           title_size=6.8)
    channel_legend(ax, 34.0, 25.0, n=3,
                   labels=["channel id 0", "channel id 1", "channel id 2"],
                   title="channel identity", pitch=2.5, sw=1.4, fontsize=6.1,
                   title_size=6.8)

    ax.text(34.0, 15.0,
            "channel colours reuse the module palette\n"
            "(blue $=$ fused, orange $=$ fusion): read the\n"
            "box labels / subscripts, not the hue alone.",
            ha="left", va="top", fontsize=6.0, color=MUTED, style="italic",
            linespacing=1.55, zorder=5)
    ax.text(4.0, 15.0,
            "ChannelFusion consumes visible tokens only and emits $[K_f, D]$:\n"
            "one vector per present location. $C$ lives only in the input length\n"
            "$\\Sigma C\\cdot N$; $N$ never enters a tensor.",
            ha="left", va="top", fontsize=6.0, color=MUTED, style="italic",
            linespacing=1.55, zorder=5)

    # ---- dimension key: only symbols actually drawn ----------------------- #
    rrect(ax, 70.0, 3.0, 78.0, 24.0, edge=INK, face=tint(INK, 0.985),
          lw=1.2, rounding=0.6, z=1)
    ax.text(72.0, 25.0, "dimension key", ha="left", va="center",
            fontsize=6.8, fontweight="bold", color=INK)

    left = [
        "$B$ \u2014 batch size",
        "$N$ \u2014 patches / channel, $(H/P)^2$",
        "$\\Sigma C$ \u2014 channels across batch",
        "$C$ \u2014 present channels (order-free)",
        "$V$ \u2014 visible tokens, $V\\leq\\Sigma C{\\cdot}N$",
    ]
    right = [
        "$K_f$ \u2014 present $(b,n)$ locations",
        "$D$ \u2014 token / embedding dim",
        "$D_{attn}$ \u2014 Q/K/V width ($3D$ default)",
        "$D_h=D_{attn}/H$ \u2014 per-head dim",
        "$H$ \u2014 attention heads",
        "$q$ \u2014 one learned query, shared",
    ]
    yy = 22.3
    for e in left:
        ax.text(72.0, yy, e, ha="left", va="center", fontsize=6.1, color=INK)
        yy -= 2.7
    yy = 22.3
    for e in right:
        ax.text(108.0, yy, e, ha="left", va="center", fontsize=6.1, color=INK)
        yy -= 2.7

    ax.text(72.0, 9.4,
            "cost is per $(b,n)$ location: $O(C{\\cdot}D{\\cdot}D_{attn})$;\n"
            "$k=W_K\\mathrm{norm}_{kv}(x)$, $v=W_V\\mathrm{norm}_{kv}(x)$, "
            "$q=W_Q\\mathrm{norm}_q(q)$. No positional table \u2014 the encoder "
            "owns layout via 2-D axial RoPE.",
            ha="left", va="top", fontsize=5.9, color=MUTED, style="italic",
            linespacing=1.55, zorder=5)


# --------------------------------------------------------------------------- #
# Main figure assembly
# --------------------------------------------------------------------------- #
def build() -> plt.Figure:
    fig = plt.figure(figsize=(15.0, 9.0))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_xlim(0, 150)
    ax.set_ylim(0, 90)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    # title + subtitle
    ax.text(2, 87.0, "L4", ha="left", va="center", fontsize=15,
            fontweight="bold", color=INK)
    ax.text(6.5, 87.0, "ChannelFusion: Panopticon-style channel pooling",
            ha="left", va="center", fontsize=12.2, fontweight="bold",
            color=INK)
    ax.text(2.0, 83.4,
            "one shared learned query per $(b,n)$ location; learned Q/K/V "
            "projections to $D_{attn}=3D$; no padding, no positional table.",
            ha="left", va="center", fontsize=7.0, color=MUTED,
            style="italic")

    draw_fusion(ax)
    draw_legends(ax)

    # footnote
    ax.text(2.0, 0.7,
            "ChannelFusion consumes visible tokens and produces $[K_f, D]$ "
            "fused embeddings (no positional table); $C$ lives only in the "
            "input token count $\\Sigma C\\cdot N$.",
            ha="left", va="bottom", fontsize=6.8, color=MUTED,
            style="italic")

    return fig


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--formats", nargs="+", default=["pdf", "svg", "png"])
    ap.add_argument("--stem", default="level4_fusion")
    args = ap.parse_args()

    install_rcparams()

    fig = build()
    save_figure(fig, OUT, args.stem, args.formats)
    plt.close(fig)


if __name__ == "__main__":
    main()
