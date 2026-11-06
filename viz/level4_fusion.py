"""L4 -- ChannelFusion: single-query PMA pooling over the channel axis.

Shows :class:`ChannelFusion`, the **default** fusion.  It is a single-query
PMA-style pooling over the channel axis at each spatial location: learned
``k_proj`` / ``v_proj`` are applied to ``norm_kv(tokens)``, a single shared
learned query goes through ``q_proj`` after ``norm_q``, and attention runs in
an attention width ``D_attn`` (3·D by default) before ``proj_out`` maps back to
``D``.  One learned query is shared by every location, and there is no
``n_queries`` axis: the fused tensor is ``[K_f, D]``, one vector per present
``(b, n)`` location.

Channel identity is a routing bias and an output skip, never added to the
content tokens: ``identity_bias: Linear(D, n_heads)`` adds a per-head bias to
the channel logits *before* the channel softmax, and after ``proj_out``,
``identity_skip: Linear(D, D)`` adds a residual of the attention-pooled
identity ``Σ_c α_c z_c``.

It adds **no positional table**: the encoder owns layout through 2-D axial
RoPE, so the fused ``[K_f, D]`` carries content only.  The module consumes the
visible tokens only (``ChannelSet``) and returns ``fused: [K_f, D]`` plus
``present: [B, N]``, which is both the index of those rows and the source of
their grid coordinates.  ``C`` lives only in the input token count
``sum C * N``.

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
    legend, channel_legend,
)

OUT = Path(__file__).resolve().parent / "out"

# Short-hand role colours (the figure is built entirely from role colours).
FUS = FUSION          # orange: every ChannelFusion operation / module box
OUT_C = FUSED         # blue: the fused output vector, one per location
TOK = TOKENIZER       # green: the data tokens (content and channel identity)


# --------------------------------------------------------------------------- #
# Local helpers
# --------------------------------------------------------------------------- #
def stage(ax, x, y, w, h, title, sub=None, edge=FUS, tcolor=INK,
          tsize=7.6, ssize=6.2, face=None, lw=1.5, rounding=0.6, z=3):
    """Rounded **operation/module** box (never crossed)."""
    rrect(ax, x, y, w, h, edge=edge, face=face if face else tint(edge, 0.84),
          lw=lw, rounding=rounding, z=z)
    cx = x + w / 2
    if sub:
        ax.text(cx, y + h * 0.66, title, ha="center", va="center",
                fontsize=tsize, color=tcolor, fontweight="bold", zorder=4)
        ax.text(cx, y + h * 0.27, sub, ha="center", va="center",
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
    # opaque white backing so titles stay legible over the crossed diagonals
    bbox = dict(boxstyle="round,pad=0.18", facecolor="white",
                edgecolor="none", alpha=1.0)
    if sub:
        ax.text(cx, y + h * 0.66, title, ha="center", va="center",
                fontsize=tsize, color=edge, fontweight="bold", zorder=5,
                bbox=bbox)
        ax.text(cx, y + h * 0.27, sub, ha="center", va="center",
                fontsize=5.7, color=MUTED, zorder=5, bbox=bbox)
    else:
        ax.text(cx, y + h / 2, title, ha="center", va="center",
                fontsize=tsize, color=edge, fontweight="bold", zorder=5,
                bbox=bbox)


def route(ax, pts, color=INK, lw=1.4, ms=8):
    """Orthogonal poly-line with an arrowhead on the final segment."""
    for a, b in zip(pts[:-2], pts[1:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=lw,
                solid_capstyle="round", zorder=6)
    arrow(ax, pts[-2], pts[-1], color=color, lw=lw, ms=ms)


# --------------------------------------------------------------------------- #
# The single ChannelFusion panel
# --------------------------------------------------------------------------- #
def draw_fusion(ax):
    """ChannelFusion: the per-location channel pool, then the collapse."""
    # dashed alignment band around the whole panel
    rrect(ax, 2.0, 32.0, 146.0, 88.0, edge=GHOST, face="none", lw=1.0,
          dashed=True, rounding=1.0, z=0)

    # ---- geometry --------------------------------------------------------- #
    CX = [10.0, 24.0, 38.0]           # content-token columns
    CW = 11.0
    CTR = [cx + CW / 2 for cx in CX]
    IX = [58.0, 72.0, 86.0]           # channel-identity columns
    IW = 11.0
    ITR = [ix + IW / 2 for ix in IX]

    # ===================================================================== #
    # 1. the two parallel inputs: content tokens + channel identity
    # ===================================================================== #
    data_cell(ax, 10.0, 114.0, 39.0, 3.4, TOK, "$[V, D]$ visible token list",
              tsize=7.2)
    data_cell(ax, 55.0, 114.0, 44.0, 3.4, TOK,
              "$z_c$ channel identity $[V, D]$", tsize=7.0)

    for i, cx in enumerate(CX):
        data_cell(ax, cx, 105.0, CW, 6.5, CHANNEL_COLORS[i], f"$x_{i+1}$",
                  sub=f"$c_{i+1}$", tsize=9.0)
    for i, cx in enumerate(IX):
        data_cell(ax, cx, 105.0, IW, 6.5, CHANNEL_COLORS[i], f"$z_{i+1}$",
                  sub="identity", tsize=9.0)


    # ===================================================================== #
    # 2. LayerNorms: norm_kv for tokens, norm_q for the query (separate)
    # ===================================================================== #
    stage(ax, 10.0, 96.0, 39.0, 5.0, "$\\mathrm{norm}_{kv}(tokens)$",
          tsize=7.4)
    stage(ax, 55.0, 96.0, 44.0, 5.0, "$\\mathrm{identity\\_bias}$",
          "$\\mathrm{Linear}(D, H):\\; b_{c,h}$", tsize=7.4, ssize=6.0)

    # ===================================================================== #
    # 3. parallel k_proj / v_proj from the normed tokens
    # ===================================================================== #
    stage(ax, 10.0, 87.0, 19.0, 5.0, "$v\\_proj$", tsize=7.4)
    stage(ax, 32.0, 87.0, 19.0, 5.0, "$k\\_proj$", tsize=7.4)

    data_cell(ax, 10.0, 76.5, 19.0, 6.5, TOK, "$v_c$",
              sub="$[V, D_{attn}]$", tsize=8.6)
    data_cell(ax, 32.0, 76.5, 19.0, 6.5, TOK, "$k_c$",
              sub="$[V, D_{attn}]$", tsize=8.6)

    # ===================================================================== #
    # 4. logits + identity bias, then the channel softmax
    # ===================================================================== #
    stage(ax, 55.0, 64.5, 44.0, 7.5,
          "$\\mathrm{logits}_{c,h}=s_{c,h}+b_{c,h}$",
          "$\\alpha_{c,h}=\\mathrm{softmax}_c(s_{c,h}+b_{c,h})$  over $C$, "
          "per head", tsize=7.8, ssize=6.0)
    # the shared learned query (a parameter: plain box)
    stage(ax, 63.0, 73.5, 12.0, 6.5, "$q$",
          "$W_Q(\\mathrm{norm}_q(\\theta))$", tsize=10.0, ssize=5.0)

    # ===================================================================== #
    # 5. the two weighted sums + the collapse to y_j
    # ===================================================================== #
    stage(ax, 10.0, 55.5, 26.0, 7.5, "$\\sum_c \\alpha_{c,h}\\, v_{c,h}$",
          "per head $h$", tsize=8.4, ssize=6.0)
    stage(ax, 42.0, 55.5, 30.0, 7.5, "$\\sum_c \\bar{w}_c z_c$",
          "attention-pooled identity; $\\bar{w}_c=\\mathrm{mean}_h\\,\\alpha_{c,h}$",
          tsize=7.6, ssize=5.5)

    stage(ax, 10.0, 43.0, 26.0, 7.0, "$\\mathrm{proj\\_out}$",
          "concat heads $D_{attn}\\to D$", tsize=7.4, ssize=5.7)
    stage(ax, 42.0, 43.0, 30.0, 7.0, "$\\mathrm{identity\\_skip}$",
          "$\\mathrm{Linear}(D, D)$", tsize=7.4, ssize=6.0)

    ax.text(80.0, 38.0, "$\\oplus$", ha="center", va="center",
            fontsize=12.0, color=INK, fontweight="bold", zorder=6)
    data_cell(ax, 84.0, 34.5, 22.0, 7.0, OUT_C, "$y_j$",
              sub="fused $[K_f, D]$", tsize=10.0)
    data_cell(ax, 110.0, 34.5, 20.0, 7.0, GHOST, "$present$",
              sub="$[B, N]$", tsize=8.2)
    ax.text(107.0, 43.0, "returns", ha="center", va="center", fontsize=6.2,
            color=MUTED, style="italic")

    # ===================================================================== #
    # connectors
    # ===================================================================== #
    # content: token list -> tokens -> norm_kv -> parallel projections
    for cx in CTR:
        arrow(ax, (cx, 114.0), (cx, 111.5), lw=1.0, ms=7, color=TOK)
        arrow(ax, (cx, 105.0), (cx, 101.0), lw=1.0, ms=7, color=TOK)
    arrow(ax, (19.5, 96.0), (19.5, 92.0), lw=1.2, ms=8)
    arrow(ax, (41.5, 96.0), (41.5, 92.0), lw=1.2, ms=8)
    arrow(ax, (19.5, 87.0), (19.5, 83.0), lw=1.2, ms=8)
    arrow(ax, (41.5, 87.0), (41.5, 83.0), lw=1.2, ms=8)
    # v_c straight down into the weighted content sum
    arrow(ax, (19.5, 76.5), (19.5, 63.0), lw=1.4, ms=8, color=TOK)
    # k_c around the left into the logits box
    route(ax, [(51.0, 79.75), (60.0, 79.75), (60.0, 72.0)], color=TOK, ms=8)

    # identity: z_c -> identity_bias -> b -> logits
    for cx in ITR:
        arrow(ax, (cx, 114.0), (cx, 111.5), lw=1.0, ms=7, color=TOK)
        arrow(ax, (cx, 105.0), (cx, 101.0), lw=1.0, ms=7, color=TOK)
    arrow(ax, (77.0, 96.0), (77.0, 92.5), lw=1.2, ms=8, color=TOK)
    arrow(ax, (77.0, 92.5), (77.0, 72.0), lw=1.4, ms=8, color=TOK)
    # z_c also runs to the identity-pooled sum
    route(ax, [(97.0, 108.25), (101.0, 108.25), (101.0, 59.25),
               (72.0, 59.25)], color=TOK, ms=8)

    # the shared query feeds the logits box
    arrow(ax, (69.0, 73.5), (69.0, 72.0), lw=1.3, ms=8, color=FUS)

    # alpha fans out to both weighted sums
    route(ax, [(55.0, 68.5), (23.0, 68.5), (23.0, 63.0)], color=FUS, ms=8)
    arrow(ax, (60.0, 64.5), (60.0, 63.0), lw=1.4, ms=8, color=FUS)

    # collapse: content sum -> proj_out -> (+)
    arrow(ax, (23.0, 55.5), (23.0, 50.0), lw=1.4, ms=8, color=FUS)
    route(ax, [(23.0, 43.0), (23.0, 38.0), (78.0, 38.0)], color=FUS, ms=8)
    # identity sum -> identity_skip -> (+)
    arrow(ax, (57.0, 55.5), (57.0, 50.0), lw=1.4, ms=8, color=FUS)
    route(ax, [(72.0, 46.5), (80.0, 46.5), (80.0, 39.2)], color=FUS, ms=8)
    arrow(ax, (80.0, 37.0), (84.0, 38.0), lw=1.6, ms=9, color=OUT_C)

    # ---- compact side-note box ------------------------------------------- #
    rrect(ax, 103.0, 91.5, 43.0, 24.0, edge=GHOST, face=tint(GHOST, 0.965),
          lw=1.1, rounding=0.6, z=1)
    ax.text(105.5, 113.2, "notes", ha="left", va="center", fontsize=6.8,
            fontweight="bold", color=INK, zorder=5)
    note_x, note_y, note_pitch = 105.5, 109.7, 2.55
    notes = [
        (r"$V \leq \Sigma_b C_b{\cdot}N_b$ rows; no padding", MUTED,
         "italic"),
        ("one row per present location; absent locations own no row", MUTED,
         "italic"),
        (r"cost $O(C{\cdot}D{\cdot}D_{attn}) + O(D{\cdot}D_{attn})$ per "
         "location, plus the small identity bias/skip", MUTED, "italic"),
        ("no positional table; encoder owns layout via 2-D axial RoPE", MUTED,
         "italic"),
        (r"$q$ = one shared learned query; $c$ = channel token index",
         FUSION, "normal"),
        (r"$present\,[B,N]$ = 2nd return, grid coords", GHOST, "normal"),
        (r"$z_c$ = channel identity, never added to tokens", TOKENIZER,
         "normal"),
    ]
    for j, (txt, col, sty) in enumerate(notes):
        ax.text(note_x, note_y - j * note_pitch, txt, ha="left", va="center",
                fontsize=5.9, color=col, style=sty, zorder=5)


# --------------------------------------------------------------------------- #
# Bottom legends + dimension key
# --------------------------------------------------------------------------- #
def draw_legends(ax):
    # ---- legend box: roles, channels and the geometry convention ---------- #
    rrect(ax, 2.0, 3.0, 72.0, 26.0, edge=INK, face=tint(INK, 0.975),
          lw=1.2, rounding=0.6, z=1)

    legend(ax, 5.0, 27.0,
           [("channel token / value", TOKENIZER, "cross"),
            ("channel identity $z_c$ (never in tokens)", TOKENIZER, "cross"),
            ("channel fusion op", FUSION, "fill"),
            ("fused output $y_j$", FUSED, "cross"),
            ("crossed = data tensor; plain = operation", INK, "line")],
           title="module roles / data", pitch=2.4, sw=1.3, fontsize=6.0,
           title_size=6.8)
    channel_legend(ax, 46.0, 27.0, n=3,
                   labels=["channel id 0", "channel id 1", "channel id 2"],
                   title="channel identity", pitch=2.4, sw=1.3, fontsize=6.0,
                   title_size=6.8)

    ax.text(5.0, 12.0,
            "channel colours reuse the module palette: channel id 0 is blue\n"
            "(= fused output) and channel id 1 is green (= tokenizer role).\n"
            "Read the box labels / subscripts, not the hue alone.",
            ha="left", va="top", fontsize=5.9, color=MUTED, style="italic",
            linespacing=1.5, zorder=5)

    # ---- dimension key / glossary ---------------------------------------- #
    rrect(ax, 78.0, 3.0, 70.0, 26.0, edge=INK, face=tint(INK, 0.985),
          lw=1.2, rounding=0.6, z=1)
    ax.text(80.0, 27.0, "dimension key / glossary", ha="left", va="center",
            fontsize=6.8, fontweight="bold", color=INK)

    left = [
        "$B$ \u2014 batch size",
        "$N$ \u2014 patches / channel",
        "$N_b=\\lfloor H_{\\mathrm{img}}/p\\rfloor^2$ \u2014 sample $b$",
        "$P$ \u2014 canonical patch size (kernel)",
        "$p$ \u2014 runtime patch size (pixels)",
        "$\\Sigma C$ \u2014 channels across batch",
        "$C$ \u2014 present channels (order-free)",
        "$V$ \u2014 visible tokens, $V\\leq\\Sigma_b C_b{\\cdot}N_b$",
        "$present[B,N]$ \u2014 2nd return (grid coords)",
        "$\\oplus$ \u2014 element-wise addition",
        "$\\theta$ \u2014 shared learned query vector",
        "RoPE \u2014 encoder 2-D axial PE",
    ]
    right = [
        "$K_f$ \u2014 present $(b,n)$ locations",
        "$D$ \u2014 token / embedding dim",
        "$D_{attn}$ \u2014 Q/K/V width ($3D$)",
        "$H$ \u2014 attention heads; $H_{\\mathrm{img}}$ \u2014 image side",
        "$D_h=D_{attn}/H$ \u2014 per-head dim",
        "$z_c$ \u2014 channel identity (never in tokens)",
        "$b_{c,h}$ \u2014 identity bias (per head)",
        "$\\bar{w}_c$ \u2014 mean-over-heads identity weight",
        "$s_{c,h}=q_h\\!\\cdot\\!k_{c,h}/\\sqrt{D_h}$ \u2014 logit",
        "$\\alpha_{c,h}$ \u2014 channel softmax weights",
        "$c$ \u2014 channel token index",
    ]
    yy = 25.0
    for e in left:
        ax.text(80.0, yy, e, ha="left", va="center", fontsize=5.7, color=INK)
        yy -= 1.9
    yy = 25.0
    for e in right:
        ax.text(113.0, yy, e, ha="left", va="center", fontsize=5.7, color=INK)
        yy -= 1.9


# --------------------------------------------------------------------------- #
# Main figure assembly
# --------------------------------------------------------------------------- #
def build() -> plt.Figure:
    fig = plt.figure(figsize=(15.0, 13.0))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_xlim(0, 150)
    ax.set_ylim(0, 130)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    # title + subtitle
    ax.text(2, 126.5, "L4", ha="left", va="center", fontsize=15,
            fontweight="bold", color=INK)
    ax.text(6.5, 126.5, "ChannelFusion: single-query channel pooling",
            ha="left", va="center", fontsize=12.2, fontweight="bold",
            color=INK)
    ax.text(2.0, 122.5,
            "one shared learned query per $(b,n)$; learned Q/K/V to "
            "$D_{attn}=3D$; identity routing bias + skip; no padding, no "
            "positional table.", ha="left", va="center", fontsize=7.0,
            color=MUTED, style="italic")

    draw_fusion(ax)
    draw_legends(ax)

    # footnote
    ax.text(2.0, 0.7,
            "ChannelFusion consumes visible tokens and produces $[K_f, D]$ "
            "fused embeddings (no positional table); $C$ lives only in the "
            "input token count $\\Sigma C\\cdot N$; the padded "
            "$[B, C_{\\max}, N, D]$ tensor is never materialised.",
            ha="left", va="bottom", fontsize=6.6, color=MUTED,
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
