"""L4 -- channel fusion internals: two interchangeable pooling strategies.

Panel A shows :class:`ChannelFusion`, the default PMA-style pooling over the
channel axis at each spatial location. Panel B shows
:class:`IdentityGatedSum`, the identity-routed alternative for
high-channel-count panels. Both are ``O(C)`` per location: the default
``ChannelFusion`` uses a single shared query per location, so it never forms a
``C x C`` attention map. A bottom inset shows the rejected ``O(C^2)`` dense
``C x C`` channel self-attention baseline (ChannelViT / ChAda-ViT), which
requires padding and an attention mask.

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
    BLUE, GREEN, ORANGE, RED, PURPLE, VIOLET, INK, MUTED, SKY,
    CHANNEL_COLORS, FRONTEND, FUSION, ENCODER, DECODER, MEMORY, LOSS, FUSED,
    GHOST, tint, rrect, arrow, panel_header, install_rcparams, save_figure,
    legend, role_legend, channel_legend, padded_ghost, edge_label,
)

OUT = Path(__file__).resolve().parent / "out"

# Short-hand role colours (the figure is built entirely from role colours).
FUS = FUSION          # orange: every ChannelFusion / Hypernetwork box
OUT_C = FUSED         # blue: the fused output vector [B, N, D]
FE = FRONTEND         # green: patch-embed tokens, additive identity z_c
REJ = LOSS            # red: the rejected O(C^2) baseline / omitted accent


# --------------------------------------------------------------------------- #
# Local helpers
# --------------------------------------------------------------------------- #
def stage(ax, x, y, w, h, title, sub=None, edge=FUS, tcolor=INK,
          tsize=7.6, ssize=6.2, face=None, lw=1.5, rounding=0.6, z=3):
    """Rounded stage box; title at 63 % height, muted sub-line at 27 %."""
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


def note(ax, x, y, lines, color=MUTED, size=6.1):
    """Small multi-line annotation, top-left anchored."""
    ax.text(x, y, lines, ha="left", va="top", fontsize=size, color=color,
            style="italic", linespacing=1.55, zorder=5)


# --------------------------------------------------------------------------- #
# Panel A: ChannelFusion (default shared-query PMA, O(C) per location)
# --------------------------------------------------------------------------- #
def draw_panel_a(ax):
    """ChannelFusion: top-to-bottom chain, then the collapse to one token."""
    # dashed alignment band -- same x-width and y-band as panel B
    rrect(ax, 2.0, 28.0, 70.0, 53.5, edge=GHOST, face="none", lw=1.0,
          dashed=True, rounding=1.0, z=0)
    panel_header(ax, 3.0, 83.6, "A",
                 "ChannelFusion (default) \u2014 $O(C)$ per location")

    # ---- geometry --------------------------------------------------------- #
    CX = [11.0, 19.0, 27.0]           # left edge of each channel column
    CW = 6.0
    CTR = [cx + CW / 2 for cx in CX]  # column centres
    GX0, GX1 = CX[0], CX[-1] + CW     # channel group extent: 11 .. 33
    GC = (GX0 + GX1) / 2

    # ---- ragged token list: the solid primary object ---------------------- #
    rrect(ax, GX0, 78.0, GX1 - GX0, 3.0, edge=FE, face=tint(FE, 0.86),
          lw=1.4, rounding=0.5)
    ax.text(GC, 79.5, "ragged $[\\Sigma C\\cdot N,\\, D]$  $V$ visible rows",
            ha="center", va="center", fontsize=6.2, color=INK,
            fontweight="bold", zorder=4)

    # ---- stage 1: channel tokens at this (b, n) location ------------------ #
    y_tok, h_tok = 71.5, 5.2
    for i, cx in enumerate(CX):
        col = CHANNEL_COLORS[i]
        rrect(ax, cx, y_tok, CW, h_tok, edge=col, face=tint(col, 0.82),
              lw=1.5, rounding=0.6)
        ax.text(cx + CW / 2, y_tok + 3.5, f"$x_{i+1}$", ha="center",
                va="center", fontsize=8.6, color=col, fontweight="bold")
        ax.text(cx + CW / 2, y_tok + 1.5, f"$m_{i+1}$", ha="center",
                va="center", fontsize=5.9, color=MUTED, style="italic")
    ax.text(35.0, 75.2, "$C$ present channels", ha="left", va="center",
            fontsize=6.2, color=INK)
    ax.text(35.0, 73.5, "order-free \u00b7 one $(b,n)$", ha="left",
            va="center", fontsize=6.0, color=MUTED, style="italic")

    # ---- stage 2: LayerNorm -> keys = values ------------------------------ #
    y_ln, h_ln = 66.0, 3.0
    stage(ax, GX0, y_ln, GX1 - GX0, h_ln, "LayerNorm $\\mathrm{norm}_{kv}$",
          edge=FUS, tcolor=FUS, tsize=7.0)

    y_kv, h_kv = 58.5, 5.5
    for i, cx in enumerate(CX):
        col = CHANNEL_COLORS[i]
        rrect(ax, cx, y_kv, CW, h_kv, edge=col, face=tint(col, 0.86),
              lw=1.5, rounding=0.6)
        ax.text(cx + CW / 2, y_kv + h_kv / 2, f"$k_{i+1}\\!=\\!v_{i+1}$",
                ha="center", va="center", fontsize=7.6, color=col,
                fontweight="bold")
    note(ax, 48.0, 64.4,
         "keys = values = $\\mathrm{norm}_{kv}(x_c)$\n"
         "no $W_Q/W_K/W_V$")
    note(ax, 50.0, 56.0,
         "attention over the $C$ present channels:\n"
         "real rows only \u2014 no padding,\n"
         "no $-\\infty$ attention mask")

    # ---- stage 3: scores q_h . k_c,h / sqrt(D_h) -------------------------- #
    y_s, h_s = 49.0, 6.0
    stage(ax, GX0, y_s, GX1 - GX0, h_s,
          "$s_{c,h} = q_h \\cdot k_{c,h} / \\sqrt{D_h}$",
          "$H$ heads  \u00b7  $D_h = D/H$", edge=FUS, tcolor=FUS, tsize=8.0)

    # ---- stage 4: softmax over the channel axis --------------------------- #
    y_a, h_a = 39.5, 6.0
    stage(ax, GX0, y_a, GX1 - GX0, h_a,
          "$\\alpha_{c} = \\mathrm{softmax}_c\\,(s_{c})$",
          "softmax over the channel ($C$) axis", edge=FUS, tcolor=FUS,
          tsize=7.8)

    # ---- connectors: tokens -> LayerNorm -> k=v -> scores -> alpha -------- #
    for cx in CTR:
        arrow(ax, (cx, 78.0), (cx, y_tok + h_tok), lw=1.0, ms=7, color=FE)
        arrow(ax, (cx, y_tok), (cx, y_ln + h_ln), lw=1.1, ms=7)
        arrow(ax, (cx, y_ln), (cx, y_kv + h_kv), lw=1.1, ms=7)
        arrow(ax, (cx, y_kv), (cx, y_s + h_s), lw=1.1, ms=7)
        arrow(ax, (cx, y_s), (cx, y_a + h_a), lw=1.1, ms=7, color=FUS)

    # ---- the learned query q_j (shared by every location) ----------------- #
    qx, qw, qh = 2.0, 7.0, 6.0
    qy = y_s + (h_s - qh) / 2
    rrect(ax, qx, qy, qw, qh, edge=FUS, face=tint(FUS, 0.74), lw=1.7,
          rounding=0.6)
    ax.text(qx + qw / 2, qy + qh / 2, "$q_j$", ha="center", va="center",
            fontsize=9.6, color=FUS, fontweight="bold")
    ax.text(2.5, 47.6, "$Q{=}1$ learned query", ha="left",
            va="center", fontsize=6.0, color=FUS)
    ax.text(2.5, 45.8, "shared by all $(b,n)$", ha="left",
            va="center", fontsize=5.8, color=MUTED, style="italic")
    arrow(ax, (qx + qw, qy + qh / 2), (GX0, y_s + h_s / 2), lw=1.5,
          color=FUS, ms=9)

    # ---- collapse: weighted sum -> proj_out -> y_j ------------------------ #
    y_c = y_a + h_a / 2               # centre of the alpha row (42.5)
    sx, sw, sh = 36.0, 12.0, 11.0
    sy = y_c - sh / 2
    stage(ax, sx, sy, sw, sh, "$\\sum_c \\alpha_c\\, v_c$", "per head $h$",
          edge=FUS, tcolor=FUS, tsize=9.4, rounding=0.7)

    px, pw, ph = 51.0, 9.0, 6.0
    py = y_c - ph / 2
    stage(ax, px, py, pw, ph, "proj_out", "concat heads", edge=FUS,
          tcolor=FUS, tsize=7.2)

    yx, yw, yh = 63.0, 7.0, 7.0
    yy = y_c - yh / 2
    rrect(ax, yx, yy, yw, yh, edge=OUT_C, face=tint(OUT_C, 0.80), lw=1.8,
          rounding=0.6)
    ax.text(yx + yw / 2, y_c + 1.4, "$y_j$", ha="center", va="center",
            fontsize=10.0, color=OUT_C, fontweight="bold")
    ax.text(yx + yw / 2, y_c - 1.9, "fused", ha="center", va="center",
            fontsize=6.0, color=MUTED)

    arrow(ax, (GX1, y_c), (sx, y_c), lw=1.5, color=FUS, ms=9)
    arrow(ax, (sx + sw, y_c), (px, y_c), lw=1.5, color=FUS, ms=9)
    arrow(ax, (px + pw, y_c), (yx, y_c), lw=1.5, color=OUT_C, ms=9)
    edge_label(ax, (GX1 + sx) / 2, y_c + 1.4, "$\\alpha_c$", color=FUS)

    ax.text(36.0, 34.4, "scatter \u2192 $[B, N, Q{\\cdot}D]$ \u2192 "
            "$[B, N, D]$", ha="left", va="center", fontsize=6.2,
            color=OUT_C, fontweight="bold")

    # ---- values v_c tap into the weighted sum ----------------------------- #
    vx, vw, vh = 36.0, 10.0, 5.5
    vy = y_kv + (h_kv - vh) / 2
    rrect(ax, vx, vy, vw, vh, edge=GHOST, face=tint(GHOST, 0.92), lw=1.3,
          rounding=0.6)
    ax.text(vx + vw / 2, vy + vh / 2 + 1.1, "$v_c$", ha="center",
            va="center", fontsize=8.2, color=INK, fontweight="bold")
    ax.text(vx + vw / 2, vy + vh / 2 - 1.2, "values", ha="center",
            va="center", fontsize=5.9, color=MUTED)
    arrow(ax, (GX1, y_kv + h_kv / 2), (vx, vy + vh / 2), lw=1.3, ms=8)
    arrow(ax, (vx + vw / 2, vy), (vx + vw / 2, sy + sh), lw=1.3, ms=8)
    edge_label(ax, vx + vw / 2 + 1.3, (vy + sy + sh) / 2, "$v_c$")


# --------------------------------------------------------------------------- #
# Panel B: IdentityGatedSum (O(C) alternative)
# --------------------------------------------------------------------------- #
def draw_panel_b(ax):
    """IdentityGatedSum: channel embedding -> weights, tokens -> values."""
    rrect(ax, 75.0, 28.0, 70.0, 53.5, edge=GHOST, face="none", lw=1.0,
          dashed=True, rounding=1.0, z=0)
    panel_header(ax, 76.0, 83.6, "B",
                 "IdentityGatedSum \u2014 $O(C)$")

    # two parallel branches: (left) identity -> weights, (right) values
    LX = [79.0, 85.0, 91.0]           # weight-branch column left edges
    RX = [114.0, 120.0, 126.0]        # value-branch column left edges
    W = 5.0
    LC = LX[1] + W / 2                # 87.5  -- left group centre
    RC = RX[1] + W / 2                # 122.5 -- right group centre

    # ---- branch inputs ---------------------------------------------------- #
    rrect(ax, LX[0], 78.0, LX[2] + W - LX[0], 3.0, edge=FE,
          face=tint(FE, 0.86), lw=1.4, rounding=0.5)
    ax.text(LC, 79.5, "channel embedding $z_c$  ($D$)", ha="center",
            va="center", fontsize=6.2, color=INK, fontweight="bold")

    rrect(ax, RX[0], 78.0, RX[2] + W - RX[0], 3.0, edge=FE,
          face=tint(FE, 0.86), lw=1.4, rounding=0.5)
    ax.text(RC, 79.5, "ragged tokens $[\\Sigma C\\cdot N, D]$", ha="center",
            va="center", fontsize=6.2, color=INK, fontweight="bold")

    y_in, h_in = 71.5, 5.2
    for i in range(3):
        col = CHANNEL_COLORS[i]
        rrect(ax, LX[i], y_in, W, h_in, edge=col, face=tint(col, 0.82),
              lw=1.4, rounding=0.5)
        ax.text(LX[i] + W / 2, y_in + h_in / 2, f"$z_{i+1}$", ha="center",
                va="center", fontsize=8.2, color=col, fontweight="bold")

    stage(ax, RX[0], y_in, RX[2] + W - RX[0], h_in, "tokens $t_c$",
          edge=GHOST, tcolor=INK, tsize=7.4, face=tint(GHOST, 0.92), lw=1.3)

    # ---- weights branch: MLP -> softmax -> w_c ---------------------------- #
    y_mlp, h_mlp = 62.0, 6.0
    stage(ax, 77.0, y_mlp, 21.0, h_mlp, "MLP weight_net",
          "Linear $\\rightarrow$ GELU $\\rightarrow$ Linear", edge=FUS,
          tcolor=FUS, tsize=7.4)

    y_pill, h_pill = 52.5, 5.0
    stage(ax, 77.0, y_pill, 21.0, h_pill,
          "$\\mathrm{softmax}_c$ over the $C$ axis", edge=FUS, tcolor=FUS,
          tsize=6.5)

    y_w, h_w = 44.5, 5.5
    for i in range(3):
        rrect(ax, LX[i], y_w, W, h_w, edge=FUS, face=tint(FUS, 0.88),
              lw=1.4, rounding=0.5)
        ax.text(LX[i] + W / 2, y_w + h_w / 2, f"$w_{i+1}$", ha="center",
                va="center", fontsize=7.8, color=FUS, fontweight="bold")

    for i in range(3):
        arrow(ax, (LX[i] + W / 2, y_in), (LX[i] + W / 2, y_mlp + h_mlp),
              lw=1.1, ms=8, color=CHANNEL_COLORS[i])
    arrow(ax, (LC, y_mlp), (LC, y_pill + h_pill), lw=1.2, ms=8, color=FUS)
    for i in range(3):
        arrow(ax, (LC, y_pill), (LX[i] + W / 2, y_w + h_w), lw=1.1, ms=8,
              color=FUS)

    # ---- values branch: value_proj + out_norm -> V_c --------------------- #
    y_vp, h_vp = 62.0, 6.0
    stage(ax, RX[0] - 4.0, y_vp, (RX[2] + W) - (RX[0] - 4.0), h_vp,
          "value_proj", "then $\\mathrm{out\\_norm}$", edge=FUS, tcolor=FUS,
          tsize=7.4)

    y_v, h_v = 44.5, 5.5
    for i in range(3):
        col = CHANNEL_COLORS[i]
        rrect(ax, RX[i], y_v, W, h_v, edge=col, face=tint(col, 0.86),
              lw=1.4, rounding=0.5)
        ax.text(RX[i] + W / 2, y_v + h_v / 2, f"$V_{i+1}$", ha="center",
                va="center", fontsize=7.8, color=col, fontweight="bold")

    arrow(ax, (RC, y_in), (RC, y_vp + h_vp), lw=1.2, ms=8)
    for i in range(3):
        arrow(ax, (RC, y_vp), (RX[i] + W / 2, y_v + h_v), lw=1.1, ms=8,
              color=CHANNEL_COLORS[i])

    # ---- merge: sum_c w_c V_c -------------------------------------------- #
    y_sum, h_sum = 33.0, 9.0
    stage(ax, 77.0, y_sum, 65.0, h_sum, "$\\sum_c w_c\\, V_c$",
          "per location $(b,n)$", edge=FUS, tcolor=FUS, tsize=9.4,
          rounding=0.7)

    arrow(ax, (LC, y_w), (LC, y_sum + h_sum), lw=1.4, color=FUS, ms=9)
    arrow(ax, (RC, y_v), (RC, y_sum + h_sum), lw=1.4, ms=9)
    edge_label(ax, LC + 1.4, y_sum + h_sum + 1.5, "$w_c$", color=FUS)
    edge_label(ax, RC + 1.4, y_sum + h_sum + 1.5, "$V_c$")

    # fused output chip (blue)
    stage(ax, 84.0, 28.6, 50.0, 3.0,
          "fused $[B, N, Q{\\cdot}D]$ $\\rightarrow$ $[B, N, D]$",
          edge=OUT_C, tcolor=OUT_C, tsize=6.6, rounding=0.5, lw=1.5)
    arrow(ax, (109.5, y_sum), (109.5, 31.6), lw=1.4, color=OUT_C, ms=9)


# --------------------------------------------------------------------------- #
# Bottom inset: the rejected dense C x C baseline (O(C^2))
# --------------------------------------------------------------------------- #
def draw_inset(ax):
    """Dense C x C channel self-attention -- the rejected O(C^2) baseline."""
    x0, x1 = 82.0, 145.0
    y0, y1 = 3.0, 27.0
    cxm = (x0 + x1) / 2

    rrect(ax, x0, y0, x1 - x0, y1 - y0, edge=REJ, face=tint(REJ, 0.965),
          lw=1.4, dashed=True, rounding=0.8, z=1)

    ax.text(cxm, y1 - 2.4,
            "Rejected baseline \u2014 dense $C\\times C$ channel "
            "self-attention",
            ha="center", va="center", fontsize=7.0, color=REJ,
            fontweight="bold", zorder=4)
    ax.text(cxm, y1 - 4.6,
            "ChannelViT / ChAda-ViT  \u00b7  quadratic in $C$",
            ha="center", va="center", fontsize=6.0, color=MUTED,
            style="italic", zorder=4)

    # row 1: tokens -> W_Q / W_K / W_V -> C x C attention
    rrect(ax, 85.0, 15.0, 18.0, 5.0, edge=GHOST, face="white", lw=1.1,
          rounding=0.4)
    ax.text(94.0, 18.2, "tokens $X$", ha="center", va="center",
            fontsize=6.8, color=INK, fontweight="bold")
    ax.text(94.0, 16.3, "$[\\Sigma C\\cdot N, D]$", ha="center", va="center",
            fontsize=5.9, color=MUTED)

    WX = [106.5, 111.0, 115.5]
    labels = ["$W_Q$", "$W_K$", "$W_V$"]
    for i, wx in enumerate(WX):
        rrect(ax, wx, 15.5, 4.2, 4.0, edge=REJ, face=tint(REJ, 0.9),
              lw=1.2, dashed=True, rounding=0.3)
        ax.text(wx + 2.1, 17.5, labels[i], ha="center", va="center",
                fontsize=6.4, color=REJ, fontweight="bold")
    arrow(ax, (103.0, 17.5), (106.5, 17.5), lw=1.0, ms=7, color=REJ,
          dashed=True)
    arrow(ax, (WX[-1] + 4.2, 17.5), (122.0, 17.5), lw=1.0, ms=7, color=REJ,
          dashed=True)

    rrect(ax, 122.0, 14.0, 20.0, 7.0, edge=REJ, face=tint(REJ, 0.93),
          lw=1.3, dashed=True, rounding=0.5)
    ax.text(132.0, 18.7, "$\\mathrm{softmax}(QK^{\\top}/\\sqrt{d_h})\\,V$",
            ha="center", va="center", fontsize=6.0, color=REJ,
            fontweight="bold")
    ax.text(132.0, 16.1, "$C\\times C$ scores \u00b7 needs mask",
            ha="center", va="center", fontsize=5.6, color=MUTED)

    # row 2: the padded ghost this baseline would need
    padded_ghost(ax, 85.0, 5.0, 30.0, 8.0, z=2)
    ax.text(117.0, 10.6, "$O(C^2)$ per location",
            ha="left", va="center", fontsize=7.0, color=REJ,
            fontweight="bold")
    ax.text(117.0, 8.2, "padding + attention mask required,",
            ha="left", va="center", fontsize=6.0, color=MUTED)
    ax.text(117.0, 6.2, "so it is rejected \u2014 only the "
            "$O(C)$ paths above ship.", ha="left", va="center",
            fontsize=6.0, color=MUTED, style="italic")


# --------------------------------------------------------------------------- #
# Bottom legends + dimension key
# --------------------------------------------------------------------------- #
def draw_legends(ax):
    # ---- legend box ------------------------------------------------------- #
    rrect(ax, 2.0, 3.0, 38.0, 24.0, edge=INK, face=tint(INK, 0.975),
          lw=1.2, rounding=0.6, z=1)

    role_legend(ax, 4.0, 25.0,
                [("frontend / $z_c$", FRONTEND),
                 ("channel fusion", FUSION),
                 ("encoder", ENCODER),
                 ("decoder", DECODER),
                 ("fused output", FUSED),
                 ("loss / rejected", LOSS)],
                title="module roles", pitch=2.5, sw=1.4, fontsize=6.1,
                title_size=6.8)
    channel_legend(ax, 24.0, 25.0, n=3, title="channel identity",
                   pitch=2.5, sw=1.4, fontsize=6.1, title_size=6.8)

    # ---- dimension key ---------------------------------------------------- #
    rrect(ax, 42.0, 3.0, 38.0, 24.0, edge=INK, face=tint(INK, 0.985),
          lw=1.2, rounding=0.6, z=1)
    ax.text(44.0, 25.0, "dimension key", ha="left", va="center",
            fontsize=6.8, fontweight="bold", color=INK)

    entries = [
        "$D$ \u2014 token / embedding dim",
        "$D_h = D/H$ \u2014 per-head dim",
        "$H$ \u2014 attention heads",
        "$Q$ \u2014 queries per location (default 1)",
        "$V$ \u2014 visible channel-patch tokens",
        "$C$ \u2014 present channels (order-free)",
    ]
    yy = 22.2
    for e in entries:
        ax.text(44.0, yy, e, ha="left", va="center", fontsize=6.1, color=INK)
        yy -= 2.7

    note(ax, 44.0, 6.4,
         "keys = values = $\\mathrm{norm}_{kv}(\\mathrm{tokens})$,\n"
         "there is no $W_Q/W_K/W_V$ in either fusion.")


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

    # title
    ax.text(2, 87.0, "L4", ha="left", va="center", fontsize=15,
            fontweight="bold", color=INK)
    ax.text(6.5, 87.0, "Channel fusion: PMA pooling vs identity-gated routing",
            ha="left", va="center", fontsize=12.2, fontweight="bold",
            color=INK)

    # panels A and B: equal width, shared top/bottom band
    draw_panel_a(ax)

    # vertical separator
    ax.plot([73.5, 73.5], [29.0, 82.0], color=MUTED, lw=1.0,
            linestyle="--", zorder=1)

    draw_panel_b(ax)

    # bottom strip: legends + dimension key, and the rejected-baseline inset
    draw_legends(ax)
    draw_inset(ax)

    # footnote
    ax.text(2.0, 0.7,
            "Both shipped variants are padding-free and produce "
            "$[B, N, D]$ fused embeddings; $C$ lives only in the ragged input "
            "length $\\Sigma C\\cdot N$. The dashed $[B, C_{\\max}, N, D]$ "
            "tensor is never materialised.",
            ha="left", va="bottom", fontsize=6.8, color=MUTED, style="italic")

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
