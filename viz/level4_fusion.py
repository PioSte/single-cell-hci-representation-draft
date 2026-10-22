"""L4 -- channel fusion internals: two interchangeable pooling strategies.

Panel A shows :class:`ChannelFusion`, the default PMA-style pooling over the
channel axis at each spatial location. Panel B shows :class:`HypernetworkWeightedSum`,
the marker-identity-routed alternative for high-channel-count panels. Both are
O(C) per location: the default `ChannelFusion` uses a single fixed query per
location, so it never forms a C x C attention map. A small inset contrasts with
dense C x C channel self-attention (the ChannelViT / ChAda-ViT baseline), which
is O(C^2), and with standard fused QKV self-attention.

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
    BLUE, GREEN, ORANGE, RED, PURPLE, VIOLET, INK, MUTED, CHANNEL_COLORS,
    tint, rrect, arrow, panel_header, install_rcparams, save_figure,
)

OUT = Path(__file__).resolve().parent / "out"


# --------------------------------------------------------------------------- #
# Local helper: module / stage box with an optional muted sub-line
# --------------------------------------------------------------------------- #
def stage(ax, x, y, w, h, title, sub=None, edge=BLUE, tcolor=INK,
          tsize=7.6, ssize=6.2, face=None, lw=1.5, rounding=0.6):
    """Rounded stage box; title at 62 % height, muted sub-line at 27 %."""
    rrect(ax, x, y, w, h, edge=edge, face=face if face else tint(edge, 0.84),
          lw=lw, rounding=rounding)
    cx = x + w / 2
    if sub:
        ax.text(cx, y + h * 0.63, title, ha="center", va="center",
                fontsize=tsize, color=tcolor, fontweight="bold", zorder=4)
        ax.text(cx, y + h * 0.27, sub, ha="center", va="center",
                fontsize=ssize, color=MUTED, zorder=4)
    else:
        ax.text(cx, y + h / 2, title, ha="center", va="center",
                fontsize=tsize, color=tcolor, fontweight="bold", zorder=4)


def note(ax, x, y, lines, color=MUTED, size=6.3):
    """Small multi-line annotation, top-left anchored."""
    ax.text(x, y, lines, ha="left", va="top", fontsize=size, color=color,
            style="italic", linespacing=1.55, zorder=5)


# --------------------------------------------------------------------------- #
# Panel A: ChannelFusion (default fixed-query PMA, O(C) per location)
# --------------------------------------------------------------------------- #
def draw_panel_a(ax):
    """ChannelFusion: top-to-bottom chain, then the collapse to one token."""
    panel_header(ax, 2.0, 57.5, "A",
                 "ChannelFusion (default) \u2014 $O(C)$ per location")

    C = 3
    CX = [14.0, 22.5, 31.0]          # left edge of each channel column
    CW = 7.0
    CTR = [cx + CW / 2 for cx in CX]  # column centres
    GX0, GX1 = CX[0], CX[-1] + CW     # channel group extent: 14 .. 38

    # ---- stage 1: channel tokens at this (b, n) location ------------------- #
    y_tok, h_tok = 47.5, 6.0
    for i, cx in enumerate(CX):
        col = CHANNEL_COLORS[i]
        rrect(ax, cx, y_tok, CW, h_tok, edge=col, face=tint(col, 0.82),
              lw=1.5, rounding=0.6)
        ax.text(cx + CW / 2, y_tok + 4.1, f"$x_{i+1}$", ha="center",
                va="center", fontsize=8.8, color=col, fontweight="bold")
        ax.text(cx + CW / 2, y_tok + 1.7, f"$m_{i+1}$", ha="center",
                va="center", fontsize=6.2, color=MUTED, style="italic")
    ax.text((GX0 + GX1) / 2, 55.0,
            "channel tokens at one $(b,n)$  \u00b7  $C$ varies, order-free",
            ha="center", va="center", fontsize=6.6, color=INK)

    # ---- stage 2: LayerNorm -> keys = values ------------------------------ #
    y_ln, h_ln = 40.5, 3.5
    stage(ax, GX0, y_ln, GX1 - GX0, h_ln,
          "LayerNorm $\\mathrm{norm}_{kv}$", edge=MUTED, tcolor=INK,
          tsize=7.2, face=tint(MUTED, 0.93), lw=1.3)

    y_kv, h_kv = 30.5, 6.5
    for i, cx in enumerate(CX):
        col = CHANNEL_COLORS[i]
        rrect(ax, cx, y_kv, CW, h_kv, edge=col, face=tint(col, 0.86),
              lw=1.5, rounding=0.6)
        ax.text(cx + CW / 2, y_kv + h_kv / 2, f"$k_{i+1}\\!=\\!v_{i+1}$",
                ha="center", va="center", fontsize=7.8, color=col,
                fontweight="bold")

    # ---- stage 3: scores q_h . k_c,h / sqrt(D_h) -------------------------- #
    y_s, h_s = 20.5, 6.5
    stage(ax, GX0, y_s, GX1 - GX0, h_s,
          "$s_{c,h} = q_h \\cdot k_{c,h} / \\sqrt{D_h}$",
          "one score per channel, per head", edge=GREEN, tcolor=GREEN,
          tsize=8.0)

    # ---- stage 4: softmax over the channel axis --------------------------- #
    y_a, h_a = 10.5, 6.5
    stage(ax, GX0, y_a, GX1 - GX0, h_a,
          "$\\alpha_{c} = \\mathrm{softmax}_c\\,(s_{c})$",
          "softmax over the channel ($C$) axis", edge=ORANGE, tcolor=ORANGE,
          tsize=8.0)

    # connectors: tokens -> LayerNorm -> k=v -> scores -> alpha
    for cx in CTR:
        arrow(ax, (cx, y_tok), (cx, y_ln + h_ln), lw=1.2, ms=8)
        arrow(ax, (cx, y_ln), (cx, y_kv + h_kv), lw=1.2, ms=8)
        arrow(ax, (cx, y_kv), (cx, y_s + h_s), lw=1.2, ms=8)
        arrow(ax, (cx, y_s), (cx, y_a + h_a), lw=1.2, ms=8, color=ORANGE)

    # ---- the learned query q_j (shared by every location) ----------------- #
    qx, qw, qh = 2.5, 8.5, 6.5
    rrect(ax, qx, y_s, qw, qh, edge=BLUE, face=tint(BLUE, 0.78), lw=1.8,
          rounding=0.6)
    ax.text(qx + qw / 2, y_s + qh / 2, "$q_j$", ha="center", va="center",
            fontsize=10.0, color=BLUE, fontweight="bold")
    ax.text(qx + qw / 2, 18.6, "learned query", ha="center", va="center",
            fontsize=6.1, color=BLUE)
    ax.text(qx + qw / 2, 16.8, "shared by all $(b,n)$", ha="center",
            va="center", fontsize=6.0, color=MUTED, style="italic")
    arrow(ax, (qx + qw, y_s + qh / 2), (GX0, y_s + qh / 2), lw=1.5,
          color=BLUE, ms=9)

    # ---- collapse: weighted sum -> proj_out -> y_j ------------------------ #
    y_c = y_a + h_a / 2               # 13.75, centre of the alpha row
    sx, sw, sh = 42.0, 12.0, 11.0
    sy = y_c - sh / 2
    rrect(ax, sx, sy, sw, sh, edge=VIOLET, face=tint(VIOLET, 0.80), lw=1.8,
          rounding=0.7)
    ax.text(sx + sw / 2, y_c + 1.6, "$\\sum_c \\alpha_c\\, v_c$", ha="center",
            va="center", fontsize=9.6, color=VIOLET, fontweight="bold")
    ax.text(sx + sw / 2, y_c - 2.2, "per head $h$", ha="center", va="center",
            fontsize=6.2, color=MUTED)

    px, pw, ph = 58.0, 10.0, 6.5
    py = y_c - ph / 2
    stage(ax, px, py, pw, ph, "proj_out", "concat heads", edge=RED,
          tcolor=RED, tsize=7.4)

    yx, yw, yh = 71.0, 7.0, 7.5
    yy = y_c - yh / 2
    rrect(ax, yx, yy, yw, yh, edge=RED, face=tint(RED, 0.80), lw=1.8,
          rounding=0.6)
    ax.text(yx + yw / 2, y_c + 1.5, "$y_j$", ha="center", va="center",
            fontsize=10.0, color=RED, fontweight="bold")
    ax.text(yx + yw / 2, y_c - 2.0, "at $(b,n)$", ha="center", va="center",
            fontsize=6.1, color=MUTED)

    arrow(ax, (GX1, y_c), (sx, y_c), lw=1.5, color=ORANGE, ms=9)
    arrow(ax, (sx + sw, y_c), (px, y_c), lw=1.5, color=INK, ms=9)
    arrow(ax, (px + pw, y_c), (yx, y_c), lw=1.5, color=RED, ms=9)

    ax.text(74.5, 8.0, "scatter $\\rightarrow\\ [B, N, Q{\\cdot}D]$",
            ha="right", va="center", fontsize=6.6, color=MUTED, style="italic")

    # ---- values v_c tap into the weighted sum ----------------------------- #
    vx, vw, vh = 42.0, 10.0, 6.5
    vy = y_kv + (h_kv - vh) / 2
    rrect(ax, vx, vy, vw, vh, edge=MUTED, face=tint(MUTED, 0.92), lw=1.3,
          rounding=0.6)
    ax.text(vx + vw / 2, vy + vh / 2 + 1.3, "$v_c$", ha="center",
            va="center", fontsize=8.4, color=INK, fontweight="bold")
    ax.text(vx + vw / 2, vy + vh / 2 - 1.4, "values", ha="center",
            va="center", fontsize=6.1, color=MUTED)
    arrow(ax, (GX1, y_kv + h_kv / 2), (vx, vy + vh / 2), lw=1.4, ms=9)
    arrow(ax, (vx + vw / 2, vy), (vx + vw / 2, sy + sh), lw=1.4, ms=9)
    ax.text(vx + vw / 2 + 1.3, (vy + sy + sh) / 2, "$v_c$", ha="left",
            va="center", fontsize=6.8, color=MUTED)

    # ---- faithful side notes ---------------------------------------------- #
    note(ax, 57.0, 45.5,
         "keys = values = $\\mathrm{norm}_{kv}(x_c)$\n"
         "no $W_Q/W_K/W_V$ matrices")
    note(ax, 57.0, 37.5,
         "one learned query set $q_j$\n"
         "shared by every $(b,n)$")
    note(ax, 57.0, 29.5,
         "attention over the $C$ present\n"
         "channels \u2014 real rows only,\n"
         "no padding, no attention mask")


# --------------------------------------------------------------------------- #
# Panel B: HypernetworkWeightedSum (O(C) alternative)
# --------------------------------------------------------------------------- #
def draw_panel_b(ax):
    """HypernetworkWeightedSum: identity -> weights, tokens -> values."""
    panel_header(ax, 85.0, 57.5, "B",
                 "HypernetworkWeightedSum \u2014 $O(C)$")

    # two parallel branches: (left) identity -> weights, (right) values
    LX = [86.5, 93.0, 99.5]          # weight-branch column left edges
    RX = [113.0, 119.5, 126.0]       # value-branch column left edges
    W = 5.5
    LC = LX[1] + W / 2               # 95.75 -- left group centre
    RC = RX[1] + W / 2               # 122.25 -- right group centre

    # ---- branch inputs ----------------------------------------------------- #
    y_in, h_in = 47.0, 5.5
    for i in range(3):
        col = CHANNEL_COLORS[i]
        rrect(ax, LX[i], y_in, W, h_in, edge=col, face=tint(col, 0.82),
              lw=1.4, rounding=0.5)
        ax.text(LX[i] + W / 2, y_in + h_in / 2, f"$z_{i+1}$", ha="center",
                va="center", fontsize=8.4, color=col, fontweight="bold")
    ax.text(LC, 54.2, "identity $z_c$ \u2014 additive marker vector",
            ha="center", va="center", fontsize=6.5, color=INK)

    stage(ax, RX[0], y_in, RX[2] + W - RX[0], h_in, "tokens $t_c$",
          edge=MUTED, tcolor=INK, tsize=7.6, face=tint(MUTED, 0.92), lw=1.3)
    ax.text(RC, 54.2, "patch-embed tokens $[V, D]$", ha="center", va="center",
            fontsize=6.5, color=INK)

    # ---- weights branch: MLP -> softmax -> w_c ---------------------------- #
    y_mlp, h_mlp = 38.0, 6.5
    stage(ax, LX[0], y_mlp, LX[2] + W - LX[0], h_mlp, "MLP weight_net",
          "Linear $\\rightarrow$ GELU $\\rightarrow$ Linear", edge=BLUE,
          tcolor=BLUE, tsize=7.6)

    y_pill, h_pill = 31.0, 5.0
    rrect(ax, LX[0], y_pill, LX[2] + W - LX[0], h_pill, edge=ORANGE,
          face=tint(ORANGE, 0.86), lw=1.4, rounding=0.5)
    ax.text(LC, y_pill + 3.2, "$\\mathrm{softmax}_c$ over the", ha="center",
            va="center", fontsize=6.5, color=ORANGE, fontweight="bold")
    ax.text(LC, y_pill + 1.5, "channel ($C$) axis", ha="center",
            va="center", fontsize=6.5, color=ORANGE, fontweight="bold")

    y_w, h_w = 23.0, 5.5
    for i in range(3):
        rrect(ax, LX[i], y_w, W, h_w, edge=ORANGE, face=tint(ORANGE, 0.90),
              lw=1.4, rounding=0.5)
        ax.text(LX[i] + W / 2, y_w + h_w / 2, f"$w_{i+1}$", ha="center",
                va="center", fontsize=8.0, color=ORANGE, fontweight="bold")

    for i in range(3):
        arrow(ax, (LX[i] + W / 2, y_in), (LX[i] + W / 2, y_mlp + h_mlp),
              lw=1.1, ms=8, color=CHANNEL_COLORS[i])
    arrow(ax, (LC, y_mlp), (LC, y_pill + h_pill), lw=1.2, ms=8)
    for i in range(3):
        arrow(ax, (LC, y_pill), (LX[i] + W / 2, y_w + h_w), lw=1.1, ms=8,
              color=ORANGE)

    # ---- values branch: value_proj + out_norm -> V_c --------------------- #
    y_vp, h_vp = 38.0, 6.5
    stage(ax, RX[0], y_vp, RX[2] + W - RX[0], h_vp, "value_proj",
          "then $\\mathrm{out\\_norm}$", edge=GREEN, tcolor=GREEN, tsize=7.6)

    y_v, h_v = 23.0, 5.5
    for i in range(3):
        col = CHANNEL_COLORS[i]
        rrect(ax, RX[i], y_v, W, h_v, edge=col, face=tint(col, 0.86),
              lw=1.4, rounding=0.5)
        ax.text(RX[i] + W / 2, y_v + h_v / 2, f"$V_{i+1}$", ha="center",
                va="center", fontsize=8.0, color=col, fontweight="bold")

    arrow(ax, (RC, y_in), (RC, y_vp + h_vp), lw=1.2, ms=8)
    for i in range(3):
        arrow(ax, (RC, y_vp), (RX[i] + W / 2, y_v + h_v), lw=1.1, ms=8,
              color=CHANNEL_COLORS[i])

    # ---- merge: sum_c w_c V_c -------------------------------------------- #
    sx, sw = 86.5, 45.0
    y_sum, h_sum = 8.5, 10.0
    rrect(ax, sx, y_sum, sw, h_sum, edge=VIOLET, face=tint(VIOLET, 0.80),
          lw=1.8, rounding=0.7)
    ax.text(sx + sw / 2, y_sum + 6.6, "$\\sum_c w_c\\, V_c$", ha="center",
            va="center", fontsize=9.6, color=VIOLET, fontweight="bold")
    ax.text(sx + sw / 2, y_sum + 2.6,
            "per location $(b,n)$  $\\rightarrow$  fused $[B,N,Q{\\cdot}D]$",
            ha="center", va="center", fontsize=6.4, color=MUTED)

    arrow(ax, (LC, y_w), (LC, y_sum + h_sum), lw=1.4, color=ORANGE, ms=9)
    arrow(ax, (RC, y_v), (RC, y_sum + h_sum), lw=1.4, ms=9)
    ax.text(LC + 1.4, y_sum + h_sum + 1.8, "$w_c$", ha="left", va="center",
            fontsize=6.6, color=ORANGE)
    ax.text(RC + 1.4, y_sum + h_sum + 1.8, "$V_c$", ha="left", va="center",
            fontsize=6.6, color=MUTED)

    # ---- cost callout ----------------------------------------------------- #
    ax.text(sx + sw / 2, 6.4, "Complexity: $O(C)$ per location",
            ha="center", va="center", fontsize=8.2, color=GREEN,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.45", facecolor=tint(GREEN, 0.88),
                      edgecolor=GREEN, lw=1.4), zorder=5)
    ax.text(sx + sw / 2, 3.1,
            "routing keyed on marker identity, not token content",
            ha="center", va="center", fontsize=6.4, color=MUTED,
            style="italic")


# --------------------------------------------------------------------------- #
# Inset: standard fused QKV self-attention (for contrast)
# --------------------------------------------------------------------------- #
def draw_inset(ax):
    """Small inset showing standard fused QKV attention (for contrast)."""
    x0, x1 = 137.5, 170.5
    y0, y1 = 10.5, 52.5
    cxm = (x0 + x1) / 2

    rrect(ax, x0, y0, x1 - x0, y1 - y0, edge=MUTED, face=tint(MUTED, 0.955),
          lw=1.3, dashed=True, rounding=0.6, z=1)

    ax.text(cxm, y1 - 2.4, "Standard fused QKV attention",
            ha="center", va="center", fontsize=7.4, color=INK,
            fontweight="bold")
    ax.text(cxm, y1 - 4.9, "for contrast \u2014 channels self-attend",
            ha="center", va="center", fontsize=6.2, color=MUTED,
            style="italic")

    # input tokens X
    x_x, w_x, y_x, h_x = 148.5, 11.0, 42.5, 3.8
    stage(ax, x_x, y_x, w_x, h_x, "tokens $X$", edge=MUTED, tcolor=INK,
          tsize=6.8, face="white", lw=1.2)

    # the three projection matrices
    y_w, h_w = 34.0, 5.0
    WX = [142.5, 150.5, 158.5]
    labels = ["$W_Q$", "$W_K$", "$W_V$"]
    cols = [BLUE, GREEN, ORANGE]
    for i in range(3):
        rrect(ax, WX[i], y_w, 6.5, h_w, edge=cols[i], face=tint(cols[i], 0.90),
              lw=1.4, rounding=0.5)
        ax.text(WX[i] + 3.25, y_w + h_w / 2, labels[i], ha="center",
                va="center", fontsize=7.6, color=cols[i], fontweight="bold")
        arrow(ax, (x_x + 2.0 + 3.0 * i, y_x), (WX[i] + 3.25, y_w + h_w),
              lw=1.0, ms=7)

    # attention op
    y_at, h_at = 25.5, 5.6
    stage(ax, 140.0, y_at, 28.0, h_at, "Attn$(Q, K, V)$",
          "$\\mathrm{softmax}(QK^{\\top}\\,/\\sqrt{d})\\,V$", edge=VIOLET,
          tcolor=VIOLET, tsize=7.6, ssize=6.1)
    ends = [147.5, 154.0, 160.5]
    for i in range(3):
        arrow(ax, (WX[i] + 3.25, y_w), (ends[i], y_at + h_at),
              lw=1.0, ms=7, color=cols[i])

    # output
    stage(ax, 146.0, 17.0, 16.0, 4.4,
          "one vector per token", "$[T, D]$ \u2014 no fixed-size $C$-axis",
          edge=INK, tcolor=INK, tsize=6.4, ssize=5.9,
          face=tint(INK, 0.94), lw=1.1)
    arrow(ax, (cxm, y_at), (cxm, 21.4), lw=1.1, ms=8)

    ax.text(cxm, 14.2,
            "no shared learned query: $Q$ comes from the tokens",
            ha="center", va="center", fontsize=6.0, color=MUTED,
            style="italic")
    ax.text(cxm, 12.2,
            "variable $C$ needs padding or an attention mask",
            ha="center", va="center", fontsize=6.0, color=MUTED,
            style="italic")


# --------------------------------------------------------------------------- #
# Main figure assembly
# --------------------------------------------------------------------------- #
def build() -> plt.Figure:
    fig = plt.figure(figsize=(14.0, 7.0))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_xlim(0, 172)
    ax.set_ylim(0, 64)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    # title
    ax.text(2, 61.5, "L4", ha="left", va="center", fontsize=15,
            fontweight="bold", color=INK)
    ax.text(6.5, 61.5, "Channel fusion: PMA pooling vs hypernetwork routing",
            ha="left", va="center", fontsize=12.2, fontweight="bold", color=INK)

    # panels
    draw_panel_a(ax)

    # vertical separator
    ax.plot([80.0, 80.0], [4.0, 56.0], color=MUTED, lw=1.0,
            linestyle="--", zorder=1)

    draw_panel_b(ax)
    draw_inset(ax)

    # footnote
    ax.text(2.0, 0.8,
            "Both variants are padding-free and produce $[B, N, Q{\\cdot}D]$ fused embeddings. "
            "ChannelFusion uses learned queries (attention over channels); "
            "HypernetworkWeightedSum uses marker-identity routing (linear cost).",
            ha="left", va="bottom", fontsize=7.0, color=MUTED, style="italic")

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
