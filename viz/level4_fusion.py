"""L4 -- channel fusion internals: two interchangeable pooling strategies.

Panel A shows :class:`ChannelFusion`, the default PMA-style pooling over channel
axis at each spatial location. Panel B shows :class:`HypernetworkWeightedSum`,
the O(C) alternative for high-channel-count panels. A small inset contrasts with
standard fused QKV self-attention.

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
    tint, rrect, box, arrow, edge_label, sq_bracket, panel_header,
    install_rcparams, save_figure,
)

OUT = Path(__file__).resolve().parent / "out"


# --------------------------------------------------------------------------- #
# Panel A: ChannelFusion (default PMA-style, O(C²) per location)
# --------------------------------------------------------------------------- #
def draw_panel_a(ax):
    """ChannelFusion: attention-pooling over channels at one location."""
    y_base = 58.0
    panel_header(ax, 2.0, y_base, "A",
                 "ChannelFusion (default) \u2014 $O(C^2)$ per location")

    # ---- channel tokens as columns ---------------------------------------- #
    C = 3
    x_tokens = 6.0
    y_tok = 32.0
    cw, pitch = 3.8, 5.0
    for c in range(C):
        rrect(ax, x_tokens + c * pitch, y_tok, cw, 18.0, edge=CHANNEL_COLORS[c],
              face=tint(CHANNEL_COLORS[c], 0.82), lw=1.5, rounding=0.6)
        ax.text(x_tokens + c * pitch + cw / 2, y_tok + 19.2, f"$x_{{{c+1}}}$",
                ha="center", va="center", fontsize=9.0, color=CHANNEL_COLORS[c],
                fontweight="bold")
        # marker identity indication
        ax.text(x_tokens + c * pitch + cw / 2, y_tok - 1.5, f"marker {c+1}",
                ha="center", va="center", fontsize=6.5, color=MUTED, style="italic")
    ax.text(x_tokens + C * pitch / 2 - 0.5, y_tok + 22.0, "C channel tokens",
            ha="center", va="center", fontsize=7.8, color=INK)

    # ---- LayerNorm to get k = v -------------------------------------------- #
    x_kv = 28.0
    for c in range(C):
        rrect(ax, x_kv + c * pitch, y_tok, cw, 18.0, edge=CHANNEL_COLORS[c],
              face=tint(CHANNEL_COLORS[c], 0.85), lw=1.5, rounding=0.6)
        ax.text(x_kv + c * pitch + cw / 2, y_tok + 19.2, f"$k_{{{c+1}}}=v_{{{c+1}}}$",
                ha="center", va="center", fontsize=8.0, color=CHANNEL_COLORS[c],
                fontweight="bold")
        arrow(ax, (x_tokens + c * pitch + cw + 0.8, y_tok + 9.0),
              (x_kv + c * pitch - 0.8, y_tok + 9.0), lw=1.3)
    ax.text(x_kv + C * pitch / 2 - 0.5, y_tok + 22.0, "keys = values (no $W_Q/W_K/W_V$)",
            ha="center", va="center", fontsize=7.8, color=INK)
    
    # LayerNorm label
    ax.text((x_tokens + cw + x_kv) / 2, y_tok + 13.5, "LayerNorm",
            ha="center", va="center", fontsize=7.0, color=MUTED,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor=MUTED, lw=1.0))

    # ---- learned query q_j ------------------------------------------------- #
    qw, qh = 5.5, 8.0
    qx, qy = 7.0, 16.0
    rrect(ax, qx, qy, qw, qh, edge=BLUE, face=tint(BLUE, 0.78), lw=1.8, rounding=0.5)
    ax.text(qx + qw / 2, qy + qh / 2, "$q_j$", ha="center", va="center",
            fontsize=10.0, color=BLUE, fontweight="bold")
    ax.text(qx + qw / 2, qy - 1.6, "learned query", ha="center", va="center",
            fontsize=6.5, color=BLUE)
    ax.text(qx + qw / 2, qy - 2.7, "(shared ∀ locations)", ha="center", va="center",
            fontsize=6.0, color=MUTED, style="italic")

    # ---- attention scores = q · k / sqrt(D_h) ------------------------------ #
    x_scores = 28.0
    y_scores = 10.0
    score_w, score_h = 3.8, 2.8
    for c in range(C):
        rrect(ax, x_scores + c * pitch, y_scores, score_w, score_h,
              edge=CHANNEL_COLORS[c], face=tint(CHANNEL_COLORS[c], 0.88),
              lw=1.2, rounding=0.4)
        ax.text(x_scores + c * pitch + score_w / 2, y_scores + score_h / 2,
                f"$s_{c+1}$", ha="center", va="center", fontsize=8.0,
                color=CHANNEL_COLORS[c], fontweight="bold")
    ax.text(x_scores + C * pitch + 4.0, y_scores - 1.8, "scores per channel",
            ha="left", va="center", fontsize=6.5, color=MUTED)
    
    # arrow from q and k to scores
    arrow(ax, (qx + qw / 2, qy), (x_scores + pitch, y_scores + score_h + 0.5),
          lw=1.3, rad=-0.3)
    arrow(ax, (x_kv + pitch + cw / 2, y_tok), (x_scores + pitch, y_scores + score_h + 0.5),
          lw=1.3, rad=0.3)
    ax.text(20.0, 20.0, "$q_h \\cdot k_{c,h} / \\sqrt{D_h}$", ha="center", va="center",
            fontsize=7.0, color=MUTED,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor=MUTED, lw=1.0))

    # ---- softmax over C ---------------------------------------------------- #
    y_alpha = 4.5
    for c in range(C):
        rrect(ax, x_scores + c * pitch, y_alpha, score_w, score_h,
              edge=CHANNEL_COLORS[c], face=tint(ORANGE, 0.80),
              lw=1.2, rounding=0.4)
        ax.text(x_scores + c * pitch + score_w / 2, y_alpha + score_h / 2,
                f"$\\alpha_{c+1}$", ha="center", va="center", fontsize=8.0,
                color=ORANGE, fontweight="bold")
        arrow(ax, (x_scores + c * pitch + score_w / 2, y_scores),
              (x_scores + c * pitch + score_w / 2, y_alpha + score_h + 0.3),
              lw=1.1, color=ORANGE)
    ax.text(x_scores + C * pitch / 2 - 0.5, y_alpha - 1.8,
            "softmax over C (channels)", ha="center", va="center",
            fontsize=6.5, color=ORANGE, fontweight="bold")
    ax.text(x_scores + C * pitch + 3.5, y_alpha + score_h / 2, "softmax$_c$",
            ha="center", va="center", fontsize=7.0, color=ORANGE,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor=ORANGE, lw=1.0))

    # ---- weighted sum Σ α_c v_c -------------------------------------------- #
    x_fused = 50.0
    y_fused = 35.0
    fused_w, fused_h = 12.0, 10.0
    rrect(ax, x_fused, y_fused, fused_w, fused_h, edge=VIOLET, face=tint(VIOLET, 0.78),
          lw=1.8, rounding=0.7)
    ax.text(x_fused + fused_w / 2, y_fused + fused_h / 2 + 1.5,
            "$\\sum_c \\alpha_c v_c$", ha="center", va="center",
            fontsize=10.0, color=VIOLET, fontweight="bold")
    ax.text(x_fused + fused_w / 2, y_fused + fused_h / 2 - 1.8,
            "per head", ha="center", va="center",
            fontsize=6.8, color=MUTED)

    # arrows from alpha and v to fused
    arrow(ax, (x_scores + pitch, y_alpha + score_h), (x_fused, y_fused + fused_h / 2),
          lw=1.3, rad=-0.25, color=ORANGE)
    arrow(ax, (x_kv + pitch + cw, y_tok + 9.0), (x_fused, y_fused + fused_h / 2),
          lw=1.3, rad=0.2)

    # ---- concat heads + proj_out ------------------------------------------- #
    x_out = 68.0
    y_out = 35.0
    out_w, out_h = 10.0, 10.0
    rrect(ax, x_out, y_out, out_w, out_h, edge=RED, face=tint(RED, 0.80),
          lw=1.8, rounding=0.7)
    ax.text(x_out + out_w / 2, y_out + out_h / 2 + 1.5, "$y_j$", ha="center",
            va="center", fontsize=11.0, color=RED, fontweight="bold")
    ax.text(x_out + out_w / 2, y_out + out_h / 2 - 1.8, "fused token",
            ha="center", va="center", fontsize=6.8, color=MUTED)
    arrow(ax, (x_fused + fused_w, y_fused + fused_h / 2),
          (x_out, y_out + out_h / 2), lw=1.5, color=RED)
    ax.text((x_fused + fused_w + x_out) / 2, y_out + out_h + 2.2,
            "concat heads → proj_out", ha="center", va="center",
            fontsize=7.0, color=RED,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor=RED, lw=1.0))

    # ---- scatter into [B, N, Q·D] ------------------------------------------ #
    ax.text(x_out + out_w / 2, y_out - 2.5, "scatter → $[B, N, Q{\\cdot}D]$",
            ha="center", va="center", fontsize=7.0, color=MUTED, style="italic")



# --------------------------------------------------------------------------- #
# Panel B: HypernetworkWeightedSum (O(C) alternative)
# --------------------------------------------------------------------------- #
def draw_panel_b(ax):
    """HypernetworkWeightedSum: MLP-based routing keyed on marker identity."""
    y_base = 58.0
    panel_header(ax, 85.0, y_base, "B", "HypernetworkWeightedSum (O(C) alternative)")

    C = 3
    x_identity = 88.0
    y_id = 36.0
    iw, ih, ipitch = 4.5, 4.0, 6.0

    # ---- marker identities z_c --------------------------------------------- #
    for c in range(C):
        rrect(ax, x_identity + c * ipitch, y_id, iw, ih, edge=CHANNEL_COLORS[c],
              face=tint(CHANNEL_COLORS[c], 0.82), lw=1.5, rounding=0.5)
        ax.text(x_identity + c * ipitch + iw / 2, y_id + ih / 2, f"$z_{c+1}$",
                ha="center", va="center", fontsize=9.0, color=CHANNEL_COLORS[c],
                fontweight="bold")
    ax.text(x_identity + C * ipitch / 2 - 0.5, y_id + ih + 1.5,
            "marker identities", ha="center", va="center",
            fontsize=7.8, color=INK)

    # ---- MLP weight_net ---------------------------------------------------- #
    x_mlp = 88.0
    y_mlp = 27.0
    mlp_w, mlp_h = 18.0, 6.0
    rrect(ax, x_mlp, y_mlp, mlp_w, mlp_h, edge=BLUE, face=tint(BLUE, 0.85),
          lw=1.6, rounding=0.6)
    ax.text(x_mlp + mlp_w / 2, y_mlp + mlp_h / 2, "MLP weight_net",
            ha="center", va="center", fontsize=8.5, color=BLUE, fontweight="bold")
    ax.text(x_mlp + mlp_w / 2, y_mlp + mlp_h / 2 - 1.8, "Linear → GELU → Linear",
            ha="center", va="center", fontsize=6.5, color=MUTED)
    for c in range(C):
        arrow(ax, (x_identity + c * ipitch + iw / 2, y_id),
              (x_mlp + mlp_w / 2, y_mlp + mlp_h), lw=1.1, color=CHANNEL_COLORS[c])

    # ---- softmax over C ---------------------------------------------------- #
    y_weights = 18.0
    for c in range(C):
        rrect(ax, x_identity + c * ipitch, y_weights, iw, ih, edge=ORANGE,
              face=tint(ORANGE, 0.80), lw=1.5, rounding=0.5)
        ax.text(x_identity + c * ipitch + iw / 2, y_weights + ih / 2, f"$w_{c+1}$",
                ha="center", va="center", fontsize=9.0, color=ORANGE, fontweight="bold")
        arrow(ax, (x_mlp + (c + 1) * (mlp_w / (C + 1)), y_mlp),
              (x_identity + c * ipitch + iw / 2, y_weights + ih), lw=1.1, color=ORANGE)
    ax.text(x_identity + C * ipitch / 2 - 0.5, y_weights - 1.6,
            "softmax$_c$(logits)", ha="center", va="center",
            fontsize=7.0, color=ORANGE, fontweight="bold")

    # ---- value projection -------------------------------------------------- #
    x_values = 120.0
    y_val = 36.0
    vw, vh = 4.5, 4.0
    for c in range(C):
        rrect(ax, x_values + c * ipitch, y_val, vw, vh, edge=CHANNEL_COLORS[c],
              face=tint(CHANNEL_COLORS[c], 0.85), lw=1.5, rounding=0.5)
        ax.text(x_values + c * ipitch + vw / 2, y_val + vh / 2, f"$V_{c+1}$",
                ha="center", va="center", fontsize=9.0, color=CHANNEL_COLORS[c],
                fontweight="bold")
    ax.text(x_values + C * ipitch / 2 - 0.5, y_val + vh + 1.5,
            "value_proj(tokens)", ha="center", va="center",
            fontsize=7.8, color=INK)

    # ---- weighted sum ------------------------------------------------------ #
    x_sum = 115.0
    y_sum = 6.0
    sum_w, sum_h = 14.0, 9.0
    rrect(ax, x_sum, y_sum, sum_w, sum_h, edge=VIOLET, face=tint(VIOLET, 0.78),
          lw=1.8, rounding=0.7)
    ax.text(x_sum + sum_w / 2, y_sum + sum_h / 2 + 1.5,
            "$\\sum_c w_c V_c$", ha="center", va="center",
            fontsize=10.0, color=VIOLET, fontweight="bold")
    ax.text(x_sum + sum_w / 2, y_sum + sum_h / 2 - 1.5,
            "out_norm", ha="center", va="center",
            fontsize=6.8, color=MUTED)

    # arrows to sum
    arrow(ax, (x_identity + ipitch, y_weights), (x_sum + sum_w / 2, y_sum + sum_h),
          lw=1.3, rad=-0.2, color=ORANGE)
    arrow(ax, (x_values + ipitch, y_val), (x_sum + sum_w / 2, y_sum + sum_h),
          lw=1.3, rad=0.2)

    # ---- output ------------------------------------------------------------ #
    ax.text(x_sum + sum_w / 2, y_sum - 2.0, "scatter → $[B, N, Q{\\cdot}D]$",
            ha="center", va="center", fontsize=7.0, color=MUTED, style="italic")

    # ---- complexity callout ------------------------------------------------ #
    ax.text(115.0, 54.0, "Complexity: $O(C)$ per location",
            ha="center", va="center", fontsize=8.5, color=GREEN, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.6", facecolor=tint(GREEN, 0.88),
                      edgecolor=GREEN, lw=1.5))

    # ---- key difference note ----------------------------------------------- #
    ax.text(115.0, 50.0, "Weights keyed on marker identity, not token content",
            ha="center", va="center", fontsize=7.0, color=MUTED, style="italic")


# --------------------------------------------------------------------------- #
# Inset: standard fused QKV self-attention (for contrast)
# --------------------------------------------------------------------------- #
def draw_inset(ax):
    """Small inset showing standard fused QKV for comparison."""
    x_in = 148.0
    y_in = 40.0
    w_in, h_in = 22.0, 16.0
    
    # frame
    rrect(ax, x_in, y_in, w_in, h_in, edge=MUTED, face=tint(MUTED, 0.95),
          lw=1.5, dashed=True, rounding=0.6, z=1)
    ax.text(x_in + w_in / 2, y_in + h_in + 1.5, "Standard attention (for contrast)",
            ha="center", va="center", fontsize=7.5, color=MUTED, fontweight="bold")

    # QKV projection
    box_w, box_h = 5.0, 4.5
    x_qkv = x_in + 2.0
    y_qkv = y_in + h_in - box_h - 2.0
    for i, label in enumerate(["Q", "K", "V"]):
        bx = x_qkv + i * 5.5
        col = [BLUE, GREEN, ORANGE][i]
        rrect(ax, bx, y_qkv, box_w, box_h, edge=col, face=tint(col, 0.85),
              lw=1.2, rounding=0.4)
        ax.text(bx + box_w / 2, y_qkv + box_h / 2, f"$W_{label}$", ha="center",
                va="center", fontsize=8.0, color=col, fontweight="bold")

    # input X
    ax.text(x_in + w_in / 2, y_qkv - 1.2, "X", ha="center", va="center",
            fontsize=8.0, color=INK)
    arrow(ax, (x_in + w_in / 2, y_qkv - 0.5), (x_in + w_in / 2, y_qkv), lw=1.0)

    # attention result
    y_attn = y_in + 3.0
    rrect(ax, x_in + 6.0, y_attn, 10.0, 5.0, edge=VIOLET, face=tint(VIOLET, 0.85),
          lw=1.2, rounding=0.5)
    ax.text(x_in + 11.0, y_attn + 2.5, "Attn$(Q, K, V)$", ha="center", va="center",
            fontsize=7.5, color=VIOLET, fontweight="bold")

    # note
    ax.text(x_in + w_in / 2, y_in - 1.2,
            "Channels self-attend;\nno shared query",
            ha="center", va="top", fontsize=6.0, color=MUTED,
            style="italic", linespacing=1.4)


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
    ax.plot([82.0, 82.0], [0.0, 57.0], color=MUTED, lw=1.0, linestyle="--", zorder=1)
    
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
