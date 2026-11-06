"""L2 -- tensor/axis layout of the padding-free channel-fusion architecture.

The hero figure: a left-to-right diagram of literal, axis-labelled array stacks
in the Perceiver-IO Fig 2 / Perceiver Fig 1 idiom, showing where the variable
channel count ``C`` appears (input token list length only) and how it is
summed away by fusion to produce one fixed-size fused vector per location.

Shows:

1. input set ``[ΣC, 1, H, W]`` (uneven: height = C_b, colour per channel)
2. post-mask visible token list ``[V, D]`` with the ``(b, channel_idx, n)``
   index triple annotated; one omitted row is drawn as an external dashed
   callout
3. the per-patch grouping highlighted: one row per channel at the same ``n``
4. the fused vectors on the ``[B, N]`` location grid, after channel fusion
5. encoder module: the ``[K_f, D]`` visible tokens (no memory tokens; the
   ``memory`` arm would prepend ``[B·M, D]`` gold rows)
6. spatial pooling of ``tok_out`` to ``latents [B, L, D]``; the decoder KV is
   ``enc_all = latents`` with fixed length ``[B·L, D]`` and does **not**
   contain the raw ``tok_out``
7. decoder target queries ``[K_t, D]`` -> flat ``[Σ p²]`` predicted pixels
8. the ghost ``[B, C_max, N, D]`` tensor (dashed, struck) that is never
   materialised, plus role / channel legends and a faithful symbol glossary.

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
    GREEN, FUSION, ENCODER, LOSS, FUSED, POOL, VIOLET, INK, MUTED,
    CHANNEL_COLORS,
    tint, rrect, arrow, edge_label, sq_bracket, token_rows, grid_cells,
    legend, channel_legend, padded_ghost,
    install_rcparams, save_figure,
)

OUT = Path(__file__).resolve().parent / "out"

# The single main dataflow line every stack is vertically centred on.
YMID = 30.5

# Shared horizontal anchors so arrows in ``build`` line up with the stacks.
ENC_X, ENC_W = 62.5, 9.0
KV_X, KV_W = 78.0, 9.5
TGT_X, TGT_W = 95.0, 7.0
PRED_X, PRED_W = 108.0, 7.0


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
    """Set of single-channel images [ΣC, 1, H, W].

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
    ax.text(x + cw / 2, y0 - 2.7, "$p_b{\\times}p_b$ patches $\\to N_b$",
            ha="center", va="center", fontsize=6.2, color=MUTED)
    ax.text(x + cw / 2, y0 - 4.2,
            "stored $P{\\times}P$ kernel, PI-resized to $p_b$",
            ha="center", va="center", fontsize=6.0, color=GREEN)


def _token_column(ax, x, w, h, pitch, n_rows, specs):
    y0 = YMID - ((n_rows - 1) * pitch + h) / 2.0
    top = token_rows(ax, x, y0, w, h, pitch, specs, rounding=0.35, z=3)
    return y0, top


def _visible_specs():
    """Visible rows of a channel-major 3-channel x 3-patch schematic.

    Rows are listed bottom-to-top; reversing the channel order here puts
    channel 0 at the top, matching the input tiles.  The top channel's final
    ``(channel 0, patch 2)`` row is omitted by DCP and drawn as the external
    dashed callout, so every row in the solid stack is visible.
    """
    specs = []
    for ch in (2, 1, 0):
        for _ in range(2 if ch == 0 else 3):
            specs.append((CHANNEL_COLORS[ch], False))
    return specs


def draw_token_list(ax):
    """Post-mask visible token list [V, D] with index-triple annotation."""
    x, w, h, pitch, n = 11.5, 9.0, 2.0, 2.3, 8
    y0, top = _token_column(ax, x, w, h, pitch, n, _visible_specs())

    # (b, channel_idx, n) index-triple annotation just right of the stack
    anno_y = y0 + 1 * pitch + h / 2
    ax.text(x + w + 0.6, anno_y,
            "$(b,\\ \\mathrm{channel\\_idx},\\ n)$",
            ha="left", va="center", fontsize=5.8, color=INK, style="italic",
            zorder=5)

    sq_bracket(ax, x - 0.7, y0, top, color=MUTED, lw=1.1, tick=0.55)
    axis_text(ax, x - 1.7, y0 + 0.30 * (top - y0), "$V$", rotation=90)
    h_bracket(ax, x, x + w, top + 0.6, color=MUTED, down=True)
    axis_text(ax, x + w / 2, top + 1.6, "$D$", size=7.5)
    ax.text(x + w / 2, y0 - 1.1, "visible tokens $[V,D]$", ha="center",
            va="center", fontsize=7.8, fontweight="bold", color=INK)
    ax.text(x + w / 2, y0 - 2.6,
            "full list $[\\sum_b C_b{\\cdot}N_b,D]$ $\\to$ DCP omits rows",
            ha="center", va="center", fontsize=6.0, color=MUTED,
            style="italic")
    ax.text(x + w / 2, y0 - 4.0,
            "visible $[V,D]$, $V\\leq\\sum_b C_b N_b$ "
            "$\\cdot$ schematic 3ch $\\times$ 3p",
            ha="center", va="center", fontsize=6.0, color=MUTED,
            style="italic")


def draw_omitted_callout(ax):
    """External dashed callout for the one row DCP omitted.

    Kept outside the solid stack so the stack reads as the post-mask ``[V,D]``
    object, not as a padded/zeroed tensor.
    """
    x, y, w, h = 9.8, 14.8, 8.0, 2.0
    rrect(ax, x, y, w, h, edge=LOSS, face="white", lw=1.2, dashed=True,
          rounding=0.35, z=3)
    ax.text(x + w + 0.8, y + h / 2, "omitted row\n(masked by DCP)",
            ha="left", va="center", fontsize=6.0, color=LOSS,
            linespacing=1.25, zorder=5)


def draw_channel_grouping(ax):
    """Highlight one row per channel at the same patch index ``n``."""
    x, w, h, pitch, n = 25.5, 9.0, 2.0, 2.3, 8
    y0, top = _token_column(ax, x, w, h, pitch, n, _visible_specs())

    # Rows 1, 4, 7 are patch n=1 for channels 2, 1, 0 (channel-major layout,
    # channel 0 at the top to match the input tiles).
    for row in (1, 4, 7):
        rrect(ax, x - 0.4, y0 + row * pitch - 0.2, w + 0.8, h + 0.4,
              edge=FUSION, face="none", lw=2.0, dashed=False,
              rounding=0.45, z=4)

    sq_bracket(ax, x - 0.7, y0, top, color=MUTED, lw=1.1, tick=0.55)
    axis_text(ax, x - 1.7, y0 + 0.30 * (top - y0), "$V$", rotation=90)
    h_bracket(ax, x, x + w, top + 0.6, color=MUTED, down=True)
    axis_text(ax, x + w / 2, top + 1.6, "$D$", size=7.5)
    ax.text(x + w / 2, y0 - 1.1, "same patch $n$ across channels",
            ha="center", va="center", fontsize=7.5, fontweight="bold",
            color=INK)
    ax.text(x + w / 2, y0 - 2.55,
            "one row per channel $\\cdot$ $z_c$ pooled at $(b,n)$",
            ha="center", va="center", fontsize=6.1, color=FUSION)


def draw_fused_grid(ax):
    """Fused vectors on the [B, N] location grid (FUSED blue)."""
    x0, cw, pitch = 42.5, 2.4, 2.7
    B, N = 4, 5
    gh = (B - 1) * pitch + cw
    y0 = YMID - gh / 2.0
    top, right = grid_cells(ax, x0, y0, B, N, cw, pitch, edge=FUSED,
                            face=tint(FUSED, 0.76), lw=1.1, rounding=0.25, z=3)

    h_bracket(ax, x0, right, top + 0.6, color=MUTED, down=True)
    axis_text(ax, (x0 + right) / 2, top + 1.6, "$N=\\max_b N_b$", size=7.0)
    sq_bracket(ax, x0 - 0.7, y0, top, color=MUTED, lw=1.1, tick=0.55)
    axis_text(ax, x0 - 1.7, y0 + 0.22 * (top - y0), "$B$", rotation=90)
    axis_text(ax, right + 0.9, top - 0.5, "$D$", ha="left", size=7.5)
    ax.text(right + 0.9, top - 1.7, "no fusion PE", ha="left",
            fontsize=5.6, color=FUSION)
    ax.text((x0 + right) / 2, y0 - 1.1, "fused $[K_f,D]$", ha="center",
            va="center", fontsize=7.8, fontweight="bold", color=INK)


def draw_fusion(ax):
    """ChannelFusion stage: dashed module frame + z_c routing-bias/skip note."""
    rrect(ax, 34.6, 29.2, 7.6, 2.6, edge=FUSION, face="none", lw=1.3,
          dashed=True, rounding=0.6, z=2)
    arrow(ax, (35.0, YMID), (42.1, YMID), lw=1.5, color=FUSION)
    ax.text(38.4, 33.0, "ChannelFusion", ha="center", va="center",
            fontsize=6.8, fontweight="bold", color=FUSION)
    ax.text(38.5, 34.4,
            "$z_c$: routing bias $+$ skip\n"
            "(never added to tokens)",
            ha="center", va="center", fontsize=4.8, color=GREEN,
            linespacing=1.3, zorder=7)


def draw_encoder(ax):
    """Encoder module: the ``[K_f, D]`` fused tokens (no memory tokens)."""
    x, w, h, pitch = ENC_X, ENC_W, 1.9, 2.2
    n_vis = 6
    y0 = YMID - ((n_vis - 1) * pitch + h) / 2.0

    for i in range(n_vis):
        rrect(ax, x, y0 + i * pitch, w, h, edge=FUSED,
              face=tint(FUSED, 0.80), lw=1.2, rounding=0.3, z=3, crossed=True)
    top = y0 + (n_vis - 1) * pitch + h

    # ENCODER-tinted dashed module frame so the role colour reads.
    rrect(ax, x - 0.4, y0 - 0.6, w + 0.9, (top - y0) + 1.2, edge=ENCODER,
          face="none", lw=1.4, dashed=True, rounding=0.6, z=2)

    h_bracket(ax, x, x + w, top + 0.7, color=MUTED, down=True)
    axis_text(ax, x + w + 0.7, top + 0.7, "$D$", ha="left", size=7.5)

    # module title / role sits above the frame, clear of the flow arrows
    ax.text(x + w / 2, top + 4.5, "encoder $\\cdot$ TransformerEncoder",
            ha="center", va="center", fontsize=7.8, fontweight="bold",
            color=INK)
    ax.text(x + w / 2, top + 3.0,
            "no memory tokens $\\cdot$ depth$=$4 blocks $\\cdot$ "
            "2-D axial RoPE (metric token centres)",
            ha="center", va="center", fontsize=5.9, color=ENCODER)
    ax.text(x + w / 2, top + 1.9,
            "scale: Fourier$(\\log p_b,\\ \\log\\,\\mu m/px)$ "
            "added to encoder tokens",
            ha="center", va="center", fontsize=5.9, color=MUTED)

    content_c = y0 + ((n_vis - 1) * pitch + h) / 2

    sq_bracket(ax, x - 1.0, y0, top, color=FUSED, lw=1.1, tick=0.55)
    axis_text(ax, x - 2.1, content_c, "$K_f$", color=FUSED, rotation=90,
              size=7.0)

    return {"right": x + w + 0.5, "content_c": content_c}


def draw_kv(ax):
    """Decoder KV ``enc_all``: the fixed latent set.

    The KV is a single fixed-length tensor ``[B·L, D]``; the ``memory`` arm
    would prepend ``[B·M, D]`` ``mem_out`` rows.  ``tok_out`` is deliberately
    absent.
    """
    x, w, h, pitch = KV_X, KV_W, 1.8, 2.1
    sections = [
        ("latents", 6, POOL, 0.72),
    ]
    n = sum(count for _, count, _, _ in sections)
    y0 = YMID - ((n - 1) * pitch + h) / 2.0
    top = y0 + (n - 1) * pitch + h

    centers = {}
    j = 0
    for name, count, color, amt in sections:
        ys = []
        for _ in range(count):
            yy = y0 + j * pitch
            rrect(ax, x, yy, w, h, edge=color, face=tint(color, amt), lw=1.2,
                  rounding=0.3, z=3, crossed=True)
            ys.append(yy + h / 2)
            j += 1
        centers[name] = sum(ys) / len(ys)

    # one bracket for the whole concatenated tensor; its row extent is named by
    # the top shape label, so the vertical bracket itself carries no text (keeps
    # the incoming routing arrows clear).
    sq_bracket(ax, x - 1.0, y0, top, color=MUTED, lw=1.1, tick=0.55)
    h_bracket(ax, x, x + w, top + 0.6, color=MUTED, down=True)
    axis_text(ax, x + w / 2, top + 1.7,
              "$[B\\cdot L,\\,D]$", size=7.2)

    # section label on the right, kept clear of the dec arrow at YMID
    ax.text(x + w + 0.8, centers["latents"], "latents $[B,L,D]$",
            ha="left", va="center", fontsize=5.4, color=POOL)

    # The drawn stack is a schematic: it uses fewer rows than the shipped
    # default (L=16), so say so next to the top of the stack.
    ax.text(x + w + 1.4, top + 0.2,
            "rows illustrative\n(default $L{=}16$)",
            ha="left", va="center", fontsize=4.8, color=MUTED, style="italic",
            linespacing=1.25, zorder=7,
            bbox=dict(boxstyle="round,pad=0.16", facecolor="white",
                      edgecolor="none", alpha=0.9))

    # the fixed-length / not-tok_out message, below the stack
    ax.text(x + w / 2, 21.6,
            "$enc\\_all = latents$: fixed "
            "$[B\\cdot L,D]$\n"
            "length independent of $C$ and of $N$; raw $tok\\_out\\,[K_f,D]$ "
            "is NOT in the KV",
            ha="center", va="center", fontsize=5.3, color=MUTED,
            linespacing=1.35,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                      edgecolor=MUTED, linewidth=0.8))

    return {"right": x + w + 0.5, **centers}


def draw_decoder_targets(ax):
    """Decoder: [Kt, D] queries -> flat [Σ p²] predictions (VIOLET)."""
    x, w, h, pitch = TGT_X, TGT_W, 1.9, 2.2
    n = 4
    y0 = YMID - ((n - 1) * pitch + h) / 2.0
    top = y0 + (n - 1) * pitch + h

    # -- target queries ----------------------------------------------------- #
    for i in range(n):
        rrect(ax, x, y0 + i * pitch, w, h, edge=VIOLET, face=tint(VIOLET, 0.82),
              lw=1.2, rounding=0.3, z=3, crossed=True)
    sq_bracket(ax, x - 0.7, y0, top, color=MUTED, lw=1.1, tick=0.55)
    axis_text(ax, x - 1.7, y0 + 0.22 * (top - y0), "$K_t$", rotation=90)
    h_bracket(ax, x, x + w, top + 0.6, color=MUTED, down=True)
    axis_text(ax, x + w / 2, top + 1.6, "$D$", size=7.5)
    ax.text(x + w / 2, y0 - 1.1, "targets", ha="center", va="center",
            fontsize=7.8, fontweight="bold", color=INK)
    ax.text(x + w / 2, y0 - 2.9,
            "mask token $+$ metric fractional PE\n"
            "$+$ target_proj$(z_c + scale)$",
            ha="center", va="center", fontsize=5.2, color=VIOLET,
            linespacing=1.3)

    # -- arrow queries -> predictions --------------------------------------- #
    arrow(ax, (x + w + 0.4, YMID), (PRED_X - 0.4, YMID), color=VIOLET, lw=1.4)

    # -- predictions -------------------------------------------------------- #
    px, pw = PRED_X, PRED_W
    for i in range(n):
        rrect(ax, px, y0 + i * pitch, pw, h, edge=VIOLET,
              face=tint(VIOLET, 0.82), lw=1.2, rounding=0.3, z=3, crossed=True)
    sq_bracket_mirror(ax, px + pw + 0.7, y0, top, color=MUTED, lw=1.1, tick=0.55)
    axis_text(ax, px + pw + 1.9, (y0 + top) / 2, "$K_t$", rotation=90)
    h_bracket(ax, px, px + pw, top + 0.6, color=MUTED, down=True)
    axis_text(ax, px + pw / 2, top + 1.6, "$\\Sigma\\,p^2$", size=7.5)
    ax.text(px + pw / 2, y0 - 1.1, "predictions", ha="center", va="center",
            fontsize=7.8, fontweight="bold", color=INK)
    ax.text(px + pw / 2, y0 - 3.3,
            "head per_p: exact head per configured $p$\n"
            "unseen $p$: nearest $p_{head}{\\geq}p$, no upsample\n"
            "output flat $[\\Sigma\\,p^2]$",
            ha="center", va="center", fontsize=4.9, color=VIOLET,
            linespacing=1.3)


def draw_ghost_padded(ax):
    """The ghost [B, C_max, N, D] tensor that is never allocated/built."""
    padded_ghost(ax, 2.2, 2.2, 12.8, 7.6,
                 shape="$[B,\\,C_{\\max},\\,N,\\,D]$",
                 note="", color=MUTED, z=1)
    # Explicit definition: this is a dashed *padded* tensor, not a real buffer.
    ax.text(3.0, 3.9,
            "ghost: a padded $[B,C_{\\max},N,D]$\n"
            "tensor, never allocated or built",
            ha="left", va="center", fontsize=5.5, color=MUTED, style="italic",
            linespacing=1.35, zorder=6,
            bbox=dict(boxstyle="round,pad=0.22", facecolor="white",
                      edgecolor="none", alpha=0.92))


def draw_glossary(ax):
    """Compact three-column symbol glossary; every figure symbol is here."""
    ax.text(70.0, 17.4, "symbols / glossary", ha="left", va="center",
            fontsize=6.6, fontweight="bold", color=INK)
    col1 = [
        ("$B$", "batch size"),
        ("$C_b$ / $C$", "channels, sample $b$ / count"),
        ("$\\Sigma C$", "total channels in batch"),
        ("$N_b$", "patches / channel, sample $b$"),
        ("$N$", "$\\max_b N_b$ (fused grid)"),
        ("$H,\\,W$", "image side (square)"),
        ("$n$", "patch index"),
        ("$V$", "visible tokens"),
        ("$D$", "token / embedding dim"),
    ]
    col2 = [
        ("$L$", "latents / sample ($n_{latents}$)"),
        ("$K_f$", "present $(b,n)$ locations"),
        ("$K_t$", "decoder targets (masked)"),
        ("$P$", "canonical patch (kernel)"),
        ("$p$", "configured patch size"),
        ("$p_b$", "runtime patch, $\\max(1,\\lfloor t_{um}/\\mu{+}0.5\\rfloor)$"),
        ("pixel_size", "$\\mu$m/px (required)"),
        ("$D_{attn}$", "fusion attn dim ($=3D$)"),
    ]
    col3 = [
        ("$enc\\_all$", "decoder KV $=$ latents $[B,L,D]$"),
        ("PMA", "multi-head attn pooling ($L$ queries)"),
        ("RoPE", "2-D axial rotary PE"),
        ("fractional PE", "metric decoder PE"),
        ("$z_c$", "channel identity vector"),
        ("$channel\\_idx$", "per-token channel position"),
        ("PI-resize", "FlexiViT kernel resize"),
        ("DCP", "row-omitting mask sampler"),
    ]
    pitch = 1.42
    for col_x, entries in ((70.0, col1), (84.0, col2), (98.0, col3)):
        for i, (sym, meaning) in enumerate(entries):
            y = 15.9 - i * pitch
            ax.text(col_x, y, sym, ha="left", va="center", fontsize=5.2,
                    color=INK, fontweight="bold")
            ax.text(col_x + 4.7, y, meaning, ha="left", va="center",
                    fontsize=5.0, color=MUTED)


def draw_legends(ax):
    """Fill the empty bottom band with role + channel legends and a note."""
    legend(
        ax, 27.0, 16.3,
        [("tokenizer patch embed", GREEN, "fill"),
         ("channel fusion", FUSION, "fill"),
         ("encoder (TransformerEncoder)", ENCODER, "fill"),
         ("fused vectors $[K_f,D]$", FUSED, "cross"),
         ("latent pool", POOL, "cross"),
         ("decoder / targets", VIOLET, "fill"),
         ("omitted row (masked)", LOSS, "omit")],
        title="roles & data tensors", pitch=1.62, sw=1.05, fontsize=6.0,
        title_size=6.6, label_gap=0.95,
    )
    channel_legend(
        ax, 47.0, 16.3, n=len(CHANNEL_COLORS), title="channel identity",
        labels=[f"channel {i}" for i in range(len(CHANNEL_COLORS))],
        pitch=1.62, sw=1.05, fontsize=6.0, title_size=6.6, label_gap=0.95,
    )

    ax.text(48.0, 20.3,
            "Channel count $C$ lives only in the input length $\\sum_b C_b{\\cdot}N_b$.\n"
            "Fusion pools the visible channel tokens at each present $(b,n)$ with one\n"
            "shared learned query: $O(C{\\cdot}D{\\cdot}D_{attn})+O(D{\\cdot}D_{attn})$, "
            "$D_{attn}=3D$.\n"
            "Channel identity $z_c$ is a per-head routing bias and an attention-pooled\n"
            "skip, never added to tokens. No fusion PE; fusion returns $[K_f,D]$ directly,\n"
            "so no $[B,N,D]$ buffer is built. Vector size is fixed, independent of $C,N$.",
            ha="center", va="center", fontsize=5.9, color=MUTED, style="italic",
            linespacing=1.35,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                      edgecolor=MUTED, linewidth=0.9))


# --------------------------------------------------------------------------- #
# Main figure
# --------------------------------------------------------------------------- #
def build() -> plt.Figure:
    fig = plt.figure(figsize=(18.3, 7.5))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_xlim(0, 122)
    ax.set_ylim(0, 50)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    # ---- title ------------------------------------------------------------ #
    ax.text(1, 47.8, "L2", ha="left", va="center", fontsize=15,
            fontweight="bold", color=INK)
    ax.text(5.5, 47.8,
            "Tensor layout: literal array stacks with axis labels (Perceiver-IO Fig 2 idiom)",
            ha="left", va="center", fontsize=12.2, fontweight="bold", color=INK)
    ax.text(1, 45.0,
            "The fused vector is independent of the channel count $C$; the decoder KV length $B\\cdot L$ is fixed and independent of $C$ and $N$. "
            "Omitted rows are absent from the token list (dashed); the token list is the only token container.",
            ha="left", va="center", fontsize=7.0, color=MUTED, style="italic")

    # ---- array stacks ----------------------------------------------------- #
    draw_input_images(ax)
    draw_token_list(ax)
    draw_omitted_callout(ax)
    draw_channel_grouping(ax)
    draw_fusion(ax)
    draw_fused_grid(ax)
    enc = draw_encoder(ax)
    kv = draw_kv(ax)
    draw_decoder_targets(ax)
    draw_ghost_padded(ax)
    draw_legends(ax)
    draw_glossary(ax)

    # ---- main dataflow arrows (all on the shared centre line) ------------- #
    arrow(ax, (6.9, YMID), (11.1, YMID), lw=1.5, color=GREEN)
    edge_label(ax, 9.0, YMID + 1.4, "embed", size=6.5, color=GREEN)

    # grouping / reindexing operation: same tokens, regrouped by (b, n)
    arrow(ax, (21.0, YMID), (25.1, YMID), lw=1.5)
    ax.text(22.6, YMID + 2.9,
            "group by\n$(b,n)$\nlocation\n(same tokens)",
            ha="center", va="center", fontsize=5.8, color=INK,
            linespacing=1.25, zorder=7,
            bbox=dict(boxstyle="round,pad=0.16", facecolor="white",
                      edgecolor="none", alpha=0.9))

    # fused grid -> encoder
    arrow(ax, (56.2, YMID), (61.0, YMID), lw=1.5)
    edge_label(ax, 58.6, YMID + 1.4, "enc", size=6.5)

    # ---- encoder -> decoder KV -------------------------------------------- #
    # tok_out is pooled (PMA) into the latent set, which is the whole decoder KV.
    arrow(ax, (enc["right"], enc["content_c"]),
          (KV_X - 1.1, kv["latents"]), color=POOL, lw=1.3)
    ax.text(74.6, 24.6, "PMA($L$)", ha="center", va="center", fontsize=5.0,
            color=POOL)

    # ---- KV -> decoder ---------------------------------------------------- #
    arrow(ax, (kv["right"], YMID), (TGT_X - 0.4, YMID), lw=1.5, color=VIOLET)
    edge_label(ax, (kv["right"] + TGT_X) / 2, YMID + 1.4, "dec", size=6.5,
               color=VIOLET)

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
