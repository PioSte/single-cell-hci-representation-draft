"""L6 -- MAE objective: encoder-decoder asymmetry and channel-aware reconstruction.

Shows the two-branch objective with:
* Encoder branch: the tokenizer emits the ragged visible channel-patches
  [V, D] (masked rows are omitted, never zeroed). Channel fusion pools each
  present (b, n) location and ``enc_tokens = fused[present]`` selects the
  [Kf, D] channel-agnostic fused location tokens fed to the encoder
  (``Kf <= V``). A PMA pooler over ``tok_out`` (content tokens only) produces
  the trained descriptor [B, D].
* Decoder KV: ``enc_all = [mem_out; tok_out; desc]`` -> [B*M + Kf + B, D].
* Decoder branch: target queries conditioned on the target channel embedding;
  each block runs target self-attention over a sample's queries, then
  cross-attention to ``enc_all``.
* One shared decoder for all channels.
* Asymmetry: V visible channel-patches + Kt masked channel-patches = ΣC·N,
  with Kf <= V.
* Default loss: per-patch-standardised MSE.

Usage::

    uv run --extra viz python viz/level6_objective.py
    uv run --extra viz python viz/level6_objective.py --formats pdf svg png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from figure_style import (  # noqa: E402
    BLUE, GREEN, ORANGE, RED, PURPLE, VIOLET, INK, MUTED, CHANNEL_COLORS,
    TOKENIZER, FUSION, ENCODER, DECODER, MEMORY, LOSS, FUSED, POOL, GHOST,
    tint, rrect, box, arrow, edge_label, sq_bracket_mirror, token_rows,
    channel_color, install_rcparams, save_figure, legend, channel_legend,
)

OUT = Path(__file__).resolve().parent / "out"

# --------------------------------------------------------------------------- #
# One shared row grid for every array stack, so the columns line up.
# --------------------------------------------------------------------------- #
PITCH = 3.8
H = 2.9

# Frame geometry (equal width, one gap, shared top and bottom).
FRAME_X0, FRAME_X1 = 2.5, 74.5
FRAME_Y0, FRAME_Y1 = 28.0, 80.0
DFRAME_X0, DFRAME_X1 = 79.5, 151.5


# --------------------------------------------------------------------------- #
# Array stacks
# --------------------------------------------------------------------------- #
def draw_channel_patches(ax, x=6.5, y0=47.5, w=9.0):
    """Ragged visible channel-patches [V, D] with the DCP-omitted row below.

    Row 0 (bottom) is the omitted channel-patch -- dashed, LOSS red, outside
    the [V, D] bracket (absent, never zeroed). Rows 1-4 are the visible
    channel-patches grouped by the bracket.
    """
    specs = [
        (LOSS, True),               # omitted channel-patch (bottom, outside V)
        (channel_color(0), False),
        (channel_color(1), False),
        (channel_color(2), False),
        (channel_color(3), False),
    ]
    token_rows(ax, x, y0, w, H, PITCH, specs)
    vis_bot = y0 + 1 * PITCH
    vis_top = y0 + 4 * PITCH + H
    sq_bracket_mirror(ax, x + w + 1.0, vis_bot, vis_top)
    ax.text(x + w / 2, vis_top + 3.6, "$[V, D]$ visible",
            ha="center", va="center", fontsize=7.4, color=INK,
            fontweight="bold")
    ax.text(x + w / 2, vis_top + 2.0, "channel-patches",
            ha="center", va="center", fontsize=6.0, color=MUTED)
    # DCP omission sits below the dashed row, fully outside the [V, D] bracket.
    ax.text(x + w / 2, y0 + H / 2 - 3.2,
            "DCP omission\n(from $\\Sigma C\\cdot N$)",
            ha="center", va="center", fontsize=6.0, color=LOSS,
            fontweight="bold", linespacing=1.2)


def draw_fused_locations(ax, x=23.0, y0=51.3, w=9.0):
    """Encoder input: [Kf, D] channel-agnostic fused location tokens (FUSED)."""
    specs = [(FUSED, False)] * 4
    top = token_rows(ax, x, y0, w, H, PITCH, specs)
    sq_bracket_mirror(ax, x + w + 0.5, y0, top)
    ax.text(x + w / 2, top + 2.4, "$[K_f, D]$ fused locations",
            ha="center", va="center", fontsize=7.4, color=INK,
            fontweight="bold")
    return top


def draw_memory(ax, x=6.5, y0=35.5, w=9.0):
    """Fixed memory tokens [B·M, D] (MEMORY gold)."""
    specs = [(MEMORY, False), (MEMORY, False)]
    top = token_rows(ax, x, y0, w, H, PITCH, specs)
    sq_bracket_mirror(ax, x + w + 1.0, y0, top)
    ax.text(x + w / 2, y0 - 2.7, "$[B{\\cdot}M, D]$ memory", ha="center",
            va="center", fontsize=7.4, color=INK, fontweight="bold")
    return top


def draw_enc_all(ax, x=54.5, y0=41.0, w=10.0):
    """Decoder KV: ``enc_all = cat([mem_out, tok_out, desc])``.

    ``token_rows`` draws bottom-up, so the specs are listed bottom-to-top to
    give the same top-to-bottom order as the code: memory-out on top, then the
    visible fused ``tok_out``, then the descriptor at the bottom.
    """
    specs = [
        (POOL, False),                                                  # desc [B, D]
        (FUSED, False), (FUSED, False), (FUSED, False), (FUSED, False),  # tok_out [Kf, D]
        (MEMORY, False), (MEMORY, False),                               # mem_out [B·M, D]
    ]
    top = token_rows(ax, x, y0, w, H, PITCH, specs)
    sq_bracket_mirror(ax, x + w + 1.0, y0, top)
    ax.text(x + w / 2, top + 2.4, "$[B{\\cdot}M + K_f + B, D]$", ha="center",
            va="center", fontsize=7.2, color=INK, fontweight="bold")
    # Row-group labels on the right, centred on their actual row bands.
    # Top band = memory-out, middle = visible fused tokens, bottom = descriptor.
    ax.text(x + w + 1.2, y0 + 5.5 * PITCH + H / 2, "mem ($M$)", ha="left",
            va="center", fontsize=6.1, color=MEMORY, zorder=5)
    ax.text(x + w + 1.2, y0 + 2.5 * PITCH + H / 2, "vis ($K_f$)", ha="left",
            va="center", fontsize=6.1, color=FUSED, zorder=5)
    ax.text(x + w + 1.2, y0 + 0.5 * PITCH + H / 2, "desc ($B$)", ha="left",
            va="center", fontsize=6.1, color=POOL, zorder=5)
    # Descriptor provenance: the pooler reads content tokens only.
    ax.text(x + w / 2, y0 - 3.0,
            "desc $=$ PMA pooler($\\mathrm{tok}_{out}$), content tokens only",
            ha="center", va="center", fontsize=6.1, color=POOL, style="italic")
    return top


def draw_target_queries(ax, x=84.0, y0=51.0, w=12.0):
    """Decoder target queries [Kt, D], conditioned on target channel identity.

    Queries are the shared fused/neutral objects (FUSED blue); the target
    channel identity enters additively through ``target_proj``.
    """
    specs = [(FUSED, False)] * 3
    top = token_rows(ax, x, y0, w, H, PITCH, specs)
    sq_bracket_mirror(ax, x + w + 1.0, y0, top)
    ax.text(x + w / 2, top + 2.4, "$[K_t, D]$ queries", ha="center", va="center",
            fontsize=7.6, color=INK, fontweight="bold")
    # Query recipe (decoder.py): mask + fracPE + target_proj(chan_emb + scale_emb)
    ax.text(x + w / 2, top + 6.0,
            r"$q = \mathrm{mask} + \mathrm{fracPE}(r,c,g_h,g_w)$" "\n"
            r"$+\, \mathrm{target\_proj}(\mathrm{chan\_emb} + \mathrm{scale\_emb})$",
            ha="center", va="center", fontsize=6.2, color=MUTED,
            style="italic", linespacing=1.3)
    return top


def draw_predictions(ax, x=123.0, y0=48.0, w=15.0):
    """Decoder predictions [Kt, P²] -- DECODER violet."""
    specs = [(DECODER, False)] * 3
    top = token_rows(ax, x, y0, w, H, PITCH, specs)
    sq_bracket_mirror(ax, x + w + 1.0, y0, top)
    ax.text(x + w / 2, top + 2.4, "$[K_t, p^2]$ predictions", ha="center",
            va="center", fontsize=6.9, color=INK, fontweight="bold")
    return top


def draw_targets(ax, x=123.0, y0=31.0, w=15.0):
    """Raw pixel targets [Kt, P²] for the masked channel-patches.

    Targets are per-channel -> identity rows use CHANNEL_COLORS.
    """
    specs = [
        (channel_color(2), False),
        (channel_color(3), False),
        (channel_color(0), False),
    ]
    top = token_rows(ax, x, y0, w, H, PITCH, specs)
    sq_bracket_mirror(ax, x + w + 1.0, y0, top)
    ax.text(x + w / 2, top + 1.8, "$[K_t, p^2]$ targets", ha="center",
            va="center", fontsize=7.6, color=INK, fontweight="bold")
    return top


# --------------------------------------------------------------------------- #
# The figure
# --------------------------------------------------------------------------- #
def build() -> plt.Figure:
    fig = plt.figure(figsize=(13.0, 8.0))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_xlim(0, 156)
    ax.set_ylim(0, 88)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    # ---- title ------------------------------------------------------------ #
    ax.text(2, 84.0, "L6", ha="left", va="center", fontsize=15,
            fontweight="bold", color=INK)
    ax.text(6.5, 84.0,
            "MAE objective: encoder-decoder asymmetry & channel-aware reconstruction",
            ha="left", va="center", fontsize=12.2, fontweight="bold", color=INK)

    # ---- two equal-width branch frames, shared top ------------------------ #
    rrect(ax, FRAME_X0, FRAME_Y0, FRAME_X1 - FRAME_X0, FRAME_Y1 - FRAME_Y0,
          edge=ENCODER, face=tint(ENCODER, 0.96), lw=1.5, dashed=True,
          rounding=1.0, z=1)
    ax.text(3.5, 78.0, "Encoder branch (sees only visible tokens)",
            ha="left", va="center", fontsize=9.5, fontweight="bold",
            color=ENCODER)

    rrect(ax, DFRAME_X0, FRAME_Y0, DFRAME_X1 - DFRAME_X0, FRAME_Y1 - FRAME_Y0,
          edge=DECODER, face=tint(DECODER, 0.96), lw=1.5, dashed=True,
          rounding=1.0, z=1)
    ax.text(80.5, 78.0, "Decoder branch (channel-conditioned reconstruction)",
            ha="left", va="center", fontsize=9.5, fontweight="bold",
            color=DECODER)

    # ---- encoder pathway -------------------------------------------------- #
    draw_channel_patches(ax, x=5.0, y0=47.5)
    draw_memory(ax, x=5.0, y0=35.5)
    draw_fused_locations(ax, x=23.2, y0=51.3)
    draw_enc_all(ax, x=57.0, y0=41.0)

    # Channel fusion: pool the [V, D] ragged list per (b, n) -> [Kf, D] fused.
    # Readable stage label sits in the open corridor between the two stacks,
    # just below the orange arrow (replaces the old unreadable tiny box).
    arrow(ax, (15.5, 58.5), (23.2, 58.5), color=FUSION, lw=1.5)
    ax.text(18.5, 57.7, "Channel fusion\n1 query / $(b,n)$\nno PE",
            ha="center", va="top", fontsize=5.3, color=FUSION,
            fontweight="bold", linespacing=1.3, zorder=7,
            bbox=dict(boxstyle="round,pad=0.1", facecolor="white",
                      edgecolor=FUSION, linewidth=0.7, alpha=0.95))

    # Encoder module (TransformerEncoder) -- ENCODER role colour.
    box(ax, 34.0, 44.0, 21.0, 22.0, "TransformerEncoder",
        "grouped self-attn $\\times L_e$\n"
        "2-D axial RoPE on norm $(r,c)$\n"
        "mem 0 rot $\\cdot$ $+M$ memory\n"
        "add scale Fourier($\\log p$, $\\log\\,\\mu$m/px)",
        edge=ENCODER, sub_size=6.0)

    # Visible fused tokens + memory feed the encoder; encoder -> enc_all.
    arrow(ax, (32.8, 53.5), (34.0, 53.5), color=FUSED, lw=1.5)
    arrow(ax, (14.0, 42.0), (34.0, 48.0), color=MEMORY, lw=1.5)
    arrow(ax, (55.0, 53.85), (57.0, 53.85), color=ENCODER, lw=1.5)

    # ---- decoder pathway -------------------------------------------------- #
    draw_target_queries(ax, x=84.0, y0=51.0)

    box(ax, 101.0, 45.0, 17.0, 22.0, "Decoder",
        "target self-attn\n$\\rightarrow$ grouped cross-attn\n$\\times L_d$ blocks",
        edge=DECODER, sub_size=6.1)

    # queries -> decoder
    arrow(ax, (97.8, 56.0), (101.0, 56.0), color=DECODER, lw=1.5)

    # enc_all -> decoder cross-attention bus (keys = values = enc_all).
    ax.plot([68.0, 98.0], [43.0, 43.0], color=ENCODER, lw=1.2,
            linestyle="--", zorder=4)
    ax.plot([98.0, 98.0], [43.0, 48.0], color=ENCODER, lw=1.2,
            linestyle="--", zorder=4)
    arrow(ax, (98.0, 48.0), (101.0, 48.0), color=ENCODER, lw=1.2, dashed=True)
    ax.text(76.0, 44.7, "keys / values: mem + tokens + desc",
            ha="left", va="center", fontsize=6.2, color=ENCODER, zorder=7,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                      edgecolor="none", alpha=0.9))

    # decoder -> head -> predictions (routed clear of the labels).
    box(ax, 120.0, 64.0, 26.0, 11.0, "Head",
        "$\\mathrm{Linear}(D, p^2)$ per configured $p$\n"
        "$\\rightarrow$ $[K_t, p^2]$ (nearest-head fallback)",
        edge=DECODER, title_size=7.6, sub_size=6.0)
    arrow(ax, (118.0, 65.5), (120.0, 65.5), color=DECODER, lw=1.5)
    # show the resize step on the short head->predictions arrow
    arrow(ax, (130.5, 64.0), (130.5, 62.3), color=DECODER, lw=1.5)
    ax.text(131.5, 63.15, "resize", ha="left", va="center", fontsize=6.0,
            color=DECODER, zorder=7,
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                      edgecolor="none", alpha=0.9))

    draw_predictions(ax, x=123.0, y0=48.0)
    ax.text(130.5, 46.5,
            "mixed $p$: padded $[K_t, \\max p^2]$ + validity mask",
            ha="center", va="center", fontsize=6.0, color=MUTED,
            style="italic")

    # ---- targets (raw pixels) -------------------------------------------- #
    draw_targets(ax, x=123.0, y0=31.0)

    # Short, light route: targets are the raw pixels of the masked patches.
    ax.plot([DFRAME_X0], [36.25], marker="o", ms=3.0, color=GHOST, zorder=4)
    arrow(ax, (DFRAME_X0, 36.25), (123.0, 36.25), color=GHOST, lw=1.0, dashed=True)
    ax.text(101.0, 39.0, "target = raw pixels of masked channel-patches",
            ha="center", va="center", fontsize=6.8, color=MUTED, style="italic")

    # ---- loss ------------------------------------------------------------- #
    box(ax, 123.0, 11.0, 19.0, 11.0, "Loss",
        "per-patch-standardised MSE",
        edge=LOSS, title_size=7.6, sub_size=6.0)

    # predictions -> loss (right), targets -> loss (top), merged at the box.
    ax.plot([139.0, 149.0, 149.0], [53.0, 53.0, 17.5], color=DECODER, lw=1.3,
            zorder=4)
    arrow(ax, (149.0, 17.5), (142.0, 17.5), color=LOSS, lw=1.5)
    ax.plot([130.5, 130.5], [31.0, 24.5], color=GHOST, lw=1.3, linestyle="--",
            zorder=4)
    arrow(ax, (130.5, 24.5), (130.5, 22.0), color=LOSS, lw=1.5)
    ax.text(146.0, 55.3, "pred", ha="center", va="center", fontsize=6.6,
            color=DECODER)
    ax.text(132.0, 27.0, "tgt", ha="left", va="center", fontsize=6.6,
            color=MUTED)

    # ---- role + channel legends (empty lower-left) ------------------------ #
    # Module roles stay plain; data tensors use crossed swatches; omitted rows
    # keep the dashed-white swatch (never zeroed).
    legend(
        ax, 3.0, 27.0,
        [
            ("channel fusion", FUSION, "fill"),
            ("encoder (TransformerEncoder)", ENCODER, "fill"),
            ("channel-aware decoder", DECODER, "fill"),
            ("memory tokens", MEMORY, "cross"),
            ("objective / omitted rows", LOSS, "omit"),
            ("fused tokens", FUSED, "cross"),
            ("patch-token descriptor", POOL, "cross"),
        ],
        title="module roles", pitch=1.9, sw=1.4, fontsize=6.0,
    )
    channel_legend(
        ax, 39.0, 25.5, n=len(CHANNEL_COLORS), title="channel identity",
        pitch=2.2, sw=1.4, fontsize=6.2,
    )

    # ---- compact symbol glossary (empty lower-right area) ---------------- #
    rrect(ax, 82.0, 9.0, 33.0, 17.0, edge=GHOST, face=tint(GHOST, 0.95),
          lw=1.1, dashed=True, rounding=0.5, z=2)
    ax.text(83.0, 24.6, "symbols / glossary", ha="left", va="center",
            fontsize=6.4, fontweight="bold", color=INK, zorder=4)
    ax.text(83.0, 22.2,
            "B  batch size\n"
            "C  channels\n"
            "$\\Sigma C$ total channels\n"
            "N  patches / channel\n"
            "M  memory tokens\n"
            "V  visible patches\n"
            "$K_f$ fused locations\n"
            "$K_t$ masked targets\n"
            "D  embedding dim",
            ha="left", va="top", fontsize=6.0, color=INK,
            linespacing=1.35, zorder=4)
    ax.text(97.0, 22.2,
            "p  patch size\n"
            "c  canonical patch size\n"
            "PMA pooled descriptor\n"
            "RoPE 2-D axial rotation\n"
            "DCP visibility sampler\n"
            "$L_e$ encoder blocks\n"
            "$L_d$ decoder blocks\n"
            "s  scale embedding",
            ha="left", va="top", fontsize=6.0, color=INK,
            linespacing=1.35, zorder=4)

    # ---- key points ------------------------------------------------------- #
    rrect(ax, FRAME_X0, 2.0, FRAME_X1 - FRAME_X0, 5.0, edge=ENCODER,
          face=tint(ENCODER, 0.88), lw=1.3, rounding=0.5, z=2)
    ax.text((FRAME_X0 + FRAME_X1) / 2, 4.5,
            r"Asymmetry: encoder sees $K_f \leq V$ fused locations; "
            r"decoder reconstructs $K_t$ masked channel-patches;  $V + K_t = \Sigma C \cdot N$",
            ha="center", va="center", fontsize=6.4, color=INK,
            fontweight="bold")

    rrect(ax, DFRAME_X0, 2.0, DFRAME_X1 - DFRAME_X0, 5.0, edge=DECODER,
          face=tint(DECODER, 0.88), lw=1.3, rounding=0.5, z=2)
    ax.text((DFRAME_X0 + DFRAME_X1) / 2, 4.5,
            r"One shared decoder: target self-attn, then cross-attn to mem/tokens/descriptor",
            ha="center", va="center", fontsize=7.6, color=INK,
            fontweight="bold")

    return fig


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--formats", nargs="+", default=["pdf", "svg", "png"])
    ap.add_argument("--stem", default="level6_objective")
    args = ap.parse_args()

    install_rcparams()
    fig = build()
    save_figure(fig, OUT, args.stem, args.formats)
    plt.close(fig)


if __name__ == "__main__":
    main()
