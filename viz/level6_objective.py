"""L6 -- MAE objective: encoder-decoder asymmetry and channel-aware reconstruction.

Shows the two-branch objective with:
* Encoder branch: the tokenizer emits the visible channel-patches
  [V, D] (masked rows are omitted, never zeroed). Channel fusion pools the
  channels at each present location and returns those rows as
  ``enc_tokens`` directly: [Kf, D] channel-agnostic location tokens, Kf <= V.
  A PMA pooler over ``tok_out`` (content tokens only) produces the fixed
  latent set [B, L, D] (``L = n_latents``) by spatial pooling to fixed size
  latent.
* Decoder KV: ``enc_all = latents`` -> [B*L, D] (the ``memory`` arm would
  prepend ``mem_out``). ``tok_out`` itself is *not* in the KV.
* Decoder branch: target queries conditioned on the target channel embedding;
  each block runs target self-attention over a sample's queries, then
  cross-attention to ``enc_all``.
* One shared decoder for all channels.
* Asymmetry: V visible channel-patches + Kt masked channel-patches <=
  Sigma_b C_b·N_b (equality without the off-cell mask); the encoder sees Kf
  fused locations.
* Default loss: standardise each target patch (mean/std), then MSE, averaged
  with ``loss_reduce="sample"`` (one vote per crop), masked rows only.
* The head emits the flat [Sigma p^2] pixel array (row i contiguous p_i^2);
  a mixed-p batch needs no padded block and no validity mask.

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
    TOKENIZER, FUSION, ENCODER, DECODER, LOSS, FUSED, POOL, GHOST,
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
    """Visible channel-patches [V, D] with the DCP-omitted row below.

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
            "DCP omission\n(from $\\Sigma_b C_b{\\cdot}N_b$)",
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


def draw_enc_all(ax, x=57.0, y0=41.0, w=10.0):
    """Decoder KV: the fixed latent set from spatial pooling to fixed size latent.

    The ``memory`` arm would prepend ``[B·M, D]`` ``mem_out`` rows on top;
    ``tok_out`` is not part of this KV.
    """
    specs = [(POOL, False)] * 6                                      # latents [B, L, D]
    top = token_rows(ax, x, y0, w, H, PITCH, specs)
    sq_bracket_mirror(ax, x + w + 1.0, y0, top)
    ax.text(x + w / 2, top + 2.4, "$[B{\\cdot}L,\\,D]$",
            ha="center", va="center", fontsize=6.9, color=INK,
            fontweight="bold")
    # Row-group label on the right, centred on the latent band.
    ax.text(x + w + 1.2, y0 + 2.5 * PITCH + H / 2, "latents ($L$)", ha="left",
            va="center", fontsize=6.1, color=POOL, zorder=5)
    return top


def draw_target_queries(ax, x=84.0, y0=51.0, w=12.0):
    """Decoder target queries [Kt, D], conditioned on target channel identity.

    Queries are decoder-side objects (DECODER violet); the target channel
    identity ``z_c`` enters additively through ``target_proj``.
    """
    specs = [(DECODER, False)] * 3
    top = token_rows(ax, x, y0, w, H, PITCH, specs)
    sq_bracket_mirror(ax, x + w + 1.0, y0, top)
    ax.text(x + w / 2, top + 2.4, "$[K_t, D]$ queries", ha="center", va="center",
            fontsize=7.6, color=INK, fontweight="bold")
    # Query recipe (decoder.py): mask + fracPE + target_proj(z_c + scale Fourier)
    ax.text(x + w / 2, top + 6.0,
            r"$q = \mathrm{mask} + \mathrm{fracPE}(\mathrm{target})$" "\n"
            r"$+\, \mathrm{target\_proj}(z_c + \mathrm{Fourier}(p,\mu))$",
            ha="center", va="center", fontsize=6.2, color=MUTED,
            style="italic", linespacing=1.3)
    return top


def draw_predictions(ax, x=123.0, y0=48.0, w=15.0):
    """Decoder predictions: flat [Sigma p^2] pixels -- DECODER violet."""
    specs = [(DECODER, False)] * 3
    top = token_rows(ax, x, y0, w, H, PITCH, specs)
    sq_bracket_mirror(ax, x + w + 1.0, y0, top)
    ax.text(x + w / 2, top + 2.4, "$[\\Sigma p^2]$ flat predictions",
            ha="center", va="center", fontsize=6.9, color=INK,
            fontweight="bold")
    ax.text(x + w / 2, y0 - 2.1,
            "row $i$ = $p_i^2$ contiguous pixels",
            ha="center", va="center", fontsize=6.0, color=MUTED,
            style="italic")
    return top


def draw_targets(ax, x=123.0, y0=31.0, w=15.0):
    """Raw pixel targets: flat [Sigma p^2] pixels -- DECODER violet."""
    specs = [(DECODER, False)] * 3
    top = token_rows(ax, x, y0, w, H, PITCH, specs)
    sq_bracket_mirror(ax, x + w + 1.0, y0, top)
    ax.text(x + w / 2, top + 1.8, "$[\\Sigma p^2]$ flat targets", ha="center",
            va="center", fontsize=7.0, color=INK, fontweight="bold")
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
    draw_fused_locations(ax, x=23.2, y0=51.3)
    draw_enc_all(ax, x=56.2, y0=41.0, w=9.0)

    # Channel fusion: pool the [V, D] token list per location -> [Kf, D] fused.
    # Compact stage label in the corridor between the two stacks.
    arrow(ax, (15.8, 58.5), (23.2, 58.5), color=FUSION, lw=1.5)
    ax.text(18.5, 57.3, "channel fusion\n1 query / location\nno PE",
            ha="center", va="top", fontsize=4.5, color=FUSION,
            fontweight="bold", linespacing=1.3, zorder=7)

    # Encoder module (TransformerEncoder) -- ENCODER role colour.
    box(ax, 34.0, 44.0, 21.0, 22.0, "TransformerEncoder",
        "grouped self-attn $\\times L_e$\n"
        "2-D axial RoPE on metric\n"
        "token centres $(i{+}0.5)\\cdot p\\cdot\\mu$\n"
        "no memory tokens (default)\n"
        "add scale Fourier($\\log p,\\log\\mu$)",
        edge=ENCODER, sub_size=5.7)

    # Visible fused tokens feed the encoder; encoder -> enc_all.
    arrow(ax, (32.8, 53.5), (34.0, 53.5), color=FUSED, lw=1.5)
    arrow(ax, (55.0, 53.85), (56.2, 53.85), color=ENCODER, lw=1.5)

    # Pooled-latent provenance, inside the encoder frame.
    ax.text(57.0, 39.6,
            "latents $=$ PMA pool($\\mathrm{tok}_{out}$); $L=n_{latents}=16$\n"
            "trained through the loss (KV $+$ reconstruction)\n"
            "$\\mathrm{tok}_{out}$: content tokens; not in the KV",
            ha="center", va="top", fontsize=5.5, color=MUTED,
            style="italic", linespacing=1.35, zorder=6)

    # ---- decoder pathway -------------------------------------------------- #
    draw_target_queries(ax, x=84.0, y0=51.0)

    box(ax, 101.0, 45.0, 17.0, 22.0, "Decoder",
        "target self-attn\n$\\rightarrow$ grouped cross-attn\n$\\times L_d$ blocks",
        edge=DECODER, sub_size=6.1)

    # queries -> decoder
    arrow(ax, (97.8, 56.0), (101.0, 56.0), color=DECODER, lw=1.5)

    # enc_all -> decoder cross-attention bus (keys = values = enc_all).
    ax.plot([66.2, 98.0], [43.0, 43.0], color=ENCODER, lw=1.2,
            linestyle="--", zorder=4)
    ax.plot([98.0, 98.0], [43.0, 48.0], color=ENCODER, lw=1.2,
            linestyle="--", zorder=4)
    arrow(ax, (98.0, 48.0), (101.0, 48.0), color=ENCODER, lw=1.2, dashed=True)
    # KV label clear of the bus line and of the decoder frame border.
    ax.text(81.5, 41.5, "keys / values: latents",
            ha="left", va="center", fontsize=6.0, color=ENCODER, zorder=7)

    # decoder -> head -> predictions.
    box(ax, 120.0, 64.0, 26.0, 11.0, "Head",
        "one $\\mathrm{Linear}(D,p^2)$ per configured $p$\n"
        "unseen $p$: nearest $p_{head}\\geq p$\n"
        "bilinear, never upsamples",
        edge=DECODER, title_size=7.6, sub_size=5.8)
    arrow(ax, (118.0, 65.5), (120.0, 65.5), color=DECODER, lw=1.5)
    # Configured p uses its exact per-p head; the resize path is only the
    # unseen-p fallback (drawn for completeness, never applied to configured p).
    arrow(ax, (130.5, 64.0), (130.5, 61.9), color=DECODER, lw=1.5, ms=10)
    ax.text(130.5, 63.05, "exact per-$p$ head", ha="center", va="center",
            fontsize=6.0, color=DECODER, zorder=7,
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                      edgecolor="none", alpha=0.95))

    draw_predictions(ax, x=123.0, y0=48.0)

    # ---- targets (raw pixels) -------------------------------------------- #
    draw_targets(ax, x=123.0, y0=31.0)

    # Short, light route: targets are the raw pixels of the masked patches.
    ax.plot([DFRAME_X0], [36.25], marker="o", ms=3.0, color=GHOST, zorder=4)
    arrow(ax, (DFRAME_X0, 36.25), (123.0, 36.25), color=GHOST, lw=1.0,
          dashed=True)
    ax.text(101.0, 39.0, "target = raw pixels of masked channel-patches",
            ha="center", va="center", fontsize=6.8, color=MUTED, style="italic")

    # ---- loss ------------------------------------------------------------- #
    box(ax, 122.0, 10.5, 23.0, 12.5, "Loss",
        "standardise each target patch $\\rightarrow$ MSE\n"
        "mean per sample (loss_reduce=sample)\n"
        "masked rows only",
        edge=LOSS, title_size=7.6, sub_size=5.8)

    # predictions -> loss (right), targets -> loss (top), merged at the box.
    ax.plot([139.0, 149.0, 149.0], [53.0, 53.0, 17.5], color=DECODER, lw=1.3,
            zorder=4)
    arrow(ax, (149.0, 17.5), (145.0, 17.5), color=LOSS, lw=1.5)
    ax.plot([130.5, 130.5], [31.0, 25.5], color=GHOST, lw=1.3, linestyle="--",
            zorder=4)
    arrow(ax, (130.5, 25.5), (130.5, 23.0), color=LOSS, lw=1.5)
    ax.text(146.0, 55.3, "pred", ha="center", va="center", fontsize=6.6,
            color=DECODER)
    ax.text(132.2, 26.0, "tgt", ha="left", va="center", fontsize=6.6,
            color=MUTED)

    # ---- role + channel legends (empty lower-left) ------------------------ #
    # Module roles stay plain; data tensors use crossed swatches; omitted rows
    # keep the dashed-white swatch (never zeroed). LOSS is the loss fill, the
    # dashed "omit" swatch is the absent row.
    legend(
        ax, 3.0, 27.0,
        [
            ("channel fusion ($D_{attn}=3D$)", FUSION, "fill"),
            ("encoder (TransformerEncoder)", ENCODER, "fill"),
            ("channel-aware decoder", DECODER, "fill"),
            ("reconstruction loss", LOSS, "fill"),
            ("omitted row (absent, never zeroed)", LOSS, "omit"),
            ("fused tokens $[K_f,D]$", FUSED, "cross"),
            ("pooled latents", POOL, "cross"),
        ],
        title="roles & row kinds", pitch=1.65, sw=1.4, fontsize=5.8,
    )
    channel_legend(
        ax, 39.0, 25.5, n=len(CHANNEL_COLORS), title="channel identity",
        pitch=2.0, sw=1.4, fontsize=6.2,
    )

    # ---- compact 3-column symbol glossary (empty lower-right area) -------- #
    rrect(ax, 77.5, 9.0, 43.5, 17.5, edge=GHOST, face=tint(GHOST, 0.95),
          lw=1.1, dashed=True, rounding=0.5, z=2)
    ax.text(78.5, 25.2, "symbols / glossary", ha="left", va="center",
            fontsize=6.4, fontweight="bold", color=INK, zorder=4)
    ax.text(78.0, 23.4,
            "$B$  batch\n"
            "$C_b$  channels of sample $b$\n"
            "$\\Sigma C$  total channels\n"
            "$N$  patches / channel\n"
            "$M$  memory tokens (arm)\n"
            "$L$  latents (pooled, 16)\n"
            "$V$  visible patches\n"
            "pred / tgt  prediction / target",
            ha="left", va="top", fontsize=5.0, color=INK,
            linespacing=1.5, zorder=4)
    ax.text(93.5, 23.4,
            "$K_t$  masked targets\n"
            "$D$  embedding dim\n"
            "$p$  patch size\n"
            "$p_{head}$  head grid ($\\geq p$)\n"
            "$\\mu$  pixel size ($\\mu$m/px)\n"
            "$z_c$  channel identity\n"
            "$D_{attn}$  fusion attn dim\n"
            "$enc_{all}$  decoder KV ($lat$)",
            ha="left", va="top", fontsize=5.0, color=INK,
            linespacing=1.5, zorder=4)
    ax.text(108.0, 23.4,
            "$K_f$  fused locations\n"
            "PMA  attention pooler\n"
            "RoPE  2-D axial rot.\n"
            "fracPE  fractional PE\n"
            "DCP  visibility sampler\n"
            "mask  learned mask token\n"
            "$q$  decoder query\n"
            "$L_e$ / $L_d$  enc / dec blocks",
            ha="left", va="top", fontsize=5.0, color=INK,
            linespacing=1.5, zorder=4)
    ax.text(78.5, 12.4,
            "drawn row counts are schematic (default $L{=}16$)",
            ha="left", va="center", fontsize=5.2, color=MUTED,
            style="italic", zorder=4)

    # ---- key points ------------------------------------------------------- #
    rrect(ax, FRAME_X0, 2.0, FRAME_X1 - FRAME_X0, 5.0, edge=ENCODER,
          face=tint(ENCODER, 0.88), lw=1.3, rounding=0.5, z=2)
    ax.text((FRAME_X0 + FRAME_X1) / 2, 5.2,
            r"Asymmetry: encoder sees $K_f$ fused locations; "
            r"decoder reconstructs $K_t$ masked channel-patches",
            ha="center", va="center", fontsize=6.4, color=INK,
            fontweight="bold")
    ax.text((FRAME_X0 + FRAME_X1) / 2, 3.6,
            r"$V + K_t \leq \Sigma_b\, C_b \cdot N_b$"
            r"   (equality without the off-cell mask)",
            ha="center", va="center", fontsize=5.9, color=INK,
            fontweight="bold")

    rrect(ax, DFRAME_X0, 2.0, DFRAME_X1 - DFRAME_X0, 5.0, edge=DECODER,
          face=tint(DECODER, 0.88), lw=1.3, rounding=0.5, z=2)
    ax.text((DFRAME_X0 + DFRAME_X1) / 2, 4.5,
            r"One shared decoder: target self-attn, then cross-attn to the latents",
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
