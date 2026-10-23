"""L6 -- MAE objective: encoder-decoder asymmetry and channel-aware reconstruction.

Shows the two-branch objective with:
* Encoder branch: sees only visible tokens [Kf, D] + memory [M, D] -> enc_all
* Decoder branch: target queries conditioned on channel identity -> predictions
* One shared decoder for all channels (virtual staining capability)
* Asymmetry: Kf (visible) vs Kt (masked), Kf + Kt = ΣC·N

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
    FRONTEND, FUSION, ENCODER, DECODER, MEMORY, LOSS, FUSED, GHOST,
    tint, rrect, box, arrow, edge_label, sq_bracket, token_rows, channel_color,
    install_rcparams, save_figure, role_legend, channel_legend,
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
def draw_encoder_input(ax, x=9.0, y0=47.5, w=11.0):
    """Channel-patch tokens before/after DCP omission; one row is a dashed ghost.

    The omitted row is drawn with the LOSS accent (absent, never zeroed).
    """
    specs = [
        (BLUE, False),
        (GREEN, False),
        (LOSS, True),   # omitted channel-patch -- absent, never zeroed
        (PURPLE, False),
        (BLUE, False),
        (ORANGE, False),
    ]
    top = token_rows(ax, x, y0, w, H, PITCH, specs)
    sq_bracket(ax, x + w + 1.0, y0, top)
    ax.text(x + w / 2, top + 2.4, "$[K_f, D]$ visible tokens", ha="center",
            va="center", fontsize=7.8, color=INK, fontweight="bold")
    omitted_y = y0 + 2 * PITCH + H / 2
    return top, omitted_y


def draw_memory(ax, x=9.0, y0=38.0, w=11.0):
    """Fixed memory tokens [B·M, D] (MEMORY gold)."""
    specs = [(MEMORY, False), (MEMORY, False)]
    top = token_rows(ax, x, y0, w, H, PITCH, specs)
    sq_bracket(ax, x + w + 1.0, y0, top)
    ax.text(x + w / 2, y0 - 2.6, "$[B{\\cdot}M, D]$ memory", ha="center",
            va="center", fontsize=7.8, color=INK, fontweight="bold")
    return top


def draw_enc_all(ax, x=54.0, y0=42.0, w=11.0):
    """Encoder output [B·M + Kf, D] = memory-out then visible token-out."""
    specs = [
        (MEMORY, False), (MEMORY, False),
        (BLUE, False), (GREEN, False), (PURPLE, False), (BLUE, False),
        (ORANGE, False),
    ]
    top = token_rows(ax, x, y0, w, H, PITCH, specs)
    sq_bracket(ax, x + w + 1.0, y0, top)
    ax.text(x + w / 2, top + 2.4, "$[B{\\cdot}M + K_f, D]$", ha="center",
            va="center", fontsize=7.8, color=INK, fontweight="bold")
    # Row-group labels to the left of the stack (clear of the feeder bus).
    ax.text(x - 1.0, 45.2, "mem (M)", ha="right", va="center", fontsize=5.8,
            color=MEMORY)
    ax.text(x - 1.0, 57.5, "vis ($K_f$)", ha="right", va="center", fontsize=5.8,
            color=BLUE)
    return top


def draw_target_queries(ax, x=86.0, y0=53.0, w=12.0):
    """Decoder target queries [Kt, D], conditioned on target channel identity.

    Queries are the shared fused/neutral objects (FUSED blue); the target
    channel identity enters additively through ``target_proj``.
    """
    specs = [(FUSED, False)] * 3
    top = token_rows(ax, x, y0, w, H, PITCH, specs)
    sq_bracket(ax, x + w + 1.0, y0, top)
    ax.text(x + w / 2, top + 2.4, "$[K_t, D]$ queries", ha="center", va="center",
            fontsize=7.8, color=INK, fontweight="bold")
    # Query recipe (decoder.py): mask_token + spatial_pos[patch] + target_proj(z_c)
    ax.text(x + w / 2 - 2.0, y0 - 3.0,
            r"$q = \mathrm{mask} + \mathrm{pos}_n + W z_c$",
            ha="center", va="center", fontsize=6.8, color=MUTED, style="italic")
    return top


def draw_predictions(ax, x=126.0, y0=53.0, w=14.0):
    """Decoder predictions [Kt, P²] -- DECODER violet."""
    specs = [(DECODER, False)] * 3
    top = token_rows(ax, x, y0, w, H, PITCH, specs)
    ax.text(x + w / 2, top + 2.4, "$[K_t, P^2]$ predictions", ha="center",
            va="center", fontsize=7.8, color=INK, fontweight="bold")
    return top


def draw_targets(ax, x=126.0, y0=34.0, w=14.0):
    """Raw pixel targets [Kt, P²] for the masked channel-patches.

    Targets are per-channel -> identity rows use CHANNEL_COLORS.
    """
    specs = [
        (channel_color(2), False),
        (channel_color(3), False),
        (channel_color(0), False),
    ]
    top = token_rows(ax, x, y0, w, H, PITCH, specs)
    ax.text(x + w / 2, top + 2.7, "$[K_t, P^2]$ targets", ha="center",
            va="center", fontsize=7.8, color=INK, fontweight="bold")
    ax.text(x + w / 2, top + 0.9, "(raw masked pixels)", ha="center",
            va="center", fontsize=6.3, color=MUTED, style="italic")
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
    draw_encoder_input(ax, x=9.0, y0=47.5)
    draw_memory(ax, x=9.0, y0=38.0)
    draw_enc_all(ax, x=54.0, y0=42.0)

    # DCP omission annotation, fully inside the frame and clear of the dashed row.
    ax.text(8.4, 56.55, "DCP\nomission\n(not zeroed)", ha="right",
            va="center", fontsize=5.4, color=FRONTEND, fontweight="bold",
            linespacing=1.15)

    # Encoder module (TransformerEncoder) -- ENCODER role colour.
    box(ax, 27.0, 43.0, 19.0, 21.0, "TransformerEncoder",
        "grouped self-attn\n$\\times L_e$ blocks", edge=ENCODER, sub_size=6.6)

    # Visible tokens + memory feed the encoder; encoder -> enc_all.
    arrow(ax, (21.0, 58.5), (27.0, 56.0), color=FUSED, lw=1.5)
    arrow(ax, (21.0, 41.5), (27.0, 48.0), color=MEMORY, lw=1.5)
    arrow(ax, (46.0, 53.5), (54.0, 53.5), color=ENCODER, lw=1.5)

    # ---- decoder pathway -------------------------------------------------- #
    draw_target_queries(ax, x=86.0, y0=53.0)

    box(ax, 104.0, 46.0, 17.0, 21.0, "Decoder",
        "grouped cross-attn\n$\\times L_d$ blocks", edge=DECODER, sub_size=6.6)

    # queries -> decoder
    arrow(ax, (98.0, 58.5), (104.0, 58.5), color=DECODER, lw=1.5)

    # enc_all -> decoder cross-attention bus (keys = values = enc_all).
    ax.plot([66.0, 101.0], [44.0, 44.0], color=ENCODER, lw=1.2,
            linestyle="--", zorder=4)
    ax.plot([101.0, 101.0], [44.0, 54.0], color=ENCODER, lw=1.2,
            linestyle="--", zorder=4)
    arrow(ax, (101.0, 54.0), (104.0, 54.0), color=ENCODER, lw=1.2, dashed=True)
    edge_label(ax, 87.5, 46.2, "keys / values (enc_all)", color=ENCODER,
               size=6.4)

    # decoder -> head -> predictions
    box(ax, 126.0, 69.0, 17.0, 9.0, "Head",
        "$\\mathrm{Linear}(D, P^2)$", edge=DECODER, title_size=7.6,
        sub_size=6.4)
    ax.plot([121.0, 124.0], [60.0, 60.0], color=DECODER, lw=1.4, zorder=4)
    ax.plot([124.0, 124.0], [60.0, 73.5], color=DECODER, lw=1.4, zorder=4)
    arrow(ax, (124.0, 73.5), (126.0, 73.5), color=DECODER, lw=1.5)

    pred_top = draw_predictions(ax, x=126.0, y0=53.0)
    arrow(ax, (138.5, 69.0), (138.5, pred_top), color=DECODER, lw=1.5)

    # ---- targets (raw pixels) -------------------------------------------- #
    draw_targets(ax, x=126.0, y0=34.0)

    # Short, light route: targets are the raw pixels of the masked patches.
    arrow(ax, (81.0, 38.25), (126.0, 38.25), color=GHOST, lw=1.0, dashed=True)
    ax.text(103.0, 40.6, "target = raw pixels of masked channel-patches",
            ha="center", va="center", fontsize=6.8, color=MUTED, style="italic")

    # ---- loss ------------------------------------------------------------- #
    box(ax, 124.0, 12.0, 18.0, 11.0, "Loss",
        "per-patch standardize target,\nthen MSE (no per-channel norm)",
        edge=LOSS, title_size=7.6, sub_size=5.4)

    # predictions -> loss (right), targets -> loss (top), merged at the box.
    ax.plot([140.0, 148.0, 148.0], [58.0, 58.0, 17.5], color=DECODER, lw=1.3,
            zorder=4)
    arrow(ax, (148.0, 17.5), (142.0, 17.5), color=LOSS, lw=1.5)
    ax.plot([133.0, 133.0], [34.0, 25.0], color=GHOST, lw=1.3, linestyle="--",
            zorder=4)
    arrow(ax, (133.0, 25.0), (133.0, 23.0), color=LOSS, lw=1.5)
    ax.text(147.0, 40.0, "pred", ha="right", va="center", fontsize=6.6,
            color=DECODER)
    ax.text(134.6, 29.0, "tgt", ha="left", va="center", fontsize=6.6,
            color=MUTED)

    # ---- role + channel legends (empty lower-left) ------------------------ #
    role_legend(
        ax, 3.0, 25.5,
        [
            ("front-end / patch embed", FRONTEND),
            ("channel fusion", FUSION),
            ("spatial encoder", ENCODER),
            ("channel-aware decoder", DECODER),
            ("memory tokens", MEMORY),
            ("objective / omitted rows", LOSS),
            ("fused joint tokens", FUSED),
        ],
        title="module roles", pitch=2.2, sw=1.4, fontsize=6.2,
    )
    channel_legend(
        ax, 39.0, 25.5, n=len(CHANNEL_COLORS), title="channel identity",
        pitch=2.2, sw=1.4, fontsize=6.2,
    )

    # ---- key points ------------------------------------------------------- #
    rrect(ax, FRAME_X0, 2.0, FRAME_X1 - FRAME_X0, 5.0, edge=ENCODER,
          face=tint(ENCODER, 0.88), lw=1.3, rounding=0.5, z=2)
    ax.text((FRAME_X0 + FRAME_X1) / 2, 4.5,
            r"Asymmetry: encoder sees $K_f$ (visible), decoder reconstructs $K_t$ (masked); $K_f + K_t = \Sigma C \cdot N$",
            ha="center", va="center", fontsize=7.6, color=INK,
            fontweight="bold")

    rrect(ax, DFRAME_X0, 2.0, DFRAME_X1 - DFRAME_X0, 5.0, edge=DECODER,
          face=tint(DECODER, 0.88), lw=1.3, rounding=0.5, z=2)
    ax.text((DFRAME_X0 + DFRAME_X1) / 2, 4.5,
            r"One shared decoder for ALL channels — conditioned on target identity $z_c$ (virtual staining)",
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
