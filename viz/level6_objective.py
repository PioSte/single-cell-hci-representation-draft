"""L6 -- MAE objective: encoder-decoder asymmetry and channel-aware reconstruction.

Shows the two-branch objective with:
* Encoder branch: sees only visible tokens [Kf, D] + memory [M, D] → enc_all
* Decoder branch: target queries conditioned on channel identity → predictions
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
    tint, rrect, box, arrow, edge_label, sq_bracket, token_rows,
    install_rcparams, save_figure,
)

OUT = Path(__file__).resolve().parent / "out"

# --------------------------------------------------------------------------- #
# One shared row grid for every array stack, so the columns line up.
# --------------------------------------------------------------------------- #
PITCH = 3.8
H = 2.9

# Frame geometry (equal width, one gap).
FRAME_X0, FRAME_X1 = 2.5, 74.5
FRAME_Y0, FRAME_Y1 = 28.0, 80.0
DFRAME_X0, DFRAME_X1 = 79.5, 151.5


# --------------------------------------------------------------------------- #
# Array stacks
# --------------------------------------------------------------------------- #
def draw_encoder_input(ax, x=10.0, y0=45.6, w=11.0):
    """Channel-patch tokens before/after DCP omission; one row is a dashed ghost."""
    specs = [
        (BLUE, False),
        (BLUE, False),
        (GREEN, False),
        (ORANGE, True),  # omitted channel-patch -- absent, never zeroed
        (ORANGE, False),
        (PURPLE, False),
    ]
    top = token_rows(ax, x, y0, w, H, PITCH, specs)
    sq_bracket(ax, x + w + 1.0, y0, top)
    ax.text(x + w / 2, top + 2.4, "$[K_f, D]$ visible tokens", ha="center",
            va="center", fontsize=7.8, color=INK, fontweight="bold")
    omitted_y = y0 + 3 * PITCH + H / 2
    return top, omitted_y


def draw_memory(ax, x=10.0, y0=38.0, w=11.0):
    """Memory tokens [B·M, D]."""
    specs = [(ORANGE, False), (ORANGE, False)]
    top = token_rows(ax, x, y0, w, H, PITCH, specs)
    sq_bracket(ax, x + w + 1.0, y0, top)
    ax.text(x + w / 2, y0 - 2.6, "$[B{\\cdot}M, D]$ memory", ha="center",
            va="center", fontsize=7.8, color=INK, fontweight="bold")
    return top


def draw_enc_all(ax, x=55.0, y0=41.8, w=11.0):
    """Encoder output [B·M + Kf, D] = memory-out then visible token-out."""
    specs = [
        (ORANGE, False),
        (ORANGE, False),
        (BLUE, False),
        (BLUE, False),
        (GREEN, False),
        (ORANGE, False),
        (PURPLE, False),
    ]
    top = token_rows(ax, x, y0, w, H, PITCH, specs)
    sq_bracket(ax, x + w + 1.0, y0, top)
    ax.text(x + w / 2, top + 2.4, "$[B{\\cdot}M + K_f, D]$", ha="center",
            va="center", fontsize=7.8, color=INK, fontweight="bold")
    ax.text(x + w / 2, y0 - 3.0, "encoder memory", ha="center", va="center",
            fontsize=7.0, color=INK)
    ax.text(x - 1.2, 43.7, "mem", ha="right", va="center", fontsize=6.2,
            color=ORANGE)
    ax.text(x - 1.2, 57.0, "vis", ha="right", va="center", fontsize=6.2,
            color=BLUE)
    return top


def draw_target_queries(ax, x=86.0, y0=52.5, w=12.0):
    """Decoder target queries [Kt, D], conditioned on target channel identity."""
    specs = [
        (ORANGE, False),
        (VIOLET, False),
        (RED, False),
    ]
    top = token_rows(ax, x, y0, w, H, PITCH, specs)
    sq_bracket(ax, x + w + 1.0, y0, top)
    ax.text(x + w / 2, top + 2.4, "$[K_t, D]$ queries", ha="center", va="center",
            fontsize=7.8, color=INK, fontweight="bold")
    # Query recipe, placed just below the stack and clear of the cross-attn bus.
    ax.text(x + w / 2, y0 - 3.0,
            r"$q = \mathrm{mask} + \mathrm{pos}_n + W z_c$",
            ha="center", va="center", fontsize=6.8, color=MUTED, style="italic")
    return top


def draw_predictions(ax, x=127.0, y0=52.5, w=14.0):
    """Decoder predictions [Kt, P²]."""
    specs = [
        (ORANGE, False),
        (VIOLET, False),
        (RED, False),
    ]
    top = token_rows(ax, x, y0, w, H, PITCH, specs)
    sq_bracket(ax, x + w + 1.0, y0, top)
    ax.text(x + w / 2, top + 2.4, "$[K_t, P^2]$ predictions", ha="center",
            va="center", fontsize=7.8, color=INK, fontweight="bold")
    return top


def draw_targets(ax, x=127.0, y0=33.0, w=14.0):
    """Raw pixel targets [Kt, P²] for the masked channel-patches."""
    specs = [
        (ORANGE, False),
        (VIOLET, False),
        (RED, False),
    ]
    top = token_rows(ax, x, y0, w, H, PITCH, specs)
    ax.text(x + w / 2, top + 2.6, "$[K_t, P^2]$ targets", ha="center",
            va="center", fontsize=7.8, color=INK, fontweight="bold")
    ax.text(x + w / 2, top + 0.7, "(raw masked pixels)", ha="center",
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

    # ---- two equal-width branch frames ------------------------------------ #
    rrect(ax, FRAME_X0, FRAME_Y0, FRAME_X1 - FRAME_X0, FRAME_Y1 - FRAME_Y0,
          edge=BLUE, face=tint(BLUE, 0.96), lw=1.5, dashed=True, rounding=1.0,
          z=1)
    ax.text(3.5, 78.5, "Encoder branch (sees only visible tokens)",
            ha="left", va="center", fontsize=9.5, fontweight="bold", color=BLUE)

    rrect(ax, DFRAME_X0, FRAME_Y0, DFRAME_X1 - DFRAME_X0, FRAME_Y1 - FRAME_Y0,
          edge=VIOLET, face=tint(VIOLET, 0.96), lw=1.5, dashed=True,
          rounding=1.0, z=1)
    ax.text(80.5, 78.5, "Decoder branch (channel-conditioned reconstruction)",
            ha="left", va="center", fontsize=9.5, fontweight="bold", color=VIOLET)

    # ---- encoder pathway -------------------------------------------------- #
    enc_top, omitted_y = draw_encoder_input(ax, x=10.0, y0=45.6)
    draw_memory(ax, x=10.0, y0=38.0)
    draw_enc_all(ax, x=55.0, y0=41.8)

    # DCP omission annotation, fully inside the frame and on the dashed row.
    ax.text(5.8, omitted_y, "DCP\nomission", ha="center", va="center",
            fontsize=6.0, color=GREEN, fontweight="bold", linespacing=1.1)
    ax.text(5.8, omitted_y - 3.2, "(never\nmasked)", ha="center", va="center",
            fontsize=5.6, color=GREEN, style="italic", linespacing=1.0)

    # Encoder module
    box(ax, 28.0, 44.5, 20.0, 20.0, "Spatial encoder",
        "grouped self-attn\n$\\times L$ blocks", edge=PURPLE, sub_size=6.8)

    # Visible tokens + memory feed the encoder (concat inside the module).
    arrow(ax, (22.0, 56.55), (28.0, 55.5), color=BLUE, lw=1.5)
    arrow(ax, (22.0, 41.35), (28.0, 48.0), color=ORANGE, lw=1.5)
    arrow(ax, (48.0, 54.5), (55.0, 54.5), color=PURPLE, lw=1.5)

    # ---- decoder pathway -------------------------------------------------- #
    draw_target_queries(ax, x=86.0, y0=52.5)

    box(ax, 105.0, 47.75, 16.0, 20.0, "Decoder",
        "cross-attn\n$\\times D$ blocks", edge=VIOLET, sub_size=6.8)

    # queries -> decoder
    arrow(ax, (99.0, 57.75), (105.0, 57.75), color=VIOLET, lw=1.5)

    # enc_all -> decoder cross-attention bus (routed below the query formula).
    ax.plot([67.0, 100.0], [46.0, 46.0], color=PURPLE, lw=1.1,
            linestyle="--", zorder=4)
    ax.plot([100.0, 100.0], [46.0, 52.0], color=PURPLE, lw=1.1,
            linestyle="--", zorder=4)
    arrow(ax, (100.0, 52.0), (105.0, 52.0), color=PURPLE, lw=1.1, dashed=True)
    edge_label(ax, 82.0, 42.6, "encoder memory (keys, values)", color=PURPLE,
               size=6.4)

    # decoder -> head -> predictions
    box(ax, 127.0, 69.0, 16.0, 8.0, "Head",
        "$\\mathrm{Linear}(D, P^2)$", edge=VIOLET, title_size=7.6, sub_size=6.4)
    ax.plot([121.0, 124.0], [65.0, 65.0], color=VIOLET, lw=1.4, zorder=4)
    ax.plot([124.0, 124.0], [65.0, 73.0], color=VIOLET, lw=1.4, zorder=4)
    arrow(ax, (124.0, 73.0), (127.0, 73.0), color=VIOLET, lw=1.5)

    pred_top = draw_predictions(ax, x=127.0, y0=52.5)
    arrow(ax, (135.0, 69.0), (135.0, pred_top), color=VIOLET, lw=1.5)

    # ---- targets (raw pixels) -------------------------------------------- #
    targ_top = draw_targets(ax, x=127.0, y0=33.0)

    # Short, light route: targets are the raw pixels of the masked patches.
    arrow(ax, (80.0, 38.25), (127.0, 38.25), color=MUTED, lw=1.0, dashed=True)
    ax.text(101.0, 40.6, "target = raw pixels of masked channel-patches",
            ha="center", va="center", fontsize=6.8, color=MUTED, style="italic")

    # ---- loss ------------------------------------------------------------- #
    box(ax, 126.0, 11.0, 17.0, 10.0, "Loss",
        "per-patch standardize target,\nthen MSE (no per-channel norm)",
        edge=RED, title_size=7.6, sub_size=5.4)

    # predictions -> loss (right) and targets -> loss (bottom), merged.
    ax.plot([141.0, 147.0, 147.0], [57.75, 57.75, 25.0], color=VIOLET, lw=1.3,
            zorder=4)
    ax.plot([147.0, 135.0], [25.0, 25.0], color=RED, lw=1.3, zorder=4)
    ax.plot([135.0, 135.0], [33.0, 25.0], color=MUTED, lw=1.3, linestyle="--",
            zorder=4)
    arrow(ax, (135.0, 25.0), (135.0, 21.0), color=RED, lw=1.5)
    ax.text(148.4, 42.0, "pred", ha="left", va="center", fontsize=6.6,
            color=VIOLET)
    ax.text(136.5, 29.0, "tgt", ha="left", va="center", fontsize=6.6,
            color=MUTED)

    # ---- key points ------------------------------------------------------- #
    rrect(ax, FRAME_X0, 2.0, FRAME_X1 - FRAME_X0, 5.0, edge=BLUE,
          face=tint(BLUE, 0.88), lw=1.3, rounding=0.5, z=2)
    ax.text((FRAME_X0 + FRAME_X1) / 2, 4.5,
            r"Asymmetry: encoder sees $K_f$ (visible), decoder reconstructs $K_t$ (masked); $K_f + K_t = \Sigma C \cdot N$",
            ha="center", va="center", fontsize=7.6, color=INK, fontweight="bold")

    rrect(ax, DFRAME_X0, 2.0, DFRAME_X1 - DFRAME_X0, 5.0, edge=VIOLET,
          face=tint(VIOLET, 0.88), lw=1.3, rounding=0.5, z=2)
    ax.text((DFRAME_X0 + DFRAME_X1) / 2, 4.5,
            r"One shared decoder for ALL channels — conditioned on target identity $z_c$ (virtual staining)",
            ha="center", va="center", fontsize=7.6, color=INK, fontweight="bold")

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
