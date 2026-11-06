"""L3 -- patch embed + channel identity (carried separately) tokenizer detail.

Shows the channel-tied patch embedding (one PI-resized Conv2d kernel shared by
all channels) and the channel identity `z_c`, which is carried alongside the
content token instead of being added to it.  Emphasizes:

- Tokenizer and fusion add **no** spatial table; the encoder supplies 2-D axial
  RoPE on normalized `(row, col)`, and the decoder a fractional absolute PE.
- The Conv2d kernel is stored at a canonical `P` and **PI-resized** only when
  `p_b != P` (identity at the shipped `P`); each sample has
  `N_b = floor(H/p_b) * floor(W/p_b)` patches, the border smaller than `p_b`
  being cropped.
- `z_c` is indexed by the dataset-level `channel_id`, not the panel position.
- DCP-omitted tokens are rejected at flatten as absent rows; they have no
  content token and no `z_c` entry.
- The output is one concatenated token list `[V, D]` (`V <= sum_b C_b * N_b`).

Usage::

    uv run --extra viz python viz/level3_tokenizer.py
    uv run --extra viz python viz/level3_tokenizer.py --formats pdf svg png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from figure_style import (  # noqa: E402
    GREEN, RED, INK, MUTED, CHANNEL_COLORS,
    TOKENIZER, LOSS,
    tint, rrect, box, arrow, token_rows, install_rcparams,
    legend, channel_legend, padded_ghost, sq_bracket_mirror, save_figure,
)

OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Components
# --------------------------------------------------------------------------- #
def draw_channel_tile(ax, x, y, size, color):
    """Single channel image tile [1, H, W]."""
    rrect(ax, x, y, size, size, edge=color, face=tint(color, 0.72), lw=1.5)
    ax.plot([x + 0.8, x + size - 0.8], [y + 1.5, y + size - 1.5],
            color=color, lw=0.9, alpha=0.6, zorder=4)
    ax.plot([x + 0.8, x + size - 0.8], [y + size - 1.5, y + 1.5],
            color=color, lw=0.9, alpha=0.4, zorder=4)


def draw_patch_stack(ax, x, y0, w, h, pitch, n, color, omitted_idx=None):
    """Vertical stack of N patch tokens; the omitted row is a red dashed row."""
    specs = []
    for i in range(n):
        omitted = (omitted_idx is not None and i == omitted_idx)
        specs.append((RED if omitted else color, omitted))
    top = token_rows(ax, x, y0, w, h, pitch, specs, rounding=0.35, z=3)
    return top


def draw_identity_card(ax, x, y, w, h, color, label):
    """A small identity embedding card z_c (per-channel colour).

    z_c is a data vector flowing through the diagram, so the card is crossed
    like every other data cell; only module boxes stay plain.
    """
    rrect(ax, x, y, w, h, edge=color, face=tint(color, 0.75), lw=1.5,
          rounding=0.4, crossed=True)
    # white backing keeps the label legible where it crosses the diagonals
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            fontsize=8.0, fontweight="bold", color=INK, zorder=5,
            bbox=dict(boxstyle="round,pad=0.10", facecolor="white",
                      edgecolor="none", alpha=0.88))


def dashed_route(ax, pts, color=INK, lw=1.4):
    """Orthogonal dashed polyline ending in an arrow head (for lookups)."""
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    ax.plot(xs, ys, color=color, lw=lw, linestyle=(0, (4, 2.6)), zorder=6,
            solid_capstyle="round")
    arrow(ax, pts[-2], pts[-1], color=color, lw=lw, dashed=True, ms=10)


# --------------------------------------------------------------------------- #
# Main figure
# --------------------------------------------------------------------------- #
def build() -> plt.Figure:
    fig = plt.figure(figsize=(15, 8))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_xlim(0, 150)
    ax.set_ylim(0, 80)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    # ---- Title ------------------------------------------------------------ #
    ax.text(2, 77.5, "L3", ha="left", va="center", fontsize=15,
            fontweight="bold", color=INK)
    ax.text(6.5, 77.5, "Tokenizer: channel-tied patch embed + channel identity (carried separately)",
            ha="left", va="center", fontsize=12.2, fontweight="bold", color=INK)
    ax.text(2, 73.8,
            "Two channels share one projection kernel $\\theta$ (PI-resized only "
            "when $p_b \\neq P$); "
            "each looks up its own z_c, carried alongside the content token "
            "(fusion routing bias + skip); no spatial table.",
            ha="left", va="center", fontsize=7.4, color=MUTED, style="italic")
    ax.text(2, 70.9,
            "No spatial table in tokenizer or fusion  |  encoder = 2-D axial RoPE "
            "(rotary PE)  |  decoder = fractional absolute PE  |  "
            "DCP-omitted rows are absent",
            ha="left", va="center", fontsize=7.2, color=RED, style="italic",
            fontweight="bold")
    ax.text(2, 67.9,
            "Shapes:  $[1,H,W]$  \u2192  Conv2d($P \\to p_b$; PI-resized only if "
            "$p_b \\neq P$)  \u2192  $[N_b, D]$ content  +  $z_c$ $[D]$ carried  \u2192  "
            "flat $[V,D]$ tokens + $[V,D]$ channel_embed,  "
            "$V \\leq \\Sigma_b C_b \\cdot N_b$",
            ha="left", va="center", fontsize=6.8, color=INK)

    # ---- Shared geometry: both lanes on one grid -------------------------- #
    lane_yc = [57.0, 29.0]                      # channel 0 / channel 1 centres
    lane_color = [CHANNEL_COLORS[0], CHANNEL_COLORS[1]]
    lane_name = ["channel 0", "channel 1"]

    tile_x, tile_s = 4.0, 9.0
    conv_x, conv_w = 18.0, 9.0
    stack_x, stack_w, stack_h, stack_pitch, n_patches = 32.0, 7.0, 1.5, 2.0, 4
    z_x, z_w, z_h = 45.0, 6.0, 4.0
    out_x = 63.0
    omit_idx = 1                                # DCP-omitted row in channel 1
    stack_htot = (n_patches - 1) * stack_pitch + stack_h

    # ---- Channel lanes ---------------------------------------------------- #
    for lane, yc in enumerate(lane_yc):
        color = lane_color[lane]

        # Image tile [1, H, W]
        draw_channel_tile(ax, tile_x, yc - tile_s / 2, tile_s, color)
        ax.text(tile_x + tile_s / 2, yc + tile_s / 2 + 1.9, lane_name[lane],
                ha="center", va="center", fontsize=7.4, fontweight="bold",
                color=color)
        ax.text(tile_x + tile_s / 2, yc - tile_s / 2 - 1.9, "$[1, H, W]$",
                ha="center", va="center", fontsize=6.8, color=MUTED)

        # tile -> shared Conv2d -> projected patches
        arrow(ax, (tile_x + tile_s, yc), (conv_x, yc), color=color)
        arrow(ax, (conv_x + conv_w, yc), (stack_x, yc), color=color)

        # patch stack [N_b, D]
        st_y0 = yc - stack_htot / 2
        top = draw_patch_stack(ax, stack_x, st_y0, stack_w, stack_h,
                               stack_pitch, n_patches, color,
                               omitted_idx=(omit_idx if lane == 1 else None))
        ax.text(stack_x + stack_w / 2, top + 1.9, "$[N_b, D]$", ha="center",
                va="center", fontsize=6.8, color=MUTED)

        # identity z_c is carried alongside the content token (never added)
        arrow(ax, (stack_x + stack_w, yc), (z_x, yc), color=color)
        ax.text((stack_x + stack_w + z_x) / 2, yc + 2.0, "content", ha="center",
                va="center", fontsize=6.2, color=MUTED, style="italic")
        # card sits ON the lane so the content arrow enters its left edge and
        # leaves its right edge (no arrow stopping mid-air)
        z_cy = yc
        draw_identity_card(ax, z_x, z_cy - z_h / 2, z_w, z_h, color,
                           f"$z_{{c_{lane}}}$")
        ax.text(z_x + z_w + 0.6, z_cy + 3.2,
                "carried, not added", ha="left", va="center",
                fontsize=6.0, color=MUTED)
        ax.text(z_x + z_w + 0.6, z_cy + 1.4,
                "channel_id $\\to z_c$ $[D]$", ha="left", va="center",
                fontsize=6.0, color=MUTED, zorder=5,
                bbox=dict(boxstyle="round,pad=0.10", facecolor="white",
                          edgecolor="none", alpha=0.88))

        # -> per-channel content tokens: still N_b rows, still [N_b, D]
        arrow(ax, (z_x + z_w, yc), (out_x, yc), color=color)
        o_y0 = yc - stack_htot / 2
        top = draw_patch_stack(ax, out_x, o_y0, stack_w, stack_h,
                               stack_pitch, n_patches, color,
                               omitted_idx=(omit_idx if lane == 1 else None))
        ax.text(out_x + stack_w / 2, top + 1.9, "$[N_b, D]$",
                ha="center", va="center", fontsize=6.8, color=MUTED)

    # ---- Merged output object --------------------------------------------- #
    # The two per-channel [N_b, D] blocks concatenate into ONE token list
    # ChannelSet.tokens [V, D] (V <= sum_b C_b * N_b) with omitted rows dropped.
    br_x = out_x + stack_w + 1.4
    y_lo = lane_yc[1] - stack_htot / 2
    y_hi = lane_yc[0] + stack_htot / 2
    # Bracket sits to the RIGHT of the output stacks, so its ticks must point
    # LEFT (mirror) to embrace the stack rather than away from it.
    sq_bracket_mirror(ax, br_x, y_lo, y_hi, color=INK, lw=1.3, tick=0.9)
    ax.text(br_x + 1.4, (y_lo + y_hi) / 2 + 1.6, "$[V, D]$", ha="left",
            va="center", fontsize=8.6, fontweight="bold", color=INK)
    ax.text(br_x + 1.4, (y_lo + y_hi) / 2 - 1.2, "content tokens", ha="left",
            va="center", fontsize=6.4, color=MUTED)
    ax.text(br_x + 1.4, (y_lo + y_hi) / 2 - 3.6,
            "+ channel_embed $[V,D]$ (parallel)", ha="left", va="center",
            fontsize=6.2, color=MUTED)

    # annotate the omitted row on the (upper) projected stack of channel 1:
    # it is dropped at flatten, before any identity use
    row_y = lane_yc[1] - stack_htot / 2 + omit_idx * stack_pitch + stack_h / 2
    ax.plot([stack_x + stack_w, stack_x + stack_w + 1.0], [row_y, row_y - 1.3],
            color=RED, lw=1.0, zorder=7)
    ax.text(stack_x + stack_w + 1.2, row_y - 1.7, "omitted", ha="left",
            va="center", fontsize=6.1, color=RED, style="italic",
            fontweight="bold", zorder=7)

    # annotate the omitted row on the merged output stack too (to the left of
    # the stack so it stays clear of the [V, D] bracket)
    ax.plot([out_x, out_x - 0.9], [row_y, row_y - 1.5], color=RED, lw=1.0,
            zorder=7)
    ax.text(out_x - 1.1, row_y - 1.8, "omitted row", ha="right",
            va="center", fontsize=6.1, color=RED, style="italic",
            fontweight="bold")

    # ---- Shared Conv2d module (one canonical kernel, PI-resized per sample) #
    rrect(ax, conv_x, 22.0, conv_w, 40.0, edge=TOKENIZER, face=tint(TOKENIZER, 0.86),
          lw=1.5, rounding=0.7, z=3)
    cx = conv_x + conv_w / 2
    ax.text(cx, 60.2, "Conv2d", ha="center", va="center", fontsize=8.6,
            fontweight="bold", color=INK, zorder=4)
    ax.text(cx, 56.9, "1 \u2192 D", ha="center", va="center", fontsize=6.8,
            color=INK, zorder=4)
    ax.text(cx, 54.4, "canonical kernel P", ha="center", va="center",
            fontsize=6.6, color=MUTED, zorder=4)
    ax.text(cx, 52.1, "stride $p_b$", ha="center", va="center",
            fontsize=6.6, color=MUTED, zorder=4)
    ax.plot([cx - 2.8, cx + 2.8], [49.5, 49.5], color=TOKENIZER, lw=1.0,
            linestyle=(0, (3, 2)), zorder=4)
    ax.text(cx, 46.6, "PI-resize if $p_b \\neq P$", ha="center",
            va="center", fontsize=6.6, fontweight="bold", color=TOKENIZER,
            zorder=4)
    ax.text(cx, 42.8, "shared kernel $\\theta$", ha="center", va="center",
            fontsize=7.0, fontweight="bold", color=TOKENIZER, zorder=4)
    ax.text(cx, 38.4, "one kernel for\nevery channel", ha="center", va="center",
            fontsize=6.3, color=MUTED, zorder=4, linespacing=1.5)
    ax.text(cx, 33.6, "(tied across\nchannels)", ha="center", va="center",
            fontsize=6.1, color=MUTED, zorder=4, linespacing=1.5)

    # ---- Identity embedding source ---------------------------------------- #
    emb_x, emb_y, emb_w, emb_h = 90.0, 37.0, 16.0, 10.0
    box(ax, emb_x, emb_y, emb_w, emb_h, "nn.Embedding", "num_channel_ids \u00d7 D",
        edge=TOKENIZER, title_size=7.8, sub_size=6.3)
    ax.text(emb_x + emb_w / 2, emb_y - 2.2, "$z_c$ lookup table",
            ha="center", va="center", fontsize=6.4, color=TOKENIZER,
            style="italic")

    # dashed lookup routes: table -> each z_c card.  Routed above / below the
    # lane output so the merged-output bracket stays unobstructed.
    dashed_route(ax, [(emb_x + emb_w * 0.5, emb_y + emb_h),
                      (emb_x + emb_w * 0.5, 64.0),
                      (z_x + 3.0, 64.0),
                      (z_x + 3.0, lane_yc[0] + z_h / 2)],
                 color=lane_color[0])
    dashed_route(ax, [(emb_x + emb_w * 0.12, emb_y),
                      (emb_x + emb_w * 0.12, 20.0),
                      (z_x + 3.0, 20.0),
                      (z_x + 3.0, lane_yc[1] - z_h / 2)],
                 color=lane_color[1])

    # ---- Identity handling (centre band, clear of everything) ------------- #
    rrect(ax, 28.3, 37.0, 14.4, 11.0, edge=MUTED, face=tint(MUTED, 0.97),
          lw=1.1, dashed=True, rounding=0.5, z=2)
    ax.text(35.5, 46.0, "identity", ha="center", va="center",
            fontsize=6.4, fontweight="bold", color=INK, zorder=4)
    ax.text(35.5, 43.3, "(per channel)", ha="center", va="center",
            fontsize=6.4, fontweight="bold", color=INK, zorder=4)
    ax.text(35.5, 40.2, "not summed into token content\n"
            "routing bias: logits $+=$ bias\n"
            "output skip: fused $+=$ skip",
            ha="center", va="center", fontsize=5.2, color=MUTED, zorder=4,
            linespacing=1.4)

    # ---- DCP omission strip (full width) ---------------------------------- #
    strip_x, strip_y, strip_w, strip_h = 4.0, 2.0, 144.0, 16.0
    rrect(ax, strip_x, strip_y, strip_w, strip_h, edge=INK,
          face=tint(INK, 0.97), lw=1.2, rounding=0.6, z=2)
    ax.text(strip_x + 2.0, strip_y + strip_h - 2.6,
            "DCP omission = absent row", ha="left", va="center", fontsize=7.8,
            fontweight="bold", color=INK, zorder=4)

    # mini token stack: omitted row dashed red, absent from the list
    mini_specs = [(GREEN, False), (RED, True), (GREEN, False), (GREEN, False)]
    token_rows(ax, strip_x + 2.0, strip_y + 4.4, 6.0, 1.3, 1.8, mini_specs,
               rounding=0.3, z=4)
    ax.text(strip_x + 2.0, strip_y + 2.6, "omitted\nrow", ha="left", va="center",
            fontsize=5.7, color=RED, style="italic", zorder=4, linespacing=1.4)

    ax.text(strip_x + 12.5, strip_y + 11.6,
            "Conv2d runs on every patch; the visible mask then selects which "
            "rows survive.", ha="left", va="center", fontsize=6.4, color=INK,
            zorder=4)
    ax.text(strip_x + 12.5, strip_y + 8.8,
            "Omission = row selection after the conv: a dropped row is never "
            "zeroed, and owns no $z_c$.", ha="left", va="center",
            fontsize=6.4, color=INK, zorder=4)
    ax.text(strip_x + 12.5, strip_y + 6.0,
            "Absent rows are not concatenated into the flat token list "
            "$[V, D]$.", ha="left", va="center", fontsize=6.4,
            color=MUTED, zorder=4)

    # ---- symbols / glossary (bottom strip, right centre) ------------------ #
    nx, ny, nw, nh = 58.0, 3.0, 64.0, 13.0
    rrect(ax, nx, ny, nw, nh, edge=MUTED, face="#FFFFFF", lw=1.0, dashed=True,
          rounding=0.5, z=3)
    ax.text(nx + 1.5, ny + nh - 1.9, "symbols / glossary", ha="left",
            va="center", fontsize=6.6, fontweight="bold", color=INK, zorder=4)
    glossary = [
        (59.5, [
            ("$B$", "batch (samples)"),
            ("$C_b$ / $C$", "channels, sample $b$ / count"),
            ("$C_{\\max}$", "max$_b\\,C_b$"),
            ("$D$", "token / embedding dim"),
            ("$H,\\,W$", "image side (square)"),
        ]),
        (80.0, [
            ("$N_b$", "$\\lfloor H/p_b\\rfloor\\cdot\\lfloor W/p_b\\rfloor$; "
                     "border cropped"),
            ("$N$", "max$_b\\,N_b$"),
            ("$V$", "visible channel-patch tokens"),
            ("$P$", "canonical kernel size"),
            ("$p_b$", "per-sample patch size"),
        ]),
        (100.5, [
            ("$\\theta$", "shared Conv2d kernel"),
            ("$z_c$", "channel identity (bias + skip)"),
            ("PI-resize", "FlexiViT kernel resize if $p_b\\neq P$"),
            ("PE", "positional embedding"),
            ("RoPE", "2-D axial rotary PE"),
            ("DCP", "omission = omit rows, never zero"),
        ]),
    ]
    y0, pitch, off = ny + nh - 3.7, 1.72, 6.6
    for col_x, entries in glossary:
        for i, (sym, meaning) in enumerate(entries):
            yy = y0 - i * pitch
            ax.text(col_x, yy, sym, ha="left", va="center", fontsize=5.4,
                    fontweight="bold", color=INK, zorder=4)
            ax.text(col_x + off, yy, meaning, ha="left", va="center",
                    fontsize=5.2, color=MUTED, zorder=4)

    # dashed ghost of the padded tensor that is never allocated/built
    # dashed ghost: the padded tensor is never allocated nor built
    padded_ghost(ax, strip_x + strip_w - 24.0, strip_y + 2.5, 22.0, 10.5,
                 note="never allocated / built", z=3)

    # ---- Right column ----------------------------------------------------- #
    # Key design points
    leg_x, leg_y, leg_w, leg_h = 110.0, 54.0, 38.0, 23.5
    rrect(ax, leg_x, leg_y, leg_w, leg_h, edge=INK, face=tint(INK, 0.97),
          lw=1.3, rounding=0.6, z=2)
    ax.text(leg_x + leg_w / 2, leg_y + leg_h - 2.7, "Key design points",
            ha="center", va="center", fontsize=8.6, fontweight="bold", color=INK)

    bullets = [
        "\u2022 One shared Conv2d kernel $\\theta$; PI-resized $P \\to p_b$ when "
        "$p_b \\neq P$",
        "\u2022 No spatial table here: encoder = 2-D axial RoPE",
        "\u2022 Decoder = fractional absolute PE (no grid table)",
        "\u2022 $z_c$ = Embedding(channel_id), not panel position",
        "\u2022 DCP omission \u2192 row absent from the list",
        "\u2022 $C$ enters only as the list length $V$ (visible rows)",
    ]
    bullet_y = leg_y + leg_h - 5.9
    for bullet in bullets:
        ax.text(leg_x + 1.8, bullet_y, bullet, ha="left", va="center",
                fontsize=6.5, color=INK)
        bullet_y -= 3.1

    # Legends: module roles stay plain; z_c is a data vector so its swatch is
    # crossed; omitted rows are dashed/uncrossed; channels are crossed too.
    legend(ax, 112.0, 49.0,
           [("tokenizer (module)", TOKENIZER, "fill"),
            ("$z_c$ embedding", TOKENIZER, "cross"),
            ("omitted row", LOSS, "omit")],
           title="module roles / data", pitch=2.4, sw=1.4, fontsize=6.2)
    channel_legend(ax, 131.0, 49.0, n=2, title="channel identity",
                   pitch=2.4, sw=1.4, fontsize=6.2)

    # Output-object note (fills the lower-right quadrant, reinforces [V, D])
    rrect(ax, 110.0, 28.0, 38.0, 12.0, edge=INK, face=tint(INK, 0.97),
          lw=1.3, rounding=0.6, z=2)
    ax.text(129.0, 37.4, "One flat token list", ha="center", va="center",
            fontsize=8.0, fontweight="bold", color=INK)
    ax.text(112.0, 34.7, "one concatenated ChannelSet.tokens list",
            ha="left", va="center", fontsize=6.3, color=INK)
    ax.text(112.0, 32.2, "content tokens $[V, D]$  ++  parallel channel_embed $[V, D]$",
            ha="left", va="center", fontsize=6.3, color=INK)
    ax.text(112.0, 29.7, "masked rows absent (no content, no $z_c$)",
            ha="left", va="center", fontsize=6.3, color=MUTED)

    return fig


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--formats", nargs="+", default=["pdf", "svg", "png"])
    ap.add_argument("--stem", default="level3_tokenizer")
    args = ap.parse_args()

    install_rcparams()

    fig = build()
    save_figure(fig, OUT, args.stem, args.formats)
    plt.close(fig)


if __name__ == "__main__":
    main()
