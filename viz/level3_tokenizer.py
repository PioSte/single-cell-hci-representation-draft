"""L3 -- patch embed + additive identity tokenizer detail.

Shows the channel-tied patch embedding (one shared Conv2d projection for all
channels) plus the additive channel embedding `z_c`. Emphasizes:
- The tokenizer has **no spatial positional embedding**; position is added
  post-fusion by `ChannelFusion` / `IdentityGatedSum`.
- `z_c` is per channel id, not a position.
- DCP-omitted tokens are drawn as absent dashed rows, never zero cards.
- Tied vs untied projection comparison (ChannelViT shows tied is more robust).

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
    BLUE, GREEN, ORANGE, RED, PURPLE, VIOLET, INK, MUTED, CHANNEL_COLORS,
    TOKENIZER, FUSION, ENCODER, DECODER, LOSS, GHOST,
    tint, rrect, box, arrow, token_rows, install_rcparams,
    legend, role_legend, channel_legend, padded_ghost, save_figure,
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
    """A small identity embedding card z_c (per-channel colour)."""
    rrect(ax, x, y, w, h, edge=color, face=tint(color, 0.75), lw=1.5,
          rounding=0.4)
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            fontsize=8.0, fontweight="bold", color=INK, zorder=4)


def dashed_route(ax, pts, color=INK, lw=1.4):
    """Orthogonal dashed polyline ending in an arrow head (for lookups)."""
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    ax.plot(xs, ys, color=color, lw=lw, linestyle=(0, (4, 2.6)), zorder=6,
            solid_capstyle="round")
    arrow(ax, pts[-2], pts[-1], color=color, lw=lw, dashed=True, ms=10)


def draw_tied_inset(ax, ix, iy, iw, ih):
    """Small inset comparing tied vs untied projections."""
    rrect(ax, ix, iy, iw, ih, edge=MUTED, face=tint(MUTED, 0.96), lw=1.3,
          dashed=True, rounding=0.6, z=2)

    cx = ix + iw / 2
    ax.text(cx, iy + ih - 2.8, "Tied vs untied projection",
            ha="center", va="center", fontsize=8.4, fontweight="bold", color=INK)

    # ---- tied (one W for all channels) ----
    tx, ty, tw, th = ix + 3.0, iy + 10.0, 16.0, 6.0
    rrect(ax, tx, ty, tw, th, edge=GREEN, face=tint(GREEN, 0.8), lw=1.3,
          rounding=0.4)
    ax.text(tx + tw / 2, ty + th / 2, "Tied: W", ha="center", va="center",
            fontsize=8.0, fontweight="bold", color=INK)
    ax.text(tx + tw / 2, ty - 2.4, "one W for all channels", ha="center",
            va="center", fontsize=6.3, color=MUTED)
    ax.text(tx + tw / 2, ty + th + 2.4, "tied \u2713", ha="center", va="center",
            fontsize=6.8, style="italic", color=GREEN, fontweight="bold")

    # ---- untied (per-channel W_c) ----
    ux, uw, uh = ix + 23.0, 9.0, 3.0
    rrect(ax, ux, ty + 3.2, uw, uh, edge=CHANNEL_COLORS[0],
          face=tint(CHANNEL_COLORS[0], 0.8), lw=1.0, rounding=0.3)
    ax.text(ux + uw / 2, ty + 3.2 + uh / 2, "$W_0$", ha="center", va="center",
            fontsize=6.6, fontweight="bold", color=INK)
    rrect(ax, ux, ty + 0.2, uw, uh, edge=CHANNEL_COLORS[1],
          face=tint(CHANNEL_COLORS[1], 0.8), lw=1.0, rounding=0.3)
    ax.text(ux + uw / 2, ty + 0.2 + uh / 2, "$W_1$", ha="center", va="center",
            fontsize=6.6, fontweight="bold", color=INK)
    ax.text(ux + uw / 2, ty - 2.4, "per channel", ha="center", va="center",
            fontsize=6.3, color=MUTED)
    ax.text(ux + uw / 2, ty + th + 2.4, "untied", ha="center", va="center",
            fontsize=6.8, style="italic", color=MUTED)

    ax.text(cx, iy + 3.2,
            "ChannelViT: tied filters are more robust\nto varying channel sets",
            ha="center", va="center", fontsize=6.4, color=MUTED, style="italic",
            linespacing=1.5)


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
    ax.text(6.5, 77.5, "Tokenizer: channel-tied patch embed + additive channel embedding",
            ha="left", va="center", fontsize=12.2, fontweight="bold", color=INK)
    ax.text(2, 73.8,
            "Two channels share one projection W, but each adds its own channel "
            "embedding z_c; the spatial PE is added after fusion.",
            ha="left", va="center", fontsize=7.4, color=MUTED, style="italic")
    ax.text(2, 70.9,
            "No positional embedding here  |  DCP-omitted = absent row (never a zero card)",
            ha="left", va="center", fontsize=7.2, color=RED, style="italic",
            fontweight="bold")
    ax.text(2, 67.9,
            "Shapes:  $[1,H,W]$  \u2192  Conv2d  \u2192  $[N,D]$  "
            "$+[D]$ ($z_c$)  \u2192  ragged "
            "$[\\Sigma C \\cdot N, D]$     (D = token embedding dim)",
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

        # patch stack [N, D]
        st_y0 = yc - stack_htot / 2
        top = draw_patch_stack(ax, stack_x, st_y0, stack_w, stack_h,
                               stack_pitch, n_patches, color,
                               omitted_idx=(omit_idx if lane == 1 else None))
        ax.text(stack_x + stack_w / 2, top + 1.9, "$[N, D]$", ha="center",
                va="center", fontsize=6.8, color=MUTED)

        # + identity z_c  (per-channel colour)
        arrow(ax, (stack_x + stack_w, yc), (z_x, yc), color=color)
        ax.text((stack_x + stack_w + z_x) / 2, yc + 2.4, "$+$", ha="center",
                va="center", fontsize=9.5, fontweight="bold", color=INK)
        draw_identity_card(ax, z_x, yc - z_h / 2, z_w, z_h, color,
                           f"$z_{lane}$")
        ax.text(z_x - 0.5, yc - 3.9,
                f"channel id  \u00b7  $[D]$", ha="center", va="center",
                fontsize=6.0, color=MUTED)

        # -> output tokens (ragged list, omission preserved)
        arrow(ax, (z_x + z_w, yc), (out_x, yc), color=color)
        o_y0 = yc - stack_htot / 2
        top = draw_patch_stack(ax, out_x, o_y0, stack_w, stack_h,
                               stack_pitch, n_patches, color,
                               omitted_idx=(omit_idx if lane == 1 else None))
        ax.text(out_x + stack_w / 2, top + 1.9, "$[\\Sigma C \\cdot N, D]$",
                ha="center", va="center", fontsize=6.6, color=MUTED)
        ax.text(out_x + stack_w / 2, o_y0 - 1.9, "ragged tokens",
                ha="center", va="center", fontsize=6.6, fontweight="bold",
                color=color)

    # annotate the omitted row on the (upper) projected stack of channel 1
    row_y = lane_yc[1] - stack_htot / 2 + omit_idx * stack_pitch + stack_h / 2
    ax.plot([stack_x + stack_w, stack_x + stack_w + 1.1],
            [row_y, row_y - 0.6], color=RED, lw=1.0, zorder=7)
    ax.text(stack_x + stack_w + 1.3, row_y - 1.7, "omitted", ha="left",
            va="center", fontsize=6.1, color=RED, style="italic",
            fontweight="bold")

    # ---- Shared Conv2d module (one kernel for every channel) -------------- #
    rrect(ax, conv_x, 22.0, conv_w, 40.0, edge=TOKENIZER, face=tint(TOKENIZER, 0.86),
          lw=1.5, rounding=0.7, z=3)
    cx = conv_x + conv_w / 2
    ax.text(cx, 60.0, "Conv2d", ha="center", va="center", fontsize=8.6,
            fontweight="bold", color=INK, zorder=4)
    ax.text(cx, 56.6, "1 \u2192 D", ha="center", va="center", fontsize=6.8,
            color=INK, zorder=4)
    ax.text(cx, 54.2, "kernel P", ha="center", va="center", fontsize=6.6,
            color=MUTED, zorder=4)
    ax.text(cx, 51.8, "stride P", ha="center", va="center", fontsize=6.6,
            color=MUTED, zorder=4)
    ax.plot([cx - 2.8, cx + 2.8], [49.0, 49.0], color=TOKENIZER, lw=1.0,
            linestyle=(0, (3, 2)), zorder=4)
    ax.text(cx, 45.4, "shared W", ha="center", va="center", fontsize=7.6,
            fontweight="bold", color=TOKENIZER, zorder=4)
    ax.text(cx, 40.0, "one kernel for\nevery channel", ha="center", va="center",
            fontsize=6.3, color=MUTED, zorder=4, linespacing=1.5)
    ax.text(cx, 33.6, "(tied, not\nper-channel)", ha="center", va="center",
            fontsize=6.1, color=MUTED, zorder=4, linespacing=1.5)

    # ---- Identity embedding source ---------------------------------------- #
    emb_x, emb_y, emb_w, emb_h = 90.0, 37.0, 16.0, 10.0
    box(ax, emb_x, emb_y, emb_w, emb_h, "nn.Embedding", "num_channel_ids \u00d7 D",
        edge=TOKENIZER, title_size=7.8, sub_size=6.3)
    ax.text(emb_x + emb_w / 2, emb_y - 2.0, "$z_c$ lookup table",
            ha="center", va="center", fontsize=6.4, color=TOKENIZER,
            style="italic")

    # dashed lookup routes: table -> each z_c card (enter card from behind)
    route_x = z_x + z_w - 1.0
    dashed_route(ax, [(emb_x, emb_y + emb_h * 0.75), (route_x, emb_y + emb_h * 0.75),
                      (route_x, lane_yc[0] - z_h / 2)],
                 color=lane_color[0])
    dashed_route(ax, [(emb_x, emb_y + emb_h * 0.25), (route_x, emb_y + emb_h * 0.25),
                      (route_x, lane_yc[1] + z_h / 2)],
                 color=lane_color[1])

    # ---- Additive-token formula (centre band, clear of everything) -------- #
    rrect(ax, 29.0, 38.5, 12.0, 9.0, edge=MUTED, face=tint(MUTED, 0.97),
          lw=1.1, dashed=True, rounding=0.5, z=2)
    ax.text(35.0, 45.4, "per token", ha="center", va="center",
            fontsize=6.4, fontweight="bold", color=INK, zorder=4)
    ax.text(35.0, 42.9, "(additive)", ha="center", va="center",
            fontsize=6.4, fontweight="bold", color=INK, zorder=4)
    ax.text(35.0, 40.2, "proj(patch)\n$+$ $z_c$",
            ha="center", va="center", fontsize=5.9, color=MUTED, zorder=4,
            linespacing=1.5)

    # ---- DCP omission strip ----------------------------------------------- #
    strip_x, strip_y, strip_w, strip_h = 4.0, 2.0, 102.0, 16.0
    rrect(ax, strip_x, strip_y, strip_w, strip_h, edge=INK,
          face=tint(INK, 0.97), lw=1.2, rounding=0.6, z=2)
    ax.text(strip_x + 2.0, strip_y + strip_h - 2.6,
            "DCP omission = absent row", ha="left", va="center", fontsize=7.8,
            fontweight="bold", color=INK, zorder=4)

    # mini ragged stack: omitted row dashed red, never a zero card
    mini_specs = [(GREEN, False), (RED, True), (GREEN, False), (GREEN, False)]
    token_rows(ax, strip_x + 2.0, strip_y + 4.4, 6.0, 1.3, 1.8, mini_specs,
               rounding=0.3, z=4)
    ax.text(strip_x + 2.0, strip_y + 2.6, "omitted\nrow", ha="left", va="center",
            fontsize=5.7, color=RED, style="italic", zorder=4, linespacing=1.4)

    ax.text(strip_x + 12.5, strip_y + 11.2,
            "Masked channel-patch tokens are removed from the ragged list "
            "$[\\Sigma C \\cdot N, D]$.", ha="left", va="center", fontsize=6.4,
            color=INK, zorder=4)
    ax.text(strip_x + 12.5, strip_y + 8.4,
            "Never zero cards \u00b7 never padded \u00b7 no $-\\infty$ attention mask.",
            ha="left", va="center", fontsize=6.4, color=MUTED, zorder=4)
    ax.text(strip_x + 12.5, strip_y + 5.6,
            "$N = (H/P)^2$ patches per channel  \u00b7  $D$ = token dim.",
            ha="left", va="center", fontsize=6.4, color=MUTED, zorder=4)

    # dashed ghost of the padded tensor that is never materialised
    padded_ghost(ax, strip_x + 80.0, strip_y + 3.0, 18.0, 10.0, z=3)

    # ---- Right column ----------------------------------------------------- #
    # Key design points
    leg_x, leg_y, leg_w, leg_h = 110.0, 54.0, 38.0, 24.0
    rrect(ax, leg_x, leg_y, leg_w, leg_h, edge=INK, face=tint(INK, 0.97),
          lw=1.3, rounding=0.6, z=2)
    ax.text(leg_x + leg_w / 2, leg_y + leg_h - 2.8, "Key design points",
            ha="center", va="center", fontsize=8.6, fontweight="bold", color=INK)

    bullets = [
        "\u2022 One shared Conv2d W for every channel (tied)",
        "\u2022 spatial pos is added after channel fusion",
        "\u2022 $z_c$ = Embedding(channel_id), added per token",
        "\u2022 DCP omission \u2192 absent row, never a zero card",
        "\u2022 No padding tokens, no zero embeddings",
        "\u2022 $C$ lives only in the ragged length $\\Sigma C \\cdot N$",
    ]
    bullet_y = leg_y + leg_h - 6.0
    for bullet in bullets:
        ax.text(leg_x + 1.8, bullet_y, bullet, ha="left", va="center",
                fontsize=6.5, color=INK)
        bullet_y -= 2.7

    # Legends: roles + channel identity
    role_legend(ax, 112.0, 50.0,
                [("tokenizer / $z_c$", TOKENIZER),
                 ("channel fusion", FUSION),
                 ("encoder", ENCODER),
                 ("decoder", DECODER),
                 ("loss / omitted", LOSS)],
                title="module roles", pitch=2.4, sw=1.4, fontsize=6.2)
    channel_legend(ax, 131.0, 50.0, n=2, title="channel identity",
                   pitch=2.4, sw=1.4, fontsize=6.2)

    # ---- Tied vs untied inset --------------------------------------------- #
    draw_tied_inset(ax, 110.0, 6.0, 38.0, 26.0)

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