"""L3 -- patch embed + additive identity frontend detail.

Shows the channel-tied patch embedding (one shared Conv2d projection for all
channels) plus the additive channel-identity embeddings. Emphasizes:
- Position != channel identity (spatial_pos is shared; z_c is per-marker).
- DCP-omitted tokens are drawn as absent dashed rows, never zero cards.
- Tied vs untied projection comparison (ChannelViT shows tied is more robust).

Usage::

    uv run --extra viz python viz/level3_frontend.py
    uv run --extra viz python viz/level3_frontend.py --formats pdf svg png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from figure_style import (  # noqa: E402
    BLUE, GREEN, ORANGE, RED, PURPLE, VIOLET, INK, MUTED, CHANNEL_COLORS,
    tint, rrect, box, arrow, edge_label, sq_bracket, token_rows, install_rcparams,
    save_figure,
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
    """Vertical stack of N patch tokens."""
    specs = []
    for i in range(n):
        omitted = (omitted_idx is not None and i == omitted_idx)
        specs.append((color, omitted))
    top = token_rows(ax, x, y0, w, h, pitch, specs, rounding=0.35, z=3)
    return top


def draw_identity_card(ax, x, y, w, h, color, label):
    """A small identity embedding card z_c."""
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
    ax.text(cx, iy + ih - 2.6, "Tied vs untied projection",
            ha="center", va="center", fontsize=8.2, fontweight="bold", color=INK)

    # ---- tied (one W for all channels) ----
    tx, ty, tw, th = ix + 2.5, iy + 9.0, 14.0, 5.0
    rrect(ax, tx, ty, tw, th, edge=GREEN, face=tint(GREEN, 0.8), lw=1.3,
          rounding=0.4)
    ax.text(tx + tw / 2, ty + th / 2, "Tied: W", ha="center", va="center",
            fontsize=7.4, fontweight="bold", color=INK)
    ax.text(tx + tw / 2, ty - 2.2, "all channels", ha="center", va="center",
            fontsize=6.2, color=MUTED)
    ax.text(tx + tw / 2, ty + th + 2.2, "tied \u2713", ha="center", va="center",
            fontsize=6.6, style="italic", color=GREEN, fontweight="bold")

    # ---- untied (per-channel W_c) ----
    ux, uw, uh = ix + 19.5, 6.5, 2.5
    rrect(ax, ux, ty + 3.0, uw, uh, edge=BLUE, face=tint(BLUE, 0.8), lw=1.0,
          rounding=0.3)
    ax.text(ux + uw / 2, ty + 3.0 + uh / 2, "$W_0$", ha="center", va="center",
            fontsize=6.4, fontweight="bold", color=INK)
    rrect(ax, ux, ty + 0.2, uw, uh, edge=GREEN, face=tint(GREEN, 0.8), lw=1.0,
          rounding=0.3)
    ax.text(ux + uw / 2, ty + 0.2 + uh / 2, "$W_1$", ha="center", va="center",
            fontsize=6.4, fontweight="bold", color=INK)
    ax.text(ux + uw / 2, ty - 2.2, "per channel", ha="center", va="center",
            fontsize=6.2, color=MUTED)
    ax.text(ux + uw / 2, ty + th + 2.2, "untied", ha="center", va="center",
            fontsize=6.6, style="italic", color=MUTED)

    ax.text(cx, iy + 2.8,
            "ChannelViT: tied filters are more robust\nto varying channel sets",
            ha="center", va="center", fontsize=6.3, color=MUTED, style="italic",
            linespacing=1.5)


# --------------------------------------------------------------------------- #
# Main figure
# --------------------------------------------------------------------------- #
def build() -> plt.Figure:
    fig = plt.figure(figsize=(13, 7))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_xlim(0, 130)
    ax.set_ylim(0, 70)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    # ---- Title ------------------------------------------------------------ #
    ax.text(2, 67, "L3", ha="left", va="center", fontsize=15,
            fontweight="bold", color=INK)
    ax.text(6.5, 67, "Frontend: channel-tied patch embed + additive identity",
            ha="left", va="center", fontsize=12.2, fontweight="bold", color=INK)
    ax.text(2, 63.6,
            "Two channels share one projection W and one spatial_pos, but each "
            "adds its own marker identity z_c.",
            ha="left", va="center", fontsize=7.4, color=MUTED, style="italic")
    ax.text(2, 60.9,
            "Position \u2260 channel identity  |  DCP-omitted = absent row (never a zero card)",
            ha="left", va="center", fontsize=7.2, color=RED, style="italic",
            fontweight="bold")

    # ---- Shared geometry -------------------------------------------------- #
    lane_yc = [51.0, 25.0]                     # channel 0 / channel 1 centres
    lane_color = [CHANNEL_COLORS[0], CHANNEL_COLORS[1]]
    lane_name = ["channel 0", "channel 1"]
    tile_x, tile_s = 3.0, 9.0
    stack_x, stack_w, stack_h, stack_pitch, n_patches = 26.0, 6.0, 1.6, 2.1, 4
    sp_x, sp_w, sp_h = 40.0, 10.0, 8.0        # spatial_pos box
    z_x, z_w, z_h = 56.0, 6.0, 4.0            # identity card
    out_x = 66.0                              # output token stack
    omit_idx = 1                              # DCP-omitted row in channel 1

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
        arrow(ax, (tile_x + tile_s, yc), (15.0, yc), color=color)
        arrow(ax, (23.5, yc), (stack_x, yc), color=color)

        st_y0 = yc - (n_patches - 1) * stack_pitch / 2 - stack_h / 2
        top = draw_patch_stack(ax, stack_x, st_y0, stack_w, stack_h,
                               stack_pitch, n_patches, color,
                               omitted_idx=(omit_idx if lane == 1 else None))
        ax.text(stack_x + stack_w / 2, top + 1.9, "$[N, D]$", ha="center",
                va="center", fontsize=6.8, color=MUTED)

        # + spatial_pos
        arrow(ax, (stack_x + stack_w, yc), (sp_x, yc), color=color)
        ax.text((stack_x + stack_w + sp_x) / 2, yc + 2.6, "$+$", ha="center",
                va="center", fontsize=9.5, fontweight="bold", color=INK)
        rrect(ax, sp_x, yc - sp_h / 2, sp_w, sp_h, edge=ORANGE,
              face=tint(ORANGE, 0.82), lw=1.3, rounding=0.4)
        ax.text(sp_x + sp_w / 2, yc + 1.8, "spatial_pos", ha="center",
                va="center", fontsize=7.2, fontweight="bold", color=INK,
                zorder=4)
        ax.text(sp_x + sp_w / 2, yc - 1.8, "$[1, N, D]$", ha="center",
                va="center", fontsize=6.3, color=MUTED, zorder=4)

        # + identity z_c
        arrow(ax, (sp_x + sp_w, yc), (z_x, yc), color=color)
        ax.text((sp_x + sp_w + z_x) / 2, yc + 2.6, "$+$", ha="center",
                va="center", fontsize=9.5, fontweight="bold", color=INK)
        draw_identity_card(ax, z_x, yc - z_h / 2, z_w, z_h, color,
                           f"$z_{lane}$")
        ax.text(z_x + z_w / 2, yc - z_h / 2 - 2.0, f"marker {lane}",
                ha="center", va="center", fontsize=6.1, color=MUTED)

        # -> output tokens (ragged list, omission preserved)
        arrow(ax, (z_x + z_w, yc), (out_x, yc), color=color)
        o_y0 = yc - (n_patches - 1) * stack_pitch / 2 - stack_h / 2
        top = draw_patch_stack(ax, out_x, o_y0, stack_w, stack_h,
                               stack_pitch, n_patches, color,
                               omitted_idx=(omit_idx if lane == 1 else None))
        ax.text(out_x + stack_w / 2, top + 1.9, "tokens", ha="center",
                va="center", fontsize=7.2, fontweight="bold", color=color)

    # annotate the omitted row on the (upper) projected stack of channel 1
    row_y = lane_yc[1] - (n_patches - 1) * stack_pitch / 2 - stack_h / 2 \
        + omit_idx * stack_pitch + stack_h / 2
    ax.plot([stack_x + stack_w, stack_x + stack_w + 1.1],
            [row_y, row_y - 0.6], color=RED, lw=1.0, zorder=7)
    ax.text(stack_x + stack_w + 1.3, row_y - 1.0, "omitted", ha="left",
            va="center", fontsize=6.1, color=RED, style="italic",
            fontweight="bold")

    # ---- Shared Conv2d module (one kernel for every channel) -------------- #
    rrect(ax, 15.0, 19.0, 8.5, 38.0, edge=GREEN, face=tint(GREEN, 0.86),
          lw=1.5, rounding=0.7, z=3)
    cx = 19.25
    ax.text(cx, 54.6, "Conv2d", ha="center", va="center", fontsize=8.6,
            fontweight="bold", color=INK, zorder=4)
    ax.text(cx, 50.4, "1 \u2192 D", ha="center", va="center", fontsize=6.8,
            color=INK, zorder=4)
    ax.text(cx, 47.9, "kernel P", ha="center", va="center", fontsize=6.6,
            color=MUTED, zorder=4)
    ax.text(cx, 45.4, "stride P", ha="center", va="center", fontsize=6.6,
            color=MUTED, zorder=4)
    ax.plot([cx - 2.8, cx + 2.8], [42.5, 42.5], color=GREEN, lw=1.0,
            linestyle=(0, (3, 2)), zorder=4)
    ax.text(cx, 38.8, "shared W", ha="center", va="center", fontsize=7.6,
            fontweight="bold", color=GREEN, zorder=4)
    ax.text(cx, 33.2, "one kernel for\nevery channel", ha="center", va="center",
            fontsize=6.3, color=MUTED, zorder=4, linespacing=1.5)
    ax.text(cx, 26.5, "(tied, not\nper-channel)", ha="center", va="center",
            fontsize=6.1, color=MUTED, zorder=4, linespacing=1.5)

    # ---- Shared spatial_pos emphasis -------------------------------------- #
    rrect(ax, sp_x - 1.5, lane_yc[1] - sp_h / 2 - 1.5, sp_w + 3.0,
          (lane_yc[0] - lane_yc[1]) + sp_h + 3.0, edge=ORANGE,
          face="none", lw=1.1, dashed=True, rounding=0.6, z=2)
    ax.text(sp_x + sp_w / 2, (lane_yc[0] + lane_yc[1]) / 2,
            "spatial_pos is shared\n(position \u2260 identity)", ha="center",
            va="center", fontsize=6.6, color=ORANGE, fontweight="bold",
            style="italic", linespacing=1.5)

    # ---- Identity embedding source ---------------------------------------- #
    emb_x, emb_y, emb_w, emb_h = 76.0, 34.0, 14.0, 10.0
    box(ax, emb_x, emb_y, emb_w, emb_h, "nn.Embedding", "num_markers \u00d7 D",
        edge=PURPLE, title_size=7.8, sub_size=6.3)

    # dashed lookup routes: table -> each z_c card (routed clear of tokens)
    route_x = z_x + z_w - 1.0
    dashed_route(ax, [(emb_x + 1.0, emb_y + emb_h), (emb_x + 1.0, 58.3),
                      (route_x, 58.3), (route_x, lane_yc[0] + z_h / 2)],
                 color=lane_color[0])
    dashed_route(ax, [(emb_x + 1.0, emb_y), (emb_x + 1.0, 18.5),
                      (route_x, 18.5), (route_x, lane_yc[1] - z_h / 2)],
                 color=lane_color[1])
    ax.text(emb_x + emb_w / 2, emb_y + emb_h + 1.7,
            "$z_c$ lookup", ha="center", va="center", fontsize=6.4,
            color=PURPLE, style="italic")

    # ---- Additive-token formula (fills the centre) ------------------------ #
    rrect(ax, 53.0, 32.5, 21.0, 10.0, edge=MUTED, face=tint(MUTED, 0.97),
          lw=1.1, dashed=True, rounding=0.5, z=2)
    ax.text(63.5, 39.4, "per token (additive):", ha="center", va="center",
            fontsize=6.6, fontweight="bold", color=INK, zorder=4)
    ax.text(63.5, 35.7, "proj(patch) + spatial_pos[n] + $z_c$", ha="center",
            va="center", fontsize=6.0, color=MUTED, zorder=4)

    # ---- DCP omission strip ----------------------------------------------- #
    strip_x, strip_y, strip_w, strip_h = 3.0, 3.0, 90.0, 13.0
    rrect(ax, strip_x, strip_y, strip_w, strip_h, edge=INK,
          face=tint(INK, 0.97), lw=1.2, rounding=0.6, z=2)
    ax.text(strip_x + 2.0, strip_y + strip_h - 2.4,
            "DCP omission = absent row", ha="left", va="center", fontsize=7.8,
            fontweight="bold", color=INK, zorder=4)

    # mini ragged stack: omitted row dashed, never a zero card
    mini_specs = [(GREEN, False), (GREEN, True), (GREEN, False), (GREEN, False)]
    token_rows(ax, strip_x + 2.0, strip_y + 2.0, 6.0, 1.3, 1.7, mini_specs,
               rounding=0.3, z=4)
    ax.text(strip_x + 2.0, strip_y + 1.2, "omitted\nrow", ha="left", va="center",
            fontsize=5.8, color=RED, style="italic", zorder=4, linespacing=1.4)

    ax.text(strip_x + 12.5, strip_y + 8.2,
            "Masked channel-patch tokens are removed from the ragged list "
            "$[\\Sigma C \\cdot N, D]$.", ha="left", va="center", fontsize=6.4,
            color=INK, zorder=4)
    ax.text(strip_x + 12.5, strip_y + 5.2,
            "Never zero cards \u00b7 never padded \u00b7 no $-\\infty$ attention mask.",
            ha="left", va="center", fontsize=6.4, color=MUTED, zorder=4)

    # dashed ghost of the padded tensor that is never materialised
    ghost_x, ghost_y, ghost_w, ghost_h = strip_x + 75.0, strip_y + 2.6, 12.0, 7.4
    rrect(ax, ghost_x, ghost_y, ghost_w, ghost_h, edge=MUTED,
          face=tint(MUTED, 0.99), lw=1.2, dashed=True, rounding=0.4, z=3)
    ax.plot([ghost_x + 1.0, ghost_x + ghost_w - 1.0],
            [ghost_y + 1.0, ghost_y + ghost_h - 1.0], color=RED, lw=1.6,
            alpha=0.7, zorder=5)
    ax.text(ghost_x + ghost_w / 2, ghost_y + ghost_h / 2,
            "$[B, C_{max}, N, D]$", ha="center", va="center", fontsize=6.0,
            color=MUTED, zorder=4)
    ax.text(ghost_x + ghost_w / 2, ghost_y + ghost_h + 1.2,
            "never materialised", ha="center", va="center", fontsize=5.9,
            color=RED, fontweight="bold", style="italic", zorder=4)

    # ---- Tied vs untied inset --------------------------------------------- #
    draw_tied_inset(ax, 95.0, 6.0, 33.0, 22.0)

    # ---- Key design points ------------------------------------------------- #
    leg_x, leg_y, leg_w, leg_h = 95.0, 33.0, 33.0, 26.0
    rrect(ax, leg_x, leg_y, leg_w, leg_h, edge=INK, face=tint(INK, 0.97),
          lw=1.3, rounding=0.6, z=2)
    ax.text(leg_x + leg_w / 2, leg_y + leg_h - 2.6, "Key design points",
            ha="center", va="center", fontsize=8.6, fontweight="bold", color=INK)

    bullets = [
        "\u2022 One shared Conv2d W for every channel (tied)",
        "\u2022 spatial_pos $[1, N, D]$ is shared (\u2260 identity)",
        "\u2022 $z_c$ = Embedding(marker_id) added per token",
        "\u2022 DCP omission \u2192 absent row, never a zero card",
        "\u2022 No padding tokens, no zero embeddings",
        "\u2022 $C$ lives only in the ragged input length $\\Sigma C \\cdot N$",
    ]
    bullet_y = leg_y + leg_h - 5.6
    for bullet in bullets:
        ax.text(leg_x + 1.6, bullet_y, bullet, ha="left", va="center",
                fontsize=6.5, color=INK)
        bullet_y -= 2.6

    return fig


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--formats", nargs="+", default=["pdf", "svg", "png"])
    ap.add_argument("--stem", default="level3_frontend")
    args = ap.parse_args()

    install_rcparams()

    fig = build()
    save_figure(fig, OUT, args.stem, args.formats)
    plt.close(fig)


if __name__ == "__main__":
    main()
