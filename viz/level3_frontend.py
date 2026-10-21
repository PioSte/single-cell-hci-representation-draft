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
    # Faint image content squiggle
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
    rrect(ax, x, y, w, h, edge=color, face=tint(color, 0.75), lw=1.5, rounding=0.4)
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            fontsize=7.5, fontweight="bold", color=INK, zorder=4)


def draw_tied_inset(ax):
    """Small inset comparing tied vs untied projections."""
    # Box frame
    ix, iy, iw, ih = 88, 5, 34, 20
    rrect(ax, ix, iy, iw, ih, edge=MUTED, face=tint(MUTED, 0.96), lw=1.3,
          dashed=True, rounding=0.6, z=2)
    
    ax.text(ix + iw / 2, iy + ih - 2.5, "Tied vs untied projection",
            ha="center", va="center", fontsize=8.0, fontweight="bold", color=INK)
    
    # Tied: one W
    ty, th = iy + 9, 5
    rrect(ax, ix + 2, ty, 13, th, edge=GREEN, face=tint(GREEN, 0.8), lw=1.2, rounding=0.4)
    ax.text(ix + 8.5, ty + th / 2, "Tied: W", ha="center", va="center",
            fontsize=7.2, fontweight="bold", color=INK)
    ax.text(ix + 8.5, ty - 1.8, "all channels", ha="center", va="center",
            fontsize=6.0, color=MUTED)
    
    # Untied: W_c per channel
    uy = ty
    rrect(ax, ix + 17, uy, 6, th * 0.45, edge=BLUE, face=tint(BLUE, 0.8), lw=0.9, rounding=0.3)
    ax.text(ix + 20, uy + th * 0.225, "$W_0$", ha="center", va="center",
            fontsize=6.0, fontweight="bold", color=INK)
    rrect(ax, ix + 17, uy + th * 0.55, 6, th * 0.45, edge=GREEN, face=tint(GREEN, 0.8),
          lw=0.9, rounding=0.3)
    ax.text(ix + 20, uy + th * 0.775, "$W_1$", ha="center", va="center",
            fontsize=6.0, fontweight="bold", color=INK)
    ax.text(ix + 20, uy - 1.8, "per channel", ha="center", va="center",
            fontsize=6.0, color=MUTED)
    ax.text(ix + 20, uy + th + 1.8, "untied", ha="center", va="center",
            fontsize=6.4, style="italic", color=MUTED)
    ax.text(ix + 8.5, uy + th + 1.8, "tied ✓", ha="center", va="center",
            fontsize=6.4, style="italic", color=GREEN, fontweight="bold")
    
    # Note about robustness
    ax.text(ix + iw / 2, iy + 2.0,
            "ChannelViT: tied is more robust to varying channel sets",
            ha="center", va="center", fontsize=6.2, color=MUTED, style="italic")


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

    # ---- Subtitle / key concept ------------------------------------------- #
    ax.text(2, 63.5,
            "Two channels sharing the same projection W and spatial_pos, but "
            "each adding its own marker identity z_c.",
            ha="left", va="center", fontsize=7.4, color=MUTED, style="italic")
    ax.text(2, 60.8,
            "Position ≠ channel identity  |  DCP-omitted = absent row (never a zero card)",
            ha="left", va="center", fontsize=7.2, color=RED, style="italic",
            fontweight="bold")

    # ---- Channel 0 (BLUE) path -------------------------------------------- #
    c0_color = CHANNEL_COLORS[0]  # BLUE
    c0_y = 42
    tile_size = 9
    
    # Image tile
    draw_channel_tile(ax, 4, c0_y, tile_size, c0_color)
    ax.text(8.5, c0_y - 1.8, "$[1, H, W]$", ha="center", va="center",
            fontsize=6.8, color=MUTED)
    ax.text(8.5, c0_y + tile_size + 1.8, "channel 0", ha="center", va="center",
            fontsize=7.2, fontweight="bold", color=c0_color)
    
    # Conv2d projection (shared)
    arrow(ax, (13.5, c0_y + tile_size / 2), (21, c0_y + tile_size / 2), color=c0_color)
    
    # Patches after projection
    patch_x, patch_w, patch_h, patch_pitch = 21, 6, 1.6, 2.1
    n_patches = 4
    patch_top = draw_patch_stack(ax, patch_x, c0_y, patch_w, patch_h, patch_pitch,
                                  n_patches, c0_color, omitted_idx=None)
    ax.text(patch_x + patch_w / 2, patch_top + 1.8, "$[N, D]$", ha="center",
            va="center", fontsize=6.8, color=MUTED)
    
    # + spatial_pos
    arrow(ax, (28, c0_y + 4.5), (35, c0_y + 4.5), color=c0_color)
    ax.text(31.5, c0_y + 6.2, "$+$", ha="center", va="center",
            fontsize=9, fontweight="bold", color=INK)
    
    pos_x = 35
    rrect(ax, pos_x, c0_y + 0.5, 8, 7.5, edge=ORANGE, face=tint(ORANGE, 0.82),
          lw=1.3, rounding=0.4)
    ax.text(pos_x + 4, c0_y + 4.25, "spatial_pos", ha="center", va="center",
            fontsize=7.0, fontweight="bold", color=INK)
    ax.text(pos_x + 4, c0_y + 1.8, "$[1, N, D]$", ha="center", va="center",
            fontsize=6.2, color=MUTED)
    
    # + identity z_0
    arrow(ax, (44, c0_y + 4.5), (52, c0_y + 4.5), color=c0_color)
    ax.text(48, c0_y + 6.2, "$+$", ha="center", va="center",
            fontsize=9, fontweight="bold", color=INK)
    
    id_x = 52
    draw_identity_card(ax, id_x, c0_y + 2.5, 5.5, 4, c0_color, "$z_0$")
    ax.text(id_x + 2.75, c0_y + 0.8, "marker 0", ha="center", va="center",
            fontsize=6.0, color=MUTED)
    
    # Output tokens
    arrow(ax, (58, c0_y + 4.5), (64, c0_y + 4.5), color=c0_color)
    out_x = 64
    out_top = draw_patch_stack(ax, out_x, c0_y, patch_w, patch_h, patch_pitch,
                                n_patches, c0_color, omitted_idx=None)
    ax.text(out_x + patch_w / 2, out_top + 1.8, "tokens", ha="center",
            va="center", fontsize=7.2, fontweight="bold", color=c0_color)

    # ---- Channel 1 (GREEN) path ------------------------------------------- #
    c1_color = CHANNEL_COLORS[1]  # GREEN
    c1_y = 26
    
    # Image tile
    draw_channel_tile(ax, 4, c1_y, tile_size, c1_color)
    ax.text(8.5, c1_y - 1.8, "$[1, H, W]$", ha="center", va="center",
            fontsize=6.8, color=MUTED)
    ax.text(8.5, c1_y + tile_size + 1.8, "channel 1", ha="center", va="center",
            fontsize=7.2, fontweight="bold", color=c1_color)
    
    # Conv2d projection (shared - same box as above)
    arrow(ax, (13.5, c1_y + tile_size / 2), (21, c1_y + tile_size / 2), color=c1_color)
    
    # Patches after projection (with one omitted)
    omit_idx = 2
    patch_top_1 = draw_patch_stack(ax, patch_x, c1_y, patch_w, patch_h, patch_pitch,
                                    n_patches, c1_color, omitted_idx=omit_idx)
    # Label the omitted row
    ax.text(patch_x + patch_w + 0.8, c1_y + omit_idx * patch_pitch + patch_h / 2,
            "omitted", ha="left", va="center", fontsize=6.0, color=RED, style="italic")
    
    # + spatial_pos (same box as above, arrow from c1)
    arrow(ax, (28, c1_y + 4.5), (35, c1_y + 4.5), color=c1_color)
    ax.text(31.5, c1_y + 6.2, "$+$", ha="center", va="center",
            fontsize=9, fontweight="bold", color=INK)
    
    # + identity z_1
    arrow(ax, (44, c1_y + 4.5), (52, c1_y + 4.5), color=c1_color)
    ax.text(48, c1_y + 6.2, "$+$", ha="center", va="center",
            fontsize=9, fontweight="bold", color=INK)
    
    draw_identity_card(ax, id_x, c1_y + 2.5, 5.5, 4, c1_color, "$z_1$")
    ax.text(id_x + 2.75, c1_y + 0.8, "marker 1", ha="center", va="center",
            fontsize=6.0, color=MUTED)
    
    # Output tokens (with omitted row still absent)
    arrow(ax, (58, c1_y + 4.5), (64, c1_y + 4.5), color=c1_color)
    out_top_1 = draw_patch_stack(ax, out_x, c1_y, patch_w, patch_h, patch_pitch,
                                  n_patches, c1_color, omitted_idx=omit_idx)
    ax.text(out_x + patch_w / 2, out_top_1 + 1.8, "tokens", ha="center",
            va="center", fontsize=7.2, fontweight="bold", color=c1_color)
    ax.text(out_x + patch_w + 0.8, c1_y + omit_idx * patch_pitch + patch_h / 2,
            "DCP omission", ha="left", va="center", fontsize=6.0, color=RED,
            style="italic")

    # ---- Shared Conv2d module box ----------------------------------------- #
    box(ax, 15, 18, 11, 38, "Conv2d", "1 → D\nkernel P\nstride P\n\nshared W",
        edge=GREEN, title_size=8.5, sub_size=6.6)

    # ---- Shared spatial_pos annotation ------------------------------------ #
    ax.text(pos_x + 4, c0_y + 8.5, "shared across channels", ha="center",
            va="center", fontsize=6.6, color=ORANGE, fontweight="bold",
            style="italic")
    
    # ---- Identity embedding source ---------------------------------------- #
    # nn.Embedding table
    emb_x, emb_y = 77, 30
    box(ax, emb_x, emb_y, 13, 10, "nn.Embedding", "num_markers × D",
        edge=PURPLE, title_size=7.8, sub_size=6.2)
    
    # Arrows from identity cards to embedding
    arrow(ax, (id_x + 2.75, c0_y + 2.5), (emb_x + 2, emb_y + 8), color=c0_color,
          lw=1.3, rad=-0.2, dashed=True)
    arrow(ax, (id_x + 2.75, c1_y + 6.5), (emb_x + 2, emb_y + 2), color=c1_color,
          lw=1.3, rad=0.2, dashed=True)
    ax.text(emb_x - 8, emb_y + 5, "lookup", ha="center", va="center",
            fontsize=6.6, color=PURPLE, style="italic")

    # ---- "Never a zero card" emphasis ------------------------------------- #
    zero_x, zero_y = 77, 14
    rrect(ax, zero_x, zero_y, 13, 8, edge=RED, face=tint(RED, 0.92), lw=2.0,
          dashed=True)
    ax.plot([zero_x + 1.5, zero_x + 11.5], [zero_y + 1.5, zero_y + 6.5],
            color=RED, lw=2.5, zorder=5)
    ax.plot([zero_x + 1.5, zero_x + 11.5], [zero_y + 6.5, zero_y + 1.5],
            color=RED, lw=2.5, zorder=5)
    ax.text(zero_x + 6.5, zero_y + 4, "zero card", ha="center", va="center",
            fontsize=7.0, fontweight="bold", color=MUTED, zorder=4)
    ax.text(zero_x + 6.5, zero_y - 1.5, "never used", ha="center", va="center",
            fontsize=6.8, color=RED, fontweight="bold", style="italic")

    # ---- Tied vs untied inset --------------------------------------------- #
    draw_tied_inset(ax)

    # ---- Legend / key concepts box ---------------------------------------- #
    leg_x, leg_y, leg_w, leg_h = 93, 42, 35, 16
    rrect(ax, leg_x, leg_y, leg_w, leg_h, edge=INK, face=tint(INK, 0.97),
          lw=1.3, rounding=0.6, z=2)
    
    ax.text(leg_x + leg_w / 2, leg_y + leg_h - 2.2, "Key design points",
            ha="center", va="center", fontsize=8.5, fontweight="bold", color=INK)
    
    # Bullet points
    bullets = [
        "• Projection W is tied (shared) across all channels",
        "• Spatial position is shared (position ≠ identity)",
        "• Channel identity z_c is additive (per marker)",
        "• DCP omission → absent row (never zeroed)",
        "• No padding tokens, no zero embeddings",
    ]
    
    bullet_y = leg_y + leg_h - 5.0
    for bullet in bullets:
        ax.text(leg_x + 1.5, bullet_y, bullet, ha="left", va="center",
                fontsize=6.6, color=INK)
        bullet_y -= 2.2

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
