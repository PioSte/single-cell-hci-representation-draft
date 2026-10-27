"""L9 -- FlexiViT: variable patch size without resizing positional tables.

One canonical patch-embed kernel ``W_P`` is stored for
``Conv2d(1, D, kernel_size=P, stride=P)``.  At a runtime patch size ``p`` the
kernel is **PI-resized** -- ``W_p = pinv(B.T) @ W_P`` where ``B`` is the
bilinear resize operator on the patch grid -- and the shared bias is kept.
The layout modules need no resized table: the encoder uses 2-D axial RoPE on
normalized ``(row, col)``, memory tokens get zero rotation, the decoder uses a
factorized fractional absolute PE from ``(r, c, g_h, g_w)``, and
``ChannelFusion`` adds no spatial table.

Panel A: the canonical kernel, the PI-resize operator and the runtime conv.
Panel B: where ``p`` comes from, what is *not* resized, and scale conditioning.
Panel C: the variable-patch-size contract (input PI-resize vs output bilinear).

Usage::

    uv run --extra viz python viz/level9_flexivit.py
    uv run --extra viz python viz/level9_flexivit.py --formats pdf svg png
    uv run --extra viz python viz/level9_flexivit.py --stem level9_flexivit
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from figure_style import (  # noqa: E402
    BLUE, GREEN, ORANGE, PURPLE, VIOLET, INK, MUTED,
    TOKENIZER, FUSION, ENCODER, DECODER, LOSS, FUSED, GHOST,
    tint, rrect, box, arrow, sq_bracket_mirror, panel_header, legend,
    channel_color, install_rcparams, save_figure, token_rows, grid_cells,
    padded_ghost,
)

OUT = Path(__file__).resolve().parent / "out"


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def note_box(ax, x, y, w, h, title, lines, edge, title_size=7.4,
             body_size=5.7, pitch=1.6):
    """Plain module/annotation card with a bold title and left-aligned lines."""
    rrect(ax, x, y, w, h, edge=edge, face=tint(edge, 0.90), lw=1.25,
          rounding=0.5, z=2)
    ax.text(x + 1.3, y + h - 1.55, title, ha="left", va="center",
            fontsize=title_size, fontweight="bold", color=INK, zorder=4)
    yy = y + h - 3.55
    for line in lines:
        ax.text(x + 1.3, yy, line, ha="left", va="center", fontsize=body_size,
                color=INK, zorder=4)
        yy -= pitch


def cell(ax, x, y, w, h, text, edge, face=None, size=5.4, bold=False,
         align="left", color=INK, z=3):
    rrect(ax, x, y, w, h, edge=edge,
          face=face if face is not None else "white", lw=0.9, rounding=0.35,
          z=z)
    if align == "left":
        ax.text(x + 1.0, y + h / 2, text, ha="left", va="center",
                fontsize=size, color=color, zorder=z + 2,
                fontweight="bold" if bold else "normal")
    else:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=size, color=color, zorder=z + 2,
                fontweight="bold" if bold else "normal")


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #
def build() -> plt.Figure:
    fig = plt.figure(figsize=(16, 9.2))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_xlim(0, 160)
    ax.set_ylim(0, 92)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    # ---- title ------------------------------------------------------------- #
    panel_header(ax, 2, 89.4, "L9",
                 "FlexiViT: variable patch size without resizing positional tables")

    # ======================================================================= #
    # Panel A -- one canonical kernel, PI-resized per runtime patch size
    # ======================================================================= #
    rrect(ax, 2, 56.0, 156, 30.3, edge=TOKENIZER, face=tint(TOKENIZER, 0.97),
          lw=1.5, dashed=True, rounding=0.9, z=1)
    panel_header(ax, 4.5, 84.3, "A",
                 "one canonical kernel, PI-resized per runtime patch size")

    # --- top row: stored kernel -> PI-resize -------------------------------- #
    k_top, k_right = grid_cells(ax, 8.0, 73.3, 3, 3, cw=2.2, pitch=2.6,
                                edge=TOKENIZER, crossed=True)
    ax.text(11.7, 81.7, "canonical stored kernel  $W_P$", ha="center",
            va="center", fontsize=7.0, fontweight="bold", color=INK)
    ax.text(11.7, 72.4, r"$[D,1,P,P]$  $k{=}s{=}P$", ha="center", va="center",
            fontsize=5.9, color=MUTED)
    arrow(ax, (k_right + 0.3, 77.0), (21.8, 77.0), color=TOKENIZER, lw=1.5)

    box(ax, 22.0, 71.8, 34.0, 10.0,
        r"PI-resize  $W_p=\mathrm{PI}(W_P;\,P{\to}p)$",
        r"$\mathrm{PI\text{-}resize}=\mathrm{pinv}(B^\top)$" "\n"
        r"$B$ = bilinear resize on patch grid" "\n"
        r"bias is shared, never resized",
        edge=TOKENIZER, title_size=7.7, sub_size=5.8)

    # PI-resize feeds the runtime conv below.
    arrow(ax, (39.0, 71.8), (39.0, 68.6), color=TOKENIZER, lw=1.4)
    ax.text(40.0, 70.2, "$W_p$", ha="left", va="center", fontsize=6.4,
            color=TOKENIZER, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.12", facecolor="white",
                      edgecolor="none", alpha=0.9), zorder=7)

    # --- bottom row: image -> conv; z_c carried -> ragged ------------------- #
    i_top, i_right = grid_cells(ax, 8.0, 59.4, 3, 3, cw=2.2, pitch=2.6,
                                edge=BLUE, crossed=True)
    ax.text(11.7, 68.3, "single-channel image  $[1,H,W]$", ha="center",
            va="center", fontsize=6.6, fontweight="bold", color=INK)
    ax.text(11.7, 58.4, "one channel at a time", ha="center", va="center",
            fontsize=5.7, color=MUTED)
    arrow(ax, (i_right + 0.3, 63.2), (21.8, 63.2), color=TOKENIZER, lw=1.5)

    box(ax, 22.0, 58.3, 34.0, 10.0,
        r"$\mathtt{F.conv2d}(x,\,W_p,\,\mathtt{bias},\,\mathtt{stride}{=}p)$",
        r"one shared kernel, tied across" "\n"
        r"every channel  $\to\ [N_b,D]$ per channel",
        edge=TOKENIZER, title_size=7.0, sub_size=5.9)

    arrow(ax, (56.3, 63.2), (58.3, 63.2), color=TOKENIZER, lw=1.5)
    box(ax, 58.3, 58.3, 15.5, 10.0, "carry  $z_c$",
        r"$\mathtt{ChannelEmbedding}$" "\n"
        r"routing bias + skip" "\n"
        r"not added to token",
        edge=TOKENIZER, title_size=8.2, sub_size=5.5)

    # ragged, crossed token column -- the primary object
    specs = [(channel_color(1), False), (channel_color(2), False),
             (channel_color(0), False), (channel_color(3), False)]
    rx, ry0, rw, rh, rpitch = 78.5, 58.3, 8.5, 2.3, 2.9
    r_top = token_rows(ax, rx, ry0, rw, rh, rpitch, specs, crossed=True)
    sq_bracket_mirror(ax, rx + rw + 0.8, ry0, r_top)
    ax.text(rx + rw / 2, r_top + 2.1, "ragged  $[\\Sigma C\\cdot N,\\,D]$",
            ha="center", va="center", fontsize=7.0, fontweight="bold",
            color=INK)
    ax.text(rx + rw / 2, r_top + 0.7, "visible channel-patches (primary)",
            ha="center", va="center", fontsize=5.7, color=MUTED)
    arrow(ax, (74.2, 63.2), (78.3, 63.2), color=TOKENIZER, lw=1.5)

    # --- right: what p means + the never-allocated padded ghost ------------- #
    ax.text(104.0, 75.0, "$p$ = runtime patch size;  may differ per sample",
            ha="left", va="center", fontsize=6.8, fontweight="bold",
            color=INK)
    ax.text(104.0, 72.6,
            r"grid $g_h\times g_w = N_b=(H/p)(W/p)$",
            ha="left", va="center", fontsize=6.2, color=MUTED)
    padded_ghost(ax, 104.0, 60.5, 34.0, 10.2,
                 shape=r"$[B,\,C_{\max},\,N,\,D]$",
                 note="never allocated / built")
    ax.text(121.0, 58.9,
            "the ragged token list stays the solid primary object;"
            " no padded tensor is built",
            ha="center", va="center", fontsize=5.8, color=MUTED,
            style="italic")

    # ======================================================================= #
    # Panel B -- where p comes from; what is NOT resized
    # ======================================================================= #
    rrect(ax, 2, 23.5, 74, 29.0, edge=ENCODER, face=tint(ENCODER, 0.97),
          lw=1.5, dashed=True, rounding=0.9, z=1)
    panel_header(ax, 4.5, 50.2, "B", "where $p$ comes from; what is NOT resized")

    note_box(ax, 4.5, 42.5, 69.0, 7.0, "where $p$ comes from  (one knob)",
             [r"train default: $\mathtt{patch\_sizes}{=}(\ldots)$ + "
              r"$\mathtt{flexi\_random}{=}\mathtt{True}$  $\to$  one $p$ per step/batch",
              r"override: $\mathtt{forward}(\ldots,\mathtt{patch\_sizes}{=}p)$  "
              r"or a per-sample list",
              r"eval: canonical $P$ unless $\mathtt{patch\_sizes}$ is passed"],
             edge=TOKENIZER, title_size=7.3, body_size=5.6, pitch=1.55)

    note_box(ax, 4.5, 33.5, 69.0, 8.2,
             "layout needs no resized table",
             [r"encoder: 2-D axial RoPE on normalized $(r,c)$; memory tokens get zero rotation",
              r"decoder: factorized fractional absolute PE from $(r,c,g_h,g_w)$",
              r"$\mathtt{ChannelFusion}$: adds no spatial table at all"],
             edge=ENCODER, title_size=7.3, body_size=5.6, pitch=1.55)

    note_box(ax, 4.5, 24.5, 69.0, 8.2,
             r"scale conditioning  (default $\mathtt{'add'}$)",
             [r"$[\log p,\ \log \mu\mathrm{m/px},\ \mathtt{has}_{\mu}]$  $\to$  Fourier features + MLP",
              r"added to encoder tokens and to decoder target channel embeddings",
              r"$\mu$m/px is optional; $\mathtt{has}_{\mu}$ marks presence -- physical size is not required"],
             edge=ENCODER, title_size=7.3, body_size=5.6, pitch=1.55)

    # ======================================================================= #
    # Panel C -- variable-patch-size contract
    # ======================================================================= #
    rrect(ax, 82, 23.5, 76, 29.0, edge=DECODER, face=tint(DECODER, 0.97),
          lw=1.5, dashed=True, rounding=0.9, z=1)
    panel_header(ax, 84.5, 50.2, "C", "variable-patch-size contract")

    # ---- table ------------------------------------------------------------ #
    tx0, cols = 84.0, [26.0, 30.0, 16.0]
    xs = [tx0, tx0 + cols[0], tx0 + cols[0] + cols[1]]
    total_w = sum(cols)
    header_y, header_h = 45.4, 3.2
    for i, (xx, ww, label) in enumerate(zip(xs, cols,
                                            ["component", "at runtime patch size $p_b$",
                                             "resize"])):
        cell(ax, xx, header_y, ww, header_h, label, edge=INK,
             face=tint(INK, 0.88), size=5.8, bold=True, align="center")

    rows = [
        ("patch-embed kernel",
         r"PI-resize $P{\to}p_b$ (pseudoinverse), not bilinear",
         ("PI (input)", TOKENIZER)),
        ("token count / grid",
         r"$N_b=(H/p_b)^2$; DCP masks are $[C_b,N_b]$",
         ("none", MUTED)),
        ("targets / predictions",
         r"$[K_t,p_b^2]$; mixed $p$ padded to $\max_b p_b^2$ + validity mask",
         ("mask only", MUTED)),
        ("reconstruction head",
         r"shared $\mathrm{Linear}(D,c^2)$, $c=\mathtt{head\_patch\_size}$ or $\max p$",
         (r"bilinear $c{\to}p_b$", DECODER)),
        ("fusion / encoder / decoder PE",
         "no grid-sized table to resize",
         ("none", MUTED)),
    ]
    row_h, row_pitch = 3.1, 3.5
    for i, (comp, behav, (tag, tag_c)) in enumerate(rows):
        yy = header_y - 0.45 - (i + 1) * row_pitch
        cell(ax, xs[0], yy, cols[0], row_h, comp, edge=INK, face="white",
             size=5.5, bold=True)
        cell(ax, xs[1], yy, cols[1], row_h, behav, edge=INK, face="white",
             size=5.3)
        cell(ax, xs[2], yy, cols[2], row_h, tag, edge=tag_c,
             face=tint(tag_c, 0.80), size=5.4, bold=True, align="center")

    # PI-resize (input) vs bilinear (output head) distinction -- explicit.
    ax.text(84.0, 25.7,
            r"input kernel = PI-resize $\mathrm{pinv}(B^\top)$ (FlexiViT specifies this).",
            ha="left", va="center", fontsize=5.6, color=LOSS,
            fontweight="bold")
    ax.text(84.0, 24.3,
            r"output head = plain bilinear $c{\to}p_b$ -- a repo addition, not FlexiViT.",
            ha="left", va="center", fontsize=5.6, color=LOSS,
            fontweight="bold")

    # ======================================================================= #
    # Bottom: glossary (left) + code hooks & role legend (right)
    # ======================================================================= #
    rrect(ax, 2, 3.0, 74, 18.0, edge=GHOST, face="#FAFAFA", lw=1.1,
          dashed=True, rounding=0.6, z=2)
    ax.text(4.0, 19.3, "symbols / glossary", ha="left", va="center",
            fontsize=7.2, fontweight="bold", color=INK)
    left = [
        r"$P$  canonical patch size (stored kernel)",
        r"$p_b$  runtime patch size of sample $b$",
        r"$N_b=(H/p_b)(W/p_b)$  patches / channel",
        r"$D$  token / embedding dim",
        r"$V$  visible channel-patches",
        r"$c=\mathtt{head\_patch\_size}$ or $\max(\mathtt{patch\_sizes})$",
        r"$W_P,\,W_p$  stored / resized conv kernel",
    ]
    right = [
        r"$B$  bilinear resize operator on the patch grid",
        r"$\mathrm{PI}(W_P;P{\to}p)=\mathrm{pinv}(B^\top)W_P$  PI-resize",
        r"$z_c$  channel identity (routing bias + skip)",
        r"RoPE  2-D axial rotary position embedding",
        r"fracPE  factorized fractional absolute PE",
        r"$\mathtt{has}_{\mu}$  marks whether $\mu$m/px is supplied",
        r"$\Sigma C\,N$  total channel-patches across the batch",
    ]
    gy = 17.1
    for lft, rgt in zip(left, right):
        ax.text(4.0, gy, lft, ha="left", va="center", fontsize=5.3,
                color=INK)
        ax.text(40.0, gy, rgt, ha="left", va="center", fontsize=5.3,
                color=INK)
        gy -= 1.65
    ax.text(4.0, 5.6,
            r"$H,W$  image side;  $K_t$  masked channel-patches (decoder targets);  "
            r"$C_b$  channel count of sample $b$",
            ha="left", va="center", fontsize=5.3, color=INK)

    rrect(ax, 82, 3.0, 76, 18.0, edge=FUSED, face=tint(FUSED, 0.95),
          lw=1.3, rounding=0.6, z=2)
    ax.text(84.0, 19.3, "code hooks", ha="left", va="center", fontsize=7.2,
            fontweight="bold", color=INK)
    hooks = [
        r"$\mathtt{ChannelFusionMAE}(\mathtt{patch\_sizes}{=}\ldots,\ "
        r"\mathtt{flexi\_random}{=}\mathtt{True},\ \mathtt{head\_patch\_size}{=}\ldots)$",
        r"$\mathtt{forward}(\ldots,\ \mathtt{patch\_sizes}{=}\ldots)$  "
        r"$\to$ int, per-sample list, or $\mathtt{None}$ (canonical $P$)",
        r"$\mathtt{ChannelTiedPatchEmbed._project}(x,p)$:  "
        r"$\mathtt{pi\_resize\_patch\_embed}(\cdot,p)$ then $\mathtt{F.conv2d}$",
        r"$\mathtt{PatchReconstructionHead}(D,c)$:  $\mathrm{Linear}(D,c^2)$ "
        r"then bilinear resize $c{\to}p_b$",
    ]
    hy = 17.0
    for line in hooks:
        ax.text(84.0, hy, line, ha="left", va="center", fontsize=5.5,
                color=INK)
        hy -= 1.95

    # compact role legend inside the code-hook card
    roles = [
        ("tokenizer", TOKENIZER, "fill"),
        ("fusion", FUSION, "fill"),
        ("encoder", ENCODER, "fill"),
        ("decoder", DECODER, "fill"),
        ("fused", FUSED, "cross"),
        ("loss", LOSS, "fill"),
        ("ghost", GHOST, "ghost"),
    ]
    lx = 84.0
    for label, color, style in roles:
        rrect(ax, lx, 6.3, 1.5, 1.5, edge=color,
              face="#FFFFFF" if style == "ghost" else tint(color, 0.8),
              lw=1.0, dashed=(style == "ghost"), rounding=0.2, z=4,
              crossed=(style == "cross"))
        ax.text(lx + 2.1, 7.05, label, ha="left", va="center", fontsize=5.0,
                color=INK)
        lx += 10.4
    ax.text(84.0, 4.4,
            "role colours: module operations are plain; data tensors are crossed; "
            "ghost = never allocated",
            ha="left", va="center", fontsize=5.0, color=MUTED, style="italic")

    return fig


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--formats", nargs="+", default=["pdf", "svg", "png"])
    ap.add_argument("--stem", default="level9_flexivit")
    args = ap.parse_args()

    install_rcparams()
    fig = build()
    save_figure(fig, OUT, args.stem, args.formats)
    plt.close(fig)


if __name__ == "__main__":
    main()
