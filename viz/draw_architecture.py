"""Generate architecture diagrams for the repo.

Produces (into ``viz/out/``):

* ``architecture_torchview.png/.svg`` -- the traced ``ChannelFusionMAE`` graph
  (best-effort; with some torchview/torch pairs tracing raises and is skipped).
* ``padding_free_flow.png/.svg`` -- a hand-drawn schematic of the set-native
  data flow, which generic graph tracers cannot express.

The schematic follows the shipped **default** path only: token list -> fusion ->
encoder -> spatial pooling to fixed size latent (``latents [B, L, D]``) ->
decoder KV ``latents`` -> channel-aware decoder.
The padded tensor ``[B, C_max, N, D]`` appears only as a dashed ghost, and DCP
masking is drawn as row omission in the patch embed (before fusion).

Usage::

    uv run python viz/draw_architecture.py
    uv run python viz/draw_architecture.py --formats png
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import torch

OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Graphviz discovery (graphviz may live in a Nix store rather than on PATH)
# --------------------------------------------------------------------------- #
def ensure_graphviz() -> str | None:
    dot = shutil.which("dot")
    if dot:
        return dot
    import glob

    for pat in (
        "/nix/store/*graphviz*/bin/dot",
        "/usr/bin/dot",
        "/usr/local/bin/dot",
    ):
        hits = sorted(glob.glob(pat))
        if hits:
            os.environ["PATH"] = os.path.dirname(hits[-1]) + os.pathsep + os.environ["PATH"]
            return hits[-1]
    return None


# --------------------------------------------------------------------------- #
def draw_torchview() -> None:
    from fuse_channels import ChannelFusionMAE
    from torchview import draw_graph

    torch.manual_seed(0)
    model = ChannelFusionMAE(
        image_size=16,
        patch_size=8,
        dim=64,
        encoder_depth=2,
        decoder_depth=2,
        n_heads=4,
        num_channel_ids=16,
        pixel_size=1.0,
    )
    model.eval()
    images = [torch.randn(2, 1, 16, 16), torch.randn(4, 1, 16, 16)]
    ids = [torch.randint(0, 16, (2,)), torch.randint(0, 16, (4,))]
    visible = [torch.ones(2, 4, dtype=torch.bool), torch.ones(4, 4, dtype=torch.bool)]

    g = draw_graph(
        model,
        input_data=[images, ids, visible],
        graph_name="ChannelFusionMAE",
        expand_nested=False,
        depth=2,
        device="cpu",
        save_graph=False,
    )
    for fmt in ("png", "svg"):
        g.visual_graph.render(str(OUT / "architecture_torchview"), format=fmt, cleanup=True)
    print(f"wrote {OUT/'architecture_torchview.png'} and .svg")


# --------------------------------------------------------------------------- #
# Hand-drawn schematic
# --------------------------------------------------------------------------- #
def draw_schematic(formats: tuple[str, ...] = ("png", "svg"),
                   stem: str = "padding_free_flow") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    # Sibling module: the shared design system (run from ``viz/``).
    from figure_style import (  # noqa: E402
        INK, MUTED, CHANNEL_COLORS,
        TOKENIZER, FUSION, ENCODER, DECODER, LOSS, FUSED, POOL, GHOST,
        tint, rrect, box, arrow, legend, install_rcparams, save_figure,
        padded_ghost,
    )

    install_rcparams()

    fig = plt.figure(figsize=(16.5, 7.5))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_xlim(-2, 196)
    ax.set_ylim(6, 97)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    # ---------------------------------------------------------------- helpers
    def data_cell(x, y, w, h, edge, title, sub="", title_size=8.2,
                  sub_size=6.0, crossed=True):
        """Crossed data cell (token list / embedding), optional sub-line."""
        rrect(ax, x, y, w, h, edge=edge, face=tint(edge, 0.80), lw=1.5,
              rounding=0.6, crossed=crossed, z=3)
        bbox = dict(boxstyle="round,pad=0.14", facecolor="white",
                    edgecolor="none", alpha=0.85)
        cx = x + w / 2
        if sub:
            ax.text(cx, y + h * 0.64, title, ha="center", va="center",
                    fontsize=title_size, color=edge, fontweight="bold",
                    zorder=5, bbox=bbox)
            ax.text(cx, y + h * 0.26, sub, ha="center", va="center",
                    fontsize=sub_size, color=MUTED, zorder=5, bbox=bbox,
                    linespacing=1.35)
        else:
            ax.text(cx, y + h / 2, title, ha="center", va="center",
                    fontsize=title_size, color=edge, fontweight="bold",
                    zorder=5, bbox=bbox)

    def route(pts, color=INK, lw=1.5, ms=9):
        """Orthogonal poly-line with an arrowhead on the final segment."""
        for a, b in zip(pts[:-2], pts[1:-1]):
            ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=lw,
                    solid_capstyle="round", zorder=4)
        arrow(ax, pts[-2], pts[-1], color=color, lw=lw, ms=ms)

    def note(x, y, text, color=MUTED, size=6.4, ha="left", style="italic"):
        ax.text(x, y, text, ha=ha, va="center", fontsize=size, color=color,
                style=style, zorder=7,
                bbox=dict(boxstyle="round,pad=0.14", facecolor="white",
                          edgecolor="none", alpha=0.9))

    # ================================================================ title
    ax.text(2, 94.0, "Padding-free channel-fusion MAE (default path)",
            ha="left", va="center", fontsize=15.5, fontweight="bold",
            color=INK)
    ax.text(2, 90.5,
            "Shipped default path. No $[B, C_{\\max}, N, D]$ tensor "
            "anywhere in this path.",
            ha="left", va="center", fontsize=7.6, color=MUTED,
            style="italic")

    # ====================================================== row 1: tokenizer
    Y1, H = 76.0, 14.0
    Y01 = Y1 - H / 2  # 71

    # images (data stack, variable channel count)
    ix, iw, ih, ip = 2.0, 13.0, 4.2, 5.0
    total = 2 * ip + ih
    iy0 = Y1 - total / 2
    for i in range(3):
        c = CHANNEL_COLORS[i]
        rrect(ax, ix, iy0 + i * ip, iw, ih, edge=c, face=tint(c, 0.72),
              lw=1.4, rounding=0.4, crossed=True, z=3)
    ax.text(ix + iw / 2, iy0 - 1.9, "images", ha="center", va="center",
            fontsize=8.4, fontweight="bold", color=INK)
    ax.text(ix + iw / 2, iy0 - 4.3, "$[C_b,1,H,W]$, $C_b$ order-free",
            ha="center", va="center", fontsize=6.0, color=MUTED)

    # channel-tied patch embed  (DCP visible mask / row omission lives here)
    box(ax, 20, Y01, 23, H, "channel-tied patch embed",
        "shared Conv $1{\\to}D$, kernel $P$\nDCP visible mask: keep the\nvisible rows, omit the rest",
        edge=TOKENIZER, title_size=7.6, sub_size=5.7)
    # ragged visible token list -- the solid primary object
    data_cell(48, Y01, 16, H, TOKENIZER, "tokens",
              sub="$[\\Sigma C\\cdot N, D]$\nvisible $[V, D]$\ncarry $z_c$",
              title_size=8.2, sub_size=6.0)
    # channel fusion
    box(ax, 69, Y01, 21, H, "channel fusion",
        "single-query PMA\nper $(b,n)$ location\nidentity bias $+$ skip\nno post-fusion PE",
        edge=FUSION, title_size=7.6, sub_size=5.5)
    # fused output
    data_cell(95, Y01, 16, H, FUSED, "fused",
              sub="$[K_f, D]$\none row per\npresent $(b,n)$",
              title_size=8.2, sub_size=6.0)

    arrow(ax, (ix + iw, Y1), (20, Y1), lw=1.6, ms=10)
    arrow(ax, (43, Y1), (48, Y1), lw=1.6, ms=10)
    arrow(ax, (64, Y1), (69, Y1), lw=1.6, ms=10)
    arrow(ax, (90, Y1), (95, Y1), lw=1.6, ms=10)

    # DCP callout points at the patch embed / visible mask
    ax.text(31.5, 87.2, "DCP masking $=$ token omission", ha="center",
            va="center", fontsize=8.4, color=LOSS, fontweight="bold")
    ax.text(31.5, 85.0, "(visible mask in the patch embed, before fusion)",
            ha="center", va="center", fontsize=6.3, color=LOSS, style="italic")
    arrow(ax, (31.5, 84.0), (31.5, Y01 + H), color=LOSS, lw=1.4, ms=9)

    # ====================================== row 2: encoder, pooling, decoder
    Y2, H2 = 52.0, 14.0
    Y02 = Y2 - H2 / 2  # 47

    box(ax, 94, Y02, 23, H2, "TransformerEncoder",
        "grouped self-attention\n(no memory tokens)\n2-D axial RoPE (metric)\ndepth 4",
        edge=ENCODER, title_size=7.4, sub_size=5.5)
    box(ax, 121, Y02, 24, H2, "spatial pooling",
        "attention pooling on $tok_{out}$:\n$L$ learned queries $\\to$ latents $[B,L,D]$\n(trained by the loss)",
        edge=POOL, title_size=7.2, sub_size=5.4)
    box(ax, 149, Y02, 20, H2, "decoder KV",
        "$latents\\;[B\\cdot L,D]$\n(from pooling; no memory tokens)",
        edge=POOL, title_size=7.6, sub_size=5.6)
    box(ax, 173, Y02, 20, H2, "channel-aware\ndecoder",
        "target self-attention\nover $K_t$ queries\ncross-attention to the KV",
        edge=DECODER, title_size=7.4, sub_size=5.5)

    arrow(ax, (117, Y2), (121, Y2), lw=1.6, ms=10)
    arrow(ax, (145, Y2), (149, Y2), lw=1.6, ms=10)
    arrow(ax, (169, Y2), (173, Y2), lw=1.6, ms=10)

    # wrap from row 1 down-right into the encoder (all stages read L->R)
    route([(103, Y01), (103, Y02 + H2 + 3.0), (105.5, Y02 + H2 + 3.0),
           (105.5, Y02 + H2)], color=FUSED, ms=10)
    note(108.0, 64.5,
         "$tok_{out}\\,[K_f, D]$: content tokens only, NOT in the KV",
         color=FUSED, size=6.2)

    # ===================================================== token layout grid
    gx, gy, cw, pitch = 6.0, 23.0, 5.0, 6.0
    nrows, ncols = 5, 4
    omitted_row = 2
    for r in range(nrows):
        for c in range(ncols):
            col = CHANNEL_COLORS[r % len(CHANNEL_COLORS)]
            omitted = r == omitted_row
            rrect(ax, gx + c * pitch, gy + (nrows - 1 - r) * pitch, cw, cw,
                  edge=(LOSS if omitted else col),
                  face=("#FFFFFF" if omitted else tint(col, 0.78)),
                  lw=1.3, rounding=0.3, z=3,
                  dashed=omitted, crossed=not omitted)
    top = gy + (nrows - 1) * pitch + cw

    # highlight one (sample, patch) attention group = a whole column
    ax.add_patch(Rectangle((gx + 1 * pitch - 0.4, gy - 0.4), cw + 0.8,
                           (nrows - 1) * pitch + cw + 0.8, fill=False,
                           edgecolor=LOSS, lw=2.2, zorder=5))
    ax.annotate("", xy=(gx + 1 * pitch + cw / 2, top + 0.2),
                xytext=(gx + 1 * pitch + cw / 2, top + 3.2),
                arrowprops=dict(arrowstyle="-|>", color=LOSS, lw=1.6),
                zorder=6)
    ax.text(gx + 1 * pitch + cw / 2, top + 4.6,
            "one $(b, n)$ attention group $\\to$\nattention over channels",
            ha="center", va="center", fontsize=6.8, color=LOSS)

    ax.text(gx, top + 9.5, "flattened visible token list  $[V, D]$",
            ha="left", va="center", fontsize=10.2, fontweight="bold", color=INK)
    ax.text(gx, top + 6.8,
            "solid crossed = real row  $\\cdot$  dashed = omitted row  "
            "$\\cdot$  $[\\Sigma C\\cdot N, D] \\to [V, D]$",
            ha="left", va="center", fontsize=6.2, color=MUTED, style="italic")

    note(36.0, gy + (nrows - 1) * pitch + 2.0,
         "a dropped channel row is simply absent:\n"
         "no pad row, no zero fill, no $[-\\inf]$ attention mask.",
         color=LOSS, size=6.4)
    note(36.0, gy + 3.0,
         "$V \\leq \\Sigma_b\\, C_b\\cdot N_b$ real rows;  empty zones\n"
         "are ragged, not padded.",
         color=MUTED, size=6.2)

    # ===================================================== legend + glossary
    legend(
        ax, 66.0, 67.0,
        [("tokenizer (embed / tokens)", TOKENIZER, "fill"),
         ("channel fusion", FUSION, "fill"),
         ("encoder", ENCODER, "fill"),
         ("channel-aware decoder", DECODER, "fill"),
         ("fused tokens $[K_f, D]$", FUSED, "cross"),
         ("latent pooling", POOL, "cross"),
         ("omitted row / loss", LOSS, "omit"),
         ("ghost padded tensor", GHOST, "ghost")],
        title="module roles & data", pitch=2.6, sw=1.3, fontsize=6.2,
        title_size=7.0, label_gap=1.0,
    )

    # glossary strip
    rrect(ax, 98.0, 9.0, 94.0, 31.0, edge=MUTED, face="#FAFAFA", lw=1.1,
          dashed=True, rounding=0.6, z=1)
    ax.text(100.0, 36.5, "symbols / glossary", ha="left", va="center",
            fontsize=7.4, fontweight="bold", color=INK, zorder=5)
    col_a = [
        ("$B$", "batch size (samples)"),
        ("$C_b$", "channels of sample $b$ (order-free)"),
        ("$H, W$", "image side (square)"),
        ("$\\Sigma C$", "channels across the batch"),
        ("$N$", "patches per channel (max$_b\\,N_b$)"),
        ("$P$", "canonical patch size (kernel)"),
        ("$D$", "token / embedding dim"),
        ("$V$", "visible channel-patch tokens"),
    ]
    col_b = [
        ("$K_f$", "present $(b,n)$ locations"),
        ("$K_t$", "masked ch-patches (targets)"),
        ("$M$", "memory tokens (memory arm only)"),
        ("$L$", "latents from pooling (default 16)"),
        ("$z_c$", "channel identity (bias $+$ skip)"),
        ("DCP", "dynamic channel-patch masking"),
        ("RoPE", "rotary position embedding"),
    ]
    y0, pitch_g, off = 32.5, 2.9, 14.0
    for entries, x in ((col_a, 100.0), (col_b, 146.0)):
        for i, (term, desc) in enumerate(entries):
            yy = y0 - i * pitch_g
            ax.text(x, yy, term, ha="left", va="center", fontsize=6.1,
                    fontweight="bold", color=INK, zorder=5)
            ax.text(x + off, yy, desc, ha="left", va="center", fontsize=6.1,
                    color=MUTED, zorder=5)

    # ghost padded tensor (never materialised)
    padded_ghost(ax, 6.0, 9.0, 44.0, 9.0,
                 shape=r"$[B,\,C_{\max},\,N,\,D]$")

    save_figure(fig, OUT, stem, formats)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--formats", nargs="+", default=["png", "svg"])
    ap.add_argument("--stem", default="padding_free_flow")
    args = ap.parse_args()

    dot = ensure_graphviz()
    if dot is None:
        print("WARNING: graphviz 'dot' not found; skipping torchview diagram.")
    else:
        try:
            draw_torchview()
        except Exception as exc:  # torchview/torch tracing is best-effort
            print(
                "WARNING: torchview graph tracing failed "
                f"({type(exc).__name__}: {exc}); skipping traced diagram."
            )
    draw_schematic(tuple(args.formats), args.stem)


if __name__ == "__main__":
    main()
