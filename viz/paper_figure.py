"""Paper-grade figure of the ``fuse-channels`` core architecture.

One hand-authored, vector-first figure (PDF/SVG/PNG) with two panels:

* **A** -- the end-to-end padding-free channel-fusion tokenizer.
* **B** -- the set-native token layout and the interchangeable components
  (channel fusion, DCP masking).

Unlike ``draw_architecture.py`` (a traced op graph + a diagnostic schematic),
this file is the *submission* figure: modules, not ops; tensor shapes on every
block; the set-native semantics made visually explicit.

Usage::

    uv run python viz/paper_figure.py                 # pdf + svg + png
    uv run python viz/paper_figure.py --formats pdf png
    uv run python viz/paper_figure.py --stem fig1     # custom output stem
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Design system
# --------------------------------------------------------------------------- #
# Okabe-Ito, colour-blind safe.
BLUE = "#0072B2"
GREEN = "#009E73"
ORANGE = "#E69F00"
SKY = "#56B4E9"
RED = "#D55E00"
PURPLE = "#CC79A7"
VIOLET = "#6A4C93"
INK = "#1A1A1A"
MUTED = "#5C5C5C"


def tint(color: str, amount: float = 0.86):
    """Mix ``color`` toward white by ``amount`` (fills for node bodies)."""
    r, g, b = mcolors.to_rgb(color)
    return (r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount)


def node(
    ax,
    x,
    y,
    w,
    h,
    title,
    sub="",
    edge=BLUE,
    title_size=8.2,
    sub_size=6.7,
    lw=1.5,
):
    """Rounded module box with a title and a smaller shape/role annotation."""
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.0,rounding_size=0.7",
            linewidth=lw,
            edgecolor=edge,
            facecolor=tint(edge),
            zorder=2,
        )
    )
    cx = x + w / 2
    if sub:
        ax.text(
            cx, y + h * 0.64, title, ha="center", va="center",
            fontsize=title_size, fontweight="bold", color=INK, zorder=3,
        )
        ax.text(
            cx, y + h * 0.30, sub, ha="center", va="center",
            fontsize=sub_size, color=MUTED, linespacing=1.35, zorder=3,
        )
    else:
        ax.text(
            cx, y + h / 2, title, ha="center", va="center",
            fontsize=title_size, fontweight="bold", color=INK, zorder=3,
        )
    return (x, y, w, h)


def arrow(ax, p0, p1, color=INK, lw=1.7, style="-|>", rad=0.0, ms=11):
    ax.add_patch(
        FancyArrowPatch(
            p0, p1,
            arrowstyle=style,
            mutation_scale=ms,
            linewidth=lw,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
            shrinkA=1.5, shrinkB=1.5,
            zorder=4,
        )
    )


def panel_header(ax, x, y, letter, text, color=INK):
    ax.text(
        x, y, letter, ha="left", va="center",
        fontsize=15, fontweight="bold", color=color,
    )
    ax.text(
        x + 2.6, y, text, ha="left", va="center",
        fontsize=12.5, fontweight="bold", color=color,
    )


# --------------------------------------------------------------------------- #
# Panel A -- end-to-end pipeline
# --------------------------------------------------------------------------- #
def draw_pipeline(ax) -> None:
    ax.set_xlim(-2, 102)
    ax.set_ylim(0, 24)
    ax.axis("off")
    panel_header(ax, -1, 22.8, "A", "Padding-free channel-fusion tokenizer")

    stages = [
        ("Variable channel set", "$\\{I_c\\in\\mathbb{R}^{1\\times H\\times W}\\}$\n"
         "$C_b\\in\\{1,\\dots,60{+}\\}$; order-free", BLUE),
        ("Channel-tied embed", "shared $1{\\to}D$ conv\n"
         "no spatial PE", GREEN),
        ("Channel identity", "$z_c$ routing bias\n+ output skip\nper channel id", ORANGE),
        ("Padding-free tokens", "$[\\,\\Sigma C\\cdot N,\\; D\\,]$ $+$ idx\n"
         "$(b,c,n)$; ragged, no pad row", SKY),
        ("Channel fusion (PMA)", "single query per $(b,n)$\n"
         "$+$ post-fusion pos $\\to[B,N,D]$", RED),
        ("Transformer encoder", "$+M$ memory tokens\n"
         "grouped self-attn\n$\\to[B{\\cdot}M{+}K_f,D]$", PURPLE),
        ("Channel-aware decoder", "target self-attn\n"
         "cross-attn: mem$+$tok$+$desc\n$\\to$ pixels", VIOLET),
    ]

    n = len(stages)
    x0, x1 = 1.0, 101.0
    w = 12.8
    gap = (x1 - x0 - n * w) / (n - 1)
    y, h = 5.6, 9.8

    centers = []
    for i, (title, sub, color) in enumerate(stages):
        x = x0 + i * (w + gap)
        node(ax, x, y, w, h, title, sub, edge=color)
        centers.append((x, w))

    for i in range(n - 1):
        xa = centers[i][0] + centers[i][1]
        xb = centers[i + 1][0]
        arrow(ax, (xa, y + h / 2), (xb, y + h / 2))

    # headline callout: the tensor that never exists
    cx = centers[3][0] + centers[3][1] / 2
    ax.text(
        cx, 20.4, "no $[B,\\,C_{\\max},\\,N,\\,D]$ tensor anywhere",
        ha="center", va="center", fontsize=9.2, fontweight="bold", color=RED,
    )
    arrow(ax, (cx, 19.7), (cx, y + h + 0.4), color=RED, lw=1.3, style="-|>", ms=9)

    # footnote ribbon: what makes the path a set function
    ax.text(
        1.0, 2.2,
        "Masking, padding and cardinality are handled by omitting rows — not by zeroing or "
        "attention masks: the encoder is a set function of the present tokens.",
        ha="left", va="center", fontsize=8.2, color=MUTED, style="italic",
    )


# --------------------------------------------------------------------------- #
# Panel B -- token layout + interchangeable components
# --------------------------------------------------------------------------- #
def draw_ragged_grid(ax) -> None:
    cell, pitch = 5.4, 6.25

    # column labels
    for n in range(4):
        ax.text(
            4 + n * pitch + cell / 2, 44.6, f"$n_{n}$",
            ha="center", va="center", fontsize=8.2, color=MUTED,
        )

    def draw_sample(x0, n_ch, label, missing=None, group_col=None, gcolor=RED):
        rows = []
        for c in range(n_ch):
            yy = 38.0 - c * pitch
            rows.append(yy)
            for n in range(4):
                xx = x0 + n * pitch
                is_missing = missing == (c, n)
                ax.add_patch(
                    FancyBboxPatch(
                        (xx, yy), cell, cell,
                        boxstyle="round,pad=0.0,rounding_size=0.5",
                        linewidth=1.1,
                        edgecolor=(MUTED if is_missing else BLUE),
                        facecolor=("#FFFFFF" if is_missing else tint(BLUE, 0.8)),
                        linestyle=("--" if is_missing else "solid"),
                        zorder=2,
                    )
                )
                if is_missing:
                    ax.text(
                        xx + cell / 2, yy + cell / 2, "omitted",
                        ha="center", va="center", fontsize=6.3,
                        color=RED, style="italic", zorder=3,
                    )
            ax.text(
                x0 - 1.1, yy + cell / 2, f"$c_{c}$",
                ha="right", va="center", fontsize=7.6, color=MUTED,
            )
        if group_col is not None:
            gx = x0 + group_col * pitch
            ax.add_patch(
                Rectangle(
                    (gx - 0.35, rows[-1] - 0.35), cell + 0.7,
                    (rows[0] - rows[-1]) + cell + 0.7,
                    fill=False, edgecolor=gcolor, linewidth=2.3, zorder=3,
                )
            )
        ax.text(
            x0 + 2 * pitch - 0.15, 15.4, label,
            ha="center", va="center", fontsize=8.4, color=INK, fontweight="bold",
        )

    draw_sample(4.0, 4, "sample A  ($C=4$)", group_col=1)
    draw_sample(34.0, 3, "sample B  ($C=3$)", missing=(1, 2), group_col=2)

    # group annotation
    gx = 4.0 + 1 * pitch + cell / 2
    ax.text(
        gx, 46.8, "attention group $(b,n)$",
        ha="center", va="center", fontsize=8.0, fontweight="bold", color=RED,
    )
    arrow(ax, (gx, 46.2), (gx, 38.0 + cell + 0.35), color=RED, lw=1.2, ms=9)
    ax.text(
        34.0 + 2 * pitch + cell / 2, 46.8, "same rule, other sample",
        ha="center", va="center", fontsize=8.0, color=RED,
    )

    # the padded tensor that never exists
    ax.add_patch(
        FancyBboxPatch(
            (3.0, 2.5), 52.0, 8.2,
            boxstyle="round,pad=0.0,rounding_size=0.6",
            linewidth=1.4, edgecolor=MUTED, facecolor="#F4F4F4",
            linestyle="--", zorder=2,
        )
    )
    ax.plot([3.0, 55.0], [2.9, 10.3], color=RED, lw=2.4, zorder=4, alpha=0.85)
    ax.text(
        29.0, 6.6,
        "the padded tensor $[B,\\,C_{\\max},\\,N,\\,D]$ is never materialised",
        ha="center", va="center", fontsize=8.8, color=MUTED,
        style="italic", zorder=5, fontweight="bold",
    )


def draw_components(ax) -> None:
    # --- fusion alternatives -------------------------------------------------
    ax.add_patch(
        FancyBboxPatch(
            (60.0, 24.0), 41.0, 21.0,
            boxstyle="round,pad=0.0,rounding_size=0.7",
            linewidth=1.5, edgecolor=RED, facecolor=tint(RED, 0.9), zorder=2,
        )
    )
    ax.text(
        80.5, 42.9, "Channel fusion — interchangeable",
        ha="center", va="center", fontsize=10.0, fontweight="bold", color=INK,
    )
    ax.text(
        62.0, 38.9, "PMA attention",
        ha="left", va="center", fontsize=8.6, fontweight="bold", color=RED,
    )
    ax.text(
        62.0, 36.1,
        "$\\mathrm{softmax}\\!\\left(qK^{\\top}/\\sqrt{d}\\right)V$",
        ha="left", va="center", fontsize=9.6, color=INK,
    )
    ax.text(
        62.0, 33.5, "$O(C)$ per location $\\cdot$ data-dependent weights",
        ha="left", va="center", fontsize=7.4, color=MUTED,
    )
    ax.text(
        62.0, 30.2, "IdentityGatedSum (channel-routed)",
        ha="left", va="center", fontsize=8.6, fontweight="bold", color=RED,
    )
    ax.text(
        62.0, 27.4,
        "$\\sum_c \\mathrm{softmax}_c(\\mathrm{MLP}(z_c))\\,V_c$",
        ha="left", va="center", fontsize=9.2, color=INK,
    )
    ax.text(
        62.0, 25.0, "$O(C)$; MLP once per channel, broadcast to patches",
        ha="left", va="center", fontsize=7.0, color=MUTED,
    )

    # --- DCP masking ---------------------------------------------------------
    ax.add_patch(
        FancyBboxPatch(
            (60.0, 2.0), 41.0, 19.0,
            boxstyle="round,pad=0.0,rounding_size=0.7",
            linewidth=1.5, edgecolor=GREEN, facecolor=tint(GREEN, 0.9), zorder=2,
        )
    )
    ax.text(
        80.5, 19.1, "DCP masking = token omission",
        ha="center", va="center", fontsize=10.0, fontweight="bold", color=INK,
    )
    bullets = [
        "Eq. 1: patch, channel, or combined region",
        "patch: fixed $\\lfloor N \\cdot r_p\\rfloor$ per channel",
        "channel: unified HCS $m{\\sim}U\\{1..C\\}$, then subset",
        "masked channel-patch = dropped row; never zeroed/padded",
    ]
    for i, b in enumerate(bullets):
        yy = 16.2 - i * 3.3
        ax.text(61.6, yy, "$\\bullet$", ha="left", va="center",
                fontsize=8.0, color=GREEN)
        ax.text(62.8, yy, b, ha="left", va="center", fontsize=7.6, color=INK)


def draw_token_panel(ax) -> None:
    ax.set_xlim(-2, 102)
    ax.set_ylim(0, 52)
    ax.axis("off")
    panel_header(ax, -1, 50.6, "B", "Set-native token layout and components")
    draw_ragged_grid(ax)
    draw_components(ax)


# --------------------------------------------------------------------------- #
def build() -> plt.Figure:
    figsize = (13.0, 9.5)
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(
        2, 1, height_ratios=[24, 52], hspace=0.02,
        left=0.01, right=0.99, top=0.985, bottom=0.01,
    )
    ax_a = fig.add_subplot(gs[0])
    ax_a.set_aspect("equal", adjustable="box")
    draw_pipeline(ax_a)

    ax_b = fig.add_subplot(gs[1])
    ax_b.set_aspect("equal", adjustable="box")
    draw_token_panel(ax_b)
    return fig


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--formats", nargs="+", default=["pdf", "svg", "png"])
    ap.add_argument("--stem", default="architecture")
    args = ap.parse_args()

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "mathtext.fontset": "dejavusans",
    })

    fig = build()
    stem = args.stem
    for fmt in args.formats:
        path = OUT / f"{stem}.{fmt}"
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"wrote {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
