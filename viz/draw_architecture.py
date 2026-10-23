"""Generate architecture diagrams for the repo.

Produces (into ``viz/out/``):

* ``architecture_torchview.png/.svg`` -- the traced ``ChannelFusionMAE`` graph.
* ``padding_free_flow.png/.svg`` -- a schematic of the set-native data flow,
  which generic graph tracers cannot express.

Usage::

    uv run python viz/draw_architecture.py
"""

from __future__ import annotations

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
        n_memory=4,
        num_channel_ids=16,
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
def draw_schematic() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(15, 8))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.85], hspace=0.25)
    ax = fig.add_subplot(gs[0])
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 4)
    ax.axis("off")

    def box(x, y, w, h, text, color, fontsize=9, edge="#333"):
        ax.add_patch(
            mpatches.FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.06,rounding_size=0.12",
                linewidth=1.2,
                edgecolor=edge,
                facecolor=color,
            )
        )
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize)

    def arrow(x0, y0, x1, y1):
        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops=dict(arrowstyle="-|>", lw=1.3, color="#333"),
        )

    y, h = 2.2, 0.9
    box(0.1, y, 2.1, h, "images\nlist of [C_b,1,H,W]\n(variable C)", "#e3f2fd", 8.5)
    box(2.9, y, 2.2, h, "channel-tied\npatch embed\n(shared conv)", "#e8f5e9", 8.5)
    box(5.8, y, 2.0, h, "tokens\n[ΣC·N, D]\n+ identity", "#fff8e1", 8.5)
    box(8.5, y, 2.4, h, "group by\n(sample, location)\nPMA over channels", "#fce4ec", 8.5)
    box(11.6, y, 3.2, h, "fused [B, N, D]\n+ spatial pos\n(size independent of C)", "#ede7f6", 8.5)
    for a, b in ((2.2, 2.9), (5.1, 5.8), (7.8, 8.5), (10.9, 11.6)):
        arrow(a, y + h / 2, b, y + h / 2)

    # second stage
    y2 = 0.5
    box(8.5, y2, 2.4, h, "memory tokens +\nspatial transformer", "#e0f7fa", 8.5)
    box(11.6, y2, 3.2, h, "channel-aware decoder\n→ masked-channel pixels", "#f3e5f5", 8.5)
    arrow(13.2, y, 9.7, y2 + h)
    arrow(10.9, y2 + h / 2, 11.6, y2 + h / 2)

    ax.text(
        0.1,
        3.7,
        "Padding-free channel-fusion front-end  (no [B, C_max, D] tensor anywhere in this path)",
        fontsize=12,
        fontweight="bold",
    )
    ax.text(
        7.6,
        3.35,
        "masking = token omission",
        fontsize=8.5,
        color="#b71c1c",
        ha="center",
    )
    ax.add_patch(
        mpatches.FancyArrowPatch(
            (7.5, 3.25), (8.7, 3.12), arrowstyle="-|>", color="#b71c1c", lw=1.2
        )
    )

    # ---- token layout: C rows x N columns, columns are attention groups ---- #
    ax2 = fig.add_subplot(gs[1])
    C, N = 5, 4
    ax2.set_xlim(-0.6, 7.2)
    ax2.set_ylim(-0.4, C + 1.7)
    ax2.axis("off")

    for c in range(C):
        for n in range(N):
            ax2.add_patch(
                mpatches.FancyBboxPatch(
                    (n, C - 1 - c),
                    0.86,
                    0.86,
                    boxstyle="round,pad=0.02,rounding_size=0.08",
                    facecolor="#e8eaf6",
                    edgecolor="#9fa8da",
                    lw=1,
                )
            )
            ax2.text(n + 0.43, C - 1 - c + 0.43, f"c{c}n{n}", ha="center", va="center", fontsize=7)

    # highlight column 1 as one attention group
    ax2.add_patch(
        mpatches.Rectangle(
            (0.93, -0.05),
            1.0,
            C + 0.05,
            fill=False,
            edgecolor="#d81b60",
            lw=2.4,
        )
    )
    ax2.text(1.43, C + 0.45, "one (sample, patch) group → attention over channels", ha="center", fontsize=8.5, color="#d81b60")
    ax2.annotate(
        "",
        xy=(1.43, C - 0.05),
        xytext=(1.43, C + 0.32),
        arrowprops=dict(arrowstyle="-|>", color="#d81b60", lw=1.4),
    )
    ax2.text(4.5, C - 1.3, "only REAL channels\nare present — a dropped\nchannel is simply absent\n(no pad row, no mask,\nno [-inf] attention)", fontsize=8.5, color="#1b5e20")
    ax2.text(
        -0.55,
        C + 1.25,
        "flattened token layout  [ΣC·N, D]",
        fontsize=10.5,
        fontweight="bold",
    )

    fig.savefig(OUT / "padding_free_flow.png", dpi=170, bbox_inches="tight")
    fig.savefig(OUT / "padding_free_flow.svg", bbox_inches="tight")
    print(f"wrote {OUT/'padding_free_flow.png'} and .svg")


def main() -> None:
    dot = ensure_graphviz()
    if dot is None:
        print("WARNING: graphviz 'dot' not found; skipping torchview diagram.")
    else:
        draw_torchview()
    draw_schematic()


if __name__ == "__main__":
    main()
