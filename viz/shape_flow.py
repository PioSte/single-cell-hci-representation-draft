"""kinemo animation of the tensor-shape flow through the channel-fusion MAE.

It walks the shapes of :class:`fuse_channels.mae.ChannelFusionMAE` on a toy
input and shows the two things that matter for a padding-free channel-set
model:

* the channel axis ``C`` (which varies per sample) is *summed away* by the
  channel fusion -- it is never padded to a ``C_max``;
* every shape downstream of the fusion is independent of ``C``.

Two simplifications: DCP masking is shown as location-level removal although the
real masker selects per channel-patch token before the fusion, and ``[B, D]`` is
the inference-time :meth:`~fuse_channels.mae.ChannelFusionMAE.encode` descriptor,
not a training-forward intermediate.

The grid of cells is a literal picture of the tensor axes: rows are channels
(grouped by sample), columns are spatial patches, and the third axis ``D`` is
the feature width.  The channel axis is the thing you watch collapse.

Render::

    .venv/bin/kinemo check --strict viz/shape_flow.py
    .venv/bin/kinemo render viz/shape_flow.py --format mp4
    .venv/bin/kinemo render viz/shape_flow.py --format gif
"""

from __future__ import annotations

import kinemo as k

# --------------------------------------------------------------------------- #
# Toy input, chosen to mirror the module defaults in mae.py.
#   B samples, channels C_b in {3, 2}, H = W = 32, patch P = 8 -> N = 16 patches,
#   feature width D = 192, Q = 1 learned query, M = 4 memory tokens.
# --------------------------------------------------------------------------- #
B = 2
C = (3, 2)
N = 16
M = 4

COLS = 4  # we draw 4 of the N = 16 patch columns
CX = [-0.75, 0.05, 0.85, 1.65]
ROW_Y = [1.55, 0.95, 0.35, -0.45, -1.05]
FUSED_Y = [0.75, -0.05]
CS = 0.56

CH = [k.BLUE, k.ORANGE, k.PURPLE, k.GREEN, k.TEAL]
FUSED = k.YELLOW
MEM = k.TEAL
DESC = k.PINK


def cell(x, y, color, size=CS, opacity=0.0, fill=0.6):
    """One rounded cell = one tensor element (a row/column picture)."""
    return k.RoundedRect(
        w=size,
        h=size,
        radius=0.12,
        x=x,
        y=y,
        fill=color,
        fill_opacity=fill,
        stroke=color,
        stroke_width=1.8,
        opacity=opacity,
    )


@k.scene(size=(1920, 1080), fps=60)
def shape_flow(s: k.Scene):
    # -- persistent chrome ------------------------------------------------- #
    title = k.Text("From an image set to a fixed output", size=0.62).place(
        at="top", margin=0.5
    )
    kicker = k.Text(
        "tensor shapes through the padding-free channel-fusion MAE",
        size=0.3,
        color=k.GRAY,
    ).place(below=title, gap=0.15)

    step = k.Text("1 / 10", size=0.3, mono=True, color=k.GRAY).place(
        at="top-right", margin=0.55
    )
    desc = k.Text("", size=0.34, color=k.GRAY, x=0.0, y=-2.5, width=13.0, align="center")
    shape = k.Text("", size=0.34, mono=True, x=0.0, y=-3.4)

    def cap(description, formula, color=k.WHITE, dur=0.55):
        desc.set(text=description)
        shape.set(text=formula, color=color)
        s.wait(dur)

    def nxt(label):
        s.play(step.to(text=label), duration=0.25)

    s.play(k.write(title), k.fade_in(kicker))
    s.play(k.fade_in(desc, shape, step))

    # ===================================================================== #
    # 1 -- the input set: one single-channel image per marker.
    # ===================================================================== #
    lab_in = []
    img = {}
    tiles_in = []
    for b in range(B):
        x0 = -2.55 if b == 0 else -1.15
        base = sum(C[:b])
        for c in range(C[b]):
            gi = base + c
            y = ROW_Y[0 + c]  # side by side at first
            img[(b, c)] = cell(x0, y, CH[gi])
            tiles_in.append(img[(b, c)])
        lab_in.append(
            k.Text(
                f"sample {b + 1} · C={C[b]}",
                size=0.26,
                color=k.GRAY,
                x=-2.9 if b == 0 else -0.85,
                y=1.95,
            )
        )
    s.add(*tiles_in)
    s.play(
        *[t.to(opacity=1.0) for t in tiles_in],
        k.fade_in(*lab_in, shift=(0, 0.3)),
        duration=0.8,
    )
    cap("one image per marker — the channel count varies per sample", "[C_b, 1, H, W]")
    s.wait(0.5)

    # ===================================================================== #
    # 2 -- concatenate the real channels: [ΣC, 1, H, W], no pad slot.
    # ===================================================================== #
    nxt("2 / 10")
    moves = []
    for b in range(B):
        base = sum(C[:b])
        for c in range(C[b]):
            gi = base + c
            moves.append(img[(b, c)].to(x=-2.55, y=ROW_Y[gi]))
    s.play(*moves, duration=0.7)
    s.play(
        lab_in[0].to(text="ΣC real channels", x=-3.35, y=0.25),
        k.fade_out(lab_in[1]),
        duration=0.4,
    )
    cap("stack every real channel — nothing is padded", "[ΣC, 1, H, W] = [5, 1, 32, 32]", k.YELLOW)
    s.wait(0.5)

    # ===================================================================== #
    # 3 -- channel-tied patch embedding + additive identity -> token set.
    # ===================================================================== #
    nxt("3 / 10")
    patch = {}
    for b in range(B):
        base = sum(C[:b])
        for c in range(C[b]):
            gi = base + c
            for j in range(COLS):
                patch[(b, c, j)] = cell(CX[j], ROW_Y[gi], CH[gi])
    s.add(*patch.values())

    # move each image tile to the first patch column, then split into tokens
    moves = []
    for key, tile in img.items():
        b, c = key
        gi = sum(C[:b]) + c
        moves.append(tile.to(x=CX[0], y=ROW_Y[gi]))
    s.play(*moves, duration=0.6)
    s.play(
        *[t.to(opacity=0.0) for t in tiles_in],
        *[p.to(opacity=1.0) for p in patch.values()],
        duration=0.5,
    )
    cap(
        "one tied 1→D filter + a shared spatial position + identity z_c",
        "[ΣC, N, D] = [5, 16, 192]",
    )
    s.play(*[k.indicate(p) for p in patch.values()], duration=0.6)
    legend = k.Text(
        f"rows = channels (grouped by sample) · cols = patches ({COLS} of N={N}) · depth = D",
        size=0.24,
        color=k.GRAY,
        x=0.45,
        y=2.45,
    )
    s.play(k.fade_in(legend), duration=0.4)
    s.wait(0.4)

    # ===================================================================== #
    # 4 -- channel fusion: collapse the channel axis at every location.
    # ===================================================================== #
    nxt("4 / 10")
    s.play(k.fade_out(lab_in[0]), duration=0.3)
    fused = {}
    for b in range(B):
        for j in range(COLS):
            fused[(b, j)] = cell(CX[j], FUSED_Y[b], FUSED)
    s.add(*fused.values())

    merge = []
    for b in range(B):
        base = sum(C[:b])
        for c in range(C[b]):
            gi = base + c
            for j in range(COLS):
                merge.append(patch[(b, c, j)].to(x=CX[j], y=FUSED_Y[b], opacity=0.0))
    s.play(*merge, duration=0.9, ease=k.ease.in_out)
    s.play(*[f.to(opacity=1.0) for f in fused.values()], duration=0.4)
    cap(
        "a learned query pools the channels at each location — C is summed away",
        "[B, N, Q·D] → [B, N, D] = [2, 16, 192]",
        k.YELLOW,
    )
    s.play(*[k.indicate(f) for f in fused.values()], duration=0.7)
    s.wait(0.5)

    # ===================================================================== #
    # 5 -- DCP masking = token omission (never zeros, never a pad slot).
    # ===================================================================== #
    nxt("5 / 10")
    omitted = [(0, 1), (1, 3)]
    s.play(k.fade_out(fused[omitted[0]], fused[omitted[1]]), duration=0.5)
    cap(
        "DCP masking omits tokens, so the encoder sees Kf ≤ B·N locations",
        "[B, N, D]  →  [Kf, D]   (Kf = visible locations)",
        k.RED,
    )
    s.wait(0.5)

    # ===================================================================== #
    # 6 -- memory-augmented spatial transformer; fixed memory count.
    # ===================================================================== #
    nxt("6 / 10")
    mem = [cell(CX[m], 2.05, MEM, size=0.4) for m in range(M)]
    s.add(*mem)
    mem_label = k.Text("× M memory tokens", size=0.26, color=MEM, x=3.25, y=2.05)
    arrow_enc = k.Arrow(start=(2.0, 0.35), end=(3.3, 0.35), stroke=k.GRAY, fill=k.GRAY)
    s.play(
        *[m.to(opacity=1.0) for m in mem],
        k.fade_in(mem_label, shift=(0, 0.2)),
        duration=0.5,
    )
    s.play(k.draw(arrow_enc))
    cap(
        "memory + visible tokens go through the transformer (per-sample groups)",
        "[B·M + Kf, D] = [8 + Kf, 192]",
        k.TEAL,
    )
    s.wait(0.5)

    # ===================================================================== #
    # 7 -- fixed-size descriptor for the whole sample.
    # ===================================================================== #
    nxt("7 / 10")
    bars = []
    for b in range(B):
        bars.append(
            k.RoundedRect(
                w=0.44,
                h=1.15,
                radius=0.14,
                x=4.35,
                y=FUSED_Y[b],
                fill=DESC,
                fill_opacity=0.6,
                stroke=DESC,
                stroke_width=2.0,
                opacity=0.0,
            )
        )
    s.add(*bars)
    desc_label = k.Text("[B, D]", size=0.32, mono=True, color=DESC, x=4.35, y=-1.15)
    s.play(
        *[b.to(opacity=1.0) for b in bars],
        k.fade_in(desc_label, shift=(0.3, 0)),
        duration=0.6,
    )
    cap(
        "pooled representation: fixed [B, D], independent of C",
        "[B, D] = [2, 192]",
        k.PINK,
    )
    s.play(*[k.indicate(b) for b in bars], duration=0.6)
    s.wait(0.5)

    # ===================================================================== #
    # 8 -- channel-aware shared decoder -> raw pixels for masked targets.
    # ===================================================================== #
    nxt("8 / 10")
    # clear the front-end stage to make room
    alive_fused = [f for key, f in fused.items() if key not in omitted]
    s.play(
        k.fade_out(
            *patch.values(), *alive_fused, *mem, mem_label, arrow_enc, *bars, desc_label
        ),
        duration=0.5,
    )
    q1 = k.Text("mask token\n+ position\n+ target z_c", size=0.26, x=-2.4, y=0.0)
    qcell = cell(-1.0, 0.0, k.ORANGE, size=0.9, opacity=0.0)
    arrow_dec = k.Arrow(start=(-0.45, 0.0), end=(0.75, 0.0), stroke=k.GRAY, fill=k.GRAY)
    px = k.Text("[Kt, P²]", size=0.4, mono=True, x=2.1, y=0.0)
    tgt = k.RoundedRect(
        w=1.5, h=1.5, radius=0.18, x=4.2, y=0.0,
        fill=k.GREEN, fill_opacity=0.35, stroke=k.GREEN, stroke_width=2.0, opacity=0.0,
    )
    tgt_label = k.Text("target patch\n[Kt, P²]", size=0.26, color=k.GREEN, x=4.2, y=-1.3)
    s.add(qcell, tgt)
    s.play(
        *[qcell.to(opacity=1.0), tgt.to(opacity=1.0)],
        k.fade_in(q1, px, tgt_label),
        duration=0.5,
    )
    s.play(k.draw(arrow_dec))
    cap(
        "one shared decoder predicts masked channel-patches, conditioned on identity",
        "[Kt, D]  →  pixels [Kt, P²] = [Kt, 64]",
        k.GREEN,
    )
    s.play(k.indicate(px, scale=1.15), k.indicate(tgt, scale=1.1), duration=0.7)
    s.wait(0.4)

    # ===================================================================== #
    # 9 -- the objective.
    # ===================================================================== #
    nxt("9 / 10")
    loss = k.Text("loss = masked MSE, normalised per patch", size=0.36, color=k.YELLOW, x=-1.6, y=-1.55)
    s.play(k.write(loss))
    cap(
        "the only place C appears is the input length — summed, not padded",
        "C ∈ {1 … 60+}  ·  no C_max  ·  no mask  ·  no pad token",
        k.YELLOW,
    )
    s.wait(0.5)

    # ===================================================================== #
    # 10 -- summary: the shape chain.
    # ===================================================================== #
    nxt("10 / 10")
    s.play(
        k.fade_out(q1, qcell, px, tgt, tgt_label, arrow_dec, loss, *img.values()),
        duration=0.4,
    )
    chain = [
        ("[ΣC, 1, H, W]", k.BLUE),
        ("[ΣC, N, D]", k.BLUE),
        ("[B, N, D]", k.YELLOW),
        ("[B, D]", k.PINK),
        ("[Kt, P²]", k.GREEN),
    ]
    pills = []
    arrows = []
    for i, (text, color) in enumerate(chain):
        x = -5.6 + i * 2.8
        pills.append(
            k.RoundedRect(
                w=2.2, h=0.9, radius=0.16, x=x, y=0.2,
                fill=color, fill_opacity=0.22, stroke=color, stroke_width=2.0,
            )
        )
        pills.append(k.Text(text, size=0.24, mono=True, color=k.WHITE, x=x, y=0.2))
        if i < len(chain) - 1:
            arrows.append(
                k.Arrow(start=(x + 1.15, 0.2), end=(x + 1.65, 0.2), stroke=k.GRAY, fill=k.GRAY)
            )
    s.play(k.fade_in(*pills, shift=(0, 0.2)), duration=0.6)
    s.play(k.stagger([k.draw(a) for a in arrows], lag=0.1), duration=0.5)
    summary = k.Text(
        "variable length → fixed everything else", size=0.34, color=k.GREEN, x=0.0, y=-1.1
    )
    s.play(k.write(summary))
    cap("", "", k.WHITE, dur=0.3)

    s.wait(1.2)
