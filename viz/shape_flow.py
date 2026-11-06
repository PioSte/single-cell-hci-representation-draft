"""kinemo animation of the tensor-shape flow through the channel-fusion MAE.

It walks the shapes of :class:`fuse_channels.mae.ChannelFusionMAE` on a toy
input and shows the two things that matter for a padding-free channel-set
model:

* the channel axis ``C`` (which varies per sample) is *summed away* by the
  channel fusion -- it is never padded to a ``C_max``;
* every shape downstream of the fusion is independent of ``C``.

One simplification: DCP masking is shown as location-level removal although the
real masker selects per channel-patch token before the fusion.  Spatial pooling
to fixed size latent gives ``[B, L, D]`` (``L = n_latents``, independent of both
``C`` and ``N``); in the shipped default it *is* the decoder KV, and in the
``memory`` arm it is the KV's second component, so the KV never grows with
``N``.  The reconstruction target is the flat pixel layout ``[Σ p²]``: row
``i`` owns ``p_i²`` contiguous pixels, never a padded ``[Kt, max p²]`` block.
Fusion adds no post-fusion spatial PE (the legacy learned/sincos table is an
ablation).

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
#   feature width D = 192, one learned fusion query, no memory tokens,
#   L = 16 latents.
# --------------------------------------------------------------------------- #
B = 2
C = (3, 2)
N = 16

COLS = 4  # we draw 4 of the N = 16 patch columns
CX = [-0.75, 0.05, 0.85, 1.65]
ROW_Y = [1.55, 0.95, 0.35, -0.45, -1.05]
FUSED_Y = [0.75, -0.05]
CS = 0.56

CH = [k.BLUE, k.ORANGE, k.PURPLE, k.GREEN, k.TEAL]
FUSED = k.YELLOW
LAT = "#56B4E9"  # sky: fixed latents from spatial pooling (distinct from channels)


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
    # Persistent symbol key: every axis name used in the animation, on screen.
    symbol_key = k.Text(
        "SYMBOLS\n"
        "ΣC total channels\n"
        "C_b channels/sample\n"
        "H,W image side (px)\n"
        "N  patches/channel\n"
        "N_b patches/sample\n"
        "D  token width\n"
        "B  batch size\n"
        "M  memory (arm only)\n"
        "L  latents/sample\n"
        "V  visible tokens\n"
        "Kf present locs\n"
        "Kt masked targets\n"
        "p  patch side (px)\n"
        "p_b patch/sample\n"
        "P  canonical side\n"
        "colour = channel id",
        size=0.16,
        color=k.GRAY,
        x=-6.55,
        y=1.5,
    )

    def cap(description, formula, color=k.WHITE, dur=0.55):
        desc.set(text=description)
        shape.set(text=formula, color=color)
        s.wait(dur)

    def nxt(label):
        s.play(step.to(text=label), duration=0.25)

    s.play(k.write(title), k.fade_in(kicker))
    s.play(k.fade_in(desc, shape, step, symbol_key))

    # ===================================================================== #
    # 1 -- the input set: one single-channel image per channel.
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
    cap("one image per channel — the channel count varies per sample", "[C_b, 1, H, W]")
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
    # 3 -- channel-tied patch embedding; identity carried -> token set.
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
        "one tied P×P 1→D filter (shared across channels); identity z_c carried alongside",
        "[ΣC, N, D] = [5, 16, 192]   ·   visible set [V, D]",
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
        "a learned query pools the channels at each location · no spatial PE",
        "[Kf, D] = [32, 192]  (Kf = Σ_b N_b, one row per present location)",
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
        "DCP drops channel-patch tokens before fusion (simplified here: whole locations)",
        "[Kf, D]  →  [Kf', D]   (omitted rows never exist, never zeroed)",
        k.RED,
    )
    s.wait(0.5)

    # ===================================================================== #
    # 6 -- grouped spatial transformer over the fused location tokens.
    # ===================================================================== #
    nxt("6 / 10")
    arrow_enc = k.Arrow(start=(2.0, 0.35), end=(3.3, 0.35), stroke=k.GRAY, fill=k.GRAY)
    s.play(k.draw(arrow_enc))
    cap(
        "visible location tokens go through the transformer (per-sample groups)",
        "[Kf, D] = [Kf, 192]   (no memory tokens; the memory arm adds M)",
        k.TEAL,
    )
    s.wait(0.5)

    # ===================================================================== #
    # 7 -- spatial pooling: fixed latents for the sample.
    # ===================================================================== #
    nxt("7 / 10")
    # latents [B, L, D]: the fixed-size set from spatial pooling (L = n_latents).
    lat = k.RoundedRect(
        w=0.5,
        h=1.5,
        radius=0.14,
        x=5.9,
        y=0.33,
        fill=LAT,
        fill_opacity=0.55,
        stroke=LAT,
        stroke_width=2.0,
        opacity=0.0,
    )
    s.add(lat)
    lat_label = k.Text("[B, L, D]", size=0.30, mono=True, color=LAT, x=5.9, y=-1.15)
    s.play(
        lat.to(opacity=1.0),
        k.fade_in(lat_label, shift=(0.3, 0)),
        duration=0.6,
    )
    cap(
        "spatial pooling → fixed latents [B,L,D], trained by the loss",
        "[B, L, D] = [2, 16, 192]   →  decoder KV",
        k.PINK,
    )
    s.play(k.indicate(lat), duration=0.6)
    s.wait(0.5)

    # ===================================================================== #
    # 8 -- channel-aware shared decoder -> raw pixels for masked targets.
    # ===================================================================== #
    nxt("8 / 10")
    # clear the tokenizer stage to make room
    alive_fused = [f for key, f in fused.items() if key not in omitted]
    s.play(
        k.fade_out(
            *patch.values(), *alive_fused, arrow_enc,
            lat, lat_label,
        ),
        duration=0.5,
    )
    q1 = k.Text("mask token\n+ position\n+ target z_c", size=0.26, x=-2.4, y=0.0)
    qcell = cell(-1.0, 0.0, k.ORANGE, size=0.9, opacity=0.0)
    arrow_dec = k.Arrow(start=(-0.45, 0.0), end=(0.75, 0.0), stroke=k.GRAY, fill=k.GRAY)
    px = k.Text("[Σ p²]", size=0.4, mono=True, x=2.0, y=0.0)
    px_note = k.Text(
        "flat pixels: row i = p_i² contiguous (p_i = p_b)\nno padded block",
        size=0.19,
        color=k.GREEN,
        x=0.3,
        y=-1.05,
    )
    tgt = k.RoundedRect(
        w=1.5, h=1.5, radius=0.18, x=4.5, y=0.0,
        fill=k.GREEN, fill_opacity=0.35, stroke=k.GREEN, stroke_width=2.0, opacity=0.0,
    )
    tgt_label = k.Text("conceptual\nKt × p² grid", size=0.24, color=k.GREEN, x=4.5, y=-1.35)
    s.add(qcell, tgt)
    s.play(
        *[qcell.to(opacity=1.0), tgt.to(opacity=1.0)],
        k.fade_in(q1, px, px_note, tgt_label),
        duration=0.5,
    )
    s.play(k.draw(arrow_dec))
    cap(
        "shared decoder: target self-attn, then cross-attn to the latents",
        "KV = latents (fixed, does not grow with N)",
        k.GREEN,
    )
    s.play(k.indicate(px, scale=1.15), k.indicate(tgt, scale=1.1), duration=0.7)
    s.wait(0.4)

    # ===================================================================== #
    # 9 -- the objective.
    # ===================================================================== #
    nxt("9 / 10")
    loss = k.Text(
        "default: standardised-target MSE, mean per sample",
        size=0.26,
        color=k.YELLOW,
        x=0.0,
        y=-1.55,
        align="center",
    )
    loss_abl = k.Text(
        "ablation: (1-λ)·MSE(raw) + λ·L1(|rFFT|), λ = 0.01",
        size=0.20,
        color=k.GRAY,
        x=0.0,
        y=-1.93,
        align="center",
    )
    s.play(k.write(loss), k.fade_in(loss_abl, shift=(0, 0.15)))
    cap(
        "the only place C appears is the input length — summed, not padded",
        "C ∈ {1 … 60+}  ·  no C_max  ·  no pad token\nno -inf attention mask (masking = omission)",
        k.YELLOW,
    )
    s.wait(0.5)

    # ===================================================================== #
    # 10 -- summary: the shape chain.
    # ===================================================================== #
    nxt("10 / 10")
    s.play(
        k.fade_out(
            q1, qcell, px, px_note, tgt, tgt_label, arrow_dec, loss, loss_abl,
            *img.values(),
        ),
        duration=0.4,
    )
    # Two rows: the pooled outputs are parallel siblings of ``tok_out``, not a
    # chain into each other; the decoder KV is the fixed latent set.
    pills = []
    arrows = []

    def pill(text, color, x, y, w=1.9, h=0.82, size=0.22):
        pills.append(
            k.RoundedRect(
                w=w, h=h, radius=0.16, x=x, y=y,
                fill=color, fill_opacity=0.22, stroke=color, stroke_width=2.0,
            )
        )
        pills.append(k.Text(text, size=size, mono=True, color=k.WHITE, x=x, y=y))

    y1 = 0.25
    pill("[ΣC, 1, H, W]", k.BLUE, -4.6, y1)
    pill("[ΣC, N, D]", k.BLUE, -2.4, y1)
    pill("[Kf, D]", k.YELLOW, -0.2, y1, w=1.7)
    arrows += [
        k.Arrow(start=(-3.63, y1), end=(-3.37, y1), stroke=k.GRAY, fill=k.GRAY),
        k.Arrow(start=(-1.43, y1), end=(-1.07, y1), stroke=k.GRAY, fill=k.GRAY),
    ]

    # the decoder KV: the fixed latent set (memory rows are the arm only).
    y2 = -1.1
    pill("[B, L, D]", LAT, -0.2, y2, w=1.9, h=1.45, size=0.20)
    arrows.append(k.Arrow(start=(-0.2, -0.18), end=(-0.2, -0.33), stroke=k.GRAY, fill=k.GRAY))

    pill("[B·L, D]", k.TEAL, 2.2, y2, w=1.6, size=0.20)
    pill("[Kt, D]", k.GREEN, 4.5, y2, w=1.5)
    pill("[Σ p²]", k.GREEN, 6.4, y2, w=1.6)
    arrows += [
        k.Arrow(start=(0.76, y2), end=(1.03, y2), stroke=k.GRAY, fill=k.GRAY),
        k.Arrow(start=(3.37, y2), end=(3.73, y2), stroke=k.GRAY, fill=k.GRAY),
        k.Arrow(start=(5.27, y2), end=(5.58, y2), stroke=k.GRAY, fill=k.GRAY),
    ]
    s.play(k.fade_in(*pills, shift=(0, 0.2)), duration=0.6)
    s.play(k.stagger([k.draw(a) for a in arrows], lag=0.08), duration=0.5)
    s.play(
        k.fade_in(
            k.Text("latent pooler\nreads tok_out", size=0.17, color=LAT, x=-2.45, y=y2),
            k.Text("KV = latents", size=0.18, color=k.TEAL, x=2.2, y=-0.5),
        ),
        duration=0.4,
    )
    cap("variable length → fixed everything else", "", k.GREEN, dur=0.3)

    s.wait(1.2)
