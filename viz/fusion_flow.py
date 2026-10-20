"""kinemo animation: what the channel-fusion module actually computes.

The default fusion is :class:`fuse_channels.fusion.ChannelFusion` in
``mode="per_location"`` -- PMA-style pooling of the channel axis at every
spatial location.  This scene zooms into **one** location (one sample, one
patch) and follows its channel tokens through every operation:

    gather channels -> LayerNorm (k = v) -> per-head query · key / sqrt(Dh)
    -> softmax over the C channels -> sum(alpha * v) -> concat heads
    -> shared Linear -> scatter into [B, N, Q*D]

The numbers come from a real ``ChannelFusion`` forward pass with an
illustrative ``dim=4, n_heads=2, n_queries=1`` (the model default is ``dim=192,
n_heads=6``; a 192-wide vector is unreadable, the math is identical).  That
probe also confirmed the structural facts this animation leans on:

* keys and values are the **same** tensor -- ``k = v = norm_kv(tokens)``;
* there are **no** separate ``W_Q/W_K/W_V`` -- the query is a learned per-patch
  parameter and the only learned linear is the output projection ``proj_out``;
* the learned query is indexed by *patch*, so it is shared across samples.

The two dot-product steps use ``k.matrix_product`` because that is literally
what they are: ``q_h · K_hᵀ -> scores`` and ``alpha_h @ V_h -> o_h``.

Render::

    kinemo check --strict viz/fusion_flow.py
    kinemo render viz/fusion_flow.py --format mp4 --out viz/out/fusion_flow.mp4
    kinemo render viz/fusion_flow.py --format gif --size 720p --out viz/out/fusion_flow.gif
"""

from __future__ import annotations

import kinemo as k

# --------------------------------------------------------------------------- #
# Numbers from one real ChannelFusion forward pass
#   dim=4, heads=2, Dh=2, n_queries=1, C=3 channels at patch j=1.
# --------------------------------------------------------------------------- #
D = 4
H = 2
DH = D // H
C = 3

X = [  # raw channel tokens x_c, [C][D]
    [-1.29, -0.79, -0.02, -0.72],
    [0.52, -1.31, 0.19, 0.54],
    [-2.22, 0.26, -1.03, -0.50],
]
KV = [  # k = v = LayerNorm(x)  [C][D]
    [-1.29, -0.19, 1.51, -0.03],
    [0.70, -1.70, 0.27, 0.73],
    [-1.49, 1.25, -0.17, 0.41],
]
Q = [-0.37, -1.43, 1.21, 0.59]
Q_RAW = [-0.13, -0.45, 0.34, 0.16]  # the learned parameter before LayerNorm

ALPHA = [[0.25, 0.57], [0.69, 0.27], [0.06, 0.16]]  # [channel][head]
HEADS = [[0.07, -1.15], [0.90, 0.25]]  # [head][Dh]
CONCAT = HEADS[0] + HEADS[1]
W = [  # proj_out weight, [D][D]
    [-0.00, 0.27, -0.41, -0.37],
    [-0.19, 0.13, -0.01, 0.40],
    [-0.04, 0.13, -0.15, -0.10],
    [-0.48, -0.33, -0.21, 0.02],
]
BIAS = [0.20, 0.30, -0.34, -0.22]
FUSED = [-0.57, 0.22, -0.65, -0.05]

CH = [k.BLUE, k.ORANGE, k.PURPLE]
QCOL = k.TEAL
H0COL = k.YELLOW
H1COL = k.GREEN
FUSEDCOL = k.PINK

CELL = 0.5
GAP = 0.1
STEP = CELL + GAP
YROWS = [1.35, 0.6, -0.15]
XL = -5.5
XR = 0.5


def vrow(values, x0, y, color, opacity=0.0, cell=CELL, gap=GAP, stroke=None):
    """A tensor as a row of cells (one cell = one element)."""
    out = []
    for i, _v in enumerate(values):
        x = x0 + i * (cell + gap)
        c = color[i] if isinstance(color, (list, tuple)) else color
        out.append(
            k.RoundedRect(
                w=cell, h=cell, radius=0.06, x=x, y=y,
                fill=c, fill_opacity=0.85, stroke=stroke or c,
                stroke_width=1.3, opacity=opacity,
            )
        )
    return out


def nrow(values, x0, y, size=0.16, color=k.GRAY, opacity=0.0):
    """Numeric labels under a token row."""
    out = []
    for i, v in enumerate(values):
        x = x0 + i * STEP
        out.append(k.Text(f"{v:+.2f}", size=size, x=x, y=y, color=color, opacity=opacity))
    return out


def box_around(x0, i0, i1, y, color, opacity=0.0):
    """Outline a contiguous span of a token row (i0..i1 inclusive)."""
    x_left = x0 + i0 * STEP - CELL / 2 - 0.05
    x_right = x0 + i1 * STEP + CELL / 2 + 0.05
    return k.RoundedRect(
        w=x_right - x_left, h=CELL + 0.10, radius=0.10,
        x=(x_left + x_right) / 2, y=y, fill_opacity=0.0,
        stroke=color, stroke_width=2.2, opacity=opacity,
    )


@k.scene(size=(1920, 1080), fps=60)
def fusion_flow(s: k.Scene):
    title = k.Text("Inside channel fusion", size=0.6).place(at="top", margin=0.5)
    kicker = k.Text(
        "pooling by multi-head attention at every spatial location",
        size=0.28, color=k.GRAY,
    ).place(below=title, gap=0.14)
    step = k.Text("1 / 11", size=0.28, mono=True, color=k.GRAY).place(
        at="top-right", margin=0.55
    )
    desc = k.Text("", size=0.32, color=k.GRAY, x=0.0, y=-2.62, width=13.5, align="center")
    formula = k.Text("", size=0.32, mono=True, x=0.0, y=-3.3)

    def cap(description, f, color=k.WHITE, dur=0.45):
        desc.set(text=description)
        formula.set(text=f, color=color)
        s.wait(dur)

    def nxt(label):
        s.play(step.to(text=label), duration=0.22)

    s.play(k.write(title), k.fade_in(kicker))
    s.play(k.fade_in(desc, formula, step))

    # ===================================================================== #
    # 1 -- context: channel-patch tokens, locations, segments.
    # ===================================================================== #
    loc_labels, stacks, grid = [], [], []
    counts = [3, 2]
    for si in range(2):
        for pj in range(3):
            cx = -3.1 + pj * 2.6
            cy = 1.0 - si * 1.85
            grid.append(
                k.RoundedRect(
                    w=1.5, h=1.3, radius=0.14, x=cx, y=cy,
                    fill=k.GRAY, fill_opacity=0.06, stroke=k.GRAY,
                    stroke_width=1.2, opacity=0.0,
                )
            )
            for c in range(counts[si]):
                stacks.append(
                    k.RoundedRect(
                        w=0.28, h=0.20, radius=0.05,
                        x=cx - 0.28, y=cy + (counts[si] - 1) / 2 * 0.26 - c * 0.26,
                        fill=CH[c], fill_opacity=0.85, stroke=CH[c],
                        stroke_width=1.0, opacity=0.0,
                    )
                )
            loc_labels.append(
                k.Text(f"{counts[si]} ch", size=0.2, color=k.GRAY, x=cx + 0.42, y=cy, opacity=0.0)
            )
    hl = k.RoundedRect(
        w=1.62, h=1.42, radius=0.16, x=-0.5, y=1.0,
        fill_opacity=0.0, stroke=H0COL, stroke_width=2.4, opacity=0.0,
    )
    axis_p = k.Text("patch →", size=0.24, color=k.GRAY, x=0.0, y=-2.05, opacity=0.0)
    axis_s = k.Text("sample ↓", size=0.24, color=k.GRAY, x=-4.5, y=0.1, opacity=0.0)
    s.add(*grid, *stacks, *loc_labels, hl, axis_p, axis_s)
    s.play(
        k.stagger([g.to(opacity=1.0) for g in grid], lag=0.05),
        k.stagger([st.to(opacity=1.0) for st in stacks], lag=0.015),
        *[t.to(opacity=1.0) for t in loc_labels],
        axis_p.to(opacity=1.0), axis_s.to(opacity=1.0),
        duration=0.6,
    )
    cap(
        "a token is one channel-patch; a segment groups channels at one location",
        "seg = sample_idx · N + patch_idx",
    )
    s.play(hl.to(opacity=1.0), duration=0.3)
    s.play(k.indicate(hl), duration=0.5)
    s.wait(0.3)

    # ===================================================================== #
    # 2 -- zoom into the highlighted location: its C channel tokens.
    # ===================================================================== #
    nxt("2 / 11")
    s.play(k.fade_out(*grid, *stacks, *loc_labels, hl, axis_p, axis_s), duration=0.5)
    x_cells, x_nums, x_lab = [], [], []
    for c in range(C):
        x_cells += vrow(X[c], XL, YROWS[c], CH[c])
        x_nums += nrow(X[c], XL, YROWS[c] - 0.4)
        x_lab.append(
            k.Text(f"x_{c + 1}", size=0.28, mono=True, x=XL - 0.55, y=YROWS[c],
                   color=CH[c], opacity=0.0)
        )
    s.add(*x_cells, *x_nums, *x_lab)
    s.play(
        k.stagger([c.to(opacity=1.0) for c in x_cells], lag=0.01),
        *[n.to(opacity=1.0) for n in x_nums],
        *[l.to(opacity=1.0) for l in x_lab],
        duration=0.7,
    )
    cap(
        "zoom in: these C channel tokens are the whole attention set",
        "tokens x_c ∈ R^D,  c = 1 … C      (C=3, D=4 here; D=192 real)",
    )
    s.wait(0.3)

    # ===================================================================== #
    # 3 -- LayerNorm: keys and values are the same tensor.
    # ===================================================================== #
    nxt("3 / 11")
    k_cells, k_nums = [], []
    for c in range(C):
        k_cells += vrow(KV[c], XR, YROWS[c], CH[c], stroke=k.WHITE)
        k_nums += nrow(KV[c], XR, YROWS[c] - 0.4)
    k_head = k.Text("k = v = LayerNorm(x)", size=0.3, mono=True, x=2.0, y=1.95,
                    color=k.WHITE, opacity=0.0)
    arrows = [
        k.Arrow(start=(XL + 2.35, YROWS[c]), end=(XR - 2.35, YROWS[c]),
                stroke=k.GRAY, fill=k.GRAY, opacity=0.0)
        for c in range(C)
    ]
    norm_lab = k.Text("LayerNorm", size=0.26, color=k.GRAY, x=(XL + XR) / 2 + 0.1, y=1.0,
                      opacity=0.0)
    s.add(*k_cells, *k_nums, k_head, *arrows, norm_lab)
    s.play(
        k.stagger([c.to(opacity=1.0) for c in k_cells], lag=0.04),
        *[n.to(opacity=1.0) for n in k_nums],
        k_head.to(opacity=1.0), norm_lab.to(opacity=1.0),
        k.stagger([a.to(opacity=1.0) for a in arrows], lag=0.05),
        duration=0.8,
    )
    cap(
        "normalize every token; keys and values are literally the same tensor",
        "k_c = v_c = (x_c − μ) / σ ⊙ γ + β        (γ = 1, β = 0 fresh here)",
    )
    s.wait(0.3)

    # ===================================================================== #
    # 4 -- one learned query per patch, shared across samples.
    # ===================================================================== #
    nxt("4 / 11")
    s.play(k.fade_out(*x_cells, *x_nums, *x_lab), duration=0.5)
    q_cells = vrow(Q, XL, -1.3, QCOL)
    q_nums = nrow(Q, XL, -1.7)
    q_lab = k.Text("q_j", size=0.28, mono=True, x=XL - 0.55, y=-1.3, color=QCOL, opacity=0.0)
    s.add(*q_cells, *q_nums, q_lab)
    s.play(
        k.stagger([c.to(opacity=1.0) for c in q_cells], lag=0.02),
        *[n.to(opacity=1.0) for n in q_nums],
        q_lab.to(opacity=1.0),
        duration=0.6,
    )
    cap(
        "the query is a parameter, not a projection: there is no W_Q; it is shared across samples",
        "q_j = LayerNorm(queries[j]) ∈ R^D",
    )
    s.wait(0.3)

    # ===================================================================== #
    # 5 -- split D into H heads (attention is independent per head).
    # ===================================================================== #
    nxt("5 / 11")
    boxes = [box_around(XL, 0, DH - 1, -1.3, H0COL), box_around(XL, DH, D - 1, -1.3, H1COL)]
    for c in range(C):
        boxes += [
            box_around(XR, 0, DH - 1, YROWS[c], H0COL),
            box_around(XR, DH, D - 1, YROWS[c], H1COL),
        ]
    leg = [
        k.RoundedRect(w=0.22, h=0.22, radius=0.05, x=3.35, y=0.75,
                      fill=H0COL, fill_opacity=0.85, stroke=H0COL, opacity=0.0),
        k.Text("head 1", size=0.24, color=H0COL, x=4.15, y=0.75, opacity=0.0),
        k.RoundedRect(w=0.22, h=0.22, radius=0.05, x=3.35, y=0.3,
                      fill=H1COL, fill_opacity=0.85, stroke=H1COL, opacity=0.0),
        k.Text("head 2", size=0.24, color=H1COL, x=4.15, y=0.3, opacity=0.0),
    ]
    s.add(*boxes, *leg)
    s.play(
        k.stagger([b.to(opacity=1.0) for b in boxes], lag=0.05),
        *[t.to(opacity=1.0) for t in leg],
        duration=0.7,
    )
    cap(
        "split every token into H heads; attention runs independently per head",
        "D = H · D_h        (here 4 = 2 × 2)",
    )
    s.wait(0.3)

    # ===================================================================== #
    # 6 -- scores = q_h · k_c,h / sqrt(D_h), shown as a dot product.
    # ===================================================================== #
    nxt("6 / 11")
    s.play(
        k.fade_out(*boxes, *leg, *q_cells, *q_nums, q_lab,
                   *k_cells, *k_nums, k_head, *arrows, norm_lab),
        duration=0.4,
    )
    A1 = k.Matrix([[Q[0], Q[1]]], size=0.38, x=-4.2, y=0.3)
    B1 = k.Matrix([[KV[c][0] for c in range(C)], [KV[c][1] for c in range(C)]],
                  size=0.38, x=-0.5, y=0.3)
    P1 = A1 @ B1
    P1.set(x=3.4, y=0.3)
    score_lab = k.Text("head 1:  dot products  q_h · k_{c,h}   (then ÷ √D_h)",
                       size=0.28, mono=True, x=-3.0, y=1.45, color=H0COL)
    s.play(k.draw(A1), k.draw(B1))
    s.play(k.matrix_product(A1, B1, P1))
    s.play(k.fade_in(score_lab), duration=0.4)
    cap(
        "each score is a dot product of the head query with one channel's head key",
        "s_{c,h} = (q_h · k_{c,h}) / √D_h",
        H0COL,
    )
    s.wait(0.3)

    # ===================================================================== #
    # 7 -- softmax over the C channels (per head).
    # ===================================================================== #
    nxt("7 / 11")
    s.play(k.fade_out(A1, B1, P1, score_lab), duration=0.4)
    soft_lab = k.Text("α = softmax over the C channels", size=0.28, x=-3.0, y=1.8,
                      color=k.WHITE)
    alpha_m = k.Matrix(ALPHA, size=0.5, x=-0.4, y=0.2)  # rows = channels, cols = heads
    s.play(k.fade_in(soft_lab), k.draw(alpha_m), duration=0.6)
    cap(
        "softmax runs over the channel axis: each head's weights sum to 1 across channels",
        "α_{c,h} = softmax_c(s_{c,h})        Σ_c α_{c,h} = 1",
    )
    s.wait(0.3)

    # ===================================================================== #
    # 8 -- weighted sum of the value vectors -> head outputs.
    # ===================================================================== #
    nxt("8 / 11")
    s.play(k.fade_out(soft_lab, alpha_m), duration=0.4)
    a_h0 = k.Matrix([[ALPHA[0][0], ALPHA[1][0], ALPHA[2][0]]], size=0.4, x=-4.0, y=0.2)
    Vt0 = k.Matrix([[KV[0][0], KV[0][1]], [KV[1][0], KV[1][1]], [KV[2][0], KV[2][1]]],
                   size=0.4, x=0.0, y=0.2)
    o0 = a_h0 @ Vt0
    o0.set(x=3.6, y=0.2)
    sum_lab = k.Text("output o_h = α_h @ V_h   (weights the same k = v tensor)",
                     size=0.28, mono=True, x=-2.3, y=1.75, color=H0COL)
    s.play(k.draw(a_h0), k.draw(Vt0))
    s.play(k.matrix_product(a_h0, Vt0, o0))
    s.play(k.fade_in(sum_lab), duration=0.4)
    cap(
        "each head output is the attention-weighted average of its channel values",
        "o_h = Σ_c α_{c,h} · v_{c,h}",
        H0COL,
    )
    s.wait(0.3)

    # ===================================================================== #
    # 9 -- concatenate heads and apply the one shared Linear projection.
    # ===================================================================== #
    nxt("9 / 11")
    s.play(k.fade_out(a_h0, Vt0, o0, sum_lab), duration=0.4)
    Wm = k.Matrix(W, size=0.3, x=-3.2, y=0.2)
    cm = k.Matrix([[v] for v in CONCAT], size=0.3, x=0.5, y=0.2)
    pm = Wm @ cm
    pm.set(x=2.6, y=0.2)
    proj_lab = k.Text("proj_out:  y = W · concat(o_h) + b", size=0.28, mono=True,
                      x=-2.0, y=2.05, color=k.WHITE)
    s.play(k.draw(Wm), k.draw(cm))
    s.play(k.matrix_product(Wm, cm, pm))
    s.play(k.fade_in(proj_lab), duration=0.4)
    cap(
        "concatenate the heads, then one shared Linear projection — the only learned mix",
        "y = W · concat_h(o_h) + b        y ∈ R^D",
    )
    s.wait(0.3)

    # ===================================================================== #
    # 10 -- scatter back into [B, N, Q·D]: presence, not padding.
    # ===================================================================== #
    nxt("10 / 11")
    s.play(k.fade_out(Wm, cm, pm, proj_lab), duration=0.4)
    out_grid, out_fill = [], []
    for si in range(2):
        for pj in range(4):
            x = -3.9 + pj * 2.6
            y = 0.8 - si * 1.9
            out_grid.append(
                k.RoundedRect(w=1.7, h=1.3, radius=0.14, x=x, y=y,
                              fill=k.GRAY, fill_opacity=0.06, stroke=k.GRAY,
                              stroke_width=1.2, opacity=0.0)
            )
            is_hi = si == 0 and pj == 1
            col = FUSEDCOL if is_hi else k.GRAY
            out_fill.append(
                k.RoundedRect(w=1.1, h=0.4, radius=0.08, x=x, y=y,
                              fill=col, fill_opacity=0.55 if is_hi else 0.0,
                              stroke=col, stroke_width=1.5, opacity=0.0)
            )
    out_lab = k.Text("fused[sample,  patch]  →  [B, N, Q·D]", size=0.3, mono=True,
                     x=0.0, y=2.05, color=FUSEDCOL)
    out_note = k.Text(
        "the vector we just computed lands at (sample 1, patch 2);\n"
        "locations with no visible channel are simply absent from the encoder input",
        size=0.24, color=k.GRAY, x=0.0, y=-2.05, width=11.5,
    )
    s.add(*out_grid, *out_fill)
    s.play(
        k.stagger([g.to(opacity=1.0) for g in out_grid], lag=0.04),
        k.stagger([f.to(opacity=1.0) for f in out_fill], lag=0.04),
        k.fade_in(out_lab, out_note),
        duration=0.7,
    )
    cap(
        "one fused vector per present location: no C_max, no pad slot",
        "[ΣC, N, D]  →  [B, N, Q·D]        present = any visible channel",
        FUSEDCOL,
    )
    s.wait(0.3)

    # ===================================================================== #
    # 11 -- summary of the whole pipeline.
    # ===================================================================== #
    nxt("11 / 11")
    s.play(k.fade_out(*out_grid, *out_fill, out_lab, out_note), duration=0.4)
    ops = [
        ("gather", k.BLUE),
        ("LayerNorm", k.WHITE),
        ("q·kᵀ/√D", k.TEAL),
        ("softmax_c", k.YELLOW),
        ("Σ α·v", k.ORANGE),
        ("concat", k.PURPLE),
        ("proj_out", k.PINK),
        ("scatter", k.GREEN),
    ]
    pills, arow = [], []
    for i, (txt, col) in enumerate(ops):
        x = -5.25 + i * 1.5
        pills.append(
            k.RoundedRect(w=1.4, h=0.8, radius=0.14, x=x, y=0.3,
                          fill=col, fill_opacity=0.22, stroke=col,
                          stroke_width=1.8, opacity=0.0)
        )
        pills.append(
            k.Text(txt, size=0.22, mono=True, color=k.WHITE, x=x, y=0.3, opacity=0.0)
        )
        if i < len(ops) - 1:
            arow.append(
                k.Arrow(start=(x + 0.72, 0.3), end=(x + 0.78, 0.3),
                        stroke=k.GRAY, fill=k.GRAY, opacity=0.0)
            )
    s.add(*pills, *arow)
    s.play(k.stagger([p.to(opacity=1.0) for p in pills], lag=0.05),
           k.stagger([a.to(opacity=1.0) for a in arow], lag=0.05), duration=0.7)
    closing = k.Text(
        "C is summed away at every location: cost O(C), no C_max, no pad token",
        size=0.34, color=k.GREEN, x=0.0, y=-0.9,
    )
    s.play(k.write(closing))
    cap(
        "variable channel count in, fixed-size fused tokens out",
        "[C_b, 1, H, W] → [ΣC, N, D] → [B, N, D]",
        k.GREEN,
    )
    s.wait(1.0)
