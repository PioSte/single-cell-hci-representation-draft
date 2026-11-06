"""kinemo animation: what the default channel-fusion module actually computes.

The default fusion is :class:`fuse_channels.fusion.ChannelFusion` -- single-query
PMA-style pooling of the channel axis at every spatial location.  This scene
zooms into **one** location (one sample, one patch) and follows its channel
tokens through every operation of the default path:

    gather channels -> norm_kv -> W_K/W_V and norm_q/W_Q -> per-head query
    -> scores q_h . k_c,h / sqrt(D_h)
    -> + identity_bias(z_c)  ==  logits s_{c,h} + b_{c,h}
    -> softmax over the C channels (per head) -> alpha
    -> sum_c alpha_c,h v_c,h -> concat heads -> proj_out -> content path
    -> + identity_skip(sum_c w_c z_c) -> one [D] row per present location

No post-fusion spatial positional embedding exists on the default path (the
encoder owns layout through 2-D axial RoPE).  ``IdentityGatedSum`` is the
ablation and is **not** drawn.

The numbers come from a real ``ChannelFusion`` forward pass with an
illustrative ``dim=4, n_heads=2, attn_dim=4``.  ``attn_dim = D`` is a
*readability* choice: the model default is ``attn_dim = 3D`` (Panopticon's
``D_attn``).  The probe set the ``identity_bias`` / ``identity_skip`` weights
explicitly so the routing bias and the skip are visible; the arithmetic is
still exactly the module's ``forward``:

* keys and values come from separate learned projections applied to
  ``norm_kv(tokens)``;
* the query is projected by ``W_Q`` after ``norm_q`` and shared by every sample
  and every spatial location (there is no ``n_queries`` argument);
* ``identity_bias: Linear(D, n_heads)`` adds a per-head bias ``b_{c,h}`` to the
  logits **before** the grouped channel softmax;
* after ``proj_out``, ``identity_skip: Linear(D, D)`` adds a residual of the
  attention-pooled identity ``sum_c w_c z_c`` with ``w_c = mean_h alpha_{c,h}``.

Render::

    kinemo check --strict viz/fusion_flow.py
    kinemo render viz/fusion_flow.py --format mp4 --out viz/out/fusion_flow.mp4
    kinemo render viz/fusion_flow.py --format gif --size 720p --out viz/out/fusion_flow.gif
"""

from __future__ import annotations

import kinemo as k

# --------------------------------------------------------------------------- #
# Numbers from one real ChannelFusion forward pass
#   dim=4, heads=2, Dh=2, attn_dim=4, one learned query, C=3 channels at one
#   (sample, patch).  identity_bias / identity_skip weights were set so the
#   bias and skip vectors are legible; every number below is that forward pass.
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
Z = [  # channel identity z_c, [C][D] (never added to the content tokens)
    [0.80, -0.40, 0.20, 0.60],
    [-0.60, 1.00, -0.80, 0.40],
    [0.40, 0.20, 1.20, -1.00],
]
K = [  # k = k_proj(norm_kv(x))  [C][D]
    [0.36, -0.71, 0.29, -0.99],
    [0.19, 0.87, -0.11, -1.38],
    [0.05, -1.19, -0.25, 0.10],
]
V = [  # v = v_proj(norm_kv(x))  [C][D]
    [0.77, 0.09, 0.85, -0.26],
    [-0.41, 0.53, 0.55, -0.78],
    [1.18, 0.04, -0.27, 0.50],
]
Q = [0.68, -0.93, -0.11, 0.02]  # q = q_proj(norm_q(queries))

BIAS = [[0.62, -0.16], [-0.72, 0.72], [0.28, -0.42]]          # b_{c,h}
SCORES = [[0.63, -0.03], [-0.48, -0.01], [0.80, 0.02]]        # s_{c,h}
LOGITS = [[1.25, -0.19], [-1.20, 0.71], [1.08, -0.40]]        # s_{c,h}+b_{c,h}
ALPHA = [[0.52, 0.23], [0.05, 0.58], [0.44, 0.19]]            # [channel][head]
WMEAN = [0.38, 0.31, 0.31]                                    # w_c = mean_h alpha
HEADS = [[0.89, 0.08], [0.46, -0.41]]                         # [head][Dh]
CONCAT = HEADS[0] + HEADS[1]
W = [  # proj_out weight, [D][D]
    [-0.27, -0.24, -0.28, 0.09],
    [0.10, -0.21, 0.33, -0.28],
    [0.10, -0.14, -0.43, -0.33],
    [0.20, 0.20, -0.31, -0.16],
]
BIAS_OUT = [-0.47, 0.27, 0.43, 0.16]                          # proj_out bias
Y_CONTENT = [-0.89, 0.61, 0.44, 0.28]                         # proj_out(concat)+b
IPOOL = [0.24, 0.22, 0.20, 0.04]                              # sum_c w_c z_c
SKIP = [0.29, 0.29, 0.22, 0.04]                               # identity_skip(IPOOL)
FUSED = [-0.61, 0.89, 0.66, 0.32]                             # y_content + skip

CH = [k.BLUE, k.ORANGE, k.PURPLE]
QCOL = k.TEAL
H0COL = k.YELLOW
H1COL = k.GREEN
FUSEDCOL = k.PINK
IDCOL = "#56B4E9"

CELL = 0.5
GAP = 0.1
STEP = CELL + GAP
YROWS = [1.55, 0.70, -0.15]      # three channel rows
XL = -5.5                        # first-cell centre of a left token block


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


def tag_box(x0, n, y, color, opacity=0.0, pad_x=0.12, pad_y=0.14):
    """A highlight box around `n` consecutive cells of a row."""
    w = n * CELL + (n - 1) * GAP + 2 * pad_x
    h = CELL + 2 * pad_y
    cx = x0 + (n - 1) * STEP / 2
    return k.RoundedRect(
        w=w, h=h, radius=0.12, x=cx, y=y, fill_opacity=0.0,
        stroke=color, stroke_width=2.6, opacity=opacity,
    )


def callout(text, target, side, color, style="box", gap=0.35, size=0.26, max_width=2.6):
    """A named label with a leader line pointing at `target`."""
    return k.Callout(text, to=target, side=side, style=style, gap=gap, size=size,
                     max_width=max_width, color=color, opacity=0.0)


def opbox(text, x, y, color=k.WHITE, w=1.7, h=0.62, size=0.22, sub=None,
          opacity=0.0):
    """An operation / module block (plain, never crossed)."""
    box = k.RoundedRect(w=w, h=h, radius=0.12, x=x, y=y, fill=color,
                        fill_opacity=0.12, stroke=color, stroke_width=1.7,
                        opacity=opacity)
    if sub is None:
        label = k.Text(text, size=size, mono=True, color=color, x=x, y=y,
                       opacity=opacity)
        return box, label
    label = k.Text(text, size=size, mono=True, color=color, x=x, y=y + h * 0.16,
                   opacity=opacity)
    sublabel = k.Text(sub, size=size - 0.05, color=k.GRAY, x=x, y=y - h * 0.20,
                      opacity=opacity)
    return box, label, sublabel


def arrow(a, b, color=k.GRAY, opacity=0.0):
    return k.Arrow(start=a, end=b, stroke=color, fill=color, opacity=opacity)


@k.scene(size=(1920, 1080), fps=60)
def fusion_flow(s: k.Scene):
    title = k.Text("Inside channel fusion", size=0.6).place(at="top", margin=0.5)
    kicker = k.Text(
        "the default ChannelFusion, one location at a time",
        size=0.28, color=k.GRAY,
    ).place(below=title, gap=0.14)
    step = k.Text("1 / 14", size=0.28, mono=True, color=k.GRAY).place(
        at="top-right", margin=0.55
    )
    desc = k.Text("", size=0.32, color=k.GRAY, x=0.0, y=-2.62, width=13.5,
                  align="center")
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
    # 1 -- context: channel-patch tokens, locations, groups.
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
                k.Text(f"{counts[si]} ch", size=0.2, color=k.GRAY, x=cx + 0.42,
                       y=cy, opacity=0.0)
            )
    hl = k.RoundedRect(
        w=1.62, h=1.42, radius=0.16, x=2.1, y=1.0,
        fill_opacity=0.0, stroke=H0COL, stroke_width=2.4, opacity=0.0,
    )
    axis_p = k.Text("patch →", size=0.24, color=k.GRAY, x=0.0, y=-2.05, opacity=0.0)
    axis_s = k.Text("sample ↓", size=0.24, color=k.GRAY, x=-4.5, y=0.1, opacity=0.0)
    c_group = callout("one group = the channels at one location", hl, "right", H0COL)
    c_tok = callout("a channel token = one channel-patch", stacks[9], "bottom", CH[0])
    s.add(*grid, *stacks, *loc_labels, hl, axis_p, axis_s, c_group, c_tok)
    s.play(
        k.stagger([g.to(opacity=1.0) for g in grid], lag=0.05),
        k.stagger([st.to(opacity=1.0) for st in stacks], lag=0.015),
        *[t.to(opacity=1.0) for t in loc_labels],
        axis_p.to(opacity=1.0), axis_s.to(opacity=1.0),
        duration=0.6,
    )
    cap(
        "a token is one channel-patch; a group holds the channels at one location",
        "group_id = sample_idx · N + patch_idx",
    )
    s.play(hl.to(opacity=1.0), duration=0.3)
    s.play(k.indicate(hl), duration=0.5)
    s.play(c_group.to(opacity=1.0), c_tok.to(opacity=1.0), duration=0.5)
    s.wait(0.3)

    # ===================================================================== #
    # 2 -- zoom in: content tokens x_c AND channel identity z_c.
    # ===================================================================== #
    nxt("2 / 14")
    s.play(k.fade_out(*grid, *stacks, *loc_labels, hl, axis_p, axis_s, c_group, c_tok),
           duration=0.5)
    # content tokens
    x_cells, x_nums, x_lab = [], [], []
    for c in range(C):
        x_cells += vrow(X[c], XL, YROWS[c], CH[c])
        x_nums += nrow(X[c], XL, YROWS[c] - 0.34)
        x_lab.append(k.Text(f"x_{c + 1}", size=0.28, mono=True, x=XL - 0.55,
                            y=YROWS[c], color=CH[c], opacity=0.0))
    span_x = tag_box(XL, D, YROWS[1], CH[0], pad_x=0.16, pad_y=1.05)
    c_x = callout("content tokens x_c — the attention set", span_x, "bottom", CH[0])
    # channel identity
    ZX = -0.6
    z_cells, z_nums, z_lab = [], [], []
    for c in range(C):
        z_cells += vrow(Z[c], ZX, YROWS[c], CH[c])
        z_nums += nrow(Z[c], ZX, YROWS[c] - 0.34)
        z_lab.append(k.Text(f"z_{c + 1}", size=0.28, mono=True, x=ZX - 0.55,
                            y=YROWS[c], color=CH[c], opacity=0.0))
    span_z = tag_box(ZX, D, YROWS[1], IDCOL, pad_x=0.16, pad_y=1.05)
    c_z = callout("channel identity z_c — parallel input, never added to tokens",
                  span_z, "bottom", IDCOL, max_width=3.0)
    # notation glossary
    panel = k.RoundedRect(w=4.5, h=4.2, radius=0.14, x=4.35, y=-0.15,
                          fill=k.GRAY, fill_opacity=0.05, stroke=k.GRAY,
                          stroke_width=1.0, opacity=0.0)
    panel_title = k.Text("notation", size=0.22, color=k.WHITE, x=2.55, y=1.76,
                         opacity=0.0)
    NOTE = [
        "D — token/embedding dim (4 here; 192 real)",
        "H — attention heads (2 here; 6 real)",
        "D_h = D_attn / H — per-head dim (=2)",
        "D_attn — Q/K/V width (3D default; =D here)",
        "C — channels at one location (order-free)",
        "K_f — present (b,n) locations = rows",
        "x_c — content token (patch embedding)",
        "z_c — channel identity; never added to tokens",
        "s_{c,h} = q_h·k_{c,h}/√D_h — the score",
        "b_{c,h} — per-head identity routing bias",
        "α_{c,h} — channel softmax weights, Σ_c α=1",
        "w_c = mean_h α_{c,h} — identity pool weight",
        "θ — one learned query, shared by every location",
        "⊕ — add, the residual junction",
    ]
    note_lines = [
        k.Text(t, size=0.17, color=k.GRAY, x=4.35, y=1.47 - i * 0.265,
               width=4.2, align="left", opacity=0.0)
        for i, t in enumerate(NOTE)
    ]
    s.add(*x_cells, *x_nums, *x_lab, span_x, c_x,
          *z_cells, *z_nums, *z_lab, span_z, c_z,
          panel, panel_title, *note_lines)
    s.play(
        k.stagger([c.to(opacity=1.0) for c in x_cells], lag=0.01),
        *[n.to(opacity=1.0) for n in x_nums],
        *[l.to(opacity=1.0) for l in x_lab],
        span_x.to(opacity=1.0), c_x.to(opacity=1.0),
        duration=0.7,
    )
    s.play(
        k.stagger([c.to(opacity=1.0) for c in z_cells], lag=0.01),
        *[n.to(opacity=1.0) for n in z_nums],
        *[l.to(opacity=1.0) for l in z_lab],
        span_z.to(opacity=1.0), c_z.to(opacity=1.0),
        duration=0.7,
    )
    s.play(panel.to(opacity=1.0), panel_title.to(opacity=1.0),
           k.stagger([n.to(opacity=1.0) for n in note_lines], lag=0.03),
           duration=0.6)
    cap(
        "two parallel inputs per location: patch content x_c and channel identity z_c",
        "x_c, z_c ∈ R^D   c = 1 … C   (C=3, D=4 here; D=192 real)",
    )
    s.wait(0.3)

    # ===================================================================== #
    # 3 -- norm_kv -> separate W_K / W_V projections, split into H heads.
    # ===================================================================== #
    nxt("3 / 14")
    s.play(k.fade_out(*x_cells, *x_nums, *x_lab, span_x, c_x,
                      *z_cells, *z_nums, *z_lab, span_z, c_z,
                      panel, panel_title, *note_lines), duration=0.5)
    XK, XV = -1.5, 1.5
    k_cells, k_nums, v_cells, v_nums = [], [], [], []
    for c in range(C):
        k_cells += vrow(K[c], XK, YROWS[c], CH[c], stroke=k.WHITE)
        k_nums += nrow(K[c], XK, YROWS[c] - 0.34)
        v_cells += vrow(V[c], XV, YROWS[c], CH[c], stroke=k.WHITE)
        v_nums += nrow(V[c], XV, YROWS[c] - 0.34)
    span_k = tag_box(XK, D, YROWS[1], k.WHITE, pad_x=0.16, pad_y=1.05)
    span_v = tag_box(XV, D, YROWS[1], H1COL, pad_x=0.16, pad_y=1.05)
    c_k = callout("keys k = W_K norm_kv(x)", span_k, "left", k.WHITE, max_width=2.4)
    c_v = callout("values v = W_V norm_kv(x)", span_v, "bottom", H1COL, max_width=2.6)
    c_split = callout("split D into H heads: D = H · D_h", span_v, "right", H0COL,
                      style="line", max_width=3.0)
    heads_boxes = []
    for x0 in (XK, XV):
        for c in range(C):
            heads_boxes += [box_around(x0, 0, DH - 1, YROWS[c], H0COL),
                            box_around(x0, DH, D - 1, YROWS[c], H1COL)]
    s.add(*k_cells, *k_nums, *v_cells, *v_nums, span_k, span_v,
          c_k, c_v, c_split, *heads_boxes)
    s.play(
        k.stagger([c.to(opacity=1.0) for c in k_cells], lag=0.03),
        *[n.to(opacity=1.0) for n in k_nums],
        k.stagger([c.to(opacity=1.0) for c in v_cells], lag=0.03),
        *[n.to(opacity=1.0) for n in v_nums],
        span_k.to(opacity=1.0), span_v.to(opacity=1.0),
        c_k.to(opacity=1.0), c_v.to(opacity=1.0),
        duration=0.8,
    )
    s.play(k.stagger([b.to(opacity=1.0) for b in heads_boxes], lag=0.03),
           c_split.to(opacity=1.0), duration=0.6)
    cap(
        "normalize once, project with separate learned W_K and W_V, split into H heads",
        "k_c = W_K norm_kv(x_c)   v_c = W_V norm_kv(x_c)   D = H · D_h",
    )
    s.wait(0.3)

    # ===================================================================== #
    # 4 -- the single learned query, shared by every location.
    # ===================================================================== #
    nxt("4 / 14")
    s.play(k.fade_out(*k_cells, *k_nums, *v_cells, *v_nums, span_k, span_v,
                      c_k, c_v, c_split, *heads_boxes), duration=0.5)
    QX = -2.0
    q_cells = vrow(Q, QX, 0.2, QCOL)
    q_nums = nrow(Q, QX, -0.14)
    q_lab = k.Text("q", size=0.28, mono=True, x=QX - 0.55, y=0.2, color=QCOL,
                   opacity=0.0)
    span_q = tag_box(QX, D, 0.2, QCOL)
    c_q = callout("one learned query θ ∈ R^{1×1×D}, shared by every sample and "
                  "location", span_q, "right", QCOL, max_width=3.2)
    attn_note = k.Text(
        "attn width D_attn = D here for readability; the default is D_attn = 3D",
        size=0.24, color=H0COL, x=0.0, y=-1.5, width=10.0, align="center",
        opacity=0.0)
    s.add(*q_cells, *q_nums, q_lab, span_q, c_q, attn_note)
    s.play(
        k.stagger([c.to(opacity=1.0) for c in q_cells], lag=0.03),
        *[n.to(opacity=1.0) for n in q_nums],
        q_lab.to(opacity=1.0), span_q.to(opacity=1.0), c_q.to(opacity=1.0),
        attn_note.to(opacity=1.0),
        duration=0.7,
    )
    cap(
        "the query is one learned vector, projected by W_Q after norm_q and shared",
        "q = W_Q norm_q(θ) ∈ R^D   (θ = queries: [1, 1, D])",
    )
    s.wait(0.3)

    # ===================================================================== #
    # 5 -- scores s = q_h . k_c,h / sqrt(D_h), shown as a dot product (head 1).
    # ===================================================================== #
    nxt("5 / 14")
    s.play(k.fade_out(*q_cells, *q_nums, q_lab, span_q, c_q, attn_note),
           duration=0.4)
    A1 = k.Matrix([[Q[0], Q[1]]], size=0.38, x=-4.6, y=0.0)
    B1 = k.Matrix([[K[0][0], K[1][0], K[2][0]],
                   [K[0][1], K[1][1], K[2][1]]], size=0.38, x=-0.9, y=0.0)
    P1 = k.Matrix([[0.90, -0.68, 1.14]],
                  size=0.38, x=4.0, y=0.0)
    e_a = k.RoundedRect(w=A1.width + 0.35, h=A1.height + 0.35, radius=0.14,
                        x=-4.6, y=0.0, fill_opacity=0.0, stroke=QCOL,
                        stroke_width=2.4, opacity=0.0)
    e_b = k.RoundedRect(w=B1.width + 0.35, h=B1.height + 0.35, radius=0.14,
                        x=-0.9, y=0.0, fill_opacity=0.0, stroke=H0COL,
                        stroke_width=2.4, opacity=0.0)
    e_p = k.RoundedRect(w=P1.width + 0.35, h=P1.height + 0.35, radius=0.14,
                        x=4.0, y=0.0, fill_opacity=0.0, stroke=H0COL,
                        stroke_width=2.4, opacity=0.0)
    c_a = callout("query, head 1  q_h ∈ R^{D_h}", e_a, "top", QCOL, gap=0.4)
    c_b = callout("keys, head 1  k_c,h  [D_h, C]", e_b, "top", H0COL, gap=0.4)
    c_p = callout("raw q·k (head 1); ÷√D_h → s", e_p, "top", H0COL, gap=0.4,
                  max_width=2.7)
    dot_eq = k.Text("=", size=0.5, mono=True, color=k.WHITE, x=1.55, y=0.0,
                    opacity=0.0)
    s.add(e_a, e_b, e_p, c_a, c_b, c_p, dot_eq)
    s.play(k.draw(A1), k.draw(B1))
    s.play(dot_eq.to(opacity=1.0), k.draw(P1))
    s.play(e_a.to(opacity=1.0), e_b.to(opacity=1.0), e_p.to(opacity=1.0),
           c_a.to(opacity=1.0), c_b.to(opacity=1.0), c_p.to(opacity=1.0),
           duration=0.5)
    cap(
        "each score is the head query dotted with one channel's head key (head 1 shown)",
        "s_{c,h} = (q_h · k_{c,h}) / √D_h       stack heads → s ∈ R^{C×H}",
        H0COL,
    )
    s.wait(0.3)

    # ===================================================================== #
    # 6 -- identity routing bias: z_c -> identity_bias -> b_{c,h}.
    # ===================================================================== #
    nxt("6 / 14")
    s.play(k.fade_out(A1, B1, P1, e_a, e_b, e_p, c_a, c_b, c_p, dot_eq),
           duration=0.4)
    ic_cells, ic_lab = [], []
    for c in range(C):
        ic_cells += vrow(Z[c], XL, YROWS[c], CH[c])
        ic_lab.append(k.Text(f"z_{c + 1}", size=0.28, mono=True, x=XL - 0.55,
                             y=YROWS[c], color=CH[c], opacity=0.0))
    span_z2 = tag_box(XL, D, YROWS[1], IDCOL, pad_x=0.16, pad_y=1.05)
    c_z2 = callout("channel identity z_c ∈ R^{C×D}, never in the content tokens",
                   span_z2, "bottom", IDCOL, max_width=3.2)
    ib_box, ib_lab, ib_sub = opbox("identity_bias", -1.4, 0.0, color=IDCOL,
                                   w=2.2, h=0.9, size=0.22,
                                   sub="Linear(D, H), no bias")
    bias_m = k.Matrix(BIAS, size=0.40, x=1.8, y=0.0)
    e_bias = k.RoundedRect(w=bias_m.width + 0.35, h=bias_m.height + 0.35,
                           radius=0.14, x=1.8, y=0.0, fill_opacity=0.0,
                           stroke=IDCOL, stroke_width=2.4, opacity=0.0)
    arr_ib = [arrow((XL + 2.05, YROWS[c]), (-2.5, 0.0), color=k.GRAY)
              for c in range(C)]
    arr_b = arrow((-0.28, 0.0), (0.25, 0.0), color=IDCOL)
    c_bias = callout("b_{c,h} — per-head routing bias", e_bias,
                     "top", IDCOL, gap=0.4, max_width=2.6)
    s.add(*ic_cells, *ic_lab, span_z2, c_z2, ib_box, ib_lab, ib_sub, bias_m,
          e_bias, *arr_ib, arr_b, c_bias)
    s.play(
        k.stagger([c.to(opacity=1.0) for c in ic_cells], lag=0.02),
        *[l.to(opacity=1.0) for l in ic_lab],
        span_z2.to(opacity=1.0), c_z2.to(opacity=1.0),
        ib_box.to(opacity=1.0), ib_lab.to(opacity=1.0), ib_sub.to(opacity=1.0),
        k.stagger([a.to(opacity=1.0) for a in arr_ib], lag=0.05),
        arr_b.to(opacity=1.0), duration=0.8,
    )
    s.play(k.draw(bias_m), e_bias.to(opacity=1.0), c_bias.to(opacity=1.0),
           duration=0.5)
    cap(
        "channel identity is projected to one bias per head; it is not added to the tokens",
        "b_{c,h} = identity_bias(z_c) = W_b z_c ∈ R^{C×H}",
        IDCOL,
    )
    s.wait(0.3)

    # ===================================================================== #
    # 7 -- the softmax input: logits = s + b.
    # ===================================================================== #
    nxt("7 / 14")
    s.play(k.fade_out(*ic_cells, *ic_lab, span_z2, c_z2, ib_box, ib_lab, ib_sub,
                      bias_m, e_bias, *arr_ib, arr_b, c_bias), duration=0.4)
    s_m = k.Matrix(SCORES, size=0.40, x=-4.3, y=0.0)
    b_m = k.Matrix(BIAS, size=0.40, x=0.2, y=0.0)
    plus1 = k.Text("+", size=0.5, mono=True, color=k.WHITE, x=-2.05, y=0.0,
                   opacity=0.0)
    eq1 = k.Text("=", size=0.5, mono=True, color=k.WHITE, x=2.3, y=0.0,
                 opacity=0.0)
    l_m = k.Matrix(LOGITS, size=0.40, x=4.4, y=0.0)
    e_s = k.RoundedRect(w=s_m.width + 0.35, h=s_m.height + 0.35, radius=0.14,
                        x=-4.3, y=0.0, fill_opacity=0.0, stroke=H0COL,
                        stroke_width=2.4, opacity=0.0)
    e_bm = k.RoundedRect(w=b_m.width + 0.35, h=b_m.height + 0.35, radius=0.14,
                         x=0.2, y=0.0, fill_opacity=0.0, stroke=IDCOL,
                         stroke_width=2.4, opacity=0.0)
    e_l = k.RoundedRect(w=l_m.width + 0.35, h=l_m.height + 0.35, radius=0.14,
                        x=4.4, y=0.0, fill_opacity=0.0, stroke=H1COL,
                        stroke_width=2.8, opacity=0.0)
    c_s = callout("scores s_{c,h}", e_s, "top", H0COL, gap=0.4)
    c_bm = callout("identity bias b_{c,h}", e_bm, "top", IDCOL, gap=0.4)
    c_l = callout("logits = softmax input", e_l, "top", H1COL, gap=0.4)
    s.add(s_m, b_m, plus1, eq1, l_m, e_s, e_bm, e_l, c_s, c_bm, c_l)
    s.play(k.draw(s_m), k.draw(b_m))
    s.play(plus1.to(opacity=1.0), eq1.to(opacity=1.0), k.draw(l_m))
    s.play(e_s.to(opacity=1.0), e_bm.to(opacity=1.0), e_l.to(opacity=1.0),
           c_s.to(opacity=1.0), c_bm.to(opacity=1.0), c_l.to(opacity=1.0),
           duration=0.5)
    cap(
        "the bias is added BEFORE the softmax: the softmax sees s + b, not s alone",
        "logits_{c,h} = s_{c,h} + b_{c,h}",
        H1COL,
    )
    s.wait(0.3)

    # ===================================================================== #
    # 8 -- softmax over the C channels, per head -> alpha.
    # ===================================================================== #
    nxt("8 / 14")
    s.play(k.fade_out(s_m, b_m, plus1, eq1, e_s, e_bm, c_s, c_bm,
                      l_m, e_l, c_l), duration=0.4)
    l_m2 = k.Matrix(LOGITS, size=0.40, x=-3.2, y=0.0)
    e_l2 = k.RoundedRect(w=l_m2.width + 0.35, h=l_m2.height + 0.35, radius=0.14,
                         x=-3.2, y=0.0, fill_opacity=0.0, stroke=H1COL,
                         stroke_width=2.4, opacity=0.0)
    soft = k.Text("softmax_c  →", size=0.30, mono=True, color=k.WHITE, x=-0.3,
                  y=0.0, opacity=0.0)
    alpha_m = k.Matrix(ALPHA, size=0.44, x=2.6, y=0.0)
    e_alpha = k.RoundedRect(w=alpha_m.width + 0.35, h=alpha_m.height + 0.35,
                            radius=0.14, x=2.6, y=0.0, fill_opacity=0.0,
                            stroke=H0COL, stroke_width=2.6, opacity=0.0)
    c_l2 = callout("softmax input = logits", e_l2, "top", H1COL, gap=0.35)
    c_alpha = callout("α_{c,h} = channel softmax weights",
                      e_alpha, "top", H0COL, gap=0.35, max_width=2.8)
    s.add(l_m2, e_l2, soft, alpha_m, e_alpha, c_l2, c_alpha)
    s.play(k.draw(l_m2), e_l2.to(opacity=1.0), c_l2.to(opacity=1.0))
    s.play(soft.to(opacity=1.0), k.draw(alpha_m), e_alpha.to(opacity=1.0),
           c_alpha.to(opacity=1.0), duration=0.6)
    cap(
        "softmax runs over the channel axis (per head); the bias changed the routing",
        "α_{c,h} = softmax_c(s_{c,h} + b_{c,h})        Σ_c α_{c,h} = 1",
    )
    s.wait(0.3)

    # ===================================================================== #
    # 9 -- weighted sum of values -> head outputs.
    # ===================================================================== #
    nxt("9 / 14")
    s.play(k.fade_out(l_m2, e_l2, soft, alpha_m, e_alpha, c_l2, c_alpha),
           duration=0.4)
    a_h0 = k.Matrix([[ALPHA[0][0], ALPHA[1][0], ALPHA[2][0]]], size=0.40,
                    x=-4.4, y=0.0)
    Vt0 = k.Matrix([[V[0][0], V[0][1]], [V[1][0], V[1][1]], [V[2][0], V[2][1]]],
                   size=0.40, x=-0.8, y=0.0)
    o0 = a_h0 @ Vt0
    o0.set(x=3.0, y=0.0)
    e_w = k.RoundedRect(w=a_h0.width + 0.35, h=a_h0.height + 0.35, radius=0.14,
                        x=-4.4, y=0.0, fill_opacity=0.0, stroke=H0COL,
                        stroke_width=2.4, opacity=0.0)
    e_v = k.RoundedRect(w=Vt0.width + 0.35, h=Vt0.height + 0.35, radius=0.14,
                        x=-0.8, y=0.0, fill_opacity=0.0, stroke=H1COL,
                        stroke_width=2.4, opacity=0.0)
    e_o = k.RoundedRect(w=o0.width + 0.35, h=o0.height + 0.35, radius=0.14,
                        x=3.0, y=0.0, fill_opacity=0.0, stroke=H0COL,
                        stroke_width=2.6, opacity=0.0)
    c_w = callout("weights α_h (head 1)", e_w, "top", H0COL, gap=0.4)
    c_v = callout("values V_h (head 1)", e_v, "top", H1COL, gap=0.4,
                  max_width=2.4)
    c_o = callout("head output o_h", e_o, "top", H0COL, gap=0.4)
    s.add(e_w, e_v, e_o, c_w, c_v, c_o)
    s.play(k.draw(a_h0), k.draw(Vt0))
    s.play(k.matrix_product(a_h0, Vt0, o0))
    s.play(e_w.to(opacity=1.0), e_v.to(opacity=1.0), e_o.to(opacity=1.0),
           c_w.to(opacity=1.0), c_v.to(opacity=1.0), c_o.to(opacity=1.0),
           duration=0.5)
    cap(
        "each head output is the attention-weighted average of its channel values",
        "o_h = Σ_c α_{c,h} · v_{c,h}        (head 1 shown)",
        H0COL,
    )
    s.wait(0.3)

    # ===================================================================== #
    # 10 -- concat heads + proj_out -> content path.
    # ===================================================================== #
    nxt("10 / 14")
    s.play(k.fade_out(a_h0, Vt0, o0, e_w, e_v, e_o, c_w, c_v, c_o), duration=0.4)
    Wm = k.Matrix(W, size=0.32, x=-3.9, y=-0.1)
    cm = k.Matrix([[v] for v in CONCAT], size=0.32, x=-0.2, y=-0.1)
    pm = Wm @ cm
    pm.set(x=3.2, y=-0.1)
    e_W = k.RoundedRect(w=Wm.width + 0.35, h=Wm.height + 0.35, radius=0.14,
                        x=-3.9, y=-0.1, fill_opacity=0.0, stroke=k.WHITE,
                        stroke_width=2.2, opacity=0.0)
    e_c = k.RoundedRect(w=cm.width + 0.35, h=cm.height + 0.35, radius=0.14,
                        x=-0.2, y=-0.1, fill_opacity=0.0, stroke=k.PURPLE,
                        stroke_width=2.4, opacity=0.0)
    e_p10 = k.RoundedRect(w=pm.width + 0.35, h=pm.height + 0.35, radius=0.14,
                          x=3.2, y=-0.1, fill_opacity=0.0, stroke=FUSEDCOL,
                          stroke_width=2.6, opacity=0.0)
    c_W = callout("W, b from proj_out", e_W, "top", k.WHITE,
                  gap=0.4, max_width=2.6)
    c_c = callout("concat heads  c = [o_1; …; o_H]", e_c, "top", k.PURPLE,
                  gap=0.4, max_width=2.8)
    c_p10 = callout("proj_out(concat)+b = content path", e_p10, "top", FUSEDCOL,
                    gap=0.4, max_width=2.8)
    s.add(e_W, e_c, e_p10, c_W, c_c, c_p10)
    s.play(k.draw(Wm), k.draw(cm))
    s.play(k.matrix_product(Wm, cm, pm))
    s.play(e_W.to(opacity=1.0), e_c.to(opacity=1.0), e_p10.to(opacity=1.0),
           c_W.to(opacity=1.0), c_c.to(opacity=1.0), c_p10.to(opacity=1.0),
           duration=0.5)
    cap(
        "concatenate the heads, then one shared Linear projection (content path only)",
        "y_content = W · concat_h(o_h) + b_out        y_content ∈ R^D",
        FUSEDCOL,
    )
    s.wait(0.3)

    # ===================================================================== #
    # 11 -- identity pool + identity_skip.
    # ===================================================================== #
    nxt("11 / 14")
    s.play(k.fade_out(Wm, cm, pm, e_W, e_c, e_p10, c_W, c_c, c_p10), duration=0.4)
    z2_cells, z2_lab = [], []
    for c in range(C):
        z2_cells += vrow(Z[c], XL, YROWS[c], CH[c])
        z2_lab.append(k.Text(f"z_{c + 1}", size=0.28, mono=True, x=XL - 0.55,
                             y=YROWS[c], color=CH[c], opacity=0.0))
    pool_box, pool_lab, pool_sub = opbox("Σ_c w_c z_c", -1.9, 0.78, color=IDCOL,
                                         w=2.2, h=0.72, size=0.22,
                                         sub="identity pool")
    skip_box, skip_lab, skip_sub = opbox("identity_skip", 0.6, 0.78, color=IDCOL,
                                         w=2.0, h=0.72, size=0.20,
                                         sub="Linear(D, D)")
    skip_row = k.Matrix([SKIP], size=0.30, x=4.2, y=0.78)
    e_skiprow = k.RoundedRect(w=skip_row.width + 0.30, h=skip_row.height + 0.30,
                              radius=0.14, x=4.2, y=0.78, fill_opacity=0.0,
                              stroke=IDCOL, stroke_width=2.6, opacity=0.0)
    c_pool = callout("w_c = mean_h α_{c,h}", pool_box, "top", H0COL, gap=0.4,
                     max_width=2.6)
    c_skip = callout("identity_skip → skip", e_skiprow, "top",
                     IDCOL, gap=0.35, max_width=2.4)
    alpha_m2 = k.Matrix(ALPHA, size=0.38, x=-1.9, y=-1.05)
    e_alpha2 = k.RoundedRect(w=alpha_m2.width + 0.35, h=alpha_m2.height + 0.35,
                             radius=0.14, x=-1.9, y=-1.05, fill_opacity=0.0,
                             stroke=H0COL, stroke_width=2.4, opacity=0.0)
    arr_pool = [arrow((XL + 2.05, YROWS[c]), (-3.05, 0.78), color=k.GRAY)
                for c in range(C)]
    arr_pskip = arrow((-0.75, 0.78), (-0.4, 0.78), color=IDCOL)
    arr_skip = arrow((1.6, 0.78), (2.1, 0.78), color=IDCOL)
    arr_up = arrow((-1.9, 0.15), (-1.9, 0.42), color=H0COL)
    s.add(*z2_cells, *z2_lab, pool_box, pool_lab, pool_sub,
          skip_box, skip_lab, skip_sub, skip_row, e_skiprow, c_pool, c_skip,
          alpha_m2, e_alpha2, *arr_pool, arr_pskip, arr_skip, arr_up)
    s.play(k.stagger([c.to(opacity=1.0) for c in z2_cells], lag=0.02),
           *[l.to(opacity=1.0) for l in z2_lab],
           k.stagger([a.to(opacity=1.0) for a in arr_pool], lag=0.05),
           duration=0.7)
    s.play(k.draw(alpha_m2), e_alpha2.to(opacity=1.0), arr_up.to(opacity=1.0),
           pool_box.to(opacity=1.0), pool_lab.to(opacity=1.0),
           pool_sub.to(opacity=1.0), c_pool.to(opacity=1.0), duration=0.6)
    s.play(arr_pskip.to(opacity=1.0), skip_box.to(opacity=1.0),
           skip_lab.to(opacity=1.0), skip_sub.to(opacity=1.0), duration=0.4)
    s.play(arr_skip.to(opacity=1.0), k.draw(skip_row), e_skiprow.to(opacity=1.0),
           c_skip.to(opacity=1.0), duration=0.5)
    cap(
        "identity is also pooled with the same attention weights and passed through a skip",
        "w_c = mean_h α_{c,h};   skip = identity_skip(Σ_c w_c z_c)",
        IDCOL,
    )
    s.wait(0.3)

    # ===================================================================== #
    # 12 -- add junction: fused y = content + identity skip.
    # ===================================================================== #
    nxt("12 / 14")
    s.play(k.fade_out(*z2_cells, *z2_lab, pool_box, pool_lab,
                      pool_sub, skip_box, skip_lab, skip_sub, skip_row, e_skiprow,
                      c_pool, c_skip, alpha_m2, e_alpha2, *arr_pool, arr_pskip,
                      arr_skip, arr_up), duration=0.4)
    yc_row = k.Matrix([Y_CONTENT], size=0.30, x=-3.6, y=0.9)
    sk_row = k.Matrix([SKIP], size=0.30, x=-3.6, y=0.0)
    y_row = k.Matrix([FUSED], size=0.30, x=2.6, y=0.45)
    e_yc = k.RoundedRect(w=yc_row.width + 0.30, h=yc_row.height + 0.30,
                         radius=0.14, x=-3.6, y=0.9, fill_opacity=0.0,
                         stroke=FUSEDCOL, stroke_width=2.4, opacity=0.0)
    e_sk = k.RoundedRect(w=sk_row.width + 0.30, h=sk_row.height + 0.30,
                         radius=0.14, x=-3.6, y=0.0, fill_opacity=0.0,
                         stroke=IDCOL, stroke_width=2.4, opacity=0.0)
    e_y = k.RoundedRect(w=y_row.width + 0.35, h=y_row.height + 0.35, radius=0.14,
                        x=2.6, y=0.45, fill_opacity=0.0, stroke=FUSEDCOL,
                        stroke_width=2.8, opacity=0.0)
    plus2 = k.Text("⊕", size=0.55, mono=True, color=k.WHITE, x=-0.5, y=0.45,
                   opacity=0.0)
    a_yc = arrow((-1.55, 0.9), (-0.95, 0.62), color=FUSEDCOL)
    a_sk = arrow((-1.55, 0.0), (-0.95, 0.28), color=IDCOL)
    a_out = arrow((-0.05, 0.45), (0.55, 0.45), color=FUSEDCOL)
    c_yc = callout("content  y_content = proj_out(concat)", e_yc, "top",
                   FUSEDCOL, gap=0.4, max_width=3.0)
    c_sk = callout("identity skip", e_sk, "bottom", IDCOL, gap=0.4)
    c_y = callout("fused y = y_content + skip  →  one [D] row", e_y, "top",
                  FUSEDCOL, gap=0.4, max_width=3.2)
    s.add(yc_row, sk_row, y_row, e_yc, e_sk, e_y, plus2, a_yc, a_sk, a_out,
          c_yc, c_sk, c_y)
    s.play(k.draw(yc_row), k.draw(sk_row), e_yc.to(opacity=1.0),
           e_sk.to(opacity=1.0), c_yc.to(opacity=1.0), c_sk.to(opacity=1.0),
           a_yc.to(opacity=1.0), a_sk.to(opacity=1.0), duration=0.6)
    s.play(plus2.to(opacity=1.0), a_out.to(opacity=1.0), k.draw(y_row),
           e_y.to(opacity=1.0), c_y.to(opacity=1.0), duration=0.6)
    cap(
        "the identity skip is added AFTER proj_out, at this junction — not inside the softmax",
        "y = proj_out(concat_h o_h) + identity_skip(Σ_c w_c z_c),   y ∈ R^D",
        FUSEDCOL,
    )
    s.wait(0.3)

    # ===================================================================== #
    # 13 -- one row per present location; no post-fusion spatial PE (default).
    # ===================================================================== #
    nxt("13 / 14")
    s.play(k.fade_out(yc_row, sk_row, y_row, e_yc, e_sk, e_y, plus2, a_yc, a_sk,
                      a_out, c_yc, c_sk, c_y), duration=0.4)
    out_grid, out_fill = [], []
    for si in range(2):
        for pj in range(4):
            x = -3.9 + pj * 2.6
            y = 0.7 - si * 1.9
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
    out_note = k.Text(
        "no post-fusion spatial PE by default: the encoder owns layout via 2-D axial RoPE",
        size=0.26, color=H0COL, x=0.0, y=-2.1, width=12.0, align="center",
        opacity=0.0)
    c_present = callout("present location → one fused [D] row", out_fill[1], "top",
                        FUSEDCOL, gap=0.5, max_width=3.0)
    c_absent = callout("absent → no row", out_grid[3], "right", k.GRAY,
                       max_width=2.0, gap=0.35)
    s.add(*out_grid, *out_fill, out_note, c_present, c_absent)
    s.play(
        k.stagger([g.to(opacity=1.0) for g in out_grid], lag=0.04),
        k.stagger([f.to(opacity=1.0) for f in out_fill], lag=0.04),
        out_note.to(opacity=1.0),
        c_present.to(opacity=1.0), c_absent.to(opacity=1.0),
        duration=0.7,
    )
    cap(
        "one [D] row per present location; absent locations own no row",
        "[ΣC·N, D]  →  [K_f, D]     K_f = present (b,n) locations;  present: [B, N]",
        FUSEDCOL,
    )
    s.wait(0.3)

    # ===================================================================== #
    # 14 -- summary of the default path.
    # ===================================================================== #
    nxt("14 / 14")
    s.play(k.fade_out(*out_grid, *out_fill, out_note, c_present, c_absent),
           duration=0.4)
    row1 = [("gather", k.BLUE), ("norm_kv", k.WHITE), ("Q/K/V proj", k.WHITE),
            ("q·kᵀ/√D_h", k.TEAL), ("+b (identity)", IDCOL),
            ("softmax_c", k.YELLOW)]
    row2 = [("Σ α·v", k.ORANGE), ("concat heads", k.PURPLE),
            ("proj_out", k.PINK), ("Σ w_c z_c", IDCOL),
            ("identity_skip", IDCOL), ("⊕ → [K_f,D]", FUSEDCOL)]
    pills, arow = [], []
    for row_i, ops in enumerate((row1, row2)):
        y = 0.75 - row_i * 1.05
        for i, (txt, col) in enumerate(ops):
            x = -5.0 + i * 2.0
            pills.append(
                k.RoundedRect(w=1.62, h=0.72, radius=0.14, x=x, y=y,
                              fill=col, fill_opacity=0.20, stroke=col,
                              stroke_width=1.7, opacity=0.0)
            )
            pills.append(
                k.Text(txt, size=0.17, mono=True, color=k.WHITE, x=x, y=y,
                       opacity=0.0)
            )
            if i < len(ops) - 1:
                arow.append(arrow((x + 0.83, y), (x + 1.15, y)))
    s.add(*pills, *arow)
    s.play(k.stagger([p.to(opacity=1.0) for p in pills], lag=0.04),
           k.stagger([a.to(opacity=1.0) for a in arow], lag=0.04), duration=0.8)
    closing = k.Text(
        "O(C) per location · no C_max · no padding · identity = bias + skip\n"
        "no scatter (rows are already in grouped order) · no post-fusion spatial PE",
        size=0.28, color=k.GREEN, x=0.0, y=-1.55, width=12.5, align="center")
    s.play(k.write(closing))
    cap(
        "variable channel count in, fixed-size fused rows out",
        "[C_b, 1, H, W] → [ΣC·N, D] → [K_f, D]        (default path, no spatial PE)",
        k.GREEN,
    )
    s.wait(1.0)
