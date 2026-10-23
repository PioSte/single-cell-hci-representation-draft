# Visualizations

Several kinds of artifact, because they answer different questions.

## 0. The level ladder — L0–L8

A coherent set of hand-authored, vector-first figures (PDF/SVG/PNG) that walk
from *where the model sits in the literature* down to *the attention primitive*.
They share one design system so they read as one authored set:

* `figure_style.py` — the palette (Okabe-Ito), module-box / arrow / array-stack
  primitives, the channel colour table, and the `save_figure` helper. It also
  exposes a shared **role palette** so the same architecture role always gets
  the same colour in every level (`FRONTEND` green, `FUSION` orange, `ENCODER`
  pink, `DECODER` violet, `MEMORY` gold, `LOSS` red, `FUSED` blue, `POOL` sky,
  `GHOST` muted), plus `legend` / `role_legend` / `channel_legend` and
  `padded_ghost` helpers. Per-channel colours stay in `CHANNEL_COLORS` and are
  used only for the channel axis (image tiles and channel-patch token rows).
* `figure_facts.md` — the authoritative tensor shapes and semantics each figure
  must draw, read from `src/fuse_channels/`. Every figure is built from this.

| level | file | what it shows | reference idiom |
| --- | --- | --- | --- |
| **L0** | `level0_positioning.py` | padded dense ViT · dense cross-channel attention · ours | ChAda-ViT Fig 13 |
| **L1** | `level1_system.py` | end-to-end system: training path + `encode()` branch | Panopticon Fig 1 / Perceiver-IO Fig 2 |
| **L2** | `level2_layout.py` | axis-labelled tensor layout (the hero figure) | Perceiver-IO Fig 2 |
| **L3** | `level3_frontend.py` | tied patch embed + additive channel embedding + post-fusion PE | ChannelViT Fig 1 / ChAda-ViT Fig 2 |
| **L4** | `level4_fusion.py` | fusion internals (single query, projection-free K/V) + `IdentityGatedSum` channel-routed `O(C)` ablation | Perceiver-IO Fig 5 inset |
| **L5** | `level5_masking.py` | DCP Eq. 1 (patch / channel / combined) + unified HCS channel sampler; token omission | ChAda-ViT Fig 2 MASK row |
| **L6** | `level6_objective.py` | MAE objective: trained content-only descriptor, target self-attention, default/ablation losses | Panopticon teacher–student |
| **L7** | `level7_invariance.py` | measured invariance & cost (runs the real model) | ChAda-ViT scaling panels |
| **L8** | `level8_segment_ops.py` | the padding-free grouped-attention primitive | Perceiver-IO Fig 5 |

Run any level (writes into the git-ignored `viz/out/`):

```bash
uv run --extra viz python viz/level2_layout.py                 # pdf + svg + png
uv run --extra viz python viz/level7_invariance.py             # measures the real model
uv run --extra viz python viz/level4_fusion.py --formats pdf svg
```

Two visual rules carry the paper's thesis across every level: the ragged token
list `[ΣC·N, D]` is always the solid primary object, and the padded tensor
`[B, C_max, N, D]` only ever appears as a dashed **ghost** marked
*never materialised*.

## 1. Interactive behaviour — `dashboard.py` (marimo)

An architecture diagram proves the *shape* of the code; it cannot prove that the
module behaves like a set function. The marimo dashboard runs the **real**
modules and recomputes every number when you move a control:

| control | demonstrates |
| --- | --- |
| channel count `C` (1–60) | cardinality invariance — output shape never changes |
| permute channel order | permutation invariance (exact) |
| use channel embedding | the additive `z_c` changes routing |
| channel index to omit | omission correctness vs. the masked-padded reference |
| MAE training steps | channel-masked reconstruction (virtual staining) |

Panels: a live measurements table, a per-location **attention heatmap over
channels**, a **fused-token mean vs. C** curve, and a **masked-channel
reconstruction** grid. The curve is the mean of the fused `[B,N,D]` tokens — a
shape proxy, **not** the trained `[B,D]` descriptor returned by
`encode(pool="attention")`.

```bash
# interactive (opens a browser)
uv run marimo edit viz/dashboard.py
uv run marimo run  viz/dashboard.py

# self-contained HTML snapshot (interactive via WASM), written to viz/out/
uv run marimo export html viz/dashboard.py -o viz/out/dashboard.html
```

## 2. Structure — `draw_architecture.py`

```bash
uv run python viz/draw_architecture.py
```

Writes into `viz/out/`:

* `architecture_torchview.png/.svg` — the `ChannelFusionMAE` graph traced from a
  forward pass with [torchview](https://github.com/mert-kurttutan/torchview).
  Needs the Graphviz `dot` binary (the script finds it on `PATH` or in a Nix
  store). Torchview tracing is best-effort: with some torchview/torch pairs it
  raises a `KeyError`; the script warns and still writes the schematic below.
* `padding_free_flow.png/.svg` — a hand-drawn schematic of the set-native data
  flow: the flattened `[ΣC·N, D]` token layout, the per-`(sample, patch)`
  attention group, and the "masking = token omission" rule. Generic graph
  tracers cannot express this.

## 3. Paper figure — `paper_figure.py`

The hand-authored, submission-quality figure in `viz/out/architecture.{pdf,svg,png}`.
It is vector-first (PDF/SVG for the paper, PNG for preview), uses a
colour-blind-safe Okabe-Ito palette, and annotates every module with its tensor
shape. Two panels:

* **A** — the end-to-end padding-free channel-fusion front-end.
* **B** — the set-native token layout (ragged grid, `(b,n)` attention groups,
  an omitted row, the padded tensor that is never materialised) plus the
  interchangeable fusion options and the DCP masking rule.

```bash
uv run python viz/paper_figure.py                    # pdf + svg + png
uv run python viz/paper_figure.py --formats pdf svg  # subset
```

Why not a tracer for this one? A tracer shows ops; a paper figure must show
*modules* and make the set-native semantics explicit — that is exactly what
`architecture_torchview` cannot express.

## 4. Cardinality invariance — `cardinality_figure.py`

A step-by-step, spatial explanation of *why* the output does not depend on the
number of channels. Written to `viz/out/cardinality.{pdf,svg,png}`.

* **A** — a filmstrip of consecutive steps for one sample: input channel set →
  tied embed + additive `z_c` (`C×N` token grid) → per-location channel fusion (the
  `C` tokens at each `(b,n)` collapse to one) → fixed `[B,N,D]` grid, memory and
  descriptor. The variable-height channel axis is the thing you watch vanish.
* **B** — the invariance itself: `C=1`, `C=3`, `C=5` token stacks (input length
  `ΣC·N`) all funnel through the *same* fusion and produce the *same*
  `[B,N,D] → [B,D]` output.

```bash
uv run python viz/cardinality_figure.py
```

## 5. Shape flow (animated) — `shape_flow.py` (kinemo)

A [kinemo](https://github.com/m4nd0r4s/kinemo) animation of the **tensor-shape
transformations** from the raw image set to the final output, walking the shapes
of `ChannelFusionMAE`. A grid of cells is a literal picture of the tensor
axes (rows = channels grouped by sample, columns = spatial patches, `D` = depth)
and the channel axis is the thing you watch collapse:

`[C_b, 1, H, W]` → `[ΣC, 1, H, W]` → `[ΣC, N, D]` → `[B, N, D]` → `[B, D]` →
`[Kt, P²]`, with the DCP masking `[B, N, D] → [Kf, D]` and the memory
`[B·M + Kf, D]` in between. The point the animation makes visually: `C` appears
only in the input length and is **summed away** by the channel fusion, never
padded to a `C_max`. One simplification: the DCP masking is drawn as
location-level removal, whereas the real masker selects per channel-patch token
*before* the fusion. `[B, D]` is the **trained** descriptor: the training
forward pools `tok_out` (content tokens only) and appends it to the decoder KV;
`encode(pool="attention")` returns the same vector.

```bash
# check, preview in the browser, or render (MP4 / GIF / frames)
uv run kinemo check --strict viz/shape_flow.py
uv run kinemo dev viz/shape_flow.py
uv run kinemo render viz/shape_flow.py --format mp4 --out viz/out/shape_flow.mp4
uv run kinemo render viz/shape_flow.py --format gif --size 720p --out viz/out/shape_flow.gif
```

This needs the `kinemo` package (in the `viz` extra). The scene is written for
agents: `kinemo check --strict` passes, so `kinemo snap --at <t>` can be used to
inspect any instant without watching the video.

## 6. Fusion internals (animated) — `fusion_flow.py` (kinemo)

A deeper animation of the **default** fusion path (`ChannelFusion`). It zooms
into one `(sample, patch)` location and follows
its channel tokens through every operation, one at a time:

`x_c` → `LayerNorm` (`k = v`) → per-head `q_h · k_{c,h} / √D_h` → `softmax` over
the `C` channels → `Σ_c α_c v_c` → `concat` heads → `proj_out` Linear →
scatter into `[B, N, D]` → add the post-fusion spatial PE. `ChannelFusion` is
single-query (one learned `[1,1,D]` query for every location) and projection-free
for K/V; the only learned linear is `proj_out`.

The two dot-product steps are drawn with `k.matrix_product` (each score/entry is
shown cell by cell while its query row and key/value column are highlighted), and
every element group at each step is **named, highlighted and pointed at with a
callout** — the channel block, the keys/values, the query, each head, the query /
keys / scores, the attention weights, the values, the head output, the projection,
the concat and the fused vector, plus present vs. absent locations. The numbers
come from a real forward pass at `dim=4, n_heads=2` — small enough to read; the
model default is `dim=192, n_heads=6`. It makes three structural points explicit:
keys and values are the *same* tensor (`k = v = norm_kv(x)`), there is no
`W_Q/W_K/W_V` (the query is one learned parameter vector shared by every sample
and every patch, so `ChannelFusion` has no `n_queries`), and the softmax axis is
the channel axis — never patches or samples.

```bash
uv run kinemo check --strict viz/fusion_flow.py
uv run kinemo render viz/fusion_flow.py --format mp4 --out viz/out/fusion_flow.mp4
uv run kinemo render viz/fusion_flow.py --format gif --size 720p --out viz/out/fusion_flow.gif
```

## Why this combination

A tracer (torchview, torch.fx, torch.export) shows ops and tensor shapes.
The dashboard shows *invariants* — padding-free equivalence, permutation
invariance, omission correctness, channel-embedding-aware routing, masked-channel
imputation. The paper figure shows the *architecture* at module level.
Together they answer "is it wired the way I sketched?", "does it actually work
as a set function?" and "what do I put in the manuscript?".
