# Visualizations

Six kinds of artifact, because they answer different questions.

## 1. Interactive behaviour — `dashboard.py` (marimo)

An architecture diagram proves the *shape* of the code; it cannot prove that the
module behaves like a set function. The marimo dashboard runs the **real**
modules and recomputes every number when you move a control:

| control | demonstrates |
| --- | --- |
| channel count `C` (1–60) | cardinality invariance — output shape never changes |
| permute channel order | permutation invariance (exact) |
| use channel identity | the additive identity embedding changes routing |
| channel index to omit | omission correctness vs. the masked-padded reference |
| MAE training steps | channel-masked reconstruction (virtual staining) |

Panels: a live measurements table, a per-location **attention heatmap over
channels**, a **fixed-size descriptor vs. C** curve, and a **masked-channel
reconstruction** grid.

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
  store).
* `padding_free_flow.png/.svg` — a hand-drawn schematic of the set/varlen data
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
*modules* and make the set/varlen semantics explicit — that is exactly what
`architecture_torchview` cannot express.

## 4. Cardinality invariance — `cardinality_figure.py`

A step-by-step, spatial explanation of *why* the output does not depend on the
number of channels. Written to `viz/out/cardinality.{pdf,svg,png}`.

* **A** — a filmstrip of consecutive steps for one sample: input channel set →
  tied embed + identity (`C×N` token grid) → per-location channel fusion (the
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
padded to a `C_max`. Two simplifications: the DCP masking is drawn as
location-level removal, whereas the real masker selects per channel-patch token
*before* the fusion; and `[B, D]` is the inference-time `encode()` descriptor,
which the training forward does not materialise as an intermediate.

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

A deeper animation of the **default** fusion path (`ChannelFusion`,
`mode="per_location"`). It zooms into one `(sample, patch)` location and follows
its channel tokens through every operation, one at a time:

`x_c` → `LayerNorm` (`k = v`) → per-head `q_h · k_{c,h} / √D_h` → `softmax` over
the `C` channels → `Σ_c α_c v_c` → `concat` heads → `proj_out` Linear →
scatter into `[B, N, Q·D]`.

The two dot-product steps are drawn with `k.matrix_product` (each score/entry is
shown cell by cell while its query row and key/value column are highlighted), and
every element group at each step is **named, highlighted and pointed at with a
callout** — the channel block, the keys/values, the query, each head, the query /
keys / scores, the attention weights, the values, the head output, the projection,
the concat and the fused vector, plus present vs. absent locations. The numbers
come from a real forward pass at `dim=4, n_heads=2` — small enough to read; the
model default is `dim=192, n_heads=6`. It makes three structural points explicit:
keys and values are the *same* tensor (`k = v = norm_kv(x)`), there is no
`W_Q/W_K/W_V` (the query is a learned per-patch parameter, so it is shared across
samples), and the softmax axis is the channel axis — never patches or samples.

```bash
uv run kinemo check --strict viz/fusion_flow.py
uv run kinemo render viz/fusion_flow.py --format mp4 --out viz/out/fusion_flow.mp4
uv run kinemo render viz/fusion_flow.py --format gif --size 720p --out viz/out/fusion_flow.gif
```

## Why this combination

A tracer (torchview, torch.fx, torch.export) shows ops and tensor shapes.
The dashboard shows *invariants* — padding-free equivalence, permutation
invariance, omission correctness, identity-aware routing, masked-channel
imputation. The paper figure shows the *architecture* at module level.
Together they answer "is it wired the way I sketched?", "does it actually work
as a set function?" and "what do I put in the manuscript?".
