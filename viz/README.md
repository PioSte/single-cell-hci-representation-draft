# Visualizations

Four kinds of artifact, because they answer different questions.

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

## Why this combination

A tracer (torchview, torch.fx, torch.export) shows ops and tensor shapes.
The dashboard shows *invariants* — padding-free equivalence, permutation
invariance, omission correctness, identity-aware routing, masked-channel
imputation. The paper figure shows the *architecture* at module level.
Together they answer "is it wired the way I sketched?", "does it actually work
as a set function?" and "what do I put in the manuscript?".
