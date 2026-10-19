# Visualizations

Two kinds of artifact, because they answer different questions.

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

## Why this combination

A tracer (torchview, torch.fx, torch.export) shows ops and tensor shapes.
The dashboard shows *invariants* — padding-free equivalence, permutation
invariance, omission correctness, identity-aware routing, masked-channel
imputation. Together they answer both "is it wired the way I sketched?" and
"does it actually work as a set function?".
