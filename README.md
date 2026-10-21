# fuse-channels

A PyTorch implementation of the **padding-free channel-fusion front-end** for a
high-content-imaging (HCI) foundation model, plus a channel-aware Masked
Autoencoder (MAE) built on top of it.

It implements the design in the handoff message *"Padding-Free Channel-Fusion
Module for an HCI Foundation Model"* and its MAE follow-up.

## What problem it solves

An HCI image is a set of per-channel embeddings `{E_c ∈ R^D}` where the channel
count `C` varies from 1 to 20–60+, channel order is meaningless, and panels
differ across labs. We need

```
f({E_1 .. E_C}) -> R^{k x D},   output size independent of C,
```

without concatenation blow-up, with channel identity preserved, data-dependent
weighting, and **no padding tokens**.

## Design (settled in the handoff)

```
per-channel (tied) patch embed
    + additive channel-identity embedding (learnable)
    + shared spatial positional embedding
    -> padding-free channel-axis attention pooling (PMA) to a fixed bottleneck
    -> memory-augmented spatial transformer
    -> (MAE) channel-aware shared decoder conditioned on channel identity
```

* **Channel-tied projection** — one shared filter across channels (ChannelViT).
* **Additive channel identity** — learnable per-marker ids for a closed panel.
  Never normalized across the channel axis.
* **Padding-free** — tokens are flattened `[ΣC, D]` with segment ids; no
  `[B, C_max, D]` tensor and no attention masks for cardinality. Masking is
  token *omission*, never zeroing.
* **PMA pooling** — a learned query cross-attends over the channel set at each
  spatial location (`ChannelFusion`), or a cheaper `O(C)` hypernetwork-weighted
  sum (`HypernetworkWeightedSum`).
* **PMA descriptor** — at inference the encoder output is reduced to a fixed
  `[B, D]` joint embedding by pooling by multi-head attention
  (`MultiheadAttentionPooling`), not a variable-length mean.
* **MAE** — DCP masking (independently per-channel spatial masks + dynamic
  channel masking), fixed-count memory tokens, and a shared decoder that is
  conditioned on the *target* channel's identity.

## Install

Managed with [uv](https://docs.astral.sh/uv/):

```bash
uv sync --extra dev
uv run pytest
```

## Usage

```python
import torch
from fuse_channels import ChannelFusionMAE

model = ChannelFusionMAE(
    image_size=16, patch_size=8, dim=192,
    encoder_depth=4, decoder_depth=2, n_heads=6,
    n_memory=4, num_markers=32,        # learnable identity table
)

# a batch is a list of variable-channel images: [C_b, 1, H, W]
images = [torch.randn(3, 1, 16, 16), torch.randn(7, 1, 16, 16)]
ids = [torch.randint(0, 32, (3,)), torch.randint(0, 32, (7,))]

out = model(images, ids)
out["loss"].backward()

# inference representation (no masking); descriptor is attention-pooled
mem, descriptor = model.encode(images, ids)   # descriptor: [B, D]
# model.encode(images, ids, pool="mean")  # or the old mean pool
```

### Just the fusion module

```python
from fuse_channels import ChannelFrontend, ChannelIdentity, ChannelFusion, ChannelSet

frontend = ChannelFrontend(image_size=16, patch_size=8, dim=192,
                           identity=ChannelIdentity(192, num_markers=32))
fusion = ChannelFusion(192, n_heads=6, n_queries=1, n_patches=4)

cs: ChannelSet = frontend(images, ids)      # real tokens only
fused, present = fusion(cs)                 # [B, 4, 192]; no channel padding
```

## Layout

| file | contents |
| --- | --- |
| `ops.py` | padding-free grouped attention primitives |
| `container.py` | `ChannelSet`: flattened variable-channel token batch |
| `identity.py` | learnable per-marker channel identities |
| `patch_embed.py` | channel-tied patch embedding + front-end |
| `fusion.py` | `ChannelFusion` (PMA), `MultiheadAttentionPooling`, `HypernetworkWeightedSum`, dense reference |
| `encoder.py` | memory-augmented spatial transformer |
| `decoder.py` | channel-aware shared decoder |
| `masking.py` | DCP masking (spatial-per-channel + dynamic channel) |
| `losses.py` | patch-normalized pixel loss |
| `mae.py` | `ChannelFusionMAE` end-to-end |

## Acceptance criteria

Covered by `tests/`:

- cardinality invariance for `C ∈ {1,3,6,20,60}` (`test_fusion.py`)
- exact padding-free vs masked-padded equivalence (`test_padding_free_equivalence`)
- omission correctness, permutation invariance, identity ablation
- no pad artifacts in the fusion path
- DCP masking never masks all channels; spatial masks are per channel
- MAE encoder sees only visible tokens; loss decreases on a fixed batch
- cost comparison: `examples/benchmark_fusion.py`

## Notes

* `flash-attn` is optional (`uv sync --extra flash`); the grouped-kernel
  fallbacks in `ops.py` are pure PyTorch and produce numerically identical
  results.
* `examples/train_mae.py` overfits a synthetic variable-channel dataset.
* `examples/benchmark_fusion.py` reports throughput for attention vs.
  hypernetwork-weighted-sum fusion at `C = 60`.

## Visualizations

There are three, because they answer different questions — see
[`viz/README.md`](viz/README.md) for details.

**Interactive behaviour (recommended).** The marimo dashboard runs the real
modules and recomputes every invariant as you move the controls:

```bash
uv sync --extra viz
uv run marimo edit viz/dashboard.py          # or: marimo run viz/dashboard.py
uv run marimo export html viz/dashboard.py -o viz/out/dashboard.html
```

It shows the live measurements table (padding-free ≡ masked-padded, permutation
invariance, omission correctness, identity ablation), a per-location attention
heatmap over channels, a fixed-size-descriptor-vs-`C` curve, and a
masked-channel reconstruction (virtual-staining) demo.

**Structure.** Generate the traced graph and the padding-free flow schematic:

```bash
uv run python viz/draw_architecture.py      # -> viz/out/{architecture_torchview,padding_free_flow}.{png,svg}
```

**Paper figure.** The hand-authored, module-level submission figure (vector
PDF/SVG + PNG, colour-blind-safe, tensor shapes on every block):

```bash
uv run python viz/paper_figure.py            # -> viz/out/architecture.{pdf,svg,png}
uv run python viz/cardinality_figure.py      # -> viz/out/cardinality.{pdf,svg,png}
```
