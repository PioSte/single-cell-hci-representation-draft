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
* **Additive channel identity** — a learned per-marker embedding, added to
  every patch token and never normalized across the channel axis.
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

### Channel identity

Each marker id indexes one row of a learned table, `ChannelIdentity`
(`nn.Embedding(num_markers, dim)`). The match is exact integer indexing — not
similarity, distance, or content:

```python
ident_per_ch = self.identity(ids_per_ch)      # [ΣC] ids -> [ΣC, dim], one row per channel
identity     = ident_per_ch[channel_global]   # broadcast to that channel's N patch tokens
tokens       = tokens + identity
```

* One row per marker id, learned; the same row is reused for every occurrence
  of that id (all patches, samples, batch) — that's the tying.
* `channel_global`/`channel_idx` only broadcast; the semantic match is entirely
  `channel_id -> row`.
* The table is a closed set of `num_markers` rows: an id with no trained row
  cannot be represented. Gradients reach only the rows that appeared; unused
  rows get no gradient (though an optimizer with weight decay, e.g. the `AdamW`
  in the example, still decays them toward zero).

The alternative — concatenating a one-hot id to the token — is deliberately not
used:

1. **After the next linear projection they are the same function — concat only
   costs more.** If you `concat(x, onehot(m))` and multiply by a weight, you get
   `W_x x + W_e · onehot(m) = W_x x + E[m]`. That second term *is* the embedding
   lookup; the one-hot path is just a verbose parameterization of the additive
   bias. The differences are: the token width becomes `D + num_markers` (so
   attention and MLP cost inflate for every layer), and the architecture is now
   hard-wired to the size of the marker registry. The embedding gives the
   identical identity signal at `num_markers × D` params without touching the
   network width.
2. **Every reference architecture uses the additive learned embedding.**
   ChannelViT: "learnable channel embeddings ... added to the patch tokens"
   (`references/channelvit/paper.md:17,99`). ChAda-ViT: additive `chan_c`
   alongside shared spatial pos (`chada-vit/paper.md:129`). ChA-MAEViT: channel
   tokens added to patch tokens in encoder and decoder
   (`cha-maevit/paper.md:113`). Panopticon: additive wavelength PE
   (`panopticon/paper.md:94,106`). Perceiver IO: identity is a learned dense
   modality/task embedding combined with the token (`perceiver-io/paper.md:103,546`)
   — a learned code, not a raw one-hot (their one-hot labels are data, not the
   modality code).
3. **The one direct one-hot ablation is negative.** ChannelViT Table 11
   (`channelvit/paper.md:704`): feeding "one-hot encoding of each channel
   combination" + MLP = 66.86, learned combination embeddings = 61.98, versus
   additive per-channel embeddings + shared classifier = 68.09. Their
   interpretation (`:714`): the shared classifier "regularizes the ViT to embed
   inputs with different channel combinations into the same space" — explicit
   identity features let the model specialize instead of sharing.
4. **One-hot codes are orthogonal; learned embeddings carry similarity.** A
   one-hot makes every marker equidistant — the model must learn all pairwise
   relations from scratch and cannot express "Hoechst is DAPI-like." Learned
   dense embeddings share statistical strength across markers and can place
   related channels close; ChannelViT shows the learned channel embeddings
   become correlated and interpretable for related channels
   (`channelvit/paper.md:245-255`). Panopticon's fine wavelength encoding is the
   explicit smooth-similarity version and beats coarse/no identity
   (`panopticon/paper.md:198`). This is the standard "learned embedding vs
   one-hot input" argument from NLP/ViT positional embeddings (Perceiver:
   identity should be a *feature* the net can use or ignore,
   `perceiver/paper.md:120,132`).

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
