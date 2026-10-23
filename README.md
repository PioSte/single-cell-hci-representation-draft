# fuse-channels

A PyTorch implementation of the **padding-free channel-fusion front-end** for a
high-content-imaging (HCI) foundation model, plus a channel-aware Masked
Autoencoder (MAE) built on top of it.

It implements a padding-free channel-fusion module for an HCI foundation model
and its MAE follow-up. The design rationale, trade-offs, and deviations from the
literature are in [`docs/design.md`](docs/design.md).

## What problem it solves

An HCI image is a set of per-channel embeddings `{E_c ∈ R^D}` where the channel
count `C` varies from 1 to 5+ (much larger in multiplexed panels), channel
order is meaningless, and panels differ across labs. We need a representation
whose size is independent of `C`:

```
f({E_1 .. E_C}) -> [N, D] per sample, then -> [D] descriptor,
```

without concatenation blow-up, with channel identity preserved, and with **no
padding tokens**. `C` appears only in the front-end input length `ΣC·N`; the
fused output is `[B, N, D]` (`N` is fixed by the image and patch size) and the
pooled descriptor is `[B, D]`.

## Architecture

```
per-channel tied Conv2d(1 -> D) patch embed        (no positional embedding here)
    + additive ChannelEmbedding (per channel id)
    -> per-location channel fusion: one learned query over the C channel tokens
       (ChannelFusion, K = V = LayerNormed tokens)  + post-fusion spatial PE
    -> padding-free transformer encoder over [present locations, D]
       with fixed-count memory tokens
    -> trained content-only PMA descriptor [B, D]   (appended to decoder KV)
    -> channel-aware shared decoder: target-identity + target-position queries,
       target self-attention, cross-attention to encoder representation + memory
```

* **Channel-tied projection.** One `Conv2d(1, D, kernel_size=patch_size,
  stride=patch_size)` is shared by every channel (ChannelViT's tied filters).
  There is no positional embedding at this stage; position is added after
  fusion.
* **Additive channel identity.** `ChannelEmbedding` is a learned
  `num_channel_ids × D` table. The row for a channel id is added to every patch
  token of that channel; it is never normalized across the channel axis.
* **Padding-free.** Front-end tokens are flattened `[ΣC·N, D]` with integer
  group ids; no `[B, C_max, N, D]` tensor and no attention masks for
  cardinality are ever materialised. Masking is token *omission*, never
  zeroing. The encoder receives one token per **present** `(sample, patch)`
  location, so its input length is the number of present locations, not the
  number of visible channel-patches.
* **Per-location fusion.** `ChannelFusion` reduces the `C` channels at each
  `(sample, patch)` location with one shared learned query, costing `O(C)` per
  location. Keys and values are the LayerNormed tokens themselves (no K/V
  projections; the Panopticon-style simplification); the query and the output
  projection are learned. `IdentityGatedSum` is the `O(C)` marker-routed
  ablation: `joint = Σ_c softmax_c(MLP(channel_embed_c)) · V_c`.
* **Post-fusion positional embedding.** Both fusion modules add the learned
  spatial PE to the `[B, N, D]` fused output, so channel attention is
  position-agnostic and position enters the encoder once per location.
* **Encoder and descriptor.** Standard pre-norm Q/K/V/O transformer blocks run
  over the flattened real tokens with fixed-count memory tokens. The `[B, D]`
  descriptor is a PMA-style pool over the encoder's *content* tokens (not the
  memory tokens), is trained through the MAE loss, and is appended as one token
  per sample to the decoder KV.
* **Decoder.** One decoder is shared across all channels. Each target query is
  `mask token + target spatial PE + target channel embedding`; queries
  self-attend within each sample, then cross-attend to the encoder
  representation + memory + descriptor. A fully dropped channel has no encoder
  evidence, only its identity query, so identity acts as a **target
  specification** as well as an input feature.
* **Masking.** DCP (ChA-MAEViT Eq. 1): patch-only, channel-only, or combined
  regions, with the region probabilities separate from `patch_ratio`. HCS is
  the channel-subset sampler used by the channel region (and is available as
  `p_patch=0, p_channel=1`). Channels dropped by the channel region stay in
  `images` and are reconstructed as targets.
* **Loss.** Default `normalized_patch_mse` (standard MAE patch-normalized MSE).
  `loss="cha_maevit"` selects the ChA-MAEViT ablation: raw-pixel L2 plus
  Fourier-amplitude L1 (`lambda_f=0.01`).

### Why additive channel embedding

A learned table is the width-efficient parameterization of the same additive
identity bias that `concat(x, onehot(m)) @ W` learns:
`W_x x + W_e · onehot(m) = W_x x + E[m]`. The concat path widens every token to
`D + num_channel_ids` and pays that width in every attention and MLP layer; the
table carries the same signal at fixed width and `num_channel_ids × D`
parameters. Identity is not optional: ChannelViT's mean-embedding ablation
(Table 10) collapses accuracy from 57.60/68.09 to 1.26/2.49 on JUMP-CP. The
full derivation, the Table 11 caveat, and the corrected reference claims are in
[`docs/design.md`](docs/design.md).

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
    n_memory=4, num_channel_ids=32,   # size of the learnable identity table
    fusion="attention",               # or "identity_gated" (ablation)
    loss="mae",                       # or "cha_maevit" (ablation)
)

# a batch is a list of variable-channel images: [C_b, 1, H, W]
images = [torch.randn(3, 1, 16, 16), torch.randn(7, 1, 16, 16)]
ids = [torch.randint(0, 32, (3,)), torch.randint(0, 32, (7,))]

out = model(images, ids)
out["loss"].backward()

# inference representation (no masking): the descriptor is the trained PMA pool
mem, descriptor = model.encode(images, ids)               # descriptor: [B, D]
_, mean_descriptor = model.encode(images, ids, pool="mean")  # ablation
```

### Just the fusion module

```python
import torch
from fuse_channels import (
    ChannelEmbedding,
    ChannelFusion,
    ChannelSet,
    ChannelTokenizer,
    IdentityGatedSum,
)

images = [torch.randn(3, 1, 16, 16), torch.randn(7, 1, 16, 16)]
ids = [torch.randint(0, 32, (3,)), torch.randint(0, 32, (7,))]

embedding = ChannelEmbedding(192, num_channel_ids=32)
frontend = ChannelTokenizer(16, 8, 192, embedding)
fusion = ChannelFusion(192, n_heads=6, n_patches=4)   # one query per location

cs: ChannelSet = frontend(images, ids)   # real tokens only; cs.channel_embed is [ΣC·N, 192]
fused, present = fusion(cs)              # [B, 4, 192]; no channel padding

# the O(C) marker-routed ablation
gated = IdentityGatedSum(192, n_patches=4)
gated_out, present = gated(cs)
```

### Identities and group ids

* `ChannelSet.channel_id` is the **identity key**: a dataset-level channel id
  that indexes the `ChannelEmbedding` table and is constant across that
  channel's patches. It is not a panel position.
* `ChannelSet.channel_idx` is the **panel position** of the channel inside its
  sample (`0 .. C_b - 1`). It is bookkeeping only; reordering panels leaves the
  fused output unchanged.
* `ChannelSet.channel_group_id` identifies one channel *instance* across the
  whole batch. `IdentityGatedSum` uses it to evaluate its channel MLP once per
  channel and broadcast to that channel's patches.
* `grouped_attention` (and the other `grouped_*` primitives) means
  padding-free batching by integer group id, **not** grouped-query attention.
  Encoder, decoder, and descriptor pooling group by sample; fusion groups by
  `(sample, patch)` location.

## Layout

| file | contents |
| --- | --- |
| `src/fuse_channels/ops.py` | padding-free grouped attention primitives |
| `src/fuse_channels/container.py` | `ChannelSet`: flattened variable-channel token batch |
| `src/fuse_channels/identity.py` | `ChannelEmbedding`: learned per-channel-id additive table |
| `src/fuse_channels/patch_embed.py` | `ChannelTiedPatchEmbed`, `ChannelTokenizer` |
| `src/fuse_channels/fusion.py` | `ChannelFusion`, `IdentityGatedSum`, `MultiheadAttentionPooling`; `_dense_fusion_reference` (private, test/viz-only) |
| `src/fuse_channels/encoder.py` | `TransformerEncoder` / `TransformerBlock` with memory tokens |
| `src/fuse_channels/decoder.py` | `ChannelAwareDecoder` / `DecoderBlock` |
| `src/fuse_channels/masking.py` | `DCPVisibilitySampler` (Eq. 1) and `HierarchicalChannelSampler` (HCS) |
| `src/fuse_channels/losses.py` | `normalized_patch_mse`, `cha_maevit_recon_loss` |
| `src/fuse_channels/mae.py` | `ChannelFusionMAE` end-to-end |
| `src/fuse_channels/data.py` | `SyntheticChannelDataset` and `collate_fn` for examples/tests |
| `examples/linear_probe.py` | frozen-encoder linear probe on a synthetic channel-presence label |
| `examples/train_mae.py` | synthetic variable-channel MAE training loop |
| `examples/benchmark_fusion.py` | fusion throughput comparison at `C = 60` |
| `docs/design.md` | design rationale, reference corrections, deviations |

`_dense_fusion_reference` is a test/viz-only dense-padded reference used to
prove the padding-free math; it is not part of the model path or the public API.

## Acceptance criteria

Covered by `tests/`:

- cardinality invariance for `C ∈ {1,3,6,20,60}` (`test_cardinality_invariance`)
- exact padding-free vs masked-padded equivalence (`test_padding_free_equivalence`)
- omission vs the dense reference (`test_omission_equivalence`)
- permutation invariance (`test_permutation_invariance`)
- identity routing: changing the channel embedding changes the fused output
  (`test_identity_changes_routing`, `test_identity_gated_sum`); no downstream
  task metric is claimed
- no pad artifacts and an exact presence mask (`test_no_pad_artifacts`)
- post-fusion PE, one query shared across locations, and attention dropout
  behaviour (`test_post_fusion_spatial_pos`,
  `test_pooling_query_is_shared_across_locations`, dropout tests)
- encoder input length = number of present locations, with front-end token count
  = visible channel-patches (`test_encoder_input_length_is_present_locations`)
- DCP Eq. 1 regions and HCS counts (`test_masking.py`); channels dropped by DCP
  are reconstruction targets (`test_dcp_dropped_channel_is_a_reconstruction_target`)
- descriptor is trained through the MAE loss, appended to the decoder KV, and
  `encode(pool="attention")` matches the forward descriptor
  (`test_descriptor_is_trained_by_the_mae_loss`,
  `test_decoder_consumes_the_descriptor`,
  `test_forward_and_encode_share_the_descriptor`)
- loss variants (`test_losses.py`) and the frozen linear-probe smoke test
  (`test_linear_probe.py`)
- single-batch MAE loss decreases (`test_overfit_single_batch`)

The cost comparison is an example, not a test:
`examples/benchmark_fusion.py` reports wall-clock for attention vs.
identity-gated fusion at `C = 60`.

## Notes

* `examples/train_mae.py` trains on a synthetic variable-channel dataset; it
  does not assert an overfit threshold.
* Attention uses `F.scaled_dot_product_attention` over real, grouped tokens. No
  custom fused kernel is required or implemented.
* The model's data path is torch-only; the visualization extras are separate.

## Visualizations

Several artefacts exist because they answer different questions. See
[`viz/README.md`](viz/README.md) for the current list and commands.

**Interactive behaviour.** The marimo dashboard runs the real modules and
recomputes the invariants as you move the controls:

```bash
uv sync --extra viz
uv run marimo edit viz/dashboard.py          # or: marimo run viz/dashboard.py
uv run marimo export html viz/dashboard.py -o viz/out/dashboard.html
```

It includes the live measurements table, a per-location channel-attention
heatmap, and a masked-channel reconstruction demo. See `viz/README.md` for the
exact panels.

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

## Design deviations

The implementation differs from the earlier reference designs in several
specifics: HCS and DCP's channel region are unified into one channel-subset
sampler; the decoder consumes the encoder representation + memory + a trained
descriptor and conditions on the target channel embedding, not "unmasked
channel tokens"; the descriptor is trained rather than inference-only; the
default loss is patch-normalized MSE with ChA-MAEViT as an ablation;
`IdentityGatedSum` replaces the misleading `HypernetworkWeightedSum` name; and
the acceptance claims are limited to what the tests cover. Rationale, trade-offs,
and the corrected reference claims are in [`docs/design.md`](docs/design.md).
