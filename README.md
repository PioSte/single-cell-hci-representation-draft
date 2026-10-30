# fuse-channels

A PyTorch implementation of the **padding-free channel-fusion tokenizer** for a
high-content-imaging (HCI) foundation model, plus a channel-aware Masked
Autoencoder (MAE) built on top of it.

The tokenizer supports the variable-patch-size target described in
[`FLEXIVIT.md`](FLEXIVIT.md). FlexiViT (Beyer et al., *One Model for All Patch
Sizes*, [`references/flexivit/paper.md`](references/flexivit/paper.md)) supplies
the patch-kernel resampling: a canonical channel-tied kernel is **PI-resized**
(**p**seudo**i**nverse resize, Eq. 4) to a per-step or per-sample patch size.
The PE and scale choices go beyond FlexiViT: the encoder uses 2-D axial
RoPE on normalized grid coordinates, the decoder uses factorized-fractional
absolute PE, and an optional scale conditioner carries ``log(p)`` and physical
pixel size. The legacy fixed-grid learned/sine-cosine absolute PE remains
available as an ablation. Design rationale, trade-offs, and deviations from the
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
padding tokens**. `C` appears only in the tokenizer input length `ΣC·N`; the
fused output is `[B, N, D]` (`N = floor(H/p)·floor(W/p)` is set by the current
patch size) and the pooled descriptor is `[B, D]`.

## Variable pixel size: `p` is a pixel count, not a physical size

High-content images come from different instruments, so the **pixel size**
(`pixel_size`, micrometres per pixel) and therefore the number of pixels per
micrometre differs across datasets. A token must cover a comparable *physical*
extent on every instrument, otherwise the same marker is a different object at
different sites. The mechanism is: **choose the patch size per sample so that
the token extent in micrometres stays constant.**

```
p               patch edge in PIXELS (a model knob; FlexiViT's axis, never a physical size)
pixel_size      micrometres per pixel (dataset metadata, varies per instrument)
p * pixel_size  physical token extent in micrometres (the quantity held constant)

p_b = round(target_um / pixel_size_b)     one p per pixel size, no snapping
```

`pixel_size` is the only name this repo uses for that quantity — in code, in
comments and in prose — and it is **required metadata**: every image has a
physical pixel size, so `forward`/`encode` raise rather than invent a default.
`log_pixel_size` is its logarithm.

**Expected data: square pixels from a finite acquisition fleet.** The
**acquisition fleet** is the set of acquisition instruments (microscopes /
scanners) and acquisition settings the model is trained *and* deployed on. It is
a set of instruments; each configuration acquires at one physical pixel size, so
the **fleet's pixel sizes** are a finite sorted table, not an interval:

```
fleet_pixel_sizes = (0.598, 0.646, 0.650, 0.652, 0.703)   # µm/px, one per configuration
```

(`FLEET_PIXEL_SIZES` in `src/fuse_channels/fleet.py`; the library itself is
general — another dataset passes its own table, or `fleet_pixel_sizes=None`.) A pixel size
outside the table implies a `p` the model never trained, so it is an
**ingest-time error**, not something to snap: `canonical_fleet_pixel_size`
maps metadata to its table entry within a tolerance (`0.59766` and
`0.6500000000000001` are the same instrument) and raises otherwise.

`pixel_size` is a scalar and the patch kernel is square, which is exact because
the target instruments acquire square pixels: `PhysicalSizeX ≠ PhysicalSizeY`
beyond a small tolerance is metadata to fix or reject, not an anisotropy the
model is expected to absorb. The fleet spans only 0.703/0.598 = 1.18×, so for a
12.8 µm token `p` takes just three values:

| pixel_size | p | extent | error |
|---|---|---|---|
| 0.598 | 21 | 12.56 µm | −1.9 % |
| 0.646 | 20 | 12.92 µm | +0.9 % |
| 0.650 | 20 | 13.00 µm | +1.6 % |
| 0.652 | 20 | 13.04 µm | +1.9 % |
| 0.703 | 18 | 12.65 µm | −1.1 % |

so `patch_sizes = (18, 20, 21)` and the residual token-extent error is ≤ ~1.9 %
(a ≈ 0.5/`p` rounding case; at a smaller token extent it is larger, e.g. ~3 % for
8 µm). Note that 0.646/0.650/0.652 all collapse to `p = 20`: `p` alone cannot
distinguish those three instruments, which is exactly why the scale conditioner
also receives `log(pixel_size)`.

So `p` *absorbs* the pixel-size variation: 0.598 µm/px uses `p = 21` for a
12.8 µm token, 0.703 µm/px uses `p = 18`, correspondingly for other target
extents. This is what
makes "train once, infer on any instrument seen in training, no finetuning"
work: one weight set, PI-resized kernels, and a token that means the same
physical thing everywhere. It is **not** extrapolation — a `p` outside the
trained set is out of contract (see
[`FLEXIVIT.md` §0](FLEXIVIT.md#0-patch-size-vs-pixel-size-the-contract)).
The scale conditioner receives both `log(p)` and `log(pixel_size)` because FlexiViT
knows nothing about physical units; `p` alone cannot distinguish "large patch at
fine sampling" from "small patch at coarse sampling".

**The invariant is the patch, not the crop.** What is held constant across
instruments is the physical footprint of a *single token*, `p · pixel_size ≈
target_um`. The physical extent of the whole window, `n_tokens · p · pixel_size`, is a
separate quantity and is free to vary: it is cell-sized when the patch count is
derived from the segmented object, which is the intended behaviour. Scaling the
number of tokens with the object does not violate the constant-patch-size
contract.

**Positional embeddings must then be metric.** With a fixed crop extent the
default grid-fraction PE is already metric, because its normalization
denominator is constant. With cell-sized crops (variable patch count) it is not:
the phase encodes `distance / crop_extent`, so an additive scale conditioner
cannot recover the physical geometry. Use `pe_space="physical"` (encoder RoPE
and decoder PE both evaluate `index * p * pixel_size / ref_um`) for cell-sized grids. The loader in `crops.py` produces exactly that layout:
`plan_patches` reads boxes straight out of the raw planes, so `crop_px` is an
exact multiple of `p`, nothing is padded, and border cells get a shifted box
instead of a lost border.

## Architecture

```
per-channel tied Conv2d(1 -> D) patch embed, PI-resized to the current p
    (content-only token; no positional embedding, no channel id here)
    -> per-location channel fusion: one learned query over the C channel tokens
       (ChannelFusion, Q/K/V projections to 3D)
       logits += identity routing bias;  fused += identity skip
       + optional scale embedding
    -> padding-free transformer encoder over [present locations, D]
       with fixed-count memory tokens and 2-D axial RoPE on q/k
    -> trained PMA descriptor over encoder patch tokens [B, D]  (appended to KV)
    -> channel-aware shared decoder: target-identity + fractional target-position
       queries, target self-attention, cross-attention to encoder representation
```

* **Channel-tied projection.** One canonical `Conv2d(1, D, kernel_size=p0,
  stride=p0)` is shared by every channel (ChannelViT's tied filters). For any
  other patch size `p` the kernel is **PI-resized** — pseudoinverse resize,
  `(B.T)^+`, not bilinearly resized (`references/flexivit/paper.md:135-171`,
  Eqs. 2-4). For `p >= p0` it is the exact inverse of the patch resize: a
  `p0`-patch that is bilinearly resized to `p` keeps its token exactly,
  `⟨B x, P ω⟩ = ⟨x, ω⟩`, so token magnitude does not drift with patch size and
  no separate intensity gain is needed. For `p < p0` the resize is
  rank-deficient and PI-resize is the least-squares fit instead. Patches tile
  `floor(H/p) × floor(W/p)`; a border smaller than `p` is cropped/ignored, as
  in FlexiViT's patchification contract. There is no positional embedding at
  this stage.
* **Channel identity: routing bias + output skip.** `ChannelEmbedding` is a
  learned `num_channel_ids × D` table, one row per channel id, never shared
  across channels. It is deliberately **not** added to the patch token before
  the fusion LayerNorm. Instead it enters twice: as a per-head additive bias on
  the per-location channel-attention logits, and as an attention-pooled skip on
  the fused token after the content projection:
  `fused = proj(Σ_c a_c LN(t_c)) + skip(Σ_c a_c e_c)`. This keeps identity
  load-bearing for routing and in the encoder representation (ChannelViT's
  finding) while making the fused representation invariant to content
  magnitude, so the identity/content balance cannot drift when the patch size
  changes (a fixed-scale vector summed into a variable-scale token before a
  LayerNorm would drift with `p`; see `FLEXIVIT.md` §3.1).
* **Padding-free.** Tokens from the tokenizer are flattened `[ΣC·N, D]` with integer
  group ids; no `[B, C_max, N, D]` tensor and no attention masks for
  cardinality are ever materialised. Masking is token *omission*, never
  zeroing. The encoder receives one token per **present** `(sample, patch)`
  location, so its input length is the number of present locations, not the
  number of visible channel-patches.
* **Per-location fusion.** `ChannelFusion` reduces the `C` channels at each
  `(sample, patch)` location with one shared learned query, costing `O(C)` per
  location. It follows Panopticon's recipe: LayerNormed tokens go through
  learned `q_proj` / `k_proj` / `v_proj` into an attention width
  `attn_dim = 3·D` (Panopticon's `D_attn = 3D`), attention runs there, and
  `proj_out` maps back to `D`. `attn_dim=D` is the parameter-matched ablation;
  Panopticon's Table 5 found `D_attn > D` important. `IdentityGatedSum` is the
  `O(C)` marker-routed ablation:
  `joint = Σ_c softmax_c(MLP(channel_embed_c)) · V_c`.
* **Encoder position: 2-D axial RoPE.** Fusion is position-agnostic. RoPE is
  applied to encoder queries and keys from **normalized** coordinates
  `(r + 0.5)/grid_h` and `(c + 0.5)/grid_w`, so the phase describes layout only;
  the patch size/scale enters through `log(p)` conditioning instead. Memory
  tokens get zero rotation. `head_dim = dim / n_heads` must be divisible by 4.
* **Decoder position: factorized-fractional absolute PE.** Target queries use
  `g_row(v) + g_col(u)` with `u = (c + 0.5)/grid_w`, `v = (r + 0.5)/grid_h`.
  This is a function, not a grid-sized table, so it is defined for any
  `(grid_h, grid_w)` and needs no resampling when `p` changes. It is
  load-bearing: masked patches of one channel share the mask token and channel
  embedding, so position is what makes their predictions differ (MAE's
  `decoder_pos_embed` argument, `references/mae/paper.md:103`).
* **Legacy absolute PE ablation.** `pos_embed="learned"` or `"sincos"` restores
  the old fixed-grid scheme: post-fusion table plus decoder target table, with
  no encoder RoPE. It cannot change patch size within a run.
* **Scale conditioning.** `scale_conditioning="add"` (default) adds a Fourier
  embedding of `log(p)` and `log(pixel_size)` to the encoder tokens
  (before the pre-LN branch; the residual stream carries it) and to the
  decoder's target channel embedding. `"adaln"` modulates the LayerNorms
  instead; `"none"` disables it. `add` is not erased by pre-LN: the additive
  vector remains in the unnormalized residual stream, and only its
  all-ones/mean component is removed from the normalized branch.
* **Encoder and descriptor.** Standard pre-norm Q/K/V/O transformer blocks run
  over the flattened real tokens with fixed-count memory tokens. The `[B, D]`
  descriptor is a PMA-style pool over the encoder's *patch* tokens (not the
  memory tokens); it therefore carries the identity skip, which the decoder can
  use as a panel-composition summary. It is trained through the MAE loss and
  appended as one token per sample to the decoder KV.
* **Decoder.** One decoder is shared across all channels. Each target query is
  `mask token + target spatial PE + target channel embedding`; queries
  self-attend within each sample, then cross-attend to the encoder
  representation + memory + descriptor. A fully dropped channel has no encoder
  evidence, only its identity query, so identity acts as a **target
  specification** as well as an input feature. The reconstruction head has
  three modes. `head_mode="per_p"` (default) keeps a separate linear
  `dim -> p²` head for every configured patch size: exact for configured `p`,
  and an unseen `p` falls back to the nearest configured head with
  `p_head >= p` + bilinear point-sampling, never upsampling.
  `head_mode="shared"` is **`p`-free**: one canonical `dim -> p0²` head
  resampled to *any* `p` in both directions by a fixed operator
  (`head_resample` = `"area"` default, `"bilinear"`, or `"pi"`), so no
  parameter depends on `p` and a checkpoint trained on one fleet serves
  another. `head_mode="canonical"` is the legacy single-head ablation. The
  decoder is pretraining-only and discarded after MAE, so heads are a
  pretraining parameter cost only.
* **Token budget.** `max_tokens` (optional) filters the random `flexi_random`
  sampler to patch sizes with `floor(H/p)·floor(W/p) <= max_tokens`. Explicit
  `patch_sizes=` overrides the budget for inference/evaluation. This caps the
  `p=8`-dominated step time without changing the configured model.
* **Masking.** DCP (ChA-MAEViT Eq. 1): patch-only, channel-only, or combined
  regions, with the region probabilities separate from `patch_ratio`. HCS is
  the channel-subset sampler used by the channel region (and is available as
  `p_patch=0, p_channel=1`). Channels dropped by the channel region stay in
  `images` and are reconstructed as targets.
* **Loss.** Default `normalized_patch_mse` (standard MAE patch-normalized MSE).
  `loss="cha_maevit"` selects the ChA-MAEViT ablation: raw-pixel L2 plus
  Fourier-amplitude L1 (`lambda_f=0.01`).

### Why a learned channel-embedding table

A learned table is the width-efficient parameterization of the same additive
identity bias that `concat(x, onehot(m)) @ W` learns:
`W_x x + W_e · onehot(m) = W_x x + E[m]` — here applied to the fused content
(`W_x` is the fusion output projection) and to the routing logits rather than
inside the patch token. The concat path widens every token to
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
    patch_sizes=(4, 8, 16),           # FlexiViT; patch_size is canonical p0
    max_tokens=1024,                  # optional training sampler cap
    head_mode="per_p",                # or "shared" (p-free) / "canonical" (legacy)
    # head_patch_size=8,              # required for head_mode="shared" (p0)
    head_resample="area",             # head_mode="shared" only: area/bilinear/pi
    scale_conditioning="add",         # or "adaln" / "none"
    pixel_size=0.5,                   # optional µm/px; log p is always used
    # pos_embed="learned",            # legacy fixed-grid PE ablation
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
tokenizer = ChannelTokenizer(16, 8, 192, embedding)
# Default target: no spatial table in fusion (encoder RoPE carries layout);
# Q/K/V project to attn_dim=3*192=576, matching Panopticon's D_attn=3D.
fusion = ChannelFusion(192, n_heads=6)
# Legacy fixed-grid ablation: post-fusion learned table.
fusion_learned = ChannelFusion(192, n_heads=6, n_patches=4, pos_embed="learned")

cs: ChannelSet = tokenizer(images, ids)   # real tokens only; cs.channel_embed is [ΣC·N, 192]
fused, present = fusion(cs)              # [B, 4, 192]; no channel padding

# the O(C) marker-routed ablation
gated = IdentityGatedSum(192)
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
| `src/fuse_channels/patch_embed.py` | `ChannelTiedPatchEmbed`, `ChannelTokenizer` (content-only tokens; identity travels separately) |
| `src/fuse_channels/fusion.py` | `ChannelFusion`, `IdentityGatedSum`, `MultiheadAttentionPooling`; `_dense_fusion_reference` (private, test/viz-only) |
| `src/fuse_channels/rope.py` | 2-D axial RoPE on normalized grid coordinates |
| `src/fuse_channels/flexi.py` | FlexiViT PI- (pseudoinverse-) resize for the patch-embedding kernel |
| `src/fuse_channels/pos_embed.py` | fractional absolute PE plus legacy learned/sincos tables |
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
  (`test_identity_changes_routing`, `test_identity_gated_sum`); identity is a
  routing bias + output skip, never summed into content before a LayerNorm
  (`tests/test_identity_decoupling.py`), so identity routing is patch-size
  invariant; no downstream task metric is claimed
- no pad artifacts and an exact presence mask (`test_no_pad_artifacts`)
- one shared pooling query per location and attention dropout behaviour
  (`test_pooling_query_is_shared_across_locations`, dropout tests); the default
  target has no fusion PE, while the legacy table remains opt-in
  (`test_post_fusion_spatial_pos_is_legacy_opt_in`)
- `pos_embed` variants: fixed `sincos` adds no parameters, is deterministic,
  keeps the padding-free ≡ dense equivalence and the cardinality/permutation
  invariants, and is row-major over the grid (`tests/test_pos_embed.py`,
  `test_sincos_pos_embed_*`); the decoder's target PE is load-bearing
  (`test_decoder_target_position_changes_predictions`)
- FlexiViT target (`tests/test_flexivit.py`): normalized RoPE is grid-size
  invariant, fractional PE is defined for non-square/dynamic grids; `B @
  vec(x)` equals `F.interpolate(x)`, PI-resize preserves `⟨B x, P ω⟩ =
  ⟨x, ω⟩` for upsampling and keeps token RMS bounded across `p` (no intensity
  scalar), per-batch and per-sample patch sizes change `n_patches` and
  the per-p reconstruction heads are exact (no resize) and the canonical
  fallback refuses to upsample, `max_tokens` filters the random sampler,
  non-divisible `H/W` uses the floor grid, and additive/AdaLN scale
  conditioning is wired into both ends
- encoder input length = number of present locations, with token count
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
* The default `head_mode="per_p"` changes decoder state-dict keys from
  `decoder.head.head.*` to `decoder.head.heads.<p>.*`. Load an older
  checkpoint with `head_mode="canonical"`, or remap the output head.
  `head_mode="shared"` uses the `decoder.head.head.*` keys and, unlike the
  other two modes, its state dict is independent of `patch_sizes`.
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
`IdentityGatedSum` replaces the misleading `HypernetworkWeightedSum` name;
position now follows the FLEXIVIT.md target (encoder RoPE + decoder fractional PE,
no fusion table), with the legacy table retained as an ablation; the decoder
uses per-patch-size output heads by default, with a nearest-head bilinear
fallback for unseen `p`, a `p`-free shared head (`head_mode="shared"`) that
resamples one canonical head to any `p`, and the canonical single head retained
as an ablation;
tokens use FlexiViT's floor/crop border
contract rather than rejecting non-divisible tiles; channel identity is
decoupled from the patch token (routing bias + output skip) so the
identity/content balance is patch-size invariant, a placement deviation from
ChannelViT's additive token that keeps its finding that per-channel identity is
load-bearing; and the acceptance claims are limited to what the tests cover. Rationale, trade-offs,
and the corrected reference claims are in [`docs/design.md`](docs/design.md).
