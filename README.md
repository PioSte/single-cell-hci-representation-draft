# single-cell-hci-representation-draft

A PyTorch implementation of the **padding-free channel-fusion tokenizer** for a
high-content-imaging (HCI) foundation model, plus a channel-aware Masked
Autoencoder (MAE) and a joint-embedding predictive (JEPA) objective built on
the same backbone.

The tokenizer supports the variable-patch-size target described in
[`FLEXIVIT.md`](FLEXIVIT.md). FlexiViT (Beyer et al., *One Model for All Patch
Sizes*, [`references/flexivit/paper.md`](references/flexivit/paper.md)) supplies
the patch-kernel resampling: a canonical channel-tied kernel is **PI-resized**
(**p**seudo**i**nverse resize, Eq. 4) to a per-step or per-sample patch size.
The PE and scale choices go beyond FlexiViT: the encoder uses 2-D axial
RoPE (metric coordinates by default, with normalized grid coordinates as the
fixed-window ablation), the decoder uses factorized-fractional absolute PE, and
an optional scale conditioner carries ``log(p)`` and physical pixel size. The legacy fixed-grid learned/sine-cosine absolute PE remains
available as an ablation. Design rationale, trade-offs, and deviations from the
literature are in [`docs/design.md`](docs/design.md).

## What problem it solves

An HCI image is a set of per-channel embeddings `{E_c ∈ R^D}` where the channel
count `C` varies from 1 to 5+ (much larger in multiplexed panels), channel
order is meaningless, and panels differ across labs. We need a representation
whose size is independent of `C`:

```
f({E_1 .. E_C}) -> [N, D] per sample -> spatial pooling to fixed size latent [L, D],
```

without concatenation blow-up, with channel identity preserved, and with **no
padding tokens**. `C` appears only in the tokenizer input length `ΣC·N`; the
fused output is `[K_f, D]`, one vector per *present* `(b, n)` location
(`N = floor(H/p)·floor(W/p)` per sample is set by the current patch size, and
absent locations own no row). **Spatial pooling to fixed size latent** then maps those encoder
tokens to `L` latents per sample, `[B, L, D]` (`n_latents = L` is a
hyperparameter), so the model-level encoder output and the decoder KV are
independent of both `C` and `N`. The decoder
still emits one prediction row per requested reconstruction target (the DCP mask
complement), not per encoder token. See “Naming this stage” below for the
nuances of the phrase.

## Variable pixel size: `p` is a pixel count, not a physical size

High-content images come from different instruments, so the **pixel size**
(`pixel_size`, micrometres per pixel) and therefore the number of pixels per
micrometre differs across datasets. A patch must cover a comparable *physical*
extent on every instrument, otherwise the same marker is a different object at
different sites. The mechanism is: **choose the patch size per sample so that
the patch extent in micrometres stays as constant as integer rounding allows.**

```
p               patch edge in PIXELS (a model knob; FlexiViT's axis, never a physical size)
pixel_size      micrometres per pixel (dataset metadata, varies per instrument)
p * pixel_size  physical patch size in micrometres (held ~constant; rounding residual)

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
12.8 µm patch `p` takes just three values:

| pixel_size | p | extent | error |
|---|---|---|---|
| 0.598 | 21 | 12.56 µm | −1.9 % |
| 0.646 | 20 | 12.92 µm | +0.9 % |
| 0.650 | 20 | 13.00 µm | +1.6 % |
| 0.652 | 20 | 13.04 µm | +1.9 % |
| 0.703 | 18 | 12.65 µm | −1.1 % |

so `patch_sizes = (18, 20, 21)` and the residual patch-extent error is ≤ ~1.9 %
(a ≈ 0.5/`p` rounding case; at a smaller patch it is larger, e.g. ~3 % for an
8 µm patch). Note that 0.646/0.650/0.652 all collapse to `p = 20`: `p` alone cannot
distinguish those three instruments, which is exactly why the scale conditioner
also receives `log(pixel_size)`.

So `p` *absorbs* the pixel-size variation: 0.598 µm/px uses `p = 21` for a
12.8 µm patch, 0.703 µm/px uses `p = 18`, correspondingly for other target
extents. This is what
makes "train once, infer on any instrument seen in training, no finetuning"
work: one weight set, PI-resized kernels, and a patch that covers the same
physical extent everywhere. It is **not** extrapolation — a `p` outside the
trained set is out of contract (see
[`FLEXIVIT.md` §0](FLEXIVIT.md#0-patch-size-vs-pixel-size-the-contract)).
The scale conditioner receives both `log(p)` and `log(pixel_size)` because FlexiViT
knows nothing about physical units; `p` alone cannot distinguish "large patch at
fine sampling" from "small patch at coarse sampling".

**The invariant is the patch, not the crop.** What is held constant across
instruments is the physical footprint of a *single patch*, `p · pixel_size ≈
target_um`. The physical extent of the whole window, `n_tokens · p · pixel_size`, is a
separate quantity and is free to vary: it is cell-sized when the patch count is
derived from the segmented object, which is the intended behaviour. Scaling the
number of tokens with the object does not violate the constant-patch-size
contract.

**Positional embeddings must then be metric.** With a fixed crop extent the
default grid-fraction PE is already metric, because its normalization
denominator is constant. With cell-sized crops (variable patch count) it is not:
the phase encodes `distance / crop_extent`, not distance, so the same physical
gap is rotated differently in differently sized cells. Use
`pe_space="physical"` (encoder RoPE and decoder PE both evaluate
`index * p * pixel_size / ref_um`) for cell-sized grids. The idea is published:
world-coordinate RoPE in `references/muvit/paper.md:97` and GSD-scaled PE in
`references/scale-mae/paper.md:91`. Whether a scale token could substitute for
it is the part that is still an untested hypothesis — see `FLEXIVIT.md` §0.1.
The loader in `crops.py` produces exactly that layout:
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
    -> spatial pooling to fixed size latent (PMA) [B, L, D]
       (L independent of C and N; trained through the loss)
    -> channel-aware shared decoder: target-identity + fractional target-position
       queries, target self-attention, cross-attention to memory + latents
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
* **Padding-free at every stage.** Tokens from the tokenizer are flattened
  `[ΣC·N, D]` with integer group ids; no `[B, C_max, N, D]` tensor and no
  attention masks for cardinality are ever materialised. Masking is token
  *omission*, never zeroing, so nothing downstream has to undo a pad row.
  Fusion returns one vector per **present** `(sample, patch)` location as
  `[K_f, D]` (not a `[B, N, D]` grid), and the decoder's reconstruction rows
  concatenate each target's `p²` pixels (`[Σ p²]`), so a mixed-`p` batch needs
  no padded pixel block either.
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
  applied to encoder queries and keys from **metric** coordinates by default
  (`pe_space="physical"`): `(r + 0.5) * p * pixel_size / ref_um` per token, so
  the phase encodes the realized physical separation, not a fraction of the
  grid. `pe_space="grid"` is the fixed-window ablation, where coordinates are
  `(r + 0.5)/grid_h` and `(c + 0.5)/grid_w` (times a reference extent) and the
  phase describes layout only; that mode stops being metric when the window
  extent varies. Memory tokens get zero rotation. `head_dim = dim / n_heads`
  must be divisible by 4.
* **Decoder position: factorized-fractional absolute PE.** Target queries use
  `g_row(v) + g_col(u)` with per-axis positions that are metric by default:
  `u, v = (index + 0.5) * p * pixel_size / ref_um`, using the same
  `target_step_um = p * pixel_size` as the encoder's frame. In the
  `pe_space="grid"` ablation they are `u = (c + 0.5)/grid_w` and
  `v = (r + 0.5)/grid_h`. This is a function, not a grid-sized table, so it is
  defined for any `(grid_h, grid_w)` and needs no resampling when `p` changes.
  It is load-bearing: masked patches of one channel share the mask token and
  channel embedding, so position is what makes their predictions differ (MAE's
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
* **Spatial pooling to fixed size latent.** Standard pre-norm Q/K/V/O
  transformer blocks run over the flattened real tokens
  (`[present locations, D]`, the fusion output as-is) with fixed-count memory
  tokens. A second PMA
  pooler then applies **spatial pooling to fixed size latent**: `L =
  n_latents` learned queries pool the per-location tokens to `[B, L, D]`.
  `L` is a hyperparameter, so the encoder output and the decoder
  KV do not grow with `C` or with the patch count `N`: varying `N` changes
  encoder compute, not the representation size. This is a hybrid — token-level
  encoder attention followed by spatial pooling to fixed size latent — not the
  Perceiver's pool-before-processing architecture; the Perceiver paper does not
  ablate this placement (`references/perceiver/paper.md:531-558`).
* **Encoder and decoder KV.** The decoder KV is the fixed set of `n_memory`
  memory tokens plus the `n_latents` pooled latents, `[B·(n_memory +
  n_latents), D]` and fixed for a given model; which of those components reach
  the decoder is selectable via `decoder_kv_components` (default: both; empty =
  blind decoder, no cross-attention).
* **Decoder.** One decoder is shared across all channels. Each target query is
  `mask token + target spatial PE + target channel embedding`; queries
  self-attend within each sample, then cross-attend to the encoder's
  memory + the fixed latent set. A fully dropped channel has no encoder
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

## JEPA objective

`ChannelFusionJEPA` keeps this backbone and changes the objective from pixel
reconstruction to feature prediction. The target is the **EMA target encoder's
per-location, channel-fused token feature**, computed from the fully observed
input (all channels, all patches); the student sees a multi-block spatial
context, optionally with whole-channel dropout, and the existing decoder blocks
predict the removed target features from the pooled latents + memory. The
target is stop-grad, the target encoder is EMA-updated, and the loss is L1
(default) or L2 in feature space.

Collapse prevention is a config choice: the shipped recipe is an EMA target
encoder with stop-gradient, but the JEPA path also supports `use_ema=False`
with a **SIGReg** (LeJEPA Epps-Pulley) or **VICReg** variance/covariance
regularizer. Both regularizers act on the pooled `latents.mean(dim=1)` and are
returned as separate `jepa_loss` / `reg_loss` terms.

This keeps the JEPA target channel-permutation invariant and padding-free, and
it keeps the predictor's fractional positional PE load-bearing. It also means
the target is a *location* representation, not a per-channel one: after channel
fusion there is no per-channel feature to predict. DCP's per-channel patch mask
is therefore not reused as the JEPA mask; a multi-block location sampler and
optional HCS channel dropout replace it. The overview, the EMA/no-EMA and
regularizer matrix, and the training-loop contract are in
**[`docs/jepa.md`](docs/jepa.md)**.

```python
from fuse_channels import JEPAConfig, build_jepa

model = build_jepa(jepa=JEPAConfig(channel_dropout=True))
# Regularizer ablation:
# model = build_jepa(jepa=JEPAConfig(regularizer="sigreg", reg_weight=0.1))
# No-EMA ablation (requires a non-zero regularizer weight):
# model = build_jepa(jepa=JEPAConfig(use_ema=False, regularizer="sigreg", reg_weight=0.1))
out = model(images, ids, pixel_sizes=pixel_sizes)
out["loss"].backward()
opt.step()
model.update_target_encoder()   # no-op when use_ema=False
```

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

## Naming this stage

The stage is called **spatial pooling to fixed size latent**. The name is
literal, but each word carries weight:

* **spatial** — it pools the per-`(sample, patch)` tokens produced after
  channel fusion. It is not a second channel fusion: channel fusion already
  ran over `C` at each patch, and this runs over patches for each sample.
- **size** — the output is `[B, L, D]`: `L = n_latents` locations of width
  `D`. The *cardinality* `L` and the *width* `D` are both fixed, so the stage
  is called fixed size even though the encoder input length (`Kf`) varies.
- **latent** — used in the singular as the name of the stage, but the tensor
  is a set of `L` latents. `L = 1` gives one latent vector `[B, D]`; `L > 1`
  gives `[B, L, D]`.
* It is a **pooling step**, not a second encoder: the encoder still runs
  token-level attention over every present location. It is also not the
  Perceiver's pool-before-processing latent bottleneck, and the Perceiver paper
  does not ablate this placement.

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
    n_memory=4, n_latents=16,         # spatial pooling to fixed size latent
    num_channel_ids=32,               # size of the learnable identity table
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

# inference representation (no masking): memory tokens + the fixed latent set
mem, latents = model.encode(images, ids)
# memory: [B*n_memory, D], latents: [B, n_latents, D]
```

### Configuration and ablations

`ChannelFusionMAE` also builds from a structured `ModelConfig`, and named
ablations live in one registry so how a run differs from the default is a
reviewable diff in code:

```python
from fuse_channels import ModelConfig, SharedCore, build_model

cfg = ModelConfig(image_size=16, patch_size=8, core=SharedCore(dim=192, n_heads=6))
model = ChannelFusionMAE.from_config(cfg)   # == ChannelFusionMAE(**cfg.to_kwargs())

build_model("blind_decoder")                  # a named arm
build_model("no_memory", encoder_depth=6)     # an arm plus a flat override
```

The registered ablations, rendered by `describe_ablations()` (a test keeps
this table in sync with `config.ABLATIONS`):

| name | diff |
| --- | --- |
| `full` | The shipped model; the identity transform. |
| `no_memory` | Drop encoder memory tokens: `encoder.n_memory = 0`. |
| `blind_decoder` | `no_memory` plus an empty decoder KV: `decoder_kv_components = ()`. |
| `no_scale_cond` | Drop scale conditioning: `scale_conditioning = "none"`. |
| `grid_pe` | Use normalized grid coordinates: `pe_space = "grid"`. |

Backbone invariants (`dim`, `n_heads`, `dropout`, `adaln`) live once in
`SharedCore` and are handed to the encoder and decoder explicitly, so they
cannot drift between modules. See `docs/design.md` §10 for the split and the
ablation policy.

The JEPA path uses the same `ModelConfig` backbone and a separate `JEPAConfig`
for objective-specific knobs (feature loss, EMA momentum, block-mask geometry,
channel dropout). `build_jepa(base, jepa, **overrides)` combines them; see
[`docs/jepa.md`](docs/jepa.md).

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
fused, present = fusion(cs)              # [K_f, 192]; one row per present location

# the O(C) marker-routed ablation
gated = IdentityGatedSum(192)
gated_out, present = gated(cs)
```

### The tokenizer stage as one module

`ChannelFusionTokenizer` composes the two stages above into the single named
stage `{per-channel planes, ids} -> ([K_f, D], present, ChannelSet)`, so callers
that only need tokenization do not have to reproduce the wiring:

```python
from fuse_channels import ChannelEmbedding, ChannelFusionTokenizer

cft = ChannelFusionTokenizer.build(
    16, 8, 192, ChannelEmbedding(192, num_channel_ids=32),
    n_heads=6, patch_sizes=(4, 8, 16),
)
fused, present, cs = cft(images, ids, pixel_sizes=0.65)
```

It owns the identity table and the two primitives; scale conditioning, the
encoder, the latent pooler, and the decoder are outside it. `build`
is the convenience constructor — pass an already-built `ChannelTokenizer` and
fusion to wrap the modules of an existing model instead (that is how the stage
can be applied to `ChannelFusionMAE`'s weights without touching its state-dict
keys).

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
  Encoder, decoder, and latent pooling group by sample; fusion groups by
  `(sample, patch)` location.

## Layout

| file | contents |
| --- | --- |
| `src/fuse_channels/ops.py` | padding-free grouped attention primitives |
| `src/fuse_channels/container.py` | `ChannelSet`: flattened variable-channel token batch |
| `src/fuse_channels/identity.py` | `ChannelEmbedding`: learned per-channel-id additive table |
| `src/fuse_channels/patch_embed.py` | `ChannelTiedPatchEmbed`, `ChannelTokenizer` (content-only tokens; identity travels separately) |
| `src/fuse_channels/fusion.py` | `ChannelFusion`, `IdentityGatedSum`, `MultiheadAttentionPooling`; `_dense_fusion_reference` (private, test/viz-only) |
| `src/fuse_channels/tokenizer.py` | `ChannelFusionTokenizer`: the tokenizer *stage* — `ChannelTokenizer` + channel fusion -> `[K_f, D]` (its docstring carries the stage contract) |
| `src/fuse_channels/rope.py` | 2-D axial RoPE; metric coordinates by default, normalized-grid ablation |
| `src/fuse_channels/flexi.py` | FlexiViT PI- (pseudoinverse-) resize for the patch-embedding kernel |
| `src/fuse_channels/pos_embed.py` | fractional absolute PE plus legacy learned/sincos tables |
| `src/fuse_channels/encoder.py` | `TransformerEncoder` / `TransformerBlock` with memory tokens |
| `src/fuse_channels/decoder.py` | `ChannelAwareDecoder` / `DecoderBlock` |
| `src/fuse_channels/masking.py` | `DCPVisibilitySampler` (Eq. 1), `HierarchicalChannelSampler` (HCS), and the JEPA `MultiBlockTargetSampler` |
| `src/fuse_channels/losses.py` | `normalized_patch_mse`, `cha_maevit_recon_loss`, `feature_prediction_loss`, flat-layout `patch_offsets` / `patch_row_terms` |
| `src/fuse_channels/mae.py` | `ChannelFusionMAE` end-to-end |
| `src/fuse_channels/jepa.py` | `ChannelFusionJEPA` and `build_jepa` |
| `src/fuse_channels/regularizers.py` | JEPA collapse regularizers: `SIGReg`, `VICReg`, `NoReg` |
| `src/fuse_channels/config.py` | `SharedCore` / `EncoderConfig` / `DecoderConfig` / `JEPAConfig` / `ModelConfig`, the ablation registry, and the shared-core split contract |
| `src/fuse_channels/data.py` | `SyntheticChannelDataset` and `collate_fn` for examples/tests |
| `examples/linear_probe.py` | frozen-encoder linear probe on a synthetic channel-presence label |
| `examples/train_mae.py` | synthetic variable-channel MAE training loop |
| `examples/train_jepa.py` | synthetic variable-channel JEPA training loop (incl. EMA update) |
| `examples/benchmark_fusion.py` | fusion throughput comparison at `C = 60` |
| `docs/design.md` | design rationale, reference corrections, deviations |
| `docs/jepa.md` | JEPA target/predictor overview and ablation matrix |

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
- encoder output / decoder KV comes from spatial pooling to fixed size latent,
  not from the per-token sequence, and does not change when the patch count does
  (`test_encoder_output_is_fixed_size_across_patch_counts`)
- DCP Eq. 1 regions and HCS counts (`test_masking.py`); channels dropped by DCP
  are reconstruction targets (`test_dcp_dropped_channel_is_a_reconstruction_target`)
- the `n_latents` pooler is trained through the MAE loss and its output is the
  decoder KV alongside the memory tokens; `encode()` matches the forward latents
  (`test_latent_pooler_is_trained_by_the_mae_loss`,
  `test_forward_and_encode_share_the_latents`,
  `test_encode_returns_the_decoder_kv`)
- loss variants (`test_losses.py`) and the frozen linear-probe smoke test
  (`test_linear_probe.py`)
- single-batch MAE loss decreases (`test_overfit_single_batch`)
- JEPA path (`tests/test_jepa.py`): feature target is stop-grad, the target
  encoder sees every location, the target is channel-permutation invariant and
  independent of `C`, the EMA update matches the student weights, optional
  channel dropout runs, the feature loss reduction is exact, no-EMA +
  SIGReg/VICReg runs, and a fixed batch overfits; the MAE path still trains in
  the same suite
- regularizers (`tests/test_regularizers.py`): SIGReg is differentiable and
  prefers a Gaussian batch over a collapsed one; VICReg matches the paper
  formula and penalizes collapse; the registry and `JEPAConfig` validation are
  covered

The cost comparison is an example, not a test:
`examples/benchmark_fusion.py` reports wall-clock for attention vs.
identity-gated fusion at `C = 60`.

## Notes

* `examples/train_mae.py` trains on a synthetic variable-channel dataset; it
  does not assert an overfit threshold.
* `ChannelFusionJEPA` holds its target encoder out of the optimizer
  (`requires_grad=False`) and keeps it in eval mode. Call
  `update_target_encoder()` after `optimizer.step()`; the helper is in
  `src/fuse_channels/jepa.py` and the training contract is summarised in
  `docs/jepa.md`.
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
sampler; the decoder consumes the pooled latent set + memory and conditions on
the target channel embedding, not "unmasked channel tokens"; the latent pooler
is trained rather than inference-only; the default loss is patch-normalized MSE with ChA-MAEViT as an ablation;
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

## License

Apache License 2.0 — see [`LICENSE`](LICENSE).
