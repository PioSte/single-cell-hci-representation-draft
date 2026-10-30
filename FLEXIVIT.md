# FlexiViT + variable patch size: PE and scale conditioning

**Encoder: 2D axial RoPE · Decoder: factorized-fractional absolute PE**

With square patches, FlexiViT-style patch-embedding resizing, and an explicit
pixel-size/scale token, the intended division of labor is:

- **PE carries layout** — where a token sits in the `(row, col)` grid, in a
  coordinate system that does not change when `p` or the grid size changes.
- **Scale token carries physical scale** — how large a token is in pixels /
  µm.

This only works cleanly if the encoder PE is truly *layout-only*. Raw token
indices do **not** satisfy that: adjacent tokens are always one index apart,
whether the patch is 8 px or 32 px. Use normalized coordinates in the encoder
RoPE, or accept that raw-index RoPE also encodes sampling density.

## 0. Patch size vs pixel size: the contract

`p` is **not** a pixel size. In FlexiViT, `p` is the patch edge length counted in
pixels (`references/flexivit/paper.md:75`: `x_i ∈ R^{p×p×c}`, `s = ⌊h/p⌋·⌊w/p⌋`);
the paper never multiplies it by a physical scale. The quantities are:

| symbol | meaning | owner |
|---|---|---|
| `pixel_size` | physical size of **one pixel** (µm/px), varies per instrument | dataset metadata → `ChannelSet.pixel_size` |
| `p` | number of **pixels per patch edge**, a model hyperparameter | `patch_sizes` / FlexiViT |
| `p · pixel_size` | physical extent of **one token** (µm) — **the invariant** | design synthesis, held ≈ constant |
| `n_tokens · p · pixel_size` | physical extent of the **whole crop** (µm) | *not* an invariant; see §0.2 |

There is exactly one name for the per-pixel quantity in this repo: **`pixel_size`**
in code and prose alike (physical size of one pixel, µm/px), and it is required
metadata. Its logarithm is `log_pixel_size`. Unit spellings such as µm/px are
units, not names.

**Objective.** High-content data is acquired at different pixel sizes. To make a
token mean the same physical thing on every instrument, hold `p · pixel_size` constant
and let `p` absorb the variation:

```
p_b = round(target_um / pixel_size_b)   then snap to the configured `patch_sizes`
```

**What is held constant is the patch, not the crop.** The invariant is the
physical size of a single token (`p · pixel_size`). The physical extent of the
*whole crop* is `n_tokens · p · pixel_size` and is free to vary — it is
cell-sized in the variable-`N` mode (§0.2) and fixed only if you choose a fixed
`n_tokens`. Do not confuse the two: a design can legitimately scale the number
of tokens with the object being imaged while every token keeps the same
physical footprint.

Two units traps when ingesting `pixel_size`: some tools report nanometres or
pixels-per-µm instead of µm/px (a factor of 1000 or an inversion), and OME-TIFF
carries `PhysicalSizeX` *and* `PhysicalSizeY` separately.

**Expected data: square pixels, `pixel_size ∈ [0.4, 0.6] µm/px`.** Anisotropy is
*not* a case this design serves. The target instruments acquire square pixels, so
`pixel_size` is a scalar and `PhysicalSizeY ≠ PhysicalSizeX` means suspect
metadata, not astigmatic optics: treat a mismatch beyond a small tolerance
(warn and/or reject the image) rather than silently averaging the two. If an
anisotropic image ever does arrive, `p` is scalar and the kernel is square, so
the options are the geometric mean with an accepted aspect-ratio error, or
resampling — but no machinery is justified for it.

The 0.4–0.6 span is also why this is a narrow problem: for a 12.8 µm token `p`
goes from `12.8/0.4 = 32` px to `12.8/0.6 = 21.33 → 21` px. That is ~1.5× on
FlexiViT's axis — under one octave, i.e. a two-entry `patch_sizes` set covers the
fleet — while resampling to the finest pixel size instead costs 2.25× in area,
hence in tokens and compute. The rounding question is therefore a ≤ 1 px
question: the residual token-extent error is ≤ ~1.6 %.

Consequences that are easy to get wrong:

- **`p` must be selected, not sampled arbitrarily.** The configured
  `patch_sizes` should be the set of `p_b` implied by the `pixel_size` range in the
  training data, not a generic compute sweep. Otherwise a valid instrument maps
  to a `p` the model never saw.
- **No extrapolation.** FlexiViT explicitly does not claim it
  (`references/flexivit/paper.md:879`: performance degrades beyond trained patch
  sizes). Inference should assert `p_b ∈ trained patch_sizes`; the decoder's
  nearest-`p` head fallback is graceful degradation, not a guarantee.
- **Anisotropy is out of scope by data assumption, not by capability.**
  `PhysicalSizeX ≠ PhysicalSizeY` should be treated as bad metadata (see above),
  not as an architecture gap to close.
- **Snap direction matters.** Rounding `p` down keeps the physical token smaller
  than the target and yields more tokens; rounding up does the opposite. Prefer
  down, and record the realized `p · pixel_size` so the resulting extent error is
  auditable.
- **`p` is confounded with instrument identity.** In FlexiViT `p` is randomized
  independently of content; here it is *determined* by `pixel_size`, so the model can
  learn `p → dataset` shortcuts. Jitter the physical target extent across a
  small set within an `pixel_size` and ablate with `scale_conditioning="none"`.
- **Rejected baseline:** resample every image to a canonical `pixel_size` and keep one
  fixed `p`. This is the standard HCI/pathology approach and is simpler. It is
  rejected here because it resamples content (destroying native PSF/noise
  statistics and cost on large multiplexed panels); FlexiViT instead resizes the
  *kernel* and leaves the pixels untouched. If a single canonical `pixel_size` covers
  the instruments acceptably, that baseline is cheaper and should be preferred.
  **Status: not implemented and not yet ablated.** Nothing in `data.py`/`crops.py`
  resamples; a runnable comparison needs a `--resample-to <µm/px>` transform in
  front of the existing loader (upsample `0.6 → 0.4 µm/px`, keep `p` fixed, and
  mask out the `p`-selection entirely so the two arms see the same token extent).
  Given the 0.4–0.6 range the resampling arm costs 2.25× the patches of the native
  arm, so the honest expectation is *not* that it is more accurate — only that it
  pins the token extent exactly. It should be run before the design is called
  settled, since it is the one ablation that could falsify the FlexiViT choice.
- **PI-resize is a smooth operator across sampling density, not a physically
  faithful resampling between instruments.** Tokens at the same physical extent
  from two devices are comparable in representation space, not equal
  (`references/flexivit/paper.md:309`, Appendix I).

## 0.1 Metric PE: the phase must encode micrometres, not grid fractions

`pe_space` selects the coordinate system the encoder RoPE and decoder
fractional PE are evaluated in:

```
grid      coord = (index + 0.5) / grid_extent * ref_extent     (default, back-compat)
physical  coord = (index + 0.5) * p * pixel_size / ref_um             (metric)
```

The identity that decides this (see §1) is

```
relative RoPE phase  =  (physical separation d / window extent W) * ref_extent * theta
```

* **Fixed physical window** (`n_tokens * target_um` constant): `W` is constant,
  so `grid` coordinates are *already* metric. `physical` is then equivalent up
  to the choice of `ref_um`; nothing is gained or lost.
* **Cell-sized window** (variable `N`): `W` varies with cell size, so `grid`
  coordinates encode `d/W` — separation as a fraction of the cell. The same
  physical gap gets a different rotation in a small cell than in a large one.
  An additive scale conditioner **cannot** repair this: the rotation is applied
  to q/k before attention and a per-sample global vector does not rescale the
  phase. Only `physical` coordinates make the phase proportional to `d`.

Consequences:

- `pe_space` defaults to `"physical"` (metric). `"grid"` is the fixed-window
  ablation; there is no `"auto"`, because pixel sizes are always present.
- **Both** encoder RoPE and decoder PE must use the same frame. The decoder
  gets `target_step_um = p * pixel_size` per target so mask tokens and visible tokens
  share coordinates; mixing `physical` encoder RoPE with per-axis grid-fraction
  decoder PE is a silent bug.
- `pe_ref_um` (default `1.0`) fixes how many micrometres one coordinate unit
  is; at the default the coordinates simply are micrometres. It is a free
  physical unit, **not** derived from the crop: with cell-sized crops the crop
  extent varies, so no `ref_um = W_um / ref_extent` mapping exists (that
  identity only holds in the fixed-window case, where it is used as an
  equivalence test).
- `grid` coordinates with a variable patch count emit a warning rather than
  silently training with non-comparable position phases.

## 0.2 Variable `N` (cell-sized grids)

A cell-sized window (`plan_patches(..., n_tokens="cell")`) makes the patch
count a function of cell size: `n_tokens = ceil(cell_side_px / p)`, clamped.

This is *consistent* with the §0 contract, not in tension with it. The invariant
is the **per-token** physical extent `p · pixel_size ≈ target_um`, which is unchanged:
with constant `p · pixel_size`,

```
n_tokens = ceil(cell_side_px / p) ≈ ceil(cell_side_um / target_um)
```

is essentially instrument-independent and tracks the cell, not the acquisition.
Only the *crop* extent `n_tokens · p · pixel_size` varies, and it is meant to: it is
cell size plus at most one token of slack. So `p` absorbs the instrument, `N`
absorbs the cell, and tokens stay physically comparable everywhere.

The consequences below follow from the crop extent being variable. They are a
correctness requirement of cell-sized windows, not a cost of a trade-off that
could instead be avoided by fixing the crop:

| consequence | handling |
|---|---|
| Non-metric position phases | `pe_space="physical"` (§0.1); warning otherwise |
| Small-cell MAE degeneracy (`N=1` disables patch masking) | `cell_n_tokens(min_n=2)`; `DCPVisibilitySampler` still falls back for `N=1` |
| Ragged batches | `extract_patches` raises on mixed `N`; `_build_targets` takes the per-sample list path whenever `p` **or** `N` varies |
| Token-count gradient weighting ∝ cell area | `loss_reduce="sample"` (default) gives each cell one vote |

**Loss reduction.** `loss_reduce="sample"` (default) averages per cell;
`"token"` weights every reconstructed row equally, so a sample's weight is
`n_i / Σ n_j` with `n_i = C_i · N_i · masked_frac`. With cell-sized crops
`N_i ∝ cell area`, so token reduction lets an 8×8-token cell take ~94% of the
gradient against a 2×2-token cell, and makes each step's weighting depend on the
randomised DCP draw. Token reduction is still available, and is the natural
choice if you care about per-patch accuracy rather than per-cell representation.

Under a *fixed* window none of these bite (the crop extent is constant, so
grid-fraction PE is already metric and `N` never varies). Cell-sized windows buy
cell-proportional compute and no excess background; the price is that the four
items above become prerequisites rather than options.

**There is no upper bound on `N`.** Nothing in the model caps the patch count —
it works at `N = 10_000` and a model *constructed* with `image_size=32` happily
processes a 320 px crop, because `image_size` only sizes legacy PE tables. A
larger cell just costs more: encoder attention is `O((C·N)²)` per sample and the
fusion buffer is `[B, max_b N_b, D]`, so the real limits are memory and time,
not architecture. Consequently `plan_patches` does **not** clamp: `max_n` is an
optional budget, and a cell that exceeds it is skipped and reported, never
silently truncated. `min_n=2` is the opposite kind of bound — it *enlarges* the
window to keep the MAE patch region usable, so it loses no content.

The genuine caveats that remain are per-`p`, not per-`N`: with `head_mode="per_p"`
(the default) the decoder head is per-`p`, so every instrument's `p` must be in
`patch_sizes`; `head_mode="shared"` removes that requirement (no parameter
depends on `p`) at the cost of resampling the prediction across `p`;
`max_tokens` guards
only the random FlexiViT sampler, not cell-sized crops; and metric PE does not
extrapolate in coordinate range, so very large `N` means attention distances
beyond any trained cell — degrade it by monitoring the `N` distribution, not by
clipping.

## Status in this repo

**Implemented.** The target architecture is now the default model path:

1. **Fractional decoder PE** — `ChannelAwareDecoder(pos_embed="fractional")`
   evaluates `g_row(v) + g_col(u)` from `(r, c, n_h, n_w)`; it has no grid-sized
   table and works with non-square or changing grids.
2. **Scale conditioning** — `scale_conditioning="add"` (default) or `"adaln"`
   embeds `log(p)` and `log(pixel_size)` with Fourier features and a
   small MLP, in both encoder and decoder.
3. **Encoder RoPE** — `encoder_pos_embed="rope"` (default) applies 2-D axial
   RoPE to q/k on normalized coordinates.  Fusion no longer owns a spatial
   table (`fusion_pos_embed="none"`).  The old fixed-grid scheme is available
   as `pos_embed="learned"` / `"sincos"` with `encoder_pos_embed="none"`.
4. **FlexiViT per batch** — `patch_sizes=(...)` and `flexi_random=True` pick one
   `p` per training step; `forward(..., patch_sizes=p)` overrides it.  The
   canonical patch kernel is PI-resized, `n_patches`, masking, target
   extraction, RoPE and fractional PE are derived per step, and the decoder
   keeps one linear `dim -> p²` reconstruction head per configured patch size
   (`head_mode="per_p"`).  An unseen `p` uses the nearest configured head with
   `p_head >= p` + bilinear point-sampling and never upsamples.  `max_tokens` optionally filters the
   random sampler; explicit `patch_sizes=` overrides it.
5. **Per-sample variable `p` / `H, W`** — `forward(..., patch_sizes=[p_b])`
   passes per-sample sizes.  Tokenization, fusion grouping, RoPE, fractional
   PE and masking use per-sample grid metadata; predictions are padded to
   `max(p_b²)` with a validity mask, and the Fourier ablation is evaluated per
   patch-size group.  The fused scatter buffer is still `[B, max N_b, D]`, but
   the encoder only ever receives present locations (flattened), so attention
   is padding-free.

Tests for these properties live in `tests/test_flexivit.py`, and the original
fixed-grid invariants remain covered in the other tests.

**Open ablation: upsample-to-finest-`pixel_size` baseline.** There is exactly one
comparison the prose above promises and the repo does not yet contain: resample
every image to the finest `pixel_size` in the fleet (0.4 µm/px), fix
`p = target_um / 0.4`, and compare against native pixels with per-sample `p_b`.
It needs only a preprocessing transform plus a fixed-`p` model config; the
FlexiViT arm is the default path. Two things make it worth running despite the
rejected-baseline note: the two arms differ by 2.25× in token count (so the
comparison must be matched on physical crop extent, not on `N`), and the
prediction to test is that the gain from exact `p·pixel_size` is smaller than the
loss from interpolated, PSF-blurred input.

---

## 1. Encoder — 2D axial RoPE on normalized coordinates

### Why RoPE

- It has no learned position table, so there is no PE table to resize when
  FlexiViT changes `p` or the grid. This removes FlexiViT’s PE-resampling step
  entirely.
- It injects position through query/key rotation. RoPE encodes an absolute
  phase per token but makes the attention logit depend on the relative
  displacement, which is the useful signal in an MAE encoder that only sees a
  sparse visible subset.
- It factorizes per axis, so non-square grids work naturally.
- It extrapolates to unseen grid extents without a lookup table.

RoPE is *relative in attention*, not “incapable of representing absolute
position.” RoFormer describes it as encoding absolute position with a rotation
matrix while producing explicit relative-position dependence in the dot
product (`references/roformer/paper.md:23`).

### Implementation

For a head dimension `d`, split it into two equal halves:

```text
d_row = d_col = d / 2          # requires d % 4 == 0
```

Use `d_row / 2` rotation pairs for the row axis and `d_col / 2` pairs for the
column axis.

Define normalized coordinates with a fixed reference extent `S` (e.g.
`S = 32` or `S = 64`):

```text
r_norm = (r + 0.5) / n_h * S
c_norm = (c + 0.5) / n_w * S

r in [0, n_h), c in [0, n_w), n_h = H / p, n_w = W / p
```

Then, for `i = 0 .. d_row/2 - 1`:

```text
theta_i = base ** (-2 * i / d_row)
rotate pair (2i, 2i+1) of the row half by r_norm * theta_i
```

and for `i = 0 .. d_col/2 - 1`:

```text
theta_i = base ** (-2 * i / d_col)
rotate pair (d_row + 2i, d_row + 2i + 1) by c_norm * theta_i
```

Recommended defaults:

- `base = 100` for vision grids, with `100–1000` as the useful tuning range.
  Vision RoPE typically uses a smaller base than language RoPE because token
  grids are tens to a few hundred per axis, not tens of thousands.
- The same frequency schedule can be shared by both axes; separate schedules
  are optional.
- If aspect ratio is meaningful, normalize both axes by the same
  `max(n_h, n_w)` instead of normalizing each axis independently to `[0, S]`.
  Per-axis normalization removes aspect ratio; common-scale normalization
  preserves it.

### Why normalized, not raw indices

Raw-index RoPE needs no resampling, but it is not invariant to `p` or grid
size. At `p = 8`, two adjacent tokens are 8 px apart; at `p = 32`, the same
index delta is 32 px. The angular difference per index is identical, so the
attention sees different physical geometry. Normalized coordinates make the
encoder PE layout-only and leave physical scale to the scale token.

If raw token indices are used deliberately, state that the encoder PE also
encodes sampling density and the scale token is partly redundant.

### Axial vs mixed frequencies

Axial 2D RoPE cannot represent diagonal directions because each frequency
depends on only one axis. RoPE-Mixed learns `(theta_x, theta_y)` per head/layer
and generally performs better across resolutions and tasks
(`references/rope-vit/paper.md:27`, `:173-199`, `:265`). Start with axial for
simplicity; treat mixed RoPE as the stronger upgrade, not as an equivalent
variant.

---

## 2. Decoder — factorized-fractional absolute PE

The decoder receives **mask tokens** that must be told where they belong. MAE
adds positional embeddings to all decoder tokens for exactly this reason
(`references/mae/paper.md:103`). Keep an explicit additive absolute PE on the
decoder input.

For every patch position `(r, c)`, visible and masked:

```text
u = (c + 0.5) / n_w     in (0, 1)
v = (r + 0.5) / n_h     in (0, 1)

PE_dec(r, c) = g_row(v) + g_col(u)
```

`g_row` / `g_col` can be:

- fixed multi-frequency sinusoids,
- a small learned MLP evaluated on the scalar coordinate,
- interpolation into a modest learned 1-D table indexed by the fraction.

Add `PE_dec` to **all** decoder tokens before the transformer blocks. The
decoder PE is a function of fractional position, not a lookup on a fixed grid,
so it is defined for any `n_h × n_w` and does not need resampling when `p`
changes.

Notes:

- `g_row(v) + g_col(u)` is separable. If diagonal interactions matter, use a
  small MLP on `(u, v)` or a learned 2-D table instead of the sum.
- `ids_restore` / unshuffle logic is unchanged; derive `(r, c)` from the
  dynamic `n_h, n_w`.
- Keep fractional coordinates, not pixel coordinates, so the grid interface is
  identical across patch sizes and densities.
- If FlexiViT keeps `n_h · n_w` near a target token count, `u, v` stay in a
  stable regime; the model interpolates rather than extrapolates.

Absolute PE in the decoder is a deliberate design choice. RoPE can also be
applied to decoder queries/keys with assigned positions, but that provides
relative offsets rather than an additive absolute anchor. Do not present
“RoPE in the decoder” as categorically wrong; treat it as a different
experiment.

---

## 3. What FlexiViT resamples

FlexiViT’s patch-embedding method is **PI-resize** (pseudoinverse resize), not
plain bilinear resize. Bilinear resizing is the baseline used for standard ViT
and for the position-embedding table; PI-resize is the proposed operation for
patch-embedding weights (`references/flexivit/paper.md:105`, `:168`, `:171`,
`:173`).

| Component | Resample with `p`? |
|---|---|
| Patch-embedding kernel `p×p` | **Yes** — PI-resize; bilinear is a simpler baseline |
| Decoder reconstruction head | **Not defined by FlexiViT** — this repo uses per-`p` linear heads; an unseen `p` uses the nearest larger head with bilinear point-sampling, and `head_mode="canonical"` is the ablation |
| Encoder RoPE | **No table / no resampling** — use normalized coordinates for layout-only PE |
| Decoder fractional PE | **No table / no resampling** |
| Decoder PE table | **No table** (there isn’t one) |
| Channel identity | **Not added to the patch token** — routing bias + pooled skip (see §3.1) |

FlexiViT assumes patches tile the image. For arbitrary `H, W`, this repo uses
FlexiViT's own floor contract: keep `floor(H/p)·floor(W/p)` patches and discard
the border smaller than `p`. Padding plus a validity mask remains an optional
future path, not the default.

**Intensity is handled by the resize operator, not by a scalar.** The reason
FlexiViT prefers PI-resize is exactly the token-norm change that plain resize
introduces (`references/flexivit/paper.md:135`); no explicit gain such as
`(p/p₀)²` is applied. PI-resize satisfies `<B x, P ω> = <x, ω>` for `p ≥ p₀`.
That equality depends on `B` being the true bilinear operator, i.e.
`B @ vec(x) == F.interpolate(x)`; this is enforced by
`test_bilinear_resize_matrix_is_the_real_resize_operator`, and the invariant
itself by `test_pi_resize_preserves_token_intensity_when_upsampling`. Token RMS
is bounded across `p` by `test_pi_resized_kernel_keeps_token_scale_bounded`.

The decoder reconstruction head is the part where `p` touches a parameter shape.
An MAE-style head maps `d -> p²`; changing `p` changes both the output basis and
the pixel target. Three modes:

* `head_mode="per_p"` (default, exact): one linear head per configured `p`. The
decoder is discarded after MAE, so the extra parameters cost nothing at
deployment. An unseen `p` uses the nearest larger per-`p` head with bilinear
point-sampling and never upsamples. Bilinear there is point-sampling
(`align_corners=False`, no antialias), which matches the discrete `p×p` *target*
of that same `p`.
* `head_mode="shared"` (`p`-free): one canonical `dim -> p0²` head plus a fixed
linear operator to the target grid. Resampling commutes with the linear head, so
resampling the prediction and resampling the head's output rows are the same
map. Because `p · pixel_size` is held constant, a smaller `p` is a genuinely
coarser sampling of the *same* field, so the physically consistent cross-`p`
relation is a low-pass: `head_resample="area"` (default) box-averages when
shrinking and uses bilinear when growing. `"bilinear"` is the same-`p`
point-sampling operator extended across `p` (the ablation arm).
* `head_mode="canonical"`: the legacy single head, bilinear, no upsampling.

PI-resize is *not* obviously right for the head. Its justification is that it
best preserves the inner product between a resampled *input* patch and the
kernel; the head's `p²` axis is an *output*, and `P = (B^T)^+` is a minimum-norm
least-squares inverse, not a low-pass -- it keeps detail and can ring, where
`"area"` blurs. It is therefore exposed as `head_resample="pi"` (the exact
FlexiViT operator applied along the output axis) for the ablation, not as the
default. `p0` is an **explicit** hyperparameter (`head_patch_size`), not derived
from `patch_sizes`: deriving it would silently change the head's shape when the
fleet changes, which is exactly what the `p`-free head is for.
`canonical_patch_size_for(patch_sizes)` suggests the log-space median so neither
resampling direction is unnecessarily large; the other two modes keep
`p0 = max(patch_sizes)` because they cannot upsample.

### 3.1 Channel identity must not ride on the patch token

PI-resize keeps the *patch kernel* well scaled, but the token magnitude still
depends on `p` because a real `p`-patch is not the resized `p₀`-patch (see the
mixture of `x ~ N(0,I)` and real content above). That is harmless on its own —
the fusion LayerNorm is scale-invariant — but it becomes a confound if a
**fixed-scale vector is added to the token before that LayerNorm**. ChannelViT's
identity embedding is exactly such a vector, and in an HCI panel the same
marker can appear at 8 px in one dataset and 40 px in another, so the
identity/content ratio would vary with acquisition scale.

The shipped design therefore **decouples identity from content** while keeping
what ChannelViT and Panopticon establish:

- `ChannelSet.tokens` is content only; the channel embedding travels separately
  (`cs.channel_embed`) and is never summed into the token.
- Identity still routes the channel attention: a per-head bias
  `identity_bias(e_c)` is added to the per-location logits. This is the
  scale-free part of adding `e_c` to the keys, which is how Panopticon uses its
  spectral embeddings (`references/panopticon/paper.md:33,45,94`).
- Identity still reaches the representation: the attention weights pool an
  identity vector `Σ_c a_c e_c`, added back as a learned skip after the content
  projection. Removing or sharing identity is catastrophic in ChannelViT
  (Table 10, `references/channelvit/paper.md:670-688`), and this preserves it
  per channel.
- Because LayerNorm is invariant to positive rescaling, the fused token is
  invariant to content magnitude up to LayerNorm ε on vanishing content. Empty
  patches still map to identity (so “marker absent here” survives); they do not
  get amplified into noise.
- No per-`p` gain or per-token normalization is needed. The earlier
  candidate — a scalar `g_p` calibrated from `‖P_p ω‖` — is unnecessary and
  only partial: the residual drift is content SNR, not the operator, and
  Token-LN would remove the empty/signal distinction.

Tests: `test_fusion_is_content_scale_invariant`,
`test_identity_routing_is_patch_size_invariant`,
`test_zero_content_is_identity_only`,
`test_padding_free_matches_dense_reference` in
`tests/test_identity_decoupling.py`.

---

## 4. Pixel-size / scale token

The dozen names this used to carry are gone; use the code's names in prose too:

```text
log_p           = log(p)                        # token size in pixels / sampling density
log_pixel_size  = log(pixel_size)               # physical size of one pixel (µm/px)
log_token_um    = log_p + log_pixel_size        # physical token extent (µm)
```

`log_token_um` is a *derived* feature — it is the sum of the other two, so feeding
all three gives the MLP no information it cannot form itself. Feed **two**, and
choose deliberately:

- If only patch size varies and `pixel_size` is fixed, `log_p` alone tells the
  model the current token density; the missing constant is absorbed by the bias.
- If both vary, feed `log_p` and `log_pixel_size` (`log_token_um` is redundant
  with them), or feed `log_token_um` and `log_pixel_size` if you would rather
  have the physically meaningful combination explicit.
- If the dataset has no `pixel_size` metadata, feed `log_p` and say so: the
  scalar is `log_p`, not physical scale.

The shipped conditioner feeds `log_p` **and** `log_pixel_size`; the derived
`log_token_um` is not supplied because it is exactly their sum.

**`pixel_size` is required metadata, not an optional field.** Every image has a
physical pixel size, so there is no "unknown" state to encode: no presence flag,
no filled placeholder, and no way for an absent value to be silently read as
`1.0` µm/px. `ChannelFusionMAE.forward`/`encode` raise if no pixel size is
supplied, and the loader must provide one per sample. An earlier design carried
a `has_pixel_size` flag to disambiguate "absent" from "1.0"; that state no longer
exists, so the flag is gone.

Embed it with Fourier features and a small MLP:

```python
e_scale = MLP(fourier(s))     # e.g. sinusoidal(s) -> 2-layer MLP
```

Injection options:

- **Cheap baseline: add `e_scale` to every token in both encoder and decoder.**
  This is global additive conditioning, like a timestep embedding. Put it in
  both ends so the decoder does not reconstruct scale-normalized content
  without knowing the physical size.
- **Stronger: AdaLN / FiLM.** Use `e_scale` to modulate `gamma, beta` in the
  encoder/decoder LayerNorms. This is a stronger conditioner, but the additive
  path is not erased by pre-LN: LayerNorm removes only the mean/all-ones
  component of an additive vector, and the unnormalized residual stream still
  carries it. `adaln` is zero-initialized and starts as a no-op. Keep `add` as
  the default until a real scale-sensitivity measurement says otherwise.
- **Alternative: prepend a dedicated scale token** that attends normally. Use
  this if the model should be able to query scale rather than only be biased
  by it.

---

## 5. Concrete wiring

```python
# ---- per cell ----
p = choose_patch_size(H, W)          # e.g. argmin |N_target - (H//p)*(W//p)|
n_h, n_w = H // p, W // p            # square patches; mask/crop borders if needed

# pixels -> tokens
x = patchify(cell, p)                # [N, p*p*C]
x = patch_embed_pi_resize(x, p) + e_scale

# ---- encoder ----
r, c = grid_coords(n_h, n_w)         # row-major, matching Conv2d.flatten(2)
r_norm = (r + 0.5) / n_h * S
c_norm = (c + 0.5) / n_w * S

h = x[ids_keep]
h = transformer_encoder(
    h,
    rope_2d(r_norm[ids_keep], c_norm[ids_keep]),
)

# ---- decoder ----
z = scatter(h, ids_restore, fill=mask_token)          # [N, d]
z = z + fractional_pe(r, c, n_h, n_w) + e_scale       # absolute anchor
pixels_hat = decode(z, p)                             # d -> p*p*C
```

For a channel-aware MAE, `C` is per-target-channel or per-sample depending on
the decoder design; the same PE/scale logic applies to the target channel’s
patch tokens.

---

## 6. Pitfalls

- **Don’t call raw-index RoPE “patch-size invariant.”** It needs no
  resampling, but it encodes sampling density. Use normalized coordinates if
  the encoder PE must be layout-only.
- **Don’t scale raw RoPE indices by `p`** and then claim scale is delegated to
  the scale token. Choose one convention: raw indices (density-aware) or
  normalized coordinates (layout-only).
- **Don’t use plain bilinear resize as “the FlexiViT method.”** PI-resize is
  the proposed patch-embedding resampling; bilinear is the baseline.
- **Don’t forget `d % 4 == 0`** for axial RoPE; each axis needs `d/4`
  frequency pairs.
- **Don’t use base 10000 by default for vision grids.** Start around 100 and
  tune in `100–1000`.
- **Don’t mix axial and mixed frequencies accidentally.** Pick one and make
  the implementation explicit.
- **Don’t assume RoPE is useless in the decoder.** It provides relative
  offsets; absolute fractional PE provides the additive anchor. Prefer the
  latter for MAE mask tokens, but keep the distinction accurate.
- **Don’t resample the decoder head naively.** FlexiViT defines input patch
  resampling, not MAE output-head resampling. Prefer per-`p` heads; if you use
  a nearest-head fallback, guard against upsampling.
- **Don’t feed only `log p` when physical µm/pixel matters.** Use `log p`,
  `log(µm/px)`, or both.
- **Don’t assume pre-LN erases an additive scale vector.** It removes only the
  mean component; the residual stream carries the vector. Use AdaLN/FiLM if you
  need stronger conditioning, not because `add` is provably lost.
- **Don’t add the channel identity to the patch token before the fusion
  LayerNorm.** A fixed-scale vector summed into a variable-scale token makes the
  identity/content ratio a function of `p`; use a routing bias + output skip
  (§3.1).
- **Don’t “fix” the identity drift with per-token normalization (Token-LN).**
  It makes vanishing patches full-norm noise, which destroys the empty/signal
  distinction that sparse HCI markers rely on.
- **Don’t forget dynamic grid metadata.** The decoder needs `(r, c, n_h, n_w)`,
  not only a flat patch index.

---

## 7. Repo integration notes

Implementation map for the target architecture:

- `src/fuse_channels/rope.py` — `axial_rope_cache` / `apply_rope`; the encoder
  applies them to q/k in every block.
- `src/fuse_channels/pos_embed.py` — `FractionalSpatialPE` and
  `get_fractional_sincos_pos_embed`; the decoder uses them by default. The old
  learned/sine-cosine tables remain for the `pos_embed="learned"` / `"sincos"`
  ablation.
- `src/fuse_channels/flexi.py` — `pi_resize_patch_embed`; `ChannelTiedPatchEmbed`
  keeps one canonical `Conv2d` kernel and PI-resizes it per patch size.
- `src/fuse_channels/patch_embed.py` — per-sample patch sizes, floor/crop
  border handling, grid metadata, and row/column coordinates on `ChannelSet`.
- `src/fuse_channels/container.py` — `grid_h`, `grid_w`,
  `n_patches_per_sample`, `patch_row`, `patch_col`, `patch_size`, `pixel_size`.
- `src/fuse_channels/decoder.py` — fractional target PE, optional AdaLN, and
  per-`p` reconstruction heads with a nearest-head bilinear fallback for unseen
  sizes.
- `src/fuse_channels/masking.py` — `DCPVisibilitySampler` accepts a scalar or
  per-sample patch count at call time.
- `src/fuse_channels/mae.py` — resolves patch sizes (with optional
  `max_tokens`), builds scale features, routes RoPE/conditioning through the
  encoder and decoder, and pads mixed-p targets with a validity mask.

## 8. References

- FlexiViT: `references/flexivit/paper.md` — patch-size randomization,
  PI-resize, PE-table resizing.
- RoFormer: `references/roformer/paper.md` — original 1-D RoPE, rotary
  formulation, relative-position property.
- RoPE-ViT: `references/rope-vit/paper.md` — 2-D axial vs mixed RoPE, vision
  frequency schedules, resolution extrapolation.
- MAE: `references/mae/paper.md` — sparse encoder, mask tokens, absolute PE on
  both encoder and decoder.

No canonical paper is provided for the “pixel-size token” as described here;
the scale conditioning is a design synthesis. FlexiViT covers patch-size
flexibility, not physical µm/pixel conditioning.
