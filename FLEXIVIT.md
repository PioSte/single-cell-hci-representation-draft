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

## Status in this repo

**This document describes a target architecture, not the current behavior of
the repo.** The current implementation is a fixed-grid, fixed-patch-size,
no-scale-metadata model:

- one `image_size` and `patch_size` per model; `ChannelSet.n_patches` is a
  scalar and fusion returns a dense `[B, N, D]` tensor;
- absolute post-fusion PE is learned or fixed sine-cosine, shared with the
  decoder `spatial_pos` table;
- the decoder head is a fixed `Linear(dim, patch_size²)`;
- masking is constructed with one fixed `n_patches`;
- there is no µm/pixel metadata, scale token, encoder RoPE, or FlexiViT
  patch-embedding resize yet.

The migration should be staged rather than implemented as one change:

1. **Fractional decoder PE at the fixed grid** — replace the decoder table
   lookup with a function of `(r, c, n_h, n_w)`; no FlexiViT yet.
2. **Scale conditioning** — add per-sample and/or per-channel pixel-size
   metadata, then additive or AdaLN/FiLM conditioning.
3. **Encoder RoPE at the fixed grid** — apply RoPE to post-fusion tokens, move
   position out of the fusion modules, and split encoder/decoder PE config.
4. **FlexiViT per batch** — one `p` per training step, with fixed `H, W`; make
   `n_patches`, masking, target extraction, and the decoder head dynamic per
   step. This is the practical first FlexiViT milestone.
5. **Per-sample variable `p` / `H, W`** — flatten fusion output, use per-sample
   grid metadata and masking, and accept that this is a larger redesign than
   the PE changes in this document.

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
| Decoder reconstruction head | **Not defined by FlexiViT** — use a canonical output grid, a conv/MLP conditioned on `p`, or carefully resample the output basis |
| Encoder RoPE | **No table / no resampling** — use normalized coordinates for layout-only PE |
| Decoder fractional PE | **No table / no resampling** |
| Decoder PE table | **No table** (there isn’t one) |

FlexiViT assumes patches tile the image. For arbitrary `H, W`, use
padding/cropping or dynamic `p` selection and mask padded cells.

The decoder reconstruction head is the open part. An MAE-style head maps
`d -> p²·C`; changing `p` changes both the output basis and the pixel target.
Resampling that linear weight is possible but is not the same operation as
PI-resizing an input patch kernel. Prefer a fixed canonical token grid,
unpatchify + resize, or a convolution/MLP decoder conditioned on `p`.

---

## 4. Pixel-size / scale token

Define the scale scalar explicitly. These are different quantities:

```text
s_p    = log(p)                    # token size in pixels / sampling density
s_phys = log(µm per pixel)         # physical pixel size
s_tok  = log(p * µm_per_px)        # physical token extent
```

- If only patch size varies and physical pixel size is fixed, `log p` is
  enough to tell the model the current token density.
- If both patch size and µm/pixel vary, one scalar cannot encode both. Feed
  both `s_p` and `s_phys`, or feed `s_tok` plus `s_phys`.
- If the dataset has no µm metadata, call the scalar what it is: `log p`, not
  physical scale.

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
  encoder/decoder LayerNorms. This avoids the risk that a per-token constant
  added before LayerNorm is largely normalized away.
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
  resampling, not MAE output-head resampling.
- **Don’t feed only `log p` when physical µm/pixel matters.** Use `log p`,
  `log(µm/px)`, or both.
- **Don’t add a constant scale vector before LayerNorm and expect strong
  conditioning.** Use AdaLN/FiLM if scale is important.
- **Don’t forget dynamic grid metadata.** The decoder needs `(r, c, n_h, n_w)`,
  not only a flat patch index.

---

## 7. Repo integration notes

Current state:

- `src/fuse_channels/pos_embed.py` supports only `"learned"` and `"sincos"`;
  `"rope"` is not implemented.
- `tests/test_pos_embed.py:106` currently asserts that `"rope"` is rejected.
- `ChannelTiedPatchEmbed` fixes `image_size` and `patch_size` at construction
  (`src/fuse_channels/patch_embed.py:26-32`); FlexiViT needs dynamic `p`,
  dynamic `n_h, n_w`, and PI-resized patch weights.
- `ChannelAwareDecoder` uses `self.spatial_pos[target_patch]`
  (`src/fuse_channels/decoder.py:149`); fractional PE needs grid coordinates
  instead of a flat table lookup.
- Fusion currently adds a post-fusion spatial table. If encoder RoPE is added,
  decide whether to remove or replace that table so position is not injected
  twice.

---

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
