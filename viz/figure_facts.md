# Figure facts — shapes and semantics drawn by the L0–L9 figures

Single source of truth so every figure agrees with `src/fuse_channels/`.
All shapes below are read from the code, not invented. Notation:

| symbol | meaning | code |
|---|---|---|
| `B` | batch size (samples) | `len(images)` |
| `C_b` | channel count of sample `b` (variable, order-free) | `images[b].shape[0]` |
| `ΣC` | total channels across the batch | `sum(C_b)` |
| `H, W` | image side (square) | `image_size` |
| `P` | canonical patch size (kernel stored at `P×P`) | `patch_size` |
| `p_b` | runtime patch size for sample `b` | `patch_sizes[b]` |
| `N_b` | patches per channel for sample `b`, `= floor(H/p_b)·floor(W/p_b)` | `n_patches_per_sample[b]` |
| `N` | batch-maximum patch count, `= max_b N_b` | `ChannelSet.n_patches` |
| `c` | canonical reconstruction-head grid, `= head_patch_size or max(patch_sizes)` (only `head_mode="shared"`/`"canonical"`) | `PatchReconstructionHead.canonical_patch_size` |
| `µ` | `pixel_size`, micrometres per pixel (required metadata) | `pixel_size` |
| `D` | token/embedding dim | `dim` |
| `M` | fixed memory/register tokens per sample | `n_memory` |
| `L` | latents per sample from spatial pooling to fixed size latent | `n_latents` |
| `T` | real channel-patch tokens in a flattened token set (`T = V` after masking) | `cs.tokens.shape[0]` |
| `V` | number of **visible** channel-patch tokens (`≤ Σ_b C_b·N_b`) | `keep.sum()` |
| `Kf` | number of present `(b,n)` locations (`≤ B·N`) | `present.sum()` |
| `Kt` | number of masked channel-patches (decoder targets) | `~visible` nonzero |
| `C_u` | number of unique channel instances in the batch | `unique(channel_group_id)` |

The fusion output is one token per location, so there is **no query-count
axis** in the fused tensor. The only learned-query set in the model is the
latent pooler's; `ChannelFusion` has no query count. `L` always means the
latent count; encoder depth is written `depth` and encoder block count as
`encoder.depth`.

## Shipped defaults — `config.py::DEFAULT_CONFIG`

These are the values a figure may print; anything else is an ablation. Read
them from the config, do not hard-code a second copy in prose.

| field | default |
|---|---|
| `image_size`, `patch_size` | `16`, `8` (canonical kernel `P`) |
| `dim`, `n_heads`, `dropout` | `192`, `6`, `0.0` |
| `encoder.depth`, `n_memory` | `4`, `4` (`M`) |
| `decoder.depth` | `2` |
| `n_latents` | `16` (`L`) |
| `decoder_kv_components` | `("memory", "latents")` |
| `pos_embed` (encoder / decoder / fusion) | `"rope"` / `"fractional"` / `"none"` |
| `pe_space` | `"physical"` (µm coordinates) |
| `fusion`, `fusion_attn_dim` | `"attention"`, `None` → `3D` |
| `head_mode`, `head_resample` | `"per_p"`, `"area"` |
| `scale_conditioning` | `"add"` (Fourier `[log p, log pixel_size]`) |
| `p_patch`, `p_channel`, `patch_ratio` | `0.5`, `0.5`, `0.75` |
| `mode_sampling`, `min_visible_channels` | `"batch"`, `1` |
| `loss`, `loss_reduce` | `"mae"`, `"sample"` |
| `flexi_random`, `pixel_size` | `True`, `None` (per-call metadata is required) |

`IdentityGatedSum`, learned/sincos tables, `attn_dim=D`, `head_mode`
`"shared"`/`"canonical"`, `scale_conditioning="adaln"`,
`pe_space="grid"` and the named registry ablations are **not** the default
path and are not drawn.

## Tokenizer — `patch_embed.py` (L3)

- `images`: list of `[C_b, 1, H, W]` → `torch.cat` → `[ΣC, 1, H, W]`.
- `ChannelTiedPatchEmbed.proj = Conv2d(1, D, kernel_size=P, stride=P)` —
  **one shared kernel for every channel** (tied, not per-channel), with the
  kernel **PI-resized** (FlexiViT) when a non-canonical patch size is used.
  It has **no positional embedding and no `spatial_pos` parameter**.
- `feat = proj(x).flatten(2).transpose(1,2)` → `[ΣC, N, D]`; flatten →
  `[ΣC·N, D]`. The optional DCP **visible** mask is applied to rows →
  `[V, D]`; the conv is evaluated on every patch, and the dropped rows are
  simply not selected. The kernel is resized only when `p != P` (with the
  shipped single canonical `P` it is the identity).
- `ChannelTokenizer` looks up the channel embedding
  `ChannelEmbedding.embed = nn.Embedding(num_channel_ids, D)`, indexed by the
  dataset-level `channel_id` (not the panel position `channel_idx`), and
  produces a `ChannelSet`:
  - `tokens`: `[V, D]`, **content only** (`feat`); identity is never added here;
  - `sample_idx`, `channel_idx`, `patch_idx`, `channel_id` (`[V]`);
  - `channel_embed`: `[V, D]` (the per-channel identity vector, kept separate
    so the fusion can use it as a routing bias and an output skip);
  - `channel_group_id`: `[V]` global channel instance in the concatenated
    batch (constant across a channel's patches), used by `IdentityGatedSum`
    to evaluate its MLP once per channel.
- Spatial position is **not** added here. In the FlexiViT target the encoder
  owns layout through 2-D axial RoPE; fusion adds no spatial table. The legacy
  learned/sincos post-fusion table is opt-in (`pos_embed="learned"`).

## FlexiViT / variable patch size — `flexi.py`, `patch_embed.py`, `mae.py`, `decoder.py` (L9)

- `ChannelTiedPatchEmbed.proj.weight` is stored at the canonical `P×P` size.
  For `p != P`, `_project` calls `pi_resize_patch_embed(weight, p)`:
  `P_resize = pinv(B^T)`, where `B` is the bilinear resize operator on the
  `P²→p²` patch grid, then convolves with `stride=p`. The bias is shared.
- `ChannelFusionMAE._resolve_patch_sizes`:
  - explicit `forward(..., patch_sizes=...)` → int (same `p` for the batch) or
    one `p_b` per sample;
  - legacy learned/sincos table PE → fixed canonical `P`;
  - otherwise, if more than one `patch_sizes` is configured and
    `flexi_random=True` in training → one `p` sampled per step/batch;
  - otherwise → canonical `P`.
- The fleet supplies the pixel sizes: `p_b = max(1, floor(target_um / pixel_size_b + 0.5))`
  (`crops.patch_size_for_pixel_size`), and `pixel_size_b` must be a member of
  `fleet.FLEET_PIXEL_SIZES` at ingest. There is no snap of `p` and no
  "unknown pixel size" state.
- Each sample derives `N_b = floor(H/p_b)·floor(W/p_b)`,
  `grid_h = H//p_b`, `grid_w = W//p_b`; the border smaller than `p_b` is
  cropped. The masker uses `[C_b, N_b]`, targets use `p_b`, and
  `ChannelSet.n_patches = max_b N_b`.
- No positional table is resized: the encoder's `2-D axial RoPE` uses metric
  token-centre coordinates `(index + 0.5)·p·pixel_size` (default
  `pe_space="physical"`; normalized grid fractions are the `grid_pe`
  ablation), the decoder's factorized fractional PE uses the same metric
  `(r + 0.5)·p·pixel_size / pe_ref_um` (`grid_h/grid_w` enter only the grid
  fraction ablation), and `ChannelFusion` adds no spatial table.
- `scale_conditioning="add"` (default) embeds `[log(p), log(pixel_size)]` with
  Fourier features + MLP and adds it to encoder tokens and to the decoder
  target channel embedding before `target_proj`.
- The reconstruction head keeps one `Linear(D, p²)` per configured `p`
  (`head_mode="per_p"`, default), so a configured `p` is exact. An unseen `p`
  falls back to the nearest configured head with `p_head >= p` and bilinear
  point-sampling, and refuses to upsample. `head_mode="shared"` (one p-free
  head at `c`, resampled both ways) and `"canonical"` (legacy single head,
  bilinear, no upsampling) are ablations.
- The head output is the flat `[Σ p²]` pixel array of `losses.patch_offsets`:
  row `i` is `p_i²` contiguous pixels. A mixed per-sample `p` batch needs no
  padded block and no validity mask.

## Masking — `masking.py` (L5)

- `DCPVisibilitySampler(n_patches, p_patch, p_channel, patch_ratio,
  min_visible_channels, max_masked_channels_frac, mode_sampling)` implements
  ChA-MAEViT Eq. 1. A scalar `s ~ U(0,1)` picks the region **per batch**
  (`mode_sampling="batch"`, default, so one region per training step) or per
  sample (`"sample"`):
  - `s < p_patch` → **patch** region;
  - `s < p_patch + p_channel` → **channel** region;
  - otherwise → **combined** region (`patch_mask | channel_mask`).
- **Patch region**: per channel, a **fixed count**
  `floor(N · patch_ratio)` of spatial positions is masked, drawn independently
  per channel with `randperm`.
- **Channel region / HCS**: `HierarchicalChannelSampler` draws the number of
  *visible* channels `m ~ U{min_visible_channels..C}` and then a uniform
  `m`-subset. The mask is constant down each dropped channel's rows. DCP's
  channel and combined regions reuse this sampler — there is one unified
  channel-subset mechanism, not a second channel-dropping rule.
- Guards: at least one token stays visible, and at least
  `min(min_visible_channels, C)` channels keep content (satisfied by HCS by
  construction; resampling is a fallback). The guard never reveals a whole
  channel that HCS dropped. `last_modes` records the region actually applied.
- Dropped channels stay in `images` and become reconstruction targets. This is
  why HCS and DCP channel masking induce the same visible-count distribution.
- Masked ⇒ **omitted row**. Never zeroed, never padded, no `-inf` attention
  mask. `p_patch=0, p_channel=1` is HCS-only, `p_patch=1, p_channel=0` is
  patch-only, and `p_patch=p_channel=0` selects the combined region.

## Channel fusion — `fusion.py` (L2, L4)

Both shipped fusions consume a `ChannelSet` (real tokens only) and emit
`fused: [Kf, D]`, one row per present location -- absent locations own no row.
The default target adds no positional table; the legacy
`pos_embed="learned"`/`"sincos"` ablation adds `spatial_pos: [1, N, D]`.
They also return `present: [B, N]` (a location exists if it had at least one
visible channel token).

### `ChannelFusion` (default, single-query PMA)

- `group_id = cs.location_group_ids() = sample_idx * N + patch_idx` → one
  group per `(b,n)` location; `n_groups = |unique(group_id)| = Kf`.
- `tok = norm_kv(tokens)[order]` (LayerNormed channel tokens).
- `k = k_proj(tok)`, `v = v_proj(tok)` into `attn_dim = 3D` by default
  (Panopticon's `D_attn = 3D`); `attn_dim=D` is the parameter-matched ablation.
- `q = q_proj(norm_q(queries))` with `queries: [1, 1, D]` — **exactly one
  learned query, shared by every sample and every spatial location** — expanded
  to `[Kf, 1, H, Dh]`. There is no `n_queries` argument and no multi-query path.
- Channel identity is a **routing bias and an output skip**, never added to
  `tokens`: `identity_bias: Linear(D, n_heads)` adds a per-head bias to the
  channel logits before the softmax; `identity_skip: Linear(D, D)` adds a
  residual of the attention-pooled `channel_embed` after `proj_out`.
- `grouped_pool_attention(q, k, v, lookup, Kf, logit_bias=...)` →
  `[Kf, 1, H, Dh]` plus the channel weights (**separate learned `W_Q/W_K/W_V`
  projections**, Panopticon's channel cross-attention recipe).
- `proj_out: Linear(D_attn, D)` maps back → `[Kf, D]`, then
  `+ identity_skip(Σ_c w_c·z_c)`; add `spatial_pos` only in legacy mode.
- Cost: `O(C·D·D_attn)` per `(b,n)` location plus `O(D·D_attn)` for
  `proj_out` — **`O(C)` scaling**, not a dense `C×C` per-location block.
  ChannelViT and ChAda-ViT attend over the full `C·N` sequence
  (`O((C·N)²)`); only ChAda-ViT pads and masks.

### `IdentityGatedSum` (ablation, `O(C)`)

- Requires `ChannelSet.channel_embed` and `ChannelSet.channel_group_id`;
  identity is a routing bias + attention-pooled skip, never added to `tokens`;
  raises otherwise.
- `weight_net` (Linear → GELU → Linear → 1) is evaluated **once per channel
  instance** (`channel_group_id`), then broadcast to all of that channel's
  patches: `logits: [V]`.
- Values `= out_norm(value_proj(content_norm(tokens))): [V, D]`.
- `weights = grouped_softmax(logits, location_group_id, Kf)` over the channel
  axis at each `(b,n)`; `Σ_c w_c · V_c` via `index_add_` → `[Kf, D]`;
  then `+ spatial_pos` only in legacy mode.
- Cost: `O(C·D²)` per location (one value projection per token) plus one small
  MLP per channel; `O(C)` scaling.

### Padding-free reference

`_dense_fusion_reference` is a private, test-only implementation of the same
`ChannelFusion` math on a materialised `[B, C_max, N, D]` tensor plus an
attention mask. It is **not** in the model path.

## Latent pooler — `fusion.py` / `mae.py` (L1, L2, L6)

- `MultiheadAttentionPooling` is a **PMA readout**, separate from the fusion
  modules: learned queries, standard `q_proj / k_proj / v_proj / out_proj`
  projections, and `grouped_pool_attention` over the real tokens of each
  **sample** group.
- `ChannelFusionMAE` applies it once, to `tok_out` (the encoder's **content
  tokens only**, not the memory tokens):
  `latents = latent_pooler(tok_out, enc_group_id, B)` with
  `n_queries = n_latents` and `readout="flat"` → `[B, L·D]`, reshaped to
  `[B, L, D]`. This is spatial pooling to fixed size latent, independent of
  `C` and `N`, and it is **trained** through the reconstruction loss.
  `encode()` returns it.

## Encoder — `encoder.py` (L2, L6)

- `enc_tokens` is the fusion output `[Kf, D]` directly (no gather);
  `enc_group_id` is one group per sample
  (`torch.arange(B).repeat_interleave(present.sum(1))`).
- `memory: [1, M, D]` expanded `[B, M, D]` → `[B·M, D]`, with per-sample
  group ids.
- `x = cat([mem, tokens])` → `[B·M + Kf, D]`.
- `TransformerEncoder` is a stack of standard pre-norm `TransformerBlock`s:
  explicit Q/K/V/O projections + MLP, with
  `grouped_projected_self_attention` (bucketed equal-length groups) per
  sample. 2-D axial RoPE rotates q/k from metric `(row, col)` token-centre
  coordinates (µm); memory tokens get zero rotation. Default
  `scale_conditioning="add"` adds a Fourier embedding of
  `[log(p), log(pixel_size)]` to the encoder tokens; `"adaln"` instead
  modulates the LayerNorms. Final `LayerNorm`.
- Returns `mem_out [B·M, D]`, `tok_out [Kf, D]`; `tok_out` is then pooled
  to `latents [B, L, D]`.
- Final decoder KV is `enc_all = cat([mem_out, latents])` →
  `[B·(M + L), D]`, with matching per-sample group ids. The KV length is
  fixed for a given model and does not grow with the patch count.

## Decoder — `decoder.py` (L6)

- Targets: per sample `~visible[b]` → `(channel, patch)` pairs; `tgt_group_id`,
  `tgt_patch`, `tgt_channel_embed [Kt, D]` indexed by the **global** channel
  instance (`channel_id` → `ChannelEmbedding`).
- `q = mask_token [D] + fractional_pe(row, col, step_um)
  + target_proj(tgt_channel_embed + scale_emb)` → `[Kt, D]`. The fractional PE
  is metric by default: `(r + 0.5)·p·pixel_size / pe_ref_um`; `grid_h/grid_w`
  are used only by the `pe_space="grid"` ablation. The same channel-embedding
  table serves two roles: input identity (fusion routing bias + output skip;
  never added to `tokens`) and target specification (decoder query). With the
  default `scale_conditioning="add"`, the per-sample scale embedding is folded
  into the target channel embedding; with `"adaln"` it conditions the decoder
  blocks.
- Each `DecoderBlock`:
  1. **target self-attention** over the queries of each sample
     (`grouped_projected_self_attention`, bucketed);
  2. **cross-attention** to the encoder representation
     (`mem_out + latents`) using the general grouped primitive
     `grouped_attention`;
  3. MLP.
- `norm` then the per-`p` head: `Linear(D, p²)` for configured `p`; an unseen
  `p` uses the nearest larger head + bilinear point-sampling. The output is the
  flat `[Σ p²]` pixel array: row `i` is `p_i²` contiguous pixels
  (`losses.patch_offsets`), so mixed per-sample `p` needs no padded block and
  no validity mask.
- Target `= ChannelTokenizer.gather_patches(...)` → flat `[Σ p²]`, the same
  layout, reading only the requested patches.

## Loss — `losses.py` (L6)

- Default objective: standardise each target patch (`(t - mean) / std`), then
  MSE, averaged with `loss_reduce="sample"` (one mean squared error per
  sample, i.e. one vote per crop; `"token"` weights rows equally and is the
  ablation). No
  per-channel normalisation. The model path uses `patch_row_terms` on the flat
  pixel layout (per-row segment sums, no mask); `normalized_patch_mse` is the
  `[K, p²]` reference form of the same math.
- Ablation `cha_maevit_recon_loss(pred, target, lambda_f=0.01)`:
  `(1-λ)·MSE(raw pixels) + λ·L1(|rFFT₂(pred)|, |rFFT₂(target)|)` —
  ChA-MAEViT Eq. 2. Selected with `loss="cha_maevit"`.

## Inference — `mae.py::encode` (L1, L2)

- No masking (`visible = all True`); same tokenizer + fusion + encoder.
- Returns `(mem_out [B·M, D], latents [B, L, D])`: the two components of the
  decoder KV. `latents` is the fixed set from spatial pooling to fixed size
  latent.

## The claim the figures must carry

- The only place `C` appears is the input length `ΣC·N`; fusion **sums it away**
  at each `(b,n)`, emitting one `[D]` vector per present location (`[Kf, D]`).
  Output vector size is independent of `C`; the row count is the number of
  present locations, so `N` never enters a tensor either.
- The encoder's token sequence is `[Kf, D]`, but the model applies spatial
  pooling to fixed size latent, giving `[B, L, D]` (`L = n_latents`). The decoder KV
  `[B·(M + L), D]` is independent of both `C` and the patch count `N`.
- `[B, C_max, N, D]` is **never allocated/built** (never materialised) — draw it
  only as a dashed ghost.
- Masking / missing channels are handled by **omitting rows**, not zeroing or
  attention masks.
- The `[B, L, D]` latent set is trained and comes from content tokens only. A
  plot of `fused.mean(dim=1)` is a **fused-token mean**, not this latent set.
