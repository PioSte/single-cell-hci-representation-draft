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
| `c` | canonical reconstruction-head grid, `= head_patch_size or max(patch_sizes)` | `PatchReconstructionHead.canonical_patch_size` |
| `D` | token/embedding dim | `dim` |
| `M` | fixed memory/register tokens per sample | `n_memory` |
| `T` | real channel-patch tokens in a flattened token set (`T = V` after masking) | `cs.tokens.shape[0]` |
| `V` | number of **visible** channel-patch tokens (`≤ Σ_b C_b·N_b`) | `keep.sum()` |
| `Kf` | number of present `(b,n)` locations (`≤ B·N`) | `present.sum()` |
| `Kt` | number of masked channel-patches (decoder targets) | `~visible` nonzero |
| `C_u` | number of unique channel instances in the batch | `unique(channel_group_id)` |

The fusion output is one token per location, so there is **no query-count
axis** in the fused tensor. `n_queries` exists only on the descriptor pooler
(see below), never on `ChannelFusion`.

## Tokenizer — `patch_embed.py` (L3)

- `images`: list of `[C_b, 1, H, W]` → `torch.cat` → `[ΣC, 1, H, W]`.
- `ChannelTiedPatchEmbed.proj = Conv2d(1, D, kernel_size=P, stride=P)` —
  **one shared kernel for every channel** (tied, not per-channel), with the
  kernel **PI-resized** (FlexiViT) when a non-canonical patch size is used.
  It has **no positional embedding and no `spatial_pos` parameter**.
- `feat = proj(x).flatten(2).transpose(1,2)` → `[ΣC, N, D]`; flatten →
  `[ΣC·N, D]`. The optional DCP **visible** mask is applied to rows →
  `[V, D]`.
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
- Each sample derives `N_b = floor(H/p_b)·floor(W/p_b)`,
  `grid_h = H//p_b`, `grid_w = W//p_b`; the border smaller than `p_b` is
  cropped. The masker uses `[C_b, N_b]`, targets use `p_b`, and
  `ChannelSet.n_patches = max_b N_b`.
- No positional table is resized: encoder `2-D axial RoPE` uses normalized
  `(row,col)`, decoder uses factorized fractional PE from `(r,c,g_h,g_w)`,
  and `ChannelFusion` adds no spatial table.
- `scale_conditioning="add"` (default) embeds `[log(p), log(pixel_size), has_pixel_size]` with
  Fourier features + MLP and adds it to encoder tokens and to the decoder
  target channel embedding before `target_proj`.
- The reconstruction head keeps one `Linear(D, p²)` per configured `p`
  (`head_mode="per_p"`, default). An unseen `p` uses the nearest configured
  head with `p_head >= p` and bilinear point-sampling; `head_mode="canonical"`
  keeps the single `Linear(D, c²)` head with `c = head_patch_size or
  max(patch_sizes)` as the ablation. With mixed per-sample `p`, predictions are
  padded to `max_b p_b²` and accompanied by a validity mask.

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
`fused: [B, N, D]`. The default target adds no positional table; the legacy
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
- `grouped_pool_attention(q, k, v, lookup, Kf)` → `[Kf, 1, H, Dh]`
  (**separate learned `W_Q/W_K/W_V` projections**, Panopticon's channel
  cross-attention recipe).
- `proj_out: Linear(D_attn, D)` maps back → `[Kf, D]`; scatter to
  `[B, N, D]`; add `spatial_pos` only in legacy mode.
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
- Values `= out_norm(value_proj(tokens)): [V, D]`.
- `weights = grouped_softmax(logits, location_group_id, Kf)` over the channel
  axis at each `(b,n)`; `Σ_c w_c · V_c` via `index_add_` → `[B, N, D]`;
  then `+ spatial_pos` only in legacy mode.
- Cost: `O(C·D²)` per location (one value projection per token) plus one small
  MLP per channel; `O(C)` scaling.

### Padding-free reference

`_dense_fusion_reference` is a private, test-only implementation of the same
`ChannelFusion` math on a materialised `[B, C_max, N, D]` tensor plus an
attention mask. It is **not** in the model path.

## Descriptor pooler — `fusion.py` / `mae.py` (L1, L2, L6)

- `MultiheadAttentionPooling` is a **PMA readout**, separate from the fusion
  modules:
  - learned queries `[1, n_queries, D]` (`n_queries` defaults to 1; this is the
    only remaining query-count knob);
  - standard `q_proj / k_proj / v_proj / out_proj` projections;
  - `grouped_pool_attention` over the real tokens of each **sample** group;
  - optional post-SAB over the `n_queries` outputs when `n_queries > 1`;
  - `readout="mean"` averages the query outputs → `[B, D]`.
- In `ChannelFusionMAE.forward` the pooler is applied to `tok_out` (the
  encoder's **content tokens only**, not the memory tokens):
  `desc = pooler(tok_out, enc_group_id, B) → [B, D]`. It is appended as one
  extra per-sample token to the decoder KV and therefore **trained** through
  the reconstruction loss. `encode(pool="attention")` returns that same
  descriptor.
- `pool="mean"` is the content-token mean ablation; `pool="memory"` does not
  exist.

## Encoder — `encoder.py` (L2, L6)

- `enc_tokens = fused[present]` → `[Kf, D]`; `enc_group_id` is one group per
  sample (`torch.arange(B).repeat_interleave(present.sum(1))`).
- `memory: [1, M, D]` expanded `[B, M, D]` → `[B·M, D]`, with per-sample
  group ids.
- `x = cat([mem, tokens])` → `[B·M + Kf, D]`.
- `TransformerEncoder` is a stack of standard pre-norm `TransformerBlock`s:
  explicit Q/K/V/O projections + MLP, with
  `grouped_projected_self_attention` (bucketed equal-length groups) per
  sample. 2-D axial RoPE rotates q/k from normalized `(row, col)` coordinates;
  memory tokens get zero rotation. Default `scale_conditioning="add"` adds a
  Fourier embedding of `[log(p), log(pixel_size), has_pixel_size]` to the encoder tokens;
  `"adaln"` instead modulates the LayerNorms. Final `LayerNorm`.
- Returns `mem_out [B·M, D]`, `tok_out [Kf, D]`.
- Final decoder KV is `enc_all = cat([mem_out, tok_out, desc])` →
  `[B·M + Kf + B, D]`, with matching per-sample group ids.

## Decoder — `decoder.py` (L6)

- Targets: per sample `~visible[b]` → `(channel, patch)` pairs; `tgt_group_id`,
  `tgt_patch`, `tgt_channel_embed [Kt, D]` indexed by the **global** channel
  instance (`channel_id` → `ChannelEmbedding`).
- `q = mask_token [D] + fractional_pe(row, col, grid_h, grid_w)
  + target_proj(tgt_channel_embed + scale_emb)` → `[Kt, D]`. The same
  channel-embedding table serves two roles: input identity (fusion routing
  bias + output skip; never added to `tokens`) and target specification
  (decoder query). With the default
  `scale_conditioning="add"`, the per-sample scale embedding is folded into the
  target channel embedding; with `"adaln"` it conditions the decoder blocks.
- Each `DecoderBlock`:
  1. **target self-attention** over the queries of each sample
     (`grouped_projected_self_attention`, bucketed);
  2. **cross-attention** to the encoder representation
     (`mem_out + tok_out + desc`) using the general grouped primitive
     `grouped_attention`;
  3. MLP.
- `norm` then the per-`p` head: `Linear(D, p²)` for configured `p`; an unseen
  `p` uses the nearest larger head + bilinear point-sampling → `[Kt, p²]`
  (padded with a mask for mixed per-sample p).
- Target `= extract_patches(images)[gidx, :, tgt_patch]` → `[Kt, p²]`
  (padded/validity-masked in the mixed-p case).

## Loss — `losses.py` (L6)

- Default `normalized_patch_mse(pred, target)`: standardise each target patch
  (`(t - mean) / std`), then MSE. No per-channel normalisation.
- Ablation `cha_maevit_recon_loss(pred, target, lambda_f=0.01)`:
  `(1-λ)·MSE(raw pixels) + λ·L1(|rFFT₂(pred)|, |rFFT₂(target)|)` —
  ChA-MAEViT Eq. 2. Selected with `loss="cha_maevit"`.

## Inference — `mae.py::encode` (L1, L2)

- No masking (`visible = all True`); same tokenizer + fusion + encoder.
- `pool="attention"` (default) applies the trained descriptor pooler to
  `tok_out`; `pool="mean"` averages the content tokens. Both return a
  `[B, D]` descriptor; `encode` also returns `mem_out [B·M, D]`.
- There is no memory-pool readout.

## The claim the figures must carry

- The only place `C` appears is the input length `ΣC·N`; fusion **sums it away**
  at each `(b,n)` to `[B, N, D]`. Output shape/size is independent of `C`.
- `[B, C_max, N, D]` is **never allocated/built** (never materialised) — draw it
  only as a dashed ghost.
- Masking / missing channels are handled by **omitting rows**, not zeroing or
  attention masks.
- The `[B, D]` descriptor is trained, comes from content tokens only, and is
  appended to the decoder KV. A plot of `fused.mean(dim=1)` is a **fused-token
  mean**, not this descriptor.
