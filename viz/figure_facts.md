# Figure facts — shapes and semantics drawn by the L0–L8 figures

Single source of truth so every figure agrees with `src/fuse_channels/`.
All shapes below are read from the code, not invented. Notation:

| symbol | meaning | code |
|---|---|---|
| `B` | batch size (samples) | `len(images)` |
| `C_b` | channel count of sample `b` (variable, order-free) | `images[b].shape[0]` |
| `ΣC` | total channels across the batch (`= B`-independent sum) | `sum(C_b)` |
| `H, W` | image side (square) | `image_size` |
| `P` | patch size | `patch_size` |
| `N` | patches per channel, `= (H/P)²` | `ChannelTiedPatchEmbed.n_patches` |
| `D` | token/embedding dim | `dim` |
| `Q` | channel-fusion queries per location (default 1) | `n_queries` |
| `M` | fixed memory/register tokens per sample | `n_memory` |
| `V` | number of **visible** channel-patch tokens (`≤ ΣC·N`) | `keep.sum()` |
| `Kf` | number of present `(b,n)` locations (`≤ B·N`) | `present.sum()` |
| `Kt` | number of masked channel-patches (decoder targets) | `~visible` nonzero |

## Front-end — `patch_embed.py` (L3)

- `images`: list of `[C_b, 1, H, W]` → `torch.cat` → `[ΣC, 1, H, W]`.
- `ChannelTiedPatchEmbed.proj = Conv2d(1, D, kernel_size=P, stride=P)` —
  **one shared kernel for every channel** (tied, not per-channel).
- `feat = proj(x).flatten(2).transpose(1,2)` → `[ΣC, N, D]`; then
  `feat = feat + spatial_pos` where `spatial_pos: [1, N, D]` is shared across
  channels (position ≠ channel identity).
- Flatten → `[ΣC·N, D]`; apply the DCP visible mask → `[V, D]`.
- `ChannelIdentity.embed = nn.Embedding(num_markers, D)`; tokens become
  `feat + identity` — an **additive** marker vector, never omitted.

## Masking — `masking.py` (L5)

- `DCPMasker` returns per sample `visible: [C_b, N]` bool (*keep*), one of two
  modes per sample with probabilities `p_channel` / `p_patch`:
  - **channel mode**: sample `k` (`count`) then `which k` via `randperm`;
    mask is constant down the channel's rows.
  - **patch mode**: `torch.rand(C_b, N) >= p_patch` — an **independent** spatial
    mask per channel.
- Guards: never mask every token; at least `min_visible_channels` channels keep
  some content.
- Masked ⇒ **omitted row**. Never zeroed, never padded, no `-inf` attention mask.

## Channel fusion — `fusion.py` (L2, L4)

- `seg = location_segments() = sample_idx * N + patch_idx` → one segment per
  `(b,n)` location; `n_seg = |unique(seg)| = Kf`.
- `tok = norm_kv(tokens)[order]` viewed `[V, H, Dh]`.
- `q = norm_q(queries)` with `queries: [1, Q, D]` — **one shared query set for
  every spatial location** — expanded to `[Kf, Q, H, Dh]`.
- `grouped_pool_attention(q, tok, tok, lookup, Kf)` → `[Kf, Q, H, Dh]`
  (**keys = values = tokens**; there is no `W_Q/W_K/W_V`).
- Cost: `O(Q·C·D)` per `(b,n)` location. With the default `Q = 1`
  (fixed-query PMA) this is **`O(C)`**, not the `O(C²)` of dense `C×C`
  channel self-attention (the ChannelViT / ChAda-ViT baseline).
- `proj_out` Linear → scatter back to `fused: [B, N, Q·D]`, `present: [B, N]`.
- `fuse_proj`: `Identity` if `Q·D == D` else `Linear` → `[B, N, D]`.
- **HypernetworkWeightedSum**: `logits = weight_net(identity) [V, Q]`,
  `weights = segment_softmax(logits)` over the channel (segment) axis, values
  `= out_norm(value_proj(tokens))`; `Σ_c weights_c · V_c` → `[B, N, Q·D]`
  — `O(C)` per location, keys on **marker identity** (`z_c`).
  (Both shipped fusions are therefore `O(C)` per location; the quadratic
  `O(C²)` baseline is dense `C×C` cross-channel self-attention.)

## Encoder — `encoder.py` (L2, L6)

- `enc_tokens = fused[present]` → `[Kf, D]`; `enc_seg` per sample.
- `memory: [1, M, D]` expanded `[B, M, D]` → `[B·M, D]`.
- `x = cat([mem, tokens])` → `[B·M + Kf, D]`, `grouped_self_attention` × depth.
- Returns `mem_out [B·M, D]`, `tok_out [Kf, D]`; `enc_all = cat` = `[B·M + Kf, D]`.

## Decoder — `decoder.py` (L6)

- Targets: per sample `~visible[b]` → `(channel, patch)` pairs; `tgt_seg`,
  `tgt_patch`, `tgt_identity [Kt, D]` (**global** channel index → identity).
- `q = mask_token [D] + spatial_pos[target_patch] + target_proj(target_identity)`
  → `[Kt, D]`.
- `grouped_cross_attention(q, tgt_seg, enc_all, enc_all_seg)` × depth.
- `head = Linear(D, P²)` → predictions `[Kt, P²]`.
- Target `= extract_patches(images)[gidx, :, tgt_patch]` → `[Kt, P²]`.

## Loss — `losses.py` (L6)

- `normalized_patch_mse(pred, target)`: per-patch standardise the target
  (`(t-mean)/std`), then MSE. No per-channel normalisation.

## Inference — `mae.py::encode` (L1, L2)

- No masking (`visible = all True`); same front-end + fusion + encoder.
- `pool`: `attention` (PMA over `[mem; tokens]`) / `mean` (over visible tokens) /
  `memory` (mean over the `M` memory tokens) → descriptor `[B, D]`.

## The claim the figures must carry

- The only place `C` appears is the input length `ΣC·N`; fusion **sums it away**
  at each `(b,n)` to `[B, N, D]`. Output shape/size is independent of `C`.
- `[B, C_max, N, D]` is **never materialised** — draw it only as a ghost.
- Masking / missing channels are handled by **omitting rows**, not zeroing or
  attention masks.
