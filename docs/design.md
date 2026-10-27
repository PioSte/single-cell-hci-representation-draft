# Design notes

This document records the design decisions behind the shipped implementation,
the trade-offs they accept, and the places where earlier prose (and the
literature summary) were wrong. The current API is in
[`../README.md`](../README.md).
Citation line numbers refer to the extracted paper texts under
`references/*/paper.md`.

## 1. Additive channel embedding

`ChannelEmbedding` is a learned `nn.Embedding(num_channel_ids, dim)` table. A
channel id selects one row; that row is added to every patch token of the
channel and is never normalized across the channel axis.

**The table is the width-efficient form of the same additive bias.** For a
token `x` and channel id `m`,

```
concat(x, onehot(m)) @ W
    = W_x x + W_e · onehot(m)
    = W_x x + E[m].
```

The concatenation path is therefore not a different function; its `W_e` rows
*are* the embedding table. The differences are parameter count and width: the
concat path widens every token to `D + num_channel_ids` and pays that width in
every attention and MLP layer, while the table needs `num_channel_ids × D`
parameters and leaves the network width independent of the registry size. This
is the argument for the table, not a claim that one-hot inputs cannot express
similarity: a learned `W_e` can encode any pairwise similarity.

**Identity matters.** ChannelViT's mean-embedding ablation (Table 10) replaces
each per-channel embedding with the mean embedding across channels and reports
57.60/68.09 on JUMP-CP collapsing to 1.26/2.49, and 47.39/63.01 collapsing to
10.44/52.13 on So2Sat (`references/channelvit/paper.md:670-688`). The tied
patch filters and shared positional embedding are what make an explicit
identity signal necessary.

**Table 11 is not a token-level one-hot ablation.** It conditions the *final
classifier* on the sampled channel combination: a one-hot combination feature
plus MLP classifier scores 66.86, a learned combination embedding plus MLP
scores 61.98, and the plain shared linear classifier on ChannelViT scores 68.09
(`references/channelvit/paper.md:692-714`). The paper's explanation is that the
shared classifier *regularizes* the backbone to embed different channel
combinations into the same space. It supports a shared readout, not the claim
that additive-per-token identity beats a one-hot token input.

The learned embeddings do become correlated and interpretable for related
channels (`references/channelvit/paper.md:245-255`), but that is a property of
training an embedding table, not an expressiveness advantage over concat.

The cost is a closed vocabulary. Only `num_channel_ids` rows exist, and an id
with no trained row cannot be represented; unseen markers require new rows (and
ChA-MAEViT shows novel channel tokens can be fine-tuned,
`references/cha-maevit/paper.md:697-709`). For a closed HCI panel this is the
intended trade-off.

Scope of the "additive learned embedding" claim: it is the ChannelViT /
ChAda-ViT / ChA-MAEViT family. Panopticon uses a *fixed* sinusoidal encoding of
the central wavelength for optical channels and concatenates learned SAR
embeddings; it is not an additive learned identity
(`references/panopticon/paper.md:94-108`). DOFA/HyperFree use
hypernetwork-generated conditioning, which this repo deliberately does not.

## 2. Per-location single-query fusion vs full `C·N` attention

Panopticon reduces the channel axis at each patch by cross-attention with a
single learned query `q ∈ R^{1×D}`, using the per-channel patch embeddings as
keys and values (`references/panopticon/paper.md:92`). Its actual recipe is
wider than "one query": the tokens are projected with learned Q/K/V
projections into an attention dimension `D_attn`, attention runs there, and a
final projection maps back to the backbone width. Panopticon sets
`D_attn = 3D` and its Table 5 ablation shows `D_attn > D` improves the metric
substantially (`references/panopticon/paper.md:92,194,219`; the earlier
"projection-free K/V" reading in this repo was wrong).

`ChannelFusion` follows that recipe: one learned query is shared across every
sample and every spatial location, the LayerNormed channel tokens go through
learned `q_proj`/`k_proj`/`v_proj` into `attn_dim = 3·dim` by default, and
`proj_out` maps the pooled vector back to `[B, N, D]`. `attn_dim=dim` is the
narrow, parameter-matched ablation. The fusion group is `(sample, patch)`, so
the softmax axis is the channel axis, never the patch axis.

The cost is `O(C)` per location, `O(N·C)` per sample. The whole-sequence
alternatives are:

* ChannelViT concatenates all `C·N` per-channel patch tokens into one ViT
  sequence and uses HCS to shorten it; it does **not** pad
  (`references/channelvit/paper.md:37-41`,
  `references/channelvit/paper.md:107-117`). Its attention is over `C·N`
  tokens, i.e. `O((C·N)²)`.
* ChAda-ViT pads every panel to `C_max × m` tokens and masks the padding
  (`references/chada-vit/paper.md:121-129`), also giving `O((C·N)²)` on the
  padded sequence.

Panopticon is not a whole-sequence baseline: it uses the same per-patch
single-query channel pooling as `ChannelFusion`, and its implementation pads
and masks the channel axis only to batch that per-location operation
(`references/panopticon/paper.md:581`). Its cost is therefore `O(N·C_max)` per
sample, not `O((C·N)²)`.

So the expensive baselines are `O((C·N)²)` full joint attention, not a
per-location `C×C` block. The repo's per-location pool is cheaper but does not
let locations interact until the encoder; that is deliberate, and the encoder
is where spatial mixing happens. A single query per location also means
`ChannelFusion` has no `n_queries`; the multi-query research ablation lives in
the descriptor pooler instead.

`IdentityGatedSum` is the other `O(C)` option (Section 8). It is an ablation,
not the default.

## 3. Position: encoder RoPE, decoder fractional PE

The tied patch embed has no positional embedding. Channel attention at a
location is over channels at the *same* patch, so it is naturally
position-agnostic; adding position before it would give the attention keys
positional offsets that cannot help distinguish channels. Fusion therefore
stays position-free in the target architecture.

**Encoder: 2-D axial RoPE.** Position enters the encoder as a rotation of
queries and keys, not as an additive table. Coordinates are normalized to
`(r + 0.5)/grid_h` and `(c + 0.5)/grid_w` (times a fixed reference extent), so
the phase describes layout only and does not change meaning when the patch size
changes. The head dimension is split into row and column halves, and each axis
gets `head_dim / 4` frequency pairs with base `100` by default
(`references/rope-vit/paper.md:157-171`; RoFormer describes the rotation as
absolute position with relative dependence in the dot product,
`references/roformer/paper.md:23`). Memory tokens get zero rotation. Axial RoPE
is the simple first choice; RoPE-Mixed is the stronger upgrade because axial
frequencies cannot express diagonal directions
(`references/rope-vit/paper.md:27`).

Raw-index RoPE would need no resampling but would also encode sampling
density: adjacent tokens are one index apart at `p = 8` and at `p = 32`, even
though the physical distance differs. Normalized coordinates avoid that and
leave physical scale to the scale conditioner.

**Decoder: factorized-fractional absolute PE.** Mask tokens need an additive
absolute anchor: without it every masked patch of a channel gets the same query
and predicts the same pixels (MAE's `decoder_pos_embed`,
`references/mae/paper.md:103`). The decoder PE is

```
u = (c + 0.5) / grid_w
v = (r + 0.5) / grid_h
PE_dec(r, c) = g_row(v) + g_col(u)
```

where `g_row` / `g_col` are fixed multi-frequency sinusoids by default (a
small learned MLP is available as `fractional_pe_mode="mlp"`). It is a function
of fractional position, not a lookup on a fixed grid, so it is defined for any
grid and needs no resampling when `p` changes.

**Legacy absolute-table ablation.** `pos_embed="learned"` or `"sincos"`
restores the pre-FlexiViT scheme: a post-fusion `[1, N, D]` table (learned or
MAE/ViT sine-cosine) plus the decoder's `spatial_pos[target_patch]` table, and
no encoder RoPE. This is fixed-grid only. The earlier design notes on MAE's
sine-cosine code and the lack of a learned-vs-fixed ablation still apply to
that legacy path (`references/mae/paper.md:546`); the target path replaces both
tables with functions.

**Scale conditioning.** `log(p)` and, when supplied, `log(µm/px)` are embedded
with Fourier features and a small MLP. The default `scale_conditioning="add"`
adds the embedding to encoder and decoder tokens; `"adaln"` modulates the
LayerNorms instead, which is stronger because an additive constant before a
LayerNorm is partly normalized away.

## 4. Trained descriptor

`MultiheadAttentionPooling` pools a padding-free token set with a learned set
of queries through standard Q/K/V/O projections (PMA-style; Set Transformer's
PMA is a multi-head attention block over `rFF(Z)` with optional post-SAB,
`references/set-transformer/paper.md:157-171`). The repo's module is a single
attention block; it does not claim to be the fully canonical PMA.

In `ChannelFusionMAE.forward` the descriptor is computed from the encoder's
**content tokens only** (`tok_out`, not the memory tokens), appended as one
extra per-sample token to the decoder KV, and trained by the reconstruction
loss. That makes it a real readout rather than an inference-only artefact.
`encode(pool="attention")` returns the same descriptor;
`encode(pool="mean")` is the content-only mean ablation. `pool="memory"` does
not exist.

`n_queries > 1` builds a small post-SAB over the query outputs and averages
back to `[B, D]`; it is an optional research ablation, not the default
bottleneck.

## 5. Masking: DCP Eq. 1 and HCS overlap

ChA-MAEViT Eq. 1 draws a selection value `s` and chooses one of three regions:
patch masking, channel masking, or their union ("combined"). Patch masking
uses a fixed fraction per channel (`patch_ratio = 0.75` by default); the region
probabilities `p_patch` and `p_channel` are separate
(`references/cha-maevit/paper.md:105-130`). ChannelViT's HCS draws a visible
channel count `m ~ U{1..C}` first, then a subset of size `m`
(`references/channelvit/paper.md:107-117`).

Because dropped channels stay in `images` and are reconstructed as targets, the
two channel-dropping mechanisms induce the same visible-count distribution:
HCS *is* DCP's channel region. The implementation therefore has one
`HierarchicalChannelSampler`, used by `DCPVisibilitySampler`'s channel region;
the combined region is that sampler plus the per-channel patch mask. There is
no second channel-dropping mechanism.

Other details:

* `mode_sampling="batch"` (default) draws `s` once per call, so all samples
  share a region (paper-faithful); `mode_sampling="sample"` draws per sample.
* A channel-only draw on `C <= min_visible_channels` falls back to patch
  masking, which keeps the guard meaningful.
* Guards resample rather than reveal a whole channel row. This matters: a
  dropped channel must not leak back into the encoder as a fully visible
  channel.
* Masking omits rows from the flattened `[ΣC·N, D]` tensor. No mask tokens and
  no attention masks are used. The encoder receives one token per present
  location; fully dropped locations disappear from its input.

## 6. Decoder target identity and self-attention

ChA-MAEViT conditions its shared decoder on encoder-optimized channel tokens
added to the patch tokens (`references/cha-maevit/paper.md:121-125`). This
repo deliberately substitutes the shared `ChannelEmbedding` table. Under the
FlexiViT target the target query is

```
mask_token + fractional_pe(r, c, grid_h, grid_w)
            + target_proj(channel_embed[target_channel]).
```

The fractional PE term is the repo's equivalent of MAE's `decoder_pos_embed`,
and it is load-bearing rather than decorative: our decoder builds one query per
masked (channel, patch) and does not run full-grid self-attention, so without it
every masked patch of a channel would get the same query and predict the same
pixels (`test_decoder_target_position_changes_predictions`,
`test_fractional_decoder_pe_is_load_bearing_for_mask_tokens`).
MAE reaches the same conclusion from the other side -- it adds PE to the
unshuffled full token set for exactly this reason
(`references/mae/paper.md:103`) -- but sums it over visible *and* mask tokens
(`decoder_pos_embed` in `models_mae.forward_decoder`), because its decoder
self-attends over the whole grid. Our visible tokens already carry position from encoder RoPE, so only the
target queries need an additive anchor. Either parameterization works:
`mask_token` / `target_proj` are learned, and the decoder PE is a function of
fractional coordinates (or the legacy learned/`sincos` table).

That is Perceiver-IO-style query construction from position and
modality/target features (`references/perceiver-io/paper.md:101-103`,
`references/perceiver-io/paper.md:546`), not the paper's encoder-optimized
channel tokens. The substitution is what lets one decoder be shared across
channels and keeps the target channel's identity independent of whether that
channel appeared in the encoder input. A fully masked channel has no encoder
evidence at all, only its identity query; that is the intended behaviour, not
an oversight.

Decoder blocks add grouped self-attention over the target queries of each
sample before cross-attention. Targets of the same sample can therefore
coordinate (e.g. correlated masked channel-patches) before attending to the
encoder representation + memory + descriptor. The decoder never receives raw
per-channel tokens; its KV is the encoder's final representation plus memory
and the trained descriptor.

## 7. Losses

The default is `normalized_patch_mse`: the standard MAE objective, MSE against
per-patch-normalized target pixels. The ChA-MAEViT objective is an explicit
ablation (`loss="cha_maevit"`): raw-pixel L2 plus an L1 distance between
Fourier amplitudes, with `lambda_f = 0.01`
(`references/cha-maevit/paper.md:137-143`). Both operate on the masked
channel-patches only (the reconstruction targets) and neither applies
per-channel normalization. The ablation is selectable rather than default so
the reference objective can be compared without changing the baseline MAE
recipe.

## 8. `IdentityGatedSum`: naming and the ablation

```
joint = Σ_c softmax_c(MLP(channel_embed_c)) · V_c
```

The small MLP emits one logit per channel and the weights are normalized over
the channels at each `(sample, patch)` location. The MLP is evaluated once per
unique channel instance (via `ChannelSet.channel_group_id`) and broadcast to
its patches, so the cost is `O(C)` per location.

It is **not** a hypernetwork: it does not generate network weights. It is also
not content-dependent in the way the attention variant is: its weights depend
only on the channel embedding, so for a fixed panel they are identical across
samples and locations. The honest description is **marker-routed** weighting.
The attention variant's weights depend on the token content. It raises an error
if `channel_embed` is missing rather than silently falling back to token
features.

The name `IdentityGatedSum` replaces `HypernetworkWeightedSum`, which described
a mechanism the module never had. It remains available as `fusion="identity_gated"`
for the cost/ablation comparison.

## 9. Reference claims corrected

| Earlier claim | Correction | Source |
| --- | --- | --- |
| ChannelViT pads to a dense channel axis | It concatenates all `C·N` tokens into one ViT sequence; HCS shortens the sequence. Of these two, only ChAda-ViT pads and masks. | `references/channelvit/paper.md:37-41`, `:107-117`; `references/chada-vit/paper.md:121-129` |
| Baselines pay `O(C²)` for a `C×C` cross-channel block | They attend over the full `C·N` (or padded `C_max·m`) sequence: `O((C·N)²)`. The repo's per-location fusion is `O(C)`. | `references/channelvit/paper.md:107-117`; `references/chada-vit/paper.md:121-129` |
| Panopticon uses an additive learned wavelength identity | Optical identity is a fixed sinusoidal encoding at the central wavelength; SAR embeddings are concatenated. | `references/panopticon/paper.md:94-108` |
| Perceiver's modality identity comes from its position-encoding lines | Use Perceiver-IO's learned modality/task embeddings and query construction. | `references/perceiver-io/paper.md:101-103`, `:546` |
| ChA-MAEViT's additive channel tokens are at `:113` | `:113` is the memory-token formula; the additive channel tokens are at `:121-125`. | `references/cha-maevit/paper.md:111-113`, `:121-125` |
| ChannelViT Table 11 is a token-level one-hot ablation | Table 11 conditions the final classifier; Table 10 is the channel-embedding ablation that supports "identity matters". | `references/channelvit/paper.md:670-688`, `:692-714` |
| Every reference architecture uses an additive learned identity | Narrow to ChannelViT/ChAda-ViT/ChA-MAEViT; Panopticon uses fixed wavelength PE plus concatenation, and DOFA/HyperFree use hypernetwork conditioning. | `references/panopticon/paper.md:94-108` |
| Panopticon's channel attention is projection-free (`k = v`, no K/V projections) | It uses learned Q/K/V projections into `D_attn`, with `D_attn = 3D` by default; Table 5 shows `D_attn > D` matters. | `references/panopticon/paper.md:92,194,219` |
| `IdentityGatedSum` provides data-dependent weighting | It is marker-routed: weights depend on the channel embedding, not on token content; only the attention fusion is content-dependent. | `src/fuse_channels/fusion.py` (class `IdentityGatedSum`) |
