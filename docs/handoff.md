# Handoff: Padding-Free Channel-Fusion Module for an HCI Foundation Model

This is the design specification this repository implements, reproduced from the
research handoff (message `83a9cb8b…`, plus the MAE follow-up `f87fe1bd…` in
`chat-export-1792446787674.json`).

**To:** implementation agent
**From:** research/design
**Deliverable:** a channel-fusion module (encoder front-end) that maps an
**arbitrary, unordered set of same-size per-channel embeddings** to a
**fixed-size joint embedding**, with **no padding tokens**, tuned for
high-content-imaging (HCI) foundation models.

---

## 1. Problem statement

Input: a set of per-channel embeddings `{E_c ∈ R^D}` for c in panel, where `C`
varies per sample (1 up to 20–60+ for multiplexed panels), channel order is
semantically meaningless, and panels/markers vary across labs and datasets.

We need `f({E_c}) → R^{k×D}` (k learned queries, typically k=1) such that:

- output size is **independent of C** (no concatenation blow-up);
- output is **permutation-invariant** over channel order but
  **channel-identity-aware**;
- aggregation weights are **data-dependent** (not fixed mean/max);
- **missing channels handled by omission**, not zero-padding;
- no pad tokens, no attention masks for cardinality.

## 2. Design decisions (settled — implement these)

**Chosen family: attention pooling over channel embeddings**
(Panopticon-style learned-query pooling), made set-native. ChannelViT and
ChAda-ViT are precedents for tied per-channel projection, additive channel
identity, and cross-channel attention; both use dense padded attention, so they
inform the design but are not padding-free and do not pool to a fixed-query
bottleneck. Explicitly **rejected**: Deep Sets mean/max as the *primary*
mechanism (channel identity required); wavelength-hypernetwork conditioning
(DOFA/HyperFree), whose supported mechanism keys on per-channel wavelength,
whereas HCI panels are identified by markers.

Pipeline, in order:

1. **Per-channel patch embedding, channel-tied.** Shared/tied linear projection
   across channels (ChannelViT: tied > untied, more robust to varying channel
   sets).
2. **Additive channel-identity embedding** per token — **do not omit**.
   ChannelViT's channel-embedding ablation (replacing per-channel embeddings
   with a shared mean) severely degrades accuracy. Closed panel → learnable
   per-marker embeddings. Add a **shared spatial positional embedding** so
   position ≠ channel identity.
3. **Channel-axis attention / pooling** to a fixed query bottleneck (PMA-style
   cross-attention), placed **before** the main spatial transformer to bound
   compute.
4. **Per-token LayerNorm** everywhere — never normalize across the channel axis.

## 3. Padding-free implementation (the core requirement)

Do **not** densify to `[B, C_max, D]`. Primary route:

```python
# Concatenate only REAL channel tokens across the whole batch -- no pad tokens.
E    = per_channel_proj(tokens) + channel_embed(ids)      # [ΣC, D]
joint = varlen_cross_attn(q=queries,                      # [B, k, D]
                          k=E, v=E,
                          cu_seqlens=offsets)              # cumulative channel counts
```

- Attention is cross-sample-independent; varlen just exploits it, so results
  match masked attention exactly.
- **Channel dropout / augmentation = token removal before the channel-attention
  stack**, never zeroing.

Alternates: PyTorch nested tensors; **hypernetwork-weighted sum**
(`w_c = MLP(channel_embed_c)`, `joint = Σ_c softmax(w)_c · E_c`) — `O(C)`
instead of `O(C²)`, preferred if 20–60-channel quadratic cost is a concern.

## 4. HCI-specific requirements (must-have)

1. **Missing channels by omission.**
2. **Hierarchical Channel Sampling (HCS)** during training: sample channel
   *count* first, then a combination.
3. **Channel-subset regularization**: train on random marker subsets.
4. **Permutation semantics**: output depends on *which* markers are present, not
   their order.

## 5. Acceptance criteria

- Cardinality invariance: identical output shape for `C ∈ {1, 3, 6, 20, 60}`.
- Padding-free equivalence vs masked-padded reference, mixed-`C` batches.
- Omission correctness (no leakage; per-token norm unaffected).
- Permutation test: shuffling channel order changes output < tolerance.
- Identity ablation: dropping the additive embedding degrades the task.
- No pad artifacts: no `-inf` masks, no `[B, C_max, D]` in the fusion path.
- Cost check: FLOPs/throughput at C=60, cross-attention vs. hypernetwork sum.

## 6. Pitfalls to avoid

- No fallback to `[B, C_max, D]` + masking.
- No mean/max pooling as the primary mechanism.
- No wavelength conditioning for HCI (marker identity instead).
- No normalization across the channel axis.
- No channel dropout by zeroing.

## 7. Reference implementations to adapt

ChannelViT (2309.16108) · ChAda-ViT (2311.15264) · Panopticon (2503.10845) ·
Perceiver / Perceiver IO (2103.03206 / 2107.14795) · FlashAttention
`varlen` (`cu_seqlens`) / HF padding-free path / PyTorch nested tensors.

## 8. Deliverable summary

Ship `ChannelFusion` as a drop-in encoder front-end: **channel-tied
per-channel embedding → additive marker-conditioned identity embedding →
varlen cross-attention into a fixed learned-query bottleneck**, trained with
HCS + channel dropout by token removal, per-token norms throughout.

---

# MAE follow-up (deltas to the spec)

Train the backbone as a **channel-aware MAE**:

- **Two masking axes** (Dynamic Channel-Patch masking, ChA-MAEViT 2503.19331):
  spatial patch masking sampled **independently per channel**, plus **dynamic
  channel masking** (sample `k`, then which `k`). Mix via `p_patch, p_channel`.
  Spatial/channel masking = **token omission** in the varlen tensor; guard:
  never mask all channels.
- **Fixed-count memory/register tokens** through encoder + decoder
  (cardinality-independent).
- **Channel-aware shared decoder** (one decoder, not per-channel), fed the
  unmasked channel tokens + memory tokens, and **conditioned on the target
  channel's identity embedding**.

Extra acceptance criteria: channel-masked reconstruction beats spatial-only
masking; memory-token ablation degrades reconstruction; padding-free property
preserved under masking (encoder input length == number of visible tokens);
downstream linear-probe transfer.
