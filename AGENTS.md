# Purpose of this repo

This repo is a PyTorch reference implementation of a padding-free "channel-fusion" front-end for a high-content-imaging (HCI) foundation model, plus a channel-aware Masked Autoencoder (MAE) built on top of it. The design rationale and deviations are in `docs/design.md`.

## The core problem it solves

An HCI image is a set of per-channel embeddings {E_c ∈ R^D} where:

- the channel count C varies per sample (1 up to 5+, much larger in multiplexed panels),
- channel order is semantically meaningless,
- panels/markers differ across labs and datasets.

The goal is a function f({E_1..E_C}) → [N, D] per sample whose size is independent of C, plus a pooled [D] descriptor — no concatenation blow-up, channel-identity-aware, permutation-invariant, with no padding tokens.

## Layout

- `src/fuse_channels/` — library: `ops` (padding-free grouped attention), `container` (`ChannelSet`), `identity` (`ChannelEmbedding`), `patch_embed` (`ChannelTiedPatchEmbed`, `ChannelTokenizer`), `fusion` (`ChannelFusion`, `IdentityGatedSum`, `MultiheadAttentionPooling`), `encoder`, `decoder`, `masking` (DCP Eq. 1 + HCS), `losses`, `mae`, `data`
- `tests/` — invariants and acceptance coverage (cardinality invariance, padding-free ≡ masked-padded, omission, permutation, identity routing, DCP/HCS, descriptor, losses, linear probe)
- `examples/` — `train_mae.py` (synthetic variable-channel training), `linear_probe.py` (frozen-encoder probe), `benchmark_fusion.py`
- `viz/` — visualizations (included in this repo)
- `references/` — background papers
- `docs/design.md` — design rationale and corrected reference claims

## Naming

`ChannelTokenizer` is the front-end (tied patch embed + additive identity); `ChannelEmbedding` is the learned per-channel-id table; `IdentityGatedSum` is the marker-routed O(C) fusion ablation; `grouped_attention` means padding-free batching by group id, not grouped-query attention.
