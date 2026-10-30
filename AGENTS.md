# Purpose of this repo

This repo is a PyTorch reference implementation of a padding-free "channel-fusion" tokenizer for a high-content-imaging (HCI) foundation model, plus a channel-aware Masked Autoencoder (MAE) built on top of it. The design rationale and deviations are in `docs/design.md`.

## The core problem it solves

An HCI image is a set of per-channel embeddings {E_c ∈ R^D} where:

- the channel count C varies per sample (1 up to 5+, much larger in multiplexed panels),
- channel order is semantically meaningless,
- panels/markers differ across labs and datasets.

The goal is a function f({E_1..E_C}) → [N, D] per sample whose size is independent of C, plus a pooled [D] descriptor — no concatenation blow-up, channel-identity-aware, permutation-invariant, with no padding tokens.

Pixel size is a separate, orthogonal axis: `p` is a pixel count, not a physical size. Expected data has square pixels, and the **acquisition fleet** — the set of acquisition instruments/scanners and settings the model is trained and deployed on — has a finite table of pixel sizes (`FLEET_PIXEL_SIZES` in `src/fuse_channels/fleet.py`, currently `(0.598, 0.646, 0.650, 0.652, 0.703)` µm/px; the table, not the fleet itself, is what the code calls `fleet_pixel_sizes`), so pixel anisotropy is **not** a design case (`PhysicalSizeX ≠ PhysicalSizeY` is bad metadata to fix or reject, not something `p` absorbs). Each *patch* is kept at a comparable physical extent across instruments by choosing `p_b = round(p_target_um / pixel_size_b)` from the image's µm/px metadata; `p` is never snapped onto a lattice, and a pixel size outside the fleet is rejected at ingest rather than served with an untrained `p`. The invariant is the per-patch footprint `p · pixel_size`; the crop/window extent `n_tokens · p · pixel_size` is a separate, deliberately variable quantity (cell-sized grids are intended). See `README.md` §"Variable pixel size" and `FLEXIVIT.md` §0 before changing patch-size, scale-conditioning, or dataloading behaviour.

## Layout

- `src/fuse_channels/` — library: `ops` (padding-free grouped attention), `container` (`ChannelSet`), `identity` (`ChannelEmbedding`), `patch_embed` (`ChannelTiedPatchEmbed`, `ChannelTokenizer`), `fusion` (`ChannelFusion`, `IdentityGatedSum`, `MultiheadAttentionPooling`), `encoder`, `decoder`, `masking` (DCP Eq. 1 + HCS), `losses`, `mae`, `data`
- `tests/` — invariants and acceptance coverage (cardinality invariance, padding-free ≡ masked-padded, omission, permutation, identity routing, DCP/HCS, descriptor, losses, linear probe)
- `examples/` — `train_mae.py` (synthetic variable-channel training), `linear_probe.py` (frozen-encoder probe), `benchmark_fusion.py`
- `viz/` — visualizations (included in this repo)
- `references/` — background papers
- `docs/design.md` — design rationale and corrected reference claims

## Naming

`ChannelTokenizer` produces the channel-set tokens (tied patch embed + additive identity); `ChannelEmbedding` is the learned per-channel-id table; `IdentityGatedSum` is the marker-routed O(C) fusion ablation; `grouped_attention` means padding-free batching by group id, not grouped-query attention.
