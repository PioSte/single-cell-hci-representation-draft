# Purpose of this repo

This repo is a PyTorch reference implementation of a padding-free "channel-fusion" front-end for a high-content-imaging (HCI) foundation model, plus a channel-aware Masked Autoencoder (MAE) built on top of it. It's the code counterpart to a written design handoff (reproduced in docs/handoff.md).

## The core problem it solves

 An HCI image is a set of per-channel embeddings {E_c ∈ R^D} where:
 - the channel count C varies per sample (1 up to 5+ markers),
 - channel order is semantically meaningless,
 - panels/markers differ across labs and datasets.

 The goal is a function f({E_1..E_C}) → R^{k×D} whose output size is independent of C — no concatenation blow-up, channel-identity-aware, permutation-invariant, with no padding tokens.
 
 ## Layout

 - src/fuse_channels/ — library (ops, container, identity, patch_embed, fusion, encoder, decoder, masking, losses, mae, data)
 - tests/ — invariance/equivalence checks (cardinality invariance, padding-free ≡ masked-padded, permutation invariance, etc.)
 - examples/ — synthetic MAE training and a fusion throughput benchmark
 - viz/ — visualizations of the archiitecture
 - references/ —  background papers