# Purpose of this repo

This repo is a PyTorch reference implementation of a padding-free "channel-fusion" tokenizer for a high-content-imaging (HCI) foundation model, plus a channel-aware Masked Autoencoder (MAE) built on top of it. The design rationale and deviations are in `docs/design.md`.

## The core problem it solves

An HCI image is a set of per-channel embeddings {E_c ∈ R^D} where:

- the channel count C varies per sample (1 up to 5+, much larger in multiplexed panels),
- channel order is semantically meaningless,
- panels/markers differ across labs and datasets.

The goal is a function f({E_1..E_C}) → [N, D] per sample whose size is independent of C, then **spatial pooling to fixed size latent** — `L = n_latents` learned latents per sample, `[L, D]`, with L a hyperparameter independent of both C and the patch count N — no concatenation blow-up, channel-identity-aware, permutation-invariant, with no padding tokens.

Pixel size is a separate, orthogonal axis: `p` is a pixel count, not a physical size. Expected data has square pixels, and the **acquisition fleet** — the set of acquisition instruments/scanners and settings the model is trained and deployed on — has a finite table of pixel sizes. That table is the code constant `FLEET_PIXEL_SIZES` in `src/fuse_channels/fleet.py`; it is documented with the expected-data assumptions in `README.md` §"Variable pixel size" and `FLEXIVIT.md` §0, not duplicated here. Pixel anisotropy is **not** a design case (`PhysicalSizeX ≠ PhysicalSizeY` is bad metadata to fix or reject, not something `p` absorbs). Each *patch* is held at a target physical extent `target_um` by deriving `p` from the image's `pixel_size`; `p` is never snapped onto a lattice, and a pixel size outside the fleet is rejected at ingest rather than served with an untrained `p`. The per-patch footprint is the design invariant; integer rounding makes it approximate, and the realized value is recorded in the geometry. The crop/window extent is a separate, deliberately variable quantity (cell-sized grids are intended). Read those two sections before changing patch-size, scale-conditioning, or dataloading behaviour.

## Layout

- `src/fuse_channels/` — library modules; consult module docstrings and the file map in `README.md`.
- `tests/` — invariants and acceptance coverage.
- `examples/` — runnable training, probing, and benchmark examples.
- `viz/` — visualizations.
- `references/` — background papers.
- `README.md`, `FLEXIVIT.md`, `docs/design.md` — usage overview, patch-size/pixel-size design, and design rationale.

## Keep documentation at the right altitude

- Code is the source of truth for mechanics, defaults, and exact formulas.
  Module docstrings carry a stage's normative contract; README/design documents
  explain intent, trade-offs, usage, and why a choice was made.
- Do not restate implementation details in higher-level prose. If an exact
  value or formula is needed there, generate it from code or state it once in
  the owning module and link to it.
- When behaviour changes, update the owning code doc/test and correct any prose
  that has become false. Do not add new prose merely to mirror the
  implementation.
- Mention a default or an ablation only where it affects how the code should be
  used; otherwise omit it.
- For rounded or approximate quantities, distinguish target from realized value
  once, in the owning description, rather than repeating the distinction
  throughout the docs.

