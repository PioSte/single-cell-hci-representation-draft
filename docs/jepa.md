# JEPA objective

`src/fuse_channels/jepa.py` implements a joint-embedding predictive objective
on the same channel-fusion backbone as the MAE. It reuses the tokenizer,
per-location channel fusion, encoder, spatial pooler, and decoder blocks; what
changes is the **target space** and the **target set**.

## Why the target is a feature, not a pixel

The MAE reconstructs the pixels of the masked channel-patches
(`src/fuse_channels/mae.py`, `src/fuse_channels/losses.py`). A JEPA predicts a
learned representation instead, so the target encoder can discard
pixel-level nuisance (noise, background, exact intensities) and keep the
predictable, semantic part of the signal. I-JEPA and V-JEPA both find that
predicting in representation space transfers better from a frozen backbone
than predicting pixels, and V-JEPA uses an L1 feature regression for stability
(`references/i-jepa/paper.md`, `references/v-jepa/paper.md`,
`references/v-jepa-2/paper.md`).

The target is not the pooled `[B, L, D]` latent set. The latents are a global
PMA summary: predicting them gives no local error signal, and they are already
the predictor's context key/value. The target is the encoder's **per-location**
token feature. That keeps the predictor's fractional positional embedding
load-bearing and makes the objective spatial.

After channel fusion there is no longer a per-channel feature to predict:
`ChannelFusion` emits one vector per present `(sample, patch)` location, and
channel identity travels as a routing bias plus an output skip. The target is
therefore **channel-fused** and permutation invariant. Predicting the
pre-fusion tokenizer output would only be a linear patch projection, not a
semantic target.

## Pipeline

```
teacher (eval, no grad; EMA copy or the student itself when use_ema=False):
    images (all C, all N) -> tokenizer_bar -> fusion_bar -> encoder_bar
    -> H_bar[N, D] -> gather target locations -> projector_bar -> sg(target) [K_t, D_proj]

student:
    images (context locations; optional visible-channel subset)
    -> tokenizer -> fusion -> encoder -> tok_out[N_ctx, D]
    -> spatial pooling (PMA) -> latents [B, L, D] (+ memory in the memory arm)

predictor:
    q = mask_token + fractional_pe(target row, col, grid, step_um[d])
        + optional scale condition
    q -> decoder blocks cross-attending to the latents
      -> LayerNorm -> predicted features [K_t, D] -> projector -> [K_t, D_proj]

loss:
    L1 (default) or L2 between the projected predicted and stop-grad target
    features, reduced per sample, plus reg_weight * regularizer(projected student
    rows).  The projector is discarded at inference.
```

The target is detached; only the student and predictor receive gradients.
`JEPAConfig.use_ema=True` (default) keeps an independent EMA copy of the
representation path (`tokenizer`, `fusion`, `encoder`, and `scale_embed` when
enabled). After the optimizer step, call
`ChannelFusionJEPA.update_target_encoder()` to apply the EMA; the target
encoder stays in eval mode so dropout is off. `use_ema=False` removes the copy
and uses the student's full-view output as the detached target; it requires a
non-zero regularizer weight because EMA is otherwise the only collapse
prevention.

## Collapse control: EMA vs regularizers

The JEPA name is shared; the collapse-prevention mechanism is a config choice:

| `use_ema` | `regularizer` | Meaning |
| --- | --- | --- |
| `True` | `"none"` | Shipped recipe: EMA target encoder + stop-gradient. |
| `True` | `"sigreg"` / `"vicreg"` | EMA plus an auxiliary distribution regularizer on the projected student rows. |
| `False` | `"sigreg"` / `"vicreg"` | No target copy; detached student full-view target plus the regularizer on the projected rows. |

`SIGReg` (`src/fuse_channels/regularizers.py`) is the LeJEPA Epps-Pulley
objective: `num_slices` random unit directions, a 17-point trapezoid quadrature
on `[-5, 5]` with the Gaussian window `exp(-t²/2)`, and an all-reduced empirical
characteristic function for DDP. `VICReg` is the single-view variance +
covariance regularizer; the invariance term is the prediction loss, so only
`mu * variance_hinge + nu * covariance` is added. Both consume the **projected
student rows** — the same tensor the prediction loss regresses — following
VICReg (the variance/covariance terms act on the discarded expander output) and
LeJEPA (SIGReg and the invariance loss share the projected embedding while the
probe reads the backbone). `encode() -> latents` remains the deployment
representation and is never constrained by the prior.

Regularizer weights are not portable across methods: SIGReg, VICReg and the
prediction loss have different numeric scales, and the statistic's value depends
on the row count and embedding width. `SIGReg(reduce="mean")` (default) drops
the explicit row-count factor of Algorithm 1; `reduce="sum"` reproduces the
published statistic verbatim. `VICReg(bias_correct=True)` removes the
`(D - 1) / (rows - 1)` covariance sampling floor. Sweep `reg_weight` per
regularizer **and** per `(rows, D)` regime, and record that regime next to the
number. The local references are
`references/lejepa/paper.md` (Algorithm 1), `references/lejepa/MINIMAL.md`,
`references/vicreg/paper.md`, and `references/visreg/paper.md`. VISReg is
checked in as a reference but is not yet exposed by `JEPAConfig`; SIGReg and
VICReg are the supported regularizer ablations.

`encode()` is inherited unchanged: it returns the student's no-mask
`(memory, latents)`, where `memory` is empty unless the memory arm is on. The
latent pooler is the predictor's key/value, so it is trained by the JEPA loss
even though the target itself is per-location.

## Target set

`MultiBlockTargetSampler` (`src/fuse_channels/masking.py`) samples a few large
target rectangles, following I-JEPA's multi-block result: random per-patch
masking and small targets underperform because they do not force the predictor
to infer a semantic region (`references/i-jepa/paper.md`, Tables 6, 8-10).
Target locations are removed for **all** channels, so the student cannot copy a
partially visible token at a target position.

The sampler also produces the student's **context block** — I-JEPA's second mask,
with a different distribution from the targets. `context_scale=(1.0, 1.0)`
(the default) makes the context the exact target complement. A lower value draws
one large rectangle and removes the targets from it, so a border strip is neither
target nor context; that is I-JEPA's context-size axis (Table 9: 31.2 at
(0.40, 1.0) vs 54.2 at (0.85, 1.0)) and, in a padding-free model, also an
encoder-cost knob. The targets are sampled *before* the context rectangle, so
changing `context_scale` does not perturb them for a fixed seed.

With `JEPAConfig.channel_dropout=True`, the student context additionally sees
only an HCS-sampled subset of whole channels while the teacher still sees every
channel. The target is then the full-channel fused feature, so the objective
becomes channel completion as well as spatial prediction (the CR-DINO
teacher/student idea, `references/cr-dino/paper.md`). This is the natural use
of the channel-fusion backbone: the student must infer the contribution of
channels it did not see into the fused representation.

DCP's per-channel patch mask is deliberately not reused as the JEPA mask. With
many channels it rarely leaves a location with no visible channel, so a
location-level target set is ill-defined; and a partially masked location
leaks the context token at the same position.

## Differences from the MAE path

| | MAE (`ChannelFusionMAE`) | JEPA (`ChannelFusionJEPA`) |
| --- | --- | --- |
| Target | masked channel-patch pixels | EMA encoder per-location features, optionally projected |
| Mask | DCP channel-patch visibility | multi-block target locations + I-JEPA context block (+ optional whole-channel dropout) |
| Decoder output | flattened `p²` pixels per target row | `D`-wide feature per target location |
| Target identity | target channel embedding in the query | none; the target is channel-fused |
| Collapse control | reconstruction objective | EMA or no-EMA + SIGReg/VICReg on projected rows |
| Inference representation | `encode() -> (memory, latents)` | same |

## Usage

```python
import torch
from fuse_channels import JEPAConfig, build_jepa

# EMA, shipped recipe
model = build_jepa(jepa=JEPAConfig(channel_dropout=True))

# Optional regularizer ablation (EMA + SIGReg)
# model = build_jepa(jepa=JEPAConfig(regularizer="sigreg", reg_weight=0.1))

# No-EMA ablation (requires a non-zero regularizer weight)
# model = build_jepa(jepa=JEPAConfig(use_ema=False, regularizer="sigreg", reg_weight=0.1))

# Disable the projector (predict in encoder feature space)
# model = build_jepa(jepa=JEPAConfig(use_projector=False))

# Restrict the student's context block (I-JEPA context-size axis)
# model = build_jepa(jepa=JEPAConfig(context_scale=(0.5, 1.0)))

opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

for images, ids, pixel_sizes in loader:
    out = model(images, ids, pixel_sizes=pixel_sizes)
    opt.zero_grad()
    out["loss"].backward()
    opt.step()
    model.update_target_encoder()   # no-op when use_ema=False

mem, latents = model.encode(images, ids, pixel_sizes=pixel_sizes)
```

`build_jepa` takes the same `ModelConfig` backbone as `build_model`; the
objective-specific knobs live in `JEPAConfig` because they belong to the
predictor/target pair, not the shared backbone.

## Novelty inventory

This inventory separates what is taken from a reference, what is derived here,
and what is genuinely novel. It is the honest map for a reader comparing the
implementation against I-JEPA/V-JEPA and LeJEPA/VICReg.

| Property | I-JEPA / V-JEPA | LeJEPA / VICReg | Ours | Status |
| --- | --- | --- | --- | --- |
| Predict in feature space | yes | yes (invariance) | yes | cited |
| Multi-block target rectangles | yes | no masking | yes | cited |
| Context block distinct from targets | yes | no | yes (`context_scale`) | cited (I-JEPA Table 9) |
| EMA target encoder + stop-grad | yes | no | yes (default) | cited |
| No-EMA masked prediction + distribution regularizer | no | no | yes | **novel** |
| Regularizer on the aligned tensor, discarded at inference | VICReg expander | LeJEPA projector | yes | cited convention |
| SIGReg `reduce="mean"` row convention | — | Algorithm 1 is `sum` | mean default | **derived** |
| VICReg `bias_correct` covariance floor | — | paper uses the floor | opt-in correction | **derived** |
| Within-sample collapse diagnostic (`feature_spread`) | — | — | yes | **novel** |
| Channel-completion as the augmentation-free second view | — | — | yes (`channel_dropout`) | **novel** |
| Per-location targets on a fused channel representation | — | — | yes | **novel** |

The last four rows are the claims to defend in an experiment; the first six are
borrowed machinery and should be attributed.

## What to ablate first

- **Spatial targets only vs. spatial + channel dropout.** This is the direct
  comparison between I-JEPA-style spatial prediction and the channel-completion
  signal that motivates the fusion tokenizer.
- **Per-location target vs. pooled-latent target.** The global target is the
  Cell-JEPA/CR-DINO-style alternative; it is cheaper and may suit whole-cell
  classification, but it is not spatially localized.
- **L1 vs. L2 feature loss.** V-JEPA reports L1 as more stable; I-JEPA uses L2.
- **EMA vs. no-EMA + SIGReg/VICReg.** The EMA path has no regularizer weight;
  the no-EMA path removes the target copy and must replace EMA with a
  regularizer. Use the same target/mask and compare frozen probes.
- **EMA + regularizer vs. EMA alone.** This isolates whether the regularizer
  helps on top of the EMA rather than replacing it.
- **Projector on vs. off.** The projector is discarded at inference; on turns
  the regularizer into a prior on a throwaway space (the reference convention),
  off regularizes the deployment representation directly. Compare frozen probes.
- **Context block size.** `context_scale=(0.5, 1.0)` vs `(1.0, 1.0)` is
  I-JEPA's Table 9 axis; in a padding-free model it also changes encoder cost.
- **Regularizer weight and slice count.** SIGReg/VICReg scales differ; do not
  reuse one weight. Record `(rows, D)` with the weight. SIGReg's paper ablates
  256/1024/2048 slices and finds the quadrature domain/point count has little
  effect.
