"""JEPA (joint-embedding predictive) objective on the channel-fusion backbone.

The target is **not** a pixel patch.  It is the exponential-moving-average
(EMA) target encoder's final per-location token feature, computed from the
fully observed input (all channels, all patches).  The student sees only a
multi-block spatial context (optionally with whole-channel dropout), and a
feature-space predictor reconstructs the target features at the removed
locations.

Pipeline::

    target:  images (all C, all N) -> tokenizer̄ -> fusion̄ -> encoder̄
             -> H̄ [N, D] -> gather target locations
             -> projector̄ -> stop-grad target [K_t, D_proj]
    student: images (context locations, optional visible channels)
             -> tokenizer -> fusion -> encoder -> tok_out
             -> spatial pooling -> latents [B, L, D] (+ memory, arm only)
    predictor: mask_token + fractional PE(target position) + scale condition
             -> decoder blocks cross-attending to the latents
             -> predicted target features [K_t, D] -> projector [K_t, D_proj]

The loss is L1 (V-JEPA) or L2 (I-JEPA) between predicted and stop-grad target
features, reduced per sample.  The target encoder's parameters are updated as
an EMA of the student's ``tokenizer``, ``fusion``, ``encoder``,
``scale_embed`` and ``jepa_projector`` after each optimizer step; the target encoder never receives a
gradient.  The predictor is the existing decoder block stack, so the pooled
latents remain the predictor's KV and are therefore trained by the JEPA loss
rather than by a pixel reconstruction head.

This module deliberately targets *locations*, not channel-patches: after
channel fusion there is no longer a per-channel feature to predict.  Spatial
multi-block targets make the target set well defined; optional whole-channel
dropout turns the objective into channel completion (the CR-DINO teacher/student
idea) while keeping the teacher's target channel-fused and permutation
invariant.

Deviations from the references in `references/`, all deliberate; the
inventory is in `docs/jepa.md`:

* **No-EMA masked prediction.**  I-JEPA/V-JEPA are masked prediction *with*
  an EMA target encoder; LeJEPA is *invariance* without a target encoder and
  without masking.  The ``use_ema=False`` recipe is neither of those, so it
  is our design and has to be validated empirically, not by citation.
* **The regularizer constrains the aligned tensor.**  The anti-collapse term
  consumes the same projected rows the prediction loss regresses, following
  VICReg (the variance/covariance terms act on the expander output, which is
  discarded) and LeJEPA (SIGReg and the invariance loss share the projected
  embedding while the probe reads the backbone).  Regularising the *pooled
  latents* instead leaves the aligned rows unconstrained and only looks safe
  because the probe reads the latents.
* **The projector is discarded at inference.**  ``encode() -> latents`` stays
  the deployment representation, so the distributional prior never has to
  hold on the tensor downstream tasks read.
"""

from __future__ import annotations

from copy import deepcopy

import torch
import torch.nn as nn

from .config import DEFAULT_CONFIG, JEPAConfig, ModelConfig
from .losses import feature_prediction_loss
from .mae import ChannelFusionMAE, _n_patches_for
from .masking import HierarchicalChannelSampler, MultiBlockTargetSampler
from .regularizers import build_regularizer


class ChannelFusionJEPA(ChannelFusionMAE):
    """Channel-fusion JEPA: EMA target encoder, multi-block targets, feature predictor.

    The model reuses every component of :class:`~fuse_channels.mae.ChannelFusionMAE`
    (tokenizer, fusion, encoder, latent pooler, decoder blocks) and changes the
    objective:

    * the target is the target encoder's per-location token feature;
    * the decoder predicts features, not pixels;
    * the mask is a spatial multi-block target mask, not DCP's per-channel
      channel-patch mask.

    The context is ``MultiBlockTargetSampler.context_masks`` (the target
    complement at ``context_scale=(1.0, 1.0)``, a context rectangle minus the
    targets below it), optionally intersected with an HCS whole-channel subset
    from ``JEPAConfig.channel_dropout``.  The student therefore drops tokens for
    two reasons: *target* locations (never visible at any channel) and *outside
    the context block*.  ``forward`` takes ``target_locations=``; there is no
    ``visible=`` alias, because a DCP-style ``[C, N]`` channel-patch mask is not
    a valid JEPA target set.  ``encode()`` is inherited and returns the
    student's no-mask ``(memory, latents)`` (``memory`` is empty unless the
    memory arm is on).

    The prediction loss and the anti-collapse regularizer act on the **same**
    tensor: :meth:`_project` maps the predicted and target locations through
    ``jepa_projector``, and the regularizer consumes the student rows of that
    projection.  The projector (and its EMA copy on the target side) is
    discarded at inference, so the deployment representation stays
    ``encode() -> latents``.  This is our adaptation of the two-reference
    convention (VICReg: the variance/covariance terms act on the expander
    output, which is discarded; LeJEPA: SIGReg and the invariance loss share the
    projected embedding while the probe reads the backbone).
    """

    jepa_projector: nn.Sequential | None
    target_projector: nn.Sequential | None

    def __init__(self, *, jepa: JEPAConfig | None = None, **mae_kwargs) -> None:
        super().__init__(**mae_kwargs)
        self.jepa = jepa or JEPAConfig()
        self.use_ema = self.jepa.use_ema

        # The target encoder is an independent, slowly moving copy of the
        # representation path.  The decoder/pooler are not copied: the target is
        # per-location `tok_out`, not a pooled or reconstructed output.  With
        # `use_ema=False` the target is a detached full-view forward of the
        # student, and a regularizer is required for collapse prevention.
        if self.use_ema:
            self.target_tokenizer = deepcopy(self.tokenizer)
            self.target_fusion = deepcopy(self.fusion)
            self.target_encoder = deepcopy(self.encoder)
            self.target_scale_embed = deepcopy(self.scale_embed)
        else:
            self.target_tokenizer = None
            self.target_fusion = None
            self.target_encoder = None
            self.target_scale_embed = None

        self.jepa_projector = self._build_projector()
        if self.use_ema and self.jepa_projector is not None:
            self.target_projector = deepcopy(self.jepa_projector)
        else:
            # No separate copy without EMA: the target is a detached student
            # forward, so the shared projector produces both sides (``_project``
            # falls back to it when ``target_projector`` is None).
            self.target_projector = None

        self.regularizer = build_regularizer(self.jepa.regularizer, **self.jepa.reg_kwargs)
        self.jepa_masker = MultiBlockTargetSampler(
            n_blocks=self.jepa.target_blocks,
            target_scale=self.jepa.target_scale,
            target_aspect=self.jepa.target_aspect,
            context_scale=self.jepa.context_scale,
        )
        self.channel_sampler = HierarchicalChannelSampler()

        for module in self._target_modules():
            for param in module.parameters():
                param.requires_grad_(False)
        self._set_target_eval()

    # ------------------------------------------------------------------ #
    def _target_modules(self) -> list[nn.Module]:
        modules: list[nn.Module] = []
        for module in (self.target_tokenizer, self.target_fusion, self.target_encoder):
            if module is not None:
                modules.append(module)
        if self.target_scale_embed is not None:
            modules.append(self.target_scale_embed)
        if self.target_projector is not None:
            modules.append(self.target_projector)
        return modules

    def _build_projector(self) -> nn.Sequential | None:
        """Per-token MLP for the prediction/regularizer space, or ``None``.

        Applied per token, so it cannot mix channels or locations and the
        channel-permutation invariance of the target is preserved.  No batch
        normalization: the projector sits on the EMA target path, and a
        batch-dependent target would make the target of one sample depend on the
        other samples in the step.
        """
        if not self.jepa.use_projector:
            return None
        D = int(self.core.dim)
        hidden = D if self.jepa.proj_hidden is None else int(self.jepa.proj_hidden)
        out = D if self.jepa.proj_dim is None else int(self.jepa.proj_dim)
        return nn.Sequential(nn.Linear(D, hidden), nn.GELU(), nn.Linear(hidden, out))

    @property
    def _proj_out_dim(self) -> int:
        """Width of the tensor the prediction loss and the regularizer share."""
        if self.jepa_projector is None:
            return int(self.core.dim)
        last = self.jepa_projector[-1]
        assert isinstance(last, nn.Linear)
        return int(last.out_features)

    def _project(self, features: torch.Tensor, *, target: bool = False) -> torch.Tensor:
        """Map location features into the shared prediction/regularizer space.

        With ``use_ema=False`` there is no separate target projector: the target
        is a detached student forward, so both sides go through the shared
        ``jepa_projector``.
        """
        projector = self.target_projector if target else self.jepa_projector
        if projector is None:
            projector = self.jepa_projector
        if projector is None:
            return features
        return projector(features)

    def _set_target_eval(self) -> None:
        for module in self._target_modules():
            module.eval()

    def train(self, mode: bool = True):
        """Keep the EMA target encoder in eval mode (dropout off) while training."""
        super().train(mode)
        self._set_target_eval()
        return self

    @torch.no_grad()
    def update_target_encoder(self, momentum: float | None = None) -> None:
        """Polyak/EMA update of the target workspace after an optimizer step.

        No-op when ``JEPAConfig.use_ema=False``; the no-EMA path uses detached
        student features on the full view instead of a target copy.
        """
        if not self.use_ema:
            return
        m = self.jepa.ema_momentum if momentum is None else float(momentum)
        if not 0.0 <= m < 1.0:
            raise ValueError(f"momentum must be in [0, 1), got {m}")
        assert self.target_tokenizer is not None
        assert self.target_fusion is not None
        assert self.target_encoder is not None
        pairs: list[tuple[nn.Module, nn.Module]] = [
            (self.tokenizer, self.target_tokenizer),
            (self.fusion, self.target_fusion),
            (self.encoder, self.target_encoder),
        ]
        if self.scale_embed is not None:
            assert self.target_scale_embed is not None
            pairs.append((self.scale_embed, self.target_scale_embed))
        if self.jepa_projector is not None:
            assert self.target_projector is not None
            pairs.append((self.jepa_projector, self.target_projector))
        for student, target in pairs:
            for sp, tp in zip(student.parameters(), target.parameters(), strict=True):
                tp.mul_(m).add_(sp, alpha=1.0 - m)
            for sb, tb in zip(student.buffers(), target.buffers(), strict=True):
                tb.copy_(sb)

    # ------------------------------------------------------------------ #
    def _context_visibility(self, images, target_locations, context_locations, ns, generator):
        """Per-sample ``[C, N]`` visible mask for the student context.

        Target locations are already removed from ``context_locations`` for
        every channel.  With ``channel_dropout`` the student additionally sees
        only an HCS-sampled subset of whole channels; the teacher still sees all
        of them.
        """
        device = images[0].device
        out = []
        for b, (im, target, context, N) in enumerate(
            zip(images, target_locations, context_locations, ns, strict=True)
        ):
            C = int(im.shape[0])
            target = target.to(device=device, dtype=torch.bool)
            if target.numel() != N:
                raise ValueError(
                    f"target_locations[{b}] must have {N} entries, got {target.numel()}"
                )
            locations = context.to(device=device, dtype=torch.bool)
            if locations.numel() != N:
                raise ValueError(
                    f"context_locations[{b}] must have {N} entries, got {locations.numel()}"
                )
            overlap = locations & target
            if bool(overlap.any()):
                raise ValueError(
                    f"context_locations[{b}] must be disjoint from the target locations "
                    f"({int(overlap.sum())} overlap)"
                )
            if self.jepa.channel_dropout:
                min_visible = min(self.jepa.min_visible_channels, C)
                channels = self.channel_sampler(
                    C, device=device, generator=generator, min_visible=min_visible
                )
                visible = locations[None, :] & channels[:, None]
            else:
                visible = locations[None, :].expand(C, N).clone()
            if not bool(visible.any()):
                # Keep the padding-free path defined for tiny grids / aggressive
                # dropout: reveal one channel at one non-target location.
                free = locations.nonzero(as_tuple=False)
                if free.numel() == 0:
                    raise ValueError("JEPA context is empty: all locations are targets")
                c = int(torch.randint(C, (1,), device=device, generator=generator).item())
                n = int(
                    free[torch.randint(free.shape[0], (1,), device=device, generator=generator)]
                )
                visible[c, n] = True
            out.append(visible)
        return out

    # ------------------------------------------------------------------ #
    def forward(
        self,
        images,
        channel_ids,
        visible=None,
        generator: torch.Generator | None = None,
        patch_sizes=None,
        pixel_sizes=None,
        valid=None,
        target_locations=None,
        context_locations=None,
    ):
        """Compute the JEPA loss and return the predicted/target features.

        ``target_locations`` is an optional list of per-sample ``[N_b]`` boolean
        location masks (True = predict).  If omitted, multi-block targets are
        sampled.  ``context_locations`` optionally overrides the student's
        spatial context (default: the sampler's context block, disjoint from the
        targets).  The returned ``pred``/``target`` are ``[K_t, D_proj]`` and
        already detached on the target side.

        ``visible`` is accepted only for signature compatibility with
        :meth:`ChannelFusionMAE.forward`; a DCP ``[C, N]`` channel-patch mask is
        not a valid JEPA target set, so passing it is an error.
        """
        if visible is not None:
            raise TypeError(
                "ChannelFusionJEPA.forward takes target_locations= or "
                "context_locations=; a DCP [C, N] channel-patch mask is not a "
                "valid JEPA target set"
            )
        B = len(images)
        device = images[0].device
        ps = self._resolve_patch_sizes(images, patch_sizes, generator)
        pixel_sizes = self._resolve_pixel_sizes(B, pixel_sizes)
        ns = [_n_patches_for(im, p) for im, p in zip(images, ps, strict=True)]
        grid_hw = [(im.shape[-2] // p, im.shape[-1] // p) for im, p in zip(images, ps, strict=True)]

        if target_locations is None:
            target_locations = self.jepa_masker.sample_targets(grid_hw, generator=generator)
        else:
            target_locations = [t.clone() for t in target_locations]
        if len(target_locations) != B:
            raise ValueError(f"expected {B} target location masks, got {len(target_locations)}")
        if context_locations is None:
            context_locations, target_locations = self.jepa_masker.context_masks(
                target_locations, grid_hw, generator=generator
            )
        elif len(context_locations) != B:
            raise ValueError(f"expected {B} context location masks, got {len(context_locations)}")
        for b, (target, N) in enumerate(zip(target_locations, ns, strict=True)):
            if target.numel() != N:
                raise ValueError(
                    f"target_locations[{b}] must have {N} entries, got {target.numel()}"
                )
        target_locations = [t.to(device=device, dtype=torch.bool) for t in target_locations]
        context_locations = [c.to(device=device, dtype=torch.bool) for c in context_locations]

        context_visible = self._context_visibility(
            images, target_locations, context_locations, ns, generator
        )
        cs, ps, ns, _visible, _target_valid, pixel_sizes = self._prepare(
            images,
            channel_ids,
            context_visible,
            generator,
            patch_sizes=ps,
            pixel_sizes=pixel_sizes,
            valid=valid,
        )
        if cs.tokens.shape[0] == 0:
            raise ValueError(
                "JEPA context is empty after the target and valid masks; "
                "reduce the target block scale or reveal at least one context token"
            )

        scale_values = None
        scale_emb = None
        if self.scale_embed is not None:
            scale_values = self._scale_values(ps, pixel_sizes, device)
            scale_emb = self.scale_embed(scale_values)

        # The teacher sees the complete observation (all channels, all patches).
        if self.use_ema:
            assert self.target_tokenizer is not None
            assert self.target_fusion is not None
            assert self.target_encoder is not None
            teacher_tokenizer = self.target_tokenizer
            teacher_fusion = self.target_fusion
            teacher_encoder = self.target_encoder
        else:
            teacher_tokenizer = self.tokenizer
            teacher_fusion = self.fusion
            teacher_encoder = self.encoder

        with torch.no_grad():
            all_visible = [
                torch.ones(int(im.shape[0]), N, dtype=torch.bool, device=device)
                for im, N in zip(images, ns, strict=True)
            ]
            cs_t, _, ns_t, _, _, pixel_sizes_t = self._prepare(
                images,
                channel_ids,
                all_visible,
                generator,
                patch_sizes=ps,
                pixel_sizes=pixel_sizes,
                valid=valid,
                tokenizer=teacher_tokenizer,
            )
            assert ns_t == ns and pixel_sizes_t == pixel_sizes
            if self.use_ema:
                if self.scale_embed is not None:
                    assert self.target_scale_embed is not None
                    target_scale_emb = self.target_scale_embed(scale_values)
                else:
                    target_scale_emb = None
            else:
                target_scale_emb = scale_emb.detach() if scale_emb is not None else None
            (
                _teacher_fused,
                teacher_present,
                _teacher_mem,
                teacher_tok,
                _teacher_group,
                teacher_present_bi,
            ) = self._fuse_and_encode(
                cs_t,
                target_scale_emb,
                fusion=teacher_fusion,
                encoder=teacher_encoder,
            )

        (
            _fused,
            _present,
            mem_out,
            _tok_out,
            latents,
            _enc_group_id,
            _present_bi,
        ) = self._encode(cs, ps, scale_emb)

        # Index teacher tokens by (sample, patch) so target locations can be
        # gathered without building a dense [B, C*N, D] tensor.
        max_n = max(ns) if ns else 0
        teacher_row = torch.full((B, max_n), -1, dtype=torch.long, device=device)
        teacher_row[teacher_present_bi[:, 0], teacher_present_bi[:, 1]] = torch.arange(
            teacher_tok.shape[0], device=device
        )

        tgt_group_id, tgt_patch = [], []
        tgt_row, tgt_col, tgt_gh, tgt_gw, tgt_p = [], [], [], [], []
        for b, (target, N, p) in enumerate(zip(target_locations, ns, ps, strict=True)):
            selected = target.clone()
            if valid is not None:
                v = valid[b].to(device=device, dtype=torch.bool)
                if v.numel() != N:
                    raise ValueError(f"valid[{b}] must have {N} entries, got {v.numel()}")
                selected &= v
            selected &= teacher_present[b]
            pi = selected.nonzero(as_tuple=True)[0]
            gw = grid_hw[b][1]
            gh = grid_hw[b][0]
            tgt_group_id.append(torch.full_like(pi, b))
            tgt_patch.append(pi)
            tgt_row.append(pi // gw)
            tgt_col.append(pi % gw)
            tgt_gh.append(torch.full_like(pi, gh))
            tgt_gw.append(torch.full_like(pi, gw))
            tgt_p.append(torch.full_like(pi, p))

        if tgt_group_id:
            tgt_group_id = torch.cat(tgt_group_id, 0)
            tgt_patch = torch.cat(tgt_patch, 0)
            tgt_row = torch.cat(tgt_row, 0)
            tgt_col = torch.cat(tgt_col, 0)
            tgt_gh = torch.cat(tgt_gh, 0)
            tgt_gw = torch.cat(tgt_gw, 0)
            tgt_p = torch.cat(tgt_p, 0)
        else:
            tgt_group_id = torch.empty(0, dtype=torch.long, device=device)
            tgt_patch = tgt_row = tgt_col = tgt_gh = tgt_gw = tgt_p = tgt_group_id

        if tgt_group_id.numel() == 0:
            zero = latents.sum() * 0.0
            return {
                "loss": zero,
                "jepa_loss": zero,
                "reg_loss": zero,
                "reg_rows": 0,
                "reg_dim": self._proj_out_dim,
                "pred": None,
                "target": None,
                "latents": latents,
                "target_locations": target_locations,
                "context_locations": context_locations,
                "present": teacher_present,
                "patch_sizes": ps,
                "grid_h": cs.grid_h,
                "grid_w": cs.grid_w,
                "n_patches_per_sample": cs.n_patches_per_sample,
                "scale_values": scale_values,
                "pixel_size": cs.pixel_size,
                "pe_metric": self._use_metric_pe(cs),
            }

        target_row = teacher_row[tgt_group_id, tgt_patch]
        # Both sides of the prediction loss live in the projector's space, and
        # the regularizer consumes exactly the student rows of that same tensor
        # (see the class docstring).  The target keeps its own EMA copy of the
        # projector, so it is a stable regression target within a step.
        target_feat = self._project(teacher_tok[target_row], target=True).detach()

        enc_all, enc_all_group_id = self._decoder_kv(mem_out, latents)
        cond_dec = None
        target_condition = None
        if scale_emb is not None:
            if self.scale_conditioning == "add":
                target_condition = scale_emb[tgt_group_id]
            elif self.scale_conditioning == "adaln":
                cond_dec = scale_emb[tgt_group_id]

        pred = self.decoder.predict_features(
            enc_all,
            enc_all_group_id,
            B,
            tgt_group_id,
            tgt_patch,
            target_row=tgt_row,
            target_col=tgt_col,
            target_grid_h=tgt_gh,
            target_grid_w=tgt_gw,
            target_step_um=self._target_step_um(cs, tgt_group_id, tgt_p, pixel_sizes),
            target_condition=target_condition,
            cond=cond_dec,
        )
        pred = self._project(pred)
        jepa_loss = feature_prediction_loss(
            pred,
            target_feat,
            tgt_group_id,
            kind=self.jepa.loss,
            reduce=self.loss_reduce,
        )
        reg_loss = self.regularizer(pred)
        loss = jepa_loss + self.jepa.reg_weight * reg_loss
        return {
            "loss": loss,
            "jepa_loss": jepa_loss,
            "reg_loss": reg_loss,
            "reg_rows": int(pred.shape[0]),
            "reg_dim": int(pred.shape[1]),
            "pred": pred,
            "target": target_feat,
            "target_group_id": tgt_group_id,
            "target_patch": tgt_patch,
            "target_patch_size": tgt_p,
            "target_row": tgt_row,
            "target_col": tgt_col,
            "target_grid_h": tgt_gh,
            "target_grid_w": tgt_gw,
            "target_locations": target_locations,
            "context_locations": context_locations,
            "latents": latents,
            "present": teacher_present,
            "patch_sizes": ps,
            "grid_h": cs.grid_h,
            "grid_w": cs.grid_w,
            "n_patches_per_sample": cs.n_patches_per_sample,
            "scale_values": scale_values,
            "pixel_size": cs.pixel_size,
            "pe_metric": self._use_metric_pe(cs),
        }


def build_jepa(
    base: ModelConfig | None = None,
    jepa: JEPAConfig | None = None,
    **overrides,
) -> ChannelFusionJEPA:
    """Build a JEPA model from the authored backbone config plus JEPA knobs."""
    cfg = base or DEFAULT_CONFIG
    kwargs = cfg.to_kwargs()
    kwargs.update(overrides)
    return ChannelFusionJEPA(jepa=jepa or JEPAConfig(), **kwargs)
