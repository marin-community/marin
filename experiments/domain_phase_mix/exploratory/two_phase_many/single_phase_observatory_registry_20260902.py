# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy"]
# ///
"""Registry of single-phase Observatory restrictions, equivalence classes, references, and ablations.

Every Observatory ``MODEL_IDS`` entry maps to exactly one single-phase model here. Two source ids
that reduce to the same tied-input model share one equivalence class and are benchmarked once.
Ablations are one-factor mechanism removals or matched controls of a parent, expressed through the
same builders so the only difference between a parent and its ablation is the named mechanism.
"""

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    single_phase_observatory_models_20260902 as models,
)

Builder = Callable[[models.Features], Any]

OBSERVATORY_MODEL_IDS = (
    "linear",
    "olmix_loglinear",
    "canonical",
    "effective_exposure",
    "effective_exposure_geometry",
    "separate_heads",
    "grp",
    "compact_retained_state",
    "bucket_family_grp",
    "hierarchical_phase_bucket_replay",
    "crs_plus",
    "crs_bounded",
    "hpr_band",
    "bucket_family_power_separate_heads",
    "bucket_family_power_separate_heads_family_onset",
    "bucket_family_weibull_shared_onset",
    "bucket_family_weibull_family_replay",
    "retained_power_law",
)
HIDDEN_OBSERVATORY_IDS = frozenset(
    {
        "bucket_family_power_separate_heads_family_onset",
        "bucket_family_weibull_shared_onset",
        "bucket_family_weibull_family_replay",
    }
)
FEATURE_TRANSFORMS = (
    "permuted_inventory",
    "weight_coordinate",
    "shuffled_families",
    "no_families",
    "outcome_permutation",
)
TRANSFORM_SEED = 20_260_902
BUCKET_FAMILY_RIDGE_GRID = (0.0, 1e-3, 1e-2, 0.1, 1.0, 3.0)
HIERARCHICAL_RIDGE_GRID = (0.0, 1e-3, 1e-2, 0.1, 1.0, 10.0)
HIERARCHICAL_RESIDUAL_GRID = (1.0, 3.0, 10.0, 30.0, 100.0)
HIERARCHICAL_TOP_SHAPES = 3
POWER_HEADS_RIDGE_GRID = (0.0, 1e-3, 1e-2, 0.1, 1.0)
RETAINED_GRP_RIDGE_GRID = (0.0, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0)
CRS_RIDGE_GRID = (0.0, 0.01, 0.1, 1.0)
# The parent grid with its zero replaced by a tiny ridge so the signed solve stays defined.
SIGNED_RIDGE_GRID = (1e-8, 1e-3, 1e-2, 0.1, 1.0, 3.0)


@dataclasses.dataclass(frozen=True)
class ModelEntry:
    """One benchmarked single-phase model and its provenance."""

    model_id: str
    role: str  # parent | reference | ablation | control
    source_model_ids: tuple[str, ...]
    equivalence_class: str
    display: str
    mechanisms: dict[str, str]
    removed_phase_terms: str
    allowed_metadata: str
    solver_id: str
    hyperparameter_grid: str
    build: Builder
    parent: str | None = None
    ablated_mechanism: str | None = None
    feature_transform: str | None = None
    note: str = ""

    @property
    def visible(self) -> str:
        if not self.source_model_ids:
            return "n/a"
        return "hidden" if all(source in HIDDEN_OBSERVATORY_IDS for source in self.source_model_ids) else "visible"


def _mechanisms(**overrides: str) -> dict[str, str]:
    base = {
        "coordinate": "exposure",
        "benefit": "-",
        "harm": "none",
        "link": "identity",
        "sharing": "shared_shape",
        "families": "none",
        "geometry": "none",
        "head": "nonnegative",
        "estimator": "squared",
    }
    base.update(overrides)
    return base


def _union_shapes(*groups: tuple[models.Shape, ...]) -> tuple[models.Shape, ...]:
    seen: dict[str, models.Shape] = {}
    for group in groups:
        for shape in group:
            seen.setdefault(models._shape_key(shape), dict(shape))
    return tuple(seen.values())


def _family_grid_model(
    model_id: str,
    options: models.FamilyOptions,
    shapes: tuple[models.Shape, ...],
    ridge_grid: tuple[float, ...],
    head: models.HeadSpec,
    shape_dof: int,
    dedupe: tuple[str, ...] = (),
) -> Builder:
    def build(features: models.Features) -> models.GridModel:
        del features
        return models.GridModel(
            model_id,
            lambda feats, shape: models.family_design(feats, shape, options),
            shapes,
            ridge_grid,
            head,
            shape_dof,
            dedupe,
        )

    return build


BUCKET_FAMILY_SHAPES = _union_shapes(models.bucket_family_shapes(24), models.retained_grp_shapes("power_global_tau", 16))
POWER_HEADS_SHAPES = models.retained_grp_shapes("power_global_tau", 16)
HPR_SCREEN_SHAPES = models.bucket_family_shapes(12)
WEIBULL_ONSET_SHAPES = models.retained_grp_shapes("weibull_global_tau", 32)
WEIBULL_REPLAY_SHAPES = models.retained_grp_shapes("weibull_family_coverage_family_replay", 32)
# Budget-matched grids for the benefit-response ablations: 42 Weibull shapes against the 42-shape power
# grid of bucket_family_power_grp, and 34 power shapes against the 34-shape Weibull grid of the shared-onset model.
WEIBULL_MATCHED_SHAPES = models.retained_grp_shapes("weibull_global_tau", 40)
POWER_MATCHED_SHAPES = models.retained_grp_shapes("power_global_tau", 32)
BUCKET_FAMILY_OPTIONS = models.FamilyOptions(family_signal="sum", harm="softplus_family", benefit="power")
HPR_OPTIONS = models.FamilyOptions(
    family_signal="sum", hierarchical=True, harm="softplus_family", member_replay=True, benefit="power"
)
WEIBULL_ONSET_OPTIONS = models.FamilyOptions(family_signal="mean", harm="softplus_family", benefit="weibull")
WEIBULL_REPLAY_OPTIONS = models.FamilyOptions(family_signal="mean", harm="literal_family", benefit="weibull")
NNLS = models.HeadSpec(kind=models.HeadKind.NNLS)
NNLS_SCALED = models.HeadSpec(kind=models.HeadKind.NNLS, scale_columns=True)
LOG_DEFICIT_SCALED = models.HeadSpec(kind=models.HeadKind.NNLS, scale_columns=True, link=models.LinkKind.LOG_DEFICIT)
SIGNED = models.HeadSpec(kind=models.HeadKind.RIDGE)


def _hpr_screen(model_id: str) -> models.GridModel:
    return models.GridModel(
        f"{model_id}:screen",
        lambda feats, shape: models.family_design(feats, shape, BUCKET_FAMILY_OPTIONS),
        HPR_SCREEN_SHAPES,
        HIERARCHICAL_RIDGE_GRID,
        NNLS,
        2,
    )


def _hierarchical(model_id: str, options: models.FamilyOptions) -> Builder:
    def build(features: models.Features) -> models.HierarchicalModel:
        del features
        return models.HierarchicalModel(
            model_id,
            _hpr_screen(model_id),
            lambda feats, shape: models.family_design(feats, shape, options),
            HIERARCHICAL_RESIDUAL_GRID,
            HIERARCHICAL_TOP_SHAPES,
        )

    return build


def _band(model_id: str) -> Builder:
    def build(features: models.Features) -> models.BandModel:
        return models.BandModel(model_id, _hierarchical(f"{model_id}:base", HPR_OPTIONS)(features))

    return build


def _family_onset(model_id: str) -> Builder:
    def build(features: models.Features) -> models.FamilyOnsetModel:
        del features
        shared = models.GridModel(
            f"{model_id}:shared",
            lambda feats, shape: models.family_design(feats, shape, BUCKET_FAMILY_OPTIONS),
            POWER_HEADS_SHAPES,
            POWER_HEADS_RIDGE_GRID,
            NNLS,
            2,
        )
        return models.FamilyOnsetModel(model_id, shared)

    return build


def _retained_power_law(model_id: str, **overrides: Any) -> Builder:
    def build(features: models.Features) -> models.RetainedPowerLawModel:
        phase_0_fraction = float(np.median(features.early_fraction))
        return models.RetainedPowerLawModel(
            model_id, models.retained_power_law_shapes(phase_0_fraction), models.RPL_RIDGE_GRID, **overrides
        )

    return build


def _grp_pair(
    model_id: str,
    *,
    harm: str = "softplus_group_sum",
    discount: bool = True,
    scrambled: bool = False,
    row_scrambled: bool = False,
) -> Builder:
    options = models.FamilyOptions(
        bucket_signal=False,
        family_signal="pair_discount",
        harm=harm,
        benefit="power",
        scrambled_harm=scrambled,
        row_scrambled_harm=row_scrambled,
    )

    def build(features: models.Features) -> models.GridModel:
        shapes = models.grp_pair_shapes(discount=discount and bool(features.families.pairs))
        return models.GridModel(
            model_id,
            lambda feats, shape: models.family_design(feats, shape, options),
            shapes,
            models.GRP_PAIR_RIDGE_GRID,
            NNLS,
            3,
        )

    return build


def _crs_plus(
    model_id: str,
    *,
    retention_gate: bool = True,
    family_benefit: bool = True,
    overload: bool = True,
    literal: bool = True,
) -> Builder:
    def builder(features: models.Features, shape: models.Shape) -> models.Design:
        design = models.crs_plus_design(features, shape)
        keep = np.ones(len(design.names), dtype=bool)
        for index, name in enumerate(design.names):
            if (
                (not family_benefit and name.startswith("family_benefit:"))
                or (not overload and name.startswith("family_overload:"))
                or (not literal and name == "shared_literal_replay")
            ):
                keep[index] = False
        return models.Design(
            design.values[:, keep],
            design.ridge[keep],
            tuple(name for name, flag in zip(design.names, keep, strict=True) if flag),
        )

    shapes = models.crs_plus_shapes()
    if not retention_gate:
        shapes = tuple(shape for shape in shapes if shape["late_multiplier"] == 1.0 and shape["forgetting_rate"] == 0.0)

    def build(features: models.Features) -> models.GridModel:
        del features
        return models.GridModel(
            model_id, builder, shapes, CRS_RIDGE_GRID, NNLS_SCALED, 5 if retention_gate else 3, ("late_multiplier",)
        )

    return build


def _crs_bounded(
    model_id: str,
    *,
    link: models.LinkKind = models.LinkKind.LOG_DEFICIT,
    retention_gate: bool = True,
    replay: bool = True,
) -> Builder:
    options = models.FamilyOptions(
        family_signal="none", harm="literal_shared" if replay else "none", benefit="weibull", retention_gate=True
    )
    shapes = models.crs_bounded_shapes()
    if not retention_gate:
        shapes = tuple(shape for shape in shapes if shape["late_multiplier"] == 1.0 and shape["forgetting_rate"] == 0.0)
    head = models.HeadSpec(kind=models.HeadKind.NNLS, scale_columns=True, link=link)

    def build(features: models.Features) -> models.GridModel:
        del features
        return models.GridModel(
            model_id,
            lambda feats, shape: models.family_design(feats, shape, options),
            shapes,
            CRS_RIDGE_GRID,
            head,
            4 if retention_gate else 2,
            ("late_multiplier",),
        )

    return build


def _dsp(model_id: str, **overrides: Any) -> Builder:
    def build(features: models.Features) -> models.ProfiledDspModel:
        del features
        return models.ProfiledDspModel(model_id, models.DspOptions(**overrides))

    return build


def _compact(model_id: str, **overrides: Any) -> Builder:
    def build(features: models.Features) -> models.CompactRetainedModel:
        del features
        return models.CompactRetainedModel(model_id, **overrides)

    return build


def _log_link(model_id: str, *, link: models.LinkKind = models.LinkKind.LOG_FLOOR_MARGIN) -> Builder:
    def build(features: models.Features) -> models.GridModel:
        del features
        shapes = models.log_link_shapes() if link is models.LinkKind.LOG_FLOOR_MARGIN else ({"floor_margin": 0.0},)
        return models.GridModel(
            model_id,
            models.log_epoch_design,
            shapes,
            models.LOG_LINK_RIDGE_GRID,
            models.HeadSpec(kind=models.HeadKind.RIDGE, link=link),
            0,
        )

    return build


def _static(model: Any) -> Builder:
    def build(features: models.Features) -> Any:
        del features
        return model

    return build


PARENTS: tuple[ModelEntry, ...] = (
    ModelEntry(
        "linear_weight",
        "parent",
        ("linear",),
        "affine_weight",
        "Linear",
        _mechanisms(
            coordinate="weight", benefit="affine", sharing="per_bucket", head="signed", estimator="least_squares"
        ),
        "phase-1 weight columns duplicate the phase-0 columns at a tied policy; one weight block remains",
        "none",
        "centered_minimum_norm_lstsq",
        "none",
        _static(models.LinearWeightModel()),
        note="Affine in weights equals affine in exposures because E_b = c_b w_b, so the coordinate flag is inert here.",
    ),
    ModelEntry(
        "olmix_loglinear_taskwise",
        "parent",
        ("olmix_loglinear",),
        "olmix_positive_loglinear",
        "OLMix log-linear (taskwise)",
        _mechanisms(
            coordinate="weight",
            benefit="exp_linear",
            link="positive_log",
            sharing="per_bucket",
            head="signed",
            estimator="huber_multistart",
        ),
        "the aggregate policy alpha0 w0 + alpha1 w1 equals w; no phase parameters exist",
        "none",
        "olmix_loglinear_fit.fit_olmix_loglinear_model (48 starts, Huber delta 0.02, numerical gradient)",
        "none",
        _static(models.OlmixTaskwiseModel(analytic_gradient=False)),
        note=(
            "The repository solver differentiates numerically and stops well short of the Huber optimum; that "
            "early stop is the only regularization the positive log-linear law has. An analytic-gradient solver "
            "on the same objective reaches lower training loss with coefficients up to 196 and explodes out of "
            "fold, so it is kept as the estimator ablation rather than the parent."
        ),
    ),
    ModelEntry(
        "dsp_total_exposure",
        "parent",
        ("canonical", "effective_exposure"),
        "dsp_total_exposure",
        "Canonical DSP (single phase)",
        _mechanisms(benefit="exp_saturation", harm="softplus_log_quadratic", sharing="per_bucket"),
        "canonical: the phase-1 share gain gamma multiplies a share that is constant at tied "
        "inputs; effective exposure: the phase-1 multiplier rescales E and is absorbed by rho_b "
        "and tau_b (the Observatory's own `no_phase` variant)",
        "none",
        "profiled_inner_cv_lbfgs_implicit_gradient (maxiter 36, 2 restarts, ridge 1e-6)",
        "continuous: log rho_b in [log 1e-4, log 2], tau_b in [-2, 8]",
        _dsp("dsp_total_exposure"),
    ),
    ModelEntry(
        "dsp_total_exposure_concentration",
        "parent",
        ("effective_exposure_geometry",),
        "dsp_total_exposure_concentration",
        "Canonical DSP + aggregate concentration",
        _mechanisms(
            benefit="exp_saturation",
            harm="softplus_log_quadratic",
            sharing="per_bucket",
            geometry="aggregate_concentration",
        ),
        "phase TV is identically zero and late-phase concentration duplicates aggregate "
        "concentration; one sum(w^2) column remains",
        "none",
        "profiled_inner_cv_lbfgs_implicit_gradient (maxiter 36, 2 restarts, ridge 1e-6)",
        "continuous: log rho_b, tau_b as canonical",
        _dsp("dsp_total_exposure_concentration", concentration=True),
    ),
    ModelEntry(
        "asymmetric_log_bowl",
        "parent",
        ("separate_heads",),
        "asymmetric_log_bowl",
        "Separate heads (one-phase bowl)",
        _mechanisms(benefit="asymmetric_log_bowl", harm="asymmetric_log_bowl", sharing="per_bucket_center"),
        "the early and late bowls collapse to one bowl in total exposure (the Observatory's `one_phase` policy)",
        "none",
        "in_sample_center_shift + nnls; ridge by inner CV",
        "mu shift in linspace(-2, 2, 9); ridge in (0.03, 0.1, 0.3, 1.0, 1.5, 3.0)",
        _static(models.BowlModel()),
    ),
    ModelEntry(
        "grp_pair_power",
        "parent",
        ("grp",),
        "grp_pair_power",
        "GRP, regularized (pairs, no semantic families)",
        _mechanisms(
            benefit="power", harm="softplus_log_quadratic_group_sum", sharing="shared_shape", families="quality_pairs"
        ),
        "eta=1, lambda=0 (retained exposure equals total exposure); semantic family totals, family "
        "curvature a_f, and family thresholds tau_f are banned and collapse to one shared exponent "
        "and one shared group penalty",
        "CC high/low pairs with quality discount beta",
        "inner_cv_grid + nnls",
        "exponent x8, discount x3, threshold x5, ridge x10",
        _grp_pair("grp_pair_power"),
        note="The Observatory froze the transferred production shape and retuned only the ridge; the "
        "reduction contract requires every remaining shape to be refit, so exponent, discount, and "
        "threshold are searched.",
    ),
    ModelEntry(
        "weibull_shared_literal_replay",
        "parent",
        ("compact_retained_state",),
        "weibull_shared_literal_replay",
        "Compact retained state (one phase)",
        _mechanisms(benefit="weibull", harm="literal_replay_shared", sharing="shared_shape", head="nonnegative_scaled"),
        "retention CONSTANT and late multiplier 1 make the retained state equal total exposure "
        "(the Observatory's `one_phase_weibull_shared_replay` config)",
        "none",
        "in_sample_lbfgs_shape (9 starts, top 2, maxiter 24) + scaled nnls; ridge by inner CV",
        "log rate in [log 0.05, log 20], power in [0.2, 1]; ridge in (0.1, 1.0)",
        _compact("weibull_shared_literal_replay"),
    ),
    ModelEntry(
        "bucket_family_power_grp",
        "parent",
        ("bucket_family_grp", "bucket_family_power_separate_heads"),
        "bucket_family_power_grp",
        "Bucket-resolved family GRP / power heads",
        _mechanisms(
            benefit="power", harm="softplus_log_quadratic_family", sharing="shared_shape", families="domain_quality"
        ),
        "late multiplier 1 and forgetting 0 (retained exposure equals total exposure); the early "
        "and late heads coincide, leaving one head; the two source models then share one design "
        "and differ only in shape/ridge grids, which are united here",
        "CC high/low pairs as families; singletons unpooled",
        "inner_cv_grid + nnls",
        f"shapes x{len(BUCKET_FAMILY_SHAPES)} (union of both source grids), ridge x{len(BUCKET_FAMILY_RIDGE_GRID)}",
        _family_grid_model(
            "bucket_family_power_grp", BUCKET_FAMILY_OPTIONS, BUCKET_FAMILY_SHAPES, BUCKET_FAMILY_RIDGE_GRID, NNLS, 2
        ),
    ),
    ModelEntry(
        "hierarchical_family_replay",
        "parent",
        ("hierarchical_phase_bucket_replay",),
        "hierarchical_family_replay",
        "Hierarchical phase replay (single phase)",
        _mechanisms(
            benefit="power",
            harm="softplus_log_quadratic_family+member_replay",
            sharing="hierarchical_shrinkage",
            families="domain_quality",
        ),
        "the phase-shift TV column is identically zero and is removed; retained exposure equals total exposure",
        "CC high/low pairs as pooled groups",
        "two_stage_inner_cv (bucket-resolved screen, top 3 shapes, structure sweep) + nnls with ridge multipliers",
        f"screen shapes x{len(HPR_SCREEN_SHAPES)}, ridge x{len(HIERARCHICAL_RIDGE_GRID)}, residual "
        "shrink x{len(HIERARCHICAL_RESIDUAL_GRID)}",
        _hierarchical("hierarchical_family_replay", HPR_OPTIONS),
    ),
    ModelEntry(
        "crs_plus_family_overload",
        "parent",
        ("crs_plus",),
        "crs_plus_family_overload",
        "Compact retained state + family",
        _mechanisms(
            benefit="weibull_retained_state",
            harm="literal_replay_shared+family_overload",
            sharing="shared_shape",
            families="domain_quality",
            head="nonnegative_scaled",
        ),
        "none removed: the Observatory single-phase entry keeps the revisit-gated retention, whose "
        "tied image (a0 e^{-f(1-w)} + L a1) E still moves predictions",
        "CC high/low pairs as families; every family, singletons included, carries a "
        "family-benefit column at the family rate",
        "inner_cv_grid + scaled nnls",
        "shapes x648 (x162 on phase-less panels), ridge x4",
        _crs_plus("crs_plus_family_overload"),
    ),
    ModelEntry(
        "weibull_literal_replay_logdeficit",
        "parent",
        ("crs_bounded",),
        "weibull_literal_replay_logdeficit",
        "Compact retained state, bounded link",
        _mechanisms(
            benefit="weibull_retained_state",
            harm="literal_replay_shared",
            link="log_deficit",
            sharing="shared_shape",
            head="nonnegative_scaled",
        ),
        "none removed: the retention gate's tied image is kept, as in the Observatory single-phase entry",
        "none",
        "inner_cv_grid (scored in BPB after link inversion) + scaled nnls on log(y - 0.95 min y)",
        "shapes x90 (x18 on phase-less panels), ridge x4",
        _crs_bounded("weibull_literal_replay_logdeficit"),
    ),
    ModelEntry(
        "hierarchical_family_replay_band",
        "parent",
        ("hpr_band",),
        "hierarchical_family_replay_band",
        "Hierarchical phase replay (band ensemble)",
        _mechanisms(
            benefit="power",
            harm="softplus_log_quadratic_family+member_replay",
            sharing="hierarchical_shrinkage",
            families="domain_quality",
            estimator="band_stack",
        ),
        "as hierarchical_family_replay; the band is built from the same candidates",
        "CC high/low pairs as pooled groups",
        "two_stage_inner_cv + simplex stacking over the 15% relative band (max 24 members)",
        "as hierarchical_family_replay",
        _band("hierarchical_family_replay_band"),
    ),
    ModelEntry(
        "family_onset_power_grp",
        "parent",
        ("bucket_family_power_separate_heads_family_onset",),
        "family_onset_power_grp",
        "Power heads, family onset",
        _mechanisms(
            benefit="power",
            harm="softplus_log_quadratic_family_onset",
            sharing="shared_shape",
            families="domain_quality",
        ),
        "late multiplier 1, forgetting 0; one head",
        "CC high/low pairs as families (one onset per family, singletons included)",
        "inner_cv_grid for shape/ridge, then L-BFGS-B family onsets with inner-CV shrinkage",
        f"shapes x{len(POWER_HEADS_SHAPES)}, ridge x{len(POWER_HEADS_RIDGE_GRID)}, tau shrink "
        "x{len(models.TAU_SHRINK_GRID)}",
        _family_onset("family_onset_power_grp"),
    ),
    ModelEntry(
        "weibull_family_grp_shared_onset",
        "parent",
        ("bucket_family_weibull_shared_onset",),
        "weibull_family_grp_shared_onset",
        "Weibull GRP, shared onset",
        _mechanisms(
            benefit="weibull", harm="softplus_log_quadratic_family", sharing="shared_shape", families="domain_quality"
        ),
        "late multiplier 1, forgetting 0",
        "CC high/low pairs as families",
        "inner_cv_grid + nnls",
        f"shapes x{len(WEIBULL_ONSET_SHAPES)}, ridge x{len(RETAINED_GRP_RIDGE_GRID)}",
        _family_grid_model(
            "weibull_family_grp_shared_onset",
            WEIBULL_ONSET_OPTIONS,
            WEIBULL_ONSET_SHAPES,
            RETAINED_GRP_RIDGE_GRID,
            NNLS,
            3,
        ),
    ),
    ModelEntry(
        "weibull_family_grp_literal_replay",
        "parent",
        ("bucket_family_weibull_family_replay",),
        "weibull_family_grp_literal_replay",
        "Weibull GRP, family replay",
        _mechanisms(benefit="weibull", harm="literal_replay_family", sharing="shared_shape", families="domain_quality"),
        "late multiplier 1, forgetting 0",
        "CC high/low pairs as families",
        "inner_cv_grid + nnls",
        f"shapes x{len(WEIBULL_REPLAY_SHAPES)}, ridge x{len(RETAINED_GRP_RIDGE_GRID)}",
        _family_grid_model(
            "weibull_family_grp_literal_replay",
            WEIBULL_REPLAY_OPTIONS,
            WEIBULL_REPLAY_SHAPES,
            RETAINED_GRP_RIDGE_GRID,
            NNLS,
            2,
        ),
    ),
    ModelEntry(
        "retained_power_law_phase_blind",
        "parent",
        ("retained_power_law",),
        "retained_power_law_phase_blind",
        "Retained power law (phase blind)",
        _mechanisms(
            coordinate="weight_benefit/exposure_damage",
            benefit="inverse_power_share",
            harm="power_excess_epochs",
            sharing="hierarchical_shrinkage",
            families="domain_quality",
            head="nonnegative_scaled",
            estimator="huber_irls",
        ),
        "retention 0, late multiplier 1, no ordering channel, concentration gap identically zero "
        "(the repaired estimator's own phase-blind image)",
        "CC high/low pairs as pooled families",
        "least_squares_screen (top 12) + huber_nnls rescoring; inner CV",
        "shapes x126, ridge x3",
        _retained_power_law("retained_power_law_phase_blind"),
    ),
)

REFERENCES: tuple[ModelEntry, ...] = (
    ModelEntry(
        "fold_mean",
        "reference",
        (),
        "constant",
        "Fold mean",
        _mechanisms(benefit="none", sharing="none", head="none", estimator="mean"),
        "n/a",
        "none",
        "mean",
        "none",
        _static(models.FoldMeanModel()),
    ),
    ModelEntry(
        "linear_epoch_log_link",
        "reference",
        (),
        "linear_epoch_log_link",
        "Taskwise ridge log-link on log1p(epochs)",
        _mechanisms(benefit="log1p", link="positive_log", sharing="per_bucket", head="signed", estimator="ridge"),
        "n/a (OLMix-swarm reference from `benchmark_olmix_swarm_single_phase_dsp_20260901`)",
        "none",
        "inner_cv_grid + standardized ridge on log(y - floor)",
        "ridge x8, floor margin x2",
        _log_link("linear_epoch_log_link"),
    ),
)


def _ablation(
    model_id: str,
    parent: str,
    mechanism: str,
    build: Builder,
    *,
    transform: str | None = None,
    note: str = "",
    role: str = "ablation",
) -> ModelEntry:
    """An ablation inherits its parent's record and overrides every mechanism named in ``mechanism``."""
    base = PARENT_BY_ID.get(parent) or REFERENCE_BY_ID.get(parent) or _SUCCESSOR_LOOKUP[parent]
    mechanisms = dict(base.mechanisms)
    for clause in mechanism.split(","):
        key, _, value = clause.partition("=")
        if value:
            mechanisms[key.strip()] = value.strip()
    if transform == "no_families":
        mechanisms["families"] = "none"
    elif transform in ("permuted_inventory", "weight_coordinate"):
        mechanisms["coordinate"] = transform
    elif transform:
        mechanisms["families"] = transform
    return ModelEntry(
        model_id,
        role,
        (),
        base.equivalence_class,
        f"{base.display} - {mechanism}",
        mechanisms,
        base.removed_phase_terms,
        base.allowed_metadata,
        base.solver_id,
        "derived from the built model; see model_registry.csv",
        build,
        parent=parent,
        ablated_mechanism=mechanism,
        feature_transform=transform,
        note=note,
    )


PARENT_BY_ID = {entry.model_id: entry for entry in PARENTS}
REFERENCE_BY_ID = {entry.model_id: entry for entry in REFERENCES}
_SUCCESSOR_LOOKUP: dict[str, ModelEntry] = {}


def _options(base: models.FamilyOptions, **overrides: Any) -> models.FamilyOptions:
    return dataclasses.replace(base, **overrides)


ABLATIONS: tuple[ModelEntry, ...] = (
    # Canonical DSP neighbourhood.
    _ablation(
        "dsp_total_exposure@no_harm",
        "dsp_total_exposure",
        "harm=none",
        _dsp("dsp_total_exposure@no_harm", penalty="none"),
    ),
    _ablation(
        "dsp_total_exposure@bounded_harm",
        "dsp_total_exposure",
        "harm=bounded",
        _dsp("dsp_total_exposure@bounded_harm", penalty="bounded"),
    ),
    _ablation(
        "dsp_total_exposure@shared_shape",
        "dsp_total_exposure",
        "sharing=shared_shape",
        _dsp("dsp_total_exposure@shared_shape", per_bucket=False),
    ),
    _ablation(
        "dsp_total_exposure@shared_bounded",
        "dsp_total_exposure",
        "sharing=shared_shape,harm=bounded",
        _dsp("dsp_total_exposure@shared_bounded", per_bucket=False, penalty="bounded"),
    ),
    _ablation(
        "dsp_total_exposure@pair_tie",
        "dsp_total_exposure",
        "sharing=+quality_pair_tie",
        _dsp("dsp_total_exposure@pair_tie", tie_pairs=True),
    ),
    _ablation(
        "dsp_total_exposure@permuted_inventory",
        "dsp_total_exposure",
        "coordinate=permuted_inventory",
        _dsp("dsp_total_exposure@permuted_inventory"),
        transform="permuted_inventory",
        role="control",
    ),
    _ablation(
        "dsp_total_exposure@weight_coordinate",
        "dsp_total_exposure",
        "coordinate=weight",
        _dsp("dsp_total_exposure@weight_coordinate"),
        transform="weight_coordinate",
    ),
    _ablation(
        "dsp_total_exposure@outcome_permutation",
        "dsp_total_exposure",
        "control=outcome_permutation",
        _dsp("dsp_total_exposure@outcome_permutation"),
        transform="outcome_permutation",
        role="control",
        note="Negative control: training outcomes permuted within the training rows.",
    ),
    # Separate heads.
    _ablation(
        "asymmetric_log_bowl@symmetric",
        "asymmetric_log_bowl",
        "harm=symmetric_bowl",
        _static(models.BowlModel("asymmetric_log_bowl@symmetric", symmetric=True)),
    ),
    _ablation(
        "asymmetric_log_bowl@permuted_inventory",
        "asymmetric_log_bowl",
        "coordinate=permuted_inventory",
        _static(models.BowlModel("asymmetric_log_bowl@permuted_inventory")),
        transform="permuted_inventory",
        role="control",
    ),
    # GRP pairs.
    _ablation(
        "grp_pair_power@no_pair_discount",
        "grp_pair_power",
        "families=pairs_without_discount",
        _grp_pair("grp_pair_power@no_pair_discount", discount=False),
    ),
    _ablation(
        "grp_pair_power@no_pairs",
        "grp_pair_power",
        "families=none",
        _grp_pair("grp_pair_power@no_pairs"),
        transform="no_families",
    ),
    _ablation("grp_pair_power@no_harm", "grp_pair_power", "harm=none", _grp_pair("grp_pair_power@no_harm", harm="none")),
    _ablation(
        "grp_pair_power@shuffled_families",
        "grp_pair_power",
        "families=shuffled",
        _grp_pair("grp_pair_power@shuffled_families"),
        transform="shuffled_families",
        role="control",
    ),
    # Compact retained state.
    _ablation(
        "weibull_shared_literal_replay@no_replay",
        "weibull_shared_literal_replay",
        "harm=none",
        _compact("weibull_shared_literal_replay@no_replay", harm="none"),
    ),
    _ablation(
        "weibull_shared_literal_replay@saturation_benefit",
        "weibull_shared_literal_replay",
        "benefit=exp_saturation",
        _compact("weibull_shared_literal_replay@saturation_benefit", benefit="saturation"),
    ),
    _ablation(
        "weibull_shared_literal_replay@power_benefit",
        "weibull_shared_literal_replay",
        "benefit=power",
        _compact("weibull_shared_literal_replay@power_benefit", benefit="power"),
    ),
    _ablation(
        "weibull_shared_literal_replay@unscaled_head",
        "weibull_shared_literal_replay",
        "head=nonnegative_unscaled",
        _compact("weibull_shared_literal_replay@unscaled_head", scale_columns=False),
    ),
    _ablation(
        "weibull_shared_literal_replay@permuted_inventory",
        "weibull_shared_literal_replay",
        "coordinate=permuted_inventory",
        _compact("weibull_shared_literal_replay@permuted_inventory"),
        transform="permuted_inventory",
        role="control",
    ),
    _ablation(
        "weibull_shared_literal_replay@weight_coordinate",
        "weibull_shared_literal_replay",
        "coordinate=weight",
        _compact("weibull_shared_literal_replay@weight_coordinate"),
        transform="weight_coordinate",
    ),
    # Bucket family power GRP.
    _ablation(
        "bucket_family_power_grp@no_family_signal",
        "bucket_family_power_grp",
        "families=no_family_signal",
        _family_grid_model(
            "bucket_family_power_grp@no_family_signal",
            _options(BUCKET_FAMILY_OPTIONS, family_signal="none"),
            BUCKET_FAMILY_SHAPES,
            BUCKET_FAMILY_RIDGE_GRID,
            NNLS,
            2,
        ),
    ),
    _ablation(
        "bucket_family_power_grp@no_harm",
        "bucket_family_power_grp",
        "harm=none",
        _family_grid_model(
            "bucket_family_power_grp@no_harm",
            _options(BUCKET_FAMILY_OPTIONS, harm="none"),
            BUCKET_FAMILY_SHAPES,
            BUCKET_FAMILY_RIDGE_GRID,
            NNLS,
            1,
        ),
    ),
    _ablation(
        "bucket_family_power_grp@literal_family_harm",
        "bucket_family_power_grp",
        "harm=literal_replay_family",
        _family_grid_model(
            "bucket_family_power_grp@literal_family_harm",
            _options(BUCKET_FAMILY_OPTIONS, harm="literal_family"),
            BUCKET_FAMILY_SHAPES,
            BUCKET_FAMILY_RIDGE_GRID,
            NNLS,
            1,
        ),
    ),
    _ablation(
        "bucket_family_power_grp@bucket_harm",
        "bucket_family_power_grp",
        "harm=softplus_bucket",
        _family_grid_model(
            "bucket_family_power_grp@bucket_harm",
            _options(BUCKET_FAMILY_OPTIONS, harm="softplus_bucket"),
            BUCKET_FAMILY_SHAPES,
            BUCKET_FAMILY_RIDGE_GRID,
            NNLS,
            2,
        ),
    ),
    _ablation(
        "bucket_family_power_grp@no_families",
        "bucket_family_power_grp",
        "families=none",
        _family_grid_model(
            "bucket_family_power_grp@no_families",
            BUCKET_FAMILY_OPTIONS,
            BUCKET_FAMILY_SHAPES,
            BUCKET_FAMILY_RIDGE_GRID,
            NNLS,
            2,
        ),
        transform="no_families",
    ),
    _ablation(
        "bucket_family_power_grp@shuffled_families",
        "bucket_family_power_grp",
        "families=shuffled",
        _family_grid_model(
            "bucket_family_power_grp@shuffled_families",
            BUCKET_FAMILY_OPTIONS,
            BUCKET_FAMILY_SHAPES,
            BUCKET_FAMILY_RIDGE_GRID,
            NNLS,
            2,
        ),
        transform="shuffled_families",
        role="control",
    ),
    _ablation(
        "bucket_family_power_grp@signed_head",
        "bucket_family_power_grp",
        "head=signed",
        _family_grid_model(
            "bucket_family_power_grp@signed_head",
            BUCKET_FAMILY_OPTIONS,
            BUCKET_FAMILY_SHAPES,
            SIGNED_RIDGE_GRID,
            SIGNED,
            2,
        ),
    ),
    _ablation(
        "bucket_family_power_grp@permuted_inventory",
        "bucket_family_power_grp",
        "coordinate=permuted_inventory",
        _family_grid_model(
            "bucket_family_power_grp@permuted_inventory",
            BUCKET_FAMILY_OPTIONS,
            BUCKET_FAMILY_SHAPES,
            BUCKET_FAMILY_RIDGE_GRID,
            NNLS,
            2,
        ),
        transform="permuted_inventory",
        role="control",
    ),
    _ablation(
        "bucket_family_power_grp@weibull_benefit",
        "bucket_family_power_grp",
        "benefit=weibull,families=mean_pooling,grid=34x3dof(confounded)",
        _family_grid_model(
            "bucket_family_power_grp@weibull_benefit",
            _options(BUCKET_FAMILY_OPTIONS, benefit="weibull", family_signal="mean"),
            WEIBULL_ONSET_SHAPES,
            BUCKET_FAMILY_RIDGE_GRID,
            NNLS,
            3,
        ),
    ),
    # Hierarchical replay.
    _ablation(
        "hierarchical_family_replay@no_hierarchy",
        "hierarchical_family_replay",
        "sharing=no_pooled_base",
        _hierarchical("hierarchical_family_replay@no_hierarchy", _options(HPR_OPTIONS, hierarchical=False)),
    ),
    _ablation(
        "hierarchical_family_replay@no_member_replay",
        "hierarchical_family_replay",
        "harm=no_member_replay",
        _hierarchical("hierarchical_family_replay@no_member_replay", _options(HPR_OPTIONS, member_replay=False)),
    ),
    _ablation(
        "hierarchical_family_replay@no_family_signal",
        "hierarchical_family_replay",
        "families=no_family_signal",
        _hierarchical("hierarchical_family_replay@no_family_signal", _options(HPR_OPTIONS, family_signal="none")),
    ),
    _ablation(
        "hierarchical_family_replay@shuffled_families",
        "hierarchical_family_replay",
        "families=shuffled",
        _hierarchical("hierarchical_family_replay@shuffled_families", HPR_OPTIONS),
        transform="shuffled_families",
        role="control",
    ),
    # crs_plus.
    _ablation(
        "crs_plus_family_overload@no_retention_gate",
        "crs_plus_family_overload",
        "retention_gate=off",
        _crs_plus("crs_plus_family_overload@no_retention_gate", retention_gate=False),
    ),
    _ablation(
        "crs_plus_family_overload@no_family_benefit",
        "crs_plus_family_overload",
        "families=no_family_benefit",
        _crs_plus("crs_plus_family_overload@no_family_benefit", family_benefit=False),
    ),
    _ablation(
        "crs_plus_family_overload@no_overload",
        "crs_plus_family_overload",
        "harm=no_family_overload",
        _crs_plus("crs_plus_family_overload@no_overload", overload=False),
    ),
    _ablation(
        "crs_plus_family_overload@no_literal_replay",
        "crs_plus_family_overload",
        "harm=no_literal_replay",
        _crs_plus("crs_plus_family_overload@no_literal_replay", literal=False),
    ),
    _ablation(
        "crs_plus_family_overload@shuffled_families",
        "crs_plus_family_overload",
        "families=shuffled",
        _crs_plus("crs_plus_family_overload@shuffled_families"),
        transform="shuffled_families",
        role="control",
    ),
    # crs_bounded.
    _ablation(
        "weibull_literal_replay_logdeficit@identity_link",
        "weibull_literal_replay_logdeficit",
        "link=identity",
        _crs_bounded("weibull_literal_replay_logdeficit@identity_link", link=models.LinkKind.IDENTITY),
    ),
    _ablation(
        "weibull_literal_replay_logdeficit@no_retention_gate",
        "weibull_literal_replay_logdeficit",
        "retention_gate=off",
        _crs_bounded("weibull_literal_replay_logdeficit@no_retention_gate", retention_gate=False),
    ),
    _ablation(
        "weibull_literal_replay_logdeficit@no_replay",
        "weibull_literal_replay_logdeficit",
        "harm=none",
        _crs_bounded("weibull_literal_replay_logdeficit@no_replay", replay=False),
    ),
    # Family onset.
    _ablation(
        "family_onset_power_grp@shared_onset",
        "family_onset_power_grp",
        "harm=shared_onset",
        _family_grid_model(
            "family_onset_power_grp@shared_onset",
            BUCKET_FAMILY_OPTIONS,
            POWER_HEADS_SHAPES,
            POWER_HEADS_RIDGE_GRID,
            NNLS,
            2,
        ),
    ),
    # Weibull family GRP, shared onset.
    _ablation(
        "weibull_family_grp_shared_onset@no_family_signal",
        "weibull_family_grp_shared_onset",
        "families=no_family_signal",
        _family_grid_model(
            "weibull_family_grp_shared_onset@no_family_signal",
            _options(WEIBULL_ONSET_OPTIONS, family_signal="none"),
            WEIBULL_ONSET_SHAPES,
            RETAINED_GRP_RIDGE_GRID,
            NNLS,
            3,
        ),
    ),
    _ablation(
        "weibull_family_grp_shared_onset@no_harm",
        "weibull_family_grp_shared_onset",
        "harm=none",
        _family_grid_model(
            "weibull_family_grp_shared_onset@no_harm",
            _options(WEIBULL_ONSET_OPTIONS, harm="none"),
            WEIBULL_ONSET_SHAPES,
            RETAINED_GRP_RIDGE_GRID,
            NNLS,
            2,
        ),
    ),
    _ablation(
        "weibull_family_grp_shared_onset@literal_family_harm",
        "weibull_family_grp_shared_onset",
        "harm=literal_replay_family",
        _family_grid_model(
            "weibull_family_grp_shared_onset@literal_family_harm",
            _options(WEIBULL_ONSET_OPTIONS, harm="literal_family"),
            WEIBULL_ONSET_SHAPES,
            RETAINED_GRP_RIDGE_GRID,
            NNLS,
            2,
        ),
    ),
    _ablation(
        "weibull_family_grp_shared_onset@shuffled_families",
        "weibull_family_grp_shared_onset",
        "families=shuffled",
        _family_grid_model(
            "weibull_family_grp_shared_onset@shuffled_families",
            WEIBULL_ONSET_OPTIONS,
            WEIBULL_ONSET_SHAPES,
            RETAINED_GRP_RIDGE_GRID,
            NNLS,
            3,
        ),
        transform="shuffled_families",
        role="control",
    ),
    _ablation(
        "weibull_family_grp_shared_onset@power_benefit",
        "weibull_family_grp_shared_onset",
        "benefit=power,families=sum_pooling,grid=42x2dof(confounded)",
        _family_grid_model(
            "weibull_family_grp_shared_onset@power_benefit",
            _options(WEIBULL_ONSET_OPTIONS, benefit="power", family_signal="sum"),
            BUCKET_FAMILY_SHAPES,
            RETAINED_GRP_RIDGE_GRID,
            NNLS,
            2,
        ),
    ),
    # Weibull family GRP, literal replay.
    _ablation(
        "weibull_family_grp_literal_replay@no_family_signal",
        "weibull_family_grp_literal_replay",
        "families=no_family_signal",
        _family_grid_model(
            "weibull_family_grp_literal_replay@no_family_signal",
            _options(WEIBULL_REPLAY_OPTIONS, family_signal="none"),
            WEIBULL_REPLAY_SHAPES,
            RETAINED_GRP_RIDGE_GRID,
            NNLS,
            2,
        ),
    ),
    _ablation(
        "weibull_family_grp_literal_replay@shared_literal_harm",
        "weibull_family_grp_literal_replay",
        "harm=literal_replay_shared",
        _family_grid_model(
            "weibull_family_grp_literal_replay@shared_literal_harm",
            _options(WEIBULL_REPLAY_OPTIONS, harm="literal_shared"),
            WEIBULL_REPLAY_SHAPES,
            RETAINED_GRP_RIDGE_GRID,
            NNLS,
            2,
        ),
    ),
    _ablation(
        "weibull_family_grp_literal_replay@no_harm",
        "weibull_family_grp_literal_replay",
        "harm=none",
        _family_grid_model(
            "weibull_family_grp_literal_replay@no_harm",
            _options(WEIBULL_REPLAY_OPTIONS, harm="none"),
            WEIBULL_REPLAY_SHAPES,
            RETAINED_GRP_RIDGE_GRID,
            NNLS,
            2,
        ),
    ),
    # Retained power law.
    _ablation(
        "retained_power_law_phase_blind@exposure_coordinate",
        "retained_power_law_phase_blind",
        "coordinate=exposure_benefit",
        _retained_power_law("retained_power_law_phase_blind@exposure_coordinate", coordinate="exposure"),
    ),
    _ablation(
        "retained_power_law_phase_blind@no_damage",
        "retained_power_law_phase_blind",
        "harm=none",
        _retained_power_law("retained_power_law_phase_blind@no_damage", damage=False),
    ),
    _ablation(
        "retained_power_law_phase_blind@squared_loss",
        "retained_power_law_phase_blind",
        "estimator=squared",
        _retained_power_law("retained_power_law_phase_blind@squared_loss", robust=False),
    ),
    _ablation(
        "retained_power_law_phase_blind@no_hierarchy",
        "retained_power_law_phase_blind",
        "sharing=per_bucket",
        _retained_power_law("retained_power_law_phase_blind@no_hierarchy", hierarchical=False),
    ),
    _ablation(
        "retained_power_law_phase_blind@shuffled_families",
        "retained_power_law_phase_blind",
        "families=shuffled",
        _retained_power_law("retained_power_law_phase_blind@shuffled_families"),
        transform="shuffled_families",
        role="control",
    ),
    # Baselines and references.
    _ablation(
        "olmix_loglinear_taskwise@analytic_gradient",
        "olmix_loglinear_taskwise",
        "estimator=analytic_gradient_to_convergence",
        _static(models.OlmixTaskwiseModel("olmix_loglinear_taskwise@analytic_gradient", analytic_gradient=True)),
        note=(
            "Same objective, start bank, bounds, and selection rule as the parent, driven by an analytic Huber "
            "gradient so L-BFGS-B converges; measures how much of the parent's out-of-fold behaviour is the "
            "numerical solver's early stopping."
        ),
    ),
    _ablation(
        "linear_weight@exposure_coordinate",
        "linear_weight",
        "coordinate=exposure",
        _static(models.LinearWeightModel("linear_weight@exposure_coordinate", coordinate="exposure")),
        note="Expected to reproduce the parent exactly: an affine map of w is an affine map of E.",
    ),
    _ablation(
        "linear_epoch_log_link@permuted_inventory",
        "linear_epoch_log_link",
        "coordinate=permuted_inventory",
        _log_link("linear_epoch_log_link@permuted_inventory"),
        transform="permuted_inventory",
        role="control",
    ),
    # Capacity-matched controls for the softplus harm: same columns, ridge, and threshold search,
    # with the harm block reading exposures through a fixed bucket permutation.
    _ablation(
        "bucket_family_power_grp@scrambled_harm",
        "bucket_family_power_grp",
        "harm=scrambled_inventory_control",
        _family_grid_model(
            "bucket_family_power_grp@scrambled_harm",
            _options(BUCKET_FAMILY_OPTIONS, scrambled_harm=True),
            BUCKET_FAMILY_SHAPES,
            BUCKET_FAMILY_RIDGE_GRID,
            NNLS,
            2,
        ),
        role="control",
    ),
    _ablation(
        "weibull_family_grp_shared_onset@scrambled_harm",
        "weibull_family_grp_shared_onset",
        "harm=scrambled_inventory_control",
        _family_grid_model(
            "weibull_family_grp_shared_onset@scrambled_harm",
            _options(WEIBULL_ONSET_OPTIONS, scrambled_harm=True),
            WEIBULL_ONSET_SHAPES,
            RETAINED_GRP_RIDGE_GRID,
            NNLS,
            3,
        ),
        role="control",
    ),
    _ablation(
        "grp_pair_power@scrambled_harm",
        "grp_pair_power",
        "harm=scrambled_inventory_control",
        _grp_pair("grp_pair_power@scrambled_harm", scrambled=True),
        role="control",
    ),
    # Benefit-response ablations with family pooling and grid budget matched to the parent.
    _ablation(
        "bucket_family_power_grp@weibull_benefit_matched",
        "bucket_family_power_grp",
        "benefit=weibull",
        _family_grid_model(
            "bucket_family_power_grp@weibull_benefit_matched",
            _options(BUCKET_FAMILY_OPTIONS, benefit="weibull"),
            WEIBULL_MATCHED_SHAPES,
            BUCKET_FAMILY_RIDGE_GRID,
            NNLS,
            3,
        ),
    ),
    _ablation(
        "weibull_family_grp_shared_onset@power_benefit_matched",
        "weibull_family_grp_shared_onset",
        "benefit=power",
        _family_grid_model(
            "weibull_family_grp_shared_onset@power_benefit_matched",
            _options(WEIBULL_ONSET_OPTIONS, benefit="power"),
            POWER_MATCHED_SHAPES,
            RETAINED_GRP_RIDGE_GRID,
            NNLS,
            2,
        ),
    ),
    _ablation(
        "linear_epoch_log_link@identity_link",
        "linear_epoch_log_link",
        "link=identity",
        _log_link("linear_epoch_log_link@identity_link", link=models.LinkKind.IDENTITY),
    ),
)

# Successor synthesis (appended to single_phase_observatory_registry_20260902.py after the review gate).
# Row-scrambled harm controls: same harm columns, mixture rows permuted, so the block carries capacity only.
ROW_SCRAMBLED_CONTROLS: tuple[ModelEntry, ...] = (
    _ablation(
        "bucket_family_power_grp@row_scrambled_harm",
        "bucket_family_power_grp",
        "harm=row_scrambled_control",
        _family_grid_model(
            "bucket_family_power_grp@row_scrambled_harm",
            _options(BUCKET_FAMILY_OPTIONS, row_scrambled_harm=True),
            BUCKET_FAMILY_SHAPES,
            BUCKET_FAMILY_RIDGE_GRID,
            NNLS,
            2,
        ),
        role="control",
    ),
    _ablation(
        "weibull_family_grp_shared_onset@row_scrambled_harm",
        "weibull_family_grp_shared_onset",
        "harm=row_scrambled_control",
        _family_grid_model(
            "weibull_family_grp_shared_onset@row_scrambled_harm",
            _options(WEIBULL_ONSET_OPTIONS, row_scrambled_harm=True),
            WEIBULL_ONSET_SHAPES,
            RETAINED_GRP_RIDGE_GRID,
            NNLS,
            3,
        ),
        role="control",
    ),
    _ablation(
        "grp_pair_power@row_scrambled_harm",
        "grp_pair_power",
        "harm=row_scrambled_control",
        _grp_pair("grp_pair_power@row_scrambled_harm", row_scrambled=True),
        role="control",
    ),
)

SUCCESSOR_RATES = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0)
SUCCESSOR_POWERS = (0.3, 0.5, 0.7, 1.0)
SUCCESSOR_THRESHOLDS = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
SUCCESSOR_RIDGE_GRID = (0.0, 1e-3, 1e-2, 0.1, 1.0)
SUCCESSOR_SHAPES = tuple(
    {"rate": rate, "power": power, "threshold": threshold}
    for rate in SUCCESSOR_RATES
    for power in SUCCESSOR_POWERS
    for threshold in SUCCESSOR_THRESHOLDS
)
SUCCESSOR_EXP_SHAPES = tuple(
    {"rate": rate, "threshold": threshold} for rate in SUCCESSOR_RATES for threshold in SUCCESSOR_THRESHOLDS
)
# Budget-matched exponential benefit: 28 log-spaced rates x 6 thresholds = 168 shapes, the successor's grid size.
SUCCESSOR_EXP_MATCHED_RATES = tuple(float(f"{rate:.4g}") for rate in np.geomspace(0.05, 4.0, 28))
# Wide successor grid for sub-epoch and hundred-epoch curves: knee from 0.005 to 20 epochs, shape down to 0.15.
SUCCESSOR_WIDE_RATES = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 10.0, 25.0, 60.0, 150.0, 400.0)
SUCCESSOR_WIDE_POWERS = (0.15, 0.3, 0.5, 0.7, 1.0)
SUCCESSOR_WIDE_SHAPES = tuple(
    {"rate": rate, "power": power, "threshold": threshold}
    for rate in SUCCESSOR_WIDE_RATES
    for power in SUCCESSOR_WIDE_POWERS
    for threshold in SUCCESSOR_THRESHOLDS
)
SUCCESSOR_EXP_MATCHED_SHAPES = tuple(
    {"rate": rate, "threshold": threshold} for rate in SUCCESSOR_EXP_MATCHED_RATES for threshold in SUCCESSOR_THRESHOLDS
)
SUCCESSOR_OPTIONS = models.FamilyOptions(family_signal="none", harm="softplus_bucket", benefit="weibull")
SUCCESSOR_HEAD = models.HeadSpec(kind=models.HeadKind.NNLS, scale_columns=True)


def _successor(
    model_id: str,
    options: models.FamilyOptions = SUCCESSOR_OPTIONS,
    shapes: tuple[models.Shape, ...] = SUCCESSOR_SHAPES,
    head: models.HeadSpec = SUCCESSOR_HEAD,
    ridge_grid: tuple[float, ...] = SUCCESSOR_RIDGE_GRID,
) -> Builder:
    dof = 3 if options.benefit == "weibull" else 2
    if options.harm == "none":
        dof -= 1
    return _family_grid_model(model_id, options, shapes, ridge_grid, head, dof)


# Significance prior from the 2026-06-23 domain-ablation p-value matrix (300M, one bucket deleted at a time,
# renormalized proportional baseline with repeats): bucket columns whose deletion has no significant effect on
# a metric get a larger ridge multiplier for that metric's head. Keyed by the benchmark's component names.
ABLATION_PVALUE_DIR = (
    Path(__file__).resolve().parent / "reference_outputs" / "domain_ablation_pvalue_matrix_with_training_eval_20260623"
)
ABLATION_PVALUE_FILE = "domain_ablation_cell_pvalues.csv"
PRIOR_ALPHA = 0.05
PRIOR_SHRINK = 10.0
PRIOR_SCRAMBLE_SEED = 20_260_905
TABLE9_COMPONENT_PREFIX = "olmo_base_eval/easy_bpb/"
LM_EVAL_SUFFIXES = ("_5shot", "_0shot", "_10shot", "")


def _component_aliases(metric: str) -> tuple[str, ...]:
    """Benchmark component names that a p-value-matrix metric stands for."""
    aliases = [metric]
    if metric.startswith("lm_eval/") and metric.endswith("/bpb"):
        task = metric[len("lm_eval/") : -len("/bpb")]
        for suffix in LM_EVAL_SUFFIXES:
            if suffix and task.endswith(suffix):
                task = task[: -len(suffix)]
                break
        aliases.append(f"{TABLE9_COMPONENT_PREFIX}{task}/bpb")
    return tuple(aliases)


@functools.cache
def ablation_prior_table(scrambled: bool = False) -> tuple[tuple[str, tuple[tuple[str, float], ...]], ...]:
    """Per-component bucket ridge multipliers: 1 where the deletion effect is significant, PRIOR_SHRINK otherwise."""
    cells = pd.read_csv(ABLATION_PVALUE_DIR / ABLATION_PVALUE_FILE)
    generator = np.random.default_rng(PRIOR_SCRAMBLE_SEED)
    table: dict[str, tuple[tuple[str, float], ...]] = {}
    shot_rank = {suffix: rank for rank, suffix in enumerate(LM_EVAL_SUFFIXES)}
    chosen_rank: dict[str, int] = {}
    for metric, block in cells.groupby("metric", sort=True):
        block = block.sort_values("target_domain")
        factors = np.where(block["p_two_sided"].to_numpy(float) < PRIOR_ALPHA, 1.0, PRIOR_SHRINK)
        if scrambled:
            factors = factors[generator.permutation(len(factors))]
        entries = tuple(zip(block["target_domain"].tolist(), factors.tolist(), strict=True))
        rank = next((shot_rank[s] for s in LM_EVAL_SUFFIXES if s and metric.endswith(f"{s}/bpb")), shot_rank[""])
        for alias in _component_aliases(str(metric)):
            if alias in table and chosen_rank.get(alias, 99) <= rank:
                continue
            table[alias] = entries
            chosen_rank[alias] = rank
    return tuple(sorted(table.items()))


def _successor_prior(model_id: str, *, scrambled: bool) -> Builder:
    def build(features: models.Features) -> models.GridModel:
        options = _options(SUCCESSOR_OPTIONS, component_ridge=ablation_prior_table(scrambled))
        return _successor(model_id, options, head=NNLS)(features)

    return build


def _refined(builder: Builder, *, bounded: bool = False) -> Builder:
    """Wrap a grid builder so the fitted grid argmin is refined continuously (GridModel.refine)."""

    def build(features: models.Features) -> models.GridModel:
        return dataclasses.replace(builder(features), refine=True, refine_bounded=bounded)

    return build


def _with_link_candidates(builder: Builder, links: tuple[models.LinkKind, ...]) -> Builder:
    """Wrap a grid builder so every candidate is also tried under each link; inner CV picks the link."""

    def build(features: models.Features) -> models.GridModel:
        return dataclasses.replace(builder(features), link_candidates=links)

    return build


SHARED_SHAPE_UNITS: dict[str, tuple[str, str]] = {
    "weibull_softplus_unscaled@shared_shape_target": ("weibull_softplus_unscaled", "target"),
    "weibull_softplus_unscaled@shared_shape_panel": ("weibull_softplus_unscaled", "panel"),
    "weibull_softplus_unscaled@shared_shape_scale": ("weibull_softplus_unscaled", "scale"),
}

SUCCESSORS: tuple[ModelEntry, ...] = (
    ModelEntry(
        "weibull_softplus_shared",
        "successor",
        (),
        "weibull_softplus_shared",
        "Successor: shared Weibull benefit + shared-threshold softplus harm",
        _mechanisms(
            benefit="weibull",
            harm="softplus_log_quadratic_bucket",
            sharing="shared_shape",
            families="none",
            head="nonnegative_scaled",
        ),
        "n/a (new model)",
        "none",
        "inner_cv_grid + scaled nnls",
        f"shapes x{len(SUCCESSOR_SHAPES)} (rate x{len(SUCCESSOR_RATES)}, power x{len(SUCCESSOR_POWERS)}, "
        f"threshold x{len(SUCCESSOR_THRESHOLDS)}), ridge x{len(SUCCESSOR_RIDGE_GRID)}",
        _successor("weibull_softplus_shared"),
        note=(
            "Synthesised from the ablations: materialized epochs with the true inventory, one shared Weibull "
            "saturating benefit per bucket, one shared-threshold softplus overexposure harm per bucket, no "
            "families, no literal replay, no retention gate, nonnegative column-scaled head, identity link. "
            "Three nonlinear parameters and 2B linear amplitudes."
        ),
    ),
    ModelEntry(
        "weibull_softplus_unscaled",
        "successor",
        (),
        "weibull_softplus_unscaled",
        "Successor v2: shared Weibull benefit + shared-threshold softplus harm, unscaled nonnegative head",
        _mechanisms(
            benefit="weibull",
            harm="softplus_log_quadratic_bucket",
            sharing="shared_shape",
            families="none",
            head="nonnegative_unscaled",
        ),
        "n/a (new model)",
        "none",
        "inner_cv_grid + nnls",
        f"shapes x{len(SUCCESSOR_SHAPES)} (rate x{len(SUCCESSOR_RATES)}, power x{len(SUCCESSOR_POWERS)}, "
        f"threshold x{len(SUCCESSOR_THRESHOLDS)}), ridge x{len(SUCCESSOR_RIDGE_GRID)}",
        _successor("weibull_softplus_unscaled", head=NNLS),
        note=(
            "Revision of weibull_softplus_shared after its matched unscaled-head ablation won 35 of 38 Screen units: "
            "identical design and grid, nonnegative head without column scaling. Column scaling with ridge amplifies "
            "harm columns that are nearly zero in training and huge on extrapolated test mixtures."
        ),
    ),
)
_SUCCESSOR_LOOKUP = {entry.model_id: entry for entry in SUCCESSORS}
SUCCESSOR_ABLATIONS: tuple[ModelEntry, ...] = (
    _ablation(
        "weibull_softplus_shared@no_harm",
        "weibull_softplus_shared",
        "harm=none",
        _successor(
            "weibull_softplus_shared@no_harm",
            _options(SUCCESSOR_OPTIONS, harm="none"),
            tuple(
                {"rate": s["rate"], "power": s["power"]}
                for s in SUCCESSOR_SHAPES
                if s["threshold"] == SUCCESSOR_THRESHOLDS[0]
            ),
        ),
    ),
    _ablation(
        "weibull_softplus_shared@exp_benefit",
        "weibull_softplus_shared",
        "benefit=exp_saturation",
        _successor(
            "weibull_softplus_shared@exp_benefit",
            _options(SUCCESSOR_OPTIONS, benefit="saturation"),
            SUCCESSOR_EXP_SHAPES,
        ),
    ),
    _ablation(
        "weibull_softplus_shared@scrambled_harm",
        "weibull_softplus_shared",
        "harm=scrambled_inventory_control",
        _successor("weibull_softplus_shared@scrambled_harm", _options(SUCCESSOR_OPTIONS, scrambled_harm=True)),
        role="control",
    ),
    _ablation(
        "weibull_softplus_shared@permuted_inventory",
        "weibull_softplus_shared",
        "coordinate=permuted_inventory",
        _successor("weibull_softplus_shared@permuted_inventory"),
        transform="permuted_inventory",
        role="control",
    ),
    _ablation(
        "weibull_softplus_shared@weight_coordinate",
        "weibull_softplus_shared",
        "coordinate=weight",
        _successor("weibull_softplus_shared@weight_coordinate"),
        transform="weight_coordinate",
    ),
    _ablation(
        "weibull_softplus_shared@signed_head",
        "weibull_softplus_shared",
        "head=signed",
        _successor(
            "weibull_softplus_shared@signed_head",
            head=models.HeadSpec(kind=models.HeadKind.RIDGE),
            ridge_grid=(1e-8, 1e-3, 1e-2, 0.1, 1.0),
        ),
    ),
    _ablation(
        "weibull_softplus_shared@unscaled_head",
        "weibull_softplus_shared",
        "head=nonnegative_unscaled",
        _successor("weibull_softplus_shared@unscaled_head", head=NNLS),
    ),
    _ablation(
        "weibull_softplus_shared@log_deficit_link",
        "weibull_softplus_shared",
        "link=log_deficit",
        _successor(
            "weibull_softplus_shared@log_deficit_link",
            head=models.HeadSpec(kind=models.HeadKind.NNLS, scale_columns=True, link=models.LinkKind.LOG_DEFICIT),
        ),
    ),
    _ablation(
        "weibull_softplus_shared@family_harm",
        "weibull_softplus_shared",
        "harm=softplus_log_quadratic_family,families=domain_quality",
        _successor("weibull_softplus_shared@family_harm", _options(SUCCESSOR_OPTIONS, harm="softplus_family")),
    ),
    _ablation(
        "weibull_softplus_shared@outcome_permutation",
        "weibull_softplus_shared",
        "control=outcome_permutation",
        _successor("weibull_softplus_shared@outcome_permutation"),
        transform="outcome_permutation",
        role="control",
    ),
    _ablation(
        "weibull_softplus_shared@row_scrambled_harm",
        "weibull_softplus_shared",
        "harm=row_scrambled_control",
        _successor("weibull_softplus_shared@row_scrambled_harm", _options(SUCCESSOR_OPTIONS, row_scrambled_harm=True)),
        role="control",
    ),
    _ablation(
        "weibull_softplus_unscaled@no_harm",
        "weibull_softplus_unscaled",
        "harm=none",
        _successor(
            "weibull_softplus_unscaled@no_harm",
            _options(SUCCESSOR_OPTIONS, harm="none"),
            tuple(
                {"rate": s["rate"], "power": s["power"]}
                for s in SUCCESSOR_SHAPES
                if s["threshold"] == SUCCESSOR_THRESHOLDS[0]
            ),
            head=NNLS,
        ),
    ),
    _ablation(
        "weibull_softplus_unscaled@exp_benefit",
        "weibull_softplus_unscaled",
        "benefit=exp_saturation",
        _successor(
            "weibull_softplus_unscaled@exp_benefit",
            _options(SUCCESSOR_OPTIONS, benefit="saturation"),
            SUCCESSOR_EXP_SHAPES,
            head=NNLS,
        ),
    ),
    _ablation(
        "weibull_softplus_unscaled@row_scrambled_harm",
        "weibull_softplus_unscaled",
        "harm=row_scrambled_control",
        _successor(
            "weibull_softplus_unscaled@row_scrambled_harm",
            _options(SUCCESSOR_OPTIONS, row_scrambled_harm=True),
            head=NNLS,
        ),
        role="control",
    ),
    _ablation(
        "weibull_softplus_unscaled@permuted_inventory",
        "weibull_softplus_unscaled",
        "coordinate=permuted_inventory",
        _successor("weibull_softplus_unscaled@permuted_inventory", head=NNLS),
        transform="permuted_inventory",
        role="control",
    ),
    _ablation(
        "weibull_softplus_unscaled@weight_coordinate",
        "weibull_softplus_unscaled",
        "coordinate=weight",
        _successor("weibull_softplus_unscaled@weight_coordinate", head=NNLS),
        transform="weight_coordinate",
    ),
    _ablation(
        "weibull_softplus_unscaled@signed_head",
        "weibull_softplus_unscaled",
        "head=signed",
        _successor(
            "weibull_softplus_unscaled@signed_head",
            head=models.HeadSpec(kind=models.HeadKind.RIDGE),
            ridge_grid=(1e-8, 1e-3, 1e-2, 0.1, 1.0),
        ),
    ),
    _ablation(
        "weibull_softplus_unscaled@scaled_head",
        "weibull_softplus_unscaled",
        "head=nonnegative_scaled",
        _successor("weibull_softplus_unscaled@scaled_head", head=SUCCESSOR_HEAD),
    ),
    _ablation(
        "weibull_softplus_unscaled@log_deficit_link",
        "weibull_softplus_unscaled",
        "link=log_deficit",
        _successor(
            "weibull_softplus_unscaled@log_deficit_link",
            head=models.HeadSpec(kind=models.HeadKind.NNLS, link=models.LinkKind.LOG_DEFICIT),
        ),
    ),
    _ablation(
        "weibull_softplus_unscaled@family_harm",
        "weibull_softplus_unscaled",
        "harm=softplus_log_quadratic_family,families=domain_quality",
        _successor(
            "weibull_softplus_unscaled@family_harm", _options(SUCCESSOR_OPTIONS, harm="softplus_family"), head=NNLS
        ),
    ),
    _ablation(
        "weibull_softplus_unscaled@exp_benefit_matched",
        "weibull_softplus_unscaled",
        "benefit=exp_saturation_budget_matched",
        _successor(
            "weibull_softplus_unscaled@exp_benefit_matched",
            _options(SUCCESSOR_OPTIONS, benefit="saturation"),
            SUCCESSOR_EXP_MATCHED_SHAPES,
            head=NNLS,
        ),
    ),
    _ablation(
        "weibull_softplus_unscaled@huber_head",
        "weibull_softplus_unscaled",
        "estimator=huber_irls",
        _successor(
            "weibull_softplus_unscaled@huber_head",
            head=models.HeadSpec(kind=models.HeadKind.HUBER_NNLS),
        ),
    ),
    _ablation(
        "weibull_softplus_unscaled@wide_grid",
        "weibull_softplus_unscaled",
        "search=wide_grid",
        _successor("weibull_softplus_unscaled@wide_grid", shapes=SUCCESSOR_WIDE_SHAPES, head=NNLS),
    ),
    _ablation(
        "weibull_softplus_unscaled@interaction_total",
        "weibull_softplus_unscaled",
        "interaction=total_benefit_square",
        _successor(
            "weibull_softplus_unscaled@interaction_total",
            _options(SUCCESSOR_OPTIONS, interaction="total_square"),
            head=NNLS,
        ),
    ),
    _ablation(
        "weibull_softplus_unscaled@interaction_family",
        "weibull_softplus_unscaled",
        "interaction=family_pair_products",
        _successor(
            "weibull_softplus_unscaled@interaction_family",
            _options(SUCCESSOR_OPTIONS, interaction="family_products"),
            head=NNLS,
        ),
    ),
    _ablation(
        "weibull_softplus_unscaled@quality_benefit",
        "weibull_softplus_unscaled",
        "families=quality_axis_benefit",
        _successor(
            "weibull_softplus_unscaled@quality_benefit", _options(SUCCESSOR_OPTIONS, quality_axis="benefit"), head=NNLS
        ),
    ),
    _ablation(
        "weibull_softplus_unscaled@quality_both",
        "weibull_softplus_unscaled",
        "families=quality_axis_benefit_and_harm",
        _successor(
            "weibull_softplus_unscaled@quality_both", _options(SUCCESSOR_OPTIONS, quality_axis="both"), head=NNLS
        ),
    ),
    _ablation(
        "weibull_softplus_unscaled@quality_both_shuffled",
        "weibull_softplus_unscaled",
        "families=quality_axis_shuffled_control",
        _successor(
            "weibull_softplus_unscaled@quality_both_shuffled",
            _options(SUCCESSOR_OPTIONS, quality_axis="both", shuffled_quality=True),
            head=NNLS,
        ),
        role="control",
    ),
    _ablation(
        "weibull_softplus_unscaled@log_deficit_bounded_link",
        "weibull_softplus_unscaled",
        "link=log_deficit_bounded",
        _successor(
            "weibull_softplus_unscaled@log_deficit_bounded_link",
            head=models.HeadSpec(kind=models.HeadKind.NNLS, link=models.LinkKind.LOG_DEFICIT_BOUNDED),
        ),
    ),
    _ablation(
        "weibull_softplus_unscaled@ablation_prior",
        "weibull_softplus_unscaled",
        "estimator=significance_prior_ridge",
        _successor_prior("weibull_softplus_unscaled@ablation_prior", scrambled=False),
    ),
    _ablation(
        "weibull_softplus_unscaled@scrambled_prior",
        "weibull_softplus_unscaled",
        "estimator=scrambled_prior_control",
        _successor_prior("weibull_softplus_unscaled@scrambled_prior", scrambled=True),
        role="control",
    ),
    _ablation(
        "weibull_softplus_unscaled@shared_shape_target",
        "weibull_softplus_unscaled",
        "search=shared_shape_per_target_group",
        _successor("weibull_softplus_unscaled@shared_shape_target", head=NNLS),
    ),
    _ablation(
        "weibull_softplus_unscaled@shared_shape_panel",
        "weibull_softplus_unscaled",
        "search=shared_shape_per_panel",
        _successor("weibull_softplus_unscaled@shared_shape_panel", head=NNLS),
    ),
    _ablation(
        "weibull_softplus_unscaled@shared_shape_scale",
        "weibull_softplus_unscaled",
        "search=shared_shape_across_39bucket_panels",
        _successor("weibull_softplus_unscaled@shared_shape_scale", head=NNLS),
    ),
    _ablation(
        "weibull_softplus_unscaled@refined_shape",
        "weibull_softplus_unscaled",
        "search=grid_then_nelder_mead",
        _refined(_successor("weibull_softplus_unscaled@refined_shape", head=NNLS)),
    ),
    _ablation(
        "weibull_softplus_unscaled@wide_grid_refined",
        "weibull_softplus_unscaled",
        "search=wide_grid_then_nelder_mead",
        _refined(_successor("weibull_softplus_unscaled@wide_grid_refined", shapes=SUCCESSOR_WIDE_SHAPES, head=NNLS)),
    ),
    _ablation(
        "weibull_softplus_unscaled@refined_bounded",
        "weibull_softplus_unscaled",
        "search=grid_then_bounded_nelder_mead",
        _refined(_successor("weibull_softplus_unscaled@refined_bounded", head=NNLS), bounded=True),
    ),
    _ablation(
        "weibull_softplus_unscaled@wide_grid_refined_bounded",
        "weibull_softplus_unscaled",
        "search=wide_grid_then_bounded_nelder_mead",
        _refined(
            _successor("weibull_softplus_unscaled@wide_grid_refined_bounded", shapes=SUCCESSOR_WIDE_SHAPES, head=NNLS),
            bounded=True,
        ),
    ),
    _ablation(
        "weibull_softplus_unscaled@link_by_cv",
        "weibull_softplus_unscaled",
        "link=selected_by_inner_cv",
        _with_link_candidates(
            _successor("weibull_softplus_unscaled@link_by_cv", head=NNLS),
            (models.LinkKind.IDENTITY, models.LinkKind.LOG_DEFICIT_BOUNDED),
        ),
    ),
    _ablation(
        "weibull_softplus_unscaled@outcome_permutation",
        "weibull_softplus_unscaled",
        "control=outcome_permutation",
        _successor("weibull_softplus_unscaled@outcome_permutation", head=NNLS),
        transform="outcome_permutation",
        role="control",
    ),
)
ALL_ENTRIES: tuple[ModelEntry, ...] = (
    PARENTS + REFERENCES + ABLATIONS + ROW_SCRAMBLED_CONTROLS + SUCCESSORS + SUCCESSOR_ABLATIONS
)
SUCCESSOR_BY_ID = {entry.model_id: entry for entry in SUCCESSORS}
ENTRY_BY_ID = {entry.model_id: entry for entry in ALL_ENTRIES}
if len(ENTRY_BY_ID) != len(ALL_ENTRIES):
    raise ValueError("duplicate model ids in the registry")
COVERED_SOURCES = {source for entry in PARENTS for source in entry.source_model_ids}
if COVERED_SOURCES != set(OBSERVATORY_MODEL_IDS):
    raise ValueError(
        "registry does not cover the Observatory inventory exactly: "
        "{sorted(set(OBSERVATORY_MODEL_IDS) ^ COVERED_SOURCES)}"
    )


def apply_transform(features: models.Features, entry: ModelEntry) -> models.Features:
    """Apply the entry's matched-control feature transform, if any."""
    transform = entry.feature_transform
    if transform is None or transform == "outcome_permutation":
        return features
    if transform == "permuted_inventory":
        return features.with_permuted_inventory(TRANSFORM_SEED)
    if transform == "weight_coordinate":
        return features.with_weight_coordinate()
    if transform == "shuffled_families":
        return features.with_families(models.shuffled_families(features.families, TRANSFORM_SEED), "shuffled_families")
    if transform == "no_families":
        return features.with_families(models.no_families(features.buckets), "no_families")
    raise ValueError(f"unknown feature transform {transform}")


def equivalence_classes() -> dict[str, tuple[str, ...]]:
    return {entry.equivalence_class: entry.source_model_ids for entry in PARENTS}
