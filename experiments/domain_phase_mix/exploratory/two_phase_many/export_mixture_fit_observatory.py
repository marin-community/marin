# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "cvxpy",
#   "fsspec",
#   "gcsfs",
#   "numpy",
#   "pandas",
#   "plotly",
#   "scikit-learn",
#   "scipy",
#   "tabulate",
# ]
# ///
"""Export the multi-swarm mixture-fit observatory data bundle.

The existing 300M debugger remains the source of truth for its carefully
accounted fit/heldout split and grouped OOF predictions. This exporter wraps
that bundle, fits the same surrogate family on the two StarCoder surfaces and
the production Grug-MoE swarm, and emits semantic parameter records for the
Fit Explorer.

Every expensive swarm/model result is cached independently. Re-running this
script skips complete fits and regenerates only stale outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.special import softplus
from scipy.stats import spearmanr
from sklearn.model_selection import KFold

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix import olmix_loglinear_fit as olmix_loglinear  # noqa: E402
from experiments.domain_phase_mix.dolma3_dolmino_top_level_domains import (  # noqa: E402
    TARGET_BUDGET_DOLMA3_COMMON_CRAWL,
    TOP_LEVEL_DOMAIN_TOKEN_COUNTS,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    analyze_original_separate_heads_policy_ablation_300m as separate_heads,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_grp_domain_saturation_phase_heads_20260714 as phase_head_grp,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_grp_family_onset_phase_heads_20260714 as family_onset_grp,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_hierarchical_coverage_grp_20260715 as hierarchical_grp,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_nested_coverage_dsp as coverage_dsp,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_partially_pooled_phase_bowls as pooled,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_production_grp_retained_hybrids_20260713 as retained_grp,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_retained_weibull_replay_20260713 as compact_retained,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    export_mixture_fit_debugger_300m as legacy_exporter,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    fit_production_grp_quality_variants as family_grp,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    materialize_symmetric_sepheads_geometry_frontier_panel_300m as symmetric_sepheads,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.standalone_code import dsp_exact as dsp  # noqa: E402
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search import (  # noqa: E402
    generic_family_penalty_calibration as grp_calibration,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search import (  # noqa: E402
    starcoder_grp,
)

SCRIPT_DIR = Path(__file__).resolve().parent
APP_DATA = SCRIPT_DIR / "mixture_fit_debugger/src/generated/dashboard_data.json"
LEGACY_DATA = SCRIPT_DIR / "reference_outputs/mixture_fit_debugger_300m_v1/dashboard_data.json"
CACHE_DIR = SCRIPT_DIR / "reference_outputs/mixture_fit_observatory_cache_20260713"
COSINE_DATA = SCRIPT_DIR.parent / "paper_plots/data/two_phase_starcoder_combined_143_from_wandb.csv"
WSD80_DATA = SCRIPT_DIR / "reference_outputs/starcoder_wsd80_surface_analysis_20260711/wsd80_observed_metrics.csv"
PRODUCTION_DATA = SCRIPT_DIR / (
    "reference_outputs/grug_moe_production_swarm_results_20260704/production_swarm_840_wide.csv"
)
PRODUCTION_MODEL = SCRIPT_DIR / (
    "reference_outputs/grug_moe_production_swarm_effective_exposure_dsp_uncheatable_20260705/model.json"
)
COMPACT_RETAINED_OUTPUT = SCRIPT_DIR / "reference_outputs/retained_weibull_replay_20260713"
BUCKET_FAMILY_MODEL = SCRIPT_DIR / (
    "reference_outputs/production_grp_quality_variants_20260713/bucket_resolved_quality_model.json"
)
ONE_PHASE_300M_DATA = SCRIPT_DIR / (
    "reference_outputs/one_phase_swarm_scores_export_300m_20260630/"
    "one_phase_augmented_fit_panel_uncheatable_table9_scores_300m.csv"
)
DELPHI_3E18_DATA = SCRIPT_DIR / (
    "reference_outputs/delphi_augmented_swarm_3e18_20260714/delphi_augmented_swarm_3e18_wide.csv"
)
DELPHI_3E18_HELDOUTS = SCRIPT_DIR / ("reference_outputs/delphi_3e18_append_only_heldouts_20260714/heldout_current.csv")
DELPHI_3E18_ONE_PHASE_DIR = SCRIPT_DIR / "reference_outputs/delphi_one_phase_augmented_swarm_3e18_20260715"
DELPHI_3E18_ONE_PHASE_MANIFEST = DELPHI_3E18_ONE_PHASE_DIR / "training_manifest.csv"
DELPHI_3E18_ONE_PHASE_WEIGHTS = DELPHI_3E18_ONE_PHASE_DIR / "phase_weights.csv"
DELPHI_3E18_ONE_PHASE_SERIES = "delphi_one_phase_augmented_swarm_3e18_20260715"
MINIMUM_DELPHI_3E18_HELDOUTS = 1_342
REQUIRED_DELPHI_3E18_HELDOUT_SERIES = {
    DELPHI_3E18_ONE_PHASE_SERIES: 238,
    "delphi_3e18_adversarial_stress_panel_20260716": 120,
    "delphi_3e18_frontier_phase_fiber_20260719": 200,
    "hpr_300m_to_3e18_optimum_validation_panel_20260720": 62,
    "hpr_3e18_to_3e18_optimum_validation_panel_20260720": 62,
    "delphi_3e18_frontier_random_phase_population_20260720": 296,
}
STARCODER_TARGET_COLUMN = "eval/paloma/dolma_100_programing_languages/bpb"
STARCODER_DOMAINS = ["nemotron_full", "starcoder"]
MODEL_IDS = (
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
    "bucket_family_power_separate_heads",
    "bucket_family_power_separate_heads_family_onset",
    "bucket_family_weibull_shared_onset",
    "bucket_family_weibull_family_replay",
)
HIDDEN_MODEL_IDS = {
    "bucket_family_power_separate_heads_family_onset",
    "bucket_family_weibull_shared_onset",
    "bucket_family_weibull_family_replay",
}
VISIBLE_MODEL_IDS = tuple(model_id for model_id in MODEL_IDS if model_id not in HIDDEN_MODEL_IDS)
DELPHI_3E18_MODEL_IDS = VISIBLE_MODEL_IDS
BASELINE_MODEL_IDS = ("linear", "olmix_loglinear")
RETAINED_GRP_MODEL_IDS = (
    "bucket_family_weibull_shared_onset",
    "bucket_family_weibull_family_replay",
)
NEW_MODEL_IDS = (
    *BASELINE_MODEL_IDS,
    "compact_retained_state",
    "bucket_family_grp",
    "hierarchical_phase_bucket_replay",
    "bucket_family_power_separate_heads",
    "bucket_family_power_separate_heads_family_onset",
    *RETAINED_GRP_MODEL_IDS,
)
VISIBLE_NEW_MODEL_IDS = tuple(model_id for model_id in NEW_MODEL_IDS if model_id in VISIBLE_MODEL_IDS)
LEGACY_MODEL_IDS = tuple(model_id for model_id in MODEL_IDS if model_id not in NEW_MODEL_IDS)
SINGLE_PHASE = "single_phase"
TWO_PHASE = "two_phase"
POLICY_CLASSES = (SINGLE_PHASE, TWO_PHASE)
MODEL_LABELS = {
    "linear": "Linear",
    "olmix_loglinear": "OLMix log-linear",
    "canonical": "Canonical DSP",
    "effective_exposure": "Effective-exposure DSP",
    "effective_exposure_geometry": "Eff-exp DSP + geometry",
    "separate_heads": "Separate heads",
    "grp": "GRP (regularized)",
    "compact_retained_state": "Compact retained state",
    "bucket_family_grp": "Bucket-resolved family GRP",
    "hierarchical_phase_bucket_replay": "Hierarchical phase replay",
    "bucket_family_power_separate_heads": "Power + separate heads",
    "bucket_family_power_separate_heads_family_onset": "Power + separate heads, family onset",
    "bucket_family_weibull_shared_onset": "Weibull GRP, shared onset",
    "bucket_family_weibull_family_replay": "Weibull GRP, family replay",
}
MODEL_DESCRIPTIONS = {
    "linear": "An affine response in policy weights; a transparent no-curvature baseline.",
    "olmix_loglinear": "The OLMix positive log-linear response fit with Huber loss.",
    "canonical": "Phase-1 share changes benefit, while overexposure uses raw total exposure.",
    "effective_exposure": "A shared phase-1 multiplier changes both saturation and overexposure exposure.",
    "effective_exposure_geometry": "Effective-exposure DSP plus phase divergence and concentration features.",
    "separate_heads": "Independent phase-specific asymmetric exposure bowls.",
    "grp": "Retained exposure, grouped response features, and explicit overexposure penalties.",
    "compact_retained_state": "Retained learning with a shared Weibull response and one literal replay-harm channel.",
    "bucket_family_grp": "Bucket-specific responses plus nonlinear family coverage and family repetition penalties.",
    "hierarchical_phase_bucket_replay": (
        "Family-pooled bucket utility, saturating family coverage, member replay harm, and one global phase-shift cost."
    ),
    "bucket_family_power_separate_heads": (
        "Bucket and family power responses with independent early- and late-phase nonnegative amplitudes."
    ),
    "bucket_family_power_separate_heads_family_onset": (
        "Independent early/late power heads with a shrinkage-selected replay onset for each semantic family."
    ),
    "bucket_family_weibull_shared_onset": (
        "Shared Weibull learning, nonlinear family coverage, and aggregate family replay with one learned onset."
    ),
    "bucket_family_weibull_family_replay": (
        "Shared Weibull learning and family coverage with literal replay harm learned independently per family."
    ),
}
MODEL_FAMILIES = {
    "linear": ("baseline", "Baseline", "Linear"),
    "olmix_loglinear": ("baseline", "Baseline", "OLMix log-linear"),
    "canonical": ("dsp", "DSP", "Canonical"),
    "effective_exposure": ("dsp", "DSP", "Effective exposure"),
    "effective_exposure_geometry": ("dsp", "DSP", "Eff-exp + geometry"),
    "separate_heads": ("phase_heads", "Phase heads", "Separate heads"),
    "grp": ("grp", "GRP", "Original"),
    "compact_retained_state": ("grp", "GRP", "Compact retained state"),
    "bucket_family_grp": ("grp", "GRP", "Bucket-resolved family"),
    "hierarchical_phase_bucket_replay": ("grp", "GRP", "Hierarchical phase replay"),
    "bucket_family_power_separate_heads": ("grp", "GRP", "Power + separate heads"),
    "bucket_family_power_separate_heads_family_onset": (
        "grp",
        "GRP",
        "Power + separate heads, family onset",
    ),
    "bucket_family_weibull_shared_onset": ("grp", "GRP", "Weibull shared onset"),
    "bucket_family_weibull_family_replay": ("grp", "GRP", "Weibull family replay"),
}
SEPARATE_L2_GRID = (0.03, 0.1, 0.3, 1.0, 1.5, 3.0)
PRODUCTION_GRP_L2_GRID = (0.0, 1e-4, 1e-3, 1e-2, 0.1, 0.5, 1.0, 3.0)
STARCODER_SINGLE_GRP_ALPHA_GRID = (0.1, 0.3, 1.0, 3.0, 10.0)
STARCODER_SINGLE_GRP_TAU_GRID = (1.0, 3.0, 5.0, 7.0)
STARCODER_SINGLE_GRP_L2_GRID = (1e-3, 1e-2, 0.1, 1.0)
GRP_SHAPE_PARAMS = legacy_exporter.GRP_SHAPE_PARAMS
COMPACT_L2_GRID = (0.1, 1.0)
BUCKET_FAMILY_L2_GRID = (0.0, 0.01, 0.1, 1.0, 3.0)
BUCKET_FAMILY_SHAPE_COUNT = 24
HIERARCHICAL_PHASE_REPLAY_SHAPE_COUNT = 12
HIERARCHICAL_PHASE_REPLAY_TOP_SHAPES = 3
POWER_HEADS_SHAPE_COUNT = 16
RETAINED_GRP_L2_GRID = retained_grp.L2_GRID
RETAINED_GRP_SHAPE_COUNT = 32
CACHE_VERSION = "mixture-fit-observatory-v9-delphi-one-two-phase"
MODEL_CACHE_VERSIONS = {model_id: "v1" for model_id in MODEL_IDS}
MODEL_CACHE_VERSIONS["bucket_family_power_separate_heads_family_onset"] = "v2"
LOWER_TAIL_FRACTION = 0.15
LOWER_TAIL_MIN_COUNT = 5


@dataclass(frozen=True)
class FittedResult:
    model: Any
    prediction: np.ndarray
    full_prediction: np.ndarray
    fit_detail: dict[str, Any]
    nike_swoosh: dict[str, Any] | None = None


class Predictable(Protocol):
    def predict(self, weights: np.ndarray) -> np.ndarray: ...


def is_dolma39_dataset(dataset: pooled.Dataset) -> bool:
    return dataset.name.startswith(("300m_", "delphi_3e18_"))


class UngroupedGRP:
    """GRP ablation with no semantic family or pair structure."""

    def __init__(
        self,
        dataset: pooled.Dataset,
        *,
        exponent: float,
        eta: float,
        retention_lambda: float,
        threshold: float,
        l2: float,
    ):
        self.dataset = dataset
        self.exponent = float(exponent)
        self.eta = float(eta)
        self.retention_lambda = float(retention_lambda)
        self.threshold = float(threshold)
        self.l2 = float(l2)
        self.intercept: float | None = None
        self.signal_coef: np.ndarray | None = None
        self.penalty_coef: np.ndarray | None = None

    def _design(self, weights: np.ndarray) -> np.ndarray:
        p0 = weights[:, 0, :]
        p1 = weights[:, 1, :]
        e0 = p0 * self.dataset.c0[None, :]
        e1 = p1 * self.dataset.c1[None, :]
        exposure = np.exp(-self.retention_lambda * (1.0 - p1)) * e0 + self.eta * e1
        signal = np.maximum(exposure, 1e-12) ** self.exponent
        penalty = softplus(np.log1p(exposure) - self.threshold) ** 2
        return np.hstack([-signal, penalty])

    def fit(self, indices: np.ndarray) -> UngroupedGRP:
        design = self._design(self.dataset.weights[indices])
        target = self.dataset.y[indices]
        design_mean = design.mean(axis=0, keepdims=True)
        target_mean = float(target.mean())
        centered = design - design_mean
        centered_target = target - target_mean
        if self.l2 > 0.0:
            centered = np.vstack([centered, np.sqrt(self.l2) * np.eye(design.shape[1])])
            centered_target = np.concatenate([centered_target, np.zeros(design.shape[1])])
        coef, _residual = nnls(centered, centered_target, maxiter=20 * design.shape[1])
        self.intercept = target_mean - float((design_mean @ coef).item())
        self.signal_coef = np.asarray(coef[: self.dataset.m], dtype=float)
        self.penalty_coef = np.asarray(coef[self.dataset.m :], dtype=float)
        return self

    def predict(self, weights: np.ndarray) -> np.ndarray:
        if self.intercept is None or self.signal_coef is None or self.penalty_coef is None:
            raise RuntimeError("Ungrouped GRP must be fit before prediction")
        design = self._design(weights)
        coef = np.concatenate([self.signal_coef, self.penalty_coef])
        return np.asarray(self.intercept + design @ coef, dtype=float)


class BucketFamilyGRP:
    """Bucket-level diminishing returns with pooled family coverage and replay harm."""

    def __init__(
        self,
        dataset: family_grp.Dataset,
        shape: family_grp.Shape,
        l2: float,
        head: family_grp.FittedHead,
    ):
        self.dataset = dataset
        self.shape = shape
        self.l2 = float(l2)
        self.head = head

    def predict(self, weights: np.ndarray) -> np.ndarray:
        prediction_dataset = replace(
            self.dataset,
            weights=np.asarray(weights, dtype=float),
            target=np.zeros(len(weights), dtype=float),
        )
        design, _names = family_grp.build_design(
            prediction_dataset,
            family_grp.Variant.BUCKET_RESOLVED,
            self.shape,
        )
        return self.head.predict_design(design)


class PowerSeparateHeadsGRP:
    """Bucket/family power responses with phase-specific nonnegative heads."""

    def __init__(
        self,
        dataset: family_grp.Dataset,
        variant: phase_head_grp.Variant,
        shape: retained_grp.Shape,
        l2: float,
        head: family_grp.FittedHead,
    ):
        self.dataset = dataset
        self.variant = variant
        self.shape = shape
        self.l2 = float(l2)
        self.head = head

    def predict(self, weights: np.ndarray) -> np.ndarray:
        prediction_dataset = replace(
            self.dataset,
            weights=np.asarray(weights, dtype=float),
            target=np.zeros(len(weights), dtype=float),
        )
        design, _names, _layout = phase_head_grp.build_design(
            prediction_dataset,
            self.variant,
            self.shape,
            None,
        )
        return self.head.predict_design(design)


class PowerSeparateHeadsFamilyOnsetGRP:
    """Power heads with shrinkage-selected family replay onsets."""

    def __init__(
        self,
        dataset: family_grp.Dataset,
        variant: family_onset_grp.Variant,
        shape: retained_grp.Shape,
        l2: float,
        tau_shrink: float,
        fitted: family_onset_grp.FittedModel,
    ):
        if fitted.family_tau is None:
            raise ValueError("Family-onset model requires fitted family thresholds")
        self.dataset = dataset
        self.variant = variant
        self.shape = shape
        self.l2 = float(l2)
        self.tau_shrink = float(tau_shrink)
        self.head = fitted.head
        self.family_tau = fitted.family_tau

    def predict(self, weights: np.ndarray) -> np.ndarray:
        prediction_dataset = replace(
            self.dataset,
            weights=np.asarray(weights, dtype=float),
            target=np.zeros(len(weights), dtype=float),
        )
        design, _names = family_onset_grp.build_design(
            prediction_dataset,
            self.variant,
            self.shape,
            self.family_tau,
        )
        return self.head.predict_design(design)


class RetainedFamilyGRP:
    """Shared Weibull learning with explicit family coverage and replay channels."""

    def __init__(
        self,
        dataset: family_grp.Dataset,
        variant: retained_grp.Variant,
        shape: retained_grp.Shape,
        l2: float,
        head: family_grp.FittedHead,
    ):
        self.dataset = dataset
        self.variant = variant
        self.shape = shape
        self.l2 = float(l2)
        self.head = head

    def predict(self, weights: np.ndarray) -> np.ndarray:
        prediction_dataset = replace(
            self.dataset,
            weights=np.asarray(weights, dtype=float),
            target=np.zeros(len(weights), dtype=float),
        )
        design, _names = retained_grp.build_design(prediction_dataset, self.variant, self.shape)
        return self.head.predict_design(design)


RETAINED_GRP_VARIANTS = {
    "bucket_family_weibull_shared_onset": retained_grp.VARIANT_BY_NAME["weibull_global_tau"],
    "bucket_family_weibull_family_replay": retained_grp.VARIANT_BY_NAME["weibull_family_coverage_family_replay"],
}


COMPACT_TWO_PHASE_CONFIG = compact_retained.ModelConfig(
    "revisit_retention_weibull_shared_replay",
    compact_retained.SignalKind.RETAINED_STATE,
    compact_retained.ResponseKind.WEIBULL,
    compact_retained.RetentionKind.REVISIT_GATED,
    compact_retained.ReplayPenaltyKind.SHARED,
)
COMPACT_ONE_PHASE_CONFIG = compact_retained.ModelConfig(
    "one_phase_weibull_shared_replay",
    compact_retained.SignalKind.TOTAL_EXPOSURE,
    compact_retained.ResponseKind.WEIBULL,
    compact_retained.RetentionKind.CONSTANT,
    compact_retained.ReplayPenaltyKind.SHARED,
)


def compact_config(policy_class: str) -> compact_retained.ModelConfig:
    return COMPACT_ONE_PHASE_CONFIG if policy_class == SINGLE_PHASE else COMPACT_TWO_PHASE_CONFIG


def compact_fit(
    dataset: pooled.Dataset,
    indices: np.ndarray,
    l2: float,
    policy_class: str,
) -> compact_retained.FittedModel:
    return compact_retained.fit_model(
        dataset,
        indices,
        compact_config(policy_class),
        l2,
        maxiter=24,
        top_k=2,
    )


def compact_checkpoint_prediction(
    dataset: pooled.Dataset,
    policy_class: str,
    l2: float,
    seed: int,
) -> np.ndarray | None:
    if policy_class != TWO_PHASE or dataset.name not in {
        "300m_uncheatable",
        "300m_table9",
        "production_uncheatable",
    }:
        return None
    stem = compact_retained.checkpoint_stem(dataset.name, compact_config(policy_class), l2, seed)
    path = COMPACT_RETAINED_OUTPUT / "checkpoints" / f"{stem}.npy"
    if not path.exists():
        return None
    prediction = np.load(path)
    if prediction.shape != (dataset.n,) or not np.isfinite(prediction).all():
        raise ValueError(f"Invalid compact retained-state checkpoint {path}")
    return np.asarray(prediction, dtype=float)


def compact_oof_prediction(
    dataset: pooled.Dataset,
    policy_class: str,
    l2: float,
    seed: int,
) -> np.ndarray:
    cached = compact_checkpoint_prediction(dataset, policy_class, l2, seed)
    if cached is not None:
        return cached
    prediction = np.full(dataset.n, np.nan, dtype=float)
    for train, test in folds(dataset, seed):
        prediction[test] = compact_fit(dataset, train, l2, policy_class).predict(dataset.weights[test])
    if not np.isfinite(prediction).all():
        raise ValueError(f"Incomplete compact retained-state OOF prediction for {dataset.name}")
    return prediction


def select_compact_l2(dataset: pooled.Dataset, policy_class: str) -> tuple[float, list[dict[str, float]]]:
    rows: list[dict[str, float]] = []
    for l2 in COMPACT_L2_GRID:
        prediction = compact_oof_prediction(dataset, policy_class, l2, seed=0)
        summary = metric_summary(dataset.y, prediction)
        rows.append(
            {
                "l2": float(l2),
                "oofRmse": float(summary["rmse"]),
                "oofSpearman": float(summary["spearman"]),
            }
        )
    selected = min(rows, key=lambda row: (row["oofRmse"], -row["oofSpearman"], row["l2"]))
    return float(selected["l2"]), rows


def family_partition(dataset: pooled.Dataset) -> tuple[tuple[str, ...], tuple[np.ndarray, ...], np.ndarray]:
    if is_dolma39_dataset(dataset):
        family_map = legacy_exporter.grp_packet(dataset).family_map
        names = tuple(sorted(family_map))
        members = tuple(np.asarray(family_map[name], dtype=int) for name in names)
    elif dataset.name == "production_uncheatable":
        grouped: dict[str, list[int]] = {}
        for index, domain in enumerate(dataset.domain_names):
            match = family_grp.DOMAIN_PATTERN.fullmatch(domain)
            family = f"c{match.group('family')}" if match is not None else domain
            grouped.setdefault(family, []).append(index)
        names = tuple(grouped)
        members = tuple(np.asarray(grouped[name], dtype=int) for name in names)
    else:
        names = tuple(dataset.domain_names)
        members = tuple(np.asarray([index], dtype=int) for index in range(dataset.m))
    covered = np.concatenate(members)
    if sorted(covered.tolist()) != list(range(dataset.m)):
        raise ValueError(f"Family partition does not cover {dataset.name} exactly once")
    quality = np.full(dataset.m, -1, dtype=int)
    return names, members, quality


def family_dataset(dataset: pooled.Dataset) -> family_grp.Dataset:
    names, members, quality = family_partition(dataset)
    return family_grp.Dataset(
        frame=dataset.frame,
        target=np.asarray(dataset.y, dtype=float),
        weights=np.asarray(dataset.weights, dtype=float),
        c0=np.asarray(dataset.c0, dtype=float),
        c1=np.asarray(dataset.c1, dtype=float),
        domains=tuple(dataset.domain_names),
        family_names=names,
        family_members=members,
        quality=quality,
    )


def bucket_shape_candidates(policy_class: str) -> tuple[family_grp.Shape, ...]:
    candidates = list(family_grp.shape_candidates(family_grp.Variant.BUCKET_RESOLVED, BUCKET_FAMILY_SHAPE_COUNT))
    if BUCKET_FAMILY_MODEL.exists():
        saved = json.loads(BUCKET_FAMILY_MODEL.read_text())["shape"]
        candidates.insert(0, family_grp.Shape(**saved))
    if policy_class == SINGLE_PHASE:
        candidates = [replace(shape, late_multiplier=1.0, forgetting_rate=0.0) for shape in candidates]
    return tuple(dict.fromkeys(candidates))


def bucket_fit(
    dataset: pooled.Dataset,
    indices: np.ndarray,
    shape: family_grp.Shape,
    l2: float,
) -> BucketFamilyGRP:
    structured = family_dataset(dataset)
    design, names = family_grp.build_design(structured, family_grp.Variant.BUCKET_RESOLVED, shape)
    head = family_grp.fit_head(design, structured.target, indices, l2, names)
    return BucketFamilyGRP(structured, shape, l2, head)


def select_bucket_hyperparameters(
    dataset: pooled.Dataset,
    policy_class: str,
) -> tuple[family_grp.Shape, float, list[dict[str, float]]]:
    rows: list[dict[str, float]] = []
    best: tuple[float, float, float, int, family_grp.Shape] | None = None
    for shape_index, shape in enumerate(bucket_shape_candidates(policy_class)):
        for l2 in BUCKET_FAMILY_L2_GRID:
            prediction = np.full(dataset.n, np.nan, dtype=float)
            for train, test in folds(dataset, seed=0):
                prediction[test] = bucket_fit(dataset, train, shape, l2).predict(dataset.weights[test])
            summary = metric_summary(dataset.y, prediction)
            row = {
                "shapeIndex": float(shape_index),
                "exponent": shape.exponent,
                "lateMultiplier": shape.late_multiplier,
                "forgettingRate": shape.forgetting_rate,
                "penaltyThreshold": shape.penalty_threshold,
                "l2": float(l2),
                "oofRmse": float(summary["rmse"]),
                "oofSpearman": float(summary["spearman"]),
            }
            rows.append(row)
            candidate = (row["oofRmse"], -row["oofSpearman"], float(l2), shape_index, shape)
            if best is None or candidate[:4] < best[:4]:
                best = candidate
    if best is None:
        raise RuntimeError(f"No bucket-family hyperparameter candidates for {dataset.name}")
    return best[4], best[2], rows


def hierarchical_phase_replay_shape_candidates(policy_class: str) -> tuple[family_grp.Shape, ...]:
    candidates = list(
        family_grp.shape_candidates(
            family_grp.Variant.BUCKET_RESOLVED,
            HIERARCHICAL_PHASE_REPLAY_SHAPE_COUNT,
        )
    )
    if policy_class == SINGLE_PHASE:
        candidates = [replace(shape, late_multiplier=1.0, forgetting_rate=0.0) for shape in candidates]
    return tuple(dict.fromkeys(candidates))


def select_hierarchical_phase_replay_config(
    dataset: pooled.Dataset,
    policy_class: str,
) -> tuple[hierarchical_grp.Config, dict[str, Any]]:
    structured = family_dataset(dataset)
    shapes = hierarchical_phase_replay_shape_candidates(policy_class)
    splits = folds(dataset, hierarchical_grp.SCREEN_SEED)
    _baseline, _baseline_prediction, baseline_rows = hierarchical_grp.score_configs(
        structured,
        hierarchical_grp.baseline_configs(shapes),
        splits,
    )
    best_by_shape: dict[int, float] = {}
    for row in baseline_rows:
        shape_index = int(row["shape_index"])
        best_by_shape[shape_index] = min(best_by_shape.get(shape_index, float("inf")), float(row["rmse"]))
    shape_indices = [
        shape_index
        for shape_index, _rmse in sorted(best_by_shape.items(), key=lambda item: item[1])[
            :HIERARCHICAL_PHASE_REPLAY_TOP_SHAPES
        ]
    ]
    config, _prediction, candidate_rows = hierarchical_grp.score_configs(
        structured,
        hierarchical_grp.structural_configs(
            hierarchical_grp.Variant.HIERARCHICAL_PHASE_BUCKET_REPLAY,
            shapes,
            shape_indices,
        ),
        splits,
    )
    return config, {
        "baselineShapeScreen": baseline_rows,
        "candidateSweep": candidate_rows,
        "screenSeed": hierarchical_grp.SCREEN_SEED,
        "topShapeIndices": shape_indices,
    }


def hierarchical_phase_replay_fit(
    dataset: pooled.Dataset,
    indices: np.ndarray,
    config: hierarchical_grp.Config,
) -> hierarchical_grp.Model:
    return hierarchical_grp.fit_model(family_dataset(dataset), config, indices)


def power_heads_variant(policy_class: str) -> phase_head_grp.Variant:
    name = "power_eta" if policy_class == SINGLE_PHASE else "power_separate_heads"
    return phase_head_grp.VARIANT_BY_NAME[name]


def power_heads_shape_candidates(policy_class: str) -> tuple[retained_grp.Shape, ...]:
    variant = power_heads_variant(policy_class)
    candidates = list(phase_head_grp.candidate_shapes(variant, POWER_HEADS_SHAPE_COUNT))
    if policy_class == SINGLE_PHASE:
        candidates = [replace(shape, late_multiplier=1.0, forgetting_rate=0.0) for shape in candidates]
    return tuple(dict.fromkeys(candidates))


def power_heads_fit(
    dataset: pooled.Dataset,
    indices: np.ndarray,
    shape: retained_grp.Shape,
    l2: float,
    policy_class: str,
) -> PowerSeparateHeadsGRP:
    structured = family_dataset(dataset)
    variant = power_heads_variant(policy_class)
    design, names, _layout = phase_head_grp.build_design(structured, variant, shape, None)
    head = family_grp.fit_head(design, structured.target, indices, l2, names)
    return PowerSeparateHeadsGRP(structured, variant, shape, l2, head)


def power_family_onset_variant(policy_class: str) -> family_onset_grp.Variant:
    return family_onset_grp.Variant(
        name=f"observatory_{policy_class}_family_tau",
        phase=power_heads_variant(policy_class).phase,
        onset=family_onset_grp.OnsetScope.FAMILY,
    )


def power_family_onset_fit(
    dataset: pooled.Dataset,
    indices: np.ndarray,
    shape: retained_grp.Shape,
    l2: float,
    tau_shrink: float,
    policy_class: str,
    *,
    multistart: bool,
) -> PowerSeparateHeadsFamilyOnsetGRP:
    structured = family_dataset(dataset)
    variant = power_family_onset_variant(policy_class)
    selection = phase_head_grp.SharedSelection(shape=shape, l2=l2, inner_rmse=float("nan"))
    fitted = family_onset_grp.fit_family_tau_model(
        structured,
        variant,
        selection,
        indices,
        tau_shrink,
        maxiter=family_onset_grp.DEFAULT_TAU_MAXITER,
        multistart=multistart,
    )
    return PowerSeparateHeadsFamilyOnsetGRP(structured, variant, shape, l2, tau_shrink, fitted)


def select_power_heads_hyperparameters(
    dataset: pooled.Dataset,
    policy_class: str,
) -> tuple[retained_grp.Shape, float, list[dict[str, float]]]:
    rows: list[dict[str, float]] = []
    best: tuple[float, float, float, int, retained_grp.Shape] | None = None
    for shape_index, shape in enumerate(power_heads_shape_candidates(policy_class)):
        for l2 in phase_head_grp.L2_GRID:
            prediction = np.full(dataset.n, np.nan, dtype=float)
            for train, test in folds(dataset, seed=0):
                prediction[test] = power_heads_fit(dataset, train, shape, l2, policy_class).predict(
                    dataset.weights[test]
                )
            summary = metric_summary(dataset.y, prediction)
            row = {
                "shapeIndex": float(shape_index),
                "exponent": shape.exponent,
                "lateMultiplier": shape.late_multiplier,
                "forgettingRate": shape.forgetting_rate,
                "penaltyThreshold": shape.penalty_threshold,
                "l2": float(l2),
                "oofRmse": float(summary["rmse"]),
                "oofSpearman": float(summary["spearman"]),
            }
            rows.append(row)
            candidate = (row["oofRmse"], -row["oofSpearman"], float(l2), shape_index, shape)
            if best is None or candidate[:4] < best[:4]:
                best = candidate
    if best is None:
        raise RuntimeError(f"No power-head hyperparameter candidates for {dataset.name}")
    return best[4], best[2], rows


def select_power_family_onset_hyperparameters(
    dataset: pooled.Dataset,
    policy_class: str,
) -> tuple[retained_grp.Shape, float, float, dict[str, Any]]:
    try:
        dataset_id = family_onset_grp.hierarchy.DatasetId(dataset.name)
    except ValueError:
        dataset_id = None

    if dataset_id is not None:
        structured = family_dataset(dataset)
        variant = power_family_onset_variant(policy_class)
        indices = np.arange(dataset.n)
        selection = family_onset_grp.select_shared_hyperparameters(
            structured,
            dataset_id,
            variant.phase,
            power_heads_shape_candidates(policy_class),
            indices,
            family_onset_grp.INNER_CV_SEED + 999,
            family_onset_grp.DEFAULT_INNER_SPLITS,
        )
        tau_shrink, tau_inner_rmse = family_onset_grp.select_tau_shrink(
            structured,
            dataset_id,
            variant,
            selection,
            indices,
            family_onset_grp.INNER_CV_SEED + 1999,
            family_onset_grp.DEFAULT_INNER_SPLITS,
            family_onset_grp.DEFAULT_TAU_MAXITER,
        )
        return (
            selection.shape,
            selection.l2,
            tau_shrink,
            {
                "innerSelection": {
                    "sharedRmse": selection.inner_rmse,
                    "tauRmse": tau_inner_rmse,
                    "innerSplits": family_onset_grp.DEFAULT_INNER_SPLITS,
                }
            },
        )

    shape, l2, shape_sweep = select_power_heads_hyperparameters(dataset, policy_class)
    tau_rows: list[dict[str, float]] = []
    best: tuple[float, float, float] | None = None
    for tau_shrink in family_onset_grp.TAU_SHRINK_GRID:
        prediction = np.full(dataset.n, np.nan, dtype=float)
        for train, test in folds(dataset, seed=0):
            model = power_family_onset_fit(
                dataset,
                train,
                shape,
                l2,
                tau_shrink,
                policy_class,
                multistart=False,
            )
            prediction[test] = model.predict(dataset.weights[test])
        summary = metric_summary(dataset.y, prediction)
        row = {
            "tauShrink": float(tau_shrink),
            "oofRmse": float(summary["rmse"]),
            "oofSpearman": float(summary["spearman"]),
        }
        tau_rows.append(row)
        candidate = (row["oofRmse"], -row["oofSpearman"], float(tau_shrink))
        if best is None or candidate < best:
            best = candidate
    if best is None:
        raise RuntimeError(f"No family-onset shrinkage candidates for {dataset.name}")
    return shape, l2, best[2], {"shapeSweep": shape_sweep, "tauShrinkSweep": tau_rows}


def retained_grp_shape_candidates(model_id: str, policy_class: str) -> tuple[retained_grp.Shape, ...]:
    variant = RETAINED_GRP_VARIANTS[model_id]
    candidates = list(retained_grp.shared_shape_candidates(variant, RETAINED_GRP_SHAPE_COUNT))
    if policy_class == SINGLE_PHASE:
        candidates = [replace(shape, late_multiplier=1.0, forgetting_rate=0.0) for shape in candidates]
    return tuple(dict.fromkeys(candidates))


def retained_grp_fit(
    dataset: pooled.Dataset,
    indices: np.ndarray,
    variant: retained_grp.Variant,
    shape: retained_grp.Shape,
    l2: float,
) -> RetainedFamilyGRP:
    structured = family_dataset(dataset)
    design, names = retained_grp.build_design(structured, variant, shape)
    head = family_grp.fit_head(design, structured.target, indices, l2, names)
    return RetainedFamilyGRP(structured, variant, shape, l2, head)


def select_retained_grp_hyperparameters(
    dataset: pooled.Dataset,
    model_id: str,
    policy_class: str,
) -> tuple[retained_grp.Shape, float, list[dict[str, float]]]:
    variant = RETAINED_GRP_VARIANTS[model_id]
    rows: list[dict[str, float]] = []
    best: tuple[float, float, float, int, retained_grp.Shape] | None = None
    for shape_index, shape in enumerate(retained_grp_shape_candidates(model_id, policy_class)):
        for l2 in RETAINED_GRP_L2_GRID:
            prediction = np.full(dataset.n, np.nan, dtype=float)
            for train, test in folds(dataset, seed=0):
                prediction[test] = retained_grp_fit(dataset, train, variant, shape, l2).predict(dataset.weights[test])
            summary = metric_summary(dataset.y, prediction)
            row = {
                "shapeIndex": float(shape_index),
                "rate": shape.rate,
                "power": shape.exponent,
                "lateMultiplier": shape.late_multiplier,
                "forgettingRate": shape.forgetting_rate,
                "penaltyThreshold": shape.penalty_threshold,
                "l2": float(l2),
                "oofRmse": float(summary["rmse"]),
                "oofSpearman": float(summary["spearman"]),
            }
            rows.append(row)
            candidate = (row["oofRmse"], -row["oofSpearman"], float(l2), shape_index, shape)
            if best is None or candidate[:4] < best[:4]:
                best = candidate
    if best is None:
        raise RuntimeError(f"No retained-family GRP hyperparameter candidates for {dataset.name}/{model_id}")
    return best[4], best[2], rows


def safe_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def file_fingerprint(paths: list[Path], payload: dict[str, Any], *, version: str = CACHE_VERSION) -> str:
    inputs = {
        str(path.relative_to(REPO_ROOT)): {"size": path.stat().st_size, "mtimeNs": path.stat().st_mtime_ns}
        for path in paths
    }
    return hashlib.sha256(
        json.dumps({"version": version, "inputs": inputs, **payload}, sort_keys=True).encode()
    ).hexdigest()


def parameter(
    key: str,
    symbol: str,
    value: float,
    role: str,
    *,
    scope: str = "global",
    domain_id: str | None = None,
    group_label: str | None = None,
    transformed_value: float | None = None,
    transformed_label: str | None = None,
    unit: str | None = None,
) -> dict[str, Any]:
    return {
        "key": key,
        "symbol": symbol,
        "value": safe_float(value),
        "role": role,
        "scope": scope,
        "domainId": domain_id,
        "groupLabel": group_label,
        "transformedValue": safe_float(transformed_value),
        "transformedLabel": transformed_label,
        "unit": unit,
    }


def metric_summary(
    observed: np.ndarray,
    prediction: np.ndarray,
    *,
    fold_test_indices: list[np.ndarray] | None = None,
) -> dict[str, float | int | None]:
    valid = np.isfinite(observed) & np.isfinite(prediction)
    if valid.sum() < 3:
        return {
            "n": int(valid.sum()),
            "rmse": None,
            "mae": None,
            "spearman": None,
            "regretAt1": None,
            "foldMeanRegretAt1": None,
            "lowerTailOptimism": None,
            "lowTailRmse": None,
            "lowerTailCount": 0,
        }
    valid_observed = observed[valid]
    valid_prediction = prediction[valid]
    residual = valid_prediction - valid_observed
    lower_tail_count = min(
        len(valid_observed),
        max(LOWER_TAIL_MIN_COUNT, math.ceil(LOWER_TAIL_FRACTION * len(valid_observed))),
    )
    lower_tail = np.argsort(valid_prediction)[:lower_tail_count]
    lower_tail_error = valid_observed[lower_tail] - valid_prediction[lower_tail]
    fold_regrets: list[float] = []
    if fold_test_indices is not None:
        for test_indices in fold_test_indices:
            fold_valid = np.asarray(test_indices, dtype=int)
            fold_valid = fold_valid[valid[fold_valid]]
            if len(fold_valid) == 0:
                continue
            selected = int(fold_valid[int(np.argmin(prediction[fold_valid]))])
            fold_regrets.append(float(observed[selected] - np.min(observed[fold_valid])))
    return {
        "n": int(valid.sum()),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
        "spearman": float(spearmanr(valid_observed, valid_prediction).statistic),
        "regretAt1": float(valid_observed[int(np.argmin(valid_prediction))] - np.min(valid_observed)),
        "foldMeanRegretAt1": float(np.mean(fold_regrets)) if fold_regrets else None,
        "lowerTailOptimism": float(np.mean(np.maximum(lower_tail_error, 0.0))),
        "lowTailRmse": float(np.sqrt(np.mean(lower_tail_error**2))),
        "lowerTailCount": int(lower_tail_count),
    }


def oof_test_indices(dataset: pooled.Dataset, seeds: tuple[int, ...]) -> list[np.ndarray]:
    return [test for seed in seeds for _train, test in folds(dataset, seed)]


def phase_fractions(dataset: pooled.Dataset) -> tuple[float, float]:
    return coverage_dsp.phase_fractions(dataset)


def baseline_feature_matrix(dataset: pooled.Dataset, weights: np.ndarray, policy_class: str) -> np.ndarray:
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 3 or weights.shape[1:] != (2, dataset.m):
        raise ValueError(f"Expected weights with shape (n, 2, {dataset.m}), got {weights.shape}")
    if policy_class == TWO_PHASE:
        return weights.reshape(len(weights), -1)
    if policy_class == SINGLE_PHASE:
        alpha0, alpha1 = phase_fractions(dataset)
        return alpha0 * weights[:, 0, :] + alpha1 * weights[:, 1, :]
    raise ValueError(f"Unknown policy class {policy_class!r}")


@dataclass(frozen=True)
class LinearBaseline:
    """Minimum-norm affine fit in policy-weight coordinates."""

    dataset: pooled.Dataset
    policy_class: str
    intercept: float
    coefficients: np.ndarray
    design_rank: int

    def predict(self, weights: np.ndarray) -> np.ndarray:
        features = baseline_feature_matrix(self.dataset, weights, self.policy_class)
        return np.asarray(self.intercept + features @ self.coefficients, dtype=float)


@dataclass(frozen=True)
class OlmixLoglinearBaseline:
    """Policy-aware wrapper around the frozen OLMix log-linear fit."""

    dataset: pooled.Dataset
    policy_class: str
    fit: olmix_loglinear.OlmixLoglinearFit

    def predict(self, weights: np.ndarray) -> np.ndarray:
        features = baseline_feature_matrix(self.dataset, weights, self.policy_class)
        return self.fit.predict(features)


def fit_linear_baseline(
    dataset: pooled.Dataset,
    indices: np.ndarray,
    policy_class: str,
) -> LinearBaseline:
    features = baseline_feature_matrix(dataset, dataset.weights[indices], policy_class)
    target = np.asarray(dataset.y[indices], dtype=float)
    feature_mean = features.mean(axis=0)
    target_mean = float(target.mean())
    centered_features = features - feature_mean
    coefficients, *_ = np.linalg.lstsq(centered_features, target - target_mean, rcond=None)
    intercept = target_mean - float(feature_mean @ coefficients)
    return LinearBaseline(
        dataset=dataset,
        policy_class=policy_class,
        intercept=intercept,
        coefficients=np.asarray(coefficients, dtype=float),
        design_rank=int(np.linalg.matrix_rank(centered_features)),
    )


def fit_olmix_loglinear_baseline(
    dataset: pooled.Dataset,
    indices: np.ndarray,
    policy_class: str,
) -> OlmixLoglinearBaseline:
    target = np.asarray(dataset.y[indices], dtype=float)
    if np.any(target <= 0.0):
        raise ValueError(f"OLMix log-linear requires positive targets for {dataset.name}")
    features = baseline_feature_matrix(dataset, dataset.weights[indices], policy_class)
    fit = olmix_loglinear.fit_olmix_loglinear_model(
        features,
        target,
        delta=olmix_loglinear.DEFAULT_HUBER_DELTA,
        seed=olmix_loglinear.FIT_START_SEED,
        n_starts=olmix_loglinear.FIT_N_STARTS,
    )
    return OlmixLoglinearBaseline(dataset=dataset, policy_class=policy_class, fit=fit)


def natural_weights(dataset: pooled.Dataset, alpha0: float) -> np.ndarray:
    phase0_tokens = alpha0 / np.maximum(dataset.c0, 1e-12)
    phase1_tokens = (1.0 - alpha0) / np.maximum(dataset.c1, 1e-12)
    token_proxy = 0.5 * (phase0_tokens + phase1_tokens)
    return token_proxy / token_proxy.sum()


def target_budget(dataset: pooled.Dataset, alpha0: float, known_budget: float | None = None) -> float:
    if known_budget is not None:
        return float(known_budget)
    natural = natural_weights(dataset, alpha0)
    implied = dataset.c0 * natural / max(alpha0, 1e-12)
    scale = float(np.median(implied))
    return scale


def folds(dataset: pooled.Dataset, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    if is_dolma39_dataset(dataset):
        return pooled.dataset_folds(dataset, seed, n_splits=5)
    splitter = KFold(n_splits=5, shuffle=True, random_state=seed)
    return [(train, test) for train, test in splitter.split(np.arange(dataset.n))]


def dsp_fit(
    dataset: pooled.Dataset,
    indices: np.ndarray,
    model_id: str,
    policy_class: str,
) -> coverage_dsp.CoverageModel:
    if policy_class not in POLICY_CLASSES:
        raise ValueError(f"Unknown policy class {policy_class!r}")
    single_phase = policy_class == SINGLE_PHASE
    config = coverage_dsp.FitConfig(
        name=model_id,
        use_coverage=model_id == "effective_exposure_geometry",
        variant_name=("no_phase" if single_phase else "canonical" if model_id == "canonical" else "effective_exposure"),
        coverage_indices=(1,) if single_phase else (0, 1, 2),
    )
    if dataset.name.startswith("production"):
        linear_reg, maxiter, top_k = 1e-6, 0, 3
    elif dataset.name.startswith("starcoder"):
        linear_reg, maxiter, top_k = 0.01, 24, 3
    else:
        linear_reg, maxiter, top_k = legacy_exporter.DSP_LINEAR_REG, legacy_exporter.DSP_MAXITER, 3
    return coverage_dsp.fit_model(
        dataset,
        indices,
        config,
        linear_reg=linear_reg,
        maxiter=maxiter,
        coarse_top_k=top_k,
    )


def dsp_predict(model: coverage_dsp.CoverageModel, dataset: pooled.Dataset, weights: np.ndarray) -> np.ndarray:
    alpha0, alpha1 = phase_fractions(dataset)
    return coverage_dsp.predict(model, weights, alpha0, alpha1)


def single_separate_predict(
    model: symmetric_sepheads.SeparateModel,
    dataset: pooled.Dataset,
    weights: np.ndarray,
) -> np.ndarray:
    exposure = weights[:, 0, :] * dataset.c0[None, :] + weights[:, 1, :] * dataset.c1[None, :]
    prediction = np.full(len(weights), model.intercept, dtype=float)
    prediction += pooled.bowl_design(exposure, model.mus[0]) @ model.coefs[0]
    return prediction


def select_separate_l2(
    dataset: pooled.Dataset,
    policy_class: str,
) -> tuple[float, list[dict[str, float | int | None]]]:
    if policy_class == TWO_PHASE and dataset.name == "300m_uncheatable":
        return 1.0, []
    if policy_class == TWO_PHASE and dataset.name == "300m_table9":
        return 1.5, []
    packet = coverage_dsp.packet(dataset, np.arange(dataset.n))
    rows = []
    for l2 in SEPARATE_L2_GRID:
        prediction = np.full(dataset.n, np.nan, dtype=float)
        for train, test in folds(dataset, seed=0):
            if policy_class == SINGLE_PHASE:
                model = symmetric_sepheads.fit_separate_model(dataset, train, policy="one_phase", l2=l2)
                prediction[test] = single_separate_predict(model, dataset, dataset.weights[test])
            else:
                model = separate_heads.fit_separate_heads(packet, train, l2)
                prediction[test] = separate_heads.predict_separate_heads(model, packet, dataset.weights[test])
        rows.append({"l2": l2, "oofRmse": metric_summary(dataset.y, prediction)["rmse"]})
    selected = min(rows, key=lambda row: (float(row["oofRmse"]), row["l2"]))
    return float(selected["l2"]), rows


def separate_fit(dataset: pooled.Dataset, indices: np.ndarray, l2: float, policy_class: str) -> Any:
    if policy_class == SINGLE_PHASE:
        return symmetric_sepheads.fit_separate_model(dataset, indices, policy="one_phase", l2=l2)
    packet = coverage_dsp.packet(dataset, np.arange(dataset.n))
    return separate_heads.fit_separate_heads(packet, indices, l2)


def separate_predict(
    model: Any,
    dataset: pooled.Dataset,
    weights: np.ndarray,
    policy_class: str,
) -> np.ndarray:
    if policy_class == SINGLE_PHASE:
        return single_separate_predict(model, dataset, weights)
    packet = coverage_dsp.packet(dataset, np.arange(dataset.n))
    return separate_heads.predict_separate_heads(model, packet, weights)


def starcoder_grp_packet(dataset: pooled.Dataset) -> dsp.PacketData:
    return coverage_dsp.packet(dataset, np.arange(dataset.n))


def starcoder_grp_fit(dataset: pooled.Dataset, indices: np.ndarray, params: dict[str, float] | None = None):
    packet = starcoder_grp_packet(dataset)
    subset = dsp.PacketData(
        frame=packet.frame.iloc[indices].reset_index(drop=True),
        name_col=packet.name_col,
        y=packet.y[indices],
        w=packet.w[indices],
        m=packet.m,
        c0=packet.c0,
        c1=packet.c1,
        domain_names=list(packet.domain_names),
    )
    return starcoder_grp.fit_starcoder_grp(subset, params=params, seed=0)


def production_grp_params(l2: float) -> dict[str, float]:
    exponent = float(np.mean([GRP_SHAPE_PARAMS[f"a_{family}"] for family in ("broad_text", "tech_code", "reasoning")]))
    threshold = float(
        np.median([GRP_SHAPE_PARAMS[f"tau_{family}"] for family in ("broad_text", "tech_code", "reasoning")])
    )
    return {
        "exponent": exponent,
        "eta": float(GRP_SHAPE_PARAMS["eta"]),
        "retention_lambda": float(GRP_SHAPE_PARAMS["lam"]),
        "threshold": threshold,
        "l2": float(l2),
    }


def select_production_grp_l2(dataset: pooled.Dataset) -> tuple[float, list[dict[str, float | int | None]]]:
    rows = []
    for l2 in PRODUCTION_GRP_L2_GRID:
        params = production_grp_params(l2)
        prediction = np.full(dataset.n, np.nan, dtype=float)
        for train, test in folds(dataset, seed=0):
            model = UngroupedGRP(dataset, **params).fit(train)
            prediction[test] = model.predict(dataset.weights[test])
        rows.append({"l2": l2, "oofRmse": metric_summary(dataset.y, prediction)["rmse"]})
    selected = min(rows, key=lambda row: (float(row["oofRmse"]), row["l2"]))
    return float(selected["l2"]), rows


def grp_300m_params(l2: float, policy_class: str) -> dict[str, float]:
    params = legacy_exporter.grp_params(l2)
    if policy_class == SINGLE_PHASE:
        params = {**params, "eta": 1.0, "lam": 0.0}
    return params


def grp_300m_fit(dataset: pooled.Dataset, indices: np.ndarray, l2: float, policy_class: str):
    packet = legacy_exporter.grp_packet(dataset)
    return grp_calibration.build_penalty_calibration_surrogate(
        packet,
        params=grp_300m_params(l2, policy_class),
        variant_name=legacy_exporter.GRP_VARIANT,
    ).fit(dataset.weights[indices], dataset.y[indices])


def select_300m_grp_l2(dataset: pooled.Dataset, policy_class: str) -> tuple[float, list[dict[str, float]]]:
    rows: list[dict[str, float]] = []
    for l2 in legacy_exporter.GRP_L2_GRID:
        prediction = np.full(dataset.n, np.nan, dtype=float)
        for train, test in folds(dataset, seed=0):
            model = grp_300m_fit(dataset, train, l2, policy_class)
            prediction[test] = model.predict(dataset.weights[test])
        summary = metric_summary(dataset.y, prediction)
        rows.append(
            {
                "l2": float(l2),
                "oofRmse": float(summary["rmse"]),
                "oofSpearman": float(summary["spearman"]),
            }
        )
    selected = min(rows, key=lambda row: (row["oofRmse"], -row["oofSpearman"], row["l2"]))
    return float(selected["l2"]), rows


def single_starcoder_grp_params(dataset: pooled.Dataset) -> tuple[dict[str, float], list[dict[str, float]]]:
    packet = starcoder_grp_packet(dataset)
    rows: list[dict[str, float]] = []
    for alpha in STARCODER_SINGLE_GRP_ALPHA_GRID:
        for tau in STARCODER_SINGLE_GRP_TAU_GRID:
            for l2 in STARCODER_SINGLE_GRP_L2_GRID:
                params = {"alpha": alpha, "eta": 1.0, "lam": 0.0, "tau": tau, "reg": l2}
                prediction = np.full(dataset.n, np.nan, dtype=float)
                for train, test in folds(dataset, seed=0):
                    model = starcoder_grp.StarcoderGRPSurrogate(packet, params).fit(
                        dataset.weights[train], dataset.y[train]
                    )
                    prediction[test] = model.predict(dataset.weights[test])
                summary = metric_summary(dataset.y, prediction)
                rows.append(
                    {
                        "alpha": float(alpha),
                        "tau": float(tau),
                        "l2": float(l2),
                        "oofRmse": float(summary["rmse"]),
                        "oofSpearman": float(summary["spearman"]),
                    }
                )
    selected = min(rows, key=lambda row: (row["oofRmse"], -row["oofSpearman"], row["l2"]))
    return {
        "alpha": selected["alpha"],
        "eta": 1.0,
        "lam": 0.0,
        "tau": selected["tau"],
        "reg": selected["l2"],
    }, rows


def single_starcoder_grp_fit(
    dataset: pooled.Dataset,
    indices: np.ndarray,
    params: dict[str, float],
) -> starcoder_grp.StarcoderGRPSurrogate:
    packet = starcoder_grp_packet(dataset)
    return starcoder_grp.StarcoderGRPSurrogate(packet, params).fit(dataset.weights[indices], dataset.y[indices])


def fit_one_model(
    dataset: pooled.Dataset,
    model_id: str,
    policy_class: str,
    seeds: tuple[int, ...],
    *,
    legacy_model_summary: dict[str, Any] | None = None,
) -> tuple[Any, np.ndarray, np.ndarray, dict[str, Any]]:
    all_indices = np.arange(dataset.n)
    tuning: dict[str, Any] = {}
    seed_predictions: list[np.ndarray] | None = None
    if model_id == "linear":
        full_model = fit_linear_baseline(dataset, all_indices, policy_class)
        tuning = {
            "featureDimension": int(full_model.coefficients.size),
            "designRank": full_model.design_rank,
            "fit": "Centered minimum-norm ordinary least squares",
        }

        def fold_predict(train: np.ndarray, test: np.ndarray) -> np.ndarray:
            return fit_linear_baseline(dataset, train, policy_class).predict(dataset.weights[test])

        full_prediction = full_model.predict(dataset.weights)
    elif model_id == "olmix_loglinear":
        full_model = fit_olmix_loglinear_baseline(dataset, all_indices, policy_class)
        tuning = {
            "featureDimension": len(full_model.fit.coefficients),
            "huberDelta": olmix_loglinear.DEFAULT_HUBER_DELTA,
            "nStarts": olmix_loglinear.FIT_N_STARTS,
            "fullFitHuberLoss": full_model.fit.huber_loss,
            "fit": "Multistart L-BFGS-B on the OLMix positive log-linear law",
        }

        def fold_predict(train: np.ndarray, test: np.ndarray) -> np.ndarray:
            return fit_olmix_loglinear_baseline(dataset, train, policy_class).predict(dataset.weights[test])

        full_prediction = full_model.predict(dataset.weights)
    elif model_id in {"canonical", "effective_exposure", "effective_exposure_geometry"}:
        full_model = dsp_fit(dataset, all_indices, model_id, policy_class)

        def fold_predict(train: np.ndarray, test: np.ndarray) -> np.ndarray:
            return dsp_predict(dsp_fit(dataset, train, model_id, policy_class), dataset, dataset.weights[test])

        full_prediction = dsp_predict(full_model, dataset, dataset.weights)
    elif model_id == "separate_heads":
        l2, sweep = select_separate_l2(dataset, policy_class)
        tuning = {"l2": l2, "l2Sweep": sweep}
        full_model = separate_fit(dataset, all_indices, l2, policy_class)

        def fold_predict(train: np.ndarray, test: np.ndarray) -> np.ndarray:
            return separate_predict(
                separate_fit(dataset, train, l2, policy_class),
                dataset,
                dataset.weights[test],
                policy_class,
            )

        full_prediction = separate_predict(full_model, dataset, dataset.weights, policy_class)
    elif model_id == "compact_retained_state":
        l2, sweep = select_compact_l2(dataset, policy_class)
        config = compact_config(policy_class)
        tuning = {
            "l2": l2,
            "l2Sweep": sweep,
            "shapeProtocol": "Nonlinear retained-state shape refit inside every OOF fold.",
        }
        full_model = compact_fit(dataset, all_indices, l2, policy_class)

        def fold_predict(train: np.ndarray, test: np.ndarray) -> np.ndarray:
            return compact_fit(dataset, train, l2, policy_class).predict(dataset.weights[test])

        full_prediction = full_model.predict(dataset.weights)
        seed_predictions = [compact_oof_prediction(dataset, policy_class, l2, seed) for seed in seeds]
        tuning["shapeParameters"] = {
            "rate": full_model.shape.rate,
            "power": full_model.shape.power,
            "lateMultiplier": full_model.shape.late_multiplier,
            "forgettingRate": full_model.shape.forgetting_rate,
            "signal": config.signal.value,
        }
    elif model_id == "bucket_family_grp":
        shape, l2, sweep = select_bucket_hyperparameters(dataset, policy_class)
        tuning = {
            "l2": l2,
            "shapeSweep": sweep,
            "shapeParameters": {
                "exponent": shape.exponent,
                "lateMultiplier": shape.late_multiplier,
                "forgettingRate": shape.forgetting_rate,
                "penaltyThreshold": shape.penalty_threshold,
            },
            "shapeProtocol": "OOF-selected shared shape; every reported OOF fold refits only the nonnegative head.",
        }
        full_model = bucket_fit(dataset, all_indices, shape, l2)

        def fold_predict(train: np.ndarray, test: np.ndarray) -> np.ndarray:
            return bucket_fit(dataset, train, shape, l2).predict(dataset.weights[test])

        full_prediction = full_model.predict(dataset.weights)
    elif model_id == "hierarchical_phase_bucket_replay":
        config, sweep = select_hierarchical_phase_replay_config(dataset, policy_class)
        tuning = {
            **sweep,
            "l2": config.l2,
            "residualShrink": config.residual_shrink,
            "shapeParameters": {
                "exponent": config.shape.exponent,
                "lateMultiplier": config.shape.late_multiplier,
                "forgettingRate": config.shape.forgetting_rate,
                "penaltyThreshold": config.shape.penalty_threshold,
            },
            "shapeProtocol": (
                "Fit-panel CV first screens shared shapes with Bucket-resolved family GRP, then jointly selects "
                "ridge and shrinkage of bucket excesses for the hierarchical phase-replay form. Every reported "
                "OOF fold refits only the nonnegative linear head."
            ),
        }
        full_model = hierarchical_phase_replay_fit(dataset, all_indices, config)

        def fold_predict(train: np.ndarray, test: np.ndarray) -> np.ndarray:
            return hierarchical_phase_replay_fit(dataset, train, config).predict(dataset.weights[test])

        full_prediction = full_model.predict(dataset.weights)
    elif model_id == "bucket_family_power_separate_heads":
        shape, l2, sweep = select_power_heads_hyperparameters(dataset, policy_class)
        tuning = {
            "variant": power_heads_variant(policy_class).name,
            "l2": l2,
            "shapeSweep": sweep,
            "shapeParameters": {
                "exponent": shape.exponent,
                "lateMultiplier": shape.late_multiplier,
                "forgettingRate": shape.forgetting_rate,
                "penaltyThreshold": shape.penalty_threshold,
            },
            "shapeProtocol": (
                "OOF-selected shared retained-exposure shape; every reported OOF fold refits the nonnegative "
                "phase-specific bucket/family head. The one-phase ablation uses one aggregate head."
            ),
        }
        full_model = power_heads_fit(dataset, all_indices, shape, l2, policy_class)

        def fold_predict(train: np.ndarray, test: np.ndarray) -> np.ndarray:
            return power_heads_fit(dataset, train, shape, l2, policy_class).predict(dataset.weights[test])

        full_prediction = full_model.predict(dataset.weights)
    elif model_id == "bucket_family_power_separate_heads_family_onset":
        shape, l2, tau_shrink, sweep = select_power_family_onset_hyperparameters(dataset, policy_class)
        tuning = {
            "variant": power_family_onset_variant(policy_class).name,
            "l2": l2,
            "tauShrink": tau_shrink,
            **sweep,
            "shapeParameters": {
                "exponent": shape.exponent,
                "lateMultiplier": shape.late_multiplier,
                "forgettingRate": shape.forgetting_rate,
                "penaltyThreshold": shape.penalty_threshold,
            },
            "shapeProtocol": (
                "OOF-selected shared retained-exposure shape followed by OOF-selected shrinkage of family replay "
                "onsets toward the shared onset; every reported fold refits the nonnegative head and family onsets."
            ),
        }
        full_model = power_family_onset_fit(
            dataset,
            all_indices,
            shape,
            l2,
            tau_shrink,
            policy_class,
            multistart=True,
        )

        def fold_predict(train: np.ndarray, test: np.ndarray) -> np.ndarray:
            return power_family_onset_fit(
                dataset,
                train,
                shape,
                l2,
                tau_shrink,
                policy_class,
                multistart=True,
            ).predict(dataset.weights[test])

        full_prediction = full_model.predict(dataset.weights)
    elif model_id in RETAINED_GRP_MODEL_IDS:
        variant = RETAINED_GRP_VARIANTS[model_id]
        shape, l2, sweep = select_retained_grp_hyperparameters(dataset, model_id, policy_class)
        tuning = {
            "variant": variant.name,
            "l2": l2,
            "shapeSweep": sweep,
            "shapeParameters": {
                "rate": shape.rate,
                "power": shape.exponent,
                "lateMultiplier": shape.late_multiplier,
                "forgettingRate": shape.forgetting_rate,
                "penaltyThreshold": shape.penalty_threshold,
            },
            "shapeProtocol": (
                "OOF-selected shared Weibull/retention shape; every reported OOF fold refits only the "
                "nonnegative bucket/family head."
            ),
        }
        full_model = retained_grp_fit(dataset, all_indices, variant, shape, l2)

        def fold_predict(train: np.ndarray, test: np.ndarray) -> np.ndarray:
            return retained_grp_fit(dataset, train, variant, shape, l2).predict(dataset.weights[test])

        full_prediction = full_model.predict(dataset.weights)
    elif is_dolma39_dataset(dataset):
        if policy_class == TWO_PHASE and dataset.name.startswith("300m_"):
            if legacy_model_summary is None:
                raise ValueError("Two-phase 300M GRP requires the legacy selected-L2 summary")
            l2 = float(legacy_model_summary["l2"])
            sweep = legacy_model_summary.get("l2Sweep", [])
        else:
            l2, sweep = select_300m_grp_l2(dataset, policy_class)
        tuning = {"l2": l2, "l2Sweep": sweep}
        full_model = grp_300m_fit(dataset, all_indices, l2, policy_class)

        def fold_predict(train: np.ndarray, test: np.ndarray) -> np.ndarray:
            return grp_300m_fit(dataset, train, l2, policy_class).predict(dataset.weights[test])

        full_prediction = full_model.predict(dataset.weights)
    elif dataset.name.startswith("starcoder"):
        if policy_class == SINGLE_PHASE:
            params, sweep = single_starcoder_grp_params(dataset)
            full_model = single_starcoder_grp_fit(dataset, all_indices, params)
            tuning = {
                "shapeParameters": params,
                "shapeSweep": sweep,
                "oofShapeProtocol": "CV-selected one-phase shape; fold-refit linear head",
            }
        else:
            params, full_model = starcoder_grp_fit(dataset, all_indices)
            tuning = {"shapeParameters": params, "oofShapeProtocol": "Full-fit shape; fold-refit linear head"}

        def fold_predict(train: np.ndarray, test: np.ndarray) -> np.ndarray:
            if policy_class == SINGLE_PHASE:
                model = single_starcoder_grp_fit(dataset, train, params)
            else:
                _params, model = starcoder_grp_fit(dataset, train, params=params)
            return model.predict(dataset.weights[test])

        full_prediction = full_model.predict(dataset.weights)
    elif dataset.name == "production_uncheatable":
        if policy_class != TWO_PHASE:
            raise ValueError("Production has no observed one-phase panel")
        l2, sweep = select_production_grp_l2(dataset)
        params = production_grp_params(l2)
        tuning = {
            "l2": l2,
            "l2Sweep": sweep,
            "shapeParameters": params,
            "ablation": "No family or pair grouping; 300M GRP shape transferred and ridge retuned.",
        }
        full_model = UngroupedGRP(dataset, **params).fit(all_indices)

        def fold_predict(train: np.ndarray, test: np.ndarray) -> np.ndarray:
            return UngroupedGRP(dataset, **params).fit(train).predict(dataset.weights[test])

        full_prediction = full_model.predict(dataset.weights)
    else:
        raise ValueError(f"Unsupported model {model_id!r} for {dataset.name!r}")

    if seed_predictions is None:
        seed_predictions = []
        for seed in seeds:
            oof = np.full(dataset.n, np.nan, dtype=float)
            for train, test in folds(dataset, seed):
                oof[test] = fold_predict(train, test)
            if not np.isfinite(oof).all():
                raise ValueError(f"Incomplete OOF prediction for {dataset.name}/{model_id}/seed={seed}")
            seed_predictions.append(oof)
    prediction = np.mean(seed_predictions, axis=0)
    return full_model, prediction, full_prediction, tuning


def predict_model(
    model: Any,
    fit_dataset: pooled.Dataset,
    model_id: str,
    policy_class: str,
    weights: np.ndarray,
) -> np.ndarray:
    if model_id in {"canonical", "effective_exposure", "effective_exposure_geometry"}:
        return dsp_predict(model, fit_dataset, weights)
    if model_id == "separate_heads":
        return separate_predict(model, fit_dataset, weights, policy_class)
    return np.asarray(model.predict(weights), dtype=float)


def dsp_parameters(
    model: coverage_dsp.CoverageModel,
    dataset: pooled.Dataset,
    model_id: str,
    policy_class: str,
) -> list[dict[str, Any]]:
    base = model.base
    records = [
        parameter(
            "intercept",
            "b_0",
            base.intercept,
            "Loss level after centering all response features.",
            unit="BPB",
        )
    ]
    for key, value in base.params.items():
        if isinstance(value, np.ndarray):
            continue
        if key == "gamma":
            role = (
                "Relative phase-1 premium on the benefit term."
                if model_id == "canonical"
                else "Phase-1 epoch value relative to a phase-0 epoch in effective exposure."
            )
        else:
            role = "Global nonlinear phase-response parameter."
        records.append(parameter(key, key.replace("_", " "), float(value), role))

    rho = np.asarray(base.params["rho"], dtype=float)
    tau = np.asarray(base.params.get("tau", np.zeros(dataset.m)), dtype=float)
    for index, domain in enumerate(dataset.domain_names):
        records.extend(
            [
                parameter(
                    f"rho:{domain}",
                    "rho",
                    rho[index],
                    "Saturation rate of useful exposure; larger values saturate sooner.",
                    scope="domain",
                    domain_id=domain,
                    transformed_value=float(np.log(2.0) / max(rho[index], 1e-12)),
                    transformed_label="Half-saturation exposure",
                    unit="effective epochs",
                ),
                parameter(
                    f"tau:{domain}",
                    "tau",
                    tau[index],
                    "Log-exposure threshold where the soft overexposure penalty turns on.",
                    scope="domain",
                    domain_id=domain,
                    transformed_value=float(np.expm1(tau[index])),
                    transformed_label="Penalty-onset exposure",
                    unit="effective epochs",
                ),
                parameter(
                    f"benefit:{domain}",
                    "a",
                    base.benefit_coef[index],
                    "Maximum fitted BPB reduction supplied by this bucket's saturation feature.",
                    scope="domain",
                    domain_id=domain,
                    unit="BPB",
                ),
                parameter(
                    f"penalty:{domain}",
                    "p",
                    base.penalty_coef[index],
                    "Strength of this bucket's overexposure penalty.",
                    scope="domain",
                    domain_id=domain,
                    unit="BPB",
                ),
            ]
        )
    if model_id == "effective_exposure_geometry":
        geometry_terms = (
            (
                "geometry:phase_tv",
                "theta_TV",
                model.coverage_coef[0],
                "Global cost assigned to total-variation distance between phase mixtures.",
            ),
            (
                "geometry:aggregate_hhi",
                "theta_agg",
                model.coverage_coef[1],
                "Global cost assigned to concentration of aggregate exposure.",
            ),
            (
                "geometry:phase1_hhi",
                "theta_1",
                model.coverage_coef[2],
                "Global cost assigned to concentration in the late phase.",
            ),
        )
        if policy_class == SINGLE_PHASE:
            geometry_terms = (geometry_terms[1],)
        for key, symbol, value, role in geometry_terms:
            records.append(parameter(key, symbol, value, role, unit="BPB"))
    return records


def separate_parameters(
    model: Any,
    dataset: pooled.Dataset,
    policy_class: str,
) -> list[dict[str, Any]]:
    m = dataset.m
    if policy_class == SINGLE_PHASE:
        coef = np.asarray(model.coefs[0], dtype=float)
        records = [
            parameter(
                "intercept",
                "b_0",
                model.intercept,
                "Loss level after centering the aggregate-exposure features.",
                unit="BPB",
            ),
            parameter("l2", "lambda_L2", model.l2, "Ridge shrinkage applied to the aggregate bowl coefficients."),
        ]
        blocks = (
            ("aggregate_under", "a-", coef[:m], "Aggregate-exposure underexposure curvature."),
            ("aggregate_over", "a+", coef[m:], "Aggregate-exposure overexposure curvature."),
        )
        for index, domain in enumerate(dataset.domain_names):
            records.append(
                parameter(
                    f"mu:{domain}",
                    "mu",
                    model.mus[0][index],
                    "Center of the aggregate-exposure asymmetric bowl.",
                    scope="domain",
                    domain_id=domain,
                    transformed_value=float(np.expm1(model.mus[0][index])),
                    transformed_label="Preferred total exposure",
                    unit="epochs",
                )
            )
            for key, symbol, values, role in blocks:
                records.append(
                    parameter(
                        f"{key}:{domain}",
                        symbol,
                        values[index],
                        role,
                        scope="domain",
                        domain_id=domain,
                        unit="BPB / log-epoch squared",
                    )
                )
        return records

    coef = np.asarray(model.coefficients, dtype=float)
    records = [
        parameter(
            "intercept",
            "b_0",
            model.intercept,
            "Loss level after centering the phase-head features.",
            unit="BPB",
        ),
        parameter("l2", "lambda_L2", model.l2, "Ridge shrinkage applied to all phase-head coefficients."),
    ]
    blocks = (
        ("phase0_under", "a0-", coef[:m], "Phase-0 underexposure curvature."),
        ("phase0_over", "a0+", coef[m : 2 * m], "Phase-0 overexposure curvature."),
        ("phase1_under", "a1-", coef[2 * m : 3 * m], "Phase-1 underexposure curvature."),
        ("phase1_over", "a1+", coef[3 * m :], "Phase-1 overexposure curvature."),
    )
    for index, domain in enumerate(dataset.domain_names):
        for phase, mu in ((0, model.mu0[index]), (1, model.mu1[index])):
            records.append(
                parameter(
                    f"mu{phase}:{domain}",
                    f"mu_{phase}",
                    mu,
                    f"Center of the phase-{phase} asymmetric exposure bowl.",
                    scope="domain",
                    domain_id=domain,
                    transformed_value=float(np.expm1(mu)),
                    transformed_label="Preferred exposure",
                    unit="epochs",
                )
            )
        for key, symbol, values, role in blocks:
            records.append(
                parameter(
                    f"{key}:{domain}",
                    symbol,
                    values[index],
                    role,
                    scope="domain",
                    domain_id=domain,
                    unit="BPB / log-epoch squared",
                )
            )
    return records


def grp_300m_parameters(model: Any, dataset: pooled.Dataset, l2: float) -> list[dict[str, Any]]:
    packet = model.packet
    records = [
        parameter("intercept", "b_0", model.intercept_, "Loss level after centering all GRP features.", unit="BPB"),
    ]
    for key, value in model.params.items():
        transformed = None
        transformed_label = None
        role = "GRP nonlinear shape parameter."
        if key.startswith("a_"):
            role = "Power-law response exponent; smaller values imply faster diminishing returns."
        elif key.startswith("tau_"):
            transformed = float(np.expm1(value))
            transformed_label = "Penalty-onset exposure"
            role = "Log-exposure threshold where the family overexposure penalty turns on."
        elif key == "eta":
            role = "Phase-1 epoch value relative to one retained phase-0 epoch."
        elif key == "lam":
            role = "Forgetting rate applied to phase-0 exposure as phase-1 mass moves away from a bucket."
        elif key == "beta":
            role = "Discount applied to low-quality Common Crawl within a paired topic."
        elif key == "reg":
            role = "Ridge shrinkage applied to all linear GRP coefficients."
        records.append(
            parameter(
                key,
                key.replace("_", " "),
                float(value),
                role,
                transformed_value=transformed,
                transformed_label=transformed_label,
                unit="effective epochs" if transformed is not None else None,
            )
        )
    parts = model.components()
    for coefficient, domain_index in zip(parts["singleton_coef"], packet.singletons, strict=True):
        domain = dataset.domain_names[domain_index]
        records.append(
            parameter(
                f"signal:{domain}",
                "beta_signal",
                coefficient,
                "BPB reduction coefficient on this singleton bucket's power-law signal.",
                scope="domain",
                domain_id=domain,
                unit="BPB",
            )
        )
    for coefficient, (high, low), topic in zip(parts["pair_coef"], packet.pairs, packet.pair_topics, strict=True):
        records.append(
            parameter(
                f"pair:{topic}",
                "beta_pair",
                coefficient,
                "Joint signal coefficient for the high/low-quality Common Crawl topic pair.",
                scope="group",
                group_label=f"CC pair · {topic}",
                unit="BPB",
            )
        )
        for domain_index in (high, low):
            records.append(
                parameter(
                    f"pair-member:{topic}:{dataset.domain_names[domain_index]}",
                    "beta_pair",
                    coefficient,
                    "Shared pair coefficient; shown on both member buckets for inspection.",
                    scope="domain",
                    domain_id=dataset.domain_names[domain_index],
                    group_label=f"CC pair · {topic}",
                    unit="BPB",
                )
            )
    for family, coefficient in parts["family_coef"].items():
        records.append(
            parameter(
                f"family-signal:{family}",
                "beta_family",
                coefficient,
                "Signal coefficient on total retained exposure within this family.",
                scope="group",
                group_label=family.replace("_", " ").title(),
                unit="BPB",
            )
        )
    for family, coefficient in parts["family_group_penalty_coef"].items():
        records.append(
            parameter(
                f"family-penalty:{family}",
                "beta_penalty",
                coefficient,
                "Strength of the summed within-family overexposure penalty.",
                scope="group",
                group_label=family.replace("_", " ").title(),
                unit="BPB",
            )
        )
    if not math.isclose(float(model.params["reg"]), l2):
        raise ValueError("GRP model and selected ridge differ")
    return records


def starcoder_grp_parameters(model: Any, params: dict[str, float], dataset: pooled.Dataset) -> list[dict[str, Any]]:
    if model.intercept_ is None or model.coef_ is None:
        raise RuntimeError("StarCoder GRP is not fit")
    records = [parameter("intercept", "b_0", model.intercept_, "Loss level after centering GRP features.", unit="BPB")]
    roles = {
        "alpha": "Scale inside the log-satiation signal for both corpora.",
        "eta": "Phase-1 epoch value relative to one retained phase-0 epoch.",
        "lam": "Forgetting rate applied to phase-0 exposure.",
        "tau": "Log-exposure threshold for the aggregate overexposure penalty.",
        "reg": "Ridge shrinkage applied to the three linear coefficients.",
    }
    for key, value in params.items():
        records.append(
            parameter(
                key,
                key,
                value,
                roles[key],
                transformed_value=float(np.expm1(value)) if key == "tau" else None,
                transformed_label="Penalty-onset exposure" if key == "tau" else None,
                unit="effective epochs" if key == "tau" else None,
            )
        )
    for index, domain in enumerate(dataset.domain_names):
        records.append(
            parameter(
                f"signal:{domain}",
                "beta_signal",
                model.coef_[index],
                "BPB reduction coefficient on this corpus's retained-exposure signal.",
                scope="domain",
                domain_id=domain,
                unit="BPB",
            )
        )
    records.append(
        parameter(
            "penalty",
            "beta_penalty",
            model.coef_[2],
            "Strength of the summed two-corpus overexposure penalty.",
            unit="BPB",
        )
    )
    return records


def production_grp_parameters(model: UngroupedGRP, dataset: pooled.Dataset) -> list[dict[str, Any]]:
    if model.intercept is None or model.signal_coef is None or model.penalty_coef is None:
        raise RuntimeError("Production GRP ablation is not fit")
    records = [
        parameter("intercept", "b_0", model.intercept, "Loss level after centering ungrouped GRP features.", unit="BPB"),
        parameter(
            "a",
            "a",
            model.exponent,
            "Shared power-law exponent; semantic family-specific exponents are ablated.",
        ),
        parameter("eta", "eta", model.eta, "Phase-1 epoch value relative to one retained phase-0 epoch."),
        parameter("lambda", "lambda", model.retention_lambda, "Shared phase-0 forgetting rate."),
        parameter(
            "tau",
            "tau",
            model.threshold,
            "Shared log-exposure penalty threshold; semantic family thresholds are ablated.",
            transformed_value=float(np.expm1(model.threshold)),
            transformed_label="Penalty-onset exposure",
            unit="effective epochs",
        ),
        parameter("l2", "lambda_L2", model.l2, "Ridge shrinkage selected by production-swarm OOF RMSE."),
    ]
    for index, domain in enumerate(dataset.domain_names):
        records.extend(
            [
                parameter(
                    f"signal:{domain}",
                    "beta_signal",
                    model.signal_coef[index],
                    "BPB reduction coefficient for this ungrouped bucket.",
                    scope="domain",
                    domain_id=domain,
                    unit="BPB",
                ),
                parameter(
                    f"penalty:{domain}",
                    "beta_penalty",
                    model.penalty_coef[index],
                    "Overexposure-penalty coefficient for this ungrouped bucket.",
                    scope="domain",
                    domain_id=domain,
                    unit="BPB",
                ),
            ]
        )
    return records


def compact_parameters(
    model: compact_retained.FittedModel,
    dataset: pooled.Dataset,
    l2: float,
) -> list[dict[str, Any]]:
    half_saturation = float(math.log(2.0) ** (1.0 / model.shape.power) / model.shape.rate)
    records = [
        parameter(
            "intercept", "b_0", model.intercept, "Loss level after centering retained-response features.", unit="BPB"
        ),
        parameter(
            "rho",
            "rho",
            model.shape.rate,
            "Shared rate of the Weibull learning curve.",
            transformed_value=half_saturation,
            transformed_label="Half-saturation exposure",
            unit="retained epochs",
        ),
        parameter(
            "power",
            "p",
            model.shape.power,
            "Shared Weibull shape; values below one represent mixed learning timescales.",
        ),
        parameter(
            "eta", "eta", model.shape.late_multiplier, "Phase-1 epoch value relative to one retained phase-0 epoch."
        ),
        parameter(
            "lambda",
            "lambda",
            model.shape.forgetting_rate,
            "Loss rate for phase-0 learning when a bucket is not revisited late.",
        ),
        parameter("l2", "lambda_L2", l2, "Ridge shrinkage selected by OOF RMSE."),
    ]
    for index, domain in enumerate(dataset.domain_names):
        records.append(
            parameter(
                f"signal:{domain}",
                "a_i",
                model.signal_coef[index],
                "Maximum BPB reduction supplied by this bucket's retained-learning curve.",
                scope="domain",
                domain_id=domain,
                unit="BPB",
            )
        )
    records.append(
        parameter(
            "shared_replay",
            "c",
            model.replay_coef[0],
            "Global harm per squared epoch replayed beyond the first pass, using literal total exposure.",
            unit="BPB / epoch squared",
        )
    )
    return records


def family_label(name: str) -> str:
    return name.upper() if name.startswith("c") and name[1:].isdigit() else name.replace("_", " ").title()


def bucket_family_parameters(model: BucketFamilyGRP) -> list[dict[str, Any]]:
    shape = model.shape
    records = [
        parameter(
            "intercept", "b_0", model.head.intercept, "Loss level after centering family-GRP features.", unit="BPB"
        ),
        parameter("a", "a", shape.exponent, "Shared diminishing-returns exponent for bucket and family coverage."),
        parameter("eta", "eta", shape.late_multiplier, "Phase-1 epoch value relative to one retained phase-0 epoch."),
        parameter("lambda", "lambda", shape.forgetting_rate, "Shared phase-0 forgetting rate."),
        parameter(
            "tau",
            "tau",
            shape.penalty_threshold,
            "Shared log-exposure onset for family replay harm.",
            transformed_value=float(np.expm1(shape.penalty_threshold)),
            transformed_label="Family penalty-onset exposure",
            unit="retained epochs",
        ),
        parameter("l2", "lambda_L2", model.l2, "Ridge shrinkage jointly selected with the shared response shape."),
    ]
    for name, coefficient in zip(model.head.feature_names, model.head.coefficients, strict=True):
        kind, identity = name.split(":", maxsplit=1)
        if kind == "bucket_signal":
            records.append(
                parameter(
                    name,
                    "a_i",
                    coefficient,
                    "Bucket-specific BPB reduction; Q tiers are not ordered or forced to be monotone.",
                    scope="domain",
                    domain_id=identity,
                    unit="BPB",
                )
            )
            continue
        label = family_label(identity)
        if kind == "family_signal":
            role = "Additional BPB reduction from nonlinear coverage of this family as a whole."
            symbol = "A_C"
        elif kind == "family_penalty":
            role = "Replay-harm coefficient on this family's total retained exposure."
            symbol = "B_C"
        else:
            raise ValueError(f"Unknown bucket-family feature {name!r}")
        records.append(
            parameter(
                name,
                symbol,
                coefficient,
                role,
                scope="group",
                group_label=label,
                unit="BPB",
            )
        )
    return records


def hierarchical_phase_replay_parameters(model: hierarchical_grp.Model) -> list[dict[str, Any]]:
    config = model.config
    shape = config.shape
    records = [
        parameter(
            "intercept",
            "b_0",
            model.intercept,
            "Loss level after centering hierarchical family-response features.",
            unit="BPB",
        ),
        parameter("a", "a", shape.exponent, "Shared diminishing-returns exponent for bucket and family coverage."),
        parameter("eta", "eta", shape.late_multiplier, "Phase-1 epoch value relative to one retained phase-0 epoch."),
        parameter("lambda", "lambda", shape.forgetting_rate, "Shared phase-0 forgetting rate."),
        parameter(
            "tau",
            "tau",
            shape.penalty_threshold,
            "Shared log-exposure onset for aggregate-family and member-level replay harm.",
            transformed_value=float(np.expm1(shape.penalty_threshold)),
            transformed_label="Replay penalty-onset exposure",
            unit="retained epochs",
        ),
        parameter("l2", "lambda_L2", config.l2, "Ridge shrinkage jointly selected with the shared response shape."),
        parameter(
            "residual_shrink",
            "kappa_res",
            config.residual_shrink,
            "Additional ridge multiplier on bucket-specific excess utility above its pooled family base.",
        ),
    ]
    feature_names = hierarchical_grp.build_design(model.dataset, config).names
    for name, coefficient in zip(feature_names, model.coefficients, strict=True):
        if name == "phase_shift_tv":
            records.append(
                parameter(
                    name,
                    "theta_TV",
                    coefficient,
                    "Global BPB cost of changing the mixture between phases.",
                    unit="BPB / TV",
                )
            )
            continue
        kind, identity = name.split(":", maxsplit=1)
        if kind == "singleton_signal":
            records.append(
                parameter(
                    name,
                    "a_i",
                    coefficient,
                    "BPB reduction from this singleton bucket's retained-exposure response.",
                    scope="domain",
                    domain_id=identity,
                    unit="BPB",
                )
            )
            continue
        if kind == "bucket_excess_signal":
            records.append(
                parameter(
                    name,
                    "delta_i",
                    coefficient,
                    "Nonnegative bucket utility in excess of the shared family base.",
                    scope="domain",
                    domain_id=identity,
                    unit="BPB",
                )
            )
            continue
        label = family_label(identity)
        group_fields = {
            "pooled_base_signal": (
                "a_C",
                "Shared BPB reduction per member-response unit for this family.",
            ),
            "family_coverage_signal": (
                "A_C",
                "Additional BPB reduction from saturating coverage of this family as a whole.",
            ),
            "family_overexposure": (
                "B_C",
                "Replay-harm coefficient on this family's aggregate retained exposure.",
            ),
            "family_member_replay": (
                "R_C",
                "Replay-harm coefficient on the mean member-level overexposure within this family.",
            ),
        }
        if kind not in group_fields:
            raise ValueError(f"Unknown hierarchical phase-replay feature {name!r}")
        symbol, role = group_fields[kind]
        records.append(
            parameter(
                name,
                symbol,
                coefficient,
                role,
                scope="group",
                group_label=label,
                unit="BPB",
            )
        )
    return records


def power_heads_parameters(
    model: PowerSeparateHeadsGRP | PowerSeparateHeadsFamilyOnsetGRP,
    policy_class: str,
) -> list[dict[str, Any]]:
    shape = model.shape
    records = [
        parameter(
            "intercept", "b_0", model.head.intercept, "Loss level after centering power-head features.", unit="BPB"
        ),
        parameter("a", "a", shape.exponent, "Shared diminishing-returns exponent for bucket and family coverage."),
        parameter("lambda", "lambda", shape.forgetting_rate, "Shared phase-0 forgetting rate."),
        parameter(
            "tau",
            "tau",
            shape.penalty_threshold,
            "Shared log-exposure onset for family replay harm.",
            transformed_value=float(np.expm1(shape.penalty_threshold)),
            transformed_label="Family penalty-onset exposure",
            unit="retained epochs",
        ),
        parameter("l2", "lambda_L2", model.l2, "Ridge shrinkage jointly selected with the shared response shape."),
    ]
    for name, coefficient in zip(model.head.feature_names, model.head.coefficients, strict=True):
        parts = name.split(":")
        if parts[0].startswith("phase"):
            phase = parts[0].removeprefix("phase")
            head_label = "aggregate" if policy_class == SINGLE_PHASE else f"phase-{phase}"
            head_symbol = "" if policy_class == SINGLE_PHASE else f"^{{({phase})}}"
            kind = parts[1]
            identity = ":".join(parts[2:])
            if kind == "bucket_signal":
                records.append(
                    parameter(
                        name,
                        f"a_i{head_symbol}",
                        coefficient,
                        f"{head_label.title()} BPB reduction from this bucket's power response.",
                        scope="domain",
                        domain_id=identity,
                        unit="BPB",
                    )
                )
                continue
            if kind == "family_signal":
                records.append(
                    parameter(
                        name,
                        f"A_C{head_symbol}",
                        coefficient,
                        f"{head_label.title()} nonlinear coverage benefit for this family.",
                        scope="group",
                        group_label=family_label(identity),
                        unit="BPB",
                    )
                )
                continue
        if parts[0] == "family_penalty":
            identity = ":".join(parts[1:])
            records.append(
                parameter(
                    name,
                    "B_C",
                    coefficient,
                    "Replay-harm coefficient on this family's total retained exposure.",
                    scope="group",
                    group_label=family_label(identity),
                    unit="BPB",
                )
            )
            continue
        raise ValueError(f"Unknown power-head feature {name!r}")
    return records


def power_heads_family_onset_parameters(
    model: PowerSeparateHeadsFamilyOnsetGRP,
    policy_class: str,
) -> list[dict[str, Any]]:
    records = [record for record in power_heads_parameters(model, policy_class) if record["key"] != "tau"]
    records.append(
        parameter(
            "tau_shrink",
            "lambda_tau",
            model.tau_shrink,
            "Cross-validated shrinkage of family onsets toward the shared-onset control.",
        )
    )
    for family, tau in zip(model.dataset.family_names, model.family_tau, strict=True):
        records.append(
            parameter(
                f"tau:{family}",
                "tau_C",
                tau,
                "Family-specific log-exposure onset for replay harm.",
                scope="group",
                group_label=family_label(family),
                transformed_value=float(np.expm1(tau)),
                transformed_label="Family penalty-onset exposure",
                unit="retained epochs",
            )
        )
    return records


def retained_family_parameters(model: RetainedFamilyGRP) -> list[dict[str, Any]]:
    shape = model.shape
    half_saturation = float(math.log(2.0) ** (1.0 / shape.exponent) / shape.rate)
    records = [
        parameter(
            "intercept", "b_0", model.head.intercept, "Loss level after centering retained-family features.", unit="BPB"
        ),
        parameter(
            "rho",
            "rho",
            shape.rate,
            "Shared rate of the bucket and family Weibull learning curves.",
            transformed_value=half_saturation,
            transformed_label="Half-saturation exposure",
            unit="retained epochs",
        ),
        parameter(
            "power",
            "p",
            shape.exponent,
            "Shared Weibull shape; values below one represent a mixture of learning timescales.",
        ),
        parameter("eta", "eta", shape.late_multiplier, "Phase-1 epoch value relative to one retained phase-0 epoch."),
        parameter("lambda", "lambda", shape.forgetting_rate, "Shared phase-0 forgetting rate."),
        parameter("l2", "lambda_L2", model.l2, "Ridge shrinkage jointly selected with the shared response shape."),
    ]
    if model.variant.replay is retained_grp.ReplayKind.FAMILY_AGGREGATE_GLOBAL_TAU:
        records.insert(
            -1,
            parameter(
                "tau",
                "tau",
                shape.penalty_threshold,
                "Shared log-exposure onset for aggregate family replay harm.",
                transformed_value=float(np.expm1(shape.penalty_threshold)),
                transformed_label="Family penalty-onset exposure",
                unit="retained epochs",
            ),
        )
    for name, coefficient in zip(model.head.feature_names, model.head.coefficients, strict=True):
        kind, identity = name.split(":", maxsplit=1)
        if kind == "bucket_signal":
            records.append(
                parameter(
                    name,
                    "a_i",
                    coefficient,
                    "Maximum BPB reduction from this bucket's shared-shape Weibull learning curve.",
                    scope="domain",
                    domain_id=identity,
                    unit="BPB",
                )
            )
            continue
        label = family_label(identity)
        if kind == "family_signal":
            symbol = "A_C"
            role = "Additional BPB reduction from Weibull coverage of the family's mean retained exposure."
        elif kind == "family_penalty":
            symbol = "B_C"
            role = "Replay-harm coefficient on this family's aggregate retained exposure beyond the shared onset."
        elif kind == "family_literal_replay":
            symbol = "B_C"
            role = "Replay-harm coefficient on literal per-bucket epochs beyond one pass, summed within this family."
        else:
            raise ValueError(f"Unknown retained-family feature {name!r}")
        records.append(
            parameter(
                name,
                symbol,
                coefficient,
                role,
                scope="group",
                group_label=label,
                unit="BPB",
            )
        )
    return records


def policy_coefficient_parameters(
    coefficients: np.ndarray,
    dataset: pooled.Dataset,
    policy_class: str,
    *,
    role: str,
) -> list[dict[str, Any]]:
    coefficients = np.asarray(coefficients, dtype=float)
    if policy_class == SINGLE_PHASE:
        blocks = (("aggregate", "agg", coefficients),)
    else:
        reshaped = coefficients.reshape(2, dataset.m)
        blocks = (("phase 0", "0", reshaped[0]), ("phase 1", "1", reshaped[1]))
    records: list[dict[str, Any]] = []
    for phase_label, phase_key, block in blocks:
        for domain, coefficient in zip(dataset.domain_names, block, strict=True):
            records.append(
                parameter(
                    f"beta:{phase_key}:{domain}",
                    f"beta^({phase_key})" if phase_key != "agg" else "beta^(agg)",
                    coefficient,
                    f"{phase_label.title()} {role}",
                    scope="domain",
                    domain_id=domain,
                )
            )
    return records


def linear_parameters(model: LinearBaseline) -> list[dict[str, Any]]:
    return [
        parameter(
            "intercept",
            "b_0",
            model.intercept,
            "Affine loss level at the origin of the policy-weight coordinate system.",
            unit="BPB",
        ),
        *policy_coefficient_parameters(
            model.coefficients,
            model.dataset,
            model.policy_class,
            role="slope in predicted BPB per unit mixture weight; only coefficient contrasts are identifiable.",
        ),
    ]


def olmix_loglinear_parameters(model: OlmixLoglinearBaseline) -> list[dict[str, Any]]:
    return [
        parameter(
            "c",
            "c",
            float(np.exp(model.fit.log_c)),
            "Positive additive loss floor in the OLMix response law.",
            unit="BPB",
        ),
        *policy_coefficient_parameters(
            np.asarray(model.fit.coefficients, dtype=float),
            model.dataset,
            model.policy_class,
            role="coefficient inside the exponential response; only coefficient contrasts are identifiable.",
        ),
    ]


def parameter_records(
    model: Any,
    dataset: pooled.Dataset,
    model_id: str,
    tuning: dict[str, Any],
    policy_class: str,
) -> list[dict[str, Any]]:
    if model_id == "linear":
        return linear_parameters(model)
    if model_id == "olmix_loglinear":
        return olmix_loglinear_parameters(model)
    if model_id in {"canonical", "effective_exposure", "effective_exposure_geometry"}:
        return dsp_parameters(model, dataset, model_id, policy_class)
    if model_id == "separate_heads":
        return separate_parameters(model, dataset, policy_class)
    if model_id == "compact_retained_state":
        return compact_parameters(model, dataset, float(tuning["l2"]))
    if model_id == "bucket_family_grp":
        return bucket_family_parameters(model)
    if model_id == "hierarchical_phase_bucket_replay":
        return hierarchical_phase_replay_parameters(model)
    if model_id == "bucket_family_power_separate_heads":
        return power_heads_parameters(model, policy_class)
    if model_id == "bucket_family_power_separate_heads_family_onset":
        return power_heads_family_onset_parameters(model, policy_class)
    if model_id in RETAINED_GRP_MODEL_IDS:
        return retained_family_parameters(model)
    if is_dolma39_dataset(dataset):
        return grp_300m_parameters(model, dataset, float(tuning["l2"]))
    if dataset.name.startswith("starcoder"):
        return starcoder_grp_parameters(model, tuning["shapeParameters"], dataset)
    if dataset.name == "production_uncheatable":
        return production_grp_parameters(model, dataset)
    raise ValueError(f"Unsupported parameter extraction for {dataset.name}/{model_id}")


def model_caveats(dataset: pooled.Dataset, model_id: str, policy_class: str) -> list[str]:
    caveats: list[str] = []
    if model_id == "linear":
        caveats.append(
            "Affine baseline: it cannot represent saturation, replay harm, or an interior U-shaped optimum. "
            "Because policy weights lie on a simplex, displayed slopes use the minimum-norm coefficient "
            "representation and only slope contrasts are identifiable."
        )
    if model_id == "olmix_loglinear":
        caveats.append(
            "OLMix log-linear baseline: one positive exponential ridge is fit with Huber delta 0.02 and 48 "
            "deterministic multistarts. The OLMix KL proposal penalty is a deployment objective and is not part "
            "of these surrogate predictions."
        )
    if model_id in BASELINE_MODEL_IDS:
        coordinate = (
            "phase-fraction-weighted aggregate weights"
            if policy_class == SINGLE_PHASE
            else "the flattened early- and late-phase weights"
        )
        caveats.append(f"The {policy_class.replace('_', '-')} fit uses {coordinate} as its policy coordinate.")
    if model_id == "compact_retained_state":
        caveats.append(
            "Compact ungrouped form: bucket benefit amplitudes are independent, while retention, Weibull shape, "
            "and replay harm are shared globally. The 3e18 validation panel did not reproduce its locally predicted "
            "phase advantage, so OOF fit quality is not deployment evidence."
        )
    if model_id == "bucket_family_grp":
        caveats.append(
            "Bucket response amplitudes are unconstrained within each family; family labels provide pooling channels "
            "but do not assert that Q tiers are monotone in quality."
        )
        if dataset.name.startswith("starcoder"):
            caveats.append(
                "Each StarCoder corpus is a singleton family, so the family-coverage channel vanishes and the model "
                "reduces to bucket power responses plus one replay penalty per corpus."
            )
    if model_id == "hierarchical_phase_bucket_replay":
        caveats.append(
            "Family pooling is soft rather than a quality-order constraint: each multi-member family shares a base "
            "utility, while CV-selected ridge shrinkage controls nonnegative bucket-specific excesses."
        )
        caveats.append(
            "The historical 3e18 archive was used only to evaluate transfer after the form and hyperparameters were "
            "selected on the fit panel; it is not part of this fit or its tuning loss."
        )
        if policy_class == TWO_PHASE:
            caveats.append(
                "One learned nonnegative phase-TV coefficient captures a global cost of schedule change; it cannot "
                "represent domain-specific phase-order interactions."
            )
        if dataset.name.startswith("starcoder"):
            caveats.append(
                "Both StarCoder corpora are singleton families, so hierarchical pooling and nonlinear family-coverage "
                "channels vanish; this view is a deliberate structural ablation of the full model."
            )
    if model_id == "bucket_family_power_separate_heads":
        caveats.append(
            "This changes only the response head relative to Bucket-resolved family GRP: early and late bucket/family "
            "benefits have independent nonnegative amplitudes, while retention, power curvature, and replay harm "
            "stay shared."
        )
        if policy_class == SINGLE_PHASE:
            caveats.append(
                "The policy-matched ablation collapses the phase-specific response to one aggregate-exposure head and "
                "retunes the shared response shape and ridge."
            )
    if model_id == "bucket_family_power_separate_heads_family_onset":
        caveats.append(
            "This is a nested extension of Power + separate heads: each family learns its own replay onset, with "
            "cross-validated shrinkage toward the shared onset. The extension improved 300M Table-9 nested OOF "
            "RMSE by 2.34% in 5/5 folds, but was neutral on 300M and production Uncheatable; it is a diagnostic "
            "variant, not a universal paper-model replacement."
        )
        if policy_class == SINGLE_PHASE:
            caveats.append(
                "The policy-matched ablation uses one aggregate response head and refits both shared shape terms "
                "and family replay onsets on the tied-policy panel."
            )
    if model_id in RETAINED_GRP_MODEL_IDS:
        caveats.append(
            "The bucket and family amplitudes remain nonnegative and unconstrained across quality tiers; semantic "
            "families provide coverage and replay channels, not an assumed monotone quality ordering."
        )
        if model_id == "bucket_family_weibull_shared_onset":
            caveats.append(
                "This form won the nested 300M benchmark, but the production winner used literal family replay; it "
                "is exposed here as a mechanism to test, not as a universal cross-swarm winner."
            )
        else:
            caveats.append(
                "This form won the nested production-swarm benchmark but transferred poorly to 300M, where the "
                "shared-onset Weibull form won."
            )
        if dataset.name.startswith("starcoder"):
            caveats.append(
                "Each StarCoder corpus is a singleton family, so there is no separate family-coverage feature."
            )
    if dataset.name == "production_uncheatable" and model_id == "grp":
        caveats.append(
            "Ungrouped GRP ablation: the production partition has no a priori family or pair grouping. "
            "The 300M GRP response shape is transferred, family features are removed, and ridge is retuned."
        )
    if dataset.name.startswith("starcoder") and model_id == "grp":
        caveats.append("Two-family StarCoder GRP; broad and code each supply one retained-exposure signal.")
    if model_id == "effective_exposure_geometry":
        if policy_class == SINGLE_PHASE:
            caveats.append(
                "The one-phase geometry ablation retains only aggregate concentration; phase TV is zero and late-phase "
                "concentration is algebraically redundant."
            )
        else:
            caveats.append("The three global geometry coefficients are nonnegative and fit jointly with the DSP head.")
    if policy_class == SINGLE_PHASE and model_id in {"canonical", "effective_exposure"}:
        caveats.append(
            "Canonical and effective-exposure DSP reduce to the same total-exposure model when the phase degree of "
            "freedom is removed."
        )
    if policy_class == SINGLE_PHASE and model_id == "grp":
        caveats.append(
            "The policy ablation fixes late-epoch value to 1 and forgetting to 0, then retunes ridge/shape terms."
        )
    if policy_class == SINGLE_PHASE and model_id in NEW_MODEL_IDS and model_id not in BASELINE_MODEL_IDS:
        caveats.append(
            "The policy-matched ablation fixes late-epoch value to 1 and forgetting to 0, then retunes all remaining "
            "response-shape and ridge hyperparameters."
        )
    if policy_class == SINGLE_PHASE and dataset.name.startswith("starcoder") and dataset.n < 10:
        caveats.append(
            f"Only {dataset.n} tied-policy observations are available; this fit is under-identified and should be read "
            "as an exploratory interpolation."
        )
    return caveats


def fit_detail(
    dataset: pooled.Dataset,
    model_id: str,
    model: Any,
    oof_prediction: np.ndarray,
    full_prediction: np.ndarray,
    tuning: dict[str, Any],
    policy_class: str,
    *,
    protocol: str,
    oof_seeds: tuple[int, ...],
) -> dict[str, Any]:
    parameters = parameter_records(model, dataset, model_id, tuning, policy_class)
    return {
        "modelId": model_id,
        "policyClass": policy_class,
        "modelLabel": MODEL_LABELS[model_id],
        "description": MODEL_DESCRIPTIONS[model_id],
        "parameterCount": len(parameters),
        "parameters": parameters,
        "diagnostics": {
            "oof": metric_summary(
                dataset.y,
                oof_prediction,
                fold_test_indices=oof_test_indices(dataset, oof_seeds),
            ),
            "train": metric_summary(dataset.y, full_prediction),
        },
        "tuning": tuning,
        "protocol": protocol,
        "caveats": model_caveats(dataset, model_id, policy_class),
    }


def subset_dataset(dataset: pooled.Dataset, indices: np.ndarray, suffix: str) -> pooled.Dataset:
    return pooled.Dataset(
        name=f"{dataset.name}_{suffix}",
        frame=dataset.frame.iloc[indices].reset_index(drop=True),
        y=np.asarray(dataset.y[indices], dtype=float),
        weights=np.asarray(dataset.weights[indices], dtype=float),
        c0=np.asarray(dataset.c0, dtype=float),
        c1=np.asarray(dataset.c1, dtype=float),
        domain_names=list(dataset.domain_names),
    )


def fit_full_model(dataset: pooled.Dataset, model_id: str, tuning: dict[str, Any]) -> Any:
    indices = np.arange(dataset.n)
    if model_id == "linear":
        return fit_linear_baseline(dataset, indices, TWO_PHASE)
    if model_id == "olmix_loglinear":
        return fit_olmix_loglinear_baseline(dataset, indices, TWO_PHASE)
    if model_id in {"canonical", "effective_exposure", "effective_exposure_geometry"}:
        return dsp_fit(dataset, indices, model_id, TWO_PHASE)
    if model_id == "separate_heads":
        l2 = float(tuning.get("l2", select_separate_l2(dataset, TWO_PHASE)[0]))
        return separate_fit(dataset, indices, l2, TWO_PHASE)
    if model_id == "compact_retained_state":
        return compact_fit(dataset, indices, float(tuning["l2"]), TWO_PHASE)
    if model_id == "bucket_family_grp":
        shape_values = tuning["shapeParameters"]
        shape = family_grp.Shape(
            exponent=float(shape_values["exponent"]),
            late_multiplier=float(shape_values["lateMultiplier"]),
            forgetting_rate=float(shape_values["forgettingRate"]),
            penalty_threshold=float(shape_values["penaltyThreshold"]),
        )
        return bucket_fit(dataset, indices, shape, float(tuning["l2"]))
    if model_id == "bucket_family_power_separate_heads":
        shape_values = tuning["shapeParameters"]
        shape = retained_grp.Shape(
            rate=1.0,
            exponent=float(shape_values["exponent"]),
            late_multiplier=float(shape_values["lateMultiplier"]),
            forgetting_rate=float(shape_values["forgettingRate"]),
            penalty_threshold=float(shape_values["penaltyThreshold"]),
        )
        return power_heads_fit(dataset, indices, shape, float(tuning["l2"]), TWO_PHASE)
    if model_id == "bucket_family_power_separate_heads_family_onset":
        shape_values = tuning["shapeParameters"]
        shape = retained_grp.Shape(
            rate=1.0,
            exponent=float(shape_values["exponent"]),
            late_multiplier=float(shape_values["lateMultiplier"]),
            forgetting_rate=float(shape_values["forgettingRate"]),
            penalty_threshold=float(shape_values["penaltyThreshold"]),
        )
        return power_family_onset_fit(
            dataset,
            indices,
            shape,
            float(tuning["l2"]),
            float(tuning["tauShrink"]),
            TWO_PHASE,
            multistart=True,
        )
    if model_id in RETAINED_GRP_MODEL_IDS:
        shape_values = tuning["shapeParameters"]
        shape = retained_grp.Shape(
            rate=float(shape_values["rate"]),
            exponent=float(shape_values["power"]),
            late_multiplier=float(shape_values["lateMultiplier"]),
            forgetting_rate=float(shape_values["forgettingRate"]),
            penalty_threshold=float(shape_values["penaltyThreshold"]),
        )
        return retained_grp_fit(
            dataset,
            indices,
            RETAINED_GRP_VARIANTS[model_id],
            shape,
            float(tuning["l2"]),
        )
    if dataset.name.startswith("starcoder"):
        _params, model = starcoder_grp_fit(dataset, indices)
        return model
    raise ValueError(f"Nike-swoosh fit is unsupported for {dataset.name}/{model_id}")


def predict_full_model(model: Any, dataset: pooled.Dataset, model_id: str, weights: np.ndarray) -> np.ndarray:
    if model_id in BASELINE_MODEL_IDS:
        return np.asarray(model.predict(weights), dtype=float)
    if model_id in {"canonical", "effective_exposure", "effective_exposure_geometry"}:
        return dsp_predict(model, dataset, weights)
    if model_id == "separate_heads":
        return separate_predict(model, dataset, weights, TWO_PHASE)
    if model_id in NEW_MODEL_IDS:
        return np.asarray(model.predict(weights), dtype=float)
    if dataset.name.startswith("starcoder"):
        return model.predict(weights)
    raise ValueError(f"Nike-swoosh prediction is unsupported for {dataset.name}/{model_id}")


def nike_swoosh_diagnostic(
    dataset: pooled.Dataset,
    model_id: str,
    full_model: Any,
    tuning: dict[str, Any],
    policy_class: str,
) -> dict[str, Any] | None:
    if policy_class != TWO_PHASE or not dataset.name.startswith("starcoder"):
        return None
    code_index = dataset.domain_names.index("starcoder")
    slice_indices = np.flatnonzero(np.isclose(dataset.weights[:, 0, code_index], 0.0, atol=1e-10))
    if len(slice_indices) < 8:
        raise ValueError(f"{dataset.name} has only {len(slice_indices)} p0=0 points")
    slice_data = subset_dataset(dataset, slice_indices, "p0_zero_slice")
    slice_tuning = dict(tuning)
    if model_id == "separate_heads":
        slice_l2, slice_sweep = select_separate_l2(slice_data, TWO_PHASE)
        slice_tuning = {"l2": slice_l2, "l2Sweep": slice_sweep}
    elif model_id == "compact_retained_state":
        slice_l2, slice_sweep = select_compact_l2(slice_data, TWO_PHASE)
        slice_tuning = {"l2": slice_l2, "l2Sweep": slice_sweep}
    elif model_id == "bucket_family_grp":
        slice_shape, slice_l2, slice_sweep = select_bucket_hyperparameters(slice_data, TWO_PHASE)
        slice_tuning = {
            "l2": slice_l2,
            "shapeSweep": slice_sweep,
            "shapeParameters": {
                "exponent": slice_shape.exponent,
                "lateMultiplier": slice_shape.late_multiplier,
                "forgettingRate": slice_shape.forgetting_rate,
                "penaltyThreshold": slice_shape.penalty_threshold,
            },
        }
    elif model_id == "bucket_family_power_separate_heads":
        slice_shape, slice_l2, slice_sweep = select_power_heads_hyperparameters(slice_data, TWO_PHASE)
        slice_tuning = {
            "l2": slice_l2,
            "shapeSweep": slice_sweep,
            "shapeParameters": {
                "exponent": slice_shape.exponent,
                "lateMultiplier": slice_shape.late_multiplier,
                "forgettingRate": slice_shape.forgetting_rate,
                "penaltyThreshold": slice_shape.penalty_threshold,
            },
        }
    elif model_id == "bucket_family_power_separate_heads_family_onset":
        slice_shape, slice_l2, slice_tau_shrink, slice_sweep = select_power_family_onset_hyperparameters(
            slice_data,
            TWO_PHASE,
        )
        slice_tuning = {
            "l2": slice_l2,
            "tauShrink": slice_tau_shrink,
            **slice_sweep,
            "shapeParameters": {
                "exponent": slice_shape.exponent,
                "lateMultiplier": slice_shape.late_multiplier,
                "forgettingRate": slice_shape.forgetting_rate,
                "penaltyThreshold": slice_shape.penalty_threshold,
            },
        }
    elif model_id in RETAINED_GRP_MODEL_IDS:
        slice_shape, slice_l2, slice_sweep = select_retained_grp_hyperparameters(
            slice_data,
            model_id,
            TWO_PHASE,
        )
        slice_tuning = {
            "variant": RETAINED_GRP_VARIANTS[model_id].name,
            "l2": slice_l2,
            "shapeSweep": slice_sweep,
            "shapeParameters": {
                "rate": slice_shape.rate,
                "power": slice_shape.exponent,
                "lateMultiplier": slice_shape.late_multiplier,
                "forgettingRate": slice_shape.forgetting_rate,
                "penaltyThreshold": slice_shape.penalty_threshold,
            },
        }
    slice_model = fit_full_model(slice_data, model_id, slice_tuning)
    grid = np.linspace(0.0, 1.0, 161)
    grid_weights = np.zeros((len(grid), 2, 2), dtype=float)
    grid_weights[:, 0, 0] = 1.0
    grid_weights[:, 1, 0] = 1.0 - grid
    grid_weights[:, 1, 1] = grid
    slice_prediction = predict_full_model(slice_model, slice_data, model_id, grid_weights)
    overall_prediction = predict_full_model(full_model, dataset, model_id, grid_weights)
    observed_order = np.argsort(dataset.weights[slice_indices, 1, code_index])
    observed_indices = slice_indices[observed_order]
    return {
        "sliceDefinition": "Phase-0 StarCoder weight = 0; phase-1 StarCoder weight varies.",
        "xLabel": "Phase-1 StarCoder weight",
        "yLabel": "Dolma 100 Programming Languages BPB",
        "observed": {
            "x": dataset.weights[observed_indices, 1, code_index].tolist(),
            "y": dataset.y[observed_indices].tolist(),
            "rowIds": [f"fit:{dataset.name}:{index}" for index in observed_indices],
        },
        "grid": grid.tolist(),
        "sliceFit": {
            "label": f"Fit only on p0=0 slice (n={len(slice_indices)})",
            "prediction": slice_prediction.tolist(),
            "minimumX": float(grid[int(np.argmin(slice_prediction))]),
            "minimumY": float(np.min(slice_prediction)),
        },
        "overallFit": {
            "label": f"Fit on full surface (n={dataset.n}), evaluated on p0=0",
            "prediction": overall_prediction.tolist(),
            "minimumX": float(grid[int(np.argmin(overall_prediction))]),
            "minimumY": float(np.min(overall_prediction)),
        },
    }


def load_cosine_starcoder() -> pooled.Dataset:
    frame = pd.read_csv(COSINE_DATA)
    frame = frame.loc[frame["status"].eq("completed") & frame[STARCODER_TARGET_COLUMN].notna()].reset_index(drop=True)
    weights = np.stack(
        [
            frame[["phase_0_nemotron_full", "phase_0_starcoder"]].to_numpy(dtype=float),
            frame[["phase_1_nemotron_full", "phase_1_starcoder"]].to_numpy(dtype=float),
        ],
        axis=1,
    )
    c0 = np.asarray(
        [
            np.median(
                frame.loc[frame["phase_0_nemotron_full"] > 0, "phase_0_nemotron_epochs"]
                / frame.loc[frame["phase_0_nemotron_full"] > 0, "phase_0_nemotron_full"]
            ),
            np.median(
                frame.loc[frame["phase_0_starcoder"] > 0, "phase_0_starcoder_epochs"]
                / frame.loc[frame["phase_0_starcoder"] > 0, "phase_0_starcoder"]
            ),
        ],
        dtype=float,
    )
    c1 = np.asarray(
        [
            np.median(
                frame.loc[frame["phase_1_nemotron_full"] > 0, "phase_1_nemotron_epochs"]
                / frame.loc[frame["phase_1_nemotron_full"] > 0, "phase_1_nemotron_full"]
            ),
            np.median(
                frame.loc[frame["phase_1_starcoder"] > 0, "phase_1_starcoder_epochs"]
                / frame.loc[frame["phase_1_starcoder"] > 0, "phase_1_starcoder"]
            ),
        ],
        dtype=float,
    )
    return pooled.Dataset(
        name="starcoder_cosine_50_50",
        frame=frame,
        y=frame[STARCODER_TARGET_COLUMN].to_numpy(dtype=float),
        weights=weights,
        c0=c0,
        c1=c1,
        domain_names=list(STARCODER_DOMAINS),
    )


def load_wsd80_starcoder(cosine: pooled.Dataset) -> pooled.Dataset:
    frame = pd.read_csv(WSD80_DATA)
    p0 = frame["phase_0_starcoder"].to_numpy(dtype=float)
    p1 = frame["phase_1_starcoder"].to_numpy(dtype=float)
    weights = np.stack(
        [np.column_stack([1.0 - p0, p0]), np.column_stack([1.0 - p1, p1])],
        axis=1,
    )
    return pooled.Dataset(
        name="starcoder_wsd_80_20",
        frame=frame,
        y=frame["wsd80_bpb"].to_numpy(dtype=float),
        weights=weights,
        c0=np.asarray(cosine.c0 * (0.8 / 0.5), dtype=float),
        c1=np.asarray(cosine.c1 * (0.2 / 0.5), dtype=float),
        domain_names=list(STARCODER_DOMAINS),
    )


def load_300m_single_phase_dataset(target_id: str, reference: pooled.Dataset) -> pooled.Dataset:
    target_column = {
        "uncheatable": "eval_uncheatable_eval_bpb",
        "table9": "table9_macro_bpb",
    }[target_id]
    frame = pd.read_csv(ONE_PHASE_300M_DATA)
    missing = [f"weight_{domain}" for domain in reference.domain_names if f"weight_{domain}" not in frame.columns]
    if missing:
        raise ValueError(f"One-phase 300M panel is missing weight columns: {missing[:5]}")
    if frame[target_column].isna().any():
        raise ValueError(f"One-phase 300M panel has missing {target_column} values")
    tied = frame[[f"weight_{domain}" for domain in reference.domain_names]].to_numpy(dtype=float)
    if not np.allclose(tied.sum(axis=1), 1.0, atol=1e-8):
        raise ValueError("One-phase 300M weights do not sum to one")
    return pooled.Dataset(
        name=f"300m_single_{target_id}",
        frame=frame,
        y=frame[target_column].to_numpy(dtype=float),
        weights=np.stack([tied, tied], axis=1),
        c0=np.asarray(reference.c0, dtype=float),
        c1=np.asarray(reference.c1, dtype=float),
        domain_names=list(reference.domain_names),
    )


def tied_policy_subset(dataset: pooled.Dataset) -> tuple[pooled.Dataset, np.ndarray]:
    tied = np.max(np.abs(dataset.weights[:, 0, :] - dataset.weights[:, 1, :]), axis=1) < 1e-9
    indices = np.flatnonzero(tied)
    if len(indices) < 3:
        raise ValueError(f"{dataset.name} has only {len(indices)} tied-policy observations")
    return subset_dataset(dataset, indices, "single_phase"), indices


def display_domain_name(domain: str) -> str:
    if domain == "nemotron_full":
        return "Nemotron broad"
    if domain == "starcoder":
        return "StarCoder"
    return legacy_exporter.display_domain_name(domain)


def domain_group(domain: str) -> str:
    if domain in STARCODER_DOMAINS:
        return "Broad / code"
    return legacy_exporter.domain_group(domain)


def domain_records(
    dataset: pooled.Dataset,
    alpha0: float,
    *,
    known_budget: float | None = None,
) -> tuple[list[dict[str, Any]], np.ndarray, float]:
    natural = natural_weights(dataset, alpha0)
    budget = target_budget(dataset, alpha0, known_budget)
    token_counts = alpha0 * budget * natural / np.maximum(dataset.c0, 1e-12)
    records = [
        {
            "id": domain,
            "label": display_domain_name(domain),
            "group": domain_group(domain),
            "proportionalWeight": float(natural[index]),
            "tokenCount": float(token_counts[index]),
            "phase0EpochFactor": float(dataset.c0[index]),
            "phase1EpochFactor": float(dataset.c1[index]),
        }
        for index, domain in enumerate(dataset.domain_names)
    ]
    return records, natural, budget


def row_name(dataset: pooled.Dataset, index: int) -> str:
    row = dataset.frame.iloc[index]
    for column in ("run_name", "wandb_run_name", "candidate_name", "wandb_run_id", "run_id"):
        if column in row and pd.notna(row[column]):
            return str(row[column])
    return f"{dataset.name}_{index:04d}"


def row_url(dataset: pooled.Dataset, index: int) -> str | None:
    row = dataset.frame.iloc[index]
    if "wandb_url" in row and pd.notna(row["wandb_url"]):
        return str(row["wandb_url"])
    if "wandb_run_id" in row and pd.notna(row["wandb_run_id"]):
        return f"https://wandb.ai/marin-community/marin/runs/{row['wandb_run_id']}"
    return None


def row_records(
    dataset: pooled.Dataset,
    target_id: str,
    natural: np.ndarray,
    alpha0: float,
    alpha1: float,
    available_policy_classes: tuple[str, ...],
) -> list[dict[str, Any]]:
    records = []
    for index in range(dataset.n):
        phase0 = dataset.weights[index, 0]
        phase1 = dataset.weights[index, 1]
        aggregate = alpha0 * phase0 + alpha1 * phase1
        phase0_epochs = phase0 * dataset.c0
        phase1_epochs = phase1 * dataset.c1
        total_epochs = phase0_epochs + phase1_epochs
        phase_tv = 0.5 * float(np.abs(phase0 - phase1).sum())
        aggregate_tv = 0.5 * float(np.abs(aggregate - natural).sum())
        aggregate_kl = float(np.sum(aggregate * np.log(np.maximum(aggregate, 1e-12) / np.maximum(natural, 1e-12))))
        record_id = f"fit:{dataset.name}:{index}"
        tied = bool(np.allclose(phase0, phase1, atol=1e-9))
        policy_classes = [TWO_PHASE]
        fit_policies = [TWO_PHASE]
        if tied and SINGLE_PHASE in available_policy_classes:
            policy_classes.insert(0, SINGLE_PHASE)
            fit_policies.insert(0, SINGLE_PHASE)
        records.append(
            {
                "id": record_id,
                "name": row_name(dataset, index),
                "split": "fit",
                "policyFamily": SINGLE_PHASE if tied else TWO_PHASE,
                "phaseFamily": SINGLE_PHASE if tied else TWO_PHASE,
                "policyClasses": policy_classes,
                "fitPolicies": fit_policies,
                "phaseStructure": "phase-tied weights" if tied else "two independent phase weights",
                "panel": dataset.name,
                "method": "observed swarm design",
                "sourceExperiment": dataset.name,
                "wandbUrl": row_url(dataset, index),
                "interventionType": None,
                "targetDomain": None,
                "directionType": None,
                "directionId": None,
                "isSharedAlias": False,
                "pairedRow": None,
                "candidateTarget": None,
                "observed": {target_id: float(dataset.y[index])},
                "phase0": phase0.tolist(),
                "phase1": phase1.tolist(),
                "aggregate": aggregate.tolist(),
                "phase0Epochs": phase0_epochs.tolist(),
                "phase1Epochs": phase1_epochs.tolist(),
                "totalEpochs": total_epochs.tolist(),
                "diagnostics": {
                    "phaseTv": phase_tv,
                    "aggregateTvToProportional": aggregate_tv,
                    "aggregateKlToProportional": aggregate_kl,
                    "maxEpoch": float(np.max(total_epochs)),
                    "nearestFitId": record_id,
                    "supportDistance": 0.0,
                },
            }
        )
    return records


def empty_metric_summary() -> dict[str, int | None]:
    return {
        "n": 0,
        "rmse": None,
        "mae": None,
        "spearman": None,
        "regretAt1": None,
        "foldMeanRegretAt1": None,
        "lowerTailOptimism": None,
        "lowTailRmse": None,
        "lowerTailCount": 0,
    }


def model_diagnostics(
    dataset: pooled.Dataset,
    prediction: np.ndarray,
    seeds: tuple[int, ...],
) -> dict[str, Any]:
    return {
        "fitOof": metric_summary(
            dataset.y,
            prediction,
            fold_test_indices=oof_test_indices(dataset, seeds),
        ),
        "heldout": empty_metric_summary(),
        "heldoutSinglePhase": empty_metric_summary(),
        "heldoutTwoPhase": empty_metric_summary(),
    }


def nearest_row(weights: np.ndarray, target: np.ndarray) -> int:
    phase0_distance = np.abs(weights[:, 0] - target[None, :]).sum(axis=1)
    phase1_distance = np.abs(weights[:, 1] - target[None, :]).sum(axis=1)
    return int(np.argmin(phase0_distance + phase1_distance))


def generic_baselines(
    dataset: pooled.Dataset,
    target_id: str,
    rows: list[dict[str, Any]],
    natural: np.ndarray,
) -> dict[str, list[dict[str, str]]]:
    options: list[dict[str, str]] = []
    natural_index = nearest_row(dataset.weights, natural)
    options.append({"id": rows[natural_index]["id"], "label": "Nearest proportional policy"})
    tied = np.max(np.abs(dataset.weights[:, 0] - dataset.weights[:, 1]), axis=1) < 1e-9
    if tied.any():
        tied_indices = np.flatnonzero(tied)
        tied_best = int(tied_indices[int(np.argmin(dataset.y[tied_indices]))])
        options.append({"id": rows[tied_best]["id"], "label": "Empirical constant-mixture frontier"})
    best = int(np.argmin(dataset.y))
    options.append({"id": rows[best]["id"], "label": "Empirical two-phase frontier"})
    if dataset.name.startswith("starcoder"):
        code_index = dataset.domain_names.index("starcoder")
        boundary = np.isclose(dataset.weights[:, 0, code_index], 0.0, atol=1e-10)
        boundary_indices = np.flatnonzero(boundary)
        boundary_best = int(boundary_indices[int(np.argmin(dataset.y[boundary_indices]))])
        options.append({"id": rows[boundary_best]["id"], "label": "Best p0=0 boundary point"})
    return {target_id: options}


def cache_path(swarm_id: str, target_id: str, policy_class: str, model_id: str) -> Path:
    return CACHE_DIR / swarm_id / target_id / policy_class / f"{model_id}.json"


def cached_swarm_fit(
    swarm_id: str,
    target_id: str,
    fit_dataset: pooled.Dataset,
    evaluation_dataset: pooled.Dataset,
    fit_row_indices: np.ndarray,
    policy_class: str,
    model_id: str,
    source_paths: list[Path],
    *,
    seeds: tuple[int, ...],
) -> dict[str, Any]:
    model_dependencies: list[Path] = []
    if model_id == "compact_retained_state":
        model_dependencies.append(Path(compact_retained.__file__))
    elif model_id == "bucket_family_grp":
        model_dependencies.extend([Path(family_grp.__file__), BUCKET_FAMILY_MODEL])
    elif model_id == "hierarchical_phase_bucket_replay":
        model_dependencies.extend([Path(family_grp.__file__), Path(hierarchical_grp.__file__)])
    elif model_id in {
        "bucket_family_power_separate_heads",
        "bucket_family_power_separate_heads_family_onset",
    }:
        model_dependencies.extend([Path(family_grp.__file__), Path(phase_head_grp.__file__)])
        if model_id == "bucket_family_power_separate_heads_family_onset":
            model_dependencies.append(Path(family_onset_grp.__file__))
    elif model_id in RETAINED_GRP_MODEL_IDS:
        model_dependencies.extend([Path(family_grp.__file__), Path(retained_grp.__file__)])
    elif model_id == "olmix_loglinear":
        model_dependencies.append(Path(olmix_loglinear.__file__))
    fingerprint_payload: dict[str, Any] = {
        "swarm": swarm_id,
        "target": target_id,
        "policy": policy_class,
        "model": model_id,
        "seeds": list(seeds),
    }
    if swarm_id == "delphi_3e18":
        # Selection grids and fit dispatch live in this module. Tie Delphi's
        # tuning cache to the Observatory version so logic changes cannot reuse
        # hyperparameters from an older implementation.
        fingerprint_payload["fitLogicVersion"] = CACHE_VERSION
        fingerprint_payload["exposureCoefficients"] = {
            "c0": hashlib.sha256(np.asarray(fit_dataset.c0, dtype="<f8").tobytes()).hexdigest(),
            "c1": hashlib.sha256(np.asarray(fit_dataset.c1, dtype="<f8").tobytes()).hexdigest(),
        }
    fingerprint = file_fingerprint(
        [*source_paths, *model_dependencies],
        fingerprint_payload,
        version=MODEL_CACHE_VERSIONS[model_id],
    )
    path = cache_path(swarm_id, target_id, policy_class, model_id)
    if path.exists():
        cached = json.loads(path.read_text())
        if (
            cached.get("modelCacheVersion") == MODEL_CACHE_VERSIONS[model_id]
            and cached.get("fingerprint") == fingerprint
        ):
            print(f"cache hit: {swarm_id}/{target_id}/{policy_class}/{model_id}", flush=True)
            return cached
    print(f"fitting: {swarm_id}/{target_id}/{policy_class}/{model_id}", flush=True)
    model, fit_prediction, fit_full_prediction, tuning = fit_one_model(
        fit_dataset,
        model_id,
        policy_class,
        seeds,
    )
    full_prediction = predict_model(model, fit_dataset, model_id, policy_class, evaluation_dataset.weights)
    prediction = np.asarray(full_prediction, dtype=float).copy()
    prediction[fit_row_indices] = fit_prediction
    detail = fit_detail(
        fit_dataset,
        model_id,
        model,
        fit_prediction,
        fit_full_prediction,
        tuning,
        policy_class,
        protocol=(
            f"Five-fold OOF averaged over seeds {list(seeds)} on {fit_dataset.n} {policy_class.replace('_', '-')} "
            f"rows; full model refit on those rows and projected onto {evaluation_dataset.n} observed policies."
        ),
        oof_seeds=seeds,
    )
    result = {
        "modelCacheVersion": MODEL_CACHE_VERSIONS[model_id],
        "fingerprint": fingerprint,
        "prediction": prediction.tolist(),
        "fullFitPrediction": full_prediction.tolist(),
        "fitDetail": detail,
        "nikeSwoosh": nike_swoosh_diagnostic(fit_dataset, model_id, model, tuning, policy_class),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, separators=(",", ":"), allow_nan=False) + "\n")
    return result


def build_generic_swarm(
    *,
    swarm_id: str,
    label: str,
    description: str,
    dataset: pooled.Dataset,
    target_id: str,
    target_label: str,
    metric_column: str,
    source_paths: list[Path],
    known_budget: float | None = None,
) -> dict[str, Any]:
    alpha0, alpha1 = phase_fractions(dataset)
    domains, natural, budget = domain_records(dataset, alpha0, known_budget=known_budget)
    available_policy_classes = (TWO_PHASE,) if dataset.name == "production_uncheatable" else POLICY_CLASSES
    rows = row_records(dataset, target_id, natural, alpha0, alpha1, available_policy_classes)
    predictions: dict[str, Any] = {target_id: {policy: {} for policy in available_policy_classes}}
    diagnostics: dict[str, Any] = {target_id: {policy: {} for policy in available_policy_classes}}
    fits: dict[str, Any] = {target_id: {policy: {} for policy in available_policy_classes}}
    nike_swoosh: dict[str, Any] = {target_id: {policy: {} for policy in available_policy_classes}}
    policy_fit_counts: dict[str, int] = {}
    for policy_class in available_policy_classes:
        if policy_class == SINGLE_PHASE:
            fit_dataset, fit_row_indices = tied_policy_subset(dataset)
        else:
            fit_dataset = dataset
            fit_row_indices = np.arange(dataset.n)
        policy_fit_counts[policy_class] = fit_dataset.n
        for model_id in VISIBLE_MODEL_IDS:
            result = cached_swarm_fit(
                swarm_id,
                target_id,
                fit_dataset,
                dataset,
                fit_row_indices,
                policy_class,
                model_id,
                source_paths,
                seeds=(0,),
            )
            predictions[target_id][policy_class][model_id] = {
                "prediction": result["prediction"],
                "fullFitPrediction": result["fullFitPrediction"],
            }
            diagnostics[target_id][policy_class][model_id] = model_diagnostics(
                fit_dataset,
                np.asarray(result["prediction"], dtype=float)[fit_row_indices],
                (0,),
            )
            fits[target_id][policy_class][model_id] = result["fitDetail"]
            if result["nikeSwoosh"] is not None:
                nike_swoosh[target_id][policy_class][model_id] = result["nikeSwoosh"]
    noise_scale = float(np.std(dataset.y, ddof=1))
    return {
        "id": swarm_id,
        "label": label,
        "description": description,
        "dataset": {
            "label": label,
            "fitDesignCount": dataset.n,
            "rawFitObservationCount": dataset.n,
            "heldoutCount": 0,
            "noiseReferenceCount": 0,
            "supplementalCandidateCount": 0,
            "phaseFractions": [alpha0, alpha1],
            "targetBudget": float(budget),
            "oofSeeds": [0],
            "fitProtocol": "Five-fold random OOF; full fit on all observed surface rows.",
            "policyClasses": list(available_policy_classes),
            "policyFitCounts": policy_fit_counts,
        },
        "domains": domains,
        "targets": {
            target_id: {
                "id": target_id,
                "label": target_label,
                "metricColumn": metric_column,
                "lowerIsBetter": True,
                "noiseReference": {
                    "n": dataset.n,
                    "mean": float(np.mean(dataset.y)),
                    "standardDeviation": noise_scale,
                    "differenceStandardDeviation": noise_scale,
                },
                "noiseLabel": "Target SD used only as a visual scale; no repeat-noise estimate is available.",
            }
        },
        "rows": rows,
        "predictions": predictions,
        "diagnostics": diagnostics,
        "baselines": generic_baselines(dataset, target_id, rows, natural),
        "fits": fits,
        "nikeSwoosh": nike_swoosh,
        "provenance": {
            "sources": [str(path.relative_to(REPO_ROOT)) for path in source_paths],
            "exporter": str(Path(__file__).relative_to(REPO_ROOT)),
        },
    }


def load_legacy_bundle() -> dict[str, Any]:
    if LEGACY_DATA.exists():
        bundle = json.loads(LEGACY_DATA.read_text())
    elif APP_DATA.exists():
        bundle = json.loads(APP_DATA.read_text())
        if bundle.get("schemaVersion") != 1:
            raise ValueError(f"{APP_DATA} is already v2, but the preserved v1 bundle is missing")
        LEGACY_DATA.parent.mkdir(parents=True, exist_ok=True)
        LEGACY_DATA.write_bytes(APP_DATA.read_bytes())
    else:
        raise FileNotFoundError("Generate the 300M debugger bundle before the multi-swarm observatory")
    if bundle.get("schemaVersion") != 1:
        raise ValueError(f"Expected v1 300M bundle, got schema {bundle.get('schemaVersion')}")
    return bundle


def cached_300m_fit_detail(
    legacy: dict[str, Any],
    target_id: str,
    dataset: pooled.Dataset,
    model_id: str,
) -> dict[str, Any]:
    tuning = dict(legacy["models"][model_id]["protocol"]["targetParameters"][target_id])
    fingerprint = file_fingerprint(
        [Path(__file__), LEGACY_DATA],
        {
            "swarm": "300m",
            "target": target_id,
            "policy": TWO_PHASE,
            "model": model_id,
            "tuning": tuning,
        },
    )
    path = cache_path("300m", target_id, TWO_PHASE, model_id)
    if path.exists():
        cached = json.loads(path.read_text())
        if "fitDetail" in cached:
            print(f"compatible cache hit: 300m/{target_id}/{model_id}", flush=True)
            return cached["fitDetail"]
    indices = np.arange(dataset.n)
    if model_id in {"canonical", "effective_exposure", "effective_exposure_geometry"}:
        model = dsp_fit(dataset, indices, model_id, TWO_PHASE)
        full_prediction = dsp_predict(model, dataset, dataset.weights)
    elif model_id == "separate_heads":
        model = separate_fit(dataset, indices, float(tuning["l2"]), TWO_PHASE)
        full_prediction = separate_predict(model, dataset, dataset.weights, TWO_PHASE)
    else:
        model = grp_300m_fit(dataset, indices, float(tuning["l2"]), TWO_PHASE)
        full_prediction = model.predict(dataset.weights)
    old_prediction = np.asarray(legacy["predictions"][target_id][model_id]["prediction"][: dataset.n], dtype=float)
    detail = fit_detail(
        dataset,
        model_id,
        model,
        old_prediction,
        full_prediction,
        tuning,
        TWO_PHASE,
        protocol=(
            "Existing three-seed, five-fold panel-stratified grouped OOF; full model refit on 280 collapsed designs."
        ),
        oof_seeds=tuple(int(seed) for seed in legacy["dataset"]["oofSeeds"]),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"fingerprint": fingerprint, "fitDetail": detail}, separators=(",", ":")) + "\n")
    return detail


def legacy_evaluation_dataset(
    legacy: dict[str, Any],
    target_id: str,
    reference: pooled.Dataset,
) -> pooled.Dataset:
    rows = legacy["rows"]
    frame = pd.DataFrame(
        {
            "run_name": [row["name"] for row in rows],
            "panel_source": [row.get("panel") or row["split"] for row in rows],
        }
    )
    observed = np.asarray(
        [float(row["observed"][target_id]) if row["observed"].get(target_id) is not None else np.nan for row in rows],
        dtype=float,
    )
    weights = np.stack(
        [
            np.stack([np.asarray(row["phase0"], dtype=float), np.asarray(row["phase1"], dtype=float)], axis=0)
            for row in rows
        ],
        axis=0,
    )
    return pooled.Dataset(
        name=f"300m_all_{target_id}",
        frame=frame,
        y=observed,
        weights=weights,
        c0=np.asarray(reference.c0, dtype=float),
        c1=np.asarray(reference.c1, dtype=float),
        domain_names=list(reference.domain_names),
    )


def augment_legacy_policy_membership(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    augmented: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        policy_class = row.get("phaseFamily") or row.get("policyFamily")
        if policy_class not in POLICY_CLASSES:
            raise ValueError(f"Unknown 300M row policy class {policy_class!r} for {row['name']}")
        row["policyClasses"] = [policy_class]
        fit_policies: list[str] = []
        if row["split"] == "fit":
            fit_policies.append(TWO_PHASE)
        if row.get("panel") == "single_phase_augmented_panel":
            fit_policies.append(SINGLE_PHASE)
        row["fitPolicies"] = fit_policies
        augmented.append(row)
    return augmented


def legacy_two_phase_diagnostics(
    fit_dataset: pooled.Dataset,
    evaluation_dataset: pooled.Dataset,
    rows: list[dict[str, Any]],
    prediction: np.ndarray,
    seeds: tuple[int, ...],
) -> dict[str, Any]:
    split = np.asarray([row["split"] for row in rows], dtype=object)
    alias = np.asarray([bool(row["isSharedAlias"]) for row in rows], dtype=bool)
    phase_family = np.asarray([row["phaseFamily"] for row in rows], dtype=object)
    heldout = (split == "heldout") & ~alias
    return {
        "fitOof": metric_summary(
            fit_dataset.y,
            prediction[: fit_dataset.n],
            fold_test_indices=oof_test_indices(fit_dataset, seeds),
        ),
        "heldout": metric_summary(evaluation_dataset.y[heldout], prediction[heldout]),
        "heldoutSinglePhase": metric_summary(
            evaluation_dataset.y[heldout & (phase_family == SINGLE_PHASE)],
            prediction[heldout & (phase_family == SINGLE_PHASE)],
        ),
        "heldoutTwoPhase": metric_summary(
            evaluation_dataset.y[heldout & (phase_family == TWO_PHASE)],
            prediction[heldout & (phase_family == TWO_PHASE)],
        ),
    }


def build_300m_swarm(legacy: dict[str, Any]) -> dict[str, Any]:
    fits: dict[str, Any] = {}
    predictions: dict[str, Any] = {}
    diagnostics: dict[str, Any] = {}
    rows = augment_legacy_policy_membership(legacy["rows"])
    for target_id in ("uncheatable", "table9"):
        dataset = pooled.load_300m_dataset(target_id)
        legacy_names = [row["name"] for row in rows[: dataset.n]]
        dataset_names = dataset.frame["run_name"].astype(str).tolist()
        if legacy_names != dataset_names:
            raise ValueError(f"300M row order differs for {target_id}")
        single_dataset = load_300m_single_phase_dataset(target_id, dataset)
        evaluation_dataset = legacy_evaluation_dataset(legacy, target_id, dataset)
        row_indices = {row["name"]: index for index, row in enumerate(rows)}
        single_row_indices = np.asarray(
            [row_indices[name] for name in single_dataset.frame["run_name"].astype(str)],
            dtype=int,
        )
        if len(np.unique(single_row_indices)) != single_dataset.n:
            raise ValueError(f"One-phase {target_id} rows do not map one-to-one into the observatory bundle")

        predictions[target_id] = {
            TWO_PHASE: dict(legacy["predictions"][target_id]),
            SINGLE_PHASE: {},
        }
        diagnostics[target_id] = {
            TWO_PHASE: {
                model_id: legacy_two_phase_diagnostics(
                    dataset,
                    evaluation_dataset,
                    rows,
                    np.asarray(legacy["predictions"][target_id][model_id]["prediction"], dtype=float),
                    tuple(int(seed) for seed in legacy["dataset"]["oofSeeds"]),
                )
                for model_id in LEGACY_MODEL_IDS
            },
            SINGLE_PHASE: {},
        }
        fits[target_id] = {
            TWO_PHASE: {
                model_id: cached_300m_fit_detail(legacy, target_id, dataset, model_id) for model_id in LEGACY_MODEL_IDS
            },
            SINGLE_PHASE: {},
        }
        for model_id in VISIBLE_NEW_MODEL_IDS:
            result = cached_swarm_fit(
                "300m",
                target_id,
                dataset,
                evaluation_dataset,
                np.arange(dataset.n),
                TWO_PHASE,
                model_id,
                [LEGACY_DATA, Path(compact_retained.__file__), Path(family_grp.__file__)],
                seeds=(0, 1, 2),
            )
            predictions[target_id][TWO_PHASE][model_id] = {
                "prediction": result["prediction"],
                "fullFitPrediction": result["fullFitPrediction"],
            }
            diagnostics[target_id][TWO_PHASE][model_id] = legacy_two_phase_diagnostics(
                dataset,
                evaluation_dataset,
                rows,
                np.asarray(result["prediction"], dtype=float),
                (0, 1, 2),
            )
            fits[target_id][TWO_PHASE][model_id] = result["fitDetail"]
        for model_id in VISIBLE_MODEL_IDS:
            result = cached_swarm_fit(
                "300m",
                target_id,
                single_dataset,
                evaluation_dataset,
                single_row_indices,
                SINGLE_PHASE,
                model_id,
                [ONE_PHASE_300M_DATA, LEGACY_DATA],
                seeds=(0, 1, 2),
            )
            predictions[target_id][SINGLE_PHASE][model_id] = {
                "prediction": result["prediction"],
                "fullFitPrediction": result["fullFitPrediction"],
            }
            diagnostics[target_id][SINGLE_PHASE][model_id] = model_diagnostics(
                single_dataset,
                np.asarray(result["prediction"], dtype=float)[single_row_indices],
                (0, 1, 2),
            )
            fits[target_id][SINGLE_PHASE][model_id] = result["fitDetail"]
    targets = dict(legacy["targets"])
    for target in targets.values():
        target["noiseLabel"] = "Difference SD from the 11 proportional observations used by the fit panel."
    return {
        "id": "300m",
        "label": legacy["dataset"]["label"],
        "description": "300M / 6B-token Dolma 3 + Dolmino panel with fit, heldout, repeat, and candidate checkpoints.",
        "dataset": {
            **legacy["dataset"],
            "policyClasses": list(POLICY_CLASSES),
            "policyFitCounts": {SINGLE_PHASE: 280, TWO_PHASE: 280},
        },
        "domains": legacy["domains"],
        "targets": targets,
        "rows": rows,
        "predictions": predictions,
        "diagnostics": diagnostics,
        "baselines": legacy["baselines"],
        "fits": fits,
        "nikeSwoosh": {target_id: {policy: {} for policy in POLICY_CLASSES} for target_id in targets},
        "provenance": legacy["provenance"],
    }


def load_delphi_3e18_fit_dataset(target_id: str) -> pooled.Dataset:
    target_column = {
        "uncheatable": "uncheatable_bpb",
        "table9": "table9_macro_bpb",
    }[target_id]
    frame = pd.read_csv(DELPHI_3E18_DATA)
    reference = pooled.load_300m_dataset(target_id)
    domains = list(reference.domain_names)
    phase0 = frame[[f"phase_0_{domain}" for domain in domains]].to_numpy(dtype=float)
    phase1 = frame[[f"phase_1_{domain}" for domain in domains]].to_numpy(dtype=float)
    if len(frame) != 280 or not np.allclose(phase0.sum(axis=1), 1.0) or not np.allclose(phase1.sum(axis=1), 1.0):
        raise ValueError("The Delphi 3e18 fit swarm must contain 280 normalized two-phase policies")
    realized_budgets = frame["realized_train_tokens"].to_numpy(dtype=float)
    phase0_fractions = frame["phase_0_fraction"].to_numpy(dtype=float)
    if np.ptp(realized_budgets) != 0.0 or np.ptp(phase0_fractions) != 0.0:
        raise ValueError("The Delphi 3e18 fit swarm does not share one token budget and phase split")
    new_alpha0 = float(phase0_fractions[0])
    new_alpha1 = 1.0 - new_alpha0
    token_counts = np.asarray([TOP_LEVEL_DOMAIN_TOKEN_COUNTS[domain] for domain in domains], dtype=float)
    # Both swarms materialize the same simulated-epoch subsets; only the realized phase boundary changes.
    c0 = new_alpha0 * TARGET_BUDGET_DOLMA3_COMMON_CRAWL / token_counts
    c1 = new_alpha1 * TARGET_BUDGET_DOLMA3_COMMON_CRAWL / token_counts
    target = frame[target_column].to_numpy(dtype=float)
    if not np.isfinite(target).all():
        raise ValueError(f"The Delphi 3e18 fit swarm has incomplete {target_column}")
    return pooled.Dataset(
        name=f"delphi_3e18_{target_id}",
        frame=frame,
        y=target,
        weights=np.stack([phase0, phase1], axis=1),
        c0=c0,
        c1=c1,
        domain_names=domains,
    )


def load_delphi_3e18_single_phase_dataset(
    target_id: str,
    reference: pooled.Dataset,
    heldout_frame: pd.DataFrame,
    heldout_weights: np.ndarray,
) -> tuple[pooled.Dataset, np.ndarray]:
    target_column = {
        "uncheatable": "uncheatable_bpb",
        "table9": "table9_macro_bpb",
    }[target_id]
    manifest = pd.read_csv(DELPHI_3E18_ONE_PHASE_MANIFEST).sort_values("run_order").reset_index(drop=True)
    phase_weights = pd.read_csv(DELPHI_3E18_ONE_PHASE_WEIGHTS)
    domains = list(reference.domain_names)
    expected_weight_rows = len(manifest) * 2 * len(domains)
    if len(manifest) != 280 or len(phase_weights) != expected_weight_rows:
        raise ValueError(
            f"Expected a 280-row one-phase manifest and {expected_weight_rows} long-form weights, "
            f"found {len(manifest)} and {len(phase_weights)}"
        )
    if manifest["run_name"].duplicated().any() or phase_weights.duplicated(["run_name", "phase", "domain"]).any():
        raise ValueError("The Delphi one-phase panel contains duplicate policy or weight keys")
    if set(phase_weights["domain"]) != set(domains):
        raise ValueError("The Delphi one-phase panel domain set differs from the two-phase fit panel")

    weight_lookup = phase_weights.set_index(["run_name", "phase", "domain"])["weight"]
    phase0 = np.asarray(
        [[weight_lookup.loc[(run_name, "phase_0", domain)] for domain in domains] for run_name in manifest["run_name"]],
        dtype=float,
    )
    phase1 = np.asarray(
        [[weight_lookup.loc[(run_name, "phase_1", domain)] for domain in domains] for run_name in manifest["run_name"]],
        dtype=float,
    )
    if not np.allclose(phase0, phase1, atol=1e-12):
        raise ValueError("The Delphi one-phase panel contains a phase-varying policy")
    if not np.allclose(phase0.sum(axis=1), 1.0, atol=1e-10):
        raise ValueError("The Delphi one-phase panel weights do not sum to one")

    reference_indices = {str(name): index for index, name in enumerate(reference.frame["run_name"])}
    heldout_indices = {str(name): index for index, name in enumerate(heldout_frame["wandb_run_base"])}
    if len(heldout_indices) != len(heldout_frame):
        raise ValueError("The Delphi heldout registry contains duplicate W&B run bases")

    targets: list[float] = []
    evaluation_indices: list[int] = []
    frame_rows: list[dict[str, Any]] = []
    disposition_counts: Counter[str] = Counter()
    for row_index, source in manifest.iterrows():
        run_name = str(source["run_name"])
        source_run_name = str(source["source_run_name"])
        disposition = str(source["disposition"])
        disposition_counts[disposition] += 1
        record = source.to_dict()
        if disposition == "reused_exact_phase_tied_alias":
            if source_run_name not in reference_indices:
                raise ValueError(f"Missing reused two-phase source row {source_run_name}")
            reference_index = reference_indices[source_run_name]
            if not np.allclose(reference.weights[reference_index], np.stack([phase0[row_index], phase1[row_index]])):
                raise ValueError(f"Reused one-phase weights differ from source row {source_run_name}")
            source_row = reference.frame.iloc[reference_index]
            targets.append(float(reference.y[reference_index]))
            evaluation_indices.append(reference_index)
            record["wandb_url"] = source_row.get("training_wandb_url")
            record["wandb_run_id"] = source_row.get("training_wandb_run_id")
        elif disposition == "scheduled_new_training":
            if run_name not in heldout_indices:
                raise ValueError(f"Missing completed one-phase heldout row {run_name}")
            heldout_index = heldout_indices[run_name]
            heldout_row = heldout_frame.iloc[heldout_index]
            if str(heldout_row["training_series"]) != DELPHI_3E18_ONE_PHASE_SERIES:
                raise ValueError(f"One-phase run {run_name} resolved to the wrong training series")
            if not np.allclose(heldout_weights[heldout_index], np.stack([phase0[row_index], phase1[row_index]])):
                raise ValueError(f"Completed one-phase weights differ from manifest row {run_name}")
            targets.append(float(heldout_row[target_column]))
            evaluation_indices.append(reference.n + heldout_index)
            record["wandb_url"] = heldout_row["wandb_url"]
            record["wandb_run_id"] = heldout_row["wandb_run_id"]
        else:
            raise ValueError(f"Unknown Delphi one-phase disposition {disposition!r}")
        frame_rows.append(record)

    expected_dispositions = {"reused_exact_phase_tied_alias": 42, "scheduled_new_training": 238}
    if disposition_counts != expected_dispositions:
        raise ValueError(f"Unexpected Delphi one-phase composition: {dict(disposition_counts)}")
    target = np.asarray(targets, dtype=float)
    if not np.isfinite(target).all() or len(set(evaluation_indices)) != len(evaluation_indices):
        raise ValueError(f"The Delphi one-phase {target_id} panel is incomplete or maps to duplicate observations")
    return (
        pooled.Dataset(
            name=f"delphi_3e18_single_{target_id}",
            frame=pd.DataFrame(frame_rows),
            y=target,
            weights=np.stack([phase0, phase1], axis=1),
            c0=np.asarray(reference.c0, dtype=float),
            c1=np.asarray(reference.c1, dtype=float),
            domain_names=domains,
        ),
        np.asarray(evaluation_indices, dtype=int),
    )


def load_delphi_3e18_heldouts(reference: pooled.Dataset) -> tuple[pd.DataFrame, np.ndarray]:
    frame = pd.read_csv(DELPHI_3E18_HELDOUTS)
    complete = (frame["training_state"] == "finished") & (frame["checkpoint_declared_complete"] == 1)
    frame = frame.loc[complete].reset_index(drop=True)
    domains = list(reference.domain_names)

    def parse_weights(value: str) -> list[float]:
        weights = json.loads(value)
        return [float(weights[domain]) for domain in domains]

    phase0 = np.asarray([parse_weights(value) for value in frame["phase_0_weights_json"]], dtype=float)
    phase1 = np.asarray([parse_weights(value) for value in frame["phase_1_weights_json"]], dtype=float)
    weights = np.stack([phase0, phase1], axis=1)
    if len(frame) < MINIMUM_DELPHI_3E18_HELDOUTS or not np.allclose(weights.sum(axis=2), 1.0):
        raise ValueError(
            f"Expected at least {MINIMUM_DELPHI_3E18_HELDOUTS} completed normalized 3e18 heldouts, found {len(frame)}"
        )
    series_counts = frame["training_series"].value_counts()
    for series, expected_count in REQUIRED_DELPHI_3E18_HELDOUT_SERIES.items():
        actual_count = int(series_counts.get(series, 0))
        if actual_count != expected_count:
            raise ValueError(f"Expected {expected_count} completed heldouts from {series}, found {actual_count}")
    if not np.isfinite(frame[["uncheatable_bpb", "table9_macro_bpb"]].to_numpy(dtype=float)).all():
        raise ValueError("Every completed 3e18 heldout must have both headline metrics")
    heldout_alpha0 = frame["phase_0_fraction"].to_numpy(dtype=float)
    fit_alpha0, _fit_alpha1 = phase_fractions(reference)
    mismatched_split = ~np.isclose(heldout_alpha0, fit_alpha0, atol=1e-12)
    if (frame.loc[mismatched_split, "policy_class"] != "single_phase_tied").any():
        raise ValueError("A phase-varying heldout uses a different phase split from the fit swarm")
    return frame, weights


def delphi_3e18_evaluation_dataset(
    fit_dataset: pooled.Dataset,
    heldout_frame: pd.DataFrame,
    heldout_weights: np.ndarray,
    target_id: str,
) -> pooled.Dataset:
    target_column = {
        "uncheatable": "uncheatable_bpb",
        "table9": "table9_macro_bpb",
    }[target_id]
    frame = pd.concat([fit_dataset.frame, heldout_frame], ignore_index=True, sort=False)
    target = np.concatenate(
        [
            np.asarray(fit_dataset.y, dtype=float),
            heldout_frame[target_column].to_numpy(dtype=float),
        ]
    )
    return pooled.Dataset(
        name=f"delphi_3e18_all_{target_id}",
        frame=frame,
        y=target,
        weights=np.concatenate([fit_dataset.weights, heldout_weights], axis=0),
        c0=np.asarray(fit_dataset.c0, dtype=float),
        c1=np.asarray(fit_dataset.c1, dtype=float),
        domain_names=list(fit_dataset.domain_names),
    )


def delphi_3e18_rows(
    fit_uncheatable: pooled.Dataset,
    fit_table9: pooled.Dataset,
    heldout_frame: pd.DataFrame,
    heldout_weights: np.ndarray,
    single_phase_fit: pooled.Dataset,
    single_phase_fit_indices: np.ndarray,
) -> list[dict[str, Any]]:
    alpha0, alpha1 = phase_fractions(fit_uncheatable)
    natural = natural_weights(fit_uncheatable, alpha0)
    rows = row_records(fit_uncheatable, "uncheatable", natural, alpha0, alpha1, POLICY_CLASSES)
    for index, row in enumerate(rows):
        row["id"] = f"fit:delphi_3e18:{index}"
        row["observed"]["table9"] = float(fit_table9.y[index])
        row["sourceExperiment"] = "delphi_3e18_augmented_swarm_20260714"
        row["wandbUrl"] = str(fit_uncheatable.frame.iloc[index]["training_wandb_url"])
        row["diagnostics"]["nearestFitId"] = row["id"]

    heldout_dataset = pooled.Dataset(
        name="delphi_3e18_heldouts",
        frame=heldout_frame,
        y=heldout_frame["uncheatable_bpb"].to_numpy(dtype=float),
        weights=heldout_weights,
        c0=np.asarray(fit_uncheatable.c0, dtype=float),
        c1=np.asarray(fit_uncheatable.c1, dtype=float),
        domain_names=list(fit_uncheatable.domain_names),
    )
    heldout_rows = row_records(heldout_dataset, "uncheatable", natural, alpha0, alpha1, POLICY_CLASSES)
    for index, row in enumerate(heldout_rows):
        source = heldout_frame.iloc[index]
        policy_class = SINGLE_PHASE if source["policy_class"] == "single_phase_tied" else TWO_PHASE
        exact_coordinate = source["fit_panel_overlap"] == "exact_coordinate"
        distances = np.abs(fit_uncheatable.weights - heldout_weights[index][None, :, :]).sum(axis=(1, 2))
        nearest = int(np.argmin(distances))
        row["id"] = f"heldout:delphi_3e18:{source['wandb_run_id']}"
        row["split"] = "heldout"
        row["policyFamily"] = policy_class
        row["phaseFamily"] = policy_class
        row["policyClasses"] = [policy_class]
        row["fitPolicies"] = []
        row["phaseStructure"] = "phase-tied weights" if policy_class == SINGLE_PHASE else "two independent phase weights"
        row["panel"] = str(source["training_series"])
        row["method"] = f"{source['objective']} validation; {source['fit_panel_overlap']}"
        row["sourceExperiment"] = str(source["training_series"])
        row["wandbUrl"] = str(source["wandb_url"])
        row["interventionType"] = str(source["objective"])
        row["isSharedAlias"] = bool(exact_coordinate)
        row["pairedRow"] = str(source["fit_panel_run_name"]) if exact_coordinate else None
        row["candidateTarget"] = str(source["objective"])
        anchor_id = str(source.get("anchor_id", "")) if pd.notna(source.get("anchor_id")) else ""
        if anchor_id:
            raw_metadata = source.get("proposal_metadata_json", "")
            proposal_metadata = json.loads(str(raw_metadata)) if pd.notna(raw_metadata) and str(raw_metadata) else {}
            if not isinstance(proposal_metadata, dict):
                raise ValueError(f"Invalid phase-population provenance for {source['wandb_run_name']}")
            seed_block = safe_float(source.get("seed_block"))
            radius_fraction = safe_float(source.get("radius_fraction"))
            phase_information_kl = safe_float(proposal_metadata.get("phase_information_kl"))
            feasible_radius = safe_float(proposal_metadata.get("feasible_radius"))
            realized_radius = safe_float(proposal_metadata.get("realized_radius"))
            if None in (
                seed_block,
                radius_fraction,
                phase_information_kl,
                feasible_radius,
                realized_radius,
            ):
                raise ValueError(f"Incomplete phase-population provenance for {source['wandb_run_name']}")
            row["directionType"] = str(source["candidate_kind"])
            row["directionId"] = str(source["direction_id"])
            row["phasePopulation"] = {
                "candidateId": str(source["candidate_id"]),
                "anchorId": anchor_id,
                "anchorRunName": str(proposal_metadata["anchor_run_name"]),
                "directionId": str(source["direction_id"]),
                "directionLabel": str(proposal_metadata["direction_label"]),
                "seedBlock": int(seed_block),
                "radiusFraction": float(radius_fraction),
                "contrastFamily": str(source["candidate_kind"]),
                "phaseInformationKl": float(phase_information_kl),
                "feasibleRadius": float(feasible_radius),
                "realizedRadius": float(realized_radius),
            }
        if str(source["training_series"]) == DELPHI_3E18_ONE_PHASE_SERIES:
            row["fitPolicies"] = [SINGLE_PHASE]
            row["method"] = "independently trained one-phase fit-panel policy"
        row["observed"] = {
            "uncheatable": float(source["uncheatable_bpb"]),
            "table9": float(source["table9_macro_bpb"]),
        }
        row["diagnostics"]["nearestFitId"] = rows[nearest]["id"]
        row["diagnostics"]["supportDistance"] = float(distances[nearest])
    combined = [*rows, *heldout_rows]
    if len(single_phase_fit_indices) != 280 or any(
        SINGLE_PHASE not in combined[index]["fitPolicies"] for index in single_phase_fit_indices
    ):
        raise ValueError("The one-phase fit-panel rows were not marked as policy-matched fit observations")
    marked_single_phase = [index for index, row in enumerate(combined) if SINGLE_PHASE in row["fitPolicies"]]
    if marked_single_phase != sorted(single_phase_fit_indices.tolist()):
        raise ValueError("The one-phase fit-panel membership differs from the independently assembled dataset")

    two_phase_indices = {row["name"]: index for index, row in enumerate(rows)}
    if len(two_phase_indices) != fit_uncheatable.n:
        raise ValueError("The Delphi two-phase fit panel contains duplicate run names")
    distinct_pair_count = 0
    shared_pair_count = 0
    for logical_index, source in single_phase_fit.frame.iterrows():
        source_name = str(source["source_run_name"])
        if source_name not in two_phase_indices:
            raise ValueError(f"The one-phase source row {source_name} is absent from the two-phase panel")
        two_phase_index = two_phase_indices[source_name]
        single_phase_index = int(single_phase_fit_indices[logical_index])
        two_phase_row = combined[two_phase_index]
        single_phase_row = combined[single_phase_index]
        if not np.allclose(single_phase_row["phase0"], two_phase_row["aggregate"], atol=1e-12):
            raise ValueError(f"The one-phase row derived from {source_name} does not equal its aggregate policy")

        disposition = str(source["disposition"])
        if disposition == "reused_exact_phase_tied_alias":
            if single_phase_index != two_phase_index:
                raise ValueError(f"The shared phase-tied row {source_name} unexpectedly maps to a separate checkpoint")
            shared_pair_count += 1
            continue
        if disposition != "scheduled_new_training":
            raise ValueError(f"Unknown one-phase pairing disposition {disposition!r}")
        if single_phase_index == two_phase_index:
            raise ValueError(f"The independently trained one-phase row for {source_name} aliases its source checkpoint")
        if two_phase_row["pairedRow"] is not None or single_phase_row["pairedRow"] is not None:
            raise ValueError(f"The policy pair for {source_name} conflicts with an existing direct counterpart")
        two_phase_row["pairedRow"] = single_phase_row["name"]
        single_phase_row["pairedRow"] = two_phase_row["name"]
        distinct_pair_count += 1

    if (distinct_pair_count, shared_pair_count) != (238, 42):
        raise ValueError(
            "The Delphi policy-pair inventory must contain 238 independently trained pairs and 42 shared rows, "
            f"found {distinct_pair_count} and {shared_pair_count}"
        )
    return combined


def delphi_3e18_baselines(rows: list[dict[str, Any]], target_id: str) -> list[dict[str, str]]:
    def observed(row: Mapping[str, Any]) -> float:
        return float(row["observed"][target_id])

    two_phase_fit = [row for row in rows if TWO_PHASE in row["fitPolicies"]]
    options: list[dict[str, str]] = []
    for run_name, label in (
        ("baseline_proportional", "Proportional (shared fit row)"),
        ("baseline_unimax", "UniMax-8 (shared fit row)"),
    ):
        matches = [row for row in two_phase_fit if row["name"] == run_name]
        if len(matches) == 1:
            options.append({"id": matches[0]["id"], "label": label})
    for policy_class, label in (
        (SINGLE_PHASE, "One-phase fit-panel frontier"),
        (TWO_PHASE, "Two-phase fit-panel frontier"),
    ):
        candidates = [row for row in rows if policy_class in row["fitPolicies"]]
        if candidates:
            options.append({"id": min(candidates, key=observed)["id"], "label": label})
    for policy_class, label in (
        (SINGLE_PHASE, "One-phase validation frontier"),
        (TWO_PHASE, "Two-phase validation frontier"),
    ):
        candidates = [
            row
            for row in rows
            if row["split"] == "heldout"
            and not row["isSharedAlias"]
            and row["phaseFamily"] == policy_class
            and policy_class not in row["fitPolicies"]
        ]
        if candidates:
            options.append({"id": min(candidates, key=observed)["id"], "label": label})
    return options


def delphi_3e18_policy_diagnostics(
    fit_dataset: pooled.Dataset,
    evaluation_dataset: pooled.Dataset,
    rows: list[dict[str, Any]],
    prediction: np.ndarray,
    fit_row_indices: np.ndarray,
    policy_class: str,
    seeds: tuple[int, ...],
) -> dict[str, Any]:
    if len(rows) != evaluation_dataset.n or len(prediction) != evaluation_dataset.n:
        raise ValueError("Delphi diagnostics require one row and prediction per evaluated policy")
    split = np.asarray([row["split"] for row in rows], dtype=object)
    alias = np.asarray([bool(row["isSharedAlias"]) for row in rows], dtype=bool)
    phase_family = np.asarray([row["phaseFamily"] for row in rows], dtype=object)
    fit_mask = np.zeros(evaluation_dataset.n, dtype=bool)
    fit_mask[fit_row_indices] = True
    heldout = (split == "heldout") & ~alias & ~fit_mask
    policy_heldout = heldout & (phase_family == policy_class)
    fit_prediction = prediction[fit_row_indices]
    return {
        "fitOof": metric_summary(
            fit_dataset.y,
            fit_prediction,
            fold_test_indices=oof_test_indices(fit_dataset, seeds),
        ),
        "heldout": metric_summary(evaluation_dataset.y[policy_heldout], prediction[policy_heldout]),
        "heldoutAllPolicies": metric_summary(evaluation_dataset.y[heldout], prediction[heldout]),
        "heldoutSinglePhase": metric_summary(
            evaluation_dataset.y[heldout & (phase_family == SINGLE_PHASE)],
            prediction[heldout & (phase_family == SINGLE_PHASE)],
        ),
        "heldoutTwoPhase": metric_summary(
            evaluation_dataset.y[heldout & (phase_family == TWO_PHASE)],
            prediction[heldout & (phase_family == TWO_PHASE)],
        ),
    }


def delphi_3e18_noise_reference(heldout_frame: pd.DataFrame, target_column: str) -> dict[str, float | int]:
    repeats = heldout_frame[heldout_frame["training_series"] == "delphi_3e18_baseline_noise_panel_20260703"][
        target_column
    ].to_numpy(dtype=float)
    if len(repeats) != 10:
        raise ValueError(f"Expected 10 proportional repeats for {target_column}, found {len(repeats)}")
    standard_deviation = float(np.std(repeats, ddof=1))
    return {
        "n": len(repeats),
        "mean": float(np.mean(repeats)),
        "standardDeviation": standard_deviation,
        "differenceStandardDeviation": math.sqrt(2.0) * standard_deviation,
    }


def build_delphi_3e18_swarm() -> dict[str, Any]:
    fit_datasets = {target_id: load_delphi_3e18_fit_dataset(target_id) for target_id in ("uncheatable", "table9")}
    fit_uncheatable = fit_datasets["uncheatable"]
    fit_table9 = fit_datasets["table9"]
    if fit_uncheatable.frame["run_name"].tolist() != fit_table9.frame["run_name"].tolist():
        raise ValueError("The two Delphi 3e18 target exports have different fit-row order")
    heldout_frame, heldout_weights = load_delphi_3e18_heldouts(fit_uncheatable)
    one_phase_panels = {
        target_id: load_delphi_3e18_single_phase_dataset(
            target_id,
            fit_dataset,
            heldout_frame,
            heldout_weights,
        )
        for target_id, fit_dataset in fit_datasets.items()
    }
    single_uncheatable, single_fit_indices = one_phase_panels["uncheatable"]
    single_table9, single_table9_indices = one_phase_panels["table9"]
    if single_uncheatable.frame["run_name"].tolist() != single_table9.frame["run_name"].tolist():
        raise ValueError("The two Delphi one-phase target panels have different row order")
    if not np.array_equal(single_fit_indices, single_table9_indices):
        raise ValueError("The two Delphi one-phase target panels map to different Observatory rows")
    rows = delphi_3e18_rows(
        fit_uncheatable,
        fit_table9,
        heldout_frame,
        heldout_weights,
        single_uncheatable,
        single_fit_indices,
    )
    alpha0, alpha1 = phase_fractions(fit_uncheatable)
    domains, _natural, budget = domain_records(fit_uncheatable, alpha0)
    predictions: dict[str, Any] = {}
    diagnostics: dict[str, Any] = {}
    fits: dict[str, Any] = {}
    seeds = (0, 1, 2)
    for target_id, fit_dataset in fit_datasets.items():
        evaluation_dataset = delphi_3e18_evaluation_dataset(
            fit_dataset,
            heldout_frame,
            heldout_weights,
            target_id,
        )
        predictions[target_id] = {policy_class: {} for policy_class in POLICY_CLASSES}
        diagnostics[target_id] = {policy_class: {} for policy_class in POLICY_CLASSES}
        fits[target_id] = {policy_class: {} for policy_class in POLICY_CLASSES}
        for policy_class in POLICY_CLASSES:
            if policy_class == SINGLE_PHASE:
                policy_fit_dataset, fit_row_indices = one_phase_panels[target_id]
                source_paths = [
                    DELPHI_3E18_DATA,
                    DELPHI_3E18_HELDOUTS,
                    DELPHI_3E18_ONE_PHASE_MANIFEST,
                    DELPHI_3E18_ONE_PHASE_WEIGHTS,
                ]
            else:
                policy_fit_dataset = fit_dataset
                fit_row_indices = np.arange(fit_dataset.n)
                source_paths = [DELPHI_3E18_DATA, DELPHI_3E18_HELDOUTS]
            for model_id in DELPHI_3E18_MODEL_IDS:
                result = cached_swarm_fit(
                    "delphi_3e18",
                    target_id,
                    policy_fit_dataset,
                    evaluation_dataset,
                    fit_row_indices,
                    policy_class,
                    model_id,
                    source_paths,
                    seeds=seeds,
                )
                prediction = np.asarray(result["prediction"], dtype=float)
                predictions[target_id][policy_class][model_id] = {
                    "prediction": result["prediction"],
                    "fullFitPrediction": result["fullFitPrediction"],
                }
                diagnostics[target_id][policy_class][model_id] = delphi_3e18_policy_diagnostics(
                    policy_fit_dataset,
                    evaluation_dataset,
                    rows,
                    prediction,
                    fit_row_indices,
                    policy_class,
                    seeds,
                )
                fits[target_id][policy_class][model_id] = result["fitDetail"]
    archive_disjoint_count = sum(row["split"] == "heldout" and not row["isSharedAlias"] for row in rows)
    exact_count = sum(row["split"] == "heldout" and row["isSharedAlias"] for row in rows)
    fit_union = set(range(fit_uncheatable.n)) | set(single_fit_indices.tolist())
    union_heldouts = [
        row
        for index, row in enumerate(rows)
        if index not in fit_union and row["split"] == "heldout" and not row["isSharedAlias"]
    ]
    policy_heldout_counts = Counter(str(row["phaseFamily"]) for row in union_heldouts)
    union_heldout_count = len(union_heldouts)
    return {
        "id": "delphi_3e18",
        "label": "Delphi 3e18 augmented swarm",
        "description": (
            "Matched 280-row one-phase and two-phase Dolma 3 + Dolmino fit panels retrained at 3e18 FLOPs, "
            "with every completed append-only 3e18 validation checkpoint projected as heldout evidence."
        ),
        "dataset": {
            "label": "Delphi 3e18 augmented swarm",
            "fitDesignCount": fit_uncheatable.n,
            "rawFitObservationCount": fit_uncheatable.n + single_uncheatable.n,
            "heldoutCount": union_heldout_count,
            "appendOnlyArchiveCount": len(heldout_frame),
            "archiveCoordinateDisjointCount": archive_disjoint_count,
            "sharedAliasCount": exact_count,
            "noiseReferenceCount": 10,
            "supplementalCandidateCount": len(heldout_frame),
            "policyHeldoutCounts": dict(policy_heldout_counts),
            "phaseFractions": [alpha0, alpha1],
            "targetBudget": float(budget),
            "oofSeeds": list(seeds),
            "fitProtocol": (
                "Independent three-seed, five-fold panel-stratified OOF on matched 280-row one-phase and "
                f"two-phase panels; metrics use {union_heldout_count} rows disjoint from the union of both fit "
                f"panels. All {len(heldout_frame)} append-only archive rows remain visible."
            ),
            "policyClasses": list(POLICY_CLASSES),
            "policyFitCounts": {SINGLE_PHASE: single_uncheatable.n, TWO_PHASE: fit_uncheatable.n},
            "distinctPolicyPairCount": 238,
            "sharedPhaseTiedPairCount": 42,
        },
        "domains": domains,
        "targets": {
            "uncheatable": {
                "id": "uncheatable",
                "label": "Uncheatable eval BPB",
                "metricColumn": "eval/uncheatable_eval/bpb",
                "lowerIsBetter": True,
                "noiseReference": delphi_3e18_noise_reference(heldout_frame, "uncheatable_bpb"),
                "noiseLabel": "Ten independent proportional-training repeats at the same 3e18 configuration.",
            },
            "table9": {
                "id": "table9",
                "label": "OLMoBaseEval Table-9 macro BPB",
                "metricColumn": "olmo_base_easy/table9_51_component_macro_bpb",
                "lowerIsBetter": True,
                "noiseReference": delphi_3e18_noise_reference(heldout_frame, "table9_macro_bpb"),
                "noiseLabel": "Ten independent proportional-training repeats at the same 3e18 configuration.",
            },
        },
        "rows": rows,
        "predictions": predictions,
        "diagnostics": diagnostics,
        "baselines": {target_id: delphi_3e18_baselines(rows, target_id) for target_id in fit_datasets},
        "fits": fits,
        "nikeSwoosh": {target_id: {policy_class: {} for policy_class in POLICY_CLASSES} for target_id in fit_datasets},
        "provenance": {
            "sources": [
                str(DELPHI_3E18_DATA.relative_to(REPO_ROOT)),
                str(DELPHI_3E18_HELDOUTS.relative_to(REPO_ROOT)),
                str(DELPHI_3E18_ONE_PHASE_MANIFEST.relative_to(REPO_ROOT)),
                str(DELPHI_3E18_ONE_PHASE_WEIGHTS.relative_to(REPO_ROOT)),
            ],
            "exporter": str(Path(__file__).relative_to(REPO_ROOT)),
        },
    }


def model_catalog() -> dict[str, Any]:
    return {
        model_id: {
            "id": model_id,
            "label": MODEL_LABELS[model_id],
            "description": MODEL_DESCRIPTIONS[model_id],
            "familyId": MODEL_FAMILIES[model_id][0],
            "familyLabel": MODEL_FAMILIES[model_id][1],
            "variantLabel": MODEL_FAMILIES[model_id][2],
        }
        for model_id in VISIBLE_MODEL_IDS
    }


def write_bundle(output_json: Path) -> dict[str, Any]:
    legacy = load_legacy_bundle()
    cosine = load_cosine_starcoder()
    wsd80 = load_wsd80_starcoder(cosine)
    production = pooled.load_production_dataset()
    production_metadata = json.loads(PRODUCTION_MODEL.read_text())["metrics"]
    swarms = {
        "300m": build_300m_swarm(legacy),
        "delphi_3e18": build_delphi_3e18_swarm(),
        "starcoder_cosine": build_generic_swarm(
            swarm_id="starcoder_cosine",
            label="StarCoder 50/50 cosine surface",
            description=(
                "Dense two-domain, two-phase surface under equal cosine-schedule phases; "
                "target is Dolma 100 Programming Languages BPB."
            ),
            dataset=cosine,
            target_id="starcoder_bpb",
            target_label="Dolma 100 Programming Languages BPB",
            metric_column=STARCODER_TARGET_COLUMN,
            source_paths=[COSINE_DATA],
        ),
        "starcoder_wsd80": build_generic_swarm(
            swarm_id="starcoder_wsd80",
            label="StarCoder 80/20 WSD surface",
            description=(
                "Pruned two-domain StarCoder surface under an 80/20 warmup-stable-decay schedule; "
                "target is Dolma 100 Programming Languages BPB."
            ),
            dataset=wsd80,
            target_id="starcoder_bpb",
            target_label="Dolma 100 Programming Languages BPB",
            metric_column="wsd80_bpb",
            source_paths=[WSD80_DATA, COSINE_DATA],
        ),
        "production": build_generic_swarm(
            swarm_id="production",
            label="Production Grug-MoE swarm",
            description=(
                "840 sampled two-phase mixtures over 168 production buckets; "
                "GRP is intentionally evaluated without semantic family or pair grouping."
            ),
            dataset=production,
            target_id="uncheatable",
            target_label="Uncheatable eval BPB",
            metric_column="eval/uncheatable_eval/bpb",
            source_paths=[PRODUCTION_DATA, PRODUCTION_MODEL],
            known_budget=float(production_metadata["production_experiment_budget_tokens"]),
        ),
    }
    bundle = {
        "schemaVersion": 5,
        "generatedAt": datetime.now(UTC).isoformat(),
        "models": model_catalog(),
        "swarms": swarms,
        "provenance": {
            "exporter": str(Path(__file__).relative_to(REPO_ROOT)),
            "cacheVersion": CACHE_VERSION,
        },
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(bundle, separators=(",", ":"), allow_nan=False) + "\n")
    print(f"Wrote {output_json} ({output_json.stat().st_size / 1_000_000:.2f} MB)", flush=True)
    return bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path, default=APP_DATA)
    args = parser.parse_args()
    write_bundle(args.output_json)


if __name__ == "__main__":
    main()
