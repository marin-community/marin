# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "cvxpy",
#   "fsspec",
#   "gcsfs",
#   "matplotlib",
#   "numpy",
#   "pandas",
#   "plotly",
#   "pyarrow",
#   "scikit-learn",
#   "scipy",
#   "tabulate",
# ]
# ///
"""Mechanism-local ablations of retained power law.

The retained-power-law model currently mixes three questions:

1. what tied aggregate mixture is useful;
2. what early state survives to the end;
3. how much harm comes from repeated use of a finite pool.

This development benchmark changes one mechanism at a time while pinning the
parent's nonlinear shape and ridge. It is deliberately cheaper than the nested
audit: a failed mechanism should not consume a new 1,620-configuration search.

The paper-inspired repetition variants use additional epochs
``r(e) = max(e - 1, 0)`` rather than charging damage from zero exposure. The
phase-local variant charges the phase-weighted harm

    beta0 r(e0)^delta + beta1 r(e1)^delta,

where ``ep`` is a phase's epoch intensity. At a tied policy both intensities
equal the aggregate epoch count, so this reduces exactly to the fitted
single-phase damage term. Its excess over aggregate damage is a nonnegative
Jensen gap when ``delta >= 1``.

The normalized-retention variant measures retained state in tied-equivalent
token-share units. It divides the latent state by ``beta0 + eta * beta1``, so
the retained benefit is exactly independent of ``eta`` on every tied policy.
This is the low-capacity implementation of

    aggregate backbone + phase-only retained-state residual.

No append-only heldout outcomes are used unless ``--include-heldout`` is
explicitly passed after the mechanism set has been frozen.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import benchmark_retained_power_law_swarm39_20260728 as retained_benchmark  # noqa: E402
import benchmark_wsd80_incumbents_20260728 as wsd_benchmark  # noqa: E402
import fast_surrogate_iteration_20260729 as fast_iteration  # noqa: E402
import retained_power_law_model_20260728 as retained  # noqa: E402
import starcoder_wsd80_panel_20260728 as wsd80  # noqa: E402
import swarm39_harness_20260725 as harness  # noqa: E402

DEFAULT_OUTPUT_DIR = harness.REFERENCE_OUTPUTS / "factorized_retained_overfit_20260729"
PARENT_SCREEN = harness.REFERENCE_OUTPUTS / "fast_surrogate_iteration_20260729.csv"
TIED_TOLERANCE = 1e-10


class DamageMode(StrEnum):
    """How repeated exposure enters the damage block."""

    LEGACY_FROM_ZERO = "legacy_from_zero"
    AGGREGATE_AFTER_ONE = "aggregate_after_one"
    PHASE_LOCAL_AFTER_ONE = "phase_local_after_one"


class ExcessPooling(StrEnum):
    """How the phase-local excess repetition cost shares amplitudes."""

    NONE = "none"
    GLOBAL = "global"
    FAMILY = "family"


@dataclass(frozen=True)
class Variant:
    """One frozen mechanism-local ablation."""

    name: str
    normalize_retained_state: bool
    damage_mode: DamageMode
    include_concentration: bool
    ordering_override: bool | None
    excess_pooling: ExcessPooling = ExcessPooling.NONE


VARIANTS = (
    Variant("rpl_parent", False, DamageMode.LEGACY_FROM_ZERO, True, None),
    Variant("normalized_retention", True, DamageMode.LEGACY_FROM_ZERO, True, None),
    Variant("one_epoch_onset", False, DamageMode.AGGREGATE_AFTER_ONE, True, None),
    Variant("normalized_retention_onset", True, DamageMode.AGGREGATE_AFTER_ONE, True, None),
    Variant("phase_local_overfit", False, DamageMode.PHASE_LOCAL_AFTER_ONE, False, None),
    Variant("factorized_retained_overfit", True, DamageMode.PHASE_LOCAL_AFTER_ONE, False, None),
    Variant("factorized_retained_overfit_no_ordering", True, DamageMode.PHASE_LOCAL_AFTER_ONE, False, False),
    Variant(
        "additive_global_overfit_excess",
        False,
        DamageMode.LEGACY_FROM_ZERO,
        True,
        None,
        ExcessPooling.GLOBAL,
    ),
    Variant(
        "global_overfit_excess_replaces_concentration",
        False,
        DamageMode.LEGACY_FROM_ZERO,
        False,
        None,
        ExcessPooling.GLOBAL,
    ),
    Variant(
        "family_overfit_excess_replaces_concentration",
        False,
        DamageMode.LEGACY_FROM_ZERO,
        False,
        None,
        ExcessPooling.FAMILY,
    ),
)
PROMOTED_VARIANT = next(variant for variant in VARIANTS if variant.name == "additive_global_overfit_excess")


@dataclass(frozen=True)
class PinnedFit:
    """A fit whose nonlinear shape and ridge were inherited from the parent."""

    variant: Variant
    shape: retained.Shape
    ridge: float
    intercept: float
    coefficients: np.ndarray
    geometry: retained.Geometry

    def predict(self, weights: np.ndarray) -> np.ndarray:
        design = design_matrix(weights, self.geometry, self.shape, self.variant)
        return self.intercept + design @ self.coefficients


def _shape_with_mechanism(shape: retained.Shape, variant: Variant) -> retained.Shape:
    threshold = 0.0 if variant.damage_mode is DamageMode.LEGACY_FROM_ZERO else 1.0
    ordering = shape.ordering_channel if variant.ordering_override is None else variant.ordering_override
    return replace(shape, damage_threshold=threshold, ordering_channel=ordering)


def normalized_retained_share(
    weights: np.ndarray,
    geometry: retained.Geometry,
    retention: float,
    late_multiplier: float,
) -> np.ndarray:
    """Retained share in units where every tied policy has state equal to its mixture weight."""
    state = retained.retained_share(weights, geometry, retention, late_multiplier)
    tied_scale = geometry.phase_0_fraction + late_multiplier * geometry.phase_1_fraction
    assert tied_scale > 0.0
    return state / tied_scale


def phase_epoch_intensities(weights: np.ndarray, geometry: retained.Geometry) -> tuple[np.ndarray, np.ndarray]:
    """Epoch rates within each phase, normalized so tied policies have equal rates."""
    beta_0 = geometry.phase_0_fraction
    beta_1 = geometry.phase_1_fraction
    intensity_0 = geometry.c0 * weights[:, 0, :] / beta_0
    intensity_1 = geometry.c1 * weights[:, 1, :] / beta_1
    return intensity_0, intensity_1


def damage_values(
    weights: np.ndarray,
    geometry: retained.Geometry,
    exponent: float,
    mode: DamageMode,
) -> np.ndarray:
    """Per-bucket repetition-harm basis for one of the frozen mechanisms."""
    aggregate_epochs = retained.total_epochs(weights, geometry)
    if mode is DamageMode.LEGACY_FROM_ZERO:
        return aggregate_epochs**exponent
    if mode is DamageMode.AGGREGATE_AFTER_ONE:
        return np.maximum(aggregate_epochs - 1.0, 0.0) ** exponent

    intensity_0, intensity_1 = phase_epoch_intensities(weights, geometry)
    repeated_0 = np.maximum(intensity_0 - 1.0, 0.0) ** exponent
    repeated_1 = np.maximum(intensity_1 - 1.0, 0.0) ** exponent
    return geometry.phase_0_fraction * repeated_0 + geometry.phase_1_fraction * repeated_1


def phase_excess_repetition(
    weights: np.ndarray,
    geometry: retained.Geometry,
    exponent: float,
) -> np.ndarray:
    """Extra convex repetition harm caused by concentrating fixed exposure into phases."""
    aggregate = damage_values(
        weights,
        geometry,
        exponent,
        DamageMode.AGGREGATE_AFTER_ONE,
    )
    phase_local = damage_values(
        weights,
        geometry,
        exponent,
        DamageMode.PHASE_LOCAL_AFTER_ONE,
    )
    excess = phase_local - aggregate
    assert float(excess.min()) >= -1e-9, "convex phase-local repetition produced negative excess"
    return np.maximum(excess, 0.0)


def design_matrix(
    weights: np.ndarray,
    geometry: retained.Geometry,
    parent_shape: retained.Shape,
    variant: Variant,
) -> np.ndarray:
    """Build a design while changing only the requested mechanism."""
    shape = _shape_with_mechanism(parent_shape, variant)
    state = retained.retained_share(weights, geometry, shape.retention, shape.late_multiplier)
    if variant.normalize_retained_state:
        state = normalized_retained_share(weights, geometry, shape.retention, shape.late_multiplier)
    benefit = (state + shape.benefit_offset) ** (-shape.benefit_exponent)
    damage = damage_values(weights, geometry, shape.damage_exponent, variant.damage_mode)

    blocks = [
        retained._hierarchical_block(benefit, geometry),
        retained._hierarchical_block(damage, geometry),
    ]
    if variant.include_concentration:
        blocks.append(retained._signed(retained.concentration_gap(weights, geometry)))
    if variant.excess_pooling is ExcessPooling.GLOBAL:
        blocks.append(phase_excess_repetition(weights, geometry, shape.damage_exponent).sum(axis=1, keepdims=True))
    elif variant.excess_pooling is ExcessPooling.FAMILY:
        blocks.append(
            retained._family_totals(
                phase_excess_repetition(weights, geometry, shape.damage_exponent),
                geometry,
            )
        )
    if shape.ordering_channel:
        blocks.append(retained.marginal_phase_block(weights, geometry, shape))
    return np.column_stack(blocks)


def penalty_multipliers(
    geometry: retained.Geometry,
    parent_shape: retained.Shape,
    variant: Variant,
) -> np.ndarray:
    """Parent hierarchical prior, adjusted only for columns removed by an ablation."""
    families = len(np.unique(geometry.families))
    excess = len(geometry.excess_domains)
    hierarchical = np.concatenate([np.zeros(families), np.ones(excess)])
    blocks = [hierarchical, hierarchical]
    if variant.include_concentration:
        blocks.append(np.zeros(2))
    if variant.excess_pooling is ExcessPooling.GLOBAL:
        blocks.append(np.zeros(1))
    elif variant.excess_pooling is ExcessPooling.FAMILY:
        blocks.append(np.zeros(families))
    shape = _shape_with_mechanism(parent_shape, variant)
    if shape.ordering_channel:
        blocks.append(np.concatenate([np.ones(4 * families), np.zeros(2)]))
    return np.concatenate(blocks)


def fit_pinned(
    weights: np.ndarray,
    target: np.ndarray,
    geometry: retained.Geometry,
    shape: retained.Shape,
    ridge: float,
    variant: Variant,
) -> PinnedFit:
    design = design_matrix(weights, geometry, shape, variant)
    intercept, coefficients = retained.solve_head(
        design,
        target,
        ridge,
        penalty_multipliers(geometry, shape, variant),
    )
    return PinnedFit(
        variant=variant,
        shape=shape,
        ridge=ridge,
        intercept=intercept,
        coefficients=coefficients,
        geometry=geometry,
    )


def _finite_target(panel: harness.Panel, target: str) -> tuple[harness.Panel, np.ndarray]:
    observed = panel.targets[target]
    usable = np.isfinite(observed)
    panel = panel.subset(usable)
    return panel, panel.targets[target]


def _error_metrics(prefix: str, observed: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    residual = predicted - observed
    return {
        f"{prefix}_rows": len(observed),
        f"{prefix}_rmse": float(np.sqrt(np.mean(residual**2))),
        f"{prefix}_median_absolute": float(np.median(np.abs(residual))),
        f"{prefix}_bias": float(np.mean(residual)),
        f"{prefix}_spearman": float(spearmanr(observed, predicted).statistic),
    }


def _policy_slice_metrics(panel: harness.Panel, observed: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    tied = panel.phase_tv <= TIED_TOLERANCE
    moved = ~tied
    metrics: dict[str, float | int] = {}
    for name, mask in (("tied", tied), ("moved", moved)):
        if mask.any():
            metrics.update(_error_metrics(name, observed[mask], predicted[mask]))

    if moved.sum() >= 4:
        moved_indices = np.flatnonzero(moved)
        order = moved_indices[np.argsort(panel.phase_tv[moved])]
        for quartile, indices in enumerate(np.array_split(order, 4), start=1):
            metrics.update(_error_metrics(f"moved_q{quartile}", observed[indices], predicted[indices]))
    return metrics


def _excess_diagnostics(
    weights: np.ndarray,
    geometry: retained.Geometry,
    shape: retained.Shape,
    variant: Variant,
    coefficients: np.ndarray,
) -> dict[str, float | int]:
    """Report whether a repetition-excess block is distinct and active."""
    if variant.excess_pooling is ExcessPooling.NONE:
        return {}

    excess = phase_excess_repetition(weights, geometry, shape.damage_exponent)
    if variant.excess_pooling is ExcessPooling.GLOBAL:
        pooled = excess.sum(axis=1, keepdims=True)
    else:
        pooled = retained._family_totals(excess, geometry)
    concentration = retained.concentration_gap(weights, geometry)
    correlations = [
        abs(float(np.corrcoef(pooled[:, index], concentration)[0, 1]))
        for index in range(pooled.shape[1])
        if np.std(pooled[:, index]) > 0.0
    ]

    hierarchical_columns = len(np.unique(geometry.families)) + len(geometry.excess_domains)
    start = 2 * hierarchical_columns + (2 if variant.include_concentration else 0)
    excess_coefficients = coefficients[start : start + pooled.shape[1]]
    return {
        "excess_max_abs_concentration_correlation": max(correlations, default=float("nan")),
        "excess_active_coefficients": int(np.count_nonzero(excess_coefficients > 1e-8)),
        "excess_max_coefficient": float(np.max(excess_coefficients)),
    }


def parent_shape_from_screen(scale: str, target: str) -> tuple[retained.Shape, float]:
    """Load a previously selected full-grid parent without reselecting it on an ablation."""
    frame = pd.read_csv(PARENT_SCREEN)
    row = frame[(frame["scale"] == scale) & (frame["target"] == target)]
    if len(row) != 1:
        raise ValueError(
            f"{PARENT_SCREEN} has {len(row)} parent rows for {scale}/{target}; "
            "run fast_surrogate_iteration_20260729.py for that cell first"
        )
    payload = json.loads(row.iloc[0]["shape"])
    return retained_benchmark._shape_of(payload), float(row.iloc[0]["ridge"])


def build_promoted_design(panel: harness.Panel, shape: dict) -> harness.Design:
    """Swarm-harness adapter for joint selection of the promoted one-coefficient variant."""
    parameters = retained_benchmark._shape_of(shape)
    geometry = retained_benchmark.geometry_of(panel)
    matrix = design_matrix(
        retained_benchmark.weights_of(panel),
        geometry,
        parameters,
        PROMOTED_VARIANT,
    )
    return harness.Design(
        matrix=matrix,
        names=tuple(f"column_{index}" for index in range(matrix.shape[1])),
    )


def promoted_penalty_scale(panel: harness.Panel, shape: dict) -> np.ndarray:
    return penalty_multipliers(
        retained_benchmark.geometry_of(panel),
        retained_benchmark._shape_of(shape),
        PROMOTED_VARIANT,
    )


def promoted_model() -> harness.Model:
    return harness.Model(
        name=PROMOTED_VARIANT.name,
        build=build_promoted_design,
        shapes=retained_benchmark.shapes,
        l2_grid=retained.RIDGE_GRID,
        penalty_scale=promoted_penalty_scale,
        head=retained.solve_head,
    )


def joint_grid_screen(
    scale: str,
    target: str,
    splits: int,
    seed: int,
    workers: int,
    output_dir: Path,
) -> None:
    """Run the full parent shape grid only for the promoted mechanism."""
    panel, _heldout = harness.load_scale(scale)
    fit = fast_iteration.fit_model(
        panel,
        promoted_model(),
        target,
        n_splits=splits,
        seed=seed,
        workers=workers,
    )
    payload = {
        "scale": scale,
        "target": target,
        "variant": PROMOTED_VARIANT.name,
        "splits": splits,
        "seed": seed,
        "selection_oof_rmse": fit.oof_rmse,
        "shape": fit.shape,
        "ridge": fit.l2,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"joint_grid_{scale}__{target.replace('_bpb', '')}.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"wrote {output}")


def algebraic_audit(panel: harness.Panel, shape: retained.Shape) -> None:
    """Verify the parent reproduction and the two promised tied invariants."""
    geometry = retained_benchmark.geometry_of(panel)
    weights = retained_benchmark.weights_of(panel)
    parent = VARIANTS[0]
    expected = retained.design_matrix(weights, geometry, shape)
    actual = design_matrix(weights, geometry, shape, parent)
    assert np.allclose(actual, expected), "the parent adapter does not reproduce retained power law"

    tied_weights = np.stack([panel.aggregate, panel.aggregate], axis=1)
    normalized = normalized_retained_share(
        tied_weights,
        geometry,
        shape.retention,
        shape.late_multiplier,
    )
    assert np.allclose(normalized, panel.aggregate, atol=1e-10), "normalized state is not tied-neutral"

    aggregate = damage_values(
        tied_weights,
        geometry,
        shape.damage_exponent,
        DamageMode.AGGREGATE_AFTER_ONE,
    )
    phase_local = damage_values(
        tied_weights,
        geometry,
        shape.damage_exponent,
        DamageMode.PHASE_LOCAL_AFTER_ONE,
    )
    assert np.allclose(aggregate, phase_local, atol=1e-10), "phase-local damage changes the tied restriction"


def evaluate_swarm39(
    scale: str,
    target: str,
    splits: int,
    seed: int,
    include_heldout: bool,
    variants: tuple[Variant, ...],
    shape_source: Path | None,
) -> pd.DataFrame:
    fit_panel, heldout_panel = harness.load_scale(scale)
    fit_panel, observed = _finite_target(fit_panel, target)
    if shape_source is None:
        shape, ridge = parent_shape_from_screen(scale, target)
    else:
        payload = json.loads(shape_source.read_text())
        if payload["scale"] != scale or payload["target"] != target:
            raise ValueError(
                f"{shape_source} selects {payload['scale']}/{payload['target']}, " f"not requested {scale}/{target}"
            )
        shape = retained_benchmark._shape_of(payload["shape"])
        ridge = float(payload["ridge"])
    algebraic_audit(fit_panel, shape)
    geometry = retained_benchmark.geometry_of(fit_panel)
    weights = retained_benchmark.weights_of(fit_panel)
    folds = harness.grouped_splits(fit_panel, splits, seed)

    rows = []
    for variant in variants:
        predictions = np.full(len(observed), np.nan)
        for train, test in folds:
            fit = fit_pinned(weights[train], observed[train], geometry, shape, ridge, variant)
            predictions[test] = fit.predict(weights[test])
        assert np.all(np.isfinite(predictions))
        full_fit = fit_pinned(weights, observed, geometry, shape, ridge, variant)
        row: dict[str, object] = {
            "scale": scale,
            "target": target,
            "variant": variant.name,
            "shape": json.dumps(shape.__dict__, sort_keys=True),
            "ridge": ridge,
            "columns": len(full_fit.coefficients),
            **_error_metrics("oof", observed, predictions),
            **_policy_slice_metrics(fit_panel, observed, predictions),
            **_excess_diagnostics(weights, geometry, shape, variant, full_fit.coefficients),
        }
        if include_heldout:
            heldout_panel, heldout_observed = _finite_target(heldout_panel, target)
            heldout_weights = retained_benchmark.weights_of(heldout_panel)
            heldout_predictions = full_fit.predict(heldout_weights)
            row.update(_error_metrics("heldout", heldout_observed, heldout_predictions))
            row.update(
                {
                    f"heldout_{key}": value
                    for key, value in harness.metric_row(heldout_observed, heldout_predictions).items()
                }
            )
            heldout_slices = _policy_slice_metrics(heldout_panel, heldout_observed, heldout_predictions)
            row.update({f"heldout_{key}": value for key, value in heldout_slices.items()})
        rows.append(row)
        print(
            f"{variant.name:<43} OOF {row['oof_rmse']:.6f}  "
            f"tied {row.get('tied_rmse', float('nan')):.6f}  "
            f"moved {row.get('moved_rmse', float('nan')):.6f}",
            flush=True,
        )
    return pd.DataFrame(rows)


def _wsd80_parent(
    weights: np.ndarray,
    target: np.ndarray,
    geometry: retained.Geometry,
    folds: tuple[tuple[np.ndarray, np.ndarray], ...],
    cache: Path,
) -> tuple[retained.Shape, float]:
    """Select the parent once, then cache only its nonlinear shape and ridge."""
    if cache.exists():
        payload = json.loads(cache.read_text())
        return retained.Shape(**payload["shape"]), float(payload["ridge"])

    parent = retained.fit(weights, target, geometry, folds)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(
        json.dumps(
            {
                "fold_protocol": "five blocked mixture-space folds",
                "shape": parent.shape.__dict__,
                "ridge": parent.ridge,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return parent.shape, parent.ridge


def evaluate_wsd80(output_dir: Path, variants: tuple[Variant, ...]) -> pd.DataFrame:
    """Pinned-mechanism screen on the dense 80/20 WSD StarCoder surface."""
    panel = wsd80.load_surface()
    replicates = wsd80.load_fiber_replicates()
    sigma = wsd80.training_seed_sigma(replicates)
    data = wsd_benchmark.as_pooled_dataset(panel)
    weights = data.weights
    target = data.y
    indices = np.arange(len(target))
    index_folds = wsd_benchmark.mixture_blocked_folds(weights, indices, 5, 100)
    bool_folds = tuple(
        (
            np.isin(indices, train),
            np.isin(indices, test),
        )
        for train, test in index_folds
    )
    geometry = retained.Geometry(
        c0=data.c0,
        c1=data.c1,
        phase_0_fraction=wsd80.REALIZED_PHASE_0_FRACTION,
    )
    cache = output_dir / "wsd80_blocked_parent.json"
    shape, ridge = _wsd80_parent(weights, target, geometry, bool_folds, cache)

    measured = {
        aggregate: wsd_benchmark.measured_phase_gain(panel, aggregate) for aggregate in wsd_benchmark.FIBER_AGGREGATES
    }
    rows = []
    for variant in variants:
        predictions = np.full(len(target), np.nan)
        for train, test in index_folds:
            fit = fit_pinned(weights[train], target[train], geometry, shape, ridge, variant)
            predictions[test] = fit.predict(weights[test])
        assert np.all(np.isfinite(predictions))
        full_fit = fit_pinned(weights, target, geometry, shape, ridge, variant)
        predict = full_fit.predict
        optimum = wsd_benchmark.predicted_optimum(predict, wsd_benchmark.OPTIMUM_GRID)
        advantage = wsd_benchmark.two_phase_advantage(predict, wsd_benchmark.OPTIMUM_GRID)
        tied = wsd_benchmark.tied_diagonal_fit(predict, panel, wsd_benchmark.OPTIMUM_GRID)
        phase_profile = {
            aggregate: wsd_benchmark.predicted_phase_gain(
                predict,
                aggregate,
                wsd_benchmark.OPTIMUM_GRID,
            )
            for aggregate in wsd_benchmark.FIBER_AGGREGATES
        }
        nonzero = [
            aggregate for aggregate in wsd_benchmark.FIBER_AGGREGATES if abs(measured[aggregate]["best_contrast"]) > 1e-9
        ]
        signs = sum(
            np.sign(phase_profile[aggregate]["best_contrast"]) == np.sign(measured[aggregate]["best_contrast"])
            for aggregate in nonzero
        )
        profile_rmse = float(
            np.sqrt(
                np.mean(
                    [
                        (phase_profile[aggregate]["phase_gain"] - measured[aggregate]["phase_gain"]) ** 2
                        for aggregate in wsd_benchmark.FIBER_AGGREGATES
                    ]
                )
            )
        )
        residual = predictions - target
        row: dict[str, object] = {
            "panel": "starcoder_wsd80",
            "variant": variant.name,
            "shape": json.dumps(shape.__dict__, sort_keys=True),
            "ridge": ridge,
            "columns": len(full_fit.coefficients),
            "blocked_oof_rmse": float(np.sqrt(np.mean(residual**2))),
            "blocked_oof_rmse_sigma": float(np.sqrt(np.mean(residual**2))) / sigma,
            "blocked_oof_median_sigma": float(np.median(np.abs(residual))) / sigma,
            "blocked_oof_spearman": float(spearmanr(target, predictions).statistic),
            "phase_gain_profile_rmse": profile_rmse,
            "phase_gain_signs": signs,
            "phase_gain_signs_total": len(nonzero),
            **optimum,
            **advantage,
            **tied,
        }
        rows.append(row)
        print(
            f"{variant.name:<43} blocked {row['blocked_oof_rmse_sigma']:.3f} sigma  "
            f"optimum ({row['phase_0']:.3f}, {row['phase_1']:.3f})  "
            f"gain {row['predicted_two_phase_gain']:.6f}",
            flush=True,
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", choices=("swarm39", "wsd80"), default="swarm39")
    parser.add_argument("--scale", choices=("60m", "delphi_3e18"), default="delphi_3e18")
    parser.add_argument(
        "--target",
        choices=(harness.UNCHEATABLE, harness.TABLE9),
        default=harness.UNCHEATABLE,
    )
    parser.add_argument("--splits", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--joint-grid", action="store_true")
    parser.add_argument("--include-heldout", action="store_true")
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=tuple(variant.name for variant in VARIANTS),
        help="Evaluate only these frozen variants. Defaults to the complete ablation ladder.",
    )
    parser.add_argument(
        "--shape-json",
        type=Path,
        help="Use a previously selected joint-grid shape and ridge instead of the parent screen.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    if args.splits < 2:
        raise ValueError("--splits must be at least two")
    if args.workers < 1:
        raise ValueError("--workers must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected_names = set(args.variants or (variant.name for variant in VARIANTS))
    variants = tuple(variant for variant in VARIANTS if variant.name in selected_names)
    if args.joint_grid:
        if args.panel != "swarm39":
            raise ValueError("--joint-grid currently applies only to 39-bucket panels")
        if args.include_heldout:
            raise ValueError("--joint-grid does not inspect heldouts")
        joint_grid_screen(
            scale=args.scale,
            target=args.target,
            splits=args.splits,
            seed=args.seed,
            workers=args.workers,
            output_dir=args.output_dir,
        )
        return
    if args.panel == "wsd80":
        if args.include_heldout:
            raise ValueError("the WSD80 panel has no separate heldout archive")
        table = evaluate_wsd80(args.output_dir, variants)
        suffix = "starcoder_wsd80"
    else:
        table = evaluate_swarm39(
            scale=args.scale,
            target=args.target,
            splits=args.splits,
            seed=args.seed,
            include_heldout=args.include_heldout,
            variants=variants,
            shape_source=args.shape_json,
        )
        suffix = f"{args.scale}__{args.target.replace('_bpb', '')}"
        if args.shape_json is not None:
            suffix += "__selected_shape"
        if args.include_heldout:
            suffix += "__with_development_heldouts"
    output = args.output_dir / f"pinned_ablation_{suffix}.csv"
    table.to_csv(output, index=False)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
