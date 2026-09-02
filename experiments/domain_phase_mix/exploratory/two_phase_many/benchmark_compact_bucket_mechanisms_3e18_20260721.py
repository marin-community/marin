# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "cvxpy>=1.7",
#   "fsspec>=2025.7",
#   "gcsfs>=2025.7",
#   "numpy>=2.0",
#   "pandas>=2.2",
#   "plotly>=6.0",
#   "pyarrow>=20.0",
#   "scikit-learn>=1.6",
#   "scipy>=1.15",
#   "tabulate>=0.9",
# ]
# ///
"""Benchmark identifiable per-bucket mechanisms in Compact Retained State.

The baseline uses one shared learning clock and one shared replay-harm channel:

    z_i = exp(-lambda (1 - w_i^1)) e_i^0 + eta e_i^1
    L = b - sum_i a_i (1 - exp(-(rho z_i)^p))
          + c sum_i [e_i^0 + e_i^1 - 1]_+^2.

This benchmark nests the baseline and varies two mechanistically distinct
scopes. A partially pooled learning clock uses a target-independent exposure
prior,

    log rho_i = log rho + delta (u_i - mean(u)),
    u_i = -log median_train(z_i),

at either family or bucket scope. ``delta=0`` is exactly the shared clock. The
replay term is resolved at shared, family, or bucket scope while retaining
nonnegative coefficients. The original Compact ridge is frozen; only ``delta``
is selected inside each outer fold. This keeps any gain attributable to the
new mechanism rather than a larger hyperparameter search.

The primary screen uses 998 unique Delphi 3e18 policies and the existing
coordinate-disjoint two-phase development pool. No 300M outcomes enter fitting
or hyperparameter selection.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    analyze_delphi_phase_policy_sample_efficiency_20260721 as analysis,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_partially_pooled_phase_bowls as pooled,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_retained_weibull_replay_20260713 as compact,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    export_mixture_fit_observatory as observatory,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    plot_delphi_expanded_fit_raw_optima_20260721 as endpoint,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs/compact_bucket_mechanisms_3e18_20260721"
TARGETS = ("uncheatable", "table9")
OUTER_SEED = 20260721
INNER_SEED = 20260722
SPREAD_GRID = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0)
RATE_BOUNDS = (1e-4, 100.0)
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}, "responsive": True}


class ClockScope(StrEnum):
    SHARED = "shared"
    FAMILY = "family"
    BUCKET = "bucket"


class ReplayScope(StrEnum):
    SHARED = "shared"
    FAMILY = "family"
    BUCKET = "bucket"


@dataclass(frozen=True)
class Variant:
    clock: ClockScope
    replay: ReplayScope

    @property
    def name(self) -> str:
        return f"clock_{self.clock.value}__replay_{self.replay.value}"


@dataclass(frozen=True)
class FittedModel:
    variant: Variant
    shape: compact.Shape
    spread: float
    l2: float
    intercept: float
    signal_coef: np.ndarray
    replay_coef: np.ndarray
    rates: np.ndarray
    c0: np.ndarray
    c1: np.ndarray
    family_members: tuple[np.ndarray, ...]

    def predict(self, weights: np.ndarray) -> np.ndarray:
        signal, replay = features(
            np.asarray(weights, dtype=float),
            self.c0,
            self.c1,
            self.shape,
            self.rates,
            self.variant.replay,
            self.family_members,
        )
        return np.asarray(self.intercept - signal @ self.signal_coef + replay @ self.replay_coef, dtype=float)


VARIANTS = tuple(Variant(clock, replay) for clock in ClockScope for replay in ReplayScope)
BASELINE = Variant(ClockScope.SHARED, ReplayScope.SHARED)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--targets", default=",".join(TARGETS))
    parser.add_argument("--outer-seed", type=int, default=OUTER_SEED)
    parser.add_argument("--inner-seed", type=int, default=INNER_SEED)
    parser.add_argument("--maxiter", type=int, default=24)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def subset(dataset: pooled.Dataset, indices: np.ndarray, name: str) -> pooled.Dataset:
    return pooled.Dataset(
        name=name,
        frame=dataset.frame.iloc[indices].reset_index(drop=True),
        y=np.asarray(dataset.y[indices], dtype=float),
        weights=np.asarray(dataset.weights[indices], dtype=float),
        c0=np.asarray(dataset.c0, dtype=float),
        c1=np.asarray(dataset.c1, dtype=float),
        domain_names=list(dataset.domain_names),
    )


def retained_exposure(
    weights: np.ndarray,
    c0: np.ndarray,
    c1: np.ndarray,
    shape: compact.Shape,
) -> tuple[np.ndarray, np.ndarray]:
    phase0_weight = weights[:, 0, :]
    phase1_weight = weights[:, 1, :]
    early = phase0_weight * c0[None, :]
    late = phase1_weight * c1[None, :]
    retained_early = np.exp(-shape.forgetting_rate * (1.0 - phase1_weight)) * early
    return np.maximum(retained_early + shape.late_multiplier * late, 0.0), early + late


def centered_inverse_median(values: np.ndarray) -> np.ndarray:
    positive = np.where(values > 1e-10, values, np.nan)
    medians = np.nanmedian(positive, axis=0)
    finite = np.isfinite(medians) & (medians > 0.0)
    if not np.any(finite):
        raise ValueError("No positive retained exposure available for a learning-clock prior")
    fallback = float(np.median(medians[finite]))
    medians = np.where(finite, medians, fallback)
    log_inverse = -np.log(np.maximum(medians, 1e-10))
    return log_inverse - float(np.mean(log_inverse))


def clock_rates(
    dataset: pooled.Dataset,
    shape: compact.Shape,
    scope: ClockScope,
    spread: float,
    prior_indices: np.ndarray,
    family_members: tuple[np.ndarray, ...],
) -> np.ndarray:
    retained, _total = retained_exposure(dataset.weights[prior_indices], dataset.c0, dataset.c1, shape)
    if scope is ClockScope.SHARED:
        offsets = np.zeros(dataset.m, dtype=float)
    elif scope is ClockScope.BUCKET:
        offsets = spread * centered_inverse_median(retained)
    elif scope is ClockScope.FAMILY:
        family_retained = np.column_stack([retained[:, members].mean(axis=1) for members in family_members])
        family_offsets = spread * centered_inverse_median(family_retained)
        offsets = np.zeros(dataset.m, dtype=float)
        for family_index, members in enumerate(family_members):
            offsets[members] = family_offsets[family_index]
    else:
        raise ValueError(f"Unsupported clock scope {scope}")
    return np.clip(shape.rate * np.exp(offsets), *RATE_BOUNDS)


def replay_features(
    repeated_epochs: np.ndarray,
    scope: ReplayScope,
    family_members: tuple[np.ndarray, ...],
) -> np.ndarray:
    if scope is ReplayScope.SHARED:
        return repeated_epochs.sum(axis=1, keepdims=True)
    if scope is ReplayScope.FAMILY:
        return np.column_stack([repeated_epochs[:, members].sum(axis=1) for members in family_members])
    if scope is ReplayScope.BUCKET:
        return repeated_epochs
    raise ValueError(f"Unsupported replay scope {scope}")


def features(
    weights: np.ndarray,
    c0: np.ndarray,
    c1: np.ndarray,
    shape: compact.Shape,
    rates: np.ndarray,
    replay_scope: ReplayScope,
    family_members: tuple[np.ndarray, ...],
) -> tuple[np.ndarray, np.ndarray]:
    retained, total = retained_exposure(weights, c0, c1, shape)
    signal = -np.expm1(-((np.maximum(retained, 0.0) * rates[None, :]) ** shape.power))
    repeated = np.maximum(total - 1.0, 0.0) ** 2
    return signal, replay_features(repeated, replay_scope, family_members)


def fit_candidate(
    dataset: pooled.Dataset,
    indices: np.ndarray,
    shape: compact.Shape,
    variant: Variant,
    spread: float,
    l2: float,
    family_members: tuple[np.ndarray, ...],
) -> FittedModel:
    rates = clock_rates(dataset, shape, variant.clock, spread, indices, family_members)
    signal, replay = features(
        dataset.weights[indices],
        dataset.c0,
        dataset.c1,
        shape,
        rates,
        variant.replay,
        family_members,
    )
    design = np.hstack([-signal, replay])
    intercept, coefficients = compact.fit_nonnegative_head(design, dataset.y[indices], l2)
    return FittedModel(
        variant=variant,
        shape=shape,
        spread=spread,
        l2=l2,
        intercept=intercept,
        signal_coef=np.asarray(coefficients[: dataset.m], dtype=float),
        replay_coef=np.asarray(coefficients[dataset.m :], dtype=float),
        rates=rates,
        c0=np.asarray(dataset.c0, dtype=float),
        c1=np.asarray(dataset.c1, dtype=float),
        family_members=family_members,
    )


def spread_candidates(scope: ClockScope) -> tuple[float, ...]:
    return (0.0,) if scope is ClockScope.SHARED else SPREAD_GRID


def select_spread(
    dataset: pooled.Dataset,
    indices: np.ndarray,
    shape: compact.Shape,
    variant: Variant,
    l2: float,
    family_members: tuple[np.ndarray, ...],
    seed: int,
) -> tuple[float, pd.DataFrame]:
    local = subset(dataset, indices, f"{dataset.name}_inner")
    rows: list[dict[str, float]] = []
    for spread in spread_candidates(variant.clock):
        prediction = np.full(local.n, np.nan, dtype=float)
        for local_train, local_test in observatory.folds(local, seed):
            model = fit_candidate(
                local,
                local_train,
                shape,
                variant,
                spread,
                l2,
                family_members,
            )
            prediction[local_test] = model.predict(local.weights[local_test])
        if not np.isfinite(prediction).all():
            raise ValueError(f"Incomplete inner prediction for {dataset.name}/{variant.name}/{spread}")
        summary = analysis.metrics(local.y, prediction)
        rows.append(
            {
                "spread": float(spread),
                "rmse": float(summary["rmse"]),
                "regret_at_1": float(summary["regret_at_1"]),
                "calibration_error": float(summary["calibration_error"]),
            }
        )
    frame = pd.DataFrame(rows)
    selected = frame.sort_values(["rmse", "regret_at_1", "spread"]).iloc[0]
    return float(selected["spread"]), frame


def nominal_parameter_count(dataset: pooled.Dataset, variant: Variant) -> int:
    replay = {
        ReplayScope.SHARED: 1,
        ReplayScope.FAMILY: len(observatory.family_partition(dataset)[0]),
        ReplayScope.BUCKET: dataset.m,
    }[variant.replay]
    clock = int(variant.clock is not ClockScope.SHARED)
    return 1 + dataset.m + replay + 4 + clock


def prefixed_metrics(prefix: str, observed: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    return {f"{prefix}_{key}": value for key, value in analysis.metrics(observed, predicted).items()}


def target_paths(output_dir: Path, target: str) -> tuple[Path, Path, Path]:
    target_dir = output_dir / target
    return target_dir / "metrics.csv", target_dir / "selections.csv", target_dir / "predictions.npz"


def run_target(
    target: str,
    output_dir: Path,
    outer_seed: int,
    inner_seed: int,
    maxiter: int,
    top_k: int,
    force: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics_path, selections_path, predictions_path = target_paths(output_dir, target)
    if not force and metrics_path.exists() and selections_path.exists() and predictions_path.exists():
        return pd.read_csv(metrics_path), pd.read_csv(selections_path)

    train, evaluation, provenance = endpoint.endpoint_training_set(target)
    target_dir = metrics_path.parent
    target_dir.mkdir(parents=True, exist_ok=True)
    provenance.to_csv(target_dir / "training_provenance.csv", index=False)
    evaluation.frame.to_csv(target_dir / "evaluation_provenance.csv", index=False)
    family_names, family_members, _family_index = observatory.family_partition(train)
    if sorted(np.concatenate(family_members).tolist()) != list(range(train.m)):
        raise ValueError("Semantic families do not partition the Delphi domains")

    spec = analysis.frozen_spec(target, observatory.TWO_PHASE, "compact_retained_state")
    l2 = float(spec.tuning["l2"])
    config = observatory.COMPACT_TWO_PHASE_CONFIG
    predictions = {variant.name: np.full(train.n, np.nan, dtype=float) for variant in VARIANTS}
    selection_rows: list[dict[str, Any]] = []
    outer_folds = observatory.folds(train, outer_seed)

    for fold_index, (outer_train, outer_test) in enumerate(outer_folds):
        baseline_model = compact.fit_model(
            train,
            outer_train,
            config,
            l2,
            maxiter=maxiter,
            top_k=top_k,
        )
        for variant in VARIANTS:
            spread, sweep = select_spread(
                train,
                outer_train,
                baseline_model.shape,
                variant,
                l2,
                family_members,
                inner_seed + fold_index,
            )
            fitted = fit_candidate(
                train,
                outer_train,
                baseline_model.shape,
                variant,
                spread,
                l2,
                family_members,
            )
            predictions[variant.name][outer_test] = fitted.predict(train.weights[outer_test])
            selection_rows.append(
                {
                    "target": target,
                    "fit_scope": "outer_fold",
                    "fold": fold_index,
                    "variant": variant.name,
                    "clock_scope": variant.clock.value,
                    "replay_scope": variant.replay.value,
                    "selected_spread": spread,
                    "l2": l2,
                    **{f"shape_{key}": value for key, value in asdict(baseline_model.shape).items()},
                    "inner_sweep_json": sweep.to_json(orient="records"),
                }
            )
        pd.DataFrame(selection_rows).to_csv(selections_path, index=False)
        np.savez_compressed(predictions_path, **predictions)

    full_baseline = compact.fit_model(
        train,
        np.arange(train.n),
        config,
        l2,
        maxiter=maxiter,
        top_k=top_k,
    )
    metric_rows: list[dict[str, Any]] = []
    heldout_predictions: dict[str, np.ndarray] = {}
    for variant in VARIANTS:
        oof_prediction = predictions[variant.name]
        if not np.isfinite(oof_prediction).all():
            raise ValueError(f"Incomplete OOF prediction for {target}/{variant.name}")
        spread, sweep = select_spread(
            train,
            np.arange(train.n),
            full_baseline.shape,
            variant,
            l2,
            family_members,
            inner_seed + len(outer_folds),
        )
        fitted = fit_candidate(
            train,
            np.arange(train.n),
            full_baseline.shape,
            variant,
            spread,
            l2,
            family_members,
        )
        heldout_prediction = fitted.predict(evaluation.weights)
        heldout_predictions[variant.name] = heldout_prediction
        nonzero_replay = int(np.sum(fitted.replay_coef > 1e-10))
        metric_rows.append(
            {
                "target": target,
                "variant": variant.name,
                "clock_scope": variant.clock.value,
                "replay_scope": variant.replay.value,
                "selected_spread": spread,
                "l2": l2,
                "nominal_parameter_count": nominal_parameter_count(train, variant),
                "nonzero_signal_coefficients": int(np.sum(fitted.signal_coef > 1e-10)),
                "nonzero_replay_coefficients": nonzero_replay,
                "replay_coefficient_count": len(fitted.replay_coef),
                "rate_min": float(fitted.rates.min()),
                "rate_median": float(np.median(fitted.rates)),
                "rate_max": float(fitted.rates.max()),
                "rate_ratio": float(fitted.rates.max() / fitted.rates.min()),
                **prefixed_metrics("oof", train.y, oof_prediction),
                **prefixed_metrics(
                    "heldout",
                    evaluation.frame[analysis.TARGET_COLUMNS[target]].to_numpy(dtype=float),
                    heldout_prediction,
                ),
            }
        )
        selection_rows.append(
            {
                "target": target,
                "fit_scope": "full",
                "fold": -1,
                "variant": variant.name,
                "clock_scope": variant.clock.value,
                "replay_scope": variant.replay.value,
                "selected_spread": spread,
                "l2": l2,
                **{f"shape_{key}": value for key, value in asdict(full_baseline.shape).items()},
                "inner_sweep_json": sweep.to_json(orient="records"),
                "family_names_json": json.dumps(family_names),
                "rates_json": json.dumps(fitted.rates.tolist(), separators=(",", ":")),
                "signal_coefficients_json": json.dumps(fitted.signal_coef.tolist(), separators=(",", ":")),
                "replay_coefficients_json": json.dumps(fitted.replay_coef.tolist(), separators=(",", ":")),
            }
        )

    metrics = pd.DataFrame(metric_rows).sort_values("heldout_rmse").reset_index(drop=True)
    selections = pd.DataFrame(selection_rows)
    metrics.to_csv(metrics_path, index=False)
    selections.to_csv(selections_path, index=False)
    np.savez_compressed(
        predictions_path,
        **{f"oof__{name}": value for name, value in predictions.items()},
        **{f"heldout__{name}": value for name, value in heldout_predictions.items()},
    )
    return metrics, selections


def add_baseline_deltas(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target, target_frame in metrics.groupby("target", sort=False):
        baseline = target_frame[target_frame["variant"] == BASELINE.name]
        if len(baseline) != 1:
            raise ValueError(f"Expected one Compact baseline for {target}")
        reference = baseline.iloc[0]
        for row in target_frame.to_dict(orient="records"):
            for metric in (
                "oof_rmse",
                "oof_regret_at_1",
                "heldout_rmse",
                "heldout_regret_at_1",
                "heldout_calibration_error",
                "heldout_optimism_gt_0p05",
                "heldout_worst_optimism",
            ):
                row[f"delta_{metric}"] = float(row[metric]) - float(reference[metric])
            rows.append(row)
    return pd.DataFrame(rows)


def plot_metrics(metrics: pd.DataFrame, output_dir: Path) -> Path:
    fields = (
        ("heldout_rmse", "Heldout RMSE"),
        ("heldout_regret_at_1", "Heldout Regret@1"),
        ("heldout_calibration_slope", "Observed-on-predicted slope"),
        ("heldout_worst_optimism", "Worst optimism"),
    )
    figure = make_subplots(
        rows=2,
        cols=4,
        subplot_titles=[label for _target in TARGETS for _field, label in fields],
        row_titles=("Uncheatable", "Table-9"),
        vertical_spacing=0.18,
        horizontal_spacing=0.08,
    )
    for row_index, target in enumerate(TARGETS, start=1):
        local = metrics[metrics["target"] == target].sort_values("heldout_rmse")
        colors = ["#1a9850" if name == BASELINE.name else "#d73027" for name in local["variant"]]
        for column_index, (field, _label) in enumerate(fields, start=1):
            figure.add_trace(
                go.Bar(
                    x=local["variant"],
                    y=local[field],
                    marker_color=colors,
                    customdata=np.column_stack([local["oof_rmse"], local["selected_spread"]]),
                    hovertemplate=(
                        "%{x}<br>%{y:.6f}<br>OOF RMSE=%{customdata[0]:.6f}"
                        "<br>selected spread=%{customdata[1]:.3f}<extra></extra>"
                    ),
                    showlegend=False,
                ),
                row=row_index,
                col=column_index,
            )
            figure.update_xaxes(tickangle=-35, row=row_index, col=column_index)
    figure.update_layout(
        title={
            "text": (
                "Compact Retained State: per-bucket mechanism screen"
                "<br><span style='font-size:14px;color:#5f6b76'>"
                "998 Delphi 3e18 fit policies; coordinate-disjoint two-phase development pool.</span>"
            ),
            "x": 0.5,
            "xanchor": "center",
        },
        template="plotly_white",
        width=2200,
        height=1250,
        margin={"l": 130, "r": 100, "t": 140, "b": 260},
        paper_bgcolor="#fbfaf6",
    )
    path = output_dir / "mechanism_metrics.html"
    path.write_text(pio.to_html(figure, include_plotlyjs=True, full_html=True, config=PLOT_CONFIG))
    return path


def write_report(metrics: pd.DataFrame, selections: pd.DataFrame, output_dir: Path) -> Path:
    columns = [
        "variant",
        "selected_spread",
        "nominal_parameter_count",
        "oof_rmse",
        "heldout_rmse",
        "heldout_calibration_slope",
        "heldout_regret_at_1",
        "heldout_optimism_gt_0p05",
        "heldout_worst_optimism",
    ]
    sections = [
        "# Compact Retained State per-bucket mechanism screen",
        "",
        (
            "This is a development diagnostic. All fits use only 998 unique Delphi 3e18 policies; "
            "the coordinate-disjoint evaluation pool is never used to select clock dispersion or coefficients. "
            "The original Compact ridge remains frozen."
        ),
        "",
    ]
    for target in TARGETS:
        local = metrics[metrics["target"] == target].sort_values("heldout_rmse")
        baseline = local[local["variant"] == BASELINE.name].iloc[0]
        winner = local.iloc[0]
        fold_rows = selections[(selections["target"] == target) & (selections["fit_scope"] == "outer_fold")]
        spread_stability = fold_rows.groupby("variant")["selected_spread"].agg(["min", "median", "max"]).reset_index()
        sections.extend(
            [
                f"## {target}",
                "",
                local[columns].to_markdown(index=False, floatfmt=".6f"),
                "",
                (
                    f"Best heldout-RMSE variant: `{winner['variant']}` ({winner['heldout_rmse']:.6f}) "
                    f"versus exact shared Compact ({baseline['heldout_rmse']:.6f})."
                ),
                "",
                "Fold-level selected clock dispersion:",
                "",
                spread_stability.to_markdown(index=False, floatfmt=".3f"),
                "",
            ]
        )
    path = output_dir / "report.md"
    path.write_text("\n".join(sections) + "\n")
    return path


def main() -> None:
    args = parse_args()
    targets = tuple(value.strip() for value in args.targets.split(",") if value.strip())
    unknown = set(targets) - set(TARGETS)
    if unknown:
        raise ValueError(f"Unknown targets: {sorted(unknown)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    protocol = {
        "version": 1,
        "targets": list(targets),
        "fit_rows": 998,
        "fit_scale": "Delphi 3e18 only",
        "outer_seed": args.outer_seed,
        "inner_seed": args.inner_seed,
        "spread_grid": list(SPREAD_GRID),
        "ridge_rule": "frozen from the original Delphi 3e18 Compact Observatory fit",
        "shape_rule": "refit baseline Compact shape inside every outer fold",
        "selection_rule": "clock spread selected by inner RMSE; replay scope is a prespecified ablation",
        "evaluation_rule": "coordinate-disjoint two-phase development policies outside extension series",
    }
    protocol_path = args.output_dir / "protocol.json"
    if protocol_path.exists() and not args.force and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError("Existing protocol differs; use a new output directory or --force")
    protocol_path.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")

    metric_frames = []
    selection_frames = []
    for target in targets:
        metrics, selections = run_target(
            target,
            args.output_dir,
            args.outer_seed,
            args.inner_seed,
            args.maxiter,
            args.top_k,
            args.force,
        )
        metric_frames.append(metrics)
        selection_frames.append(selections)
    combined_metrics = add_baseline_deltas(pd.concat(metric_frames, ignore_index=True))
    combined_selections = pd.concat(selection_frames, ignore_index=True)
    combined_metrics.to_csv(args.output_dir / "metrics.csv", index=False)
    combined_selections.to_csv(args.output_dir / "selections.csv", index=False)
    plot_metrics(combined_metrics, args.output_dir)
    write_report(combined_metrics, combined_selections, args.output_dir)
    print(combined_metrics.sort_values(["target", "heldout_rmse"]).to_string(index=False))


if __name__ == "__main__":
    main()
