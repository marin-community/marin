# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "cvxpy",
#   "gcsfs",
#   "joblib",
#   "numpy",
#   "pandas",
#   "plotly",
#   "pyarrow",
#   "scikit-learn",
#   "scipy",
#   "tabulate",
# ]
# ///
"""Compare phase-invariant aggregate models under the same 280-run designs.

This audit isolates the source of the frontier-control gain in the strict
fixed-budget protocol. It fits three aggregate spines on exactly the rows
charged to each budget arm:

* physical pooled acquisition;
* the independently fitted one-phase Compact Retained State restriction;
* the independently fitted one-phase canonical DSP restriction.

Phase-treatment rows are never used by an aggregate fit. Evaluation uses the
same coordinate-disjoint two-phase development archive and source/anchor
clusters as the strict protocol. The currently sealed targeted pairwise panel
is absent from the heldout registry and is not accessed.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd
import plotly.express as px
from joblib import Parallel, delayed

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    audit_fixed_budget_pooled_acquisition_protocol_20260724 as strict_protocol,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_orthogonal_aggregate_phase_identification_20260724 as orthogonal,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_partially_pooled_phase_bowls as pooled,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_retained_weibull_replay_20260713 as compact_retained,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_tied_backbone_phase_order_20260724 as phase_benchmark,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    export_mixture_fit_observatory as observatory,
)

SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_OUTPUTS = SCRIPT_DIR / "reference_outputs"
DEFAULT_OUTPUT_DIR = REFERENCE_OUTPUTS / "fixed_budget_aggregate_comparators_20260724"
STRICT_OUTPUT_DIR = REFERENCE_OUTPUTS / "fixed_budget_pooled_acquisition_protocol_20260724"
DEFAULT_SEEDS = strict_protocol.DEFAULT_SEEDS
MODEL_WORKERS = 8
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}


class AggregatePredictor(Protocol):
    """Predict smooth BPB from a two-phase policy."""

    def predict(self, weights: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class CanonicalPredictor:
    """Canonical DSP fitted to the tied restriction."""

    model: Any
    dataset: pooled.Dataset

    def predict(self, weights: np.ndarray) -> np.ndarray:
        return observatory.dsp_predict(self.model, self.dataset, weights)


@dataclass(frozen=True)
class AggregateFit:
    """One aggregate fit, its grouped OOF prediction, and selection metadata."""

    model: AggregatePredictor
    oof_prediction: np.ndarray
    selected_l2: float | None
    parameter_count: int
    selection_rows: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class FitTask:
    """One target, budget arm, and deterministic subset seed."""

    target: str
    arm: strict_protocol.BudgetArm
    seed: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--bootstrap-draws", type=int, default=strict_protocol.BOOTSTRAP_DRAWS)
    parser.add_argument("--workers", type=int, default=MODEL_WORKERS)
    return parser.parse_args()


def task_grid(seeds: tuple[int, ...]) -> tuple[FitTask, ...]:
    return tuple(
        FitTask(target, arm, seed)
        for target in orthogonal.TARGETS
        for arm in strict_protocol.ARMS
        for seed in ((seeds[0],) if arm.name == "all_tied" else seeds)
    )


def target_data(
    target: str,
) -> tuple[
    pooled.Dataset,
    pd.DataFrame,
    np.ndarray,
    pooled.Dataset,
    pooled.Dataset,
    pd.DataFrame,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    reference = observatory.load_delphi_3e18_fit_dataset(target)
    heldout_frame, heldout_weights = observatory.load_delphi_3e18_heldouts(reference)
    single, _single_evaluation_indices = observatory.load_delphi_3e18_single_phase_dataset(
        target,
        reference,
        heldout_frame,
        heldout_weights,
    )
    controls = strict_protocol.fixed_budget.fiber_control_dataset(target, single)
    evaluation_frame, evaluation_weights, observed, positions = phase_benchmark.coordinate_disjoint_combined_rows(
        target,
        reference,
        single,
        heldout_frame,
        heldout_weights,
    )
    clusters = strict_protocol.evaluation_clusters(
        evaluation_frame,
        positions,
        reference,
        heldout_frame,
    )
    return (
        reference,
        heldout_frame,
        heldout_weights,
        single,
        controls,
        evaluation_frame,
        evaluation_weights,
        observed,
        clusters,
    )


def compact_fit(
    dataset: pooled.Dataset,
    fold: np.ndarray,
) -> AggregateFit:
    """Nested grouped-CV selection for the fitted one-phase CRS restriction."""
    all_indices = np.arange(dataset.n)
    candidates = []
    for l2 in observatory.COMPACT_L2_GRID:
        prediction = np.full(dataset.n, np.nan, dtype=float)
        for fold_index in range(strict_protocol.N_FOLDS):
            test = np.flatnonzero(fold == fold_index)
            train = np.setdiff1d(all_indices, test, assume_unique=True)
            model = observatory.compact_fit(dataset, train, l2, observatory.SINGLE_PHASE)
            prediction[test] = model.predict(dataset.weights[test])
        metrics = orthogonal.regression_metrics(dataset.y, prediction)
        candidates.append((float(l2), prediction, metrics))
    selected_l2, selected_prediction, _selected_metrics = min(
        candidates,
        key=lambda candidate: (
            float(candidate[2]["rmse"]),
            float(candidate[2]["regret_at_1"]),
            -float(candidate[2]["spearman"]),
            -candidate[0],
        ),
    )
    model = observatory.compact_fit(
        dataset,
        all_indices,
        selected_l2,
        observatory.SINGLE_PHASE,
    )
    rows = tuple(
        {
            "l2": l2,
            **metrics,
            "selected": l2 == selected_l2,
        }
        for l2, _prediction, metrics in candidates
    )
    return AggregateFit(
        model=model,
        oof_prediction=selected_prediction,
        selected_l2=selected_l2,
        parameter_count=compact_retained.nominal_parameter_count(
            dataset,
            observatory.compact_config(observatory.SINGLE_PHASE),
        ),
        selection_rows=rows,
    )


def canonical_fit(
    dataset: pooled.Dataset,
    fold: np.ndarray,
) -> AggregateFit:
    """Grouped OOF fit for canonical DSP's fitted tied restriction."""
    all_indices = np.arange(dataset.n)
    prediction = np.full(dataset.n, np.nan, dtype=float)
    for fold_index in range(strict_protocol.N_FOLDS):
        test = np.flatnonzero(fold == fold_index)
        train = np.setdiff1d(all_indices, test, assume_unique=True)
        model = observatory.dsp_fit(
            dataset,
            train,
            model_id="canonical",
            policy_class=observatory.SINGLE_PHASE,
        )
        prediction[test] = observatory.dsp_predict(model, dataset, dataset.weights[test])
    model = observatory.dsp_fit(
        dataset,
        all_indices,
        model_id="canonical",
        policy_class=observatory.SINGLE_PHASE,
    )
    return AggregateFit(
        model=CanonicalPredictor(model, dataset),
        oof_prediction=prediction,
        selected_l2=None,
        parameter_count=model.base.total_param_count,
        selection_rows=(
            {
                "l2": None,
                **orthogonal.regression_metrics(dataset.y, prediction),
                "selected": True,
            },
        ),
    )


def run_task(task: FitTask) -> tuple[list[dict[str, Any]], list[dict[str, Any]], pd.DataFrame]:
    (
        _reference,
        _heldout_frame,
        _heldout_weights,
        single,
        controls,
        evaluation_frame,
        evaluation_weights,
        observed,
        clusters,
    ) = target_data(task.target)
    training = strict_protocol.aggregate_training_dataset(
        task.target,
        single,
        controls,
        task.arm,
        task.seed,
    )
    fold = strict_protocol.grouped_stratified_folds(training, task.seed)
    fits = {
        "compact_retained_state_tied": compact_fit(training, fold),
        "canonical_dsp_tied": canonical_fit(training, fold),
    }
    metric_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    prediction_frames = []
    for model_name, fit in fits.items():
        oof_metrics = orthogonal.regression_metrics(training.y, fit.oof_prediction)
        prediction = fit.model.predict(evaluation_weights)
        heldout_metrics = orthogonal.regression_metrics(observed, prediction)
        metadata = {
            "target": task.target,
            "arm": task.arm.name,
            "seed": task.seed,
            **asdict(task.arm),
            "fit_rows": training.n,
            "aggregate_model": model_name,
            "selected_l2": fit.selected_l2,
            "parameter_count": fit.parameter_count,
        }
        metric_rows.append(
            {
                **metadata,
                **{f"oof_{key}": value for key, value in oof_metrics.items()},
                **heldout_metrics,
            }
        )
        for row in fit.selection_rows:
            selection_rows.append({**metadata, **row})
        local = evaluation_frame.copy()
        local["target"] = task.target
        local["cluster"] = clusters
        local["observed"] = observed
        local["aggregate_model"] = model_name
        local["arm"] = task.arm.name
        local["seed"] = task.seed
        local["predicted"] = prediction
        local["residual"] = prediction - observed
        prediction_frames.append(local)
    return metric_rows, selection_rows, pd.concat(prediction_frames, ignore_index=True)


def pooled_acquisition_rows(
    seeds: tuple[int, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    strict_metrics = pd.read_csv(STRICT_OUTPUT_DIR / "combined_metrics.csv")
    strict_selections = pd.read_csv(STRICT_OUTPUT_DIR / "aggregate_selections.csv")
    strict_predictions = pd.read_csv(STRICT_OUTPUT_DIR / "predictions.csv", low_memory=False)
    metric_rows = []
    prediction_frames = []
    for task in task_grid(seeds):
        model_key = strict_protocol.model_key(task.arm.name, task.seed, "phase_null")
        metric = strict_metrics[strict_metrics["target"].eq(task.target) & strict_metrics["model_key"].eq(model_key)]
        if len(metric) != 1:
            raise ValueError(f"Expected one pooled-acquisition metric row for {task}/{model_key}")
        row = metric.iloc[0].to_dict()
        selection = strict_selections[
            strict_selections["target"].eq(task.target)
            & strict_selections["arm"].eq(task.arm.name)
            & strict_selections["seed"].eq(task.seed)
        ]
        if len(selection) != 1:
            raise ValueError(f"Expected one pooled-acquisition selection row for {task}")
        selected = selection.iloc[0]
        row["aggregate_model"] = "physical_pooled_acquisition"
        row["fit_rows"] = int(selected["fit_rows"])
        row["oof_rmse"] = float(selected["oof_rmse"])
        row["oof_spearman"] = float(selected["oof_spearman"])
        row["selected_l2"] = float(selected["l2"])
        row["parameter_count"] = int(selected["active_parameter_count"])
        metric_rows.append(row)
        prediction = strict_predictions[
            strict_predictions["target"].eq(task.target) & strict_predictions["model_key"].eq(model_key)
        ].copy()
        if prediction.empty:
            raise ValueError(f"Missing pooled-acquisition predictions for {task}/{model_key}")
        prediction["aggregate_model"] = "physical_pooled_acquisition"
        prediction_frames.append(prediction)
    return pd.DataFrame(metric_rows), pd.concat(prediction_frames, ignore_index=True)


def observatory_rows() -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows = []
    prediction_frames = []
    for target in orthogonal.TARGETS:
        (
            _reference,
            _heldout_frame,
            _heldout_weights,
            _single,
            _controls,
            evaluation_frame,
            _evaluation_weights,
            observed,
            clusters,
        ) = target_data(target)
        source = pd.read_csv(STRICT_OUTPUT_DIR / "predictions.csv", low_memory=False)
        prediction = source[source["target"].eq(target) & source["model_key"].eq("observatory_compact_retained_state")][
            "predicted"
        ].to_numpy(dtype=float)
        if len(prediction) != len(observed):
            raise ValueError(f"Observatory prediction length mismatch for {target}")
        metadata = {
            "target": target,
            "arm": "observatory",
            "seed": -1,
            "tied_count": 0,
            "control_count": 0,
            "treatment_count": 0,
            "fit_rows": 280,
            "aggregate_model": "observatory_two_phase_crs",
            "selected_l2": np.nan,
            "parameter_count": np.nan,
        }
        metric_rows.append({**metadata, **orthogonal.regression_metrics(observed, prediction)})
        local = evaluation_frame.copy()
        local["target"] = target
        local["cluster"] = clusters
        local["observed"] = observed
        local["aggregate_model"] = "observatory_two_phase_crs"
        local["arm"] = "observatory"
        local["seed"] = -1
        local["predicted"] = prediction
        local["residual"] = prediction - observed
        prediction_frames.append(local)
    return pd.DataFrame(metric_rows), pd.concat(prediction_frames, ignore_index=True)


def prediction_slice(
    predictions: pd.DataFrame,
    target: str,
    aggregate_model: str,
    arm: str,
    seed: int,
) -> pd.DataFrame:
    selected = predictions[
        predictions["target"].eq(target)
        & predictions["aggregate_model"].eq(aggregate_model)
        & predictions["arm"].eq(arm)
        & predictions["seed"].eq(seed)
    ].copy()
    if selected.empty:
        raise KeyError(f"Missing predictions for {target}/{aggregate_model}/{arm}/{seed}")
    return selected


def bootstrap_contrasts(
    predictions: pd.DataFrame,
    seeds: tuple[int, ...],
    draws: int,
) -> pd.DataFrame:
    records = []
    model_names = (
        "physical_pooled_acquisition",
        "compact_retained_state_tied",
        "canonical_dsp_tied",
    )
    for target_index, target in enumerate(orthogonal.TARGETS):
        for model_index, model_name in enumerate(model_names):
            all_tied = prediction_slice(predictions, target, model_name, "all_tied", seeds[0])
            for seed_index, seed in enumerate(seeds):
                controls = prediction_slice(
                    predictions,
                    target,
                    model_name,
                    "frontier_controls_only",
                    seed,
                )
                records.append(
                    {
                        "target": target,
                        "aggregate_model": model_name,
                        "seed": seed,
                        "contrast": "frontier_controls_vs_all_tied",
                        **strict_protocol.cluster_bootstrap(
                            controls["observed"].to_numpy(dtype=float),
                            controls["predicted"].to_numpy(dtype=float),
                            all_tied["predicted"].to_numpy(dtype=float),
                            controls["cluster"].to_numpy(dtype=object),
                            draws,
                            20260724 + 1000 * target_index + 100 * model_index + seed_index,
                        ),
                    }
                )
                for arm_index, arm in enumerate(("frontier_controls_only", "phase_probe_32", "phase_probe_112")):
                    candidate = prediction_slice(predictions, target, model_name, arm, seed)
                    incumbent = prediction_slice(
                        predictions,
                        target,
                        "observatory_two_phase_crs",
                        "observatory",
                        -1,
                    )
                    records.append(
                        {
                            "target": target,
                            "aggregate_model": model_name,
                            "seed": seed,
                            "contrast": f"{arm}_vs_observatory_two_phase_crs",
                            **strict_protocol.cluster_bootstrap(
                                candidate["observed"].to_numpy(dtype=float),
                                candidate["predicted"].to_numpy(dtype=float),
                                incumbent["predicted"].to_numpy(dtype=float),
                                candidate["cluster"].to_numpy(dtype=object),
                                draws,
                                20260724 + 10_000 * target_index + 1000 * model_index + 100 * seed_index + arm_index,
                            ),
                        }
                    )
    return pd.DataFrame(records)


def plot_metrics(metrics: pd.DataFrame, output_path: Path) -> None:
    selected = metrics[metrics["arm"].ne("observatory")].copy()
    figure = px.line(
        selected,
        x="treatment_count",
        y="rmse",
        color="aggregate_model",
        line_dash="seed",
        markers=True,
        facet_col="target",
        hover_data=["arm", "oof_rmse", "spearman", "regret_at_1", "optimism_gt_0p05"],
        title="Strict 280-run budget: phase-invariant aggregate comparators",
        color_discrete_sequence=px.colors.diverging.RdYlGn_r,
    )
    figure.update_layout(template="plotly_white", width=1450, height=650)
    figure.write_html(output_path, include_plotlyjs="cdn", config=PLOT_CONFIG)


def write_report(
    output_dir: Path,
    metrics: pd.DataFrame,
    contrasts: pd.DataFrame,
) -> None:
    metric_columns = [
        "target",
        "aggregate_model",
        "arm",
        "seed",
        "fit_rows",
        "oof_rmse",
        "rmse",
        "spearman",
        "calibration_slope",
        "regret_at_1",
        "optimism_gt_0p05",
        "worst_optimism",
    ]
    lines = [
        "# Fixed-budget aggregate comparator audit",
        "",
        "## Question",
        "",
        (
            "Does the frontier-control gain identify a better aggregate equation, or does it reproduce across "
            "phase-invariant models because the controls improve local coverage and denoise the same two anchors?"
        ),
        "",
        "## Aggregate models",
        "",
        (
            r"Physical pooled acquisition uses \(q_i=c_i[\alpha_0w_i^{(0)}+\alpha_1w_i^{(1)}]\) and "
            r"\(L=b-\sum_i\beta_i(1-e^{-(\rho q_i)^p})-\sum_fB_f(1-e^{-(\rho Q_f)^p})\)."
        ),
        "",
        (
            "Compact Retained State and canonical DSP are both refit as independent one-phase restrictions on "
            "the identical charged rows. They receive no phase-treatment outcomes."
        ),
        "",
        "## Metrics",
        "",
        metrics[metric_columns]
        .sort_values(["target", "arm", "rmse"])
        .to_markdown(
            index=False,
            floatfmt=".6f",
        ),
        "",
        "## Source-and-anchor cluster bootstrap contrasts",
        "",
        contrasts.to_markdown(index=False, floatfmt=".6f"),
        "",
        (
            "A frontier-control effect that reproduces across all three forms is evidence about experimental "
            "design, not evidence for the pooled-acquisition equation. The Observatory comparator is the existing "
            "two-phase CRS fit and is never refit in this audit."
        ),
        "",
    ]
    (output_dir / "report.md").write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seeds = tuple(int(value) for value in args.seeds.split(",") if value)
    if not seeds:
        raise ValueError("At least one deterministic subset seed is required")

    results = Parallel(n_jobs=args.workers, backend="loky")(delayed(run_task)(task) for task in task_grid(seeds))
    metric_rows = [row for result in results for row in result[0]]
    selection_rows = [row for result in results for row in result[1]]
    prediction_frames = [result[2] for result in results]
    metrics = pd.DataFrame(metric_rows)
    selections = pd.DataFrame(selection_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)

    pooled_metrics, pooled_predictions = pooled_acquisition_rows(seeds)
    observatory_metrics, observatory_predictions = observatory_rows()
    metrics = pd.concat([metrics, pooled_metrics, observatory_metrics], ignore_index=True, sort=False)
    predictions = pd.concat(
        [predictions, pooled_predictions, observatory_predictions],
        ignore_index=True,
        sort=False,
    )
    contrasts = bootstrap_contrasts(predictions, seeds, int(args.bootstrap_draws))

    metrics.to_csv(args.output_dir / "aggregate_metrics.csv", index=False)
    selections.to_csv(args.output_dir / "aggregate_cv_selections.csv", index=False)
    predictions.to_csv(args.output_dir / "predictions.csv", index=False)
    contrasts.to_csv(args.output_dir / "cluster_bootstrap_contrasts.csv", index=False)
    plot_metrics(metrics, args.output_dir / "aggregate_budget_paths.html")
    write_report(args.output_dir, metrics, contrasts)
    (args.output_dir / "protocol.json").write_text(
        json.dumps(
            {
                "total_checkpoint_budget": strict_protocol.TOTAL_BUDGET,
                "arms": [asdict(arm) for arm in strict_protocol.ARMS],
                "seeds": seeds,
                "models": [
                    "physical_pooled_acquisition",
                    "compact_retained_state_tied",
                    "canonical_dsp_tied",
                ],
                "aggregate_hyperparameter_selection": "coordinate-grouped CV on charged rows only",
                "phase_treatment_rows_used_by_aggregate_fit": 0,
                "sealed_targeted_pairwise_panel_accessed": False,
                "evaluation_archive": "coordinate-disjoint two-phase development archive",
                "bootstrap_unit": "proposal series and phase anchor cluster",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
