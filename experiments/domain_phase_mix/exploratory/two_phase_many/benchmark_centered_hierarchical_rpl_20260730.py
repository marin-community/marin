# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scikit-learn", "scipy"]
# ///
"""Nested 300M comparison of centered hierarchical RPL against repaired RPL."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import benchmark_aggregate_conditioned_replay_control_20260730 as benchmark  # noqa: E402

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    aggregate_conditioned_replay_control_20260730 as replay_control,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    centered_hierarchical_rpl_20260730 as candidate,
)

DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "centered_hierarchical_rpl_physical_20260730"
RPL_ROOT = SCRIPT_DIR / "reference_outputs" / "rpl_repaired_baseline_screen_20260730"
RPL_PREDICTIONS = {
    "uncheatable": RPL_ROOT / "diagnostic_300m_uncheatable" / "predictions_300m.csv",
    "table9": RPL_ROOT / "diagnostic_300m_table9" / "predictions_300m.csv",
}
RPL_PARAMETERS = {
    "uncheatable": RPL_ROOT / "diagnostic_300m_uncheatable" / "parameters_300m.csv",
    "table9": RPL_ROOT / "diagnostic_300m_table9" / "parameters_300m.csv",
}
BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 20260730


def exact_pair_indices(
    frame: pd.DataFrame,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    indexed = frame.reset_index().set_index(["phase_correspondence_key", "policy_family"])["index"]
    keys = sorted(
        set(frame.loc[frame["policy_family"].eq("single_phase"), "phase_correspondence_key"])
        & set(frame.loc[frame["policy_family"].eq("two_phase"), "phase_correspondence_key"])
    )
    one = np.asarray([indexed.loc[(key, "single_phase")] for key in keys], dtype=int)
    two = np.asarray([indexed.loc[(key, "two_phase")] for key in keys], dtype=int)
    asymmetric = ~replay_control.tied_rows(weights[two])
    return one[asymmetric], two[asymmetric]


def rmse(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(values**2)))


def bootstrap_pair_difference(
    candidate_error: np.ndarray,
    baseline_error: np.ndarray,
) -> dict[str, float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = rng.integers(0, len(candidate_error), size=(BOOTSTRAP_SAMPLES, len(candidate_error)))
    candidate_rmse = np.sqrt(np.mean(candidate_error[samples] ** 2, axis=1))
    baseline_rmse = np.sqrt(np.mean(baseline_error[samples] ** 2, axis=1))
    differences = candidate_rmse - baseline_rmse
    return {
        "difference": rmse(candidate_error) - rmse(baseline_error),
        "bootstrap_se": float(np.std(differences, ddof=1)),
        "ci_low": float(np.quantile(differences, 0.025)),
        "ci_high": float(np.quantile(differences, 0.975)),
        "probability_candidate_better": float(np.mean(differences < 0.0)),
    }


def bootstrap_grouped_difference(
    candidate_error: np.ndarray,
    baseline_error: np.ndarray,
    groups: np.ndarray,
) -> dict[str, float]:
    unique = np.unique(groups)
    rows_by_group = [np.flatnonzero(groups == group) for group in unique]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    differences = np.empty(BOOTSTRAP_SAMPLES)
    for sample_id in range(BOOTSTRAP_SAMPLES):
        sampled_groups = rng.integers(0, len(unique), size=len(unique))
        rows = np.concatenate([rows_by_group[index] for index in sampled_groups])
        differences[sample_id] = rmse(candidate_error[rows]) - rmse(baseline_error[rows])
    return {
        "difference": rmse(candidate_error) - rmse(baseline_error),
        "bootstrap_se": float(np.std(differences, ddof=1)),
        "ci_low": float(np.quantile(differences, 0.025)),
        "ci_high": float(np.quantile(differences, 0.975)),
        "probability_candidate_better": float(np.mean(differences < 0.0)),
    }


def frozen_rpl_prediction(target: str, dataset: benchmark.Dataset) -> np.ndarray:
    baseline = pd.read_csv(RPL_PREDICTIONS[target])
    baseline = baseline.loc[baseline["seed"].eq(0)].set_index("run_name")
    if set(baseline.index) != set(dataset.frame["run_name"]):
        raise ValueError(f"repaired RPL rows do not match 300m_{target}")
    return dataset.frame["run_name"].map(baseline["predicted"]).to_numpy(dtype=float)


def fit_target(
    target: str,
    output_dir: Path,
    workers: int,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    summary_path = output_dir / f"summary_300m_{target}.json"
    prediction_path = output_dir / f"predictions_300m_{target}.csv"
    parameter_path = output_dir / f"parameters_300m_{target}.csv"
    if summary_path.exists() and prediction_path.exists() and parameter_path.exists():
        print(f"300m_{target}: reusing completed target", flush=True)
        return (
            json.loads(summary_path.read_text()),
            pd.read_csv(prediction_path),
            pd.read_csv(parameter_path),
        )

    dataset = benchmark.load_300m(target)
    geometry = benchmark.geometry_300m(dataset)
    folds = benchmark.grouped_folds(dataset.frame, seed=0, n_splits=3)
    partial_prediction_path = output_dir / f"partial_predictions_300m_{target}.npy"
    partial_parameter_path = output_dir / f"partial_parameters_300m_{target}.csv"
    prediction = np.load(partial_prediction_path) if partial_prediction_path.exists() else np.full(dataset.n, np.nan)
    parameters = pd.read_csv(partial_parameter_path).to_dict("records") if partial_parameter_path.exists() else []
    for fold_id, (train, test) in enumerate(folds):
        if np.all(np.isfinite(prediction[test])):
            print(f"300m_{target}: fold {fold_id + 1}/{len(folds)} already complete", flush=True)
            continue
        print(f"300m_{target}: fold {fold_id + 1}/{len(folds)}", flush=True)
        training = dataset.frame.iloc[train].reset_index(drop=True)
        inner_folds = benchmark.local_folds(training, seed=10_000 + fold_id, n_splits=3)
        fitted = candidate.fit(
            dataset.weights[train],
            dataset.y[train],
            geometry,
            inner_folds,
            workers=workers,
        )
        prediction[test] = fitted.predict(dataset.weights[test])
        parameters.append(
            {
                "dataset": f"300m_{target}",
                "seed": 0,
                "fold": fold_id,
                "model": "centered_hierarchical_rpl_physical",
                "benefit_exponent": fitted.shape.benefit_exponent,
                "benefit_offset": fitted.shape.benefit_offset,
                "damage_exponent": fitted.shape.damage_exponent,
                "ridge": fitted.ridge,
                "retention": fitted.shape.retention,
                "late_multiplier": fitted.shape.late_multiplier,
                "ordering_channel": int(fitted.shape.ordering_channel),
            }
        )
        np.save(partial_prediction_path, prediction)
        pd.DataFrame(parameters).to_csv(partial_parameter_path, index=False)
    if not np.all(np.isfinite(prediction)):
        raise ValueError(f"centered hierarchy did not predict every 300m_{target} row")

    metric = benchmark.metric_row(
        f"300m_{target}",
        "centered_hierarchical_rpl_physical",
        0,
        dataset.y,
        prediction,
        dataset.weights,
        folds,
    )
    pair_metric = benchmark.paired_metric_row(
        f"300m_{target}",
        "centered_hierarchical_rpl_physical",
        0,
        dataset.y,
        prediction,
        dataset.frame,
        dataset.weights,
    )
    baseline_prediction = frozen_rpl_prediction(target, dataset)
    one, two = exact_pair_indices(dataset.frame, dataset.weights)
    observed_delta = dataset.y[two] - dataset.y[one]
    candidate_delta = prediction[two] - prediction[one]
    baseline_delta = baseline_prediction[two] - baseline_prediction[one]
    pair_bootstrap = bootstrap_pair_difference(
        candidate_delta - observed_delta,
        baseline_delta - observed_delta,
    )
    grouped_bootstrap = bootstrap_grouped_difference(
        prediction - dataset.y,
        baseline_prediction - dataset.y,
        dataset.frame["phase_correspondence_key"].astype(str).to_numpy(),
    )
    fold_pair_metrics = []
    for fold_id, (_train, test) in enumerate(folds):
        in_test = np.isin(two, test)
        if not np.array_equal(in_test, np.isin(one, test)):
            raise ValueError(f"exact pair members split across outer fold {fold_id}")
        if not np.any(in_test):
            continue
        selected = [row for row in parameters if int(row["fold"]) == fold_id]
        if len(selected) != 1:
            raise ValueError(f"expected one selected candidate parameter row for fold {fold_id}, found {len(selected)}")
        parameter = selected[0]
        fold_pair_metrics.append(
            {
                "fold": fold_id,
                "n_pairs": int(in_test.sum()),
                "candidate_pair_rmse": rmse((candidate_delta - observed_delta)[in_test]),
                "rpl_pair_rmse": rmse((baseline_delta - observed_delta)[in_test]),
                "benefit_exponent": float(parameter["benefit_exponent"]),
                "benefit_offset": float(parameter["benefit_offset"]),
                "damage_exponent": float(parameter["damage_exponent"]),
                "ridge": float(parameter["ridge"]),
                "retention": float(parameter["retention"]),
                "late_multiplier": float(parameter["late_multiplier"]),
                "ordering_channel": int(parameter["ordering_channel"]),
            }
        )

    prediction_frame = pd.DataFrame(
        {
            "dataset": f"300m_{target}",
            "model": "centered_hierarchical_rpl_physical",
            "seed": 0,
            "row": np.arange(dataset.n),
            "run_name": dataset.frame["run_name"].astype(str),
            "policy_family": dataset.frame["policy_family"].astype(str),
            "phase_correspondence_key": dataset.frame["phase_correspondence_key"].astype(str),
            "observed": dataset.y,
            "predicted": prediction,
            "rpl_predicted": baseline_prediction,
        }
    )
    target_summary: dict[str, object] = {
        "metric": metric,
        "pair_metric": pair_metric,
        "pair_rmse_bootstrap_difference": pair_bootstrap,
        "overall_rmse_bootstrap_difference": grouped_bootstrap,
        "fold_pair_metrics": fold_pair_metrics,
    }
    summary_path.write_text(json.dumps(target_summary, indent=2) + "\n")
    prediction_frame.to_csv(prediction_path, index=False)
    parameter_frame = pd.DataFrame(parameters)
    parameter_frame.to_csv(parameter_path, index=False)
    return target_summary, prediction_frame, parameter_frame


def selected_baseline_row(rows: list[dict[str, float]], model: str) -> dict[str, float]:
    selected = [row for row in rows if row.get("model") == model and ("seed" not in row or int(row["seed"]) == 0)]
    if len(selected) != 1:
        raise ValueError(f"expected one {model} seed-0 summary row, found {len(selected)}")
    return selected[0]


def baseline_summary(target: str) -> tuple[dict[str, float], dict[str, float]]:
    directory = RPL_ROOT / f"diagnostic_300m_{target}"
    summary = json.loads((directory / "summary.json").read_text())
    return (
        selected_baseline_row(summary["metrics_300m"], "retained_power_law"),
        selected_baseline_row(summary["paired_metrics_300m"], "retained_power_law"),
    )


def baseline_parameters(target: str) -> dict[int, dict[str, float]]:
    frame = pd.read_csv(RPL_PARAMETERS[target])
    frame = frame.loc[frame["seed"].eq(0)]
    rows = {int(row["fold"]): row for row in frame.to_dict("records")}
    if set(rows) != {0, 1, 2}:
        raise ValueError(f"expected three seed-0 RPL parameter rows for {target}")
    return rows


def gate(summary: dict[str, dict[str, object]]) -> dict[str, object]:
    target_rows = {}
    pair_material_improvement = False
    overall_material_improvement = False
    all_core_within_five = True
    all_pairs_within_five = True
    for target, result in summary.items():
        base_metric, base_pair = baseline_summary(target)
        metric = result["metric"]
        pair = result["pair_metric"]
        pair_bootstrap = result["pair_rmse_bootstrap_difference"]
        overall_bootstrap = result["overall_rmse_bootstrap_difference"]
        core_ratio = float(metric["rmse"]) / float(base_metric["rmse"])
        pair_ratio = float(pair["delta_rmse"]) / float(base_pair["delta_rmse"])
        pair_better = float(pair_bootstrap["difference"]) < -float(pair_bootstrap["bootstrap_se"])
        overall_better = (
            float(overall_bootstrap["difference"]) < -float(overall_bootstrap["bootstrap_se"]) and pair_ratio <= 1.0
        )
        pair_material_improvement |= pair_better
        overall_material_improvement |= overall_better
        all_core_within_five &= core_ratio <= 1.05
        all_pairs_within_five &= pair_ratio <= 1.05
        target_rows[target] = {
            "candidate_rmse": metric["rmse"],
            "rpl_rmse": base_metric["rmse"],
            "rmse_ratio": core_ratio,
            "candidate_pair_rmse": pair["delta_rmse"],
            "rpl_pair_rmse": base_pair["delta_rmse"],
            "pair_rmse_ratio": pair_ratio,
            "pair_improvement_beyond_one_se": pair_better,
            "overall_improvement_beyond_one_se_without_pair_regression": overall_better,
        }
    passed = (
        all_core_within_five and all_pairs_within_five and (pair_material_improvement or overall_material_improvement)
    )
    return {
        "targets": target_rows,
        "all_core_rmse_within_five_percent": all_core_within_five,
        "all_pair_rmse_within_five_percent": all_pairs_within_five,
        "material_improvement": pair_material_improvement or overall_material_improvement,
        "passed": passed,
    }


def write_report(output_dir: Path, results: dict[str, dict[str, object]], decision: dict[str, object]) -> None:
    rows = []
    fold_rows = []
    for target, result in results.items():
        base_metric, base_pair = baseline_summary(target)
        base_parameters = baseline_parameters(target)
        metric = result["metric"]
        pair = result["pair_metric"]
        rows.append(
            "| "
            + " | ".join(
                [
                    target,
                    f'{base_metric["rmse"]:.6f}',
                    f'{metric["rmse"]:.6f}',
                    f'{base_pair["delta_rmse"]:.6f}',
                    f'{pair["delta_rmse"]:.6f}',
                    f'{pair["delta_bias"]:+.6f}',
                    f'{pair["sign_accuracy"]:.3f}',
                ]
            )
            + " |"
        )
        for fold in result["fold_pair_metrics"]:
            configurations = (
                ("RPL", base_parameters[int(fold["fold"])], fold["rpl_pair_rmse"]),
                ("Centered", fold, fold["candidate_pair_rmse"]),
            )
            for model, parameters, pair_rmse in configurations:
                fold_rows.append(
                    "| "
                    + " | ".join(
                        [
                            target,
                            str(fold["fold"]),
                            model,
                            str(fold["n_pairs"]),
                            f"{pair_rmse:.6f}",
                            f'{parameters["benefit_exponent"]:g}',
                            f'{parameters["benefit_offset"]:g}',
                            f'{parameters["damage_exponent"]:g}',
                            f'{parameters["retention"]:g}',
                            f'{parameters["late_multiplier"]:g}',
                            str(int(float(parameters["ordering_channel"]))),
                            f'{parameters["ridge"]:g}',
                        ]
                    )
                    + " |"
                )
    report = """# Physical-amplitude centered hierarchical RPL: frozen 300M comparison

StarCoder WSD80 is unchanged exactly by construction and by the outcome-free
numerical audit. This table is the mandatory high-TPP 39-bucket comparison.
The earlier normalized-coordinate result is invalid and is not used here.

| Target | RPL RMSE | Centered RMSE | RPL pair RMSE | Centered pair RMSE | Centered pair bias | Centered sign |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
""" + "\n".join(
        rows
    )
    report += (
        """

## Fold selections

"""
        "| Target | Fold | Model | Pairs | Pair RMSE | Benefit exponent | Offset | "
        "Damage exponent | Retention | Late multiplier | Ordering | Ridge |\n"
        """
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
"""
        + "\n".join(fold_rows)
    )
    report += f"""

## Decision

**{"PROVISIONAL TRIGGER" if decision["passed"] else "REJECT"}** under the
preregistered seed-0 screen. This is not a promotion decision: model selection
is shared within each of only three outer folds, so the row-resampled bootstrap
is conditional and understates uncertainty. The next frozen test crosses both
linear heads with both selected configurations to separate the prior from
shape-selection effects.

This route changes partial pooling only; even a successful pinned contrast
would not establish a new training-dynamics mechanism. WSD80 is an exact
no-regression invariant for this estimator, and Delphi 3e18 was not used for
selection.
"""
    (output_dir / "outcome_report.md").write_text(report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--targets", default="uncheatable,table9")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    prediction_frames = []
    parameter_frames = []
    for target in tuple(value.strip() for value in args.targets.split(",") if value.strip()):
        target_summary, predictions, parameters = fit_target(target, args.output_dir, args.workers)
        results[target] = target_summary
        prediction_frames.append(predictions)
        parameter_frames.append(parameters)

    decision = gate(results)
    output = {"results": results, "gate": decision}
    (args.output_dir / "outcome_summary.json").write_text(json.dumps(output, indent=2) + "\n")
    pd.concat(prediction_frames, ignore_index=True).to_csv(args.output_dir / "predictions_300m.csv", index=False)
    pd.concat(parameter_frames, ignore_index=True).to_csv(args.output_dir / "parameters_300m.csv", index=False)
    write_report(args.output_dir, results, decision)
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
