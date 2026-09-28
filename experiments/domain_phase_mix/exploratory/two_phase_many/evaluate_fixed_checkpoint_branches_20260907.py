# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["fsspec==2026.1.0", "numpy==2.3.5", "pandas==2.2.2", "scipy==1.17.0"]
# ///
"""Compare frozen branch predictions, emphasizing choices among unseen actions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

OUTPUT = Path(__file__).resolve().parent / "reference_outputs/fixed_checkpoint_branch_wspu_20260907"
MODELS = tuple(f"BRW-{i:03d}" for i in range(6))
BOOTSTRAPS = 3000
SEED = 20260907


def metric_values(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    order = np.argsort(prediction, kind="stable")
    error = prediction - y
    best = float(y.min())
    top = int(order[0])
    pair_i, pair_j = np.triu_indices(len(y), 1)
    actual_delta, predicted_delta = y[pair_i] - y[pair_j], prediction[pair_i] - prediction[pair_j]
    rankable = np.abs(actual_delta) > 1e-12
    agreement = np.sign(actual_delta[rankable]) == np.sign(predicted_delta[rankable])
    return {
        "rows": float(len(y)),
        "regret1": float(y[top] - best),
        "regret3": float(y[order[:3]].min() - best),
        "regret5": float(y[order[:5]].min() - best),
        "selected_rank": float(1 + np.sum(y < y[top] - 1e-12)),
        "selected_observed": float(y[top]),
        "selected_predicted": float(prediction[top]),
        "selected_error": float(error[top]),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "bias": float(np.mean(error)),
        "spearman": float(stats.spearmanr(y, prediction).statistic),
        "pairwise_accuracy": float(np.mean(agreement)),
        "pairwise_delta_rmse": float(np.sqrt(np.mean((actual_delta - predicted_delta) ** 2))),
    }


def predictions(output: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = pd.read_csv(output / "data/rows.csv")
    arrays = np.load(output / "data/arrays.npz")
    cohorts = json.loads((output / "cohorts.json").read_text())
    records, metrics, odd = [], [], []
    for cohort in cohorts:
        for model in MODELS:
            directory = output / "fits" / cohort["name"] / model
            files = sorted(directory.glob("*.json"))
            if not files:
                assert model == "BRW-005" and cohort["name"] == "cap10_local", (cohort["name"], model)
                continue
            fitted = [json.loads(path.read_text()) for path in files]
            by_component = {item["component_name"]: item for item in fitted}
            components = [str(v) for v in arrays["component_names"]] if model == "BRW-005" else ["macro"]
            assert set(by_component) == set(components)
            fitted = [by_component[name] for name in components]
            weights = arrays["component_weights"] if model == "BRW-005" else np.ones(1)
            positions = fitted[0]["prediction_rows"]
            assert all(item["prediction_rows"] == positions for item in fitted)
            values = np.asarray([item["predicted"] for item in fitted]).T @ weights
            lookup = dict(zip(positions, values, strict=True))
            predicted_tied = float(np.asarray([item["predicted_tied"] for item in fitted]) @ weights)
            calibration_rows = cohort["test_calibration"] or cohort["calibration"]
            observed_tied = float(arrays["target"][calibration_rows].mean())
            training_anchor = float(arrays["target"][cohort["calibration"]].mean())
            for test, indices in cohort["tests"].items():
                measured = arrays["target"][indices]
                predicted = np.asarray([lookup[index] for index in indices])
                cell = rows.iloc[indices]
                values = metric_values(measured, predicted)
                selected = int(np.argmin(predicted))
                choose_tied = predicted_tied <= predicted[selected] + 1e-12
                values.update(
                    {
                        "cohort": cohort["name"],
                        "test": test,
                        "model": model,
                        "selected_action": str(cell.iloc[selected].action_id),
                        "choose_tied": choose_tied,
                        "tied_option_regret": float(
                            (observed_tied if choose_tied else measured[selected]) - min(observed_tied, measured.min())
                        ),
                        "selected_gain_over_tied": float(observed_tied - measured[selected]),
                        "observed_tied": observed_tied,
                        "predicted_tied": predicted_tied,
                        "training_anchor": training_anchor,
                        "anchor_shift": observed_tied - training_anchor,
                        "anchor_adjusted_rmse": float(
                            np.sqrt(np.mean((predicted + observed_tied - training_anchor - measured) ** 2))
                        ),
                    }
                )
                if test == "local10":
                    anchor_position = int(np.flatnonzero(cell.action_id.eq("local_anchor_fit079"))[0])
                    values["selected_gain_over_fit079"] = float(measured[anchor_position] - measured[selected])
                    values["predicted_gain_over_fit079"] = float(predicted[anchor_position] - predicted[selected])
                    for ray in (24, 32, 37):
                        plus = int(np.flatnonzero(cell.action_id.eq(f"local_plus_{ray}"))[0])
                        minus = int(np.flatnonzero(cell.action_id.eq(f"local_minus_{ray}"))[0])
                        actual = float((measured[plus] - measured[minus]) / 2)
                        estimate = float((predicted[plus] - predicted[minus]) / 2)
                        odd.append(
                            {
                                "cohort": cohort["name"],
                                "model": model,
                                "ray": ray,
                                "measured_odd": actual,
                                "predicted_odd": estimate,
                                "sign_correct": np.sign(actual) == np.sign(estimate),
                            }
                        )
                metrics.append(values)
                for index, truth, estimate in zip(indices, measured, predicted, strict=True):
                    records.append(
                        {
                            "cohort": cohort["name"],
                            "test": test,
                            "model": model,
                            "row": index,
                            "row_id": rows.iloc[index].row_id,
                            "action_id": rows.iloc[index].action_id,
                            "coordinate_hash": rows.iloc[index].coordinate_hash,
                            "measured": float(truth),
                            "predicted": float(estimate),
                        }
                    )
    return pd.DataFrame(records), pd.DataFrame(metrics), pd.DataFrame(odd)


def bootstrap_statistics(y: np.ndarray, p: np.ndarray, draws: np.ndarray) -> dict[str, np.ndarray]:
    errors = p - y
    selected = np.argmin(p[draws], axis=1)
    best = np.min(y[draws], axis=1)
    chosen_rows = draws[np.arange(len(draws)), selected]
    return {
        "rmse": np.sqrt(np.mean(errors[draws] ** 2, axis=1)),
        "regret1": y[chosen_rows] - best,
        "mae": np.mean(np.abs(errors[draws]), axis=1),
        "selected_error": errors[chosen_rows],
    }


def comparisons(frame: pd.DataFrame, metric_frame: pd.DataFrame) -> pd.DataFrame:
    records = []
    rng = np.random.default_rng(SEED)
    for (cohort, test), group in frame.groupby(["cohort", "test"], sort=True):
        table = group.pivot(index="row", columns="model", values="predicted").sort_index()
        y = group.drop_duplicates("row").set_index("row").loc[table.index, "measured"].to_numpy(float)
        draws = rng.integers(0, len(y), size=(BOOTSTRAPS, len(y)))
        boot = {model: bootstrap_statistics(y, table[model].to_numpy(), draws) for model in table}
        for baseline in ("BRW-000", "BRW-001"):
            for model in table:
                if model == baseline:
                    continue
                for metric in boot[model]:
                    delta = boot[model][metric] - boot[baseline][metric]
                    left = metric_frame.query("cohort == @cohort and test == @test and model == @model").iloc[0]
                    right = metric_frame.query("cohort == @cohort and test == @test and model == @baseline").iloc[0]
                    low, high = np.quantile(delta, [0.025, 0.975])
                    records.append(
                        {
                            "cohort": cohort,
                            "test": test,
                            "model": model,
                            "baseline": baseline,
                            "metric": metric,
                            "difference": float(left[metric] - right[metric]),
                            "ci_low": low,
                            "ci_high": high,
                            "scope": "Ordinary conditional paired-action bootstrap; no refitting or search correction",
                        }
                    )
    local = frame[frame.test.eq("local10")]
    if len(local):
        states = sorted(local.cohort.unique())
        actions = sorted(local.action_id.unique())
        assert len(states) == 5 and len(actions) == 10
        draws = rng.integers(0, 10, size=(BOOTSTRAPS, 10))
        boot = {}
        for model in MODELS:
            per_state = []
            for state in states:
                cell = local[local.model.eq(model) & local.cohort.eq(state)].set_index("action_id").loc[actions]
                per_state.append(bootstrap_statistics(cell.measured.to_numpy(), cell.predicted.to_numpy(), draws))
            boot[model] = {metric: np.mean([item[metric] for item in per_state], axis=0) for metric in per_state[0]}
        average = metric_frame[metric_frame.test.eq("local10")].groupby("model").mean(numeric_only=True)
        for baseline in ("BRW-000", "BRW-001"):
            for model in MODELS:
                if model == baseline:
                    continue
                for metric in boot[model]:
                    low, high = np.quantile(boot[model][metric] - boot[baseline][metric], [0.025, 0.975])
                    records.append(
                        {
                            "cohort": "five_checkpoint_mean",
                            "test": "local10",
                            "model": model,
                            "baseline": baseline,
                            "metric": metric,
                            "difference": float(average.loc[model, metric]) - float(average.loc[baseline, metric]),
                            "ci_low": low,
                            "ci_high": high,
                            "scope": (
                                "Shared ten-action resampling across five checkpoints; checkpoint uncertainty excluded"
                            ),
                        }
                    )
    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    destination = args.output / "comparison"
    destination.mkdir(exist_ok=True)
    frame, metrics, odd = predictions(args.output)
    frame.to_csv(destination / "predictions.csv", index=False)
    metrics.to_csv(destination / "metrics.csv", index=False)
    odd.to_csv(destination / "local_odd_contrasts.csv", index=False)
    metrics[metrics.test.eq("local10")].groupby("model").mean(numeric_only=True).to_csv(
        destination / "local_checkpoint_mean.csv"
    )
    comparisons(frame, metrics).to_csv(destination / "paired_bootstrap.csv", index=False)
    inputs = [args.output / "data/manifest.json", args.output / "cohorts.json", Path(__file__)]
    inputs.extend(sorted((args.output / "fits").rglob("*.json")))
    manifest = {
        "inputs": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs},
        "prediction_rows": len(frame),
        "metric_rows": len(metrics),
        "bootstrap_replicates": BOOTSTRAPS,
        "bootstrap_seed": SEED,
        "prospective": False,
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(metrics[["cohort", "test", "model", "regret1", "regret3", "rmse", "spearman"]].to_string(index=False))


if __name__ == "__main__":
    main()
