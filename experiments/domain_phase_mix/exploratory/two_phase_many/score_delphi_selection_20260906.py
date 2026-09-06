# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "scipy", "pandas", "scikit-learn", "joblib", "tabulate"]
# ///
"""Score frozen Delphi predictions and source-disjoint development decisions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse.csgraph import connected_components

from experiments.domain_phase_mix.exploratory.two_phase_many import benchmark_delphi_selection_20260906 as benchmark
from experiments.domain_phase_mix.exploratory.two_phase_many import delphi_selection_models_20260906 as alternatives

METHODS = (*benchmark.BASELINES, "wspu_direct_macro", *(spec.name for spec in alternatives.SPECS))
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 20260906


def source_blocks(sources: pd.Series) -> tuple[np.ndarray, list[set[str]]]:
    """Connected source memberships keep shared coordinates out of both sides of a split."""
    memberships = [set(text.split(";")) for text in sources]
    adjacency = np.array([[bool(left & right) for right in memberships] for left in memberships])
    _, labels = connected_components(adjacency, directed=False)
    return labels, memberships


def diverse_order(prediction: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Keep the predicted winner, then spread ten picks across the predicted top twenty."""
    order = np.argsort(prediction, kind="stable")
    pool = list(order[:20])
    selected = [pool.pop(0)]
    while pool and len(selected) < 10:
        distance = np.abs(weights[pool, None] - weights[None, selected]).sum(axis=2).min(axis=1)
        selected.append(pool.pop(int(np.argmax(distance))))
    return np.asarray(selected + [int(i) for i in order if i not in selected])


def collect(output: Path, repeats: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = benchmark.read_npz(output / "inputs" / "panel.npz")
    records = []
    diagnostics = []
    reproduction = []
    old = pd.read_csv(output / "inputs" / "historical_predictions.csv")
    for target in benchmark.TARGETS:
        bank = benchmark.read_npz(output / "inputs" / f"{target}_bank_features.npz")
        aggregate_weights = data[f"{target}_aggregation_weights"]
        for method in METHODS:
            bank_replicates = []
            for repeat in range(repeats):
                for fold in ((-1, 0, 1, 2, 3, 4) if repeat == 0 else (0, 1, 2, 3, 4)):
                    if method in benchmark.BASELINES or method == "wspu_direct_macro":
                        components = range(len(aggregate_weights)) if method in benchmark.BASELINES else [-1]
                        shards = [
                            benchmark.read_npz(
                                output / "baseline_shards" / method / target / f"r{repeat}_f{fold}_c{c}.npz"
                            )
                            for c in components
                        ]
                        weights = aggregate_weights if method in benchmark.BASELINES else np.ones(1)
                        prediction = np.stack([shard["prediction"] for shard in shards], axis=1) @ weights
                        bank_prediction = np.stack([shard["bank_prediction"] for shard in shards], axis=1) @ weights
                        test = shards[0]["test"]
                        for component, shard in zip(components, shards, strict=True):
                            diagnostic = json.loads(str(shard["diagnostics_json"]))
                            diagnostics.append(
                                {
                                    "method": method,
                                    "target": target,
                                    "repeat": repeat,
                                    "fold": fold,
                                    "component": component,
                                    "elapsed": float(shard["elapsed"]),
                                    **diagnostic,
                                }
                            )
                        if fold >= 0 and method in benchmark.BASELINES:
                            for component, shard in zip(components, shards, strict=True):
                                task = benchmark.harness.FitTask(
                                    method,
                                    benchmark.PANEL,
                                    target,
                                    component,
                                    str(data[f"{target}_components"][component]),
                                    repeat,
                                    fold,
                                )
                                original_path = benchmark.harness.shard_path(benchmark.HISTORICAL, task)
                                if original_path.exists():
                                    original = benchmark.read_npz(original_path)
                                    if np.array_equal(original["test"], test):
                                        reproduction.append(
                                            {
                                                "method": method,
                                                "target": target,
                                                "scope": "atomic_outer",
                                                "repeat": repeat,
                                                "fold": fold,
                                                "component": component,
                                                "max_absolute_difference": float(
                                                    np.max(np.abs(shard["prediction"] - original["prediction"]))
                                                ),
                                            }
                                        )
                    else:
                        shard = benchmark.read_npz(
                            output / "alternative_shards" / method / target / f"r{repeat}_f{fold}.npz"
                        )
                        prediction = shard["prediction"]
                        bank_prediction = shard["bank_prediction"]
                        test = shard["test"]
                        diagnostics.append(
                            {
                                "method": method,
                                "target": target,
                                "repeat": repeat,
                                "fold": fold,
                                "component": -1,
                                "elapsed": float(shard["elapsed"]),
                                "fitted_dof": float(shard["dof"]),
                                "effective_rows": float(shard["effective_rows"]),
                                "selected": str(shard["selected_json"]),
                            }
                        )
                    if fold == -1:
                        final = bank_prediction
                        if method in benchmark.BASELINES:
                            reference = old[old.model.eq(method) & old.target.eq(target)].set_index("coordinate_id")
                            expected = reference.loc[bank["coordinate_id"], "prediction"].to_numpy(float)
                            reproduction.append(
                                {
                                    "method": method,
                                    "target": target,
                                    "scope": "external_aggregate",
                                    "repeat": 0,
                                    "fold": -1,
                                    "component": -1,
                                    "max_absolute_difference": float(np.max(np.abs(final - expected))),
                                }
                            )
                        continue
                    bank_replicates.append(bank_prediction)
                    records.extend(
                        {
                            "method": method,
                            "target": target,
                            "population": "panel_oof",
                            "repeat": repeat,
                            "fold": fold,
                            "row_id": str(int(row)),
                            "prediction": float(value),
                            "uncertainty": np.nan,
                        }
                        for row, value in zip(test, prediction, strict=True)
                    )
            spread = np.std(np.stack(bank_replicates), axis=0, ddof=1)
            records.extend(
                {
                    "method": method,
                    "target": target,
                    "population": "external_development",
                    "repeat": 0,
                    "fold": -1,
                    "row_id": str(row),
                    "prediction": float(value),
                    "uncertainty": float(error),
                }
                for row, value, error in zip(bank["coordinate_id"], final, spread, strict=True)
            )
    frame = pd.DataFrame(records)
    frame.to_csv(output / "predictions.csv", index=False)
    benchmark.write_json(
        output / "prediction_hash.json",
        {
            "sha256": benchmark.sha256(output / "predictions.csv"),
            "labels_joined": False,
            "uncertainty": "dispersion across blocked panel refits; heuristic, not a calibrated confidence bound",
        },
    )
    return frame, pd.DataFrame(diagnostics), pd.DataFrame(reproduction)


def metrics_row(
    target: str,
    method: str,
    population: str,
    stratum: str,
    policy: str,
    measured: np.ndarray,
    predicted: np.ndarray,
    row_ids: np.ndarray,
    order: np.ndarray,
    repeat: int = 0,
    fold: int = -1,
) -> dict:
    metrics = benchmark.selection_metrics(measured, predicted, order)
    selected = metrics.pop("selected_row")
    return {
        "target": target,
        "method": method,
        "population": population,
        "stratum": stratum,
        "policy": policy,
        "repeat": repeat,
        "fold": fold,
        "selected_id": row_ids[selected],
        **metrics,
    }


def score_predictions(output: Path, predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = benchmark.read_npz(output / "inputs" / "panel.npz")
    results = []
    assignments = []
    loso = []
    for target in benchmark.TARGETS:
        labels = pd.read_csv(output / "inputs" / f"{target}_bank_labels.csv")
        features = benchmark.read_npz(output / "inputs" / f"{target}_bank_features.npz")
        blocks, memberships = source_blocks(labels.sources)
        for row, block, sources in zip(labels.coordinate_id, blocks, labels.sources, strict=True):
            assignments.append({"target": target, "coordinate_id": row, "source_block": int(block), "sources": sources})
        all_sources = sorted(set.union(*memberships))
        strata = {
            "all": np.arange(len(labels)),
            "optima": np.flatnonzero([not bool(source & benchmark.INTERVENTIONS) for source in memberships]),
            "interventions": np.flatnonzero([bool(source & benchmark.INTERVENTIONS) for source in memberships]),
        }
        strata.update(
            {f"source:{source}": np.flatnonzero([source in row for row in memberships]) for source in all_sources}
        )
        strata.update({f"source_block:{block}": np.flatnonzero(blocks == block) for block in np.unique(blocks)})
        measured = labels.measured_mean_bpb.to_numpy(float)
        row_ids = labels.coordinate_id.to_numpy(str)
        prediction_matrix = []
        block_metrics = []
        for method in METHODS:
            bank = (
                predictions[
                    predictions.method.eq(method)
                    & predictions.target.eq(target)
                    & predictions.population.eq("external_development")
                ]
                .set_index("row_id")
                .loc[row_ids]
            )
            predicted = bank.prediction.to_numpy(float)
            spread = bank.uncertainty.to_numpy(float)
            prediction_matrix.append(predicted)
            for name, rows in strata.items():
                if not len(rows):
                    continue
                for policy in ("point", "dispersion_penalty_1", "diverse_top20"):
                    if policy == "diverse_top20":
                        order = diverse_order(predicted[rows], features["weights"][rows])
                    else:
                        score = predicted[rows] + (spread[rows] if policy == "dispersion_penalty_1" else 0)
                        order = np.argsort(score, kind="stable")
                    record = metrics_row(
                        target,
                        method,
                        "external_development",
                        name,
                        policy,
                        measured[rows],
                        predicted[rows],
                        row_ids[rows],
                        order,
                    )
                    results.append(record)
                    if policy == "point" and name.startswith("source_block:"):
                        block_metrics.append(record)
            panel = predictions[
                predictions.method.eq(method) & predictions.target.eq(target) & predictions.population.eq("panel_oof")
            ]
            for repeat, frame in panel.groupby("repeat"):
                if len(frame) != 280 or frame.row_id.duplicated().any():
                    raise ValueError("Every canonical row must have one out-of-fold prediction per repeat")
                for fold in [-1, *sorted(frame.fold.unique())]:
                    sub = frame if fold == -1 else frame[frame.fold.eq(fold)]
                    rows = sub.row_id.to_numpy(int)
                    predicted = sub.prediction.to_numpy(float)
                    results.append(
                        metrics_row(
                            target,
                            method,
                            "panel_oof",
                            "all",
                            "point",
                            data[f"{target}_aggregate"][rows],
                            predicted,
                            sub.row_id.to_numpy(str),
                            np.argsort(predicted, kind="stable"),
                            int(repeat),
                            int(fold),
                        )
                    )
        block_table = pd.DataFrame(block_metrics)
        sizes = pd.Series(blocks).value_counts()
        eligible_blocks = sizes.index[sizes.ge(5)]
        matrix = np.stack(prediction_matrix)
        for block in eligible_blocks:
            train_blocks = [f"source_block:{b}" for b in eligible_blocks if b != block]
            training = block_table[block_table.stratum.isin(train_blocks)]
            ranking = training.groupby("method")[["regret_at_1", "best_of_5_regret", "rmse"]].mean()
            method = str(ranking.sort_values(["regret_at_1", "best_of_5_regret", "rmse"], kind="stable").index[0])
            rows = np.flatnonzero(blocks == block)
            train_sources = set.union(
                *(memberships[i] for i in np.flatnonzero(np.isin(blocks, [b for b in eligible_blocks if b != block])))
            )
            test_sources = set.union(*(memberships[i] for i in rows))
            if train_sources & test_sources:
                raise ValueError("Source overlap in development method selection")
            predicted = matrix[METHODS.index(method), rows]
            loso.append(
                metrics_row(
                    target,
                    method,
                    "source_disjoint_method_selection",
                    f"source_block:{block}",
                    "point",
                    measured[rows],
                    predicted,
                    row_ids[rows],
                    np.argsort(predicted, kind="stable"),
                )
            )
    return pd.DataFrame(results), pd.DataFrame(assignments), pd.DataFrame(loso)


def paired_sources(metrics: pd.DataFrame, loso: pd.DataFrame) -> pd.DataFrame:
    frame = metrics[metrics.stratum.str.startswith("source_block:") & metrics.rows.ge(5) & metrics.policy.eq("point")]
    references = [(spec.name, spec.parent) for spec in alternatives.SPECS]
    references += [(method, benchmark.BASELINES[0]) for method in METHODS if method != benchmark.BASELINES[0]]
    references += [("source_disjoint_method_selection", benchmark.BASELINES[0])]
    loso = loso.assign(method="source_disjoint_method_selection")
    frame = pd.concat([frame, loso], ignore_index=True)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    results = []
    for target in benchmark.TARGETS:
        for candidate, reference in sorted(set(references)):
            paired = frame[frame.target.eq(target) & frame.method.eq(candidate)].merge(
                frame[frame.target.eq(target) & frame.method.eq(reference)],
                on="stratum",
                suffixes=("_candidate", "_reference"),
                validate="one_to_one",
            )
            if len(paired) < 2:
                continue
            samples = rng.integers(0, len(paired), size=(BOOTSTRAP_DRAWS, len(paired)))
            for metric in ("regret_at_1", "best_of_5_regret", "best_of_10_regret", "rmse", "optimism"):
                delta = (paired[f"{metric}_candidate"] - paired[f"{metric}_reference"]).to_numpy(float)
                boot = delta[samples].mean(axis=1)
                results.append(
                    {
                        "target": target,
                        "candidate": candidate,
                        "reference": reference,
                        "metric": metric,
                        "source_blocks": len(delta),
                        "mean_delta": float(delta.mean()),
                        "ci_low": float(np.quantile(boot, 0.025)),
                        "ci_high": float(np.quantile(boot, 0.975)),
                        "fraction_better": float(np.mean(delta < 0)),
                        "inference": "descriptive retrospective source-block bootstrap; no multiplicity correction",
                    }
                )
    return pd.DataFrame(results)


def support_diagnostics(output: Path) -> pd.DataFrame:
    data = benchmark.read_npz(output / "inputs" / "panel.npz")
    panel = data["weights"]
    variables = len(panel) + panel.shape[1]
    objective = np.r_[np.zeros(len(panel)), np.ones(panel.shape[1])]
    constraints = np.vstack(
        [np.hstack([panel.T, -np.eye(panel.shape[1])]), np.hstack([-panel.T, -np.eye(panel.shape[1])])]
    )
    equality = np.r_[np.ones(len(panel)), np.zeros(panel.shape[1])][None]
    cache = {}
    rows = []
    for target in benchmark.TARGETS:
        bank = benchmark.read_npz(output / "inputs" / f"{target}_bank_features.npz")
        for coordinate, weight, exposure in zip(bank["coordinate_id"], bank["weights"], bank["exposures"], strict=True):
            if coordinate not in cache:
                projection = linprog(
                    objective,
                    A_ub=constraints,
                    b_ub=np.r_[weight, -weight],
                    A_eq=equality,
                    b_eq=[1],
                    bounds=[(0, None)] * variables,
                    method="highs",
                )
                if not projection.success:
                    raise ValueError(projection.message)
                cache[coordinate] = float(projection.fun / 2)
            entropy = -np.dot(weight, np.log(np.maximum(weight, 1e-15)))
            rows.append(
                {
                    "target": target,
                    "coordinate_id": coordinate,
                    "hull_distance_tv": cache[coordinate],
                    "nearest_panel_tv": float(np.abs(panel - weight).sum(axis=1).min() / 2),
                    "buckets_above_panel_max": int((exposure > data["exposures"].max(axis=0) + 1e-8).sum()),
                    "effective_buckets": float(np.exp(entropy)),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=benchmark.DEFAULT_OUTPUT)
    parser.add_argument("--repeats", type=int, choices=(1, 5), default=1)
    args = parser.parse_args()
    benchmark.verify_inputs(args.output_dir)
    predictions, diagnostics, reproduction = collect(args.output_dir, args.repeats)
    metrics, assignments, loso = score_predictions(args.output_dir, predictions)
    metrics.to_csv(args.output_dir / "metrics.csv", index=False)
    assignments.to_csv(args.output_dir / "source_blocks.csv", index=False)
    loso.to_csv(args.output_dir / "source_disjoint_selection.csv", index=False)
    paired_sources(metrics, loso).to_csv(args.output_dir / "paired_source_contrasts.csv", index=False)
    diagnostics.to_csv(args.output_dir / "fit_diagnostics.csv", index=False)
    reproduction.to_csv(args.output_dir / "baseline_reproduction.csv", index=False)
    support_diagnostics(args.output_dir).to_csv(args.output_dir / "support_diagnostics.csv", index=False)
    selected = metrics[metrics.stratum.eq("optima") & metrics.policy.eq("point")]
    print(
        selected[
            [
                "target",
                "method",
                "regret_at_1",
                "best_of_5_regret",
                "best_of_10_regret",
                "selected_rank",
                "optimism",
                "rmse",
                "spearman",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
