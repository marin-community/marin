# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "scipy", "pandas", "scikit-learn", "joblib", "tabulate"]
# ///
"""Score frozen Delphi coupling ablations without refitting or launching jobs."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.domain_phase_mix.exploratory.two_phase_many import audit_delphi_matched_policies_20260906 as policies
from experiments.domain_phase_mix.exploratory.two_phase_many import benchmark_delphi_selection_20260906 as benchmark
from experiments.domain_phase_mix.exploratory.two_phase_many import fit_delphi_coupling_20260906 as coupling
from experiments.domain_phase_mix.exploratory.two_phase_many import score_delphi_selection_20260906 as scoring

FITTED_METHODS = tuple(spec.name for spec in coupling.SPECS)
REMOVED_METHODS = tuple(f"{basis}_interaction_removed" for basis in coupling.Basis)
METHODS = (*benchmark.BASELINES, *FITTED_METHODS, *REMOVED_METHODS)
SELECTABLE_METHODS = (*benchmark.BASELINES, *FITTED_METHODS)
DIRECTION_METRICS = ("delta_rmse", "delta_mae", "direction_accuracy", "mean_predicted_magnitude")


def contrasts() -> tuple[tuple[str, str], ...]:
    references = [
        (method, reference) for method in (*FITTED_METHODS, *REMOVED_METHODS) for reference in benchmark.BASELINES
    ]
    for basis in coupling.Basis:
        references.extend(
            [
                (f"{basis}_coupled_exp", f"{basis}_additive_exp"),
                (f"{basis}_coupled_exp", f"{basis}_interaction_removed"),
                (f"{basis}_additive_exp", f"{basis}_identity"),
                (f"{basis}_coupled_exp", f"{basis}_identity"),
            ]
        )
    references.append((benchmark.BASELINES[2], benchmark.BASELINES[0]))
    return tuple(sorted(set(references)))


def scoring_protocol(output: Path, reference: Path) -> None:
    benchmark.verify_inputs(output)
    if benchmark.verify_inputs(reference) != benchmark.verify_inputs(output):
        raise ValueError("Reference and coupling inputs differ")
    expected = json.loads((reference / "prediction_hash.json").read_text())["sha256"]
    if expected != benchmark.sha256(reference / "predictions.csv"):
        raise ValueError("Original reference predictions changed")
    protocol = {
        "methods": METHODS,
        "method_selection_pool": SELECTABLE_METHODS,
        "zero_refit_ablation": "same coupled coefficients evaluated with the additive anchored formula",
        "outer_partitions": "frozen original repeat-zero five blocked folds, with existing inner selection",
        "source_partitions": "connected source memberships, held together before selection",
        "paired_comparisons": contrasts(),
        "pair_weighting": "equal connected-source block, never independent-pair resampling",
        "directional_competitive_window_bpb": policies.COMPETITIVE_WINDOW,
        "directional_noise_separation_bpb": policies.NOISE_SEPARATION,
        "competitive_status": "truth-conditioned retrospective diagnostic only",
        "fixed_policy_epoch_caps": policies.EPOCH_CAPS,
        "fixed_policy_kl_penalties": policies.KL_PENALTIES,
        "policy_tuning": "none; all fixed-grid cells reported, regret relative to each eligible minimum",
        "source_selection": "other eligible blocks only, equal-block regret@1 then best-of-5 then RMSE",
        "minimum_source_rows": policies.MIN_SOURCE_ROWS,
        "bootstrap_draws": policies.BOOTSTRAP_DRAWS,
        "bootstrap_seed": policies.BOOTSTRAP_SEED,
        "inference": "descriptive retrospective source-block intervals without multiplicity correction",
        "input_sha256": benchmark.verify_inputs(output),
        "original_prediction_sha256": expected,
        "fit_protocol_sha256": benchmark.sha256(output / "protocol.json"),
    }
    path = output / "scoring_protocol.json"
    normalized = json.loads(json.dumps(protocol))
    if path.exists() and json.loads(path.read_text()) != normalized:
        raise ValueError("Frozen scoring protocol changed; use a distinct output directory")
    if not path.exists():
        benchmark.write_json(path, protocol)


def removed_atomic(shard: dict, data: dict, bank: dict, basis: coupling.Basis) -> np.ndarray:
    query = np.vstack([data["weights"][shard["test"]], bank["weights"], data["weights"][shard["train"]]])
    anchor = coupling.bucket_basis(shard["anchor"][None], data["inventory"], basis)
    matrix = (coupling.bucket_basis(query, data["inventory"], basis) - anchor) / shard["feature_scale"]
    return np.column_stack(
        [
            coupling.response_jacobian(parameters, matrix, shard["projection"], coupling.Link.ADDITIVE)[0] * scale
            for parameters, scale in zip(shard["parameters"], shard["outcome_scale"], strict=True)
        ]
    )


def collect_predictions(output: Path, reference: Path) -> tuple[pd.DataFrame, dict, pd.DataFrame, pd.DataFrame]:
    """Persist predictions and component arrays before any bank outcome is joined."""
    expected_paths = [
        output / "shards" / spec.name / target / f"r0_f{fold}.npz"
        for spec in coupling.SPECS
        for target in benchmark.TARGETS
        for fold in (-1, 0, 1, 2, 3, 4)
    ]
    missing = [str(path) for path in expected_paths if not path.exists()]
    if missing:
        raise ValueError(f"All 72 fit shards must exist before scoring; missing {len(missing)}")
    data = benchmark.read_npz(output / "inputs" / "panel.npz")
    original = pd.read_csv(reference / "predictions.csv")
    original = original[original.method.isin(benchmark.BASELINES)].copy()
    records = original.to_dict("records")
    atomic = {}
    diagnostics = []
    validity = []
    fingerprints = set()
    for target in benchmark.TARGETS:
        bank = benchmark.read_npz(output / "inputs" / f"{target}_bank_features.npz")
        aggregate_weights = data[f"{target}_aggregation_weights"]
        for spec in coupling.SPECS:
            for fold in (-1, 0, 1, 2, 3, 4):
                path = output / "shards" / spec.name / target / f"r0_f{fold}.npz"
                shard = benchmark.read_npz(path)
                fingerprints.add(str(shard["fingerprint"]))
                choices = json.loads(str(shard["selected_json"]))
                for item in json.loads(str(shard["diagnostics_json"])):
                    starts = np.asarray(item.pop("start_objectives"), dtype=float)
                    successes = item.pop("start_successes")
                    diagnostics.append(
                        {
                            "target": target,
                            "method": spec.name,
                            "fold": fold,
                            "ridge": choices["ridge"],
                            "inner_aggregate_rmse": choices["aggregate_rmse"],
                            "selected_inner_failures": choices["unsuccessful"],
                            "selected_inner_maximum_gradient": choices["maximum_gradient"],
                            "restart_count": len(starts),
                            "restart_unsuccessful": sum(not value for value in successes),
                            "restart_objective_range": float(np.ptp(starts)),
                            "restart_objective_relative_range": float(np.ptp(starts) / max(1, np.abs(starts).min())),
                            **item,
                        }
                    )
                variants = [
                    (
                        spec.name,
                        shard["prediction"],
                        shard["bank_prediction"],
                        shard["atomic_prediction"],
                        shard["atomic_bank_prediction"],
                    )
                ]
                if spec.link == coupling.Link.COUPLED:
                    values = removed_atomic(shard, data, bank, spec.basis)
                    end = len(shard["test"])
                    bank_end = end + len(bank["coordinate_id"])
                    np.testing.assert_allclose(
                        values[:end] @ aggregate_weights, shard["removed_prediction"], atol=1e-12, rtol=1e-12
                    )
                    np.testing.assert_allclose(
                        values[end:bank_end] @ aggregate_weights,
                        shard["removed_bank_prediction"],
                        atol=1e-12,
                        rtol=1e-12,
                    )
                    variants.append(
                        (
                            f"{spec.basis}_interaction_removed",
                            shard["removed_prediction"],
                            shard["removed_bank_prediction"],
                            values[:end],
                            values[end:bank_end],
                        )
                    )
                for method, test_prediction, bank_prediction, atomic_test, atomic_bank in variants:
                    if not np.isfinite(test_prediction).all() or not np.isfinite(bank_prediction).all():
                        raise ValueError(f"Nonfinite predictions: {target}/{method}/{fold}")
                    if fold >= 0:
                        atomic[(target, method, fold)] = (shard["test"], atomic_test)
                        records.extend(
                            {
                                "method": method,
                                "target": target,
                                "population": "panel_oof",
                                "repeat": 0,
                                "fold": fold,
                                "row_id": str(int(row)),
                                "prediction": float(value),
                                "uncertainty": np.nan,
                            }
                            for row, value in zip(shard["test"], test_prediction, strict=True)
                        )
                    else:
                        records.extend(
                            {
                                "method": method,
                                "target": target,
                                "population": "external_development",
                                "repeat": 0,
                                "fold": -1,
                                "row_id": str(row),
                                "prediction": float(value),
                                "uncertainty": np.nan,
                            }
                            for row, value in zip(bank["coordinate_id"], bank_prediction, strict=True)
                        )
                    for population, matrix, aggregate in (
                        ("panel_oof", atomic_test, test_prediction),
                        ("bank", atomic_bank, bank_prediction),
                    ):
                        validity.append(
                            {
                                "target": target,
                                "method": method,
                                "fold": fold,
                                "population": population,
                                "rows": len(aggregate),
                                "atomic_values": matrix.size,
                                "negative_atomic_values": int((matrix < 0).sum()),
                                "nonpositive_atomic_values": int((matrix <= 0).sum()),
                                "negative_aggregate_values": int((aggregate < 0).sum()),
                                "minimum_atomic_prediction": float(matrix.min()) if matrix.size else np.nan,
                                "minimum_aggregate_prediction": float(aggregate.min()) if len(aggregate) else np.nan,
                            }
                        )
        for method in benchmark.BASELINES:
            for fold in range(5):
                shards = [
                    benchmark.read_npz(reference / "baseline_shards" / method / target / f"r0_f{fold}_c{component}.npz")
                    for component in range(len(aggregate_weights))
                ]
                atomic[(target, method, fold)] = (
                    shards[0]["test"],
                    np.stack([shard["prediction"] for shard in shards], axis=1),
                )
    if len(fingerprints) != 1:
        raise ValueError("Coupling shards have mixed source/input fingerprints")
    predictions = pd.DataFrame(records)
    if predictions.duplicated(["target", "method", "population", "repeat", "fold", "row_id"]).any():
        raise ValueError("Duplicate prediction identities")
    policies.write_frame(output / "predictions.csv", predictions)
    benchmark.write_json(
        output / "prediction_hash.json",
        {
            "sha256": benchmark.sha256(output / "predictions.csv"),
            "labels_joined": False,
            "fit_fingerprint": next(iter(fingerprints)),
        },
    )
    return predictions, atomic, pd.DataFrame(diagnostics), pd.DataFrame(validity)


def matched_rankings(output: Path, predictions: pd.DataFrame) -> pd.DataFrame:
    panel = benchmark.read_npz(output / "inputs" / "panel.npz")
    natural = policies.natural_proportions(panel["inventory"])
    rows = []
    for target in benchmark.TARGETS:
        bank = benchmark.read_npz(output / "inputs" / f"{target}_bank_features.npz")
        for method in METHODS:
            values = (
                predictions[
                    predictions.target.eq(target)
                    & predictions.method.eq(method)
                    & predictions.population.eq("external_development")
                ]
                .set_index("row_id")
                .loc[bank["coordinate_id"], "prediction"]
                .to_numpy(float)
            )
            for cap, coefficient in itertools.product(policies.EPOCH_CAPS, policies.KL_PENALTIES):
                indices, score, divergence = policies.policy_order(
                    values, bank["weights"], bank["exposures"], natural, cap, coefficient
                )
                rows.extend(
                    {
                        "target": target,
                        "method": method,
                        "epoch_cap": cap,
                        "kl_penalty": coefficient,
                        "policy_rank": rank,
                        "coordinate_id": str(bank["coordinate_id"][row]),
                        "prediction": float(values[row]),
                        "policy_score": float(value),
                        "kl_to_natural": float(kl),
                        "max_exposure": float(bank["exposures"][row].max()),
                    }
                    for rank, (row, value, kl) in enumerate(zip(indices, score, divergence, strict=True), start=1)
                )
    frame = pd.DataFrame(rows)
    policies.write_frame(output / "policy_rankings.csv", frame)
    benchmark.write_json(
        output / "policy_ranking_hash.json",
        {"sha256": benchmark.sha256(output / "policy_rankings.csv"), "labels_joined": False},
    )
    return frame


def score_predictions(output: Path, predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = benchmark.read_npz(output / "inputs" / "panel.npz")
    results, directions, assignments = [], [], []
    for target in benchmark.TARGETS:
        labels = pd.read_csv(output / "inputs" / f"{target}_bank_labels.csv")
        blocks, memberships = scoring.source_blocks(labels.sources)
        assignments.append(labels[["coordinate_id", "sources"]].assign(target=target, source_block=blocks))
        strata = {
            "all": np.arange(len(labels)),
            "optima": np.flatnonzero([not bool(source & benchmark.INTERVENTIONS) for source in memberships]),
            "interventions": np.flatnonzero([bool(source & benchmark.INTERVENTIONS) for source in memberships]),
        }
        strata.update({f"source_block:{int(block)}": np.flatnonzero(blocks == block) for block in np.unique(blocks)})
        measured = labels.measured_mean_bpb.to_numpy(float)
        ids = labels.coordinate_id.to_numpy(str)
        for method in METHODS:
            frame = predictions[predictions.target.eq(target) & predictions.method.eq(method)]
            predicted = (
                frame[frame.population.eq("external_development")]
                .set_index("row_id")
                .loc[ids, "prediction"]
                .to_numpy(float)
            )
            for name, rows in strata.items():
                if not len(rows):
                    continue
                results.append(
                    scoring.metrics_row(
                        target,
                        method,
                        "external_development",
                        name,
                        "point",
                        measured[rows],
                        predicted[rows],
                        ids[rows],
                        np.argsort(predicted[rows], kind="stable"),
                    )
                )
                if not name.startswith("source_block:") or len(rows) < policies.MIN_SOURCE_ROWS:
                    continue
                for subset in ("all", "competitive"):
                    selected = (
                        rows
                        if subset == "all"
                        else rows[measured[rows] <= measured[rows].min() + policies.COMPETITIVE_WINDOW]
                    )
                    for separation in (0.0, policies.NOISE_SEPARATION):
                        directions.append(
                            {
                                "target": target,
                                "method": method,
                                "stratum": name,
                                "subset": subset,
                                "separation_bpb": separation,
                                "eligible_rows": len(rows),
                                "subset_rows": len(selected),
                                **policies.pairwise_changes(measured[selected], predicted[selected], separation),
                            }
                        )
            panel = frame[frame.population.eq("panel_oof")]
            if len(panel) != 280 or panel.row_id.duplicated().any():
                raise ValueError("Each method requires 280 unique OOF predictions")
            for fold in (-1, 0, 1, 2, 3, 4):
                part = panel if fold == -1 else panel[panel.fold.eq(fold)]
                values = part.prediction.to_numpy(float)
                truth = data[f"{target}_aggregate"][part.row_id.to_numpy(int)]
                results.append(
                    scoring.metrics_row(
                        target,
                        method,
                        "panel_oof",
                        "all",
                        "point",
                        truth,
                        values,
                        part.row_id.to_numpy(str),
                        np.argsort(values, kind="stable"),
                        fold=fold,
                    )
                )
    return pd.DataFrame(results), pd.DataFrame(directions), pd.concat(assignments, ignore_index=True)


def error_decomposition(errors: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    """Split weighted task MSE into objective MSE and task deviations that aggregate away."""
    if not np.isclose(weights.sum(), 1):
        raise ValueError("Aggregation weights must sum to one")
    weight_sum = float(weights.sum())
    aggregate = errors @ weights
    deviations = errors - aggregate[:, None] / weight_sum
    atomic_mse = float(np.mean(errors**2 @ weights))
    aggregate_mse = float(np.mean(aggregate**2))
    contrast_mse = float(np.mean(deviations**2 @ weights))
    aggregate_contribution = aggregate_mse / weight_sum
    np.testing.assert_allclose(atomic_mse, aggregate_contribution + contrast_mse, atol=1e-12, rtol=1e-12)
    return {
        "weighted_component_mse": atomic_mse,
        "aggregate_mse": aggregate_mse,
        "aggregate_mse_component_contribution": aggregate_contribution,
        "aggregation_weight_sum": weight_sum,
        "task_deviation_mse": contrast_mse,
        "mean_component_rmse": float(np.sqrt(np.mean(errors**2, axis=0)).mean()),
        "aggregate_rmse": float(np.sqrt(aggregate_mse)),
    }


def atomic_decomposition(output: Path, atomic: dict) -> pd.DataFrame:
    data = benchmark.read_npz(output / "inputs" / "panel.npz")
    rows = []
    for target in benchmark.TARGETS:
        weights = data[f"{target}_aggregation_weights"]
        for method in METHODS:
            errors = []
            for fold in range(5):
                indices, predicted = atomic[(target, method, fold)]
                error = predicted - data[f"{target}_outcomes"][indices]
                errors.append(error)
                rows.append(
                    {
                        "target": target,
                        "method": method,
                        "fold": fold,
                        "rows": len(error),
                        **error_decomposition(error, weights),
                    }
                )
            joined = np.vstack(errors)
            rows.append(
                {
                    "target": target,
                    "method": method,
                    "fold": -1,
                    "rows": len(joined),
                    **error_decomposition(joined, weights),
                }
            )
    return pd.DataFrame(rows)


def paired_contrasts(frame: pd.DataFrame, groups: list[str], metric_names: tuple[str, ...]) -> pd.DataFrame:
    rng = np.random.default_rng(policies.BOOTSTRAP_SEED)
    records = []
    for keys, group in frame.groupby(groups, sort=True, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        for candidate, reference in (
            *contrasts(),
            *(("source_disjoint_method_selection", name) for name in benchmark.BASELINES),
        ):
            left, right = group[group.method.eq(candidate)], group[group.method.eq(reference)]
            if left.empty or right.empty:
                continue
            paired = left.merge(right, on="stratum", suffixes=("_candidate", "_reference"), validate="one_to_one")
            for metric in metric_names:
                delta = (paired[f"{metric}_candidate"] - paired[f"{metric}_reference"]).to_numpy(float)
                delta = delta[np.isfinite(delta)]
                if not len(delta):
                    continue
                bootstrap = delta[rng.integers(0, len(delta), (policies.BOOTSTRAP_DRAWS, len(delta)))].mean(axis=1)
                records.append(
                    {
                        **dict(zip(groups, keys, strict=True)),
                        "candidate": candidate,
                        "reference": reference,
                        "metric": metric,
                        "source_blocks": len(delta),
                        "mean_delta": float(delta.mean()),
                        "ci_low": float(np.quantile(bootstrap, 0.025)),
                        "ci_high": float(np.quantile(bootstrap, 0.975)),
                        "fraction_candidate_lower": float((delta < 0).mean()),
                        "inference": "descriptive equal-connected-source bootstrap, no multiplicity correction",
                    }
                )
    return pd.DataFrame(records)


def diagnostic_summary(diagnostics: pd.DataFrame) -> pd.DataFrame:
    frame = diagnostics.assign(scope=np.where(diagnostics.fold.eq(-1), "final_280", "outer_fits"))
    return frame.groupby(["target", "method", "scope"], as_index=False).agg(
        tasks=("component", "size"),
        unsuccessful=("converged", lambda values: int((~values).sum())),
        maximum_gradient=("projected_gradient_norm", "max"),
        mean_local_dof=("dof", "mean"),
        minimum_local_dof=("dof", "min"),
        maximum_local_dof=("dof", "max"),
        maximum_restart_objective_range=("restart_objective_range", "max"),
        unsuccessful_starts=("restart_unsuccessful", "sum"),
        clipped_task_query_rows=("clipped_query_rows", "sum"),
        negative_task_query_rows=("negative_query_rows", "sum"),
    )


def source_hashes(output: Path, reference: Path) -> dict:
    data = benchmark.read_npz(output / "inputs" / "panel.npz")
    fit_paths = [
        output / "shards" / spec.name / target / f"r0_f{fold}.npz"
        for spec in coupling.SPECS
        for target in benchmark.TARGETS
        for fold in (-1, 0, 1, 2, 3, 4)
    ]
    reference_paths = [
        reference / "baseline_shards" / method / target / f"r0_f{fold}_c{component}.npz"
        for method in benchmark.BASELINES
        for target in benchmark.TARGETS
        for fold in range(5)
        for component in range(len(data[f"{target}_aggregation_weights"]))
    ]
    implementations = (
        Path(__file__),
        Path(coupling.__file__),
        Path(policies.__file__),
        Path(benchmark.__file__),
        Path(scoring.__file__),
    )
    return {
        "implementation_sha256": {
            str(path.relative_to(benchmark.REPO_ROOT)): benchmark.sha256(path) for path in implementations
        },
        "fit_shard_sha256": {str(path.relative_to(output)): benchmark.sha256(path) for path in fit_paths},
        "reference_atomic_shard_sha256": {
            str(path.relative_to(reference)): benchmark.sha256(path) for path in reference_paths
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=coupling.DEFAULT_OUTPUT)
    parser.add_argument("--reference-dir", type=Path, default=benchmark.DEFAULT_OUTPUT)
    args = parser.parse_args()
    output, reference = args.output_dir.resolve(), args.reference_dir.resolve()
    scoring_protocol(output, reference)
    fingerprints = source_hashes(output, reference)
    manifest_path = output / "scoring_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if all(manifest.get(name) == value for name, value in fingerprints.items()) and all(
            (output / name).exists() and benchmark.sha256(output / name) == digest
            for name, digest in manifest["artifact_sha256"].items()
        ):
            print(json.dumps({"status": "cached", "counts": manifest["counts"]}, sort_keys=True))
            return
    predictions, atomic, diagnostics, validity = collect_predictions(output, reference)
    rankings = matched_rankings(output, predictions)
    metrics, directions, assignments = score_predictions(output, predictions)
    policy_metrics, policy_directions, policy_assignments = policies.score_policies(output, rankings)
    eligible = metrics[metrics.stratum.str.startswith("source_block:") & metrics.rows.ge(policies.MIN_SOURCE_ROWS)]
    unconstrained = eligible.assign(epoch_cap=0, kl_penalty=0.0)
    disjoint = policies.source_disjoint_selection(
        unconstrained[unconstrained.method.isin(SELECTABLE_METHODS)], assignments
    )
    policy_disjoint = policies.source_disjoint_selection(
        policy_metrics[policy_metrics.method.isin(SELECTABLE_METHODS)], policy_assignments
    )
    policy_eligible = policy_metrics[
        policy_metrics.stratum.str.startswith("source_block:") & policy_metrics.rows.ge(policies.MIN_SOURCE_ROWS)
    ]
    tables = {
        "metrics.csv": metrics,
        "directional_metrics.csv": directions,
        "source_blocks.csv": assignments,
        "source_disjoint_selection.csv": disjoint,
        "paired_source_contrasts.csv": paired_contrasts(
            pd.concat([eligible, disjoint], ignore_index=True), ["target"], policies.METRICS
        ),
        "directional_source_contrasts.csv": paired_contrasts(
            directions, ["target", "subset", "separation_bpb"], DIRECTION_METRICS
        ),
        "atomic_error_decomposition.csv": atomic_decomposition(output, atomic),
        "fit_diagnostics.csv": diagnostics,
        "fit_diagnostic_summary.csv": diagnostic_summary(diagnostics),
        "prediction_validity.csv": validity,
        "policy_metrics.csv": policy_metrics,
        "policy_directional_metrics.csv": policy_directions,
        "policy_source_disjoint_selection.csv": policy_disjoint,
        "policy_source_contrasts.csv": paired_contrasts(
            pd.concat([policy_eligible, policy_disjoint], ignore_index=True),
            ["target", "epoch_cap", "kl_penalty"],
            policies.METRICS,
        ),
        "policy_directional_source_contrasts.csv": paired_contrasts(
            policy_directions, ["target", "epoch_cap", "subset", "separation_bpb"], DIRECTION_METRICS
        ),
    }
    for name, table in tables.items():
        policies.write_frame(output / name, table)
    benchmark.write_json(
        manifest_path,
        {
            **fingerprints,
            "artifact_sha256": {
                name: benchmark.sha256(output / name)
                for name in (
                    *tables,
                    "predictions.csv",
                    "prediction_hash.json",
                    "policy_rankings.csv",
                    "policy_ranking_hash.json",
                    "scoring_protocol.json",
                )
            },
            "counts": {name: len(table) for name, table in tables.items()},
        },
    )
    print(
        metrics[metrics.population.eq("external_development") & metrics.stratum.eq("optima")][
            ["target", "method", *policies.METRICS]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
