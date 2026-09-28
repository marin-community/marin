# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "scipy", "pandas", "scikit-learn", "joblib", "tabulate"]
# ///
"""Score zero-refit WSPU coupling against frozen source-disjoint development evidence."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.domain_phase_mix.exploratory.two_phase_many import audit_delphi_matched_policies_20260906 as policies
from experiments.domain_phase_mix.exploratory.two_phase_many import benchmark_delphi_selection_20260906 as benchmark
from experiments.domain_phase_mix.exploratory.two_phase_many import score_delphi_coupling_20260906 as decomposition
from experiments.domain_phase_mix.exploratory.two_phase_many import score_delphi_selection_20260906 as scoring

DEFAULT_OUTPUT = benchmark.REFERENCE / "delphi_coupling_followup_20260906" / "incumbent_coupling"
KAPPA_TAGS = ("0", "0p25", "0p5", "1")
COUPLING_METHODS = tuple(f"wspu_coupling_kappa_{tag}" for tag in KAPPA_TAGS)
METHODS = (*benchmark.BASELINES, *COUPLING_METHODS)
SELECTABLE_METHODS = (*benchmark.BASELINES, COUPLING_METHODS[-1])


def scoring_protocol(output: Path, reference: Path) -> None:
    expected = json.loads((reference / "prediction_hash.json").read_text())["sha256"]
    if expected != benchmark.sha256(reference / "predictions.csv"):
        raise ValueError("Frozen reference prediction digest changed")
    protocol = {
        "stage": "second-stage mechanism probe after the separate fixed-basis screen; development only",
        "methods": METHODS,
        "primary_coupling": COUPLING_METHODS[-1],
        "reproduction_control": COUPLING_METHODS[0],
        "sensitivities_only": COUPLING_METHODS[1:3],
        "method_selection_pool": SELECTABLE_METHODS,
        "kappa_tuning": "none; sensitivity kappas never enter source-disjoint selection",
        "evaluation": "repeat-zero original five blocked folds, full bank, optima, connected source blocks",
        "fixed_caps": policies.EPOCH_CAPS,
        "fixed_kl_penalties": policies.KL_PENALTIES,
        "cap_comparison": "all models receive identical eligible rows; regret uses each eligible measured minimum",
        "policy_tuning": "none",
        "competitive_window_bpb": policies.COMPETITIVE_WINDOW,
        "noise_separation_bpb": policies.NOISE_SEPARATION,
        "competitive_status": "truth-conditioned retrospective diagnostic, never a model-selection input",
        "source_selection": "other eligible connected blocks only, equal mean regret@1 then best-of-5 then RMSE",
        "minimum_source_rows": policies.MIN_SOURCE_ROWS,
        "bootstrap_draws": policies.BOOTSTRAP_DRAWS,
        "bootstrap_seed": policies.BOOTSTRAP_SEED,
        "bootstrap_unit": "connected source block; equal-source weighting; no independent pair resampling",
        "inference": "descriptive retrospective development intervals, unadjusted for multiplicity",
        "input_sha256": benchmark.verify_inputs(reference),
        "original_prediction_sha256": expected,
        "producer_protocol_sha256": benchmark.sha256(output / "protocol.json"),
        "producer_prediction_sha256": benchmark.sha256(output / "predictions.csv"),
    }
    path = output / "scoring_protocol.json"
    normalized = json.loads(json.dumps(protocol))
    if path.exists() and json.loads(path.read_text()) != normalized:
        raise ValueError("Frozen scoring protocol changed; use a separate output directory")
    if not path.exists():
        benchmark.write_json(path, protocol)


def artifact_sources(output: Path, reference: Path) -> dict:
    paths = [
        output / "prediction_shards" / target / f"fold_{fold}" / f"kappa_{tag}.npz"
        for target in benchmark.TARGETS
        for fold in (-1, 0, 1, 2, 3, 4)
        for tag in KAPPA_TAGS
    ]
    data = benchmark.read_npz(reference / "inputs" / "panel.npz")
    paths.extend(
        reference / "baseline_shards" / method / target / f"r0_f{fold}_c{component}.npz"
        for method in benchmark.BASELINES
        for target in benchmark.TARGETS
        for fold in range(5)
        for component in range(len(data[f"{target}_aggregation_weights"]))
    )
    sources = (
        Path(__file__),
        Path(policies.__file__),
        Path(benchmark.__file__),
        Path(scoring.__file__),
        Path(decomposition.__file__),
    )
    return {
        "implementation_sha256": {
            str(path.relative_to(benchmark.REPO_ROOT)): benchmark.sha256(path) for path in sources
        },
        "prediction_shard_sha256": {
            str(path.relative_to(benchmark.REPO_ROOT)): benchmark.sha256(path) for path in paths
        },
    }


def collect(output: Path, reference: Path) -> tuple[pd.DataFrame, dict, pd.DataFrame, pd.DataFrame]:
    original = pd.read_csv(reference / "predictions.csv")
    original = original[original.method.isin(benchmark.BASELINES)].copy()
    generated = pd.read_csv(output / "predictions.csv")
    if set(generated.method) != set(COUPLING_METHODS):
        raise ValueError("Expected exactly the four frozen kappa variants")
    predictions = pd.concat([original, generated], ignore_index=True)
    if predictions.duplicated(["method", "target", "population", "repeat", "fold", "row_id"]).any():
        raise ValueError("Duplicate prediction identities")
    if not np.isfinite(predictions.prediction).all():
        raise ValueError("Nonfinite central predictions")
    parity = []
    match_columns = ["target", "population", "repeat", "fold", "row_id"]
    paired = predictions[predictions.method.eq(COUPLING_METHODS[0])].merge(
        predictions[predictions.method.eq(benchmark.BASELINES[0])],
        on=match_columns,
        suffixes=("_zero", "_original"),
        validate="one_to_one",
    )
    if len(paired) != len(generated[generated.method.eq(COUPLING_METHODS[0])]):
        raise ValueError("Missing kappa-zero reproduction coordinates")
    np.testing.assert_allclose(paired.prediction_zero, paired.prediction_original, atol=1e-10, rtol=1e-10)
    for (target, population, fold), frame in paired.groupby(["target", "population", "fold"]):
        parity.append(
            {
                "target": target,
                "population": population,
                "fold": fold,
                "level": "aggregate",
                "max_absolute_difference": float(np.abs(frame.prediction_zero - frame.prediction_original).max()),
            }
        )
    data = benchmark.read_npz(reference / "inputs" / "panel.npz")
    atomic, validity = {}, []
    for target in benchmark.TARGETS:
        weights = data[f"{target}_aggregation_weights"]
        for method in benchmark.BASELINES:
            for fold in range(5):
                shards = [
                    benchmark.read_npz(reference / "baseline_shards" / method / target / f"r0_f{fold}_c{component}.npz")
                    for component in range(len(weights))
                ]
                atomic[(target, method, fold)] = (
                    shards[0]["test"],
                    np.stack([shard["prediction"] for shard in shards], axis=1),
                )
        for tag, method in zip(KAPPA_TAGS, COUPLING_METHODS, strict=True):
            for fold in (-1, 0, 1, 2, 3, 4):
                shard = benchmark.read_npz(output / "prediction_shards" / target / f"fold_{fold}" / f"kappa_{tag}.npz")
                if fold >= 0:
                    atomic[(target, method, fold)] = (shard["test"], shard["atomic_prediction"])
                    if tag == "0":
                        indices, expected = atomic[(target, benchmark.BASELINES[0], fold)]
                        np.testing.assert_array_equal(indices, shard["test"])
                        np.testing.assert_allclose(shard["atomic_prediction"], expected, atol=1e-10, rtol=1e-10)
                        parity.append(
                            {
                                "target": target,
                                "population": "panel_oof",
                                "fold": fold,
                                "level": "atomic",
                                "max_absolute_difference": float(np.abs(shard["atomic_prediction"] - expected).max()),
                            }
                        )
                for population, matrix, values, factors in (
                    ("panel_oof", shard["atomic_prediction"], shard["prediction"], shard["nonpositive_factor_count"]),
                    (
                        "bank",
                        shard["atomic_bank_prediction"],
                        shard["bank_prediction"],
                        shard["bank_nonpositive_factor_count"],
                    ),
                ):
                    np.testing.assert_allclose(matrix @ weights, values, atol=1e-10, rtol=1e-10)
                    validity.append(
                        {
                            "target": target,
                            "method": method,
                            "fold": fold,
                            "population": population,
                            "rows": len(values),
                            "atomic_values": matrix.size,
                            "negative_atomic_values": int((matrix < 0).sum()),
                            "negative_aggregate_values": int((values < 0).sum()),
                            "nonpositive_factors": int(factors.sum()),
                            "task_query_rows_with_nonpositive_factors": int((factors > 0).sum()),
                            "minimum_atomic_prediction": float(matrix.min()) if matrix.size else np.nan,
                        }
                    )
    policies.write_frame(output / "scored_predictions.csv", predictions)
    benchmark.write_json(
        output / "scored_prediction_hash.json",
        {"sha256": benchmark.sha256(output / "scored_predictions.csv"), "labels_joined": False},
    )
    return predictions, atomic, pd.DataFrame(validity), pd.DataFrame(parity)


def matched_rankings(reference: Path, predictions: pd.DataFrame) -> pd.DataFrame:
    data = benchmark.read_npz(reference / "inputs" / "panel.npz")
    natural = policies.natural_proportions(data["inventory"])
    records = []
    for target in benchmark.TARGETS:
        bank = benchmark.read_npz(reference / "inputs" / f"{target}_bank_features.npz")
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
                indices, scores, divergence = policies.policy_order(
                    values, bank["weights"], bank["exposures"], natural, cap, coefficient
                )
                records.extend(
                    {
                        "target": target,
                        "method": method,
                        "epoch_cap": cap,
                        "kl_penalty": coefficient,
                        "policy_rank": rank,
                        "coordinate_id": str(bank["coordinate_id"][row]),
                        "prediction": float(values[row]),
                        "policy_score": float(score),
                        "kl_to_natural": float(kl),
                        "max_exposure": float(bank["exposures"][row].max()),
                    }
                    for rank, (row, score, kl) in enumerate(zip(indices, scores, divergence, strict=True), start=1)
                )
    return pd.DataFrame(records)


def score(
    reference: Path, predictions: pd.DataFrame, atomic: dict
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = benchmark.read_npz(reference / "inputs" / "panel.npz")
    records, directions, assignments, decompositions = [], [], [], []
    for target in benchmark.TARGETS:
        labels = pd.read_csv(reference / "inputs" / f"{target}_bank_labels.csv")
        blocks, memberships = scoring.source_blocks(labels.sources)
        assignments.append(labels[["coordinate_id", "sources"]].assign(target=target, source_block=blocks))
        strata = {
            "all": np.arange(len(labels)),
            "optima": np.flatnonzero([not bool(source & benchmark.INTERVENTIONS) for source in memberships]),
            "interventions": np.flatnonzero([bool(source & benchmark.INTERVENTIONS) for source in memberships]),
        }
        strata.update({f"source_block:{int(block)}": np.flatnonzero(blocks == block) for block in np.unique(blocks)})
        measured, ids = labels.measured_mean_bpb.to_numpy(float), labels.coordinate_id.to_numpy(str)
        for method in METHODS:
            frame = predictions[predictions.target.eq(target) & predictions.method.eq(method)]
            values = (
                frame[frame.population.eq("external_development")]
                .set_index("row_id")
                .loc[ids, "prediction"]
                .to_numpy(float)
            )
            for stratum, rows in strata.items():
                if not len(rows):
                    continue
                records.append(
                    scoring.metrics_row(
                        target,
                        method,
                        "external_development",
                        stratum,
                        "point",
                        measured[rows],
                        values[rows],
                        ids[rows],
                        np.argsort(values[rows], kind="stable"),
                    )
                )
                if not stratum.startswith("source_block:") or len(rows) < policies.MIN_SOURCE_ROWS:
                    continue
                for subset in ("all", "competitive"):
                    subset_rows = (
                        rows
                        if subset == "all"
                        else rows[measured[rows] <= measured[rows].min() + policies.COMPETITIVE_WINDOW]
                    )
                    for separation in (0.0, policies.NOISE_SEPARATION):
                        directions.append(
                            {
                                "target": target,
                                "method": method,
                                "stratum": stratum,
                                "subset": subset,
                                "separation_bpb": separation,
                                "eligible_rows": len(rows),
                                "subset_rows": len(subset_rows),
                                **policies.pairwise_changes(measured[subset_rows], values[subset_rows], separation),
                            }
                        )
            panel = frame[frame.population.eq("panel_oof")]
            if len(panel) != 280 or panel.row_id.duplicated().any():
                raise ValueError("Every method needs exactly 280 unique original OOF predictions")
            for fold in (-1, 0, 1, 2, 3, 4):
                part = panel if fold == -1 else panel[panel.fold.eq(fold)]
                predicted = part.prediction.to_numpy(float)
                observed = data[f"{target}_aggregate"][part.row_id.to_numpy(int)]
                records.append(
                    scoring.metrics_row(
                        target,
                        method,
                        "panel_oof",
                        "all",
                        "point",
                        observed,
                        predicted,
                        part.row_id.to_numpy(str),
                        np.argsort(predicted, kind="stable"),
                        fold=fold,
                    )
                )
                folds = range(5) if fold == -1 else (fold,)
                errors = np.vstack(
                    [
                        atomic[(target, method, value)][1]
                        - data[f"{target}_outcomes"][atomic[(target, method, value)][0]]
                        for value in folds
                    ]
                )
                decompositions.append(
                    {
                        "target": target,
                        "method": method,
                        "fold": fold,
                        "rows": len(errors),
                        **decomposition.error_decomposition(errors, data[f"{target}_aggregation_weights"]),
                    }
                )
    return (
        pd.DataFrame(records),
        pd.DataFrame(directions),
        pd.concat(assignments, ignore_index=True),
        pd.DataFrame(decompositions),
    )


def source_contrasts(frame: pd.DataFrame, groups: list[str], metrics: tuple[str, ...]) -> pd.DataFrame:
    """Report all fixed pairwise source contrasts; kappa one versus zero is primary."""
    result = policies.source_contrasts(frame, groups, metrics)
    for index in np.flatnonzero(result.reference.eq(COUPLING_METHODS[-1]).to_numpy(bool)):
        row = result.iloc[int(index)]
        selected = frame
        for name in groups:
            selected = selected[selected[name].eq(row[name])]
        paired = selected[selected.method.eq(row.candidate)].merge(
            selected[selected.method.eq(row.reference)],
            on="stratum",
            suffixes=("_old_candidate", "_old_reference"),
            validate="one_to_one",
        )
        delta = paired[f"{row.metric}_old_reference"] - paired[f"{row.metric}_old_candidate"]
        delta = delta[np.isfinite(delta)]
        updates = {
            "candidate": row.reference,
            "reference": row.candidate,
            "mean_delta": -row.mean_delta,
            "ci_low": -row.ci_high,
            "ci_high": -row.ci_low,
            "fraction_candidate_lower": (delta < 0).mean(),
        }
        for name, value in updates.items():
            result.at[int(index), name] = value
    result["comparison_status"] = np.where(
        (
            result.candidate.eq(COUPLING_METHODS[-1])
            & result.reference.isin((COUPLING_METHODS[0], benchmark.BASELINES[0]))
        )
        | (
            result.reference.eq(COUPLING_METHODS[-1])
            & result.candidate.isin((COUPLING_METHODS[0], benchmark.BASELINES[0]))
        ),
        "primary_kappa_one_vs_incumbent",
        "fixed_sensitivity_or_reference",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--reference-dir", type=Path, default=benchmark.DEFAULT_OUTPUT)
    args = parser.parse_args()
    output, reference = args.output_dir.resolve(), args.reference_dir.resolve()
    scoring_protocol(output, reference)
    sources = artifact_sources(output, reference)
    manifest_path = output / "scoring_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if all(manifest.get(name) == value for name, value in sources.items()) and all(
            (output / name).exists() and benchmark.sha256(output / name) == digest
            for name, digest in manifest["artifact_sha256"].items()
        ):
            print(json.dumps({"status": "cached", "counts": manifest["counts"]}, sort_keys=True))
            return
    predictions, atomic, validity, parity = collect(output, reference)
    rankings = matched_rankings(reference, predictions)
    policies.write_frame(output / "policy_rankings.csv", rankings)
    benchmark.write_json(
        output / "policy_ranking_hash.json",
        {"sha256": benchmark.sha256(output / "policy_rankings.csv"), "labels_joined": False},
    )
    metrics, directions, assignments, decompositions = score(reference, predictions, atomic)
    policy_metrics, policy_directions, policy_assignments = policies.score_policies(reference, rankings)
    eligible = metrics[metrics.stratum.str.startswith("source_block:") & metrics.rows.ge(policies.MIN_SOURCE_ROWS)]
    policy_eligible = policy_metrics[
        policy_metrics.stratum.str.startswith("source_block:") & policy_metrics.rows.ge(policies.MIN_SOURCE_ROWS)
    ]
    disjoint = policies.source_disjoint_selection(
        eligible[eligible.method.isin(SELECTABLE_METHODS)].assign(epoch_cap=0, kl_penalty=0.0), assignments
    )
    policy_disjoint = policies.source_disjoint_selection(
        policy_eligible[policy_eligible.method.isin(SELECTABLE_METHODS)], policy_assignments
    )
    tables = {
        "metrics.csv": metrics,
        "directional_metrics.csv": directions,
        "source_blocks.csv": assignments,
        "source_disjoint_selection.csv": disjoint,
        "paired_source_contrasts.csv": source_contrasts(
            pd.concat([eligible, disjoint], ignore_index=True), ["target"], policies.METRICS
        ),
        "directional_source_contrasts.csv": source_contrasts(
            directions, ["target", "subset", "separation_bpb"], decomposition.DIRECTION_METRICS
        ),
        "atomic_error_decomposition.csv": decompositions,
        "prediction_validity.csv": validity,
        "kappa_zero_reproduction.csv": parity,
        "policy_metrics.csv": policy_metrics,
        "policy_directional_metrics.csv": policy_directions,
        "policy_source_disjoint_selection.csv": policy_disjoint,
        "policy_source_contrasts.csv": source_contrasts(
            pd.concat([policy_eligible, policy_disjoint], ignore_index=True),
            ["target", "epoch_cap", "kl_penalty"],
            policies.METRICS,
        ),
        "policy_directional_source_contrasts.csv": source_contrasts(
            policy_directions, ["target", "epoch_cap", "subset", "separation_bpb"], decomposition.DIRECTION_METRICS
        ),
    }
    for name, frame in tables.items():
        policies.write_frame(output / name, frame)
    benchmark.write_json(
        manifest_path,
        {
            **sources,
            "artifact_sha256": {
                name: benchmark.sha256(output / name)
                for name in (
                    *tables,
                    "scored_predictions.csv",
                    "scored_prediction_hash.json",
                    "policy_rankings.csv",
                    "policy_ranking_hash.json",
                    "scoring_protocol.json",
                )
            },
            "counts": {name: len(frame) for name, frame in tables.items()},
        },
    )
    print(
        metrics[metrics.population.eq("external_development") & metrics.stratum.eq("optima")][
            ["target", "method", *policies.METRICS]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
