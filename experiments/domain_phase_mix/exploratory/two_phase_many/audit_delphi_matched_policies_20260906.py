# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "scipy", "pandas", "scikit-learn", "joblib", "tabulate"]
# ///
"""Audit fixed matched policies and within-source directions on the frozen Delphi bank.

This reads local frozen artifacts only. It neither refits a surrogate nor submits
training or evaluation jobs. Policy ordering is persisted before labels are joined.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import rel_entr

from experiments.domain_phase_mix.exploratory.two_phase_many import benchmark_delphi_selection_20260906 as benchmark
from experiments.domain_phase_mix.exploratory.two_phase_many import score_delphi_selection_20260906 as scoring

DEFAULT_OUTPUT = benchmark.REFERENCE / "delphi_coupling_followup_20260906" / "policies"
EPOCH_CAPS = (4, 6, 8, 16)
KL_PENALTIES = (0.0, 0.005, 0.02)
COMPETITIVE_WINDOW = 0.02
NOISE_SEPARATION = 0.01
MIN_SOURCE_ROWS = 5
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 20260906
METRICS = (
    "regret_at_1",
    "best_of_5_regret",
    "best_of_10_regret",
    "selected_rank",
    "optimism",
    "rmse",
    "spearman",
)


def natural_proportions(inventory: np.ndarray) -> np.ndarray:
    """Recover unique-token proportions from positive full-budget exposure multipliers."""
    if not np.isfinite(inventory).all() or np.any(inventory <= 0):
        raise ValueError("Exposure inventory must be finite and strictly positive")
    inverse = 1 / inventory
    return inverse / inverse.sum()


def policy_order(
    prediction: np.ndarray,
    weights: np.ndarray,
    exposures: np.ndarray,
    natural: np.ndarray,
    epoch_cap: float,
    kl_penalty: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return eligible indices, central-loss-plus-KL scores, and KL in stable policy order."""
    eligible = np.flatnonzero(exposures.max(axis=1) <= epoch_cap + 1e-8)
    divergence = rel_entr(weights[eligible], natural).sum(axis=1)
    score = prediction[eligible] + kl_penalty * divergence
    order = np.argsort(score, kind="stable")
    return eligible[order], score[order], divergence[order]


def write_frame(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_suffix(".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def freeze_protocol(output: Path, source: Path) -> dict:
    """Persist the fixed grid and source identities before any new outcome scoring."""
    source_hashes = benchmark.verify_inputs(source)
    protocol = {
        "canonical_training_rows": 280,
        "surrogate_refits": False,
        "methods": list(benchmark.BASELINES),
        "epoch_caps": list(EPOCH_CAPS),
        "kl_penalties": list(KL_PENALTIES),
        "policy": "central predicted BPB + coefficient * KL(weights || natural)",
        "natural": "normalized reciprocal frozen exposure multiplier; canonical proportional anchor 0.905353 epochs",
        "cap_units": "frozen canonical exposure units; no independent physical-token re-estimation",
        "central_metrics": "RMSE, Spearman and optimism use unpenalized predicted BPB",
        "candidate_set": "same frozen bank and same eligible rows for every method at a given cap",
        "cap_boundary_tolerance": 1e-8,
        "tie_break": "stable frozen coordinate order",
        "policy_tuning": "none; every cap/coefficient combination is reported",
        "source_blocks": "connect all source memberships before cap filtering, including shared controls",
        "source_disjoint_selection": (
            "within each fixed cap/coefficient, choose method on all other eligible blocks "
            "by equal-block mean regret@1, then best-of-5, then RMSE"
        ),
        "minimum_source_rows": MIN_SOURCE_ROWS,
        "directions": "all unordered pairs within each connected source block; central predicted changes",
        "competitive_subset": "diagnostic only: observed loss <= observed best within eligible source block + window",
        "competitive_window_bpb": COMPETITIVE_WINDOW,
        "noise_separation_bpb": NOISE_SEPARATION,
        "noise_rule": "diagnostic pair filter abs(observed change) >= threshold; threshold is not a confidence interval",
        "bootstrap": {
            "draws": BOOTSTRAP_DRAWS,
            "seed": BOOTSTRAP_SEED,
            "unit": "connected source block, equal block weights",
            "inference": "descriptive historical development intervals, unadjusted for multiplicity",
        },
        "input_sha256": source_hashes,
        "prediction_sha256": benchmark.sha256(source / "predictions.csv"),
        "evidence_status": "development only; no prospective successor claim",
    }
    output.mkdir(parents=True, exist_ok=True)
    path = output / "protocol.json"
    if path.exists():
        if json.loads(path.read_text()) != protocol:
            raise ValueError("Frozen protocol or input identities changed; use a distinct output directory")
    else:
        benchmark.write_json(path, protocol)
    return protocol


def rank_policies(source: Path, output: Path) -> pd.DataFrame:
    """Rank every fixed policy using features and existing predictions, without bank outcomes."""
    panel = benchmark.read_npz(source / "inputs" / "panel.npz")
    natural = natural_proportions(panel["inventory"])
    np.testing.assert_allclose(panel["weights"] * panel["inventory"], panel["exposures"], atol=1e-10, rtol=1e-10)
    predictions = pd.read_csv(source / "predictions.csv")
    rows = []
    availability = []
    for target in benchmark.TARGETS:
        bank = benchmark.read_npz(source / "inputs" / f"{target}_bank_features.npz")
        np.testing.assert_allclose(bank["weights"].sum(axis=1), 1, atol=1e-8, rtol=0)
        np.testing.assert_allclose(bank["weights"] * panel["inventory"], bank["exposures"], atol=1e-10, rtol=1e-10)
        for cap in EPOCH_CAPS:
            availability.append(
                {"target": target, "epoch_cap": cap, "rows": int((bank["exposures"].max(axis=1) <= cap + 1e-8).sum())}
            )
        for method in benchmark.BASELINES:
            frame = (
                predictions[
                    predictions.target.eq(target)
                    & predictions.method.eq(method)
                    & predictions.population.eq("external_development")
                ]
                .set_index("row_id")
                .loc[bank["coordinate_id"]]
            )
            prediction = frame.prediction.to_numpy(float)
            if not np.isfinite(prediction).all():
                raise ValueError("Nonfinite frozen prediction")
            for cap, coefficient in itertools.product(EPOCH_CAPS, KL_PENALTIES):
                indices, values, divergence = policy_order(
                    prediction, bank["weights"], bank["exposures"], natural, cap, coefficient
                )
                rows.extend(
                    {
                        "target": target,
                        "method": method,
                        "epoch_cap": cap,
                        "kl_penalty": coefficient,
                        "policy_rank": rank,
                        "coordinate_id": str(bank["coordinate_id"][row]),
                        "prediction": float(prediction[row]),
                        "policy_score": float(value),
                        "kl_to_natural": float(kl),
                        "max_exposure": float(bank["exposures"][row].max()),
                    }
                    for rank, (row, value, kl) in enumerate(zip(indices, values, divergence, strict=True), start=1)
                )
    result = pd.DataFrame(rows)
    write_frame(output / "policy_rankings.csv", result)
    write_frame(output / "availability.csv", pd.DataFrame(availability))
    write_frame(
        output / "natural_proportions.csv",
        pd.DataFrame({"bucket": panel["buckets"], "natural_share": natural, "exposure_multiplier": panel["inventory"]}),
    )
    benchmark.write_json(
        output / "ranking_hash.json",
        {"sha256": benchmark.sha256(output / "policy_rankings.csv"), "labels_joined": False},
    )
    return result


def pairwise_changes(measured: np.ndarray, predicted: np.ndarray, separation: float) -> dict[str, float | int]:
    """Measure signed-change accuracy without treating dependent pairs as independent samples."""
    left, right = np.triu_indices(len(measured), k=1)
    actual = measured[right] - measured[left]
    estimated = predicted[right] - predicted[left]
    selected = np.abs(actual) >= separation
    actual, estimated = actual[selected], estimated[selected]
    if not len(actual):
        return {
            "pairs": 0,
            "delta_rmse": np.nan,
            "delta_mae": np.nan,
            "direction_accuracy": np.nan,
            "mean_observed_magnitude": np.nan,
            "mean_predicted_magnitude": np.nan,
        }
    same = np.sign(actual) == np.sign(estimated)
    tied = (actual == 0) | (estimated == 0)
    accuracy = np.where(tied, 0.5, same.astype(float))
    return {
        "pairs": len(actual),
        "delta_rmse": float(np.sqrt(np.mean((estimated - actual) ** 2))),
        "delta_mae": float(np.mean(np.abs(estimated - actual))),
        "direction_accuracy": float(accuracy.mean()),
        "mean_observed_magnitude": float(np.abs(actual).mean()),
        "mean_predicted_magnitude": float(np.abs(estimated).mean()),
    }


def score_policies(source: Path, rankings: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics = []
    directions = []
    assignments = []
    for target in benchmark.TARGETS:
        labels = pd.read_csv(source / "inputs" / f"{target}_bank_labels.csv")
        blocks, memberships = scoring.source_blocks(labels.sources)
        labels = labels.assign(
            source_block=blocks, optima=[not bool(membership & benchmark.INTERVENTIONS) for membership in memberships]
        )
        assignments.append(labels[["coordinate_id", "sources", "source_block", "optima"]].assign(target=target))
        for (method, cap, coefficient), frame in rankings[rankings.target.eq(target)].groupby(
            ["method", "epoch_cap", "kl_penalty"], sort=True
        ):
            joined = frame.merge(labels, on="coordinate_id", how="left", validate="one_to_one").sort_values(
                "policy_rank"
            )
            if joined.measured_mean_bpb.isna().any():
                raise ValueError("Missing historical outcome")
            strata = {"all": joined, "optima": joined[joined.optima]}
            strata.update({f"source_block:{int(block)}": part for block, part in joined.groupby("source_block")})
            for name, part in strata.items():
                if not len(part):
                    continue
                result = scoring.metrics_row(
                    target,
                    str(method),
                    "external_development",
                    name,
                    "central_plus_kl",
                    part.measured_mean_bpb.to_numpy(float),
                    part.prediction.to_numpy(float),
                    part.coordinate_id.to_numpy(str),
                    np.arange(len(part)),
                )
                result.update(
                    {
                        "epoch_cap": cap,
                        "kl_penalty": coefficient,
                        "selected_policy_score": float(part.policy_score.iloc[0]),
                        "selected_kl": float(part.kl_to_natural.iloc[0]),
                        "selected_max_exposure": float(part.max_exposure.iloc[0]),
                    }
                )
                metrics.append(result)
                if coefficient != 0 or not name.startswith("source_block:") or len(part) < MIN_SOURCE_ROWS:
                    continue
                for subset in ("all", "competitive"):
                    competitive = (
                        part
                        if subset == "all"
                        else part[part.measured_mean_bpb <= part.measured_mean_bpb.min() + COMPETITIVE_WINDOW]
                    )
                    measured = competitive.measured_mean_bpb.to_numpy(float)
                    predicted = competitive.prediction.to_numpy(float)
                    for separation in (0.0, NOISE_SEPARATION):
                        directions.append(
                            {
                                "target": target,
                                "method": method,
                                "epoch_cap": cap,
                                "stratum": name,
                                "subset": subset,
                                "separation_bpb": separation,
                                "eligible_rows": len(part),
                                "subset_rows": len(competitive),
                                **pairwise_changes(measured, predicted, separation),
                            }
                        )
    return pd.DataFrame(metrics), pd.DataFrame(directions), pd.concat(assignments, ignore_index=True)


def source_disjoint_selection(metrics: pd.DataFrame, assignments: pd.DataFrame) -> pd.DataFrame:
    """Choose only the model on other connected blocks within each already fixed policy."""
    eligible = metrics[metrics.stratum.str.startswith("source_block:") & metrics.rows.ge(MIN_SOURCE_ROWS)]
    decisions = []
    for (target, _cap, _coefficient), frame in eligible.groupby(["target", "epoch_cap", "kl_penalty"], sort=True):
        for block in sorted(frame.stratum.unique()):
            training = frame[frame.stratum.ne(block)]
            if training.stratum.nunique() < 2:
                continue
            rank = (
                training.groupby("method")[["regret_at_1", "best_of_5_regret", "rmse"]]
                .mean()
                .sort_values(["regret_at_1", "best_of_5_regret", "rmse"], kind="stable")
            )
            choice = str(rank.index[0])
            record = frame[frame.stratum.eq(block) & frame.method.eq(choice)].iloc[0].to_dict()
            test_block = int(block.split(":")[1])
            train_blocks = [int(value.split(":")[1]) for value in training.stratum.unique()]
            sources = assignments[assignments.target.eq(target)]
            train_sources = set(";".join(sources[sources.source_block.isin(train_blocks)].sources).split(";"))
            test_sources = set(";".join(sources[sources.source_block.eq(test_block)].sources).split(";"))
            if train_sources & test_sources:
                raise ValueError("Source overlap across a development selection split")
            record.update(
                {
                    "method": "source_disjoint_method_selection",
                    "chosen_method": choice,
                    "population": "source_disjoint_method_selection",
                    "training_source_blocks": len(train_blocks),
                    "train_sources": ";".join(sorted(train_sources)),
                    "test_sources": ";".join(sorted(test_sources)),
                }
            )
            decisions.append(record)
    return pd.DataFrame(decisions)


def source_contrasts(frame: pd.DataFrame, group_columns: list[str], metrics: tuple[str, ...]) -> pd.DataFrame:
    """Bootstrap equal-source paired differences, keeping all dependent within-source pairs together."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows = []
    for keys, group in frame.groupby(group_columns, sort=True, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        methods = sorted(group.method.unique())
        pairs = list(itertools.combinations(methods, 2))
        for candidate, reference in pairs:
            paired = group[group.method.eq(candidate)].merge(
                group[group.method.eq(reference)],
                on="stratum",
                suffixes=("_candidate", "_reference"),
                validate="one_to_one",
            )
            for metric in metrics:
                delta = (paired[f"{metric}_candidate"] - paired[f"{metric}_reference"]).to_numpy(float)
                delta = delta[np.isfinite(delta)]
                if not len(delta):
                    continue
                bootstrap = delta[rng.integers(0, len(delta), size=(BOOTSTRAP_DRAWS, len(delta)))].mean(axis=1)
                rows.append(
                    {
                        **dict(zip(group_columns, keys, strict=True)),
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
    return pd.DataFrame(rows)


def run(source: Path, output: Path) -> dict:
    freeze_protocol(output, source)
    implementation = {
        str(path.relative_to(benchmark.REPO_ROOT)): benchmark.sha256(path)
        for path in (Path(__file__), Path(benchmark.__file__), Path(scoring.__file__))
    }
    complete_path = output / "complete.json"
    if complete_path.exists():
        complete = json.loads(complete_path.read_text())
        if complete["implementation_sha256"] == implementation and all(
            benchmark.sha256(output / name) == digest for name, digest in complete["artifact_sha256"].items()
        ):
            return {"status": "cached", **complete["counts"]}
    rankings = rank_policies(source, output)
    metrics, directions, assignments = score_policies(source, rankings)
    disjoint = source_disjoint_selection(metrics, assignments)
    source_metrics = metrics[metrics.stratum.str.startswith("source_block:") & metrics.rows.ge(MIN_SOURCE_ROWS)]
    tables = {
        "metrics.csv": metrics,
        "directional_metrics.csv": directions,
        "source_blocks.csv": assignments,
        "source_disjoint_selection.csv": disjoint,
        "paired_source_contrasts.csv": source_contrasts(source_metrics, ["target", "epoch_cap", "kl_penalty"], METRICS),
        "directional_source_contrasts.csv": source_contrasts(
            directions,
            ["target", "epoch_cap", "subset", "separation_bpb"],
            ("delta_rmse", "delta_mae", "direction_accuracy", "mean_predicted_magnitude"),
        ),
        "source_disjoint_contrasts.csv": source_contrasts(
            pd.concat([source_metrics, disjoint], ignore_index=True), ["target", "epoch_cap", "kl_penalty"], METRICS
        ),
    }
    for name, table in tables.items():
        write_frame(output / name, table)
    counts = {
        "policy_rankings": len(rankings),
        "selection_metrics": len(metrics),
        "directional_metrics": len(directions),
        "source_disjoint_decisions": len(disjoint),
    }
    benchmark.write_json(
        complete_path,
        {
            "counts": counts,
            "implementation_sha256": implementation,
            "artifact_sha256": {
                path.name: benchmark.sha256(path)
                for path in sorted(output.iterdir())
                if path.is_file() and path.name != "complete.json"
            },
        },
    )
    return {"status": "computed", **counts}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=benchmark.DEFAULT_OUTPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(args.source_dir.resolve(), args.output_dir.resolve()), sort_keys=True))


if __name__ == "__main__":
    main()
