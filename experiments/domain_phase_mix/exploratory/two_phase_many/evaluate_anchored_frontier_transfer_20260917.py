# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["numpy==2.3.5", "pandas==2.2.2", "scipy==1.17.0", "fsspec==2026.1.0", "threadpoolctl==3.6.0"]
# ///
"""Score frozen continuation fits on archived matched Delphi frontier outcomes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import benchmark_crossed_mariner_20260912 as crossed
import numpy as np
import pandas as pd
from fit_two_phase_link_spines_20260907 import load_module, write_json_atomic
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits

BASE = Path(__file__).resolve().parent
REFERENCE = BASE / "reference_outputs"
ROOT = REFERENCE / "two_phase_mariner_transfer_20260912"
CALIBRATION = REFERENCE / "delphi_phase_frontier_calibration_20260910"
OUTPUT = ROOT / "frontier_transfer_20260917"
SOURCE = ROOT / "crossed/sources/mixture_selection.py"
ALPHA = 2400 / 3007
SEED = 20260917
BOOTSTRAPS = 4000
CELL_NAMES = {"19b0c3282a8c": "Uncheatable anchor", "17a03f4d49cd": "Table-9 anchor"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def input_paths() -> list[Path]:
    return [
        Path(__file__),
        Path(crossed.__file__),
        BASE / "fixed_checkpoint_wspu_models_20260907.py",
        BASE / "fit_two_phase_link_spines_20260907.py",
        SOURCE,
        ROOT / "crossed/prepared.json",
        ROOT / "anchored/finite_composite_terms.csv",
        ROOT / "frozen_baselines/manifest.json",
        crossed.INPUT / "data/rows.csv",
        crossed.INPUT / "data/arrays.npz",
        CALIBRATION / "snapshot.json",
        CALIBRATION / "inputs/observations.csv",
        CALIBRATION / "inputs/coordinates.npz",
        CALIBRATION / "inputs/frontier_metadata.csv",
        CALIBRATION / "inputs/mirror_pairs.csv",
        CALIBRATION / "fits/mariner_uncheatable.json",
        CALIBRATION / "fits/canonical/uncheatable_predictions.csv",
        CALIBRATION / "matched_seed_phase_gains.csv",
        *[ROOT / "crossed/fits" / state / "MTP-002.json" for state in crossed.STATES],
    ]


def rank_correlation(observed: np.ndarray, prediction: np.ndarray) -> float:
    if len(observed) < 2 or np.ptp(prediction) < 1e-12 or np.ptp(observed) < 1e-12:
        return float("nan")
    return float(spearmanr(observed, prediction).statistic)


def gain_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    # Positive gain means tied BPB minus asymmetric BPB: prefer phase only above zero.
    regret = np.where(predicted > 1e-12, np.maximum(-observed, 0), np.maximum(observed, 0))
    return {
        "gain_rmse": float(np.sqrt(np.mean((predicted - observed) ** 2))),
        "mean_observed_gain": float(observed.mean()),
        "mean_predicted_gain": float(predicted.mean()),
        "gain_optimism": float(np.mean(predicted - observed)),
        "matched_regret": float(regret.mean()),
        "prefer_phase_fraction": float(np.mean(predicted > 1e-12)),
        "gain_spearman": rank_correlation(observed, predicted),
    }


def continuation_predictions(
    saved: dict[str, Any], weights: np.ndarray, inventory: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    aggregate = ALPHA * weights[:, 0] + (1 - ALPHA) * weights[:, 1]
    actual = crossed.predict_fit(
        saved["fit"], ALPHA * weights[:, 0] * inventory, (1 - ALPHA) * weights[:, 1] * inventory, SOURCE
    )
    tied = crossed.predict_fit(saved["fit"], ALPHA * aggregate * inventory, (1 - ALPHA) * aggregate * inventory, SOURCE)
    return actual, tied


def predictions() -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, Any]]:
    snapshot = json.loads((CALIBRATION / "snapshot.json").read_text())
    for name, digest in snapshot["frozen_hashes"].items():
        assert sha(CALIBRATION / name) == digest, name
    branch, branch_arrays = crossed.load_data()
    prepared = json.loads((ROOT / "crossed/prepared.json").read_text())
    baseline_manifest = json.loads((ROOT / "frozen_baselines/manifest.json").read_text())
    frame = pd.read_csv(CALIBRATION / "inputs/observations.csv")
    baseline = pd.read_csv(CALIBRATION / "fits/canonical/uncheatable_predictions.csv")
    assert baseline.run_id.equals(frame.run_id) and frame.run_id.is_unique
    assert np.allclose(baseline.measured, frame.uncheatable, rtol=0, atol=1e-14)
    with np.load(CALIBRATION / "inputs/coordinates.npz") as stored:
        arrays = {key: stored[key].copy() for key in stored.files}
    weights, aggregate = arrays["weights"], arrays["aggregate"]
    assert np.allclose(weights.sum(axis=2), 1, rtol=0, atol=1e-12)
    assert np.allclose(aggregate, ALPHA * weights[:, 0] + (1 - ALPHA) * weights[:, 1], rtol=0, atol=1e-14)
    source = load_module(SOURCE, "frontier_transfer_mariner")
    anchor_path = CALIBRATION / "fits/mariner_uncheatable.json"
    assert sha(anchor_path) == baseline_manifest["inputs"][str(anchor_path)]
    assert (
        sha(CALIBRATION / "fits/canonical/uncheatable_predictions.csv")
        == baseline_manifest["inputs"][str(CALIBRATION / "fits/canonical/uncheatable_predictions.csv")]
    )
    anchor = source.ObjectiveFit.from_json(json.loads(anchor_path.read_text()))
    assert anchor.buckets == tuple(branch_arrays["bucket_names"])
    physical_inventory = anchor.inventory * baseline_manifest["coordinate_sensitivity"]["mariner_inventory_scale"]
    frame["A"] = anchor.predict(aggregate)
    frame["HPR"] = baseline.hpr
    frame["HPR+A"] = baseline.replacement
    assert np.max(np.abs(frame.A - baseline.mariner)) < 1e-12
    frame["gain::zero"] = 0.0
    frame["gain::HPR"] = baseline.hpr_tied - baseline.hpr
    known = pd.read_csv(ROOT / "anchored/finite_composite_terms.csv")
    replay_errors = []
    training_policies = []
    guards = []
    for index, state in enumerate(crossed.STATES):
        saved = json.loads((ROOT / "crossed/fits" / state / "MTP-002.json").read_text())
        assert saved["input_hashes"] == prepared["hashes"]
        assert saved["fit"]["source_sha256"] == sha(SOURCE)
        train = np.asarray(saved["training_rows"], int)
        assert not branch.state_id.iloc[train].eq(state).any()
        train_weights = np.stack([branch_arrays["phase0_weight"][train], branch_arrays["phase1_weight"][train]], axis=1)
        assert np.allclose(
            branch_arrays["phase0_epochs"][train], ALPHA * train_weights[:, 0] * physical_inventory, rtol=0, atol=1e-9
        )
        training_policies.append(train_weights)
        actual, tied = continuation_predictions(saved, weights, physical_inventory)
        frame[f"W{index}"] = actual
        frame[f"F{index}"] = frame.A.to_numpy() + actual - tied
        frame[f"gain::F{index}"] = tied - actual
        assert np.max(np.abs(frame.loc[frame.tied, f"F{index}"] - frame.loc[frame.tied, "A"])) < 1e-12
        frame[f"prefix_tv::{index}"] = cdist(weights[:, 0], train_weights[:, 0], "cityblock").min(axis=1) / 2
        frame[f"action_tv::{index}"] = cdist(weights[:, 1], train_weights[:, 1], "cityblock").min(axis=1) / 2
        frame[f"tied_query_tv::{index}"] = cdist(aggregate, train_weights[:, 0], "cityblock").min(axis=1) / 2
        scaling = np.asarray([ALPHA, 1 - ALPHA])[None, :, None]
        frame[f"policy_tv::{index}"] = (
            cdist(
                (weights * scaling).reshape(len(frame), -1),
                (train_weights * scaling).reshape(len(train), -1),
                "cityblock",
            ).min(axis=1)
            / 2
        )
        check = known[known.state_id.eq(state)]
        check_rows = check.row.to_numpy(int)
        check_weights = np.stack(
            [branch_arrays["phase0_weight"][check_rows], branch_arrays["phase1_weight"][check_rows]], axis=1
        )
        check_actual, check_tied = continuation_predictions(saved, check_weights, physical_inventory)
        check_aggregate = ALPHA * check_weights[:, 0] + (1 - ALPHA) * check_weights[:, 1]
        replay_errors.extend(np.abs(anchor.predict(check_aggregate) + check_actual - check_tied - check.prediction))
        for label, prediction in (("actual", actual), ("tied", tied)):
            deficit = prediction - saved["fit"]["floor"]
            assert np.all(deficit > 0) and np.isfinite(deficit).all()
            latent = np.log(deficit)
            guards.append({"fit": index, "query": label, "guard_count": int((np.abs(latent) >= source.LOG_CLIP).sum())})
    training = np.concatenate(training_policies)
    frame["W_training_overlap"] = (
        cdist(weights.reshape(len(weights), -1), training.reshape(len(training), -1), "chebyshev").min(axis=1) < 1e-9
    )
    frame["W_prefix_overlap"] = cdist(weights[:, 0], training[:, 0], "chebyshev").min(axis=1) < 1e-9
    frame["primary"] = frame.external & ~frame.training_overlap & ~frame.W_training_overlap & ~frame.W_prefix_overlap
    assert max(replay_errors) < 1e-12
    audit = {
        "fit_labels": {f"F{i}": state for i, state in enumerate(crossed.STATES)},
        "branch_replay_max_error": max(replay_errors),
        "anchor_replay_max_error": float(np.max(np.abs(frame.A - baseline.mariner))),
        "guards": guards,
        "bank_rows": len(frame),
        "primary_rows": int(frame.primary.sum()),
        "primary_policies": int(frame.loc[frame.primary, "policy_id"].nunique()),
        "external_W_policy_overlap": int(frame.loc[frame.external, "W_training_overlap"].sum()),
        "external_W_prefix_overlap": int(frame.loc[frame.external, "W_prefix_overlap"].sum()),
        "inventory_scale": baseline_manifest["coordinate_sensitivity"]["mariner_inventory_scale"],
    }
    return frame, arrays, audit


def matched_pairs(frame: pd.DataFrame, arrays: dict[str, np.ndarray]) -> tuple[pd.DataFrame, dict[str, str]]:
    previous = pd.read_csv(CALIBRATION / "matched_seed_phase_gains.csv")
    previous = previous[previous.context.eq("canonical") & previous.objective.eq("uncheatable")]
    assert len(previous) == 450 and previous.row_id.is_unique
    meta = pd.read_csv(CALIBRATION / "inputs/frontier_metadata.csv").set_index("heldout_id").to_dict("index")
    archive = frame[frame.cohort.eq("archive")].reset_index(names="position").set_index("archive_id")
    assert archive.index.is_unique
    archive_rows = archive.to_dict("index")
    records = []
    cells: dict[str, str] = {}
    for row in previous.to_dict("records"):
        row_id, control_id = str(row["row_id"]), str(row["control_id"])
        policy, control = archive_rows[row_id], archive_rows[control_id]
        assert policy["primary"] and control["primary"] and not policy["tied"] and control["tied"]
        assert policy["source"] == control["source"] == row["source"]
        assert policy["aggregate_id"] == control["aggregate_id"]
        assert np.max(np.abs(arrays["aggregate"][policy["position"]] - arrays["aggregate"][control["position"]])) < 1e-9
        for field in ("dry_run_data_seed", "dry_run_trainer_seed", "parameter_count", "global_step", "phase_0_fraction"):
            left, right = meta[row_id][field], meta[control_id][field]
            assert pd.notna(left) and left == right, (row_id, field, left, right)
        observed = float(control["uncheatable"]) - float(policy["uncheatable"])
        assert abs(observed - float(row["observed_gain"])) < 1e-12
        cells[str(row["cell"])] = str(policy["aggregate_id"])
        common = {
            "cell": str(row["cell"]),
            "row_id": row_id,
            "control_id": control_id,
            "source": str(row["source"]),
            "data_seed": meta[row_id]["dry_run_data_seed"],
            "trainer_seed": meta[row_id]["dry_run_trainer_seed"],
            "policy_id": policy["policy_id"],
            "observed_gain": observed,
        }
        for model in ("zero", "HPR", *[f"F{i}" for i in range(6)]):
            predicted = policy[f"gain::{model}"]
            assert abs(control[f"gain::{model}"]) < 1e-10
            if model == "HPR":
                assert abs(predicted - float(row["predicted_gain"])) < 1e-12
            records.append({**common, "model": model, "predicted_gain": predicted})
    return pd.DataFrame(records), cells


def paired_summaries(pairs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries, intervals = [], []
    for cell, population in pairs.groupby("cell"):
        unique = population.drop_duplicates("row_id").reset_index(drop=True)
        models = list(dict.fromkeys(population.model))
        wide = population.pivot(index="row_id", columns="model", values="predicted_gain").loc[unique.row_id]
        observed = unique.observed_gain.to_numpy()
        blocks = list(unique.groupby(["source", "control_id"]).indices.items())
        rng = np.random.default_rng(SEED)
        counts = np.zeros((BOOTSTRAPS, len(blocks)), int)
        for source in unique.source.unique():
            positions = np.asarray(
                [i for i, (_, indices) in enumerate(blocks) if unique.source.iloc[indices[0]] == source]
            )
            counts[:, positions] = rng.multinomial(
                len(positions), np.full(len(positions), 1 / len(positions)), BOOTSTRAPS
            )
        denominator = counts @ np.asarray([len(indices) for _, indices in blocks])
        bootstrap_rmse, bootstrap_regret = {}, {}
        for model in models:
            predicted = wide[model].to_numpy()
            regret = np.where(predicted > 1e-12, np.maximum(-observed, 0), np.maximum(observed, 0))
            errors = (predicted - observed) ** 2
            bootstrap_rmse[model] = np.sqrt(counts @ np.asarray([errors[ix].sum() for _, ix in blocks]) / denominator)
            bootstrap_regret[model] = counts @ np.asarray([regret[ix].sum() for _, ix in blocks]) / denominator
            summaries.append(
                {
                    "cell": cell,
                    "source": "all",
                    "model": model,
                    "rows": len(unique),
                    "control_blocks": len(blocks),
                    **gain_metrics(observed, predicted),
                }
            )
            for source, indices in unique.groupby("source").indices.items():
                summaries.append(
                    {
                        "cell": cell,
                        "source": source,
                        "model": model,
                        "rows": len(indices),
                        "control_blocks": unique.iloc[indices].control_id.nunique(),
                        **gain_metrics(observed[indices], predicted[indices]),
                    }
                )
        for model in models:
            for reference in ("zero", "HPR"):
                if model == reference:
                    continue
                for metric, values in (("gain_rmse", bootstrap_rmse), ("matched_regret", bootstrap_regret)):
                    change = values[model] - values[reference]
                    intervals.append(
                        {
                            "cell": cell,
                            "model": model,
                            "reference": reference,
                            "metric": metric,
                            "lower": float(np.quantile(change, 0.025)),
                            "upper": float(np.quantile(change, 0.975)),
                            "draws": BOOTSTRAPS,
                            "control_blocks": len(blocks),
                        }
                    )
    return pd.DataFrame(summaries), pd.DataFrame(intervals)


def mirrors(frame: pd.DataFrame, arrays: dict[str, np.ndarray]) -> tuple[pd.DataFrame, pd.DataFrame]:
    pairs = pd.read_csv(CALIBRATION / "inputs/mirror_pairs.csv")
    archive = frame[frame.cohort.eq("archive")].reset_index(names="position").set_index("archive_id")
    records = []
    for pair in pairs.itertuples():
        left_ids, right_ids = json.loads(pair.left_row_ids_json), json.loads(pair.right_row_ids_json)
        assert len(left_ids) == len(right_ids) == 1
        left, right = archive.loc[left_ids[0]], archive.loc[right_ids[0]]
        assert left.primary and right.primary and left.aggregate_id == right.aggregate_id
        a = arrays["aggregate"][left.position]
        assert np.max(np.abs(arrays["weights"][left.position] + arrays["weights"][right.position] - 2 * a)) < 1e-8
        observed = left.uncheatable - right.uncheatable
        for model in ("zero", "HPR", *[f"F{i}" for i in range(6)]):
            predicted = right[f"gain::{model}"] - left[f"gain::{model}"]
            regret = (
                max(observed, 0) if predicted < -1e-12 else max(-observed, 0) if predicted > 1e-12 else abs(observed) / 2
            )
            records.append(
                {
                    "cell": pair.cell_id,
                    "left": left_ids[0],
                    "right": right_ids[0],
                    "model": model,
                    "observed_difference": observed,
                    "predicted_difference": predicted,
                    "sign_correct": float(predicted * observed > 0),
                    "regret": regret,
                }
            )
    data = pd.DataFrame(records)
    summaries = []
    for (cell, model), group in data.groupby(["cell", "model"]):
        observed, predicted = group.observed_difference.to_numpy(), group.predicted_difference.to_numpy()
        summaries.append(
            {
                "cell": cell,
                "model": model,
                "pairs": len(group),
                "rmse": float(np.sqrt(np.mean((observed - predicted) ** 2))),
                "sign_accuracy": group.sign_correct.mean() if model != "zero" else np.nan,
                "spearman": rank_correlation(observed, predicted),
                "regret": group.regret.mean(),
            }
        )
    return data, pd.DataFrame(summaries)


def endpoint_metrics(group: pd.DataFrame, model: str) -> dict[str, Any]:
    observed, predicted = group.uncheatable.to_numpy(), group[model].to_numpy()
    near = group[group[model] <= predicted.min() + 1e-12]
    selected = near.sort_values(["tied", "policy_id"], ascending=[False, True]).iloc[0]
    order = group.sort_values([model, "policy_id"]).head(3)
    return {
        "policies": len(group),
        "rmse": float(np.sqrt(np.mean((observed - predicted) ** 2))),
        "optimism": float(np.mean(observed - predicted)),
        "spearman": rank_correlation(observed, predicted),
        "regret": float(selected.uncheatable - observed.min()),
        "shortlist3_regret": float(order.uncheatable.min() - observed.min()),
        "selected_policy": selected.policy_id,
        "selected_measured": float(selected.uncheatable),
        "selected_prediction": float(selected[model]),
        "selected_optimism": float(selected.uncheatable - selected[model]),
        "selected_tied": bool(selected.tied),
    }


def endpoint_summaries(
    frame: pd.DataFrame, cells: dict[str, str], pairs: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    models = ["A", "HPR", "HPR+A", *[f"W{i}" for i in range(6)], *[f"F{i}" for i in range(6)]]
    aggregations = {model: "first" for model in models}
    aggregations.update({"uncheatable": "mean", "tied": "first"})
    selections = []
    for cell, aggregate in cells.items():
        all_rows = frame[frame.cohort.eq("archive") & frame.aggregate_id.eq(aggregate)]
        assert all_rows.primary.all()
        identities = set(pairs.loc[pairs.cell.eq(cell), "row_id"]) | set(pairs.loc[pairs.cell.eq(cell), "control_id"])
        for menu, rows in (("all_frontier", all_rows), ("seed_matched", all_rows[all_rows.archive_id.isin(identities)])):
            population = rows.groupby("policy_id", as_index=False).agg(aggregations)
            tied = population[population.tied]
            assert len(tied) == 1
            for model in models:
                metric = endpoint_metrics(population, model)
                tied_prediction = float(tied[model].iloc[0])
                metric.update(
                    {
                        "cell": cell,
                        "menu": menu,
                        "model": model,
                        "measured_gain_vs_pooled_tied": float(tied.uncheatable.iloc[0] - metric["selected_measured"]),
                        "predicted_gain_vs_tied": tied_prediction - metric["selected_prediction"],
                    }
                )
                matched = pairs[
                    pairs.cell.eq(cell) & pairs.policy_id.eq(metric["selected_policy"]) & pairs.model.eq("zero")
                ]
                metric["selected_same_seed_gain"] = (
                    0.0 if metric["selected_tied"] else float(matched.observed_gain.mean()) if len(matched) else np.nan
                )
                selections.append(metric)
    summaries = []
    eligible = frame[frame.primary & frame.uncheatable.notna()]
    populations = [("all_sources", eligible), *list(eligible.groupby("source"))]
    for scope, rows in populations:
        for tied, subset in rows.groupby("tied"):
            population = subset.groupby("policy_id", as_index=False).agg(aggregations)
            for model in models:
                for band, group in (
                    ("all", population),
                    ("predicted_frontier20", population.nsmallest(max(1, int(np.ceil(0.2 * len(population)))), model)),
                ):
                    summaries.append(
                        {
                            "scope": scope,
                            "tied": bool(tied),
                            "band": band,
                            "model": model,
                            **endpoint_metrics(group, model),
                        }
                    )
    return pd.DataFrame(selections), pd.DataFrame(summaries)


def main() -> None:
    identity = {str(path): sha(path) for path in input_paths()}
    manifest_path = OUTPUT / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        assert previous["inputs"] == identity, "Input changed; preserve this execution before rerunning."
        assert all(sha(OUTPUT / name) == digest for name, digest in previous["outputs"].items())
        print("Verified completed frontier transfer; no recomputation needed.")
        return
    OUTPUT.mkdir(exist_ok=True)
    with threadpool_limits(limits=1):
        frame, arrays, audit = predictions()
        pairs, cells = matched_pairs(frame, arrays)
        summaries, intervals = paired_summaries(pairs)
        mirror_rows, mirror_metrics = mirrors(frame, arrays)
        selections, endpoints = endpoint_summaries(frame, cells, pairs)
    for name, table in (
        ("predictions", frame),
        ("matched_pairs", pairs),
        ("gain_metrics", summaries),
        ("conditional_intervals", intervals),
        ("mirror_predictions", mirror_rows),
        ("mirror_metrics", mirror_metrics),
        ("cell_selections", selections),
        ("endpoint_metrics", endpoints),
    ):
        table.to_csv(OUTPUT / f"{name}.csv", index=False)
    audit.update(
        {
            "matched_unique_rows": pairs.row_id.nunique(),
            "matched_control_count": pairs.control_id.nunique(),
            "mirror_pair_count": len(mirror_rows) // 8,
            "cells": cells,
            "bootstrap": (
                "4000 paired source-stratified shared-control blocks per anchor; "
                "conditional on frozen fits and adaptive menus"
            ),
        }
    )
    write_json_atomic(OUTPUT / "validation.json", audit)
    write_json_atomic(
        manifest_path,
        {
            "inputs": identity,
            "outputs": {path.name: sha(path) for path in OUTPUT.iterdir() if path.is_file() and path != manifest_path},
            "protocol_note": "note_01m2sdp65pdye1em7hepskrckq",
            "fits_changed": False,
            "remote_calls": 0,
        },
    )
    print(
        summaries[summaries.source.eq("all")][
            ["cell", "model", "gain_rmse", "matched_regret", "gain_optimism", "prefer_phase_fraction"]
        ].to_string(index=False)
    )
    print(
        selections[selections.menu.eq("seed_matched") & selections.model.str.startswith("F")][
            ["cell", "model", "regret", "predicted_gain_vs_tied", "selected_same_seed_gain"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
