# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["fsspec==2026.1.0", "numpy==2.3.5", "pandas==2.2.2", "scipy==1.17.0"]
# ///
"""Test two prespecified continuation interactions with fully nested base fits."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import benchmark_crossed_mariner_20260912 as base
import numpy as np
import pandas as pd
from fit_two_phase_link_spines_20260907 import write_json_atomic

OUTPUT = base.OUTPUT.parent / "state_extension"
FEATURE_SPEC = base.OUTPUT.parent / "identification/FEATURE_SPEC.md"
PENALTIES = (None, 100.0, 10.0, 1.0, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.0)
CODE_BUCKETS = ("dolma3_stack_edu", "dolmino_stack_edu_fim", "dolmino_synth_code")


def feature_values(arrays: dict[str, np.ndarray], rows: np.ndarray, specification: dict[str, Any]) -> np.ndarray:
    """Evaluate the prespecified interactions using frozen training centers."""
    action = arrays["phase1_weight"][rows]
    prefix_log = np.log1p(arrays["phase0_epochs"][rows])
    code = np.asarray(specification["code_indices"], int)
    repetition = np.sum(
        (action - specification["action_center"]) * (prefix_log - specification["prefix_log_center"]), axis=1
    )
    code_product = (arrays["phase0_weight"][rows][:, code].sum(axis=1) - specification["prefix_code_center"]) * (
        action[:, code].sum(axis=1) - specification["action_code_center"]
    )
    return np.column_stack([repetition, code_product]) / specification["scale"]


def training_features(
    frame: pd.DataFrame, arrays: dict[str, np.ndarray], rows: np.ndarray
) -> tuple[dict[str, Any], np.ndarray]:
    """Derive centers from distinct training prefixes and non-control actions."""
    actions = frame.iloc[rows][~frame.is_tied_control.iloc[rows].to_numpy(bool)]
    unique_actions = actions.drop_duplicates("coordinate_hash").row.to_numpy(int)
    unique_prefixes = actions.drop_duplicates("state_id").row.to_numpy(int)
    code = [list(arrays["bucket_names"]).index(name) for name in CODE_BUCKETS]
    specification = {
        "code_indices": code,
        "action_center": arrays["phase1_weight"][unique_actions].mean(axis=0).tolist(),
        "prefix_log_center": np.log1p(arrays["phase0_epochs"][unique_prefixes]).mean(axis=0).tolist(),
        "prefix_code_center": float(arrays["phase0_weight"][unique_prefixes][:, code].sum(axis=1).mean()),
        "action_code_center": float(arrays["phase1_weight"][unique_actions][:, code].sum(axis=1).mean()),
        "scale": [1.0, 1.0],
        "prefix_rows": unique_prefixes.tolist(),
        "action_rows": unique_actions.tolist(),
    }
    raw = feature_values(arrays, actions.row.to_numpy(int), specification)
    scale = np.sqrt(np.mean(raw**2, axis=0))
    assert np.all(scale > 1e-12)
    specification["scale"] = scale.tolist()
    features = feature_values(arrays, rows, specification)
    assert np.linalg.matrix_rank(features) == 2
    return specification, features


def grouped_context(rows: np.ndarray, valid: np.ndarray, frame: pd.DataFrame, legacy: pd.DataFrame) -> dict[str, Any]:
    """Build deeper prefix-by-action CV using only a parent partition's training rows."""
    calibration = frame.is_tied_control.iloc[rows].to_numpy(bool)
    groups = base.action_groups(rows[~calibration], 3, frame, legacy)
    states = [state for state in base.STATES if state in set(frame.state_id.iloc[rows])]
    prefix_groups = {state: index % 3 for index, state in enumerate(states)}
    prefixes = np.asarray([prefix_groups[state] for state in frame.state_id.iloc[rows]])
    actions = np.asarray([groups.get(coordinate, -1) for coordinate in frame.coordinate_hash.iloc[rows]])
    folds = []
    for prefix in sorted(set(prefixes)):
        for action in range(3):
            training = np.flatnonzero((prefixes != prefix) & ((actions != action) | calibration))
            validation = np.flatnonzero((prefixes == prefix) & (actions == action) & ~calibration)
            assert len(training) and len(validation) and calibration[training].any()
            assert not set(frame.state_id.iloc[rows[training]]).intersection(frame.state_id.iloc[rows[validation]])
            assert not set(frame.coordinate_hash.iloc[rows[training]]).intersection(
                frame.coordinate_hash.iloc[rows[validation]]
            )
            folds.append({"train": training.tolist(), "validation": validation.tolist()})
    return {
        "train": rows.tolist(),
        "test": valid.tolist(),
        "predict": np.unique(np.r_[rows, valid]).tolist(),
        "calibration": calibration.tolist(),
        "inner_folds": folds,
        "prefix_groups": prefix_groups,
        "action_groups": groups,
    }


def prepare(output: Path) -> dict[str, Any]:
    """Freeze identities and run outcome-free feature checks before any added fit."""
    output.mkdir(parents=True, exist_ok=True)
    original = json.loads((base.OUTPUT / "prepared.json").read_text())
    assert all(base.digest(Path(path)) == sha for path, sha in original["hashes"].items())
    hashes = dict(original["hashes"])
    for path in (Path(__file__), output / "PROTOCOL.md", FEATURE_SPEC):
        hashes[str(path.resolve())] = base.digest(path)
    for state in base.STATES:
        path = base.OUTPUT / "fits" / state / "MTP-002.json"
        hashes[str(path.resolve())] = base.digest(path)
    frame, arrays = base.load_data()
    legacy = pd.read_csv(base.INPUT / "data/audit_frame.csv")
    contexts = {}
    checks = []
    for state in base.STATES:
        outer = original["contexts"][state]
        outer_rows = np.asarray(outer["train"], int)
        partition_sets = [(state, outer_rows, np.asarray(outer["test"], int))]
        for index, inner in enumerate(outer["inner_folds"]):
            name = f"{state}__inner{index:02d}"
            training = outer_rows[np.asarray(inner["train"], int)]
            validation = outer_rows[np.asarray(inner["validation"], int)]
            contexts[name] = grouped_context(training, validation, frame, legacy)
            partition_sets.append((name, training, validation))
        for name, training, validation in partition_sets:
            specification, matrix = training_features(frame, arrays, training)
            held = feature_values(arrays, validation, specification)
            truth = np.array([0.012, -0.007])
            recovered = np.linalg.lstsq(matrix, matrix @ truth, rcond=None)[0]
            error = float(np.max(np.abs(held @ (recovered - truth))))
            assert error < 1e-12
            checks.append(
                {
                    "partition": name,
                    "rank": int(np.linalg.matrix_rank(matrix)),
                    "correlation": float(np.corrcoef(matrix.T)[0, 1]),
                    "scale": specification["scale"],
                    "held_max_abs_standardized": np.max(np.abs(held), axis=0).tolist(),
                    "synthetic_prediction_max_error": error,
                    "training_rows": training.tolist(),
                    "validation_rows": validation.tolist(),
                    "features": specification,
                }
            )
    nested = output / "base_subfits"
    (nested / "sources").mkdir(parents=True, exist_ok=True)
    for name in ("mixture_selection.py", base.DSP_SOURCE.name):
        source = base.OUTPUT / "sources" / name
        destination = nested / "sources" / name
        if destination.exists():
            assert base.digest(destination) == base.digest(source)
        else:
            shutil.copy2(source, destination)
    prepared = {"hashes": hashes, "contexts": contexts, "outer_contexts": original["contexts"], "checks": checks}
    write_json_atomic(nested / "prepared.json", prepared)
    write_json_atomic(output / "prepared.json", prepared)
    write_json_atomic(output / "structural_checks.json", {"input_hashes": hashes, "checks": checks})
    return prepared


def ridge_coefficients(matrix: np.ndarray, residual: np.ndarray, penalty: float | None) -> np.ndarray:
    if penalty is None:
        return np.zeros(2)
    if penalty == 0:
        return np.linalg.lstsq(matrix, residual, rcond=None)[0]
    return np.linalg.solve(matrix.T @ matrix / len(matrix) + penalty * np.eye(2), matrix.T @ residual / len(matrix))


def action_pair_cells(frame: pd.DataFrame, rows: np.ndarray, error: np.ndarray) -> list[float]:
    values = []
    for state in frame.state_id.iloc[rows].unique():
        residual = error[frame.state_id.iloc[rows].eq(state).to_numpy()]
        assert len(residual) >= 2
        pair = np.triu_indices(len(residual), 1)
        values.append(float(np.mean((residual[:, None] - residual[None, :])[pair] ** 2)))
    return values


def fit_outer(output: Path, state: str, prepared: dict[str, Any]) -> None:
    """Select two-coefficient shrinkage from strictly held inner predictions."""
    destination = output / "fits" / state / "MTP-007.json"
    if destination.exists():
        cached = json.loads(destination.read_text())
        assert cached["input_hashes"] == prepared["hashes"]
        print(json.dumps({"state": state, "status": "cached"}), flush=True)
        return
    frame, arrays = base.load_data()
    source_path = base.OUTPUT / "sources/mixture_selection.py"
    candidate_scores: list[list[float]] = [[] for _ in PENALTIES]
    inner_records = []
    for index in range(9):
        name = f"{state}__inner{index:02d}"
        context = prepared["contexts"][name]
        training, valid = np.asarray(context["train"], int), np.asarray(context["test"], int)
        nested_path = output / "base_subfits/fits" / name / "MTP-002.json"
        nested = json.loads(nested_path.read_text())
        assert nested["input_hashes"] == prepared["hashes"]
        base_training = base.predict_fit(
            nested["fit"], arrays["phase0_epochs"][training], arrays["phase1_epochs"][training], source_path
        )
        base_valid = base.predict_fit(
            nested["fit"], arrays["phase0_epochs"][valid], arrays["phase1_epochs"][valid], source_path
        )
        specification, matrix = training_features(frame, arrays, training)
        valid_matrix = feature_values(arrays, valid, specification)
        coefficients = []
        for position, penalty in enumerate(PENALTIES):
            beta = ridge_coefficients(matrix, arrays["target"][training] - base_training, penalty)
            error = base_valid + valid_matrix @ beta - arrays["target"][valid]
            candidate_scores[position].extend(action_pair_cells(frame, valid, error))
            coefficients.append(beta.tolist())
        inner_records.append(
            {
                "partition": name,
                "features": specification,
                "coefficients_by_penalty": coefficients,
                "base_path": str(nested_path),
            }
        )
    scores = [float(np.mean(values)) for values in candidate_scores]
    selected = int(np.argmin(scores))
    outer = prepared["outer_contexts"][state]
    training = np.asarray(outer["train"], int)
    prediction_rows = np.asarray(outer["predict"], int)
    saved_base = json.loads((base.OUTPUT / "fits" / state / "MTP-002.json").read_text())
    base_training = base.predict_fit(
        saved_base["fit"], arrays["phase0_epochs"][training], arrays["phase1_epochs"][training], source_path
    )
    base_prediction = np.asarray(saved_base["predicted"], float)
    assert np.array_equal(prediction_rows, saved_base["prediction_rows"])
    specification, matrix = training_features(frame, arrays, training)
    beta = ridge_coefficients(matrix, arrays["target"][training] - base_training, PENALTIES[selected])
    prediction_features = feature_values(arrays, prediction_rows, specification)
    contributions = prediction_features * beta
    prediction = base_prediction + contributions.sum(axis=1)
    assert np.isfinite(prediction).all()
    payload = {
        "model": "MTP-007",
        "fold": state,
        "input_hashes": prepared["hashes"],
        "training_rows": training.tolist(),
        "test_rows": outer["test"],
        "prediction_rows": prediction_rows.tolist(),
        "features": specification,
        "penalties": PENALTIES,
        "inner_action_pair_mse": scores,
        "selected_index": selected,
        "selected_penalty": PENALTIES[selected],
        "coefficients_standardized_bpb": beta.tolist(),
        "coefficients_raw_feature_bpb": (beta / specification["scale"]).tolist(),
        "base_prediction": base_prediction.tolist(),
        "feature_contributions_bpb": contributions.tolist(),
        "predicted": prediction.tolist(),
        "inner_fits": inner_records,
        "base_floor": saved_base["fit"]["floor"],
        "primary_prediction_below_base_floor": int(
            np.sum(prediction[frame.state_id.iloc[prediction_rows].eq(state)] < saved_base["fit"]["floor"])
        ),
    }
    write_json_atomic(destination, payload)
    print(json.dumps({"state": state, "penalty": PENALTIES[selected], "beta": beta.tolist()}), flush=True)


def evaluate(output: Path) -> None:
    frame, _arrays = base.load_data()
    tables, metrics, effects = [], [], []
    for state in base.STATES:
        saved = json.loads((output / "fits" / state / "MTP-007.json").read_text())
        rows = np.asarray(saved["prediction_rows"], int)
        table = frame.iloc[rows][["row", "row_id", "state_id", "action_id", "coordinate_hash", "target"]].copy()
        table["is_tied"] = frame.is_tied_control.iloc[rows].to_numpy(bool)
        table["primary"] = table.state_id.eq(state)
        table["fold"] = state
        table["base_prediction"] = saved["base_prediction"]
        contribution = np.asarray(saved["feature_contributions_bpb"])
        table["repetition_effect_bpb"], table["code_effect_bpb"] = contribution[:, 0], contribution[:, 1]
        for model, predictions in (("MTP-002", saved["base_prediction"]), ("MTP-007", saved["predicted"])):
            current = table.copy()
            current["model"], current["prediction"] = model, predictions
            tables.append(current)
            primary = current[current.primary]
            tied = primary[primary.is_tied].iloc[0]
            for with_tied in (False, True):
                selected = primary if with_tied else primary[~primary.is_tied]
                result = base.state_metrics(selected)
                pick = selected.loc[selected.row.eq(result["selected_row"])].iloc[0]
                result.update(
                    {
                        "model": model,
                        "state_id": state,
                        "with_tied": with_tied,
                        "observed_gain_vs_tied": float(tied.target - pick.target),
                        "predicted_gain_vs_tied": float(tied.prediction - pick.prediction),
                    }
                )
                metrics.append(result)
        primary = table[table.primary]
        effects.append(
            {
                "state_id": state,
                "selected_penalty": "off" if saved["selected_penalty"] is None else saved["selected_penalty"],
                "repetition_beta_bpb": saved["coefficients_standardized_bpb"][0],
                "code_beta_bpb": saved["coefficients_standardized_bpb"][1],
                "repetition_min_bpb": float(primary.repetition_effect_bpb.min()),
                "repetition_max_bpb": float(primary.repetition_effect_bpb.max()),
                "code_min_bpb": float(primary.code_effect_bpb.min()),
                "code_max_bpb": float(primary.code_effect_bpb.max()),
                "below_base_floor": saved["primary_prediction_below_base_floor"],
            }
        )
    pd.concat(tables, ignore_index=True).to_csv(output / "predictions.csv", index=False)
    metric_frame = pd.DataFrame(metrics)
    metric_frame.to_csv(output / "metrics.csv", index=False)
    metric_frame.groupby(["model", "with_tied"]).mean(numeric_only=True).to_csv(output / "summary.csv")
    pd.DataFrame(effects).to_csv(output / "held_feature_effects.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "fit", "evaluate", "all"), default="all")
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    for variable in base.THREAD_VARIABLES:
        assert os.environ.get(variable) == "1", f"set {variable}=1 before running"
    if args.stage in {"prepare", "all"}:
        prepared = prepare(OUTPUT)
    else:
        prepared = json.loads((OUTPUT / "prepared.json").read_text())
    assert all(base.digest(Path(path)) == sha for path, sha in prepared["hashes"].items())
    if args.stage in {"fit", "all"}:
        jobs = [base.FitJob(name, "MTP-002", str(OUTPUT / "base_subfits")) for name in prepared["contexts"]]
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(base.fit_job, job) for job in jobs]
            for future in as_completed(futures):
                print(json.dumps(future.result()), flush=True)
        for state in base.STATES:
            fit_outer(OUTPUT, state, prepared)
    if args.stage in {"evaluate", "all"}:
        evaluate(OUTPUT)


if __name__ == "__main__":
    main()
