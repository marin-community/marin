# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "scipy", "pandas", "joblib"]
# ///
"""Replay the archived raw-harm screen without fitting any model.

Run with ``uv run --offline --no-sync python -m experiments.domain_phase_mix.
exploratory.two_phase_many.audit_mariner_raw_screen_20260917``. The default
requires the complete screen and its report. ``--allow-incomplete`` audits all
currently completed shards and marks its result incomplete when any are missing.

Only input/split definitions and configuration fingerprints are shared with the
benchmark. Predictions, floor calibration, RMS penalties, pooled inner-CV scores,
ridge selection, and convergence-count reconciliation are recomputed here from
the serialized coefficients, without calling a probe or its scoring routines.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import hashlib
import importlib.metadata
import io
import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    benchmark_single_phase_observatory_20260902 as bench,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    screen_mariner_harm_forms_20260917 as fixed,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    single_phase_observatory_models_20260902 as base,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    single_phase_observatory_registry_20260902 as registry,
)

HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "reference_outputs/mariner_harm_screen_20260917/raw_loss"
PREDICTION_KEYS = ["model", "panel", "target", "component_index", "repeat", "fold", "row_index"]
MATERIAL_RELATIVE_GAP = 0.01
MATERIAL_ABSOLUTE_GAP = 1e-10
RELATIVE_GAP_DENOMINATOR_FLOOR = 1e-12


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_checksum(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def close(actual: Any, expected: Any, label: str) -> float:
    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-10, err_msg=label)
    actual_array, expected_array = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    finite = np.isfinite(actual_array) & np.isfinite(expected_array)
    return float(np.max(np.abs(actual_array[finite] - expected_array[finite]), initial=0.0))


def read_shard(path: Path) -> tuple[dict[str, np.ndarray], str]:
    data = path.read_bytes()
    with np.load(io.BytesIO(data), allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}, hashlib.sha256(data).hexdigest()


def audit_sources(output: Path, manifest: dict[str, Any]) -> None:
    for filename, expected in manifest["sources"].items():
        path = Path(filename)
        require(checksum(path) == expected, f"Current source changed: {path}")
        require(checksum(output / "source_snapshot" / path.name) == expected, f"Snapshot changed: {path}")
    for name, expected in manifest["dependencies"].items():
        require(importlib.metadata.version(name) == expected, f"Dependency changed: {name}")
    fixed_manifest = Path(manifest["fixed_output"]) / "screen_manifest.json"
    require(checksum(fixed_manifest) == manifest["fixed_manifest_sha256"], "Geometry manifest changed")
    require(
        checksum(output / "source_snapshot/fixed_screen_manifest.json") == checksum(fixed_manifest),
        "Geometry manifest snapshot changed",
    )
    require(set(manifest["panel_inputs"]) == set(bench.THIRTY_NINE_BUCKET_PANELS), "Unexpected screen panels")
    panels = tuple(bench.load_panel(name) for name in bench.THIRTY_NINE_BUCKET_PANELS)
    for panel in panels:
        require(panel.input_hashes == manifest["panel_inputs"][panel.name], f"Panel inputs changed: {panel.name}")
    require(bench.manifest_hash(bench.split_manifest(panels, 1)) == manifest["split_hash"], "Split identity changed")
    require(
        manifest["floor_multiplier"] == 1.5 and manifest["starts_per_fit"] == 2,
        "Unexpected registered floor or start count",
    )
    require(manifest["prediction_cap"] is None, "The replay requires the registered uncapped model")
    fixed.register_candidates()


def audit_geometry(
    task: bench.FitTask,
    panel: bench.BenchPanel,
    split: bench.Split,
    shard: dict[str, np.ndarray],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    geometry = json.loads(str(shard["geometry_json"].item()))
    source_task = dataclasses.replace(task, model_id=manifest["geometry_model"])
    source_path = bench.shard_path(Path(manifest["fixed_output"]), source_task)
    require(str(source_path.resolve()) == geometry["path"], "Geometry path does not match the outer task")
    source, source_checksum = read_shard(source_path)
    require(source_checksum == geometry["checksum"], "Geometry shard bytes changed")
    require(str(source["status"].item()) == "ok", "Geometry fit failed")
    source_protocol = str(source["protocol_hash"].item())
    require(source_protocol == geometry["source_protocol_hash"], "Geometry protocol changed")
    accepted = bench.task_protocol_hashes(
        source_task,
        registry.ENTRY_BY_ID[source_task.model_id],
        manifest["split_hash"],
        panel,
        Path(manifest["fixed_output"]),
    )
    require(source_protocol in accepted, "Geometry does not belong to the registered outer fit")
    require(np.array_equal(source["test"], split.test), "Geometry uses different test rows")
    require(str(source["component"].item()) == task.component, "Geometry component changed")
    response = panel.group(task.target).outcomes[:, task.component_index]
    require(np.array_equal(source["observed"], response[split.test]), "Geometry observations changed")
    shape = json.loads(str(source["shape_json"].item()))
    require(shape == geometry["shape"] == json.loads(str(shard["shape_json"].item())), "Geometry shape changed")
    expected = json_checksum(
        {
            "manifest": json_checksum(manifest),
            "task": dataclasses.asdict(task),
            "geometry": geometry,
        }
    )
    require(str(shard["protocol_hash"].item()) == expected, "Raw task protocol differs from its source/geometry")
    return geometry


def independent_basis(exposures: np.ndarray, shape: dict[str, float], harm: str) -> tuple[np.ndarray, np.ndarray]:
    require(set(shape) == {"rate", "power", "threshold"}, "Expected shared three-parameter geometry")
    require(np.isfinite(exposures).all() and np.all(exposures >= 0.0), "Invalid exposures")
    rate, power, threshold = shape["rate"], shape["power"], shape["threshold"]
    benefit = 1.0 - np.exp(-np.power(rate * exposures, power))
    softplus = np.logaddexp(0.0, np.log1p(exposures) - threshold)
    origin = np.logaddexp(0.0, -threshold)
    if harm == "squared_log":
        damage = softplus**2 - origin**2
    elif harm == "unsquared_log":
        damage = softplus - origin
    elif harm == "bounded_rational_c1":
        squared = softplus**2 - origin**2
        damage = squared / (1.0 + squared)
    elif harm == "convex_linear":
        critical, scale = math.expm1(threshold), math.exp(threshold)
        damage = np.logaddexp(0.0, (exposures - critical) / scale) - np.logaddexp(0.0, -critical / scale)
    else:
        raise ValueError(f"Unsupported registered harm: {harm}")
    return benefit, damage


def independent_prediction(head: dict[str, Any], benefit: np.ndarray, harm: np.ndarray, rows: np.ndarray) -> np.ndarray:
    alpha, beta = np.asarray(head["alpha"]), np.asarray(head["beta"])
    require(alpha.shape == beta.shape == (benefit.shape[1],), "Wrong amplitude dimensions")
    require(
        np.isfinite(alpha).all() and np.isfinite(beta).all() and np.all(alpha >= 0) and np.all(beta >= 0),
        "Amplitudes must be finite and nonnegative",
    )
    signal = head["intercept"] - benefit[rows] @ alpha
    damage = harm[rows] @ beta
    with np.errstate(over="raise", invalid="raise", under="ignore"):
        if head["placement"] == "raw_loss":
            return head["floor"] + np.exp(signal) + damage
        require(head["placement"] == "log_deficit", "Unknown saved harm placement")
        return head["floor"] + np.exp(signal + damage)


def audit_runs(runs: list[dict[str, Any]]) -> None:
    require(len(runs) == 2 and [run["start"] for run in runs] == [0, 1], "Missing deterministic start records")
    for run in runs:
        require(run["nfev"] >= 0 and run["njev"] >= 0 and run["elapsed_seconds"] >= 0, "Invalid solver accounting")
        if run["converged"]:
            require(
                run["status"] > 0 and math.isfinite(run["objective"]) and run["objective"] >= 0,
                "Converged start has an invalid termination or objective",
            )


def audit_head(
    head: dict[str, Any],
    benefit: np.ndarray,
    harm: np.ndarray,
    response: np.ndarray,
    train: np.ndarray,
    ridge: float,
    spec: base.HeadSpec,
    placement: str,
) -> dict[str, float]:
    require(head["placement"] == placement, "Fit was saved under the wrong placement")
    anchor = float(np.median(response[train])) if math.isnan(spec.floor_anchor) else spec.floor_anchor
    gap = anchor - float(np.min(response[train]))
    if gap <= 0.0:
        gap = max(float(np.std(response[train])), 1e-3 * abs(anchor), 1e-6)
    floor = anchor - max(1.5 * gap, spec.noise_margin_sds * spec.noise_sd)
    floor_error = close(head["floor"], floor, "Training-only floor calibration")
    benefit_scale = np.sqrt(np.mean(benefit[train] ** 2, axis=0))
    harm_scale = np.sqrt(np.mean(harm[train] ** 2, axis=0))
    benefit_scale = np.where(benefit_scale > 0.0, benefit_scale, 1.0)
    harm_scale = np.where(harm_scale > 0.0, harm_scale, 1.0)
    close(head["benefit_scale"], benefit_scale, "Training-only benefit RMS")
    close(head["harm_scale"], harm_scale, "Training-only harm RMS")
    prediction = independent_prediction(head, benefit, harm, train)
    penalty = float(
        np.sum((np.asarray(head["alpha"]) * benefit_scale) ** 2) + np.sum((np.asarray(head["beta"]) * harm_scale) ** 2)
    )
    objective = float(np.mean((prediction - response[train]) ** 2)) + ridge * penalty
    runs = head["runs"]
    audit_runs(runs)
    converged = [run for run in runs if run["converged"]]
    require(bool(converged), "Saved fit has no converged start")
    selected = min(converged, key=lambda run: (run["objective"], run["start"]))
    require(head["selected_start"] == selected["start"], "Saved head did not select the best converged start")
    objective_error = close(objective, selected["objective"], "Selected-start raw objective and RMS penalty")
    return {"floor_error": floor_error, "objective_error": objective_error}


def objective_disagreement(runs: list[dict[str, Any]]) -> dict[str, Any]:
    converged = [float(run["objective"]) for run in runs if run["converged"]]
    if len(converged) < 2:
        absolute, relative = float("nan"), float("nan")
    else:
        absolute = max(converged) - min(converged)
        relative = absolute / max(abs(min(converged)), RELATIVE_GAP_DENOMINATOR_FLOOR)
    best = min(converged) if converged else float("inf")
    return {
        "converged_starts": len(converged),
        "start_objective_absolute_gap": absolute,
        "start_objective_relative_gap": relative,
        "material_start_disagreement": bool(absolute > MATERIAL_ABSOLUTE_GAP and relative > MATERIAL_RELATIVE_GAP),
        "nonconverged_lower_objective": any(
            not run["converged"] and math.isfinite(run["objective"]) and run["objective"] < best - MATERIAL_ABSOLUTE_GAP
            for run in runs
        ),
    }


def start_records(runs: list[dict[str, Any]], stage: str, ridge: float, fold: int, selected_ridge: float) -> list[dict]:
    converged = [run for run in runs if run["converged"]]
    best = min((run["objective"] for run in converged), default=float("nan"))
    records = []
    for run in runs:
        improvement = (
            max(0.0, best - run["objective"]) if math.isfinite(best) and math.isfinite(run["objective"]) else 0.0
        )
        relative = improvement / max(abs(best), RELATIVE_GAP_DENOMINATOR_FLOOR) if math.isfinite(best) else 0.0
        records.append(
            {
                "stage": stage,
                "fit_ridge": ridge,
                "inner_fold": fold,
                "ridge_selected": ridge == selected_ridge,
                "both_starts_failed": not converged,
                **run,
                "failed_objective_improvement": improvement if not run["converged"] else 0.0,
                "failed_relative_improvement": relative if not run["converged"] else 0.0,
                "material_failed_improvement": bool(
                    not run["converged"] and improvement > MATERIAL_ABSOLUTE_GAP and relative > MATERIAL_RELATIVE_GAP
                ),
            }
        )
    return records


def audit_success(
    task: bench.FitTask,
    shard: dict[str, np.ndarray],
    panel: bench.BenchPanel,
    split: bench.Split,
    manifest: dict[str, Any],
    candidate: dict[str, str],
) -> dict[str, Any]:
    response = panel.group(task.target).outcomes[:, task.component_index]
    shape = json.loads(str(shard["shape_json"].item()))
    head = json.loads(str(shard["head_json"].item()))
    diagnostics = json.loads(str(shard["diagnostics_json"].item()))
    benefit, harm = independent_basis(panel.features.exposures, shape, candidate["harm"])
    features = dataclasses.replace(panel.features, component=task.component)
    canonical = registry.ENTRY_BY_ID[registry.FROZEN_ID].build(features)
    if not isinstance(canonical, base.FittedFloorModel):
        raise ValueError("Expected the registered floor calibration")
    spec = canonical.base.head
    ridge_grid = manifest["ridge_grid"]
    ridge = float(shard["ridge"])
    final_errors = audit_head(head, benefit, harm, response, split.train, ridge, spec, candidate["placement"])
    outer = independent_prediction(head, benefit, harm, split.test)
    train_prediction = independent_prediction(head, benefit, harm, split.train)
    outer_error = close(shard["prediction"], outer, "Outer predictions from saved coefficients")
    train_error = close(shard["train_prediction"], train_prediction, "Training predictions from saved coefficients")
    trials = head["search_trials"]
    require(len(trials) == len(ridge_grid) * len(split.inner), "Missing inner candidate trials")
    all_runs = list(head["runs"])
    solver_records = start_records(head["runs"], "final", ridge, -1, ridge)
    scores, trial_lookup = [], {}
    max_objective_error, max_floor_error = final_errors["objective_error"], final_errors["floor_error"]
    for ridge_index, trial_ridge in enumerate(ridge_grid):
        error_sum, count, invalid = 0.0, 0, False
        for fold, (inner_train, validation) in enumerate(split.inner):
            trial = trials[ridge_index * len(split.inner) + fold]
            require(trial["ridge"] == trial_ridge and trial["fold"] == fold, "Inner trial order or identity changed")
            audit_runs(trial["runs"])
            all_runs.extend(trial["runs"])
            solver_records.extend(start_records(trial["runs"], "inner", trial_ridge, fold, ridge))
            trial_lookup[trial_ridge, fold] = trial
            if trial["head"] is None:
                require(not any(run["converged"] for run in trial["runs"]), "Discarded a converged inner fit")
                invalid = True
                continue
            trial_head = trial["head"]
            require(json_checksum(trial_head["runs"]) == json_checksum(trial["runs"]), "Inner start records disagree")
            errors = audit_head(
                trial_head, benefit, harm, response, inner_train, trial_ridge, spec, candidate["placement"]
            )
            max_objective_error = max(max_objective_error, errors["objective_error"])
            max_floor_error = max(max_floor_error, errors["floor_error"])
            try:
                prediction = independent_prediction(trial_head, benefit, harm, validation)
            except FloatingPointError:
                require(bool(trial["failure"]), "Unrecorded inner prediction failure")
                invalid = True
                continue
            require(not trial["failure"], "Reported inner prediction failure does not reproduce")
            error_sum += float(np.sum((prediction - response[validation]) ** 2))
            count += len(validation)
        scores.append(float("inf") if invalid else math.sqrt(error_sum / count))
    cv_error = close(shard["cv_table"], np.asarray([scores]), "Every candidate's own-form pooled inner CV")
    saved_scores = np.asarray(shard["cv_table"][0], dtype=float)
    finite_scores = np.isfinite(saved_scores)
    cv_relative_error = float(
        np.max(
            np.abs(saved_scores[finite_scores] - np.asarray(scores)[finite_scores])
            / np.maximum(np.abs(saved_scores[finite_scores]), 1e-12),
            initial=0.0,
        )
    )
    selected = int(np.argmin(scores))
    require(ridge == ridge_grid[selected], "Selected ridge is not the first minimum of its own CV scores")
    require(len(head["inner_fits"]) == len(split.inner), "Missing selected-fold predictions")
    for fold, (inner_train, validation) in zip(head["inner_fits"], split.inner, strict=True):
        require(
            np.array_equal(fold["train"], inner_train) and np.array_equal(fold["validation"], validation),
            "Selected inner fold row identities changed",
        )
        fold_index = next(index for index, (_, rows) in enumerate(split.inner) if np.array_equal(rows, validation))
        require(
            json_checksum(fold["head"]) == json_checksum(trial_lookup[ridge, fold_index]["head"]),
            "Selected inner head differs from the scored candidate",
        )
        close(
            fold["prediction"],
            independent_prediction(fold["head"], benefit, harm, validation),
            "Serialized selected inner predictions",
        )
    close(diagnostics["inner_rmse"], scores[selected], "Selected CV diagnostic")
    close(diagnostics["inner_reconstructed_rmse"], scores[selected], "CV reconstruction diagnostic")
    expected_counts = {
        "solver_start_count": len(all_runs),
        "solver_converged_count": sum(run["converged"] for run in all_runs),
        "solver_failed_count": sum(not run["converged"] for run in all_runs),
        "solver_nfev": sum(run["nfev"] for run in all_runs),
        "solver_njev": sum(run["njev"] for run in all_runs),
        "intercept_bound_hits": sum(run["intercept_bound_hit"] for run in all_runs),
        "exponential_underflow_rows": sum(run["exponential_underflow_rows"] for run in all_runs),
        "invalid_ridge_candidates": sum(not math.isfinite(score) for score in scores),
    }
    for key, expected in expected_counts.items():
        require(diagnostics[key] == expected, f"Solver accounting mismatch: {key}")
    return {
        "outer_prediction_error": outer_error,
        "train_prediction_error": train_error,
        "cv_score_error": cv_error,
        "cv_score_relative_error": cv_relative_error,
        "selected_cv_score_error": abs(diagnostics["inner_rmse"] - scores[selected]),
        "floor_error": max_floor_error,
        "objective_error": max_objective_error,
        "inner_rmse": scores[selected],
        "ridge": ridge,
        **expected_counts,
        **objective_disagreement(head["runs"]),
        "solver_records": solver_records,
    }


def audit_report(output: Path, tier: str, tasks: list[bench.FitTask], predictions: pd.DataFrame) -> None:
    report = output / f"report_{tier}"
    verification = json.loads((report / "verification.json").read_text())
    require(
        verification["fits"] == len(tasks) and verification["predictions"] == len(predictions),
        "Report coverage counts disagree with shards",
    )
    require(verification["tasks"] == [dataclasses.asdict(task) for task in tasks], "Report task list changed")
    saved = pd.read_csv(report / "component_predictions.csv").set_index(PREDICTION_KEYS).sort_index()
    replayed = predictions.set_index(PREDICTION_KEYS).sort_index()
    require(saved.index.equals(replayed.index) and saved.index.is_unique, "Reported prediction identities disagree")
    close(
        saved[["observed", "prediction"]].to_numpy(),
        replayed[["observed", "prediction"]].to_numpy(),
        "Report prediction values",
    )


def audit(output: Path, tier: str, allow_incomplete: bool, report_tasklist: Path | None = None) -> dict[str, Any]:
    manifest_path = output / "raw_screen_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    audit_sources(output, manifest)
    candidates = {candidate["model_id"]: candidate for candidate in manifest["candidates"]}
    if report_tasklist is None:
        plan = dataclasses.replace(bench.tier_plan(tier, panels=tuple(manifest["panel_inputs"])), curves=())
        tasks = bench.plan_tasks(plan, tuple(candidates))
        tasks.sort(key=lambda task: (task.fold, task.component_index, task.panel, task.target, task.model_id))
    else:
        require(report_tasklist.resolve().parent.parent == output.resolve(), "Tasklist must belong to this output")
        require(report_tasklist.name == "verification.json", "Expected a registered report verification tasklist")
        tier = report_tasklist.parent.name.removeprefix("report_")
        tasks = [bench.FitTask(**task) for task in json.loads(report_tasklist.read_text())["tasks"]]
        require(len({json_checksum(dataclasses.asdict(task)) for task in tasks}) == len(tasks), "Duplicate report tasks")
    records, prediction_rows, solver_rows = [], [], []
    for task in tasks:
        path = bench.shard_path(output, task)
        metadata = dataclasses.asdict(task)
        if not path.exists():
            records.append(metadata | {"status": "missing"})
            continue
        shard, shard_checksum = read_shard(path)
        panel = bench.load_panel(task.panel)
        split = next(
            s for s in bench.panel_splits(panel, task.repeat + 1) if s.repeat == task.repeat and s.fold == task.fold
        )
        geometry = audit_geometry(task, panel, split, shard, manifest)
        response = panel.group(task.target).outcomes[:, task.component_index]
        require(
            str(shard["model_id"].item()) == task.model_id and str(shard["component"].item()) == task.component,
            "Raw shard model or component identity changed",
        )
        require(int(shard["component_index"]) == task.component_index, "Raw shard component index changed")
        require(
            np.array_equal(shard["train"], split.train) and np.array_equal(shard["test"], split.test),
            "Raw shard uses different train/test rows",
        )
        require(np.array_equal(shard["observed"], response[split.test]), "Raw observations disagree with panel")
        saved_spec = json.loads(str(shard["spec_json"].item()))
        require(
            saved_spec == {key: value for key, value in candidates[task.model_id].items() if key != "model_id"},
            "Saved probe differs from the registered candidate",
        )
        status = str(shard["status"].item())
        record = metadata | {
            "status": status,
            "shard_sha256": shard_checksum,
            "geometry_sha256": geometry["checksum"],
            "harm": candidates[task.model_id]["harm"],
            "placement": candidates[task.model_id]["placement"],
        }
        if status != "ok":
            record["failure_diagnostics"] = str(shard["diagnostics_json"].item())
            records.append(record)
            continue
        checked = audit_success(task, shard, panel, split, manifest, candidates[task.model_id])
        solver_rows.extend(record | run for run in checked.pop("solver_records"))
        record.update(checked)
        records.append(record)
        for index, row in enumerate(split.test):
            prediction_rows.append(
                {
                    "model": task.model_id,
                    "panel": task.panel,
                    "target": task.target,
                    "component_index": task.component_index,
                    "repeat": task.repeat,
                    "fold": task.fold,
                    "row_index": int(row),
                    "observed": float(response[row]),
                    "prediction": float(shard["prediction"][index]),
                }
            )
    frame = pd.DataFrame(records)
    successful = frame[frame.status.eq("ok")]
    complete = len(successful) == len(tasks)
    predictions = pd.DataFrame(prediction_rows)
    if len(predictions):
        require(not predictions.duplicated(PREDICTION_KEYS).any(), "Duplicate OOF identities")
        require(np.isfinite(predictions[["observed", "prediction"]]).all().all(), "Nonfinite OOF values")
    if complete:
        audit_report(output, tier, tasks, predictions)
    relative = successful.get("start_objective_relative_gap", pd.Series(dtype=float)).dropna()
    summary = {
        "audited_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "auditor_sha256": checksum(Path(__file__)),
        "manifest_sha256": checksum(manifest_path),
        "tier": tier,
        "expected_fits": len(tasks),
        "audited_fits": len(successful),
        "complete": complete,
        "statuses": frame.status.value_counts().to_dict(),
        "out_of_fold_predictions": len(predictions),
        "all_candidate_scores_replayed": True,
        "refits": 0,
        "material_disagreement_rule": {
            "relative_to_best_gt": MATERIAL_RELATIVE_GAP,
            "absolute_gt": MATERIAL_ABSOLUTE_GAP,
            "denominator_floor": RELATIVE_GAP_DENOMINATOR_FLOOR,
        },
        "final_start_disagreement": {
            "comparable_fits": len(relative),
            "relative_gap_median": float(relative.median()),
            "relative_gap_max": float(relative.max()),
            "material_count": int(successful.get("material_start_disagreement", pd.Series(dtype=bool)).sum()),
            "one_converged_start_count": int(successful.get("converged_starts", pd.Series(dtype=int)).eq(1).sum()),
            "nonconverged_lower_objective_count": int(
                successful.get("nonconverged_lower_objective", pd.Series(dtype=bool)).sum()
            ),
        },
        "max_replay_errors": {
            key: float(successful[key].max())
            for key in (
                "outer_prediction_error",
                "train_prediction_error",
                "cv_score_error",
                "cv_score_relative_error",
                "selected_cv_score_error",
                "floor_error",
                "objective_error",
            )
            if key in successful
        },
        "solver_totals": {
            key: int(successful[key].sum())
            for key in (
                "solver_start_count",
                "solver_converged_count",
                "solver_failed_count",
                "solver_nfev",
                "solver_njev",
                "intercept_bound_hits",
                "exponential_underflow_rows",
                "invalid_ridge_candidates",
            )
            if key in successful
        },
    }
    destination = output / f"audit_{tier}"
    destination.mkdir(exist_ok=True)
    frame.to_csv(destination / "shard_checks.csv", index=False)
    solver_frame = pd.DataFrame(solver_rows)
    solver_frame.to_csv(destination / "solver_starts.csv", index=False)
    if len(solver_frame):
        failure_summary = solver_frame.groupby(["model_id", "stage", "fit_ridge"]).agg(
            starts=("start", "size"),
            converged=("converged", "sum"),
            material_failed_improvements=("material_failed_improvement", "sum"),
            both_failed_starts=("both_starts_failed", "sum"),
            max_nfev=("nfev", "max"),
        )
        failure_summary["failed"] = failure_summary["starts"] - failure_summary["converged"]
        failure_summary.to_csv(destination / "solver_failures_by_form_ridge.csv")
    shutil.copyfile(Path(__file__), destination / Path(__file__).name)
    (destination / "verification.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    require(complete or allow_incomplete, "Audit is incomplete; see shard_checks.csv for missing or failed fits")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tier", choices=("smoke", "screen", "certify"), default="screen")
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--report-tasklist", type=Path, help="Audit exactly the tasks in a report's verification.json")
    args = parser.parse_args()
    print(
        json.dumps(audit(args.output, args.tier, args.allow_incomplete, args.report_tasklist), indent=2, sort_keys=True)
    )


if __name__ == "__main__":
    main()
