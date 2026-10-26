# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "scikit-learn", "threadpoolctl"]
# ///
"""Audit common absolute-floor CV with frozen round-five synthetic controls.

Run with ``uv run --no-sync python`` after registration and source freezing.
Stages prepare and controls perform no fits; controls verifies the 64 original
receipts and preserves their budgets. Fixed-law-preflight runs eight known-law
cases. Candidates stops each estimator at its first failed context, and replay
independently reconstructs every completed head without calling a fitter.
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import audit_round5_floor_recovery_20260921 as old
import nextgen_two_phase_round6_20260921 as runner
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from two_phase_benchmark_data_20260921 import BenchPanel

benchmark = runner.benchmark
ROOT = runner.ROOT
CANDIDATES = ("NG2-016", "NG2-017")
CONTROLS = {"NG2-016": "CTRL-LOG-AGG", "NG2-017": "NG2-012"}
OLD_OUTPUT = old.ROOT / "synthetic_v1"
BUDGET_PATH = old.ROOT / "control_budgets.json"
BUDGET_SHA256 = "c6a99f8038eb57a6f5e0389ecbf7c42ffbe13a6dda7bf9f0f7e7a5276b324f5b"
REPLAY_LIMIT = 1e-12
TRANSLATION_LIMIT = 1e-6
FLOOR_LIMIT = 1e-5
GAIN_LIMIT = 1e-6
TIED_LIMIT = 1e-6
MAPPING_LIMIT = 1e-8
FIT_WORKERS = 6


def source_hashes() -> dict[str, str]:
    modules = (old, old.previous_audit, old.sampler_audit, old.round4_audit, old.menu_audit, old.boundary)
    return {
        **runner.source_hashes(),
        **{str(Path(module.__file__).resolve()): benchmark.sha256(Path(module.__file__)) for module in modules},
        str(Path(__file__).resolve()): benchmark.sha256(Path(__file__)),
    }


def identity(prepared: Path) -> dict[str, Any]:
    freeze_path = ROOT / "source_freeze.json"
    freeze = json.loads(freeze_path.read_text())
    if freeze["environment"] != benchmark.environment_identity():
        raise ValueError("Use uv run --no-sync python in the frozen repository environment")
    frozen = freeze["sources"] | freeze["driver_sources"]
    benchmark.verify_hashes(frozen)
    sources = source_hashes()
    if any(frozen.get(path) != digest for path, digest in sources.items()):
        raise ValueError("The round-six freeze must cover all model and synthetic driver sources")
    registration = json.loads((ROOT / "registration.json").read_text())
    if {row["id"]: row["control"] for row in registration["models"]} != CONTROLS:
        raise ValueError("The registered candidate/control mapping changed")
    if tuple(runner.CANDIDATES) != CANDIDATES or set(runner.MODELS) != set(CANDIDATES):
        raise ValueError("The runner differs from the registered two-candidate matrix")
    manifest_path = prepared / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    geometry = {
        f"starcoder_wsd80_dense/{name}": manifest["outputs"][f"starcoder_wsd80_dense/{name}"]
        for name in ("rows.csv", "policies.npz", "metadata.json")
    }
    benchmark.verify_hashes(geometry, prepared)
    return {
        "sources": sources,
        "source_freeze": benchmark.receipt(freeze_path),
        "registration": benchmark.receipt(ROOT / "registration.json"),
        "environment": benchmark.environment_identity(),
        "prepared_manifest": benchmark.sha256(manifest_path),
        "geometry": geometry,
        "contexts": [dataclasses.asdict(cell) for cell in old.contexts()],
        "control_budgets": benchmark.receipt(BUDGET_PATH),
        "claim": "adaptive_synthetic_rejection_screen_not_statistical_certification",
    }


def verified_budgets(prepared: Path) -> dict[str, Any]:
    """Verify the original identity without relabeling old controls as round six."""
    if benchmark.sha256(BUDGET_PATH) != BUDGET_SHA256:
        raise ValueError("The frozen round-five control budgets changed")
    budgets = json.loads(BUDGET_PATH.read_text())
    if budgets["identity"] != old.identity(prepared) or budgets["control_context_count"] != 64:
        raise ValueError("The original budget identity or 64-control coverage changed")
    expected = {
        str((cell.folder(OLD_OUTPUT, model) / "complete.json").resolve())
        for model in CONTROLS.values()
        for cell in old.contexts()
    }
    receipts = budgets["control_receipts"]
    if len(receipts) != 64 or {item["path"] for item in receipts} != expected:
        raise ValueError("The budgets refer to different control cells")
    if not benchmark.valid_receipts(receipts):
        raise ValueError("An original control completion receipt changed")
    for receipt in receipts:
        path = Path(receipt["path"])
        benchmark.verify_hashes(json.loads(path.read_text())["outputs"], path.parent)
    return budgets


def immutable_json(path: Path, record: dict[str, Any]) -> None:
    if path.exists():
        if json.loads(path.read_text()) != record:
            raise ValueError(f"Existing artifact has a different identity or result: {path}")
        return
    benchmark.write_json(path, record)


def preparation(prepared: Path) -> dict[str, Any]:
    record = old.preparation(prepared)
    return {
        "identity": identity(prepared),
        **record,
        "native_envelope_scope": "Unchanged control diagnostic; not the common absolute-floor search domain",
        "translation_policy": "Same registered absolute search domain for both floors; no world-dependent recentering",
        "contexts_per_candidate": 32,
        "stage_a_contexts": 14,
        "stage_b_contexts": 18,
    }


def replay_controls(prepared: Path, output: Path) -> dict[str, Any]:
    budgets = verified_budgets(prepared)
    rows = [old.replay_cell(prepared, OLD_OUTPUT, cell, model) for model in CONTROLS.values() for cell in old.contexts()]
    report = {
        "identity": identity(prepared),
        "original_identity": budgets["identity"],
        "control_budgets": benchmark.receipt(BUDGET_PATH),
        "control_context_count": len(rows),
        "control_receipts": budgets["control_receipts"],
        "maximum_serialized_prediction_replay_error": max(
            row["maximum_serialized_prediction_replay_error"] for row in rows
        ),
        "new_control_fits": 0,
        "new_budgets_derived": False,
    }
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output / "control_summary.csv", index=False)
    report["control_summary"] = benchmark.receipt(output / "control_summary.csv")
    immutable_json(ROOT / "control_reuse.json", report)
    return report


def load_fit(record: dict[str, Any]) -> runner.Fit:
    head = runner.previous.log_head.Head(
        record["intercept"],
        np.asarray(record["aggregate_coefficients"]),
        np.asarray(record["phase_coefficients"]),
        record["floor"],
    )
    return runner.Fit(
        model=record["model"],
        shape=record["shape"],
        q=record["q"],
        kappa=record["kappa"],
        ridge=record["ridge"],
        inner_rmse=record["inner_rmse"],
        head=head,
        floor_search=record["floor_search"],
    )


def replay_predictions(panel: BenchPanel, record: dict[str, Any]) -> np.ndarray:
    if record["phase_coefficients"] or (record["model"] == "NG2-016" and record["kappa"] != 0):
        raise ValueError("Unexpected independent phase coefficients or aggregate-arm phase state")
    head = benchmark.mariner.Head(record["intercept"], np.asarray(record["aggregate_coefficients"]), record["floor"])
    return head.predict(old.boundary.design_matrix(panel, record["shape"], record["q"], record["kappa"]))


def replay_frame(panel: BenchPanel, frame: pd.DataFrame, record: dict[str, Any], cell: old.Context) -> float:
    tied = old.previous_audit.tied_queries(panel)
    reconstructed = old.menu_audit.prediction_frame(
        panel,
        old.world_mean(panel, cell),
        replay_predictions(panel, record),
        old.world_mean(tied, cell),
        replay_predictions(tied, record),
    )
    errors = []
    for column in reconstructed:
        if column in {"prediction", "tied_prediction"}:
            error = float(np.max(np.abs(frame[column].to_numpy() - reconstructed[column].to_numpy())))
            if not np.isfinite(error) or error > REPLAY_LIMIT:
                raise ValueError(f"Independent serialized-head replay failed: {column} {error}")
            errors.append(error)
        elif not np.array_equal(frame[column].to_numpy(), reconstructed[column].to_numpy()):
            raise ValueError(f"Saved geometry or truth differs: {column}")
    return max(errors)


def mapping_checks(fit: runner.Fit, panel: BenchPanel) -> dict[str, float]:
    rows = np.flatnonzero(panel.tied)
    indices = np.arange(len(panel.rows))
    native = benchmark.mariner.Head(fit.head.intercept, fit.head.aggregate_coefficients, fit.head.floor)
    actual = fit.predict(panel, rows)
    tied = old.previous_audit.tied_queries(panel)
    subset = dataclasses.replace(
        panel,
        phase0=panel.phase0[rows],
        phase1=panel.phase1[rows],
        rows=panel.rows.iloc[rows].reset_index(drop=True),
        targets={},
        metadata={
            **panel.metadata,
            "realized_phase0_epochs": panel.phase0_epochs[rows],
            "realized_phase1_epochs": panel.phase1_epochs[rows],
        },
    )
    return {
        "native_tied_max_error": float(
            np.max(np.abs(actual - native.predict(benchmark.mariner.design_matrix(panel.epochs[rows], fit.shape))))
        ),
        "rowwise_max_error": float(
            np.max(np.abs(actual - np.concatenate([fit.predict(panel, np.array([row])) for row in rows])))
        ),
        "foreign_rows_removed_max_error": float(np.max(np.abs(actual - fit.predict(subset, np.arange(len(rows)))))),
        "full_batch_max_error": float(np.max(np.abs(actual - fit.predict(panel, indices)[rows]))),
        "counterfactual_tied_max_error": float(
            np.max(
                np.abs(
                    fit.predict(tied, indices) - native.predict(benchmark.mariner.design_matrix(tied.epochs, fit.shape))
                )
            )
        ),
    }


def inner_head_audit(panel: BenchPanel, training: pd.DataFrame, record: dict[str, Any]) -> dict[str, Any]:
    """Replay held-out risk and verify the saved heads use the common floor."""
    inner = training.inner_fold.to_numpy(int)
    response = training.response.to_numpy()
    search = record["floor_search"]
    selected = search["selected_configuration_search"]
    if search["final_floor_unchanged"] is not True or selected["floor"] != record["floor"]:
        raise ValueError("The final fit changed the selected absolute floor")
    folds = sorted(set(inner))
    if len(search["selected_configuration_inner_heads"]) != len(folds):
        raise ValueError("Saved inner heads do not cover the actual folds")
    squared_error, validation_count = 0.0, 0
    for fold, saved in zip(folds, search["selected_configuration_inner_heads"], strict=True):
        fitting, validation = np.flatnonzero(inner != fold), np.flatnonzero(inner == fold)
        if (
            saved["training_rows"] != panel.rows.iloc[fitting].row_id.tolist()
            or saved["validation_rows"] != panel.rows.iloc[validation].row_id.tolist()
        ):
            raise ValueError("Saved inner head uses different actual training or validation rows")
        if saved["floor"] != record["floor"] or saved["phase_coefficients"]:
            raise ValueError("Saved inner head changed the common floor or added phase coefficients")
        head = benchmark.mariner.Head(saved["intercept"], np.asarray(saved["aggregate_coefficients"]), saved["floor"])
        matrix = old.boundary.design_matrix(panel, record["shape"], record["q"], record["kappa"])
        predicted = head.predict(matrix[validation])
        squared_error += float(np.sum((predicted - response[validation]) ** 2))
        validation_count += len(validation)
    score = float(np.sqrt(squared_error / validation_count))
    error = max(abs(score - record["inner_rmse"]), abs(score - selected["inner_rmse"]))
    if not np.isfinite(error) or error > REPLAY_LIMIT:
        raise ValueError("Saved inner heads do not reproduce the selected pooled BPB CV score")
    return {
        "folds": len(folds),
        "validation_rows": validation_count,
        "replayed_inner_rmse": score,
        "score_replay_error": error,
        "common_floor_verified": True,
    }


def generic_mapping(base: BenchPanel, current: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for model in CANDIDATES:
        for q, kappa in runner.MODELS[model].states:
            for shape in (
                {"rate": 0.05, "power": 0.3, "threshold": 1.0},
                old.round4_audit.SHAPE,
                {"rate": 4.0, "power": 1.0, "threshold": 6.0},
            ):
                for floor in (-3.0, 0.7):
                    head = runner.previous.log_head.Head(-0.2, np.array([0.8, 0.3, 0.1, 0.04]), np.empty(0), floor)
                    fit = runner.Fit(
                        model=model,
                        shape=dict(shape),
                        q=q,
                        kappa=kappa,
                        ridge=0.0,
                        inner_rmse=0.0,
                        head=head,
                        floor_search={},
                    )
                    checks = mapping_checks(fit, base)
                    rows.append(
                        {
                            "model": model,
                            "shape": shape,
                            "q": q,
                            "kappa": kappa,
                            "floor": floor,
                            "checks": checks,
                            "passed": all(value <= MAPPING_LIMIT for value in checks.values()),
                        }
                    )
    report = {"identity": current, "passed": all(row["passed"] for row in rows), "limit": MAPPING_LIMIT, "generic": rows}
    immutable_json(ROOT / "generic_structural_mapping.json", report)
    return report


def fixed_law_case(prepared: Path, base: BenchPanel, cell: old.Context) -> dict[str, Any]:
    panel, inner = old.sampler_audit.sampler_panel(base, cell.sampler)
    response = old.world_mean(panel, cell)
    model = "NG2-017" if cell.world == "boundary" else "NG2-016"
    q, kappa = 0.5, 0.5 if cell.world == "boundary" else 0.0
    shape = dict(old.round4_audit.SHAPE)
    train = np.arange(len(panel.rows))
    folder = ROOT / "fixed_law" / cell.world / cell.sampler / cell.floor_name
    config = {
        "identity": identity(prepared),
        "context": dataclasses.asdict(cell),
        "model": model,
        "true_floor": old.sampler_audit.FLOOR + cell.floor_shift,
        "known_shape": shape,
        "known_q": q,
        "known_kappa": kappa,
        "ridge": 0.0,
        "training_rows": panel.rows.row_id.tolist(),
        "inner_folds": inner.tolist(),
        "claim": "Known-law common-floor CV recovery; not full model-selection evidence",
    }
    menu = old.menu_audit.raw_menu(base)
    tied = old.previous_audit.tied_queries(panel)
    tied_menu = old.previous_audit.tied_queries(menu)
    queries = np.arange(len(menu.rows))
    if not old.previous_audit.check_completed(folder, config):
        immutable_json(folder / "config.json", config)
        started = time.monotonic()
        with threadpool_limits(limits=1):
            matrix = runner.matrices(panel, train, shape, q, kappa)
            folds = tuple((np.flatnonzero(inner != fold), np.flatnonzero(inner == fold)) for fold in sorted(set(inner)))
            searched = runner.floor_cv.search_floor_cv(matrix, response, 0.0, folds)
            head = runner.floor_cv.fixed_floor_head(matrix, response, 0.0, searched.floor)
            inner_heads = []
            for fitting, validation in folds:
                conditional = runner.floor_cv.fixed_floor_head(matrix[fitting], response[fitting], 0.0, searched.floor)
                inner_heads.append(
                    {
                        "training_rows": panel.rows.iloc[fitting].row_id.tolist(),
                        "validation_rows": panel.rows.iloc[validation].row_id.tolist(),
                        "floor": conditional.floor,
                        "intercept": conditional.intercept,
                        "aggregate_coefficients": conditional.aggregate_coefficients.tolist(),
                        "phase_coefficients": conditional.phase_coefficients.tolist(),
                    }
                )
            search_record = {
                "selected_configuration_search": searched.to_json(),
                "selected_configuration_inner_heads": inner_heads,
                "final_floor_unchanged": head.floor == searched.floor,
            }
            fitted = runner.Fit(
                model=model,
                shape=shape,
                q=q,
                kappa=kappa,
                ridge=0.0,
                inner_rmse=searched.inner_rmse,
                head=head,
                floor_search=search_record,
            )
            training = old.menu_audit.prediction_frame(
                panel, response, fitted.predict(panel, train), old.world_mean(tied, cell), fitted.predict(tied, train)
            )
            training["response"], training["inner_fold"] = response, inner
            raw = old.menu_audit.prediction_frame(
                menu,
                old.world_mean(menu, cell),
                fitted.predict(menu, queries),
                old.world_mean(tied_menu, cell),
                fitted.predict(tied_menu, queries),
            )
        record = fitted.to_json() | {
            "training_rows": panel.rows.row_id.tolist(),
            "inner_folds": inner.tolist(),
            "runtime_seconds": time.monotonic() - started,
        }
        benchmark.write_json(folder / "fit.json", record)
        benchmark.write_json(folder / "floor_search.json", fitted.floor_search)
        pd.DataFrame(searched.to_json()["trace"]).to_csv(folder / "profile_trace.csv", index=False)
        training.to_csv(folder / "training_predictions.csv", index=False)
        raw.to_csv(folder / "raw_menu.csv", index=False)
        metrics = fixed_law_metrics(base, panel, menu, training, raw, record, cell)
        benchmark.write_json(folder / "metrics.json", metrics)
        names = (
            "config.json",
            "fit.json",
            "floor_search.json",
            "profile_trace.csv",
            "training_predictions.csv",
            "raw_menu.csv",
            "metrics.json",
        )
        benchmark.write_json(
            folder / "complete.json",
            {"config": config, "outputs": {name: benchmark.sha256(folder / name) for name in names}},
        )
    record = json.loads((folder / "fit.json").read_text())
    training = pd.read_csv(folder / "training_predictions.csv", float_precision="round_trip")
    raw = pd.read_csv(folder / "raw_menu.csv", float_precision="round_trip")
    if not np.array_equal(training.response.to_numpy(), response) or not np.array_equal(training.inner_fold, inner):
        raise ValueError("Known-law labels or actual grouped folds changed")
    if record["training_rows"] != config["training_rows"] or record["inner_folds"] != config["inner_folds"]:
        raise ValueError("Known-law training identities changed")
    metrics = fixed_law_metrics(base, panel, menu, training, raw, record, cell)
    if json.loads((folder / "metrics.json").read_text()) != metrics:
        raise ValueError("Known-law metrics differ from independent replay")
    return {"context": dataclasses.asdict(cell), **metrics, "completion": benchmark.receipt(folder / "complete.json")}


def fixed_law_metrics(
    base: BenchPanel,
    panel: BenchPanel,
    menu: BenchPanel,
    training: pd.DataFrame,
    raw: pd.DataFrame,
    record: dict[str, Any],
    cell: old.Context,
) -> dict[str, Any]:
    metrics = old.raw_metrics(raw)
    floor_error = abs(record["floor"] - old.sampler_audit.FLOOR - cell.floor_shift)
    checks = {
        "matched_gain_rmse": metrics["matched_gain_rmse"] <= GAIN_LIMIT,
        "uniform_101_tied_curve_rmse": metrics["uniform_101_tied_curve_rmse"] <= TIED_LIMIT,
        "absolute_floor_error": floor_error <= FLOOR_LIMIT,
    }
    mapping = mapping_checks(load_fit(record), base)
    return {
        "fitted_floor": record["floor"],
        "absolute_floor_error": floor_error,
        "raw_menu": metrics,
        "checks": checks,
        "mapping": mapping,
        "mapping_passed": all(value <= MAPPING_LIMIT for value in mapping.values()),
        "passed": all(checks.values()),
        "runtime_seconds": record["runtime_seconds"],
        "inner_head_audit": inner_head_audit(panel, training, record),
        "maximum_serialized_prediction_replay_error": max(
            replay_frame(panel, training, record, cell), replay_frame(menu, raw, record, cell)
        ),
    }


def fixed_law_preflight(prepared: Path) -> dict[str, Any]:
    current = identity(prepared)
    verified_budgets(prepared)
    base = old.previous_audit.geometry_panel(prepared, "starcoder_wsd80_dense")
    generic = generic_mapping(base, current)
    results = []
    for world, sampler, shift in itertools.product(
        ("null", "boundary"), old.sampler_audit.SAMPLERS, (0.0, old.NEGATIVE_SHIFT)
    ):
        cell = old.Context(world, sampler, 0.0, old.sampler_audit.SEEDS[0], shift)
        result = fixed_law_case(prepared, base, cell)
        results.append(result)
        print(
            json.dumps(
                {
                    "fixed_law": dataclasses.asdict(cell),
                    "passed": result["passed"],
                    "absolute_floor_error": result["absolute_floor_error"],
                }
            ),
            flush=True,
        )
    if current != identity(prepared):
        raise ValueError("Source identity changed during the known-law preflight")
    outputs = {"generic_structural_mapping.json": benchmark.sha256(ROOT / "generic_structural_mapping.json")}
    for row in results:
        completion = Path(row["completion"]["path"])
        outputs[str(completion.relative_to(ROOT))] = row["completion"]["sha256"]
        outputs.update(
            {
                str((completion.parent / name).relative_to(ROOT)): digest
                for name, digest in json.loads(completion.read_text())["outputs"].items()
            }
        )
    report = {
        "identity": current,
        "source_freeze_sha256": benchmark.sha256(ROOT / "source_freeze.json"),
        "passed": (
            len(results) == 8 and generic["passed"] and all(row["passed"] and row["mapping_passed"] for row in results)
        ),
        "fits": len(results),
        "limits": {"gain_rmse": GAIN_LIMIT, "uniform_tied_rmse": TIED_LIMIT, "absolute_floor_error": FLOOR_LIMIT},
        "cases": results,
        "outputs": outputs,
    }
    immutable_json(ROOT / "fixed_law_preflight.json", report)
    return report


def candidate_prerequisites(prepared: Path) -> dict[str, Any]:
    budgets = verified_budgets(prepared)
    reuse = json.loads((ROOT / "control_reuse.json").read_text())
    if reuse["identity"] != identity(prepared) or reuse["control_receipts"] != budgets["control_receipts"]:
        raise ValueError("Round-six control replay does not match frozen identities")
    if not benchmark.valid_receipts([reuse["control_summary"]]):
        raise ValueError("Replayed control summary changed")
    preflight = json.loads((ROOT / "fixed_law_preflight.json").read_text())
    if (
        preflight["passed"] is not True
        or preflight["source_freeze_sha256"] != benchmark.sha256(ROOT / "source_freeze.json")
        or preflight["identity"] != identity(prepared)
    ):
        raise ValueError("The eight known-law common-floor recovery cases have not passed")
    benchmark.verify_hashes(preflight["outputs"], ROOT)
    return budgets


def config_for(prepared: Path, cell: old.Context, model: str) -> dict[str, Any]:
    return {
        "identity": identity(prepared),
        "context": dataclasses.asdict(cell),
        "model": model,
        "control_reuse": benchmark.receipt(ROOT / "control_reuse.json"),
        "fixed_law_preflight": benchmark.receipt(ROOT / "fixed_law_preflight.json"),
    }


def audit_fit(prepared: Path, output: Path, cell: old.Context, model: str) -> None:
    folder = cell.folder(output, model)
    config = config_for(prepared, cell, model)
    if old.previous_audit.check_completed(folder, config):
        return
    immutable_json(folder / "config.json", config)
    print(json.dumps({"started": str(folder), "model": model, "context": dataclasses.asdict(cell)}), flush=True)
    base = old.previous_audit.geometry_panel(prepared, "starcoder_wsd80_dense")
    panel, inner = old.sampler_audit.sampler_panel(base, cell.sampler)
    train = np.arange(len(panel.rows))
    if panel.tied.any():
        raise ValueError("Synthetic fitting must use asymmetric endpoints alone")
    truth = old.world_mean(panel, cell)
    response = truth + np.random.default_rng(cell.seed).normal(0, cell.noise_sd, len(truth))
    tied, menu = old.previous_audit.tied_queries(panel), old.menu_audit.raw_menu(base)
    tied_menu = old.previous_audit.tied_queries(menu)
    started = time.monotonic()
    with threadpool_limits(limits=1):
        fitted = runner.fit_model(panel, response, train, inner, model, workers=FIT_WORKERS)
        training = old.menu_audit.prediction_frame(
            panel, truth, fitted.predict(panel, train), old.world_mean(tied, cell), fitted.predict(tied, train)
        )
        training["response"], training["inner_fold"] = response, inner
        indices = np.arange(len(menu.rows))
        raw = old.menu_audit.prediction_frame(
            menu,
            old.world_mean(menu, cell),
            fitted.predict(menu, indices),
            old.world_mean(tied_menu, cell),
            fitted.predict(tied_menu, indices),
        )
    record = fitted.to_json() | {"training_rows": panel.rows.row_id.tolist(), "inner_folds": inner.tolist()}
    expected_configurations = (
        len(benchmark.mariner.SHAPES) * len(benchmark.mariner.RIDGE_GRID) * len(runner.MODELS[model].states)
    )
    if len(fitted.floor_search["configuration_minima"]) != expected_configurations:
        raise ValueError("Saved floor searches do not cover every registered law/ridge/state configuration")
    metrics = {
        **old.raw_metrics(raw),
        "training_endpoint_rmse": float(np.sqrt(np.mean((training.prediction - truth) ** 2))),
        "maximum_serialized_prediction_replay_error": max(
            replay_frame(panel, training, record, cell), replay_frame(menu, raw, record, cell)
        ),
        "elapsed_seconds": time.monotonic() - started,
        "mapping": mapping_checks(fitted, base),
        "inner_head_audit": inner_head_audit(panel, training, record),
    }
    benchmark.write_json(folder / "fit.json", record)
    benchmark.write_json(folder / "floor_search.json", fitted.floor_search)
    benchmark.write_json(folder / "metrics.json", metrics)
    training.to_csv(folder / "training_predictions.csv", index=False)
    raw.to_csv(folder / "raw_menu.csv", index=False)
    names = ("config.json", "fit.json", "floor_search.json", "metrics.json", "training_predictions.csv", "raw_menu.csv")
    benchmark.write_json(
        folder / "complete.json",
        {"config": config, "outputs": {name: benchmark.sha256(folder / name) for name in names}},
    )


def replay_cell(prepared: Path, output: Path, cell: old.Context, model: str) -> dict[str, Any]:
    folder = cell.folder(output, model)
    if not old.previous_audit.check_completed(folder, config_for(prepared, cell, model)):
        raise ValueError(f"Missing completed candidate cell: {folder}")
    record = json.loads((folder / "fit.json").read_text())
    metrics = json.loads((folder / "metrics.json").read_text())
    base = old.previous_audit.geometry_panel(prepared, "starcoder_wsd80_dense")
    panel, inner = old.sampler_audit.sampler_panel(base, cell.sampler)
    menu = old.menu_audit.raw_menu(base)
    training = pd.read_csv(folder / "training_predictions.csv", float_precision="round_trip")
    raw = pd.read_csv(folder / "raw_menu.csv", float_precision="round_trip")
    response = old.world_mean(panel, cell) + np.random.default_rng(cell.seed).normal(0, cell.noise_sd, len(panel.rows))
    if (
        record["model"] != model
        or record["training_rows"] != panel.rows.row_id.tolist()
        or record["inner_folds"] != inner.tolist()
    ):
        raise ValueError("Saved candidate model or training identities differ")
    if not np.array_equal(training.response.to_numpy(), response) or not np.array_equal(training.inner_fold, inner):
        raise ValueError("Saved responses or actual grouped folds differ")
    replay = max(replay_frame(panel, training, record, cell), replay_frame(menu, raw, record, cell))
    recomputed = old.raw_metrics(raw)
    recomputed["training_endpoint_rmse"] = float(np.sqrt(np.mean((training.prediction - training.truth) ** 2)))
    recomputed["mapping"] = mapping_checks(load_fit(record), base)
    recomputed["inner_head_audit"] = inner_head_audit(panel, training, record)
    if any(metrics[key] != value for key, value in recomputed.items()):
        raise ValueError("Candidate metrics differ from independent prediction replay")
    return dataclasses.asdict(cell) | {
        "model": model,
        "stage": cell.stage,
        "floor": record["floor"],
        "ridge": record["ridge"],
        **metrics,
        "maximum_serialized_prediction_replay_error": replay,
    }


def translation_audit(output: Path, model: str, cell: old.Context) -> dict[str, Any] | None:
    if cell.floor_shift == 0:
        return None
    positive = dataclasses.replace(cell, floor_shift=0.0)
    first = pd.read_csv(positive.folder(output, model) / "raw_menu.csv", float_precision="round_trip")
    shifted = pd.read_csv(cell.folder(output, model) / "raw_menu.csv", float_precision="round_trip")
    errors = {
        column: float(np.max(np.abs(shifted[column].to_numpy() - first[column].to_numpy() - old.NEGATIVE_SHIFT)))
        for column in ("prediction", "tied_prediction")
    }
    fields = ("shape", "q", "kappa", "ridge", "floor", "intercept", "aggregate_coefficients")
    parameters = {}
    for name, context in (("positive_parameters", positive), ("negative_parameters", cell)):
        record = json.loads((context.folder(output, model) / "fit.json").read_text())
        parameters[name] = {key: record[key] for key in fields}
    return {
        "context": dataclasses.asdict(cell),
        "maximum_prediction_shift_error": errors["prediction"],
        "maximum_tied_prediction_shift_error": errors["tied_prediction"],
        "limit": TRANSLATION_LIMIT,
        "passed": errors["prediction"] <= TRANSLATION_LIMIT,
        **parameters,
    }


def budget_failures(row: dict[str, Any], budgets: dict[str, Any]) -> list[dict[str, Any]]:
    cohorts = [
        cohort
        for cohort in budgets["cohorts"]
        if cohort["control"] == CONTROLS[row["model"]]
        and cohort["world"] == row["world"]
        and cohort["noise_sd"] == row["noise_sd"]
    ]
    if len(cohorts) != 1:
        raise ValueError("No unique frozen same-loss world/noise budget")
    failures = [
        {"metric": metric, "observed": row[metric], "limit": limit}
        for metric, limit in cohorts[0]["limits"].items()
        if not np.isfinite(row[metric]) or row[metric] > limit
    ]
    failures.extend(
        {"metric": f"mapping_{name}", "observed": value, "limit": MAPPING_LIMIT}
        for name, value in row["mapping"].items()
        if not np.isfinite(value) or value > MAPPING_LIMIT
    )
    return failures


def candidate_run(arguments: tuple[str, str, str]) -> dict[str, Any]:
    prepared_name, output_name, model = arguments
    prepared, output = Path(prepared_name), Path(output_name)
    budgets = candidate_prerequisites(prepared)
    rows, decisions, translations = [], [], []
    report: dict[str, Any] = {}
    for cell in old.contexts():
        audit_fit(prepared, output, cell, model)
        row = replay_cell(prepared, output, cell, model)
        rows.append(row)
        failures = budget_failures(row, budgets)
        translation = translation_audit(output, model, cell)
        if translation is not None:
            translations.append(translation)
            if not translation["passed"]:
                failures.append(
                    {
                        "metric": "translation_max_error",
                        "observed": translation["maximum_prediction_shift_error"],
                        "limit": TRANSLATION_LIMIT,
                    }
                )
        decisions.append({"context": dataclasses.asdict(cell), "stage": cell.stage, "failures": failures})
        report = {
            "identity": identity(prepared),
            "model": model,
            "completed_contexts": len(rows),
            "unrun_contexts": 32 - len(rows),
            "status": "rejected" if failures else "incomplete",
            "decisions": decisions,
            "translations": translations,
            "control_budgets": benchmark.receipt(BUDGET_PATH),
            "control_reuse": benchmark.receipt(ROOT / "control_reuse.json"),
            "fixed_law_preflight": benchmark.receipt(ROOT / "fixed_law_preflight.json"),
        }
        benchmark.write_json(output / model / "decision.json", report)
        pd.DataFrame(rows).to_csv(output / model / "summary.csv", index=False)
        print(
            json.dumps(
                {
                    "model": model,
                    "context": dataclasses.asdict(cell),
                    "stage": cell.stage,
                    "completed": len(rows),
                    "failures": failures,
                }
            ),
            flush=True,
        )
        if failures:
            return report
    improvement = None
    if model == "NG2-017":
        selected = [
            row for row in rows if row["world"] == "boundary" and row["noise_sd"] > 0 and row["floor_shift"] == 0
        ]
        controls = [
            old.replay_cell(prepared, OLD_OUTPUT, cell, CONTROLS[model])
            for cell in old.contexts()
            if cell.world == "boundary" and cell.noise_sd > 0 and cell.floor_shift == 0
        ]
        if len(selected) != 6 or len(controls) != 6:
            raise ValueError("Phase improvement requires all six positive noisy boundary contexts")
        candidate_mse = float(np.mean([row["matched_gain_rmse"] ** 2 for row in selected]))
        control_mse = float(np.mean([row["matched_gain_rmse"] ** 2 for row in controls]))
        improvement = {
            "positive_noisy_boundary_contexts": 6,
            "candidate_mean_squared_gain_error": candidate_mse,
            "control_mean_squared_gain_error": control_mse,
            "passed": candidate_mse < control_mse,
        }
    report["phase_improvement"] = improvement
    report["status"] = "synthetic_survivor" if improvement is None or improvement["passed"] else "rejected"
    benchmark.write_json(output / model / "decision.json", report)
    return report


def replay_all(prepared: Path, output: Path) -> dict[str, Any]:
    budgets = verified_budgets(prepared)
    rows = [old.replay_cell(prepared, OLD_OUTPUT, cell, model) for model in CONTROLS.values() for cell in old.contexts()]
    translations, comparisons, receipts, decisions = [], [], list(budgets["control_receipts"]), []
    for model in CANDIDATES:
        completed = [cell for cell in old.contexts() if (cell.folder(output, model) / "complete.json").exists()]
        if completed != list(old.contexts()[: len(completed)]):
            raise ValueError("Completed candidate cells are not an ordered prefix")
        cell_decisions = []
        for index, cell in enumerate(completed):
            row = replay_cell(prepared, output, cell, model)
            rows.append(row)
            failures = budget_failures(row, budgets)
            receipts.append(benchmark.receipt(cell.folder(output, model) / "complete.json"))
            translation = translation_audit(output, model, cell)
            if translation is not None:
                translations.append({"model": model, **translation})
                if not translation["passed"]:
                    failures.append(
                        {
                            "metric": "translation_max_error",
                            "observed": translation["maximum_prediction_shift_error"],
                            "limit": TRANSLATION_LIMIT,
                        }
                    )
            cell_decisions.append({"context": dataclasses.asdict(cell), "stage": cell.stage, "failures": failures})
            if failures and index != len(completed) - 1:
                raise ValueError("A candidate has completed cells after its first gate failure")
            other = dataclasses.replace(cell, sampler="positive")
            if cell.sampler != "balanced" or other not in completed:
                continue
            first = pd.read_csv(cell.folder(output, model) / "raw_menu.csv", float_precision="round_trip")
            second = pd.read_csv(other.folder(output, model) / "raw_menu.csv", float_precision="round_trip")
            for column in ("row_id", "phase0_code", "phase1_code", "truth", "tied_truth"):
                if not np.array_equal(first[column].to_numpy(), second[column].to_numpy()):
                    raise ValueError("Sampler comparison geometry or truth differs")
            tied = first.tied.to_numpy(bool)
            drift = first.prediction.to_numpy()[tied] - second.prediction.to_numpy()[tied]
            gain_drift = (
                first.tied_prediction.to_numpy()
                - first.prediction.to_numpy()
                - second.tied_prediction.to_numpy()
                + second.prediction.to_numpy()
            )
            comparisons.append(
                dataclasses.asdict(cell)
                | {
                    "model": model,
                    "tied_curve_sampler_drift_rmse": float(np.sqrt(np.mean(drift**2))),
                    "tied_curve_sampler_drift_max_abs": float(np.max(np.abs(drift))),
                    "matched_gain_sampler_drift_rmse": float(np.sqrt(np.mean(gain_drift**2))),
                }
            )
        decision_path = output / model / "decision.json"
        if decision_path.exists():
            decision = json.loads(decision_path.read_text())
            if (
                decision["identity"] != identity(prepared)
                or decision["decisions"] != cell_decisions
                or decision["completed_contexts"] != len(completed)
            ):
                raise ValueError("Saved candidate decisions differ from the replayed ordered gate results")
            if cell_decisions and cell_decisions[-1]["failures"] and decision["status"] != "rejected":
                raise ValueError("Failed candidate was not marked rejected")
            if len(completed) == 32 and not cell_decisions[-1]["failures"]:
                improvement = None
                if model == "NG2-017":
                    cohort = [
                        row
                        for row in rows
                        if row["world"] == "boundary" and row["noise_sd"] > 0 and row["floor_shift"] == 0
                    ]
                    candidate_rows = [row for row in cohort if row["model"] == model]
                    control_rows = [row for row in cohort if row["model"] == CONTROLS[model]]
                    if len(candidate_rows) != 6 or len(control_rows) != 6:
                        raise ValueError("Phase-improvement replay requires six candidate and paired control cells")
                    candidate_mse = float(np.mean([row["matched_gain_rmse"] ** 2 for row in candidate_rows]))
                    control_mse = float(np.mean([row["matched_gain_rmse"] ** 2 for row in control_rows]))
                    improvement = {
                        "positive_noisy_boundary_contexts": 6,
                        "candidate_mean_squared_gain_error": candidate_mse,
                        "control_mean_squared_gain_error": control_mse,
                        "passed": candidate_mse < control_mse,
                    }
                status = "synthetic_survivor" if improvement is None or improvement["passed"] else "rejected"
                if decision["phase_improvement"] != improvement or decision["status"] != status:
                    raise ValueError("Saved final phase-improvement decision or status differs from independent replay")
            decisions.append(benchmark.receipt(decision_path))
    output.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(rows)
    summary.to_csv(output / "replay_summary.csv", index=False)
    pd.DataFrame(comparisons).to_csv(output / "sampler_comparison.csv", index=False)
    report = {
        "identity": identity(prepared),
        "completed_contexts": len(rows),
        "counts_by_model": summary.groupby("model").size().to_dict(),
        "maximum_serialized_prediction_replay_error": float(summary.maximum_serialized_prediction_replay_error.max()),
        "translations": translations,
        "completion_receipts": receipts,
        "decision_receipts": decisions,
        "outputs": {name: benchmark.sha256(output / name) for name in ("replay_summary.csv", "sampler_comparison.csv")},
        "new_fits": 0,
        "measured_outcomes_read": False,
        "lm_jobs": 0,
    }
    benchmark.write_json(output / "replay.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Order: prepare, controls, fixed-law-preflight, candidates, replay. No stage changes the old budgets.",
    )
    parser.add_argument("--prepared", type=Path, default=old.previous_audit.PREPARED)
    parser.add_argument("--output", type=Path, default=ROOT / "synthetic_v1")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--stage", choices=("prepare", "controls", "fixed-law-preflight", "candidates", "replay"), required=True
    )
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("Workers must be positive")
    immutable_json(args.output / "preparation.json", preparation(args.prepared))
    if args.stage == "prepare":
        verified_budgets(args.prepared)
        print(json.dumps({"contexts_per_candidate": 32, "old_control_receipts": 64, "new_fits": 0}), flush=True)
    elif args.stage == "controls":
        report = replay_controls(args.prepared, args.output)
        print(json.dumps({"verified_controls": report["control_context_count"], "new_fits": 0}), flush=True)
    elif args.stage == "fixed-law-preflight":
        report = fixed_law_preflight(args.prepared)
        print(json.dumps({"fixed_law_cases": report["fits"], "passed": report["passed"]}), flush=True)
    elif args.stage == "candidates":
        candidate_prerequisites(args.prepared)
        with ProcessPoolExecutor(max_workers=min(args.workers, len(CANDIDATES))) as pool:
            futures = [pool.submit(candidate_run, (str(args.prepared), str(args.output), model)) for model in CANDIDATES]
            for future in as_completed(futures):
                report = future.result()
                print(
                    json.dumps(
                        {key: report[key] for key in ("model", "status", "completed_contexts", "unrun_contexts")}
                    ),
                    flush=True,
                )
        replay_all(args.prepared, args.output)
    else:
        report = replay_all(args.prepared, args.output)
        print(json.dumps({"replayed_contexts": report["completed_contexts"], "new_fits": 0}), flush=True)


if __name__ == "__main__":
    main()
