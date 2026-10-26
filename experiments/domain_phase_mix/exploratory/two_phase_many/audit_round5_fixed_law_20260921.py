# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "scikit-learn", "threadpoolctl"]
# ///
"""Eight registered known-law floor profiles and native tied-class preflight.

Run only after the round-five source freeze, with uv run --no-sync python.
The synthetic fits know the generating shape and phase law but estimate their
floor from asymmetric responses alone. They do not certify model selection.
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
from pathlib import Path
from typing import Any

import audit_round4_sampler_recovery_20260921_v2 as synthetic
import benchmark_two_phase_20260921 as benchmark
import nextgen_two_phase_round5_20260921 as runner
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from two_phase_benchmark_data_20260921 import BenchPanel

ROOT = runner.ROOT
WORLDS = ("null", "boundary")
SHIFTS = (0.0, -0.51)
GAIN_LIMIT = 1e-6
TIED_LIMIT = 1e-6
FLOOR_LIMIT = 1e-5
MAPPING_LIMIT = 1e-8


def identity(prepared: Path) -> dict[str, Any]:
    """Require frozen fitting, helper, environment and geometry identities."""
    freeze_path = ROOT / "source_freeze.json"
    freeze = json.loads(freeze_path.read_text())
    benchmark.verify_hashes(freeze["sources"])
    benchmark.verify_hashes(freeze["driver_sources"])
    if freeze["environment"] != benchmark.environment_identity():
        raise ValueError("Run with uv run --no-sync python in the frozen environment")
    sources = runner.source_hashes()
    if any(freeze["sources"].get(path) != digest for path, digest in sources.items()):
        raise ValueError("Round-five runner differs from the registered source freeze")
    driver = str(Path(__file__).resolve())
    if freeze["driver_sources"].get(driver) != benchmark.sha256(Path(__file__)):
        raise ValueError("Fixed-law audit driver must be included in the source freeze")
    modules = (
        synthetic,
        synthetic.previous_audit,
        synthetic.sampler_audit,
        synthetic.menu_audit,
    )
    sources.update({str(Path(module.__file__).resolve()): benchmark.sha256(Path(module.__file__)) for module in modules})
    sources[driver] = benchmark.sha256(Path(__file__))
    frozen_sources = {**freeze["sources"], **freeze["driver_sources"]}
    if any(frozen_sources.get(path) != digest for path, digest in sources.items()):
        raise ValueError("Every fixed-law helper must be included in the source freeze")
    manifest = json.loads((prepared / "manifest.json").read_text())
    geometry = {
        f"starcoder_wsd80_dense/{name}": manifest["outputs"][f"starcoder_wsd80_dense/{name}"]
        for name in ("rows.csv", "policies.npz", "metadata.json")
    }
    benchmark.verify_hashes(geometry, prepared)
    return {
        "sources": sources,
        "fitting_sources": runner.source_hashes(),
        "environment": benchmark.environment_identity(),
        "source_freeze_sha256": benchmark.sha256(freeze_path),
        "source_freeze": benchmark.receipt(freeze_path),
        "prepared_manifest": benchmark.receipt(prepared / "manifest.json"),
        "geometry": geometry,
        "claim": "Known-law synthetic diagnosis; no full-search or held-out promotion claim",
        "observed_labels_read": False,
    }


def native_prediction(fit: runner.Fit, panel: BenchPanel, rows: np.ndarray) -> np.ndarray:
    head = benchmark.mariner.Head(fit.head.intercept, fit.head.aggregate_coefficients, fit.head.floor)
    return head.predict(benchmark.mariner.design_matrix(panel.epochs[rows], fit.shape))


def mapping_checks(fit: runner.Fit, panel: BenchPanel) -> dict[str, float]:
    """Compare the new runner with the authoritative native head on tied rows."""
    rows = np.flatnonzero(panel.tied)
    all_rows = np.arange(len(panel.rows))
    actual = fit.predict(panel, rows)
    single = np.concatenate([fit.predict(panel, np.array([row])) for row in rows])
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
    tied = synthetic.previous_audit.tied_queries(panel)
    return {
        "max_bpb_error": float(np.max(np.abs(actual - native_prediction(fit, panel, rows)))),
        "rowwise_max_bpb_error": float(np.max(np.abs(actual - single))),
        "foreign_rows_removed_max_bpb_error": float(np.max(np.abs(actual - fit.predict(subset, np.arange(len(rows)))))),
        "full_batch_max_bpb_error": float(np.max(np.abs(actual - fit.predict(panel, all_rows)[rows]))),
        "counterfactual_tied_max_bpb_error": float(
            np.max(np.abs(fit.predict(tied, all_rows) - native_prediction(fit, tied, all_rows)))
        ),
    }


def load_fit(record: dict[str, Any]) -> runner.Fit:
    head = runner.log_head.Head(
        record["intercept"],
        np.asarray(record["aggregate_coefficients"]),
        np.asarray(record["phase_coefficients"]),
        record["floor"],
    )
    return runner.Fit(
        record["model"],
        record["shape"],
        record["q"],
        record["kappa"],
        record["ridge"],
        record["gamma"],
        record["flat_profile"],
        0.0,
        head,
        record["floor_search"],
    )


def generic_mapping(base: BenchPanel, current: dict[str, Any]) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for model in runner.CANDIDATES:
        for q, kappa in runner.MODELS[model].states:
            for shape in (
                {"rate": 0.05, "power": 0.3, "threshold": 1.0},
                synthetic.SHAPE,
                {"rate": 4.0, "power": 1.0, "threshold": 6.0},
            ):
                for floor in (-3.0, 0.7):
                    head = runner.log_head.Head(-0.2, np.array([0.8, 0.3, 0.1, 0.04]), np.empty(0), floor)
                    fit = runner.Fit(model, dict(shape), q, kappa, 0.0, 1.5, False, 0.0, head, None)
                    checks = mapping_checks(fit, base)
                    records.append({"model": model, "q": q, "kappa": kappa, "shape": shape, "floor": floor, **checks})
    statuses = {
        model: all(
            value <= MAPPING_LIMIT
            for row in records
            if row["model"] == model
            for key, value in row.items()
            if key.endswith("bpb_error")
        )
        for model in runner.CANDIDATES
    }
    report = {
        **current,
        "passed": all(statuses.values()),
        "status": "pass" if all(statuses.values()) else "fail",
        "model_status": {name: "pass" if passed else "fail" for name, passed in statuses.items()},
        "numeric_limit": MAPPING_LIMIT,
        "actual_tied_rows": base.rows.loc[base.tied, "row_id"].astype(str).tolist(),
        "generic": records,
        "learned_heads": [],
        "limitation": "Generic heads only; final full-search learned heads require separate validation",
    }
    benchmark.write_json(ROOT / "generic_structural_mapping.json", report)
    return report


def fixed_law_case(base: BenchPanel, current: dict[str, Any], world: str, sampler: str, shift: float) -> dict[str, Any]:
    panel, inner = synthetic.sampler_audit.sampler_panel(base, sampler)
    if panel.tied.any():
        raise ValueError("Known-law fitting must use asymmetric endpoints only")
    response = synthetic.world_mean(panel, world) + shift
    q, kappa = 0.5, 0.5 if world == "boundary" else 0.0
    model = "NG2-015" if world == "boundary" else "NG2-014"
    folder = ROOT / "fixed_law" / world / sampler / ("positive_floor" if shift == 0 else "negative_floor")
    config: dict[str, Any] = {
        **current,
        "world": world,
        "sampler": sampler,
        "response_shift": shift,
        "true_floor": synthetic.FLOOR + shift,
        "known_shape": synthetic.SHAPE,
        "known_q": q,
        "known_kappa": kappa,
        "ridge": 0.0,
        "noise_sd": 0.0,
        "training_rows": panel.rows.row_id.astype(str).tolist(),
        "inner_folds_recorded_not_used": inner.tolist(),
    }
    completed = synthetic.previous_audit.check_completed(folder, config)
    menu = synthetic.menu_audit.raw_menu(base)
    tied_menu = synthetic.previous_audit.tied_queries(menu)
    rows = np.arange(len(menu.rows))
    if not completed:
        matrix = runner.matrices(panel, np.arange(len(response)), synthetic.SHAPE, q, kappa)
        profiled = runner.profile.fit_profile_head(matrix, response, 0.0)
        fit = runner.Fit(
            model, dict(synthetic.SHAPE), q, kappa, 0.0, profiled.gamma, False, 0.0, profiled.head, profiled.to_json()
        )
        folder.mkdir(parents=True, exist_ok=True)
        fit_record = fit.to_json()
        del fit_record["inner_rmse"]
        fit_record["training_rmse"] = float(np.sqrt(profiled.training_sse / len(response)))
        fit_record["search_scope"] = "Known generating shape/state and ridge zero; no inner validation"
        benchmark.write_json(folder / "fit.json", fit_record)
        benchmark.write_json(folder / "config.json", config)
        pd.DataFrame(profiled.to_json()["trace"]).to_csv(folder / "profile_trace.csv", index=False)
        frame = synthetic.menu_audit.prediction_frame(
            menu,
            synthetic.world_mean(menu, world) + shift,
            fit.predict(menu, rows),
            synthetic.world_mean(tied_menu, world) + shift,
            fit.predict(tied_menu, rows),
        )
        frame.to_csv(folder / "raw_menu.csv", index=False)
        training = panel.rows[["row_id", "policy_id", "group_id", "source", "tied"]].copy()
        training["response"] = response
        training["prediction"] = fit.predict(panel, np.arange(len(response)))
        training["inner_fold"] = inner
        training.to_csv(folder / "training_predictions.csv", index=False)
    fit = load_fit(json.loads((folder / "fit.json").read_text()))
    frame = pd.read_csv(folder / "raw_menu.csv", float_precision="round_trip")
    replay = float(np.max(np.abs(frame.prediction.to_numpy() - fit.predict(menu, rows))))
    tied_replay = float(np.max(np.abs(frame.tied_prediction.to_numpy() - fit.predict(tied_menu, rows))))
    if replay > 1e-12 or tied_replay > 1e-12:
        raise ValueError("Serialized known-law head does not reproduce saved predictions")
    raw = synthetic.menu_audit.menu_metrics(frame)
    tied_rmse = float(np.sqrt(np.mean((frame.loc[frame.tied, "prediction"] - frame.loc[frame.tied, "truth"]) ** 2)))
    floor_error = abs(fit.head.floor - config["true_floor"])
    checks = {
        "matched_gain_rmse": raw["matched_gain_rmse"] <= GAIN_LIMIT,
        "uniform_101_tied_curve_rmse": tied_rmse <= TIED_LIMIT,
        "absolute_floor_error": floor_error <= FLOOR_LIMIT,
    }
    mapping = mapping_checks(fit, base)
    result = {
        "world": world,
        "sampler": sampler,
        "response_shift": shift,
        "true_floor": config["true_floor"],
        "fitted_floor": fit.head.floor,
        "fitted_gamma": fit.gamma,
        "absolute_floor_error": floor_error,
        "uniform_101_tied_curve_rmse": tied_rmse,
        "raw_menu": raw,
        "checks": checks,
        "passed": all(checks.values()),
        "mapping": mapping,
        "mapping_passed": all(value <= MAPPING_LIMIT for value in mapping.values()),
        "serialized_prediction_max_error": max(replay, tied_replay),
        "fit": benchmark.receipt(folder / "fit.json"),
    }
    benchmark.write_json(folder / "metrics.json", result)
    names = ("config.json", "fit.json", "metrics.json", "profile_trace.csv", "raw_menu.csv", "training_predictions.csv")
    outputs = {name: benchmark.sha256(folder / name) for name in names}
    benchmark.write_json(folder / "complete.json", {"config": config, "outputs": outputs})
    return {**result, "completion": benchmark.receipt(folder / "complete.json"), "outputs": outputs}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=benchmark.DEFAULT_OUTPUT / "v3/prepared")
    args = parser.parse_args()
    current = identity(args.prepared)
    base = synthetic.previous_audit.geometry_panel(args.prepared, "starcoder_wsd80_dense")
    if int(base.tied.sum()) != 20:
        raise ValueError("Expected the actual 20 tied StarCoder query rows")
    with threadpool_limits(limits=1):
        generic = generic_mapping(base, current)
        results = []
        for world, sampler, shift in itertools.product(WORLDS, synthetic.SAMPLERS, SHIFTS):
            result = fixed_law_case(base, current, world, sampler, shift)
            results.append(result)
            print(json.dumps({key: result[key] for key in ("world", "sampler", "response_shift", "passed")}), flush=True)
    if identity(args.prepared) != current:
        raise ValueError("Source, environment or geometry identity changed during the preflight")
    passed = len(results) == 8 and generic["passed"] and all(row["passed"] and row["mapping_passed"] for row in results)
    outputs = {"generic_structural_mapping.json": benchmark.sha256(ROOT / "generic_structural_mapping.json")}
    for row in results:
        completion = Path(row["completion"]["path"])
        outputs[str(completion.relative_to(ROOT))] = row["completion"]["sha256"]
        outputs.update(
            {str((completion.parent / name).relative_to(ROOT)): digest for name, digest in row["outputs"].items()}
        )
    report = {
        **current,
        "passed": passed,
        "status": "pass" if passed else "fail",
        "fits": len(results),
        "limits": {"gain_rmse": GAIN_LIMIT, "tied_rmse": TIED_LIMIT, "floor_error": FLOOR_LIMIT},
        "generic_mapping": benchmark.receipt(ROOT / "generic_structural_mapping.json"),
        "cases": results,
        "outputs": outputs,
        "command": f"uv run --no-sync python {Path(__file__).relative_to(Path.cwd())}",
    }
    benchmark.write_json(ROOT / "fixed_law_preflight.json", report)
    print(json.dumps({"passed": passed, "fits": len(results)}), flush=True)


if __name__ == "__main__":
    main()
