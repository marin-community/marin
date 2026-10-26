# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "scikit-learn", "threadpoolctl"]
# ///
"""Run the registered floor-estimator controls and sequential synthetic gates.

Use ``uv run --no-sync python`` in the frozen repository environment. Controls
must finish and their empirical budgets must be frozen before candidate fits.
Each candidate stops at its first failed cell; a pass requires all 32 contexts.
The negative-floor contexts are translations, not independent replications.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import audit_nextgen_two_phase_20260921 as previous_audit
import audit_round3_sampler_recovery_20260921 as sampler_audit
import audit_round4_sampler_recovery_20260921_v2 as round4_audit
import audit_two_phase_round2_20260921 as menu_audit
import benchmark_two_phase_20260921 as benchmark
import nextgen_two_phase_round5_20260921 as surrogate
import numpy as np
import pandas as pd
import round4_boundary_benefit_20260921 as boundary
from threadpoolctl import threadpool_limits
from two_phase_benchmark_data_20260921 import BenchPanel

ROOT = surrogate.ROOT
CANDIDATES = ("NG2-014", "NG2-015")
CONTROLS = {"NG2-014": "CTRL-LOG-AGG", "NG2-015": "NG2-012"}
NEGATIVE_SHIFT = -0.51
REPLAY_TOLERANCE = 1e-12
TRANSLATION_TOLERANCE = 1e-6
BUDGET_METRICS = (
    "matched_gain_rmse",
    "raw_policy_regret",
    "absolute_selected_optimism",
    "uniform_101_tied_curve_rmse",
)


@dataclasses.dataclass(frozen=True)
class Context:
    world: str
    sampler: str
    noise_sd: float
    seed: int
    floor_shift: float

    @property
    def floor_name(self) -> str:
        return "positive_floor" if self.floor_shift == 0 else "negative_floor"

    @property
    def noise_name(self) -> str:
        return "noiseless" if self.noise_sd == 0 else "noise002"

    @property
    def stage(self) -> str:
        if self.world in {"null", "boundary"} and self.noise_sd == 0:
            return "A"
        if self.world == "null" and self.floor_shift == 0:
            return "A"
        return "B"

    def folder(self, output: Path, model: str) -> Path:
        return output / model / self.floor_name / self.world / self.noise_name / str(self.seed) / self.sampler


def contexts() -> tuple[Context, ...]:
    positive = [
        Context(world, sampler, noise_sd, seed, 0.0)
        for world in round4_audit.WORLDS
        for noise_sd in round4_audit.NOISE_LEVELS
        for seed in (sampler_audit.SEEDS[:1] if noise_sd == 0 else sampler_audit.SEEDS)
        for sampler in sampler_audit.SAMPLERS
    ]
    negative = [
        Context(world, sampler, noise_sd, sampler_audit.SEEDS[0], NEGATIVE_SHIFT)
        for world in ("null", "boundary")
        for noise_sd in round4_audit.NOISE_LEVELS
        for sampler in sampler_audit.SAMPLERS
    ]
    result = tuple(
        sorted(
            positive + negative,
            key=lambda cell: (
                cell.stage,
                cell.noise_sd,
                cell.floor_shift != 0,
                round4_audit.WORLDS.index(cell.world),
                cell.seed,
                cell.sampler,
            ),
        )
    )
    if len(result) != 32 or sum(cell.stage == "A" for cell in result) != 14:
        raise ValueError("The registered 14 plus 18 context matrix changed")
    return result


def identity(prepared: Path) -> dict[str, Any]:
    freeze_path = ROOT / "source_freeze.json"
    freeze = json.loads(freeze_path.read_text())
    if freeze["environment"] != benchmark.environment_identity():
        raise ValueError("Run uv run --no-sync python in the frozen repository environment")
    benchmark.verify_hashes(freeze["sources"])
    benchmark.verify_hashes(freeze["driver_sources"])
    modules = (previous_audit, sampler_audit, round4_audit, menu_audit, boundary)
    sources = {
        **surrogate.source_hashes(),
        **{str(Path(module.__file__).resolve()): benchmark.sha256(Path(module.__file__)) for module in modules},
        str(Path(__file__).resolve()): benchmark.sha256(Path(__file__)),
    }
    frozen_sources = freeze["sources"] | freeze["driver_sources"]
    if any(frozen_sources.get(name) != digest for name, digest in sources.items()):
        raise ValueError("The frozen sources do not cover every synthetic driver dependency")
    registration = json.loads((ROOT / "registration.json").read_text())
    if {item["id"]: item["control"] for item in registration["models"]} != CONTROLS:
        raise ValueError("The registered floor candidates or same-loss controls changed")
    if set(surrogate.MODELS) != set(CANDIDATES) | set(CONTROLS.values()):
        raise ValueError("The runner has a different registered model matrix")
    manifest = json.loads((prepared / "manifest.json").read_text())
    names = [f"starcoder_wsd80_dense/{name}" for name in ("rows.csv", "policies.npz", "metadata.json")]
    geometry = {name: manifest["outputs"][name] for name in names}
    benchmark.verify_hashes(geometry, prepared)
    return {
        "sources": sources,
        "source_freeze": benchmark.receipt(freeze_path),
        "registration": benchmark.receipt(ROOT / "registration.json"),
        "environment": benchmark.environment_identity(),
        "prepared_manifest": benchmark.sha256(prepared / "manifest.json"),
        "geometry": geometry,
        "contexts": [dataclasses.asdict(cell) for cell in contexts()],
        "claim": "adaptive_synthetic_rejection_screen_not_statistical_certification",
    }


def world_mean(panel: BenchPanel, cell: Context) -> np.ndarray:
    return round4_audit.world_mean(panel, cell.world) + cell.floor_shift


def floor_compatibility(response: np.ndarray, inner: np.ndarray, shift: float) -> list[dict[str, Any]]:
    """Evaluate the native envelope on each actual subset without changing truth."""
    true_floor = sampler_audit.FLOOR + shift
    partitions = [("outer_train", np.arange(len(response)))] + [
        (f"inner_train_{fold}", np.flatnonzero(inner != fold)) for fold in sorted(set(inner))
    ]
    records = []
    for name, indices in partitions:
        values = response[indices]
        lower = benchmark.mariner.floor_value(values, benchmark.mariner.FloorSpec(float("nan"), 0.0, 6.0))
        upper = benchmark.mariner.floor_value(values, benchmark.mariner.FloorSpec(float("nan"), 0.0, 1.0))
        median, minimum = float(np.median(values)), float(np.min(values))
        records.append(
            {
                "partition": name,
                "rows": len(indices),
                "true_floor": true_floor,
                "minimum": minimum,
                "median": median,
                "required_native_gamma": (median - true_floor) / (median - minimum),
                "native_envelope": [float(lower), float(upper)],
                "native_floor_compatible": bool(lower <= true_floor <= upper and np.all(values > true_floor)),
            }
        )
    return records


def preparation(prepared: Path) -> dict[str, Any]:
    base = previous_audit.geometry_panel(prepared, "starcoder_wsd80_dense")
    panels = {sampler: sampler_audit.sampler_panel(base, sampler) for sampler in sampler_audit.SAMPLERS}
    drift = float(np.max(np.abs(panels["balanced"][0].aggregate - panels["positive"][0].aggregate)))
    if drift > 3e-16:
        raise ValueError("Samplers no longer share an aggregate population")
    records = []
    for cell in contexts():
        panel, inner = panels[cell.sampler]
        if panel.tied.any() or set(inner) != {0, 1, 2} or len(panel.rows) != 200:
            raise ValueError("Synthetic fitting must use 200 asymmetric rows and three actual grouped folds")
        for fold in sorted(set(inner)):
            if set(panel.rows.loc[inner == fold, "group_id"]) & set(panel.rows.loc[inner != fold, "group_id"]):
                raise ValueError("An aggregate group crosses an inner split")
        response = world_mean(panel, cell) + np.random.default_rng(cell.seed).normal(0, cell.noise_sd, len(panel.rows))
        if not np.all(response > 0):
            raise ValueError("Registered negative-floor translation no longer has positive observations")
        compatibility = floor_compatibility(response, inner, cell.floor_shift)
        records.append(dataclasses.asdict(cell) | {"folds": compatibility})
    return {
        "maximum_sampler_aggregate_drift": drift,
        "all_training_rows_asymmetric": True,
        "geometry_sha256": {
            sampler: hashlib.sha256(np.hstack([panel.phase0, panel.phase1]).tobytes()).hexdigest()
            for sampler, (panel, _) in panels.items()
        },
        "floor_compatibility": records,
        "incompatible_contexts": [
            record for record in records if not all(fold["native_floor_compatible"] for fold in record["folds"])
        ],
        "negative_floor_claim": "Translation of existing worlds; not independent evidence",
    }


def replay_predictions(panel: BenchPanel, fitted: dict[str, Any]) -> np.ndarray:
    if fitted["phase_coefficients"] or (fitted["model"] in {"NG2-014", "CTRL-LOG-AGG"} and fitted["kappa"] != 0):
        raise ValueError("Unexpected independent phase coefficients or aggregate-arm boundary state")
    head = benchmark.mariner.Head(
        intercept=fitted["intercept"],
        coefficients=np.asarray(fitted["aggregate_coefficients"]),
        floor=fitted["floor"],
    )
    return head.predict(boundary.design_matrix(panel, fitted["shape"], fitted["q"], fitted["kappa"]))


def replay_frame(panel: BenchPanel, frame: pd.DataFrame, fitted: dict[str, Any], cell: Context) -> float:
    tied = previous_audit.tied_queries(panel)
    reconstructed = menu_audit.prediction_frame(
        panel,
        world_mean(panel, cell),
        replay_predictions(panel, fitted),
        world_mean(tied, cell),
        replay_predictions(tied, fitted),
    )
    errors = []
    for column in reconstructed:
        if column in {"prediction", "tied_prediction"}:
            error = float(np.max(np.abs(frame[column].to_numpy() - reconstructed[column].to_numpy())))
            if not np.isfinite(error) or error > REPLAY_TOLERANCE:
                raise ValueError(f"Independent serialized prediction replay failed: {column} {error}")
            errors.append(error)
        elif not np.array_equal(frame[column].to_numpy(), reconstructed[column].to_numpy()):
            raise ValueError(f"Saved geometry or truth differs: {column}")
    return max(errors)


def config_for(prepared: Path, cell: Context, model: str) -> dict[str, Any]:
    config = identity(prepared) | {"context": dataclasses.asdict(cell), "model": model}
    if model in CANDIDATES:
        config["control_budgets"] = benchmark.receipt(ROOT / "control_budgets.json")
        config["fixed_law_preflight"] = benchmark.receipt(ROOT / "fixed_law_preflight.json")
    return config


def raw_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    result = menu_audit.menu_metrics(frame)
    result["absolute_selected_optimism"] = abs(result["selected_optimism"])
    result["uniform_101_tied_curve_rmse"] = float(
        np.sqrt(np.mean((frame.loc[frame.tied, "prediction"] - frame.loc[frame.tied, "truth"]) ** 2))
    )
    return result


def audit_fit(arguments: tuple[str, str, Context, str]) -> dict[str, Any]:
    prepared_name, output_name, cell, model = arguments
    prepared, output = Path(prepared_name), Path(output_name)
    folder = cell.folder(output, model)
    config = config_for(prepared, cell, model)
    if previous_audit.check_completed(folder, config):
        return {"folder": str(folder), "cached": True, "model": model, "context": dataclasses.asdict(cell)}
    config_path = folder / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("An unfinished cell has different source or configuration identity")
    benchmark.write_json(config_path, config)
    print(json.dumps({"started": str(folder), "model": model, "context": dataclasses.asdict(cell)}), flush=True)
    base = previous_audit.geometry_panel(prepared, "starcoder_wsd80_dense")
    panel, inner = sampler_audit.sampler_panel(base, cell.sampler)
    train = np.arange(len(panel.rows))
    if panel.tied.any():
        raise ValueError("Tied synthetic labels cannot enter fitting")
    truth = world_mean(panel, cell)
    response = truth + np.random.default_rng(cell.seed).normal(0, cell.noise_sd, len(truth))
    tied, menu = previous_audit.tied_queries(panel), menu_audit.raw_menu(base)
    tied_menu = previous_audit.tied_queries(menu)
    started = time.monotonic()
    origin = None
    if model == "NG2-012" and cell.floor_shift == 0:
        old_folder = (
            round4_audit.ROOT / "synthetic_v2" / cell.world / cell.noise_name / str(cell.seed) / model / cell.sampler
        )
        old_config = round4_audit.identity(prepared) | {
            "world": cell.world,
            "sampler": cell.sampler,
            "seed": cell.seed,
            "noise_sd": cell.noise_sd,
            "model": model,
        }
        if not previous_audit.check_completed(old_folder, old_config):
            raise ValueError(f"Missing registered reusable boundary-control receipt: {old_folder}")
        saved_fit = json.loads((old_folder / "fit.json").read_text())
        train_frame = pd.read_csv(old_folder / "training_predictions.csv", float_precision="round_trip")
        raw_frame = pd.read_csv(old_folder / "raw_menu.csv", float_precision="round_trip")
        origin = benchmark.receipt(old_folder / "complete.json")
    else:
        with threadpool_limits(limits=1):
            fitted = surrogate.fit_model(panel, response, train, inner, model)
            train_frame = menu_audit.prediction_frame(
                panel, truth, fitted.predict(panel, train), world_mean(tied, cell), fitted.predict(tied, train)
            )
            train_frame["response"], train_frame["inner_fold"] = response, inner
            queries = np.arange(len(menu.rows))
            raw_frame = menu_audit.prediction_frame(
                menu,
                world_mean(menu, cell),
                fitted.predict(menu, queries),
                world_mean(tied_menu, cell),
                fitted.predict(tied_menu, queries),
            )
        saved_fit = fitted.to_json() | {"training_rows": panel.rows.row_id.tolist(), "inner_folds": inner.tolist()}
    if not np.array_equal(train_frame.response.to_numpy(), response) or not np.array_equal(
        train_frame.inner_fold, inner
    ):
        raise ValueError("Saved fitting responses or actual grouped folds differ")
    if saved_fit["training_rows"] != panel.rows.row_id.tolist() or saved_fit["inner_folds"] != inner.tolist():
        raise ValueError("Saved asymmetric training-row identities differ")
    replay_error = max(replay_frame(panel, train_frame, saved_fit, cell), replay_frame(menu, raw_frame, saved_fit, cell))
    metrics = {
        **raw_metrics(raw_frame),
        "training_endpoint_rmse": float(np.sqrt(np.mean((train_frame.prediction - truth) ** 2))),
        "floor_compatibility": floor_compatibility(response, inner, cell.floor_shift),
        "maximum_serialized_prediction_replay_error": replay_error,
        "elapsed_seconds": time.monotonic() - started,
        "reused_round4_receipt": origin,
    }
    benchmark.write_json(folder / "fit.json", saved_fit)
    benchmark.write_json(folder / "config.json", config)
    benchmark.write_json(folder / "metrics.json", metrics)
    train_frame.to_csv(folder / "training_predictions.csv", index=False)
    raw_frame.to_csv(folder / "raw_menu.csv", index=False)
    files = ("fit.json", "config.json", "metrics.json", "training_predictions.csv", "raw_menu.csv")
    benchmark.write_json(
        folder / "complete.json",
        {
            "config": config,
            "outputs": {name: benchmark.sha256(folder / name) for name in files},
        },
    )
    return {"folder": str(folder), "cached": False, "model": model, "context": dataclasses.asdict(cell), **metrics}


def replay_cell(prepared: Path, output: Path, cell: Context, model: str) -> dict[str, Any]:
    folder = cell.folder(output, model)
    if not previous_audit.check_completed(folder, config_for(prepared, cell, model)):
        raise ValueError(f"Missing complete synthetic cell: {folder}")
    fitted = json.loads((folder / "fit.json").read_text())
    metrics = json.loads((folder / "metrics.json").read_text())
    if fitted["model"] != model:
        raise ValueError("Serialized model identity differs")
    origin = metrics["reused_round4_receipt"]
    if origin is not None:
        if not benchmark.valid_receipts([origin]):
            raise ValueError("Reused round-four complete receipt changed")
        old_folder = Path(origin["path"]).parent
        benchmark.verify_hashes(json.loads(Path(origin["path"]).read_text())["outputs"], old_folder)
    base = previous_audit.geometry_panel(prepared, "starcoder_wsd80_dense")
    panel, inner = sampler_audit.sampler_panel(base, cell.sampler)
    menu = menu_audit.raw_menu(base)
    frame = pd.read_csv(folder / "raw_menu.csv", float_precision="round_trip")
    training = pd.read_csv(folder / "training_predictions.csv", float_precision="round_trip")
    response = world_mean(panel, cell) + np.random.default_rng(cell.seed).normal(0, cell.noise_sd, len(panel.rows))
    if not np.array_equal(training.response.to_numpy(), response) or not np.array_equal(training.inner_fold, inner):
        raise ValueError("Training labels or folds differ from the registered world")
    if fitted["training_rows"] != panel.rows.row_id.tolist() or fitted["inner_folds"] != inner.tolist():
        raise ValueError("Training-row identities differ from the asymmetric sampler")
    replay_error = max(replay_frame(panel, training, fitted, cell), replay_frame(menu, frame, fitted, cell))
    recomputed = raw_metrics(frame)
    recomputed["training_endpoint_rmse"] = float(np.sqrt(np.mean((training.prediction - training.truth) ** 2)))
    recomputed["floor_compatibility"] = floor_compatibility(response, inner, cell.floor_shift)
    if any(metrics[key] != value for key, value in recomputed.items()):
        raise ValueError("Saved metric differs from independently replayed predictions")
    return dataclasses.asdict(cell) | {
        "model": model,
        "stage": cell.stage,
        "floor": fitted["floor"],
        "gamma": fitted["gamma"],
        "ridge": fitted["ridge"],
        "maximum_serialized_prediction_replay_error": replay_error,
        **{key: value for key, value in metrics.items() if key != "floor_compatibility"},
    }


def translation_audit(output: Path, model: str, cell: Context) -> dict[str, Any] | None:
    if cell.floor_shift == 0:
        return None
    positive = dataclasses.replace(cell, floor_shift=0.0)
    first = pd.read_csv(positive.folder(output, model) / "raw_menu.csv", float_precision="round_trip")
    shifted = pd.read_csv(cell.folder(output, model) / "raw_menu.csv", float_precision="round_trip")
    errors = {
        column: float(np.max(np.abs(shifted[column].to_numpy() - first[column].to_numpy() - NEGATIVE_SHIFT)))
        for column in ("prediction", "tied_prediction")
    }
    positive_fit = json.loads((positive.folder(output, model) / "fit.json").read_text())
    negative_fit = json.loads((cell.folder(output, model) / "fit.json").read_text())
    fields = ("shape", "q", "kappa", "ridge", "gamma", "floor", "intercept", "aggregate_coefficients")
    return {
        "context": dataclasses.asdict(cell),
        "maximum_prediction_shift_error": errors["prediction"],
        "maximum_tied_prediction_shift_error": errors["tied_prediction"],
        "passed": errors["prediction"] <= TRANSLATION_TOLERANCE,
        "positive_parameters": {key: positive_fit[key] for key in fields},
        "negative_parameters": {key: negative_fit[key] for key in fields},
    }


def freeze_budgets(prepared: Path, output: Path) -> None:
    for model in CANDIDATES:
        if list((output / model).glob("**/complete.json")):
            raise ValueError("Budgets must be frozen before candidate outcomes exist")
    rows = [replay_cell(prepared, output, cell, model) for model in CONTROLS.values() for cell in contexts()]
    cohorts = []
    for model in CONTROLS.values():
        for world in round4_audit.WORLDS:
            for noise_sd in round4_audit.NOISE_LEVELS:
                cohort = [
                    row
                    for row in rows
                    if row["model"] == model
                    and row["world"] == world
                    and row["noise_sd"] == noise_sd
                    and row["floor_shift"] == 0
                ]
                metrics = BUDGET_METRICS + (("predicted_max_abs_matched_gain",) if world != "boundary" else ())
                maxima = {metric: max(row[metric] for row in cohort) for metric in metrics}
                limits = {metric: 1.05 * value + 1e-8 for metric, value in maxima.items()}
                if world == "null" and noise_sd == 0:
                    limits["matched_gain_rmse"] = limits["predicted_max_abs_matched_gain"] = 1e-8
                cohorts.append(
                    {
                        "control": model,
                        "world": world,
                        "noise_sd": noise_sd,
                        "control_context_count": len(cohort),
                        "control_maxima": maxima,
                        "limits": limits,
                    }
                )
    receipts = [
        benchmark.receipt(cell.folder(output, model) / "complete.json")
        for model in CONTROLS.values()
        for cell in contexts()
    ]
    record = {
        "identity": identity(prepared),
        "controls": list(CONTROLS.values()),
        "control_context_count": len(rows),
        "reused_round4_controls": sum(row["reused_round4_receipt"] is not None for row in rows),
        "formula": "1.05 * positive-floor cohort maximum + 1e-8; noiseless native-null effect <=1e-8",
        "claim": "Empirical rejection tolerances; not confidence bounds or independent replications",
        "cohorts": cohorts,
        "control_receipts": receipts,
        "translation_diagnostics": [
            translation_audit(output, model, cell)
            for model in CONTROLS.values()
            for cell in contexts()
            if cell.floor_shift != 0
        ],
    }
    path = ROOT / "control_budgets.json"
    if path.exists() and json.loads(path.read_text()) != record:
        raise ValueError("Previously frozen numerical control budgets differ")
    benchmark.write_json(path, record)
    pd.DataFrame(rows).to_csv(output / "control_summary.csv", index=False)
    print(json.dumps({"budgets": benchmark.receipt(path), "contexts": len(rows), "reused": 24}), flush=True)


def candidate_prerequisites(prepared: Path) -> dict[str, Any]:
    budgets = json.loads((ROOT / "control_budgets.json").read_text())
    if budgets["identity"] != identity(prepared) or budgets["control_context_count"] != 64:
        raise ValueError("Frozen budgets do not cover the current 64 controls")
    if not benchmark.valid_receipts(budgets["control_receipts"]):
        raise ValueError("Control receipts used to freeze budgets changed")
    for receipt in budgets["control_receipts"]:
        path = Path(receipt["path"])
        benchmark.verify_hashes(json.loads(path.read_text())["outputs"], path.parent)
    preflight = json.loads((ROOT / "fixed_law_preflight.json").read_text())
    if preflight["passed"] is not True or preflight["source_freeze_sha256"] != benchmark.sha256(
        ROOT / "source_freeze.json"
    ):
        raise ValueError("The fixed-law recovery preflight did not pass with frozen sources")
    benchmark.verify_hashes(preflight["outputs"], ROOT)
    return budgets


def budget_failures(row: dict[str, Any], budgets: dict[str, Any]) -> list[dict[str, Any]]:
    matches = [
        cohort
        for cohort in budgets["cohorts"]
        if cohort["control"] == CONTROLS[row["model"]]
        and cohort["world"] == row["world"]
        and cohort["noise_sd"] == row["noise_sd"]
    ]
    if len(matches) != 1:
        raise ValueError("No unique frozen same-loss world/noise budget")
    return [
        {"metric": metric, "observed": row[metric], "limit": limit}
        for metric, limit in matches[0]["limits"].items()
        if not np.isfinite(row[metric]) or row[metric] > limit
    ]


def candidate_run(arguments: tuple[str, str, str]) -> dict[str, Any]:
    prepared_name, output_name, model = arguments
    prepared, output = Path(prepared_name), Path(output_name)
    budgets = candidate_prerequisites(prepared)
    report_path = output / model / "decision.json"
    decisions, translations, rows = [], [], []
    for cell in contexts():
        audit_fit((prepared_name, output_name, cell, model))
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
                        "limit": TRANSLATION_TOLERANCE,
                    }
                )
        decisions.append({"context": dataclasses.asdict(cell), "stage": cell.stage, "failures": failures})
        report: dict[str, Any] = {
            "identity": identity(prepared),
            "model": model,
            "completed_contexts": len(rows),
            "status": "rejected" if failures else "incomplete",
            "decisions": decisions,
            "translations": translations,
            "control_budgets": benchmark.receipt(ROOT / "control_budgets.json"),
            "fixed_law_preflight": benchmark.receipt(ROOT / "fixed_law_preflight.json"),
        }
        benchmark.write_json(report_path, report)
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
    if model == "NG2-015":
        selected = [
            row for row in rows if row["world"] == "boundary" and row["noise_sd"] > 0 and row["floor_shift"] == 0
        ]
        candidate_mse = float(np.mean([row["matched_gain_rmse"] ** 2 for row in selected]))
        control = [
            replay_cell(prepared, output, cell, CONTROLS[model])
            for cell in contexts()
            if cell.world == "boundary" and cell.noise_sd > 0 and cell.floor_shift == 0
        ]
        control_mse = float(np.mean([row["matched_gain_rmse"] ** 2 for row in control]))
        improvement = {
            "positive_noisy_boundary_contexts": len(selected),
            "candidate_mean_squared_gain_error": candidate_mse,
            "control_mean_squared_gain_error": control_mse,
            "passed": candidate_mse < control_mse,
        }
    report["phase_improvement"] = improvement
    report["status"] = "synthetic_survivor" if improvement is None or improvement["passed"] else "rejected"
    benchmark.write_json(report_path, report)
    return report


def replay_all(prepared: Path, output: Path) -> None:
    rows, translations, sampler_comparisons = [], [], []
    for model in (*CONTROLS.values(), *CANDIDATES):
        for cell in contexts():
            if not (cell.folder(output, model) / "complete.json").exists():
                continue
            rows.append(replay_cell(prepared, output, cell, model))
            translation = translation_audit(output, model, cell)
            if translation is not None:
                translations.append({"model": model, **translation})
            other = dataclasses.replace(cell, sampler="positive")
            if cell.sampler != "balanced" or not (other.folder(output, model) / "complete.json").exists():
                continue
            balanced = pd.read_csv(cell.folder(output, model) / "raw_menu.csv", float_precision="round_trip")
            positive = pd.read_csv(other.folder(output, model) / "raw_menu.csv", float_precision="round_trip")
            for column in ("row_id", "phase0_code", "phase1_code", "truth", "tied_truth"):
                if not np.array_equal(balanced[column].to_numpy(), positive[column].to_numpy()):
                    raise ValueError("Sampler comparison has nonidentical raw geometry or truth")
            tied = balanced.tied.to_numpy(bool)
            drift = balanced.prediction.to_numpy()[tied] - positive.prediction.to_numpy()[tied]
            balanced_gain = balanced.tied_prediction.to_numpy() - balanced.prediction.to_numpy()
            positive_gain = positive.tied_prediction.to_numpy() - positive.prediction.to_numpy()
            sampler_comparisons.append(
                dataclasses.asdict(cell)
                | {
                    "model": model,
                    "tied_curve_sampler_drift_rmse": float(np.sqrt(np.mean(drift**2))),
                    "tied_curve_sampler_drift_max_abs": float(np.max(np.abs(drift))),
                    "matched_gain_sampler_drift_rmse": float(np.sqrt(np.mean((balanced_gain - positive_gain) ** 2))),
                }
            )
    if not rows:
        raise ValueError("No completed synthetic fits to replay")
    summary = pd.DataFrame(rows)
    summary.to_csv(output / "replay_summary.csv", index=False)
    pd.DataFrame(sampler_comparisons).to_csv(output / "sampler_comparison.csv", index=False)
    benchmark.write_json(
        output / "replay.json",
        {
            "identity": identity(prepared),
            "completed_contexts": len(rows),
            "counts_by_model": summary.groupby("model").size().to_dict(),
            "maximum_serialized_prediction_replay_error": float(
                summary.maximum_serialized_prediction_replay_error.max()
            ),
            "translations": translations,
            "measured_outcomes_read": False,
            "lm_jobs": 0,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=previous_audit.PREPARED)
    parser.add_argument("--output", type=Path, default=ROOT / "synthetic_v1")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument(
        "--stage", choices=("prepare", "controls", "freeze-budgets", "candidates", "replay"), required=True
    )
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("Workers must be positive")
    current = identity(args.prepared)
    prepared_record = {"identity": current, **preparation(args.prepared)}
    preparation_path = args.output / "preparation.json"
    if preparation_path.exists() and json.loads(preparation_path.read_text()) != prepared_record:
        raise ValueError("Existing synthetic preparation changed; use a new output directory")
    benchmark.write_json(preparation_path, prepared_record)
    if args.stage == "prepare":
        print(
            json.dumps(
                {"contexts_per_estimator": 32, "incompatible_contexts": len(prepared_record["incompatible_contexts"])}
            )
        )
    elif args.stage == "controls":
        tasks = [
            (str(args.prepared), str(args.output), cell, model) for model in CONTROLS.values() for cell in contexts()
        ]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for future in as_completed([pool.submit(audit_fit, task) for task in tasks]):
                print(json.dumps(future.result()), flush=True)
        replay_all(args.prepared, args.output)
    elif args.stage == "freeze-budgets":
        freeze_budgets(args.prepared, args.output)
    elif args.stage == "candidates":
        candidate_prerequisites(args.prepared)
        with ProcessPoolExecutor(max_workers=min(args.workers, len(CANDIDATES))) as pool:
            tasks = [(str(args.prepared), str(args.output), model) for model in CANDIDATES]
            for future in as_completed([pool.submit(candidate_run, task) for task in tasks]):
                record = future.result()
                print(
                    json.dumps(
                        {
                            "model": record["model"],
                            "status": record["status"],
                            "completed_contexts": record["completed_contexts"],
                        }
                    ),
                    flush=True,
                )
        replay_all(args.prepared, args.output)
    else:
        replay_all(args.prepared, args.output)


if __name__ == "__main__":
    main()
