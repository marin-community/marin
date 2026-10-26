# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "scipy", "pandas", "joblib", "scikit-learn"]
# ///
"""Screen raw-loss harm against matched inside-exponential controls at frozen geometry.

Each geometry comes from the matching MARINER outer-training fit in the fixed
basis screen. This is a conditional comparison, not a fresh search of geometry
or an external proposal-bank evaluation. Ridge selection uses each candidate's
own raw-BPB predictions. Both placements fit raw MSE with the same RMS-column
penalty, but their harm amplitudes have different units, so the regularization
does not imply an identical function-space prior.
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import hashlib
import importlib
import importlib.metadata
import io
import json
import os
import shutil
import time
import traceback
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    benchmark_single_phase_observatory_20260902 as bench,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    mariner_harm_screen_raw_loss_20260917 as raw,
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
MODULE_NAME = "experiments.domain_phase_mix.exploratory.two_phase_many.screen_mariner_raw_harm_20260917"
GEOMETRY_MODEL = "hs17_mariner"
MANIFEST_NAME = "raw_screen_manifest.json"


@dataclasses.dataclass(frozen=True)
class ProbeSpec:
    harm: raw.RawHarm
    placement: raw.HarmPlacement

    @property
    def model_id(self) -> str:
        return raw.RawLossProbe(self.harm, base.HeadSpec(), (), placement=self.placement).model_id


@dataclasses.dataclass(frozen=True)
class FrozenGeometry:
    path: str
    checksum: str
    source_protocol_hash: str
    shape: dict[str, float]


def candidate_specs() -> tuple[ProbeSpec, ...]:
    return tuple(ProbeSpec(harm, placement) for harm in raw.RawHarm for placement in raw.HarmPlacement)


def source_paths() -> tuple[Path, ...]:
    sources = (
        Path(__file__).resolve(),
        Path(raw.__file__).resolve(),
        Path(bench.olmix_benchmark.__file__).resolve(),
        Path(bench.starcoder_curves.__file__).resolve(),
        *fixed.source_paths(),
    )
    return tuple(dict.fromkeys(sources))


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _digest_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=_json_default).encode()).hexdigest()


def protocol(fixed_output: Path, max_nfev: int) -> dict[str, Any]:
    fixed_manifest_path = fixed_output.resolve() / "screen_manifest.json"
    fixed_manifest = json.loads(fixed_manifest_path.read_text())
    panels = tuple(bench.load_panel(name) for name in bench.THIRTY_NINE_BUCKET_PANELS)
    split_hash = bench.manifest_hash(bench.split_manifest(panels, 1))
    if split_hash != fixed_manifest["split_hash"]:
        raise ValueError("The current outer/inner splits differ from the geometry screen")
    for source, checksum in fixed_manifest["sources"].items():
        if fixed.digest(Path(source)) != checksum:
            raise ValueError(f"Geometry screen source changed: {source}")
    inputs = {panel.name: panel.input_hashes for panel in panels}
    if inputs != fixed_manifest["panel_inputs"]:
        raise ValueError("Current panel inputs differ from the geometry screen")
    result = {
        "version": 1,
        "sources": {str(path.resolve()): fixed.digest(path) for path in source_paths()},
        "dependencies": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "pandas", "joblib")},
        "fixed_output": str(fixed_output.resolve()),
        "fixed_manifest_sha256": fixed.digest(fixed_manifest_path),
        "split_hash": split_hash,
        "panel_inputs": inputs,
        "geometry_model": GEOMETRY_MODEL,
        "geometry_scope": "matching outer-training MARINER shard; geometry not refit for either placement",
        "floor_multiplier": raw.FLOOR_MULTIPLIER,
        "prediction_cap": None,
        "ridge_grid": registry.SUCCESSOR_RIDGE_GRID,
        "objective": "mean raw-BPB squared error plus ridge times squared RMS-basis amplitudes",
        "regularization_caveat": (
            "Same basis normalization and ridge grid; outside-exp beta is in BPB, inside-exp beta in log deficit"
        ),
        "canonical_ridge_difference": (
            "These mean raw-MSE RMS-amplitude penalties differ from canonical sum log-MSE unscaled penalties"
        ),
        "starts_per_fit": raw.START_COUNT,
        "max_nfev": max_nfev,
        "selection_evidence": "canonical out-of-fold component metrics; no external bank used",
        "candidates": [{"model_id": spec.model_id, **dataclasses.asdict(spec)} for spec in candidate_specs()],
    }
    # JSON round-trip makes resumption independent of tuple/list representation.
    return json.loads(json.dumps(result, sort_keys=True))


def prepare(output: Path, fixed_output: Path, max_nfev: int) -> dict[str, Any]:
    current = protocol(fixed_output, max_nfev)
    output.mkdir(parents=True, exist_ok=True)
    path = output / MANIFEST_NAME
    if path.exists():
        previous = json.loads(path.read_text())
        if current != previous:
            raise ValueError("Raw screen source, inputs, geometry identity or solver settings changed; use a new output")
        return previous
    snapshot = output / "source_snapshot"
    snapshot.mkdir(exist_ok=True)
    for source in source_paths():
        shutil.copyfile(source, snapshot / source.name)
    shutil.copyfile(fixed_output / "screen_manifest.json", snapshot / "fixed_screen_manifest.json")
    path.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n")
    return current


@functools.cache
def initialize_worker(output: Path) -> dict[str, Any]:
    manifest = json.loads((output / MANIFEST_NAME).read_text())
    for source, checksum in manifest["sources"].items():
        if fixed.digest(Path(source)) != checksum:
            raise ValueError(f"Raw screen source changed: {source}")
    fixed_manifest = Path(manifest["fixed_output"]) / "screen_manifest.json"
    if fixed.digest(fixed_manifest) != manifest["fixed_manifest_sha256"]:
        raise ValueError("The geometry screen manifest changed")
    for name, version in manifest["dependencies"].items():
        if importlib.metadata.version(name) != version:
            raise ValueError(f"Raw screen dependency changed: {name}")
    fixed.register_candidates()
    return manifest


def worker_identity(output: Path) -> dict[str, Any]:
    manifest = initialize_worker(output)
    return {"pid": os.getpid(), "protocol": _digest_json(manifest), "candidate_count": len(candidate_specs())}


def frozen_geometry(task: bench.FitTask, manifest: dict[str, Any]) -> FrozenGeometry:
    source_task = dataclasses.replace(task, model_id=GEOMETRY_MODEL)
    output = Path(manifest["fixed_output"])
    path = bench.shard_path(output, source_task)
    data = path.read_bytes()
    with np.load(io.BytesIO(data), allow_pickle=False) as archive:
        saved = {key: archive[key] for key in archive.files}
    panel = bench.load_panel(task.panel)
    split = next(
        s for s in bench.panel_splits(panel, task.repeat + 1) if s.repeat == task.repeat and s.fold == task.fold
    )
    expected_hashes = bench.task_protocol_hashes(
        source_task, registry.ENTRY_BY_ID[GEOMETRY_MODEL], manifest["split_hash"], panel, output
    )
    if str(saved["status"].item()) != "ok" or str(saved["protocol_hash"].item()) not in expected_hashes:
        raise ValueError(f"Geometry shard is failed or stale: {path}")
    if str(saved["component"].item()) != task.component or not np.array_equal(saved["test"], split.test):
        raise ValueError(f"Geometry shard component or test rows disagree: {path}")
    observed = panel.group(task.target).outcomes[split.test, task.component_index]
    if not np.array_equal(saved["observed"], observed) or not np.isfinite(saved["prediction"]).all():
        raise ValueError(f"Geometry shard observations or predictions disagree: {path}")
    shape = json.loads(str(saved["shape_json"].item()))
    if set(shape) != {"rate", "power", "threshold"}:
        raise ValueError(f"Expected shared MARINER geometry, got {shape}")
    return FrozenGeometry(
        str(path.resolve()), hashlib.sha256(data).hexdigest(), str(saved["protocol_hash"].item()), shape
    )


def task_protocol_hash(task: bench.FitTask, geometry: FrozenGeometry, manifest: dict[str, Any]) -> str:
    return _digest_json(
        {"manifest": _digest_json(manifest), "task": dataclasses.asdict(task), "geometry": dataclasses.asdict(geometry)}
    )


def probe(spec: ProbeSpec, features: base.Features, max_nfev: int) -> raw.RawLossProbe:
    canonical = registry.ENTRY_BY_ID[registry.FROZEN_ID].build(features)
    if not isinstance(canonical, base.FittedFloorModel) or not isinstance(canonical.base, base.GridModel):
        raise TypeError("Expected canonical fitted-floor grid model")
    head_spec = dataclasses.replace(
        canonical.base.head,
        floor_kappa_prior=raw.FLOOR_MULTIPLIER,
        floor_kappa_bounds=(raw.FLOOR_MULTIPLIER, raw.FLOOR_MULTIPLIER),
        cap_margin=float("inf"),
    )
    return raw.RawLossProbe(
        spec.harm, head_spec, registry.SUCCESSOR_RIDGE_GRID, placement=spec.placement, max_nfev=max_nfev
    )


def fit_task(task: bench.FitTask, output: Path) -> str:
    manifest = initialize_worker(output)
    geometry = frozen_geometry(task, manifest)
    expected_hash = task_protocol_hash(task, geometry, manifest)
    path = bench.shard_path(output, task)
    saved = bench.load_shard(path)
    if saved is not None:
        if str(saved["protocol_hash"].item()) != expected_hash:
            raise ValueError(f"Raw shard's geometry or task identity changed: {path}")
        if str(saved["status"].item()) == "ok" and np.isfinite(saved["prediction"]).all():
            return "cached"
        path.rename(path.with_suffix(f".failed-{time.time_ns()}.npz"))
    panel = bench.load_panel(task.panel)
    split = next(
        s for s in bench.panel_splits(panel, task.repeat + 1) if s.repeat == task.repeat and s.fold == task.fold
    )
    response = panel.group(task.target).outcomes[:, task.component_index]
    features = dataclasses.replace(panel.features, component=task.component)
    spec = next(spec for spec in candidate_specs() if spec.model_id == task.model_id)
    model = probe(spec, features, int(manifest["max_nfev"]))
    seed = int(
        hashlib.sha256(
            f"{task.panel}|{task.target}|{task.component_index}|{task.repeat}|{task.fold}".encode()
        ).hexdigest()[:8],
        16,
    )
    payload = {
        "protocol_hash": expected_hash,
        "model_id": task.model_id,
        "component": task.component,
        "component_index": task.component_index,
        "test": split.test,
        "train": split.train,
        "train_rows": len(split.train),
        "observed": response[split.test],
        "shape_json": json.dumps(geometry.shape, sort_keys=True),
        "geometry_json": json.dumps(dataclasses.asdict(geometry), sort_keys=True),
        "spec_json": json.dumps(dataclasses.asdict(spec), sort_keys=True),
        "ridge": float("nan"),
        "cv_table": np.empty((0, 0)),
        "head_json": "{}",
        "diagnostics_json": "{}",
    }
    started = time.monotonic()
    try:
        fitted = model.fit(features, response, split.train, split.inner, geometry.shape, seed)
        # Preserve the converged fit if an uncapped outer prediction fails.
        payload.update(
            {
                "ridge": fitted.ridge,
                "cv_table": fitted.cv_table,
                "head_json": json.dumps(dataclasses.asdict(fitted.head), default=_json_default, sort_keys=True),
                "diagnostics_json": json.dumps(fitted.diagnostics, default=_json_default, sort_keys=True),
            }
        )
        prediction = model.predict(fitted, features, split.test)
        if prediction.shape != (len(split.test),) or not np.isfinite(prediction).all():
            raise ValueError("Nonfinite or misshaped raw-screen prediction")
        payload.update(
            {
                "status": "ok",
                "error": "",
                "prediction": prediction,
                "train_prediction": model.predict(fitted, features, split.train),
            }
        )
    except (raw.RawLossFitError, FloatingPointError, ValueError) as error:
        runs = [dataclasses.asdict(run) for run in error.runs] if isinstance(error, raw.RawLossFitError) else []
        trials = [dataclasses.asdict(trial) for trial in error.trials] if isinstance(error, raw.RawLossFitError) else []
        diagnostics = json.loads(payload["diagnostics_json"])
        diagnostics.update({"failed_runs": runs, "failed_trials": trials})
        payload.update(
            {
                "status": "failed",
                "error": f"{type(error).__name__}: {error}\n{traceback.format_exc()[-3000:]}",
                "prediction": np.full(len(split.test), np.nan),
                "train_prediction": np.full(len(split.train), np.nan),
                "diagnostics_json": json.dumps(diagnostics, default=_json_default, sort_keys=True),
            }
        )
    payload["elapsed"] = time.monotonic() - started
    bench.atomic_save(path, payload)
    return "fitted" if payload["status"] == "ok" else "failed"


def collect(tasks: list[bench.FitTask], output: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest = initialize_worker(output)
    prediction_rows, fit_rows = [], []
    for task in tasks:
        panel = bench.load_panel(task.panel)
        geometry = frozen_geometry(task, manifest)
        expected_hash = task_protocol_hash(task, geometry, manifest)
        payload = bench.load_shard(bench.shard_path(output, task))
        metadata = {
            "model": task.model_id,
            "role": "conditional_geometry_probe",
            "parent": GEOMETRY_MODEL,
            "panel": task.panel,
            "panel_kind": panel.kind,
            "curve_family": "",
            "target": task.target,
            "component_index": task.component_index,
            "component": task.component,
            "repeat": task.repeat,
            "fold": task.fold,
        }
        if payload is None:
            fit_rows.append(metadata | {"status": "missing", "error": "shard not found"})
            continue
        if str(payload["protocol_hash"].item()) != expected_hash:
            raise ValueError("Cannot report a raw shard with a changed geometry or protocol")
        diagnostics = json.loads(str(payload["diagnostics_json"].item()))
        status = str(payload["status"].item())
        fit_rows.append(
            metadata
            | {
                "status": status,
                "error": str(payload["error"].item()),
                "elapsed": float(payload["elapsed"]),
                "ridge": float(payload["ridge"]),
                "shape_json": str(payload["shape_json"].item()),
                "geometry_sha256": geometry.checksum,
                **{f"diag_{key}": value for key, value in diagnostics.items()},
            }
        )
        if status != "ok":
            continue
        basin = np.zeros(panel.rows, dtype=bool)
        basin[panel.basin_rows(task.target)] = True
        for index, row in enumerate(payload["test"]):
            prediction_rows.append(
                metadata
                | {
                    "row_index": int(row),
                    "run": panel.runs[row],
                    "observed": float(payload["observed"][index]),
                    "prediction": float(payload["prediction"][index]),
                    "basin": bool(basin[row]),
                }
            )
    return pd.DataFrame(prediction_rows), pd.DataFrame(fit_rows)


def report(tasks: list[bench.FitTask], output: Path, tier: str) -> None:
    predictions, fits = collect(tasks, output)
    destination = output / f"report_{tier}"
    destination.mkdir(exist_ok=True)
    fits.to_csv(destination / "fits.csv", index=False)
    checks = fixed.verify_coverage(tasks, predictions, fits)
    predictions.to_csv(destination / "component_predictions.csv", index=False)
    keys = ["model", "panel", "target", "component", "repeat"]
    components = bench.grouped_metrics(predictions, keys)
    components.to_csv(destination / "component_metrics.csv", index=False)
    bench.grouped_metrics(predictions, [*keys, "fold"]).to_csv(destination / "component_fold_metrics.csv", index=False)
    aggregate = bench.aggregate_predictions(predictions)
    if not aggregate.empty:
        aggregate.to_csv(destination / "aggregate_predictions.csv", index=False)
        group_keys = ["model", "panel", "target", "repeat"]
        bench.grouped_metrics(aggregate, group_keys).to_csv(destination / "aggregate_metrics.csv", index=False)
        bench.grouped_metrics(aggregate, [*group_keys, "fold"]).to_csv(
            destination / "aggregate_fold_metrics.csv", index=False
        )
    checks.update(
        {
            "scope": "conditional geometry; out-of-fold components; no external bank",
            "aggregate_prediction_rows": len(aggregate),
            "tasks": [dataclasses.asdict(task) for task in tasks],
        }
    )
    (destination / "verification.json").write_text(json.dumps(checks, indent=2) + "\n")
    summary = {key: value for key, value in checks.items() if key != "tasks"}
    print(json.dumps({"report": str(destination), **summary}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fixed-output", type=Path, default=fixed.DEFAULT_OUTPUT)
    parser.add_argument("--tier", choices=("smoke", "screen", "certify"), default="screen")
    parser.add_argument("--models", nargs="*")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-nfev", type=int, default=300)
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument(
        "--verify-workers", action="store_true", help="Check frozen source imports in worker processes, without fitting"
    )
    args = parser.parse_args()
    manifest = prepare(args.output, args.fixed_output, args.max_nfev)
    available = {spec.model_id for spec in candidate_specs()}
    ids = tuple(args.models or sorted(available))
    if len(ids) != len(set(ids)) or not set(ids) <= available:
        raise ValueError(f"Models must be unique members of {sorted(available)}")
    module = importlib.import_module(MODULE_NAME)
    if args.verify_workers:
        with joblib.parallel_config(backend="loky", inner_max_num_threads=1):
            checks = joblib.Parallel(n_jobs=args.workers)(
                joblib.delayed(module.worker_identity)(args.output) for _ in range(args.workers)
            )
        print(json.dumps({"workers": checks, "protocol": _digest_json(manifest)}), flush=True)
        return
    fixed.register_candidates()
    plan = dataclasses.replace(bench.tier_plan(args.tier, panels=bench.THIRTY_NINE_BUCKET_PANELS), curves=())
    tasks = bench.plan_tasks(plan, ids)
    tasks.sort(key=lambda task: (task.fold, task.component_index, task.panel, task.target, task.model_id))
    # Fail before any solver starts if one requested outer-training geometry is unavailable.
    for task in tasks:
        frozen_geometry(task, manifest)
    if not args.report_only:
        counts: dict[str, int] = {}
        started = time.monotonic()
        with joblib.parallel_config(backend="loky", inner_max_num_threads=1):
            results = joblib.Parallel(n_jobs=args.workers, return_as="generator_unordered")(
                joblib.delayed(module.fit_task)(task, args.output) for task in tasks
            )
            for index, status in enumerate(results, 1):
                counts[status] = counts.get(status, 0) + 1
                if index % 8 == 0 or index == len(tasks) or status == "failed":
                    print(
                        json.dumps(
                            {
                                "completed": index,
                                "total": len(tasks),
                                "elapsed": round(time.monotonic() - started),
                                **counts,
                            }
                        ),
                        flush=True,
                    )
    report(tasks, args.output, args.tier)


if __name__ == "__main__":
    main()
