# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["numpy==2.3.5", "scipy==1.17.0", "pandas==2.2.2"]
# ///
"""Fit frozen single-phase spines on tied rows of registered nested contexts.

Every component uses the snapshotted standalone fit_task without changing its
shape/ridge grid, floor search, or fitting procedure. Contexts and inner folds
are reduced to physical tied rows, then remapped from global to local indices.
Completed task files are reused only after provenance and payload validation.

Example benchmark:
  uv run fit_two_phase_link_spines_20260907.py --contexts final --objectives uncheatable --components 0 --workers 1
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd
import scipy

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = SCRIPT_DIR / "reference_outputs" / "two_phase_link_transfer_20260907"
CONTEXTS = (
    "final",
    *(f"outer{outer}" for outer in range(3)),
    *(f"final_inner{inner}" for inner in range(3)),
    *(f"outer{outer}_inner{inner}" for outer in range(3) for inner in range(3)),
)
OBJECTIVES = ("uncheatable", "table9")
RUNTIME = {
    "python": platform.python_version(),
    "numpy": np.__version__,
    "scipy": scipy.__version__,
    "pandas": pd.__version__,
}
THREAD_VARIABLES = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


@dataclass(frozen=True)
class SpineJob:
    output: Path
    context: str
    objective: str
    component_index: int
    input_sha256: dict[str, str]
    runner_sha256: str


@dataclass(frozen=True)
class ContextRows:
    train: np.ndarray
    local_folds: tuple[tuple[np.ndarray, np.ndarray], ...]
    inner_names: tuple[str, ...]
    global_folds: tuple[tuple[np.ndarray, np.ndarray], ...]


def load_module(path: Path, name: str) -> ModuleType:
    """Load a frozen source file with dataclass-compatible module registration."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load Python source from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Publish complete JSON so interrupted work cannot leave a valid-looking cache."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", prefix=f".{path.name}.", dir=path.parent, delete=False) as stream:
        stream.write(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def context_rows(panel: dict[str, np.ndarray], splits: dict[str, np.ndarray], context: str) -> ContextRows:
    """Reduce a registered context and its three fitting folds to tied rows."""
    if context not in CONTEXTS:
        raise ValueError(f"unregistered context: {context}")
    tied = panel["physical_tied"]
    calibration = panel["calibration_mask"]
    eligible = np.arange(len(tied)) if context == "final" else splits[f"{context}_train"]
    train = eligible[tied[eligible]]
    local = np.full(len(tied), -1, dtype=int)
    local[train] = np.arange(len(train))
    inner_kind = "sub" if "_inner" in context else "inner"
    names = tuple(f"{context}_{inner_kind}{fold}" for fold in range(3))
    folds = []
    global_folds = []
    calibration_rows = np.flatnonzero(calibration & tied)
    if not len(calibration_rows) or not np.isin(calibration_rows, train).all():
        raise ValueError(f"calibration tied rows are absent from context {context}")
    for name in names:
        training = splits[f"{name}_train"]
        validation = splits[f"{name}_test"]
        training = training[tied[training]]
        validation = validation[tied[validation]]
        if not len(validation) or np.intersect1d(training, validation).size:
            raise ValueError(f"empty or overlapping tied validation fold {name}")
        if not np.isin(calibration_rows, training).all() or calibration[validation].any():
            raise ValueError(f"calibration pinning failed for {name}")
        if not np.array_equal(np.sort(np.concatenate((training, validation))), np.sort(train)):
            raise ValueError(f"inner fold {name} does not partition the context's tied rows")
        if np.intersect1d(panel["groups"][training], panel["groups"][validation]).size:
            raise ValueError(f"correspondence groups cross inner fold {name}")
        folds.append((local[training], local[validation]))
        global_folds.append((training, validation))
    scored = np.concatenate([validation for _, validation in global_folds])
    if not np.array_equal(np.sort(scored), np.sort(train[~calibration[train]])):
        raise ValueError(f"inner validation does not cover every noncalibration tied row once for {context}")
    return ContextRows(train, tuple(folds), names, tuple(global_folds))


def fit_spine(job: SpineJob) -> dict[str, Any]:
    """Fit or validate one component cache using the complete standalone procedure."""
    inputs = job.output / "inputs"
    with np.load(inputs / "panel.npz", allow_pickle=False) as arrays:
        panel = {name: arrays[name] for name in arrays.files}
    with np.load(inputs / "splits.npz", allow_pickle=False) as arrays:
        splits = {name: arrays[name] for name in arrays.files}
    rows = context_rows(panel, splits, job.context)
    component = str(panel[f"{job.objective}_components"][job.component_index])
    anchor_index = int(np.flatnonzero(panel["anchor_components"] == component).item())
    provenance = {
        "context": job.context,
        "objective": job.objective,
        "component_index": job.component_index,
        "component": component,
        "input_sha256": job.input_sha256,
        "runner_sha256": job.runner_sha256,
        "runtime": RUNTIME,
        "train_rows": rows.train.tolist(),
        "inner_folds": [
            {"name": name, "train_rows": train.tolist(), "validation_rows": validation.tolist()}
            for name, (train, validation) in zip(rows.inner_names, rows.global_folds, strict=True)
        ],
        "calibration_rows": rows.train[panel["calibration_mask"][rows.train]].tolist(),
        "anchor": {
            "proportional_bpb": float(panel["anchor_proportional_bpb"][anchor_index]),
            "repeat_sd": float(panel["anchor_repeat_sd"][anchor_index]),
            "sd_is_aggregate_approximation": bool(panel["anchor_sd_is_aggregate_approximation"][anchor_index]),
        },
    }
    target = job.output / "spines" / job.context / f"{job.objective}_c{job.component_index}.json"
    if target.exists():
        saved = json.loads(target.read_text())
        if saved["meta"]["provenance"] != provenance:
            raise ValueError(f"cache provenance changed; preserve old results and choose another output: {target}")
        fit_payload = {key: value for key, value in saved.items() if key != "meta"}
        digest = hashlib.sha256(json.dumps(fit_payload, sort_keys=True, allow_nan=False).encode()).hexdigest()
        if saved["meta"]["fit_sha256"] != digest:
            raise ValueError(f"cached fit payload was modified: {target}")
        return {
            "context": job.context,
            "objective": job.objective,
            "component_index": job.component_index,
            "cached": True,
        }
    single_phase = load_module(inputs / "single_phase.py", "frozen_two_phase_link_single_phase")
    outcome = panel[f"{job.objective}_outcomes"][rows.train, job.component_index]
    if not np.isfinite(outcome).all():
        raise ValueError(f"nonfinite tied outcome for {job.context}/{component}")
    swarm = single_phase.Swarm(
        runs=tuple(panel["runs"][rows.train].tolist()),
        buckets=tuple(panel["buckets"].tolist()),
        weights=panel["aggregate"][rows.train],
        inventory=panel["inventory"],
        outcomes=pd.DataFrame({component: outcome}),
        calibration=panel["calibration_mask"][rows.train],
    )
    anchor = single_phase.Anchor(
        float(panel["anchor_proportional_bpb"][anchor_index]), float(panel["anchor_repeat_sd"][anchor_index])
    )
    start = time.monotonic()
    fit = single_phase.fit_task(swarm, component, anchor, rows.local_folds)
    elapsed = time.monotonic() - start
    payload = fit.to_json()
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()
    payload["meta"] = {
        "provenance": provenance,
        "fit_sha256": digest,
        "fit_seconds": elapsed,
        "completed_at": datetime.now(UTC).isoformat(),
        "shape_candidates": len(single_phase.SHAPES),
        "ridge_candidates": len(single_phase.RIDGE_GRID),
        "fitting_entrypoint": "single_phase.fit_task",
    }
    write_json_atomic(target, payload)
    return {
        "context": job.context,
        "objective": job.objective,
        "component_index": job.component_index,
        "cached": False,
        "fit_seconds": elapsed,
        "train_rows": len(rows.train),
    }


def validate_inputs(output: Path) -> dict[str, str]:
    """Verify every frozen input file against the preparation manifest."""
    inputs = output / "inputs"
    manifest_path = inputs / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    hashes = {}
    for name, expected in manifest["output_sha256"].items():
        actual = hashlib.sha256((inputs / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"snapshot hash mismatch for {name}")
        hashes[name] = actual
    hashes["manifest.json"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    return hashes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--contexts", default=",".join(CONTEXTS))
    parser.add_argument("--objectives", default=",".join(OBJECTIVES))
    parser.add_argument("--components", help="Comma-separated component indices within each selected objective")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    contexts = tuple(args.contexts.split(","))
    objectives = tuple(args.objectives.split(","))
    if not set(contexts) <= set(CONTEXTS) or not set(objectives) <= set(OBJECTIVES):
        raise ValueError("contexts and objectives must be registered names")
    if len(set(contexts)) != len(contexts) or len(set(objectives)) != len(objectives):
        raise ValueError("contexts and objectives must be unique")
    hashes = validate_inputs(output)
    runner_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with np.load(output / "inputs" / "panel.npz", allow_pickle=False) as arrays:
        sizes = {objective: len(arrays[f"{objective}_components"]) for objective in objectives}
    jobs = []
    for context in contexts:
        for objective in objectives:
            components = (
                tuple(range(sizes[objective]))
                if args.components is None
                else tuple(map(int, args.components.split(",")))
            )
            if len(set(components)) != len(components) or any(
                index < 0 or index >= sizes[objective] for index in components
            ):
                raise ValueError(f"invalid component indices for {objective}")
            jobs.extend(SpineJob(output, context, objective, index, hashes, runner_hash) for index in components)
    single_phase = load_module(output / "inputs" / "single_phase.py", "frozen_two_phase_link_single_phase")
    temporal = load_module(SCRIPT_DIR / "two_phase_link_residual_20260907.py", "two_phase_link_residual_checks")
    checks = temporal.run_structural_checks(
        benefit=lambda exposure: single_phase.benefit(exposure, 0.7, 0.65),
        harm=lambda exposure: single_phase.harm(exposure, 1.7),
    )
    checks["source_sha256"] = hashes["single_phase.py"]
    checks["temporal_module_sha256"] = hashlib.sha256(
        (SCRIPT_DIR / "two_phase_link_residual_20260907.py").read_bytes()
    ).hexdigest()
    write_json_atomic(output / "structural_checks.json", checks)
    # Spawned workers import NumPy after these process-local thread caps are set.
    for variable in THREAD_VARIABLES:
        os.environ[variable] = "1"
    started = time.monotonic()
    results = []
    print(json.dumps({"event": "start", "jobs": len(jobs), "workers": args.workers, "runtime": RUNTIME}), flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(fit_spine, job) for job in jobs]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            progress = {
                "requested": len(jobs),
                "completed": len(results),
                "cached": sum(item["cached"] for item in results),
                "elapsed_seconds": time.monotonic() - started,
                "contexts": list(contexts),
                "objectives": list(objectives),
                "runner_sha256": runner_hash,
                "last_result": result,
            }
            write_json_atomic(output / "spines" / "run_status.json", progress)
            print(json.dumps({"event": "fit", "completed": len(results), "total": len(jobs), **result}), flush=True)
    print(
        json.dumps({"event": "complete", "jobs": len(results), "elapsed_seconds": time.monotonic() - started}),
        flush=True,
    )


if __name__ == "__main__":
    main()
