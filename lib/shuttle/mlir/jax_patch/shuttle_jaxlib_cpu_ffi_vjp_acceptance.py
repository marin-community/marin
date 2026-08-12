# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Ordinary-JAX ABI 7 Host proof for the 7x13 VJP boundaries."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import jax
import jaxlib
import numpy as np
from shuttle_jaxlib_target1_acceptance import (
    CacheHitCounter,
    Shape,
    acceptance_tuning,
    arrays,
    boundary_function,
    cache_snapshot,
    configure_cache,
    fixed_inputs,
    ready,
    require_bitwise,
)

from shuttle import ExecutionMode, Numerics, compiler_options

JAX_VERSION = "0.10.1"
JAXLIB_VERSION = "0.10.1"
PIPELINE_ABI_VERSION = 7
BOUNDARIES = ("backward", "composed")
WORKERS = ("baseline", "populate", "reuse")
SHAPE = Shape(7, 13, "81928ab3539c0f03")


def subject_options() -> dict[str, object]:
    options = compiler_options(
        execution_mode=ExecutionMode.CPU_EXECUTABLE_BUNDLE,
        numerics=Numerics.SOURCE_ORDERED,
        tuning=acceptance_tuning(),
    )
    encoded = options["xla_shuttle_options"]
    if not isinstance(encoded, str):
        raise TypeError("xla_shuttle_options must be canonical JSON text")
    payload = json.loads(encoded)
    if payload.get("pipeline_abi_version") != PIPELINE_ABI_VERSION:
        raise AssertionError("VJP Host proof requires pipeline ABI 7")
    return options


def _key(boundary: str, result: int) -> str:
    return f"{boundary}_{result}"


def save_baseline(path: Path) -> dict[str, object]:
    configure_cache(None)
    stored = {}
    for boundary in BOUNDARIES:
        values = arrays(ready(jax.jit(boundary_function(boundary))(*fixed_inputs(SHAPE, boundary))))
        for result, value in enumerate(values):
            stored[_key(boundary, result)] = value
    np.savez(path, **stored)
    return {"worker": "baseline", "boundaries": list(BOUNDARIES)}


def load_baseline(path: Path, boundary: str) -> tuple[np.ndarray, ...]:
    count = 2 if boundary == "backward" else 3
    with np.load(path) as stored:
        return tuple(stored[_key(boundary, result)].copy() for result in range(count))


def run_subject(worker: str, baseline: Path, cache_directory: Path) -> dict[str, object]:
    if worker not in {"populate", "reuse"}:
        raise ValueError(f"unknown subject worker: {worker}")
    configure_cache(cache_directory)
    initial = cache_snapshot(cache_directory)
    if worker == "populate" and initial:
        raise AssertionError("ABI 7 populate cache must start empty")
    counter = CacheHitCounter()
    jax.monitoring.register_event_listener(counter.observe)
    reports = []
    for boundary in BOUNDARIES:
        before = cache_snapshot(cache_directory)
        hits = counter.count
        compiled = jax.jit(boundary_function(boundary), compiler_options=subject_options())
        actual = ready(compiled(*fixed_inputs(SHAPE, boundary)))
        parity = require_bitwise(actual, load_baseline(baseline, boundary), boundary)
        after = cache_snapshot(cache_directory)
        if worker == "populate":
            if len(after) != len(before) + 1 or counter.count != hits:
                raise AssertionError(f"{boundary}: expected one ABI 7 cache miss")
        elif after != before or counter.count != hits + 1:
            raise AssertionError(f"{boundary}: expected one immutable ABI 7 cache hit")
        reports.append({"boundary": boundary, "output": parity})
    final = cache_snapshot(cache_directory)
    if len(final) != 2:
        raise AssertionError("VJP Host proof requires exactly two cache entries")
    if worker == "reuse" and final != initial:
        raise AssertionError("ABI 7 reuse changed persistent-cache bytes")
    return {
        "worker": worker,
        "pipeline_abi_version": PIPELINE_ABI_VERSION,
        "cache_files": sorted(final),
        "cache_hits": counter.count,
        "boundaries": reports,
    }


def run_worker(arguments: argparse.Namespace) -> dict[str, object]:
    if jax.__version__ != JAX_VERSION or jaxlib.__version__ != JAXLIB_VERSION:
        raise RuntimeError("VJP Host proof requires exact JAX/jaxlib 0.10.1")
    if arguments.worker == "baseline":
        return save_baseline(arguments.baseline)
    return run_subject(arguments.worker, arguments.baseline, arguments.cache_directory)


def run_orchestrator(work_directory: Path) -> dict[str, object]:
    work_directory.mkdir(parents=True, exist_ok=True)
    if any(work_directory.iterdir()):
        raise RuntimeError("VJP Host work directory must start empty")
    baseline = work_directory / "disabled-baseline.npz"
    cache_directory = work_directory / "persistent-cache"
    reports = {}
    for worker in WORKERS:
        report = work_directory / f"{worker}.json"
        environment = dict(os.environ)
        environment["JAX_PLATFORMS"] = "cpu"
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                worker,
                "--baseline",
                str(baseline),
                "--cache-directory",
                str(cache_directory),
                "--report",
                str(report),
            ],
            check=True,
            env=environment,
        )
        reports[worker] = json.loads(report.read_text())
    return {
        "classification": "local_cpu_architecture_and_disabled_jax_numerical_only",
        "hardware_evidence": "none",
        "scorecard_status_changed": False,
        "workers": reports,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-directory", type=Path)
    parser.add_argument("--worker", choices=WORKERS)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--cache-directory", type=Path)
    parser.add_argument("--report", required=True, type=Path)
    arguments = parser.parse_args()
    if arguments.worker:
        if arguments.baseline is None:
            parser.error("worker mode requires --baseline")
        if arguments.worker != "baseline" and arguments.cache_directory is None:
            parser.error("cache workers require --cache-directory")
        result = run_worker(arguments)
    else:
        if arguments.work_directory is None:
            parser.error("orchestrator mode requires --work-directory")
        result = run_orchestrator(arguments.work_directory)
    arguments.report.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
