# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Ordinary-JAX ABI 8 Host proof for the 7x13 identity-policy boundaries."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import jax
import jaxlib
import numpy as np
from acceptance_contract import ObserverIdentity, decode_native_snapshot
from shuttle_jaxlib_target1_acceptance import (
    CacheHitCounter,
    Shape,
    acceptance_tuning,
    arrays,
    boundary_function,
    bridge_module,
    cache_snapshot,
    configure_cache,
    fixed_inputs,
    ready,
)
from target1_acceptance_contract import target1_expectation, validate_target1_success_events

from shuttle import ExecutionMode, Numerics, compiler_options

JAX_VERSION = "0.10.1"
JAXLIB_VERSION = "0.10.1"
PIPELINE_ABI_VERSION = 8
BOUNDARIES = ("forward", "backward", "composed")
POLICIES = (Numerics.SOURCE_ORDERED, Numerics.FAST)
WORKERS = ("baseline", "populate", "reuse")
SHAPE = Shape(7, 13, "81928ab3539c0f03")


def subject_options(numerics: Numerics) -> dict[str, object]:
    options = compiler_options(
        execution_mode=ExecutionMode.CPU_EXECUTABLE_BUNDLE,
        numerics=numerics,
        tuning=acceptance_tuning(),
    )
    encoded = options["xla_shuttle_options"]
    if not isinstance(encoded, str):
        raise TypeError("xla_shuttle_options must be canonical JSON text")
    payload = json.loads(encoded)
    if payload.get("pipeline_abi_version") != PIPELINE_ABI_VERSION:
        raise AssertionError("identity-policy Host proof requires pipeline ABI 8")
    if payload.get("numerics") != numerics.value:
        raise AssertionError("identity-policy Host proof lost its numerical policy")
    return options


def cell_identities() -> tuple[tuple[str, str], ...]:
    """Return the closed boundary/policy matrix without compiling."""
    return tuple((boundary, numerics.value) for numerics in POLICIES for boundary in BOUNDARIES)


def expected_identity(numerics: Numerics) -> ObserverIdentity:
    options = subject_options(numerics)
    canonical = options["xla_shuttle_options"]
    if not isinstance(canonical, str):
        raise TypeError("xla_shuttle_options must be canonical JSON text")
    payload = json.loads(canonical)
    canonical_tuning = json.dumps(payload["tuning"], sort_keys=True, separators=(",", ":"))
    return ObserverIdentity(
        policy=numerics.value,
        policy_digest=hashlib.sha256(canonical.encode()).hexdigest(),
        tuning_digest=hashlib.sha256(canonical_tuning.encode()).hexdigest(),
        canonical_options=canonical,
        canonical_tuning=canonical_tuning,
    )


def _key(boundary: str, result: int) -> str:
    return f"{boundary}_{result}"


def save_baseline(path: Path) -> dict[str, object]:
    configure_cache(None)
    stored = {}
    for boundary in BOUNDARIES:
        values = arrays(ready(jax.jit(boundary_function(boundary))(*fixed_inputs(SHAPE, boundary))))
        for result, value in enumerate(values):
            if value.dtype != np.dtype("bfloat16"):
                raise AssertionError(f"{boundary}: disabled baseline dtype changed")
            stored[_key(boundary, result)] = value.view(np.uint16)
    np.savez(path, **stored)
    return {"worker": "baseline", "boundaries": list(BOUNDARIES)}


def load_baseline(path: Path, boundary: str) -> tuple[np.ndarray, ...]:
    expected_shapes = {
        "forward": ((7, 13),),
        "backward": ((7, 13), (13,)),
        "composed": ((7, 13), (7, 13), (13,)),
    }[boundary]
    with np.load(path) as stored:
        values = tuple(stored[_key(boundary, result)].copy() for result in range(len(expected_shapes)))
    for result, (value, shape) in enumerate(zip(values, expected_shapes, strict=True)):
        if value.dtype != np.uint16 or value.shape != shape:
            raise AssertionError(f"{boundary}: disabled baseline bit payload {result} changed")
    return values


def run_subject(worker: str, baseline: Path, cache_directory: Path, key_map: Path) -> dict[str, object]:
    if worker not in {"populate", "reuse"}:
        raise ValueError(f"unknown subject worker: {worker}")
    configure_cache(cache_directory)
    initial = cache_snapshot(cache_directory)
    if worker == "populate" and initial:
        raise AssertionError("ABI 8 populate cache must start empty")
    expected_keys = json.loads(key_map.read_text()) if worker == "reuse" else {}
    expected_labels = {f"{boundary}_{numerics}" for boundary, numerics in cell_identities()}
    if worker == "reuse" and set(expected_keys) != expected_labels:
        raise AssertionError("ABI 8 reuse key map does not cover the six exact cells")
    counter = CacheHitCounter()
    jax.monitoring.register_event_listener(counter.observe)
    bridge = bridge_module()
    reports = []
    keys = {}
    for numerics in POLICIES:
        for boundary in BOUNDARIES:
            label = f"{boundary}_{numerics.value}"
            before = cache_snapshot(cache_directory)
            hits = counter.count
            compiled = jax.jit(
                boundary_function(boundary),
                compiler_options=subject_options(numerics),
            )
            with bridge.subscribe() as capture:
                actual = ready(compiled(*fixed_inputs(SHAPE, boundary)))
                events = decode_native_snapshot(capture.snapshot())
            actual_arrays = arrays(actual)
            expected_bits = load_baseline(baseline, boundary)
            for result, (value, bits) in enumerate(zip(actual_arrays, expected_bits, strict=True)):
                if value.dtype != np.dtype("bfloat16") or value.shape != bits.shape:
                    raise AssertionError(f"{label}: output {result} shape or dtype changed")
                if not np.array_equal(value.view(np.uint16), bits):
                    raise AssertionError(f"{label}: output differs from disabled ordinary JAX")
            parity = {
                "bitwise_disabled_jax_parity": True,
                "output_digests": [hashlib.sha256(bits.tobytes(order="C")).hexdigest() for bits in expected_bits],
            }
            after = cache_snapshot(cache_directory)
            added = set(after) - set(before)
            if worker == "populate":
                if len(added) != 1 or counter.count != hits:
                    raise AssertionError(f"{label}: expected one ABI 8 cache miss")
                keys[label] = added.pop()
                observer = validate_target1_success_events(
                    events,
                    expected_identity(numerics),
                    target1_expectation(SHAPE.shape_id, boundary),
                )
            elif after != before or counter.count != hits + 1:
                raise AssertionError(f"{label}: expected one immutable ABI 8 cache hit")
            else:
                keys[label] = expected_keys[label]
                if keys[label] not in after:
                    raise AssertionError(f"{label}: attributed cache entry is missing")
                if events:
                    raise AssertionError(f"{label}: persistent-cache hit invoked Shuttle")
                observer = None
            reports.append(
                {
                    "boundary": boundary,
                    "numerics": numerics.value,
                    "observer": observer,
                    "output": parity,
                }
            )
    final = cache_snapshot(cache_directory)
    if len(final) != len(cell_identities()):
        raise AssertionError("identity-policy Host proof requires exactly six cache entries")
    if worker == "populate" and len(set(keys.values())) != len(cell_identities()):
        raise AssertionError("identity-policy cells must have distinct public cache keys")
    if worker == "populate":
        key_map.write_text(json.dumps(keys, indent=2, sort_keys=True) + "\n")
    if worker == "reuse" and final != initial:
        raise AssertionError("ABI 8 reuse changed persistent-cache bytes")
    return {
        "worker": worker,
        "pipeline_abi_version": PIPELINE_ABI_VERSION,
        "cache_files": sorted(final),
        "cell_to_cache_file": keys,
        "cache_hits": counter.count,
        "boundaries": reports,
    }


def run_worker(arguments: argparse.Namespace) -> dict[str, object]:
    if jax.__version__ != JAX_VERSION or jaxlib.__version__ != JAXLIB_VERSION:
        raise RuntimeError("VJP Host proof requires exact JAX/jaxlib 0.10.1")
    if arguments.worker == "baseline":
        return save_baseline(arguments.baseline)
    return run_subject(arguments.worker, arguments.baseline, arguments.cache_directory, arguments.key_map)


def run_orchestrator(work_directory: Path) -> dict[str, object]:
    work_directory.mkdir(parents=True, exist_ok=True)
    if any(work_directory.iterdir()):
        raise RuntimeError("VJP Host work directory must start empty")
    baseline = work_directory / "disabled-baseline.npz"
    cache_directory = work_directory / "persistent-cache"
    key_map = work_directory / "cache-keys.json"
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
                "--key-map",
                str(key_map),
                "--report",
                str(report),
            ],
            check=True,
            env=environment,
        )
        reports[worker] = json.loads(report.read_text())
    if reports["populate"]["cell_to_cache_file"] != reports["reuse"]["cell_to_cache_file"]:
        raise AssertionError("reuse process changed ABI 8 cache-key attribution")
    if reports["reuse"]["cache_hits"] != len(cell_identities()):
        raise AssertionError("reuse process omitted an identity-policy cache hit")
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
    parser.add_argument("--key-map", type=Path)
    parser.add_argument("--report", required=True, type=Path)
    arguments = parser.parse_args()
    if arguments.worker:
        if arguments.baseline is None:
            parser.error("worker mode requires --baseline")
        if arguments.worker != "baseline" and arguments.cache_directory is None:
            parser.error("cache workers require --cache-directory")
        if arguments.worker != "baseline" and arguments.key_map is None:
            parser.error("cache workers require --key-map")
        result = run_worker(arguments)
    else:
        if arguments.work_directory is None:
            parser.error("orchestrator mode requires --work-directory")
        result = run_orchestrator(arguments.work_directory)
    arguments.report.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
