# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Ordinary-JAX ABI 8 Host proof for the 7x13 identity-policy boundaries."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

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
from target1_acceptance_contract import Target1FixtureExpectation, target1_expectation

from shuttle import ExecutionMode, Numerics, compiler_options

JAX_VERSION = "0.10.1"
JAXLIB_VERSION = "0.10.1"
PIPELINE_ABI_VERSION = 8
BOUNDARIES = ("forward", "backward", "composed")
POLICIES = (Numerics.SOURCE_ORDERED, Numerics.FAST)
WORKERS = ("baseline", "populate", "reuse")
SHAPE = Shape(7, 13, "81928ab3539c0f03")
CPU_BUNDLE_PHASES = ("algebra_coverage", "final_erasure")
CPU_BUNDLE_FINAL_FINGERPRINTS = {
    (
        "81928ab3539c0f03",
        "forward",
        "source_ordered",
    ): "8e3af5d400b23fdd2924058ff242a1947e2cbf89b2c35dd2839609c28bf44467",
    (
        "81928ab3539c0f03",
        "backward",
        "source_ordered",
    ): "27e878ad4b71067810c6f63e05fa8233fba53fccaab721f719704f752f2ef04c",
    (
        "81928ab3539c0f03",
        "composed",
        "source_ordered",
    ): "80354aaede9c4779dfc542fb229cdc62688097fd395cbb9488ec587326a68885",
    ("81928ab3539c0f03", "forward", "fast"): "748116e5720695370aacd37d7d569c9498366d2083c008a86c4a4ed1d07fb6cf",
    ("81928ab3539c0f03", "backward", "fast"): "b9dbdffc10095e6f085e0194e214188c6cc88f8cffad64b5ad3e62b55722f32c",
    ("81928ab3539c0f03", "composed", "fast"): "137f69ffaeda5aec2f2be52c15fcfe34767368ebb5a34a5c420872d5e8d21206",
}


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


def validate_cpu_bundle_success_events(
    events: Sequence[Mapping[str, Any]],
    identity: ObserverIdentity,
    fixture: Target1FixtureExpectation,
) -> dict[str, Any]:
    """Require the CPU bundle pipeline's coverage and final-erasure phases."""
    if len(events) != len(CPU_BUNDLE_PHASES):
        raise AssertionError("one CPU bundle compilation must emit exactly two observer phases")
    if tuple(event["phase"] for event in events) != CPU_BUNDLE_PHASES:
        raise AssertionError("one CPU bundle compilation must emit the two ordered observer phases")
    if len({event["invocation_id"] for event in events}) != 1:
        raise AssertionError("CPU bundle observer phases do not share one invocation ID")

    expected_manifest = fixture.coverage_manifest(identity)
    if fixture.excluded_manifest != "[]":
        raise AssertionError("CPU bundle observer fixture contains an excluded source result")
    for event in events:
        if event["policy"] != identity.policy or event["policy_digest"] != identity.policy_digest:
            raise AssertionError("CPU bundle observer policy identity differs from compiler options")
        if event["tuning_digest"] != identity.tuning_digest:
            raise AssertionError("CPU bundle observer tuning identity differs from compiler options")
        if event["failure_pass"] != "":
            raise AssertionError("successful CPU bundle compilation emitted a failure pass")

    algebra, final = events
    if algebra["region_membership"] != fixture.region_membership:
        raise AssertionError(f"{fixture.label}: CPU bundle selected-region membership changed")
    if algebra["coverage_manifest"] != expected_manifest:
        raise AssertionError(f"{fixture.label}: CPU bundle coverage manifest changed")
    if algebra["unsupported_fingerprint"] != fixture.unsupported_fingerprint:
        raise AssertionError(f"{fixture.label}: CPU bundle unsupported structural island changed")
    if algebra["normalized_module_fingerprint"] != "" or algebra["no_shuttle_semantics"] is not False:
        raise AssertionError("CPU bundle algebra phase erased provenance early")

    if final["region_membership"] != "" or final["coverage_manifest"] != "":
        raise AssertionError("CPU bundle final phase retained provenance")
    if final["unsupported_fingerprint"] != "":
        raise AssertionError("CPU bundle final phase retained an unsupported-island fingerprint")
    try:
        expected_fingerprint = CPU_BUNDLE_FINAL_FINGERPRINTS[(fixture.shape_id, fixture.boundary, identity.policy)]
    except KeyError as error:
        raise AssertionError("CPU bundle observer identity is outside the closed six-cell matrix") from error
    if final["normalized_module_fingerprint"] != expected_fingerprint:
        raise AssertionError(f"{fixture.label}: CPU bundle final fingerprint changed")
    if final["no_shuttle_semantics"] is not True:
        raise AssertionError("CPU bundle final phase did not prove erasure")

    return {
        "invocation_id": algebra["invocation_id"],
        "shape_id": fixture.shape_id,
        "boundary": fixture.boundary,
        "policy": identity.policy,
        "policy_digest": identity.policy_digest,
        "tuning_digest": identity.tuning_digest,
        "complete_source_results": len(fixture.complete),
        "excluded_source_results": 0,
        "coverage_manifest": expected_manifest,
        "final_fingerprint": expected_fingerprint,
    }


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
                observer = validate_cpu_bundle_success_events(
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
