# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Installed-wheel CPU contract for the six ABI 6 BF16 rowwise fixtures."""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import jaxlib
import ml_dtypes
import numpy as np
from acceptance_contract import ObserverIdentity, decode_native_snapshot
from jaxlib import _jax
from target1_acceptance_contract import (
    Target1FixtureExpectation,
    target1_expectation,
    validate_target1_success_events,
)

from shuttle import Materialization, Numerics, Tuning, compiler_options

JAX_VERSION = "0.10.1"
JAXLIB_VERSION = "0.10.1"
PIPELINE_ABI_VERSION = 6
CACHE_HIT_EVENT = "/jax/compilation_cache/cache_hits"
BOUNDARIES = ("forward", "backward", "composed")
WORKER_MODES = ("baseline", "populate", "reuse")
MAX_CACHE_ENTRY_BYTES = 1 << 30
MAX_CACHE_BYTES = 12 * MAX_CACHE_ENTRY_BYTES
CACHE_FILE = re.compile(r"^jit_(forward|backward|composed)-[0-9a-f]{64}-cache$")


@dataclass(frozen=True)
class Shape:
    rows: int
    features: int
    shape_id: str


@dataclass(frozen=True)
class Wrapper:
    label: str
    numerics: Numerics
    fixture: Target1FixtureExpectation
    function: Callable[..., Any]
    arguments: tuple[np.ndarray, ...]
    baseline: tuple[np.ndarray, ...]


SHAPES = (
    Shape(2048, 4096, "44d152ecc3e9ff18"),
    Shape(7, 13, "81928ab3539c0f03"),
)


class CacheHitCounter:
    def __init__(self) -> None:
        self.count = 0

    def observe(self, event: str, **metadata: str | int) -> None:
        del metadata
        if event == CACHE_HIT_EVENT:
            self.count += 1


def acceptance_tuning() -> Tuning:
    """Match the default physical policy frozen by the fixture corpus."""
    return Tuning(
        tile_sizes=(),
        cluster_shape=(),
        pipeline_stages=1,
        materialization=Materialization.AUTOMATIC,
        maximum_candidates=1,
    )


def row_fold_scale(x: jax.Array, gamma: jax.Array) -> jax.Array:
    local = x.astype(jnp.float32)
    inverse = jax.lax.rsqrt(jnp.mean(local * local, axis=-1, keepdims=True) + 1e-5)
    return (local * inverse * gamma.astype(jnp.float32)).astype(jnp.bfloat16)


def boundary_function(boundary: str) -> Callable[..., Any]:
    if boundary == "forward":

        def forward(x, gamma):
            return row_fold_scale(x, gamma)

        return forward
    if boundary == "backward":

        def backward(x, gamma, dy):
            _, pullback = jax.vjp(row_fold_scale, x, gamma)
            return pullback(dy)

        return backward
    if boundary == "composed":

        def composed(x, gamma, dy):
            y, pullback = jax.vjp(row_fold_scale, x, gamma)
            dx, dgamma = pullback(dy)
            return y, dx, dgamma

        return composed
    raise ValueError(f"unsupported boundary {boundary!r}")


def fixed_inputs(shape: Shape, boundary: str) -> tuple[np.ndarray, ...]:
    dimensions = ((shape.rows, shape.features), (shape.features,))
    if boundary != "forward":
        dimensions += ((shape.rows, shape.features),)
    values = []
    for ordinal, item_shape in enumerate(dimensions):
        size = int(np.prod(item_shape))
        start = np.float32(-0.75 + ordinal * 0.125)
        stop = np.float32(0.875 + ordinal * 0.125)
        values.append(np.linspace(start, stop, size, dtype=np.float32).reshape(item_shape).astype(ml_dtypes.bfloat16))
    return tuple(values)


def ready(value: Any) -> Any:
    return jax.tree.map(lambda leaf: leaf.block_until_ready(), value)


def arrays(value: Any) -> tuple[np.ndarray, ...]:
    return tuple(np.asarray(leaf) for leaf in jax.tree.leaves(value))


def output_digest(values: Sequence[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value.dtype).encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def require_bitwise(actual: Any, expected: Sequence[np.ndarray], label: str) -> dict[str, Any]:
    actual_arrays = arrays(actual)
    if len(actual_arrays) != len(expected):
        raise AssertionError(f"{label}: output arity changed")
    for actual_array, expected_array in zip(actual_arrays, expected, strict=True):
        if actual_array.shape != expected_array.shape or actual_array.dtype != expected_array.dtype:
            raise AssertionError(f"{label}: output shape or dtype changed")
        if not np.array_equal(actual_array, expected_array):
            raise AssertionError(f"{label}: output differs from disabled ordinary JAX")
    return {
        "bitwise_disabled_jax_parity": True,
        "output_digests": [output_digest((value,)) for value in actual_arrays],
    }


def bridge_module() -> Any:
    bridge = getattr(_jax, "_shuttle_test_observer", None)
    if bridge is None:
        raise RuntimeError("jaxlib was not built with the test-only Shuttle observer bridge")
    return bridge


def expected_identity(numerics: Numerics) -> ObserverIdentity:
    options = compiler_options(numerics=numerics, tuning=acceptance_tuning())
    canonical = options["xla_shuttle_options"]
    if not isinstance(canonical, str):
        raise AssertionError("canonical options must be a string")
    payload = json.loads(canonical)
    if payload.get("pipeline_abi_version") != PIPELINE_ABI_VERSION:
        raise AssertionError("installed Shuttle options are not ABI 6")
    canonical_tuning = json.dumps(payload["tuning"], sort_keys=True, separators=(",", ":"))
    return ObserverIdentity(
        policy=numerics.value,
        policy_digest=hashlib.sha256(canonical.encode()).hexdigest(),
        tuning_digest=hashlib.sha256(canonical_tuning.encode()).hexdigest(),
        canonical_options=canonical,
        canonical_tuning=canonical_tuning,
    )


def configure_cache(cache_directory: Path | None) -> None:
    jax.config.update("jax_compilation_cache_check_contents", False)
    jax.config.update("jax_compilation_cache_expect_pgle", False)
    jax.config.update("jax_compilation_cache_include_metadata_in_key", False)
    jax.config.update("jax_enable_pgle", False)
    jax.config.update("jax_persistent_cache_enable_xla_caches", "")
    jax.config.update("jax_share_binary_between_hosts", False)
    if cache_directory is None:
        jax.config.update("jax_enable_compilation_cache", False)
        return
    cache_directory.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(cache_directory))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_compilation_cache_max_size", -1)


def cache_snapshot(cache_directory: Path) -> dict[str, tuple[int, str]]:
    """Return the closed flat public cache inventory without following links."""
    if not cache_directory.exists():
        return {}
    snapshot = {}
    total_bytes = 0
    for path in sorted(cache_directory.iterdir()):
        if path.is_symlink() or not path.is_file():
            raise AssertionError(f"persistent cache contains a non-regular entry: {path.name}")
        if not CACHE_FILE.fullmatch(path.name):
            raise AssertionError(f"persistent cache contains an unknown entry: {path.name}")
        size = path.stat().st_size
        if size > MAX_CACHE_ENTRY_BYTES:
            raise AssertionError(f"persistent cache entry exceeds the byte limit: {path.name}")
        total_bytes += size
        snapshot[path.name] = (size, hashlib.sha256(path.read_bytes()).hexdigest())
    if total_bytes > MAX_CACHE_BYTES:
        raise AssertionError("persistent cache exceeds the total byte limit")
    return snapshot


def cache_files(cache_directory: Path) -> frozenset[str]:
    return frozenset(cache_snapshot(cache_directory))


def contract_identities() -> tuple[tuple[str, str, str], ...]:
    """Return the closed shape/boundary/policy matrix without compiling."""
    return tuple(
        (shape.shape_id, boundary, numerics.value)
        for numerics in (Numerics.SOURCE_ORDERED, Numerics.FAST)
        for shape in SHAPES
        for boundary in BOUNDARIES
    )


def attributed_new_cache_entry(before: frozenset[str], after: frozenset[str], label: str) -> str:
    """Return the one public persistent-cache file added by a wrapper."""
    added = after - before
    if len(added) != 1:
        raise AssertionError(f"{label}: first ABI 6 compile did not add exactly one cache entry")
    return next(iter(added))


def _baseline_key(shape: Shape, boundary: str, result: int) -> str:
    return f"{shape.shape_id}_{boundary}_{result}"


def save_baselines(path: Path) -> dict[str, Any]:
    configure_cache(None)
    stored = {}
    evidence = {}
    for shape in SHAPES:
        for boundary in BOUNDARIES:
            values = arrays(ready(jax.jit(boundary_function(boundary))(*fixed_inputs(shape, boundary))))
            for result, value in enumerate(values):
                stored[_baseline_key(shape, boundary, result)] = value
            evidence[f"{shape.shape_id}_{boundary}"] = [output_digest((value,)) for value in values]
    np.savez(path, **stored)
    return {"disabled_jax_output_digests": evidence}


def load_baseline(path: Path, shape: Shape, boundary: str) -> tuple[np.ndarray, ...]:
    result_count = {"forward": 1, "backward": 2, "composed": 3}[boundary]
    with np.load(path) as stored:
        return tuple(stored[_baseline_key(shape, boundary, result)].copy() for result in range(result_count))


def make_wrappers(baseline_path: Path) -> tuple[Wrapper, ...]:
    return tuple(
        Wrapper(
            label=f"{shape.shape_id}_{boundary}_{numerics.value}",
            numerics=numerics,
            fixture=target1_expectation(shape.shape_id, boundary),
            function=boundary_function(boundary),
            arguments=fixed_inputs(shape, boundary),
            baseline=load_baseline(baseline_path, shape, boundary),
        )
        for numerics in (Numerics.SOURCE_ORDERED, Numerics.FAST)
        for shape in SHAPES
        for boundary in BOUNDARIES
    )


def run_cache_worker(mode: str, cache_directory: Path, baseline_path: Path, key_map_path: Path) -> dict[str, Any]:
    configure_cache(cache_directory)
    wrappers = make_wrappers(baseline_path)
    counter = CacheHitCounter()
    jax.monitoring.register_event_listener(counter.observe)
    expected_keys = json.loads(key_map_path.read_text()) if mode == "reuse" else {}
    initial_files = cache_files(cache_directory)
    initial_snapshot = cache_snapshot(cache_directory)
    if mode == "populate" and initial_files:
        raise AssertionError("ABI 6 populate cache must start empty")
    expected_cache_files = frozenset(expected_keys.values())
    if mode == "reuse" and not expected_cache_files.issubset(initial_files):
        raise AssertionError("ABI 6 reuse cache omits an attributed Target 1 entry")

    reports = []
    label_to_key = {}
    bridge = bridge_module()
    for wrapper in wrappers:
        compiled = jax.jit(
            wrapper.function,
            compiler_options=compiler_options(numerics=wrapper.numerics, tuning=acceptance_tuning()),
        )
        files_before = cache_files(cache_directory)
        snapshot_before = cache_snapshot(cache_directory)
        hits_before = counter.count
        with bridge.subscribe() as capture:
            first = ready(compiled(*wrapper.arguments))
            events_after_first = decode_native_snapshot(capture.snapshot())
            files_after_first = cache_files(cache_directory)
            hits_after_first = counter.count
            second = ready(compiled(*wrapper.arguments))
            events_after_second = decode_native_snapshot(capture.snapshot())
        parity = require_bitwise(first, wrapper.baseline, wrapper.label)
        require_bitwise(second, wrapper.baseline, wrapper.label + " second call")
        if files_after_first != cache_files(cache_directory):
            raise AssertionError(f"{wrapper.label}: second call changed the persistent cache")
        snapshot_after_second = cache_snapshot(cache_directory)
        for name, metadata in snapshot_before.items():
            if snapshot_after_second.get(name) != metadata:
                raise AssertionError(f"{wrapper.label}: compile mutated an existing cache entry")

        if mode == "populate":
            key = attributed_new_cache_entry(files_before, files_after_first, wrapper.label)
            observer = validate_target1_success_events(
                events_after_first,
                expected_identity(wrapper.numerics),
                wrapper.fixture,
            )
            if events_after_second != events_after_first or hits_after_first != hits_before:
                raise AssertionError(f"{wrapper.label}: same jitted object recompiled")
        else:
            key = expected_keys[wrapper.label]
            observer = None
            if events_after_first or events_after_second:
                raise AssertionError(f"{wrapper.label}: persistent-cache hit invoked Shuttle")
            if hits_after_first - hits_before != 1 or counter.count != hits_after_first:
                raise AssertionError(f"{wrapper.label}: expected one public cache hit")
            if files_after_first != files_before:
                raise AssertionError(f"{wrapper.label}: cache hit added an entry")
        label_to_key[wrapper.label] = key
        reports.append({"label": wrapper.label, "cache_file": key, "observer": observer, "output": parity})

    attributed_files = frozenset(label_to_key.values())
    if len(label_to_key) != 12 or len(attributed_files) != 12:
        raise AssertionError("Target 1 workers did not attribute twelve distinct ABI 6 cache entries")
    final_files = cache_files(cache_directory)
    if not attributed_files.issubset(final_files):
        raise AssertionError("Target 1 workers lost an attributed ABI 6 cache entry")
    if len(final_files) != 12:
        raise AssertionError("Target 1 cache must contain exactly twelve closed regular-file entries")
    if mode == "populate":
        key_map_path.write_text(json.dumps(label_to_key, indent=2, sort_keys=True) + "\n")
    elif counter.count != 12:
        raise AssertionError("reuse worker did not report twelve public cache hits")
    if mode == "reuse" and cache_snapshot(cache_directory) != initial_snapshot:
        raise AssertionError("reuse worker changed persistent-cache bytes")
    return {
        "mode": mode,
        "pipeline_abi_version": PIPELINE_ABI_VERSION,
        "cache_files": sorted(final_files),
        "label_to_cache_file": label_to_key,
        "public_cache_hits": counter.count,
        "wrappers": reports,
    }


def run_worker(arguments: argparse.Namespace) -> dict[str, Any]:
    if jax.__version__ != JAX_VERSION or jaxlib.__version__ != JAXLIB_VERSION:
        raise RuntimeError(f"acceptance requires JAX/jaxlib {JAX_VERSION}; found {jax.__version__}/{jaxlib.__version__}")
    if arguments.worker == "baseline":
        return save_baselines(arguments.baseline)
    return run_cache_worker(arguments.worker, arguments.cache_directory, arguments.baseline, arguments.key_map)


def run_orchestrator(work_directory: Path) -> dict[str, Any]:
    work_directory.mkdir(parents=True, exist_ok=True)
    if any(work_directory.iterdir()):
        raise RuntimeError("Target 1 acceptance work directory must start empty")
    baseline = work_directory / "disabled-baseline.npz"
    cache_directory = work_directory / "persistent-cache"
    key_map = work_directory / "cache-keys.json"
    reports = {}
    for worker in WORKER_MODES:
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
    if reports["populate"]["label_to_cache_file"] != reports["reuse"]["label_to_cache_file"]:
        raise AssertionError("reuse process changed ABI 6 cache-key attribution")
    if reports["reuse"]["public_cache_hits"] != 12:
        raise AssertionError("reuse process omitted a Target 1 cache hit")
    return {
        "classification": "local_cpu_architecture_and_disabled_jax_numerical_only",
        "hardware_evidence": "none",
        "oracle_status": "not_pinned",
        "scorecard_status_changed": False,
        "workers": reports,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-directory", type=Path)
    parser.add_argument("--worker", choices=WORKER_MODES)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--cache-directory", type=Path)
    parser.add_argument("--key-map", type=Path)
    parser.add_argument("--report", required=True, type=Path)
    arguments = parser.parse_args()
    if arguments.worker:
        if arguments.baseline is None:
            parser.error("worker mode requires --baseline")
        if arguments.worker in ("populate", "reuse") and (
            arguments.cache_directory is None or arguments.key_map is None
        ):
            parser.error("cache workers require --cache-directory and --key-map")
        result = run_worker(arguments)
    else:
        if arguments.work_directory is None:
            parser.error("orchestrator mode requires --work-directory")
        result = run_orchestrator(arguments.work_directory)
    arguments.report.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
