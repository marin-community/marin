# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CPU acceptance driver for a Shuttle-enabled pinned jaxlib wheel."""

import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import sys
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import jaxlib
import numpy as np
from acceptance_contract import (
    FIXTURE_EXPECTATIONS,
    FORWARD_EXPECTATION,
    VJP_EXPECTATION,
    FixtureExpectation,
    ObserverIdentity,
    decode_native_snapshot,
    match_fixture_contract,
    validate_success_events,
)
from jaxlib import _jax

from shuttle import Materialization, Numerics, Tuning, compiler_options

JAX_VERSION = "0.10.1"
JAXLIB_VERSION = "0.10.1"
CACHE_HIT_EVENT = "/jax/compilation_cache/cache_hits"
WORKER_MODES = ("baseline", "concurrency", "populate", "reuse")


@dataclass(frozen=True)
class Wrapper:
    label: str
    numerics: Numerics
    fixture: FixtureExpectation
    function: Callable[..., Any]
    arguments: tuple[np.ndarray, ...]
    baseline: tuple[np.ndarray, ...]


class CacheHitCounter:
    def __init__(self) -> None:
        self.count = 0

    def observe(self, event: str, **metadata: str | int) -> None:
        del metadata
        if event == CACHE_HIT_EVENT:
            self.count += 1


def tuning() -> Tuning:
    return Tuning(
        tile_sizes=(64, 128),
        cluster_shape=(2, 1, 1),
        pipeline_stages=3,
        materialization=Materialization.PREFER_FUSION,
        maximum_candidates=16,
    )


def reference_function(x: jax.Array, w0: jax.Array, w1: jax.Array) -> jax.Array:
    return jnp.tanh(x @ w0) @ w1


def reference_vjp(
    x: jax.Array,
    w0: jax.Array,
    w1: jax.Array,
    output_cotangent: jax.Array,
) -> tuple[jax.Array, ...]:
    _, pullback = jax.vjp(reference_function, x, w0, w1)
    return pullback(output_cotangent)


def fixed_inputs(shapes: Sequence[tuple[int, ...]]) -> tuple[np.ndarray, ...]:
    values = []
    for ordinal, shape in enumerate(shapes):
        size = int(np.prod(shape))
        start = np.float32(-0.75 + ordinal * 0.125)
        stop = np.float32(0.875 + ordinal * 0.125)
        values.append(np.linspace(start, stop, size, dtype=np.float32).reshape(shape))
    return tuple(values)


def fixture_arguments() -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
    forward = fixed_inputs(((2, 3), (3, 4), (4, 5)))
    return forward, forward + fixed_inputs(((2, 5),))


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
            raise AssertionError(f"{label}: enabled output is not bitwise equal to disabled JAX")
    return {
        "bitwise": True,
        "output_digests": [output_digest((value,)) for value in actual_arrays],
    }


def bridge_module() -> Any:
    bridge = getattr(_jax, "_shuttle_test_observer", None)
    if bridge is None:
        raise RuntimeError("jaxlib was not built with the test-only Shuttle observer bridge")
    return bridge


def decoded_events(capture: Any) -> tuple[dict[str, Any], ...]:
    return decode_native_snapshot(capture.snapshot())


def expected_identity(numerics: Numerics) -> ObserverIdentity:
    options = compiler_options(numerics=numerics, tuning=tuning())
    canonical = options["xla_shuttle_options"]
    if not isinstance(canonical, str):
        raise AssertionError("canonical options must be a string")
    payload = json.loads(canonical)
    canonical_tuning = json.dumps(payload["tuning"], sort_keys=True, separators=(",", ":"))
    return ObserverIdentity(
        policy=numerics.value,
        policy_digest=hashlib.sha256(canonical.encode()).hexdigest(),
        tuning_digest=hashlib.sha256(canonical_tuning.encode()).hexdigest(),
        canonical_options=canonical,
        canonical_tuning=canonical_tuning,
    )


def validate_event_group(
    events: Sequence[dict[str, Any]],
    numerics: Numerics,
    fixture: FixtureExpectation,
) -> dict[str, Any]:
    return validate_success_events(events, expected_identity(numerics), fixture)


def grouped_event_evidence(events: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        groups[event["invocation_id"]].append(event)
    evidence: list[dict[str, Any]] = []
    observed_contracts: set[tuple[str, str]] = set()
    for invocation_id, group in groups.items():
        del invocation_id
        numerics = Numerics(group[0]["policy"])
        match = match_fixture_contract(group, expected_identity(numerics))
        contract = (match["policy"], match["fixture"])
        if contract in observed_contracts:
            raise AssertionError("concurrent acceptance emitted a duplicate policy/fixture contract")
        observed_contracts.add(contract)
        evidence.append(match)
    expected_contracts = {(numerics.value, fixture.name) for numerics in Numerics for fixture in FIXTURE_EXPECTATIONS}
    if observed_contracts != expected_contracts:
        raise AssertionError("concurrent acceptance omitted an audited policy/fixture contract")
    return sorted(evidence, key=lambda item: item["invocation_id"])


def cache_files(cache_directory: Path) -> frozenset[str]:
    return frozenset(path.name for path in cache_directory.glob("*-cache"))


def configure_cache(cache_directory: Path | None) -> None:
    if cache_directory is None:
        jax.config.update("jax_enable_compilation_cache", False)
        return
    cache_directory.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(cache_directory))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_compilation_cache_max_size", -1)


def save_baselines(path: Path) -> dict[str, Any]:
    configure_cache(None)
    forward_arguments, vjp_arguments = fixture_arguments()
    forward = arrays(ready(jax.jit(reference_function)(*forward_arguments)))
    vjp = arrays(ready(jax.jit(reference_vjp)(*vjp_arguments)))
    np.savez(path, forward=forward[0], dx=vjp[0], dw0=vjp[1], dw1=vjp[2])
    return {
        "forward_digest": output_digest(forward),
        "vjp_digests": [output_digest((value,)) for value in vjp],
    }


def load_baselines(path: Path) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
    with np.load(path) as stored:
        return (stored["forward"].copy(),), (
            stored["dx"].copy(),
            stored["dw0"].copy(),
            stored["dw1"].copy(),
        )


def make_wrappers(baseline_path: Path) -> tuple[Wrapper, ...]:
    forward_baseline, vjp_baseline = load_baselines(baseline_path)
    forward_arguments, vjp_arguments = fixture_arguments()
    return tuple(
        wrapper
        for numerics in (Numerics.SOURCE_ORDERED, Numerics.FAST)
        for wrapper in (
            Wrapper(
                "forward_" + numerics.value,
                numerics,
                FORWARD_EXPECTATION,
                reference_function,
                forward_arguments,
                forward_baseline,
            ),
            Wrapper(
                "vjp_" + numerics.value,
                numerics,
                VJP_EXPECTATION,
                reference_vjp,
                vjp_arguments,
                vjp_baseline,
            ),
        )
    )


def compile_wrapper(wrapper: Wrapper) -> tuple[Any, dict[str, Any]]:
    compiled = jax.jit(
        wrapper.function,
        compiler_options=compiler_options(numerics=wrapper.numerics, tuning=tuning()),
    )
    result = ready(compiled(*wrapper.arguments))
    return result, require_bitwise(result, wrapper.baseline, wrapper.label)


def run_concurrency_worker(baseline_path: Path) -> dict[str, Any]:
    configure_cache(None)
    wrappers = make_wrappers(baseline_path)
    capture = bridge_module().subscribe()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(compile_wrapper, wrappers))
    capture.close()
    capture.close()
    retained = decoded_events(capture)
    evidence = grouped_event_evidence(retained)
    if len(evidence) != 4 or len({item["invocation_id"] for item in evidence}) != 4:
        raise AssertionError("four concurrent wrappers did not produce four unique invocations")

    new_arguments = fixed_inputs(((3, 2), (2, 6), (6, 4)))
    compiled_after_close = jax.jit(
        reference_function,
        compiler_options=compiler_options(numerics=Numerics.SOURCE_ORDERED, tuning=tuning()),
    )
    ready(compiled_after_close(*new_arguments))
    if decoded_events(capture) != retained:
        raise AssertionError("closed capture changed after a new-shape compilation")
    return {
        "invocations": evidence,
        "outputs": {wrapper.label: result[1] for wrapper, result in zip(wrappers, results, strict=True)},
        "records_retained_after_close": True,
    }


def run_cache_worker(
    mode: str,
    cache_directory: Path,
    baseline_path: Path,
    key_map_path: Path,
) -> dict[str, Any]:
    configure_cache(cache_directory)
    wrappers = make_wrappers(baseline_path)
    counter = CacheHitCounter()
    jax.monitoring.register_event_listener(counter.observe)
    expected_keys = json.loads(key_map_path.read_text()) if mode == "reuse" else {}
    label_to_key: dict[str, str] = {}
    reports = []
    bridge = bridge_module()
    initial_files = cache_files(cache_directory)
    if mode == "populate" and initial_files:
        raise AssertionError("populate cache must start empty")
    if mode == "reuse" and initial_files != frozenset(expected_keys.values()):
        raise AssertionError("reuse worker did not start with the four attributed cache files")

    for wrapper in wrappers:
        compiled = jax.jit(
            wrapper.function,
            compiler_options=compiler_options(numerics=wrapper.numerics, tuning=tuning()),
        )
        with bridge.subscribe() as capture:
            files_before = cache_files(cache_directory)
            hits_before = counter.count
            first = ready(compiled(*wrapper.arguments))
            events_after_first = decoded_events(capture)
            files_after_first = cache_files(cache_directory)
            hits_after_first = counter.count
            second = ready(compiled(*wrapper.arguments))
            events_after_second = decoded_events(capture)
            files_after_second = cache_files(cache_directory)
        metrics = require_bitwise(first, wrapper.baseline, wrapper.label)
        require_bitwise(second, wrapper.baseline, wrapper.label + " second call")

        if mode == "populate":
            added = files_after_first - files_before
            if len(added) != 1:
                raise AssertionError(f"{wrapper.label}: first call did not add exactly one cache file")
            key = next(iter(added))
            label_to_key[wrapper.label] = key
            evidence = validate_event_group(events_after_first, wrapper.numerics, wrapper.fixture)
            if events_after_second != events_after_first or hits_after_first != hits_before:
                raise AssertionError(f"{wrapper.label}: same jitted object recompiled")
        else:
            key = expected_keys[wrapper.label]
            label_to_key[wrapper.label] = key
            evidence = None
            if events_after_first or events_after_second:
                raise AssertionError(f"{wrapper.label}: persistent-cache hit invoked Shuttle")
            if hits_after_first - hits_before != 1 or counter.count != hits_after_first:
                raise AssertionError(f"{wrapper.label}: expected one public persistent-cache hit")
        if files_after_second != files_after_first:
            raise AssertionError(f"{wrapper.label}: second call changed cache files")
        if mode == "reuse" and files_after_second != files_before:
            raise AssertionError(f"{wrapper.label}: cache hit added a file")
        reports.append(
            {
                "label": wrapper.label,
                "cache_file": key,
                "cache_hit_delta": hits_after_first - hits_before,
                "observer": evidence,
                "output": metrics,
            }
        )

    final_files = cache_files(cache_directory)
    if len(final_files) != 4 or final_files != frozenset(label_to_key.values()):
        raise AssertionError("cache workers did not preserve four distinct attributed files")
    if mode == "populate":
        key_map_path.write_text(json.dumps(label_to_key, indent=2, sort_keys=True) + "\n")
    elif counter.count != 4:
        raise AssertionError("reuse worker did not report four public cache hits")
    return {
        "mode": mode,
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
    if arguments.worker == "concurrency":
        return run_concurrency_worker(arguments.baseline)
    return run_cache_worker(
        arguments.worker,
        arguments.cache_directory,
        arguments.baseline,
        arguments.key_map,
    )


def run_orchestrator(work_directory: Path) -> dict[str, Any]:
    work_directory.mkdir(parents=True, exist_ok=True)
    if any(work_directory.iterdir()):
        raise RuntimeError("acceptance work directory must start empty")
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
    populate = reports["populate"]
    reuse = reports["reuse"]
    if populate["label_to_cache_file"] != reuse["label_to_cache_file"]:
        raise AssertionError("reuse worker did not preserve label-to-key attribution")
    if populate["cache_files"] != reuse["cache_files"] or reuse["public_cache_hits"] != 4:
        raise AssertionError("second cache process did not reuse the same four entries")
    return reports


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
