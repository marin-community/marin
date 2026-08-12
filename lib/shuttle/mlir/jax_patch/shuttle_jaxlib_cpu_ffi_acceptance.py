# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Ordinary-JAX CPU typed-FFI acceptance for the 7x13 forward boundary."""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import jaxlib
import ml_dtypes
import numpy as np

from shuttle import ExecutionMode, Materialization, Numerics, Tuning, compiler_options

EXPECTED_VERSION = "0.10.1"
CACHE_HIT_EVENT = "/jax/compilation_cache/cache_hits"
CACHE_FILE = re.compile(r"^jit_row_fold_scale-[0-9a-f]{64}-cache$")
WORKERS = ("baseline", "populate", "reuse", "roundtrip")


class CacheHitCounter:
    def __init__(self) -> None:
        self.count = 0

    def observe(self, event: str, **metadata: str | int) -> None:
        del metadata
        if event == CACHE_HIT_EVENT:
            self.count += 1


def row_fold_scale(x: jax.Array, gamma: jax.Array) -> jax.Array:
    local = x.astype(jnp.float32)
    inverse = jax.lax.rsqrt(jnp.mean(local * local, axis=-1, keepdims=True) + 1e-5)
    return (local * inverse * gamma.astype(jnp.float32)).astype(jnp.bfloat16)


def _digest(value: np.ndarray) -> str:
    return hashlib.sha256(value.view(np.uint16).tobytes()).hexdigest()


def _arguments() -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(-0.75, 0.875, 7 * 13, dtype=np.float32).reshape(7, 13)
    gamma = np.linspace(-0.625, 1.0, 13, dtype=np.float32)
    return x.astype(ml_dtypes.bfloat16), gamma.astype(ml_dtypes.bfloat16)


def _tuning() -> Tuning:
    return Tuning(
        tile_sizes=(),
        cluster_shape=(),
        pipeline_stages=1,
        materialization=Materialization.AUTOMATIC,
        maximum_candidates=1,
    )


def _options(mode: ExecutionMode) -> dict[str, object]:
    return compiler_options(
        numerics=Numerics.SOURCE_ORDERED,
        execution_mode=mode,
        tuning=_tuning(),
    )


def _configure_cache(cache_directory: Path | None) -> None:
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


def _cache_snapshot(cache_directory: Path) -> dict[str, tuple[int, str]]:
    snapshot = {}
    for path in sorted(cache_directory.iterdir()):
        if path.is_symlink() or not path.is_file() or not CACHE_FILE.fullmatch(path.name):
            raise AssertionError(f"unexpected persistent-cache entry: {path.name}")
        snapshot[path.name] = (path.stat().st_size, hashlib.sha256(path.read_bytes()).hexdigest())
    return snapshot


def _require_versions() -> None:
    if jax.__version__ != EXPECTED_VERSION or jaxlib.__version__ != EXPECTED_VERSION:
        raise AssertionError("acceptance requires exact JAX and jaxlib 0.10.1")
    if jax.default_backend() != "cpu":
        raise AssertionError("acceptance requires the CPU backend")


def _execute(mode: ExecutionMode) -> np.ndarray:
    compiled = jax.jit(row_fold_scale, compiler_options=_options(mode))
    return np.asarray(compiled(*_arguments()).block_until_ready())


def _baseline(path: Path) -> dict[str, object]:
    _configure_cache(None)
    value = np.asarray(jax.jit(row_fold_scale)(*_arguments()).block_until_ready())
    np.save(path, value.view(np.uint16), allow_pickle=False)
    return {"worker": "baseline", "output_sha256": _digest(value)}


def _cache_worker(worker: str, cache_directory: Path, baseline_path: Path) -> dict[str, object]:
    _configure_cache(cache_directory)
    baseline_bits = np.load(baseline_path, allow_pickle=False)
    before = _cache_snapshot(cache_directory)
    counter = CacheHitCounter()
    jax.monitoring.register_event_listener(counter.observe)
    mode = ExecutionMode.STABLEHLO_ROUND_TRIP if worker == "roundtrip" else ExecutionMode.CPU_EXECUTABLE_BUNDLE
    value = _execute(mode)
    after = _cache_snapshot(cache_directory)
    if not np.array_equal(value.view(np.uint16), baseline_bits):
        raise AssertionError(f"{worker} result differs from ordinary-JAX baseline")
    if worker == "populate":
        if before or len(after) != 1 or counter.count:
            raise AssertionError("populate did not compile one CPU executable cache entry")
    elif worker == "reuse":
        if after != before or counter.count != 1:
            raise AssertionError("reuse did not deserialize one cache hit")
    elif worker == "roundtrip":
        if len(before) != 1 or len(after) != 2 or counter.count:
            raise AssertionError("execution-mode change did not produce a distinct cache miss")
    else:
        raise AssertionError(f"unknown cache worker {worker}")
    return {
        "worker": worker,
        "cache_files": sorted(after),
        "cache_hits": counter.count,
        "output_sha256": _digest(value),
    }


def _orchestrate(work_directory: Path) -> dict[str, object]:
    work_directory.mkdir(parents=True, exist_ok=True)
    if any(work_directory.iterdir()):
        raise AssertionError("acceptance work directory must start empty")
    baseline = work_directory / "baseline.npy"
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
    digests = {report["output_sha256"] for report in reports.values()}
    if len(digests) != 1:
        raise AssertionError("separate processes produced different outputs")
    if reports["populate"]["cache_files"] != reports["reuse"]["cache_files"]:
        raise AssertionError("reuse process changed the CPU executable cache entry")
    if len(reports["roundtrip"]["cache_files"]) != 2:
        raise AssertionError("round-trip execution mode did not use a distinct cache entry")
    return {"workers": reports}


def _single_process() -> dict[str, object]:
    _configure_cache(None)
    arguments = _arguments()
    baseline = np.asarray(jax.jit(row_fold_scale)(*arguments).block_until_ready())
    compiled = jax.jit(row_fold_scale, compiler_options=_options(ExecutionMode.CPU_EXECUTABLE_BUNDLE))
    first = np.asarray(compiled(*arguments).block_until_ready())
    second = np.asarray(compiled(*arguments).block_until_ready())
    if not np.array_equal(first, baseline) or not np.array_equal(second, baseline):
        raise AssertionError("typed-FFI result differs from ordinary-JAX baseline")
    return {"output_sha256": _digest(first)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-directory", type=Path)
    parser.add_argument("--worker", choices=WORKERS)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--cache-directory", type=Path)
    parser.add_argument("--report", type=Path)
    arguments = parser.parse_args()
    _require_versions()
    if arguments.worker:
        if arguments.baseline is None or arguments.report is None:
            parser.error("worker mode requires --baseline and --report")
        if arguments.worker == "baseline":
            result = _baseline(arguments.baseline)
        else:
            if arguments.cache_directory is None:
                parser.error("cache worker requires --cache-directory")
            result = _cache_worker(arguments.worker, arguments.cache_directory, arguments.baseline)
        arguments.report.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        return
    if arguments.work_directory is not None:
        result = _orchestrate(arguments.work_directory)
    else:
        result = _single_process()
    print(f"shuttle_cpu_ffi_acceptance=PASS output_sha256={result.get('output_sha256', 'multi-process')}")


if __name__ == "__main__":
    main()
