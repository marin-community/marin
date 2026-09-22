# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Correctness-gated, single-device GDN-2 kernel benchmark and bounded tile sweep.

Example on each TPU generation (configure LIBTPU_INIT_ARGS before launch)::

    uv run --package marin-levanter python lib/levanter/scripts/bench/bench_atomic_gdn2.py \
        --shape 1,256,1,128 --shape 4,4096,6,128 --dtype float32 --dtype bfloat16 \
        --bt 128 --bt 256 --mb 16 --mb 32 --output /tmp/gdn2.jsonl

All inputs reside on one explicitly selected device. This measures local kernel
work, not model training, distributed throughput, or accelerator utilization.
Compile and execution failures are recorded; failed parity never produces an
accepted timing. Output rows use seconds and include the exact numerical gate.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import itertools
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import tempfile
import time
import traceback
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from levanter.kernels.pallas import gdn2
from levanter.kernels.pallas.gdn2.reference import gdn2_reference

FORWARD_NAMES = ("output", "final_state")
GRADIENT_NAMES = ("loss", "dq", "dk", "dv", "dw", "db", "dg", "dh0")


def token_reference(q, k, v, w, b, g, h0):
    """Evaluate the unsanitized recurrence, promoting operands before arithmetic."""
    batch, length, heads, dim = q.shape
    # Rematerialize short groups so reference gradients do not retain a full
    # D*D state for every token of a production-length input.
    group = min(32, length)
    if length % group:
        raise ValueError("Reference sequence length must be divisible by 32 or smaller than 32")
    chunks = tuple(
        x.reshape(batch, length // group, group, heads, dim).transpose(1, 0, 2, 3, 4) for x in (q, k, v, w, b, g)
    )

    @jax.checkpoint
    def chunk(state, inputs):
        output, final_state = gdn2_reference(*inputs, dim**-0.5, state)
        return final_state, output

    final_state, out = jax.lax.scan(chunk, h0.astype(jnp.float32), chunks)
    out = out.transpose(1, 0, 2, 3, 4).reshape(batch, length, heads, dim)
    # Both Pallas implementations return FP32 even for BF16 inputs. Casting
    # here would also quantize the output cotangent before reference backprop.
    return out, final_state


def make_inputs(shape, dtype, seed, decay):
    keys = jax.random.split(jax.random.key(seed), 9)
    q = jax.random.normal(keys[0], shape)
    k = jax.random.normal(keys[1], shape)
    q /= jnp.linalg.norm(q, axis=-1, keepdims=True)
    k /= jnp.linalg.norm(k, axis=-1, keepdims=True)
    v = jax.random.normal(keys[2], shape)
    w = jax.nn.sigmoid(jax.random.normal(keys[3], shape))
    b = jax.nn.sigmoid(jax.random.normal(keys[4], shape))
    g = -decay * jax.random.uniform(keys[5], shape)
    state_shape = (shape[0], shape[2], shape[3], shape[3])
    h0 = 0.01 * jax.random.normal(keys[6], state_shape)
    inputs = tuple(x.astype(dtype) for x in (q, k, v, w, b, g)) + (h0,)
    cotangents = (jax.random.normal(keys[7], shape), jax.random.normal(keys[8], state_shape))
    return inputs, cotangents


def scalar_loss(function, cotangents):
    def loss(*inputs):
        output, state = function(*inputs)
        return jnp.sum(output.astype(jnp.float32) * cotangents[0]) + jnp.sum(state * cotangents[1])

    return loss


def reference_results(inputs, cotangents, device):
    """Copy exact input bits to the oracle device and return host results."""
    with jax.default_device(device):
        local_inputs = jax.device_put(jax.device_get(inputs), device)
        local_cotangents = jax.device_put(jax.device_get(cotangents), device)
        output = jax.jit(token_reference)(*local_inputs)
        gradient = jax.value_and_grad(scalar_loss(token_reference, local_cotangents), argnums=tuple(range(7)))
        gradients = jax.jit(gradient)(*local_inputs)
        return jax.device_get((output, gradients))


def deviations(actual, expected, names, atol, rtol):
    metrics = {}
    passed = True
    for name, value, reference in zip(names, jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        value = np.asarray(value, dtype=np.float32)
        reference = np.asarray(reference, dtype=np.float32)
        finite = bool(np.isfinite(value).all() and np.isfinite(reference).all())
        difference = np.abs(value - reference)
        bound = atol + rtol * np.abs(reference)
        match = finite and bool(np.all(difference <= bound))
        worst_index = np.unravel_index(np.argmax(difference / bound), difference.shape) if finite else None
        metrics[name] = {
            "finite": finite,
            "passed": match,
            "max_abs": float(difference.max()) if finite else None,
            "mean_abs": float(difference.mean()) if finite else None,
            "reference_max_abs": float(np.abs(reference).max()) if finite else None,
            "max_tolerance_ratio": float(np.max(difference / bound)) if finite else None,
            "violations": int(np.count_nonzero(difference > bound)) if finite else None,
            "worst_index": [int(i) for i in worst_index] if finite else None,
            "worst_actual": float(value[worst_index]) if finite else None,
            "worst_reference": float(reference[worst_index]) if finite else None,
        }
        passed = passed and match
    return passed, metrics


def compile_call(function, inputs):
    started = time.perf_counter()
    executable = jax.jit(function).lower(*inputs).compile()
    compile_time = time.perf_counter() - started
    started = time.perf_counter()
    result = jax.block_until_ready(executable(*inputs))
    first_execution = time.perf_counter() - started
    return executable, result, compile_time, first_execution


def measure(executable, inputs, warmup, repetitions):
    for _ in range(warmup):
        jax.block_until_ready(executable(*inputs))
    timings = []
    for _ in range(repetitions):
        started = time.perf_counter()
        jax.block_until_ready(executable(*inputs))
        timings.append(time.perf_counter() - started)
    return timings


def parse_shape(value):
    shape = tuple(map(int, value.split(",")))
    if len(shape) != 4 or min(shape) < 1 or shape[-1] != 128:
        raise argparse.ArgumentTypeError("Expected positive B,L,H,128")
    return shape


def tuning_key(context):
    return hashlib.sha256(json.dumps(context, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_tuning_cache(path):
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    if payload["version"] != 1 or not isinstance(payload["entries"], dict):
        raise ValueError(f"Unsupported GDN-2 tuning cache: {path}")
    return payload["entries"]


def save_tuning_cache(path, entries):
    """Atomically replace the optional single-writer benchmark cache."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=path.name, delete=False) as stream:
        json.dump({"version": 1, "entries": entries}, stream, indent=2, allow_nan=False)
        stream.write("\n")
        temporary = stream.name
    os.replace(temporary, path)


def tuning_order(entries, context, configs):
    entry = entries.get(tuning_key(context))
    if entry is None:
        return configs, None
    tile = (entry["block_sizes"]["bt"], entry["block_sizes"]["mb"])
    if tile not in configs:
        raise ValueError("Cached GDN-2 tile is outside its bounded configuration space")
    return [tile] + [config for config in configs if config != tile], tile


def tuning_winner(rows):
    """Select only a complete freshly validated forward/gradient measurement pair."""
    pairs = {}
    for row in rows:
        tile = (row["block_sizes"]["bt"], row["block_sizes"]["mb"])
        pairs.setdefault(tile, []).append(row)
    valid = []
    for pair in pairs.values():
        if len(pair) != 2 or {row["mode"] for row in pair} != {"forward", "forward_backward"}:
            continue
        for row in pair:
            names = FORWARD_NAMES if row["mode"] == "forward" else GRADIENT_NAMES
            if (
                row.get("error") is not None
                or not row.get("configuration_correctness_passed", False)
                or not row.get("timing_accepted", False)
                or any(not row.get("correctness", {}).get(name, {}).get("passed", False) for name in names)
                or not math.isfinite(row["steady_state_time"])
                or row["steady_state_time"] <= 0
            ):
                break
        else:
            valid.append(next(row for row in pair if row["mode"] == "forward_backward"))
    return min(valid, key=lambda row: row["steady_state_time"], default=None)


def tuning_trials(entries, context, configs, measured):
    """Try a hit first; only a fresh accepted pair can suppress the bounded sweep."""
    order, cached = tuning_order(entries, context, configs)
    for tile in order:
        yield tile
        if tile == cached and tuning_winner(measured) is not None:
            return


def update_tuning_entry(entries, context, rows, results_path):
    winner = tuning_winner(rows)
    key = tuning_key(context)
    if winner is None:
        entries.pop(key, None)
        return
    entries[key] = {
        "context": context,
        "block_sizes": winner["block_sizes"],
        "objective_time": winner["steady_state_time"],
        "results_path": str(results_path.resolve()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", type=parse_shape, action="append")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), action="append")
    parser.add_argument("--implementation", choices=("upstream", "candidate"), action="append")
    parser.add_argument("--bt", type=int, choices=(128, 256), action="append")
    parser.add_argument("--mb", type=int, choices=(16, 32), action="append")
    parser.add_argument("--seed", type=int, action="append")
    parser.add_argument("--decay", type=float, action="append", help="Maximum negative log gate magnitude")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--device-type", help="Required substring of actual device kind, e.g. TPU v5p")
    parser.add_argument(
        "--reference-device",
        choices=("current", "cpu"),
        default="current",
        help="CPU oracle additionally records TPU-versus-CPU reference discrepancies; requires JAX_PLATFORMS=tpu,cpu",
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--output", type=Path, help="New JSONL file; defaults to IRIS_OUTPUT_DIR or /tmp")
    parser.add_argument(
        "--tuning-cache", type=Path, help="Optional single-writer tile cache; hits are freshly validated"
    )
    args = parser.parse_args()
    if args.repetitions < 1 or args.warmup < 0:
        parser.error("Invalid repetitions or warmup")
    device = jax.local_devices()[args.device_index]
    if device.platform != "tpu":
        parser.error("Pallas benchmark requires TPU; CPU fallback timings are not kernel evidence")
    if args.device_type and args.device_type.lower() not in device.device_kind.lower():
        parser.error(f"Expected {args.device_type}, found {device.device_kind}")
    reference_device = jax.local_devices(backend="cpu")[0] if args.reference_device == "cpu" else device
    shapes = args.shape or [(1, 256, 1, 128)]
    dtypes = args.dtype or ["float32", "bfloat16"]
    implementations = args.implementation or ["upstream", "candidate"]
    configs = list(itertools.product(args.bt or [256], args.mb or [16]))
    seeds = args.seed or [0]
    decays = args.decay or [0.01]
    if any(decay < 0 for decay in decays):
        parser.error("Decay must be nonnegative")
    git_sha = os.environ.get("GDN2_GIT_SHA")
    git_dirty = None
    if git_sha is None:
        revision = subprocess.run(["git", "rev-parse", "HEAD"], text=True, capture_output=True, check=False)
        if revision.returncode == 0:
            git_sha = revision.stdout.strip()
            git_dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    source_digest = hashlib.sha256()
    source_root = Path(gdn2.__file__).parent
    for source in sorted(source_root.rglob("*.py")):
        source_digest.update(str(source.relative_to(source_root)).encode())
        source_digest.update(source.read_bytes())
    source_digest.update(Path(__file__).read_bytes())
    source_sha256 = source_digest.hexdigest()
    output_path = args.output or Path(os.environ.get("IRIS_OUTPUT_DIR", "/tmp")) / f"gdn2-{time.time_ns()}.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    environment = {
        name: os.environ.get(name, "")
        for name in ("LIBTPU_INIT_ARGS", "JAX_PLATFORMS", "JAX_DEFAULT_MATMUL_PRECISION", "GDN2_FWD_DIAG")
    }
    tuning_cache = load_tuning_cache(args.tuning_cache) if args.tuning_cache else {}
    failures = 0
    # A new output file prevents accidental mixing of runs or duplicate keys.
    with output_path.open("x") as output_file, jax.default_device(device):
        for shape, dtype, seed, decay in itertools.product(shapes, dtypes, seeds, decays):
            tolerance = 1e-2 if dtype == "bfloat16" else 1e-4
            inputs, cotangents = make_inputs(shape, jnp.dtype(dtype), seed, decay)
            inputs = jax.device_put(inputs, device)
            cotangents = jax.device_put(cotangents, device)
            ref_out, ref_grad = reference_results(inputs, cotangents, reference_device)
            reference_discrepancies = None
            if args.reference_device == "cpu":
                device_out, device_grad = reference_results(inputs, cotangents, device)
                forward_passed, forward_errors = deviations(device_out, ref_out, FORWARD_NAMES, tolerance, tolerance)
                backward_passed, backward_errors = deviations(
                    device_grad, ref_grad, GRADIENT_NAMES, tolerance, tolerance
                )
                reference_discrepancies = {
                    "actual_backend": device.platform,
                    "expected_backend": reference_device.platform,
                    "forward": {"passed": forward_passed, "errors": forward_errors},
                    "forward_backward": {"passed": backward_passed, "errors": backward_errors},
                }
            contexts = {
                implementation: {
                    "source_sha256": source_sha256,
                    "jax_version": jax.__version__,
                    "device_type": device.device_kind,
                    "device_count": 1,
                    "shape": shape,
                    "dtype": dtype,
                    "implementation": implementation,
                    "backend_env": environment,
                    "xla_flags": os.environ.get("XLA_FLAGS", ""),
                    "configs": sorted(configs),
                    "objective": "forward_backward",
                    "seed": seed,
                    "decay": decay,
                    "reference_backend": reference_device.platform,
                    "atol": tolerance,
                    "rtol": tolerance,
                    "repetitions": args.repetitions,
                    "warmup": args.warmup,
                }
                for implementation in implementations
            }
            orders = {impl: tuning_order(tuning_cache, contexts[impl], configs) for impl in implementations}
            measured = {impl: [] for impl in implementations}
            scheduled = (
                (impl, tile)
                for impl in implementations
                for tile in tuning_trials(tuning_cache, contexts[impl], configs, measured[impl])
            )
            for implementation, (bt, mb) in scheduled:
                base: dict[str, Any] = {
                    "kernel": "gdn2",
                    "implementation": implementation,
                    "shape": shape,
                    "dtype": dtype,
                    "backend": device.platform,
                    "reference_backend": reference_device.platform,
                    "reference_device_type": reference_device.device_kind,
                    "reference_cross_backend": reference_discrepancies,
                    "device_type": device.device_kind,
                    "device_count": 1,
                    "available_device_count": jax.device_count(),
                    "block_sizes": {"bt": bt, "bc": bt // 2, "mb": mb},
                    "git_sha": git_sha,
                    "git_dirty": git_dirty,
                    "source_sha256": source_sha256,
                    "xla_flags": os.environ.get("XLA_FLAGS", ""),
                    "backend_env": environment,
                    "jax_version": jax.__version__,
                    "seed": seed,
                    "decay": decay,
                    "atol": tolerance,
                    "rtol": tolerance,
                    "wy_eps": 0.0,
                    "compile_time": None,
                    "steady_state_time": None,
                    "error": None,
                }
                if args.tuning_cache:
                    base["tuning_cache_key"] = tuning_key(contexts[implementation])
                    base["tuning_cache_status"] = (
                        "revalidate_hit"
                        if orders[implementation][1] == (bt, mb)
                        else "fallback_after_failed_hit" if orders[implementation][1] is not None else "miss"
                    )
                try:
                    module_name = f"levanter.kernels.pallas.gdn2.{implementation}"
                    config_module = importlib.import_module(module_name + ".configs")
                    config_type = config_module.KernelConfig
                    kernel = importlib.import_module(module_name + ".gdn2_pipeline").gdn2_pallas_forward_trainable
                    if implementation == "candidate":
                        layout = (
                            config_module.ScoreLayout.FEATURE_FIRST
                            if "tpu v4" in device.device_kind.lower()
                            else config_module.ScoreLayout.FEATURE_LAST
                        )
                        config = config_type(bt=bt, bc=bt // 2, mb=mb, wy_eps=0.0, score_layout=layout)
                        base["score_layout"] = layout.value
                    else:
                        config = config_type(bt=bt, bc=bt // 2, mb=mb, wy_eps=0.0)
                        base["score_layout"] = "upstream"

                    def forward(q, k, v, w, b, g, h0):
                        return kernel(q, k, v, w, b, g, shape[-1] ** -0.5, h0=h0, config=config)

                    backward = jax.value_and_grad(scalar_loss(forward, cotangents), argnums=tuple(range(7)))
                    checked = []
                    for mode, function, expected, names in (
                        ("forward", forward, ref_out, FORWARD_NAMES),
                        ("forward_backward", backward, ref_grad, GRADIENT_NAMES),
                    ):
                        row = dict(base, mode=mode)
                        executable = None
                        try:
                            executable, result, compile_time, first = compile_call(function, inputs)
                            row.update(
                                compile_time=compile_time,
                                first_execution_time=first,
                                time_to_first_result=compile_time + first,
                            )
                            passed, errors = deviations(result, expected, names, tolerance, tolerance)
                            row.update(correctness=errors, correctness_passed=passed)
                            if not passed:
                                row["error"] = "correctness_failure"
                                failures += 1
                            del result
                        except Exception as exc:
                            row.update(error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
                            failures += 1
                        checked.append((row, executable))
                    eligible = all(row.get("correctness_passed", False) and row["error"] is None for row, _ in checked)
                    for row, executable in checked:
                        row["configuration_correctness_passed"] = eligible
                        row["timing_accepted"] = False
                        if eligible:
                            try:
                                timings = measure(executable, inputs, args.warmup, args.repetitions)
                                row.update(
                                    steady_state_time=statistics.median(timings), samples=timings, timing_accepted=True
                                )
                            except Exception as exc:
                                row.update(error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
                                failures += 1
                        elif row["error"] is None:
                            row["error"] = "paired_mode_correctness_failure"
                        output_file.write(json.dumps(row, allow_nan=False) + "\n")
                        output_file.flush()
                        print(json.dumps(row, allow_nan=False), flush=True)
                        measured[implementation].append(row)
                    del checked
                except Exception as exc:
                    row = dict(
                        base, mode="setup", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc()
                    )
                    output_file.write(json.dumps(row, allow_nan=False) + "\n")
                    output_file.flush()
                    print(json.dumps(row, allow_nan=False), flush=True)
                    measured[implementation].append(row)
                    failures += 1
            if args.tuning_cache:
                for implementation in implementations:
                    update_tuning_entry(tuning_cache, contexts[implementation], measured[implementation], output_path)
                save_tuning_cache(args.tuning_cache, tuning_cache)
            jax.clear_caches()
    if failures:
        raise SystemExit(f"{failures} benchmark configurations failed; see {output_path}")


if __name__ == "__main__":
    main()
