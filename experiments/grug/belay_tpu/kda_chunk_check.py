# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare larger KDA chunks against the saved-state C=16 path without updates."""

import argparse
import functools
import json
import os
import time

import jax
import jax.numpy as jnp
import levanter.tracker as tracker
import numpy as np

from experiments.grug.belay_tpu.kda_tpu import chunk_kda_tpu
from experiments.grug.belay_tpu.kernel_check import evaluate, measure_errors
from experiments.grug.belay_tpu.operator_check import initialize_operator_check
from experiments.grug.moe.kda import recurrent_kda


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--saved-solve", action="store_true")
    parser.add_argument("--fused-prep", action="store_true")
    parser.add_argument("--state-heads", action="store_true")
    parser.add_argument("--fused-recurrence", action="store_true")
    args = parser.parse_args()
    if args.fused_recurrence and (args.state_heads or not (args.saved_solve and args.fused_prep)):
        parser.error("Fused recurrence comparison requires --saved-solve --fused-prep without --state-heads")
    if args.state_heads and not (args.saved_solve and args.fused_prep):
        parser.error("Head batching comparison requires --saved-solve --fused-prep")
    if args.fused_prep and not args.saved_solve:
        parser.error("Fused preparation requires --saved-solve")
    output, operator_topology, reference_topology = initialize_operator_check()
    if args.state_heads:
        variants = [(16, 0, False, h) for h in (1, 4)] + [
            (c, 4, fused, h) for c in (16, 32, 64) for fused in (False, True) for h in (1, 4)
        ]
    elif args.fused_prep:
        variants = [(16, 0, False, 1)] + [
            (c, group, fused, 1) for c in (16, 32, 64) for group in (1, 4) for fused in (False, True)
        ]
    elif args.saved_solve:
        variants = [(16, 0, False, 1)] + [(c, group, False, 1) for c in (16, 32, 64) for group in (1, 4)]
    else:
        variants = [(c, 0, False, 1) for c in (16, 32, 64, 128)]
    variants = [(*variant, False) for variant in variants]
    if args.fused_recurrence:
        variants = [(16, 0, False, 1, False)] + [
            (c, h, True, h, combined) for c in (16, 32, 64) for h in (1, 4) for combined in (False, True)
        ]
    seed = 432 if args.fused_recurrence else 431 if args.state_heads else 426
    tracker.log_summary(
        {
            "kernel_check/input_seed": seed,
            "kernel_check/saved_solve": args.saved_solve,
            "kernel_check/fused_prep": args.fused_prep,
            "kernel_check/state_heads": args.state_heads,
            "kernel_check/fused_recurrence": args.fused_recurrence,
        }
    )
    failures = []
    cases = (("packed", 57), ("strong_decay", 128), ("long", 128), ("target", 4096), ("weak_decay", 4096))
    with jax.default_device(jax.local_devices()[0]), jax.default_matmul_precision("highest"):
        for case, length in cases:
            shape = (
                (8, 4, length, 128)
                if length == 4096
                else (1, 4 if args.state_heads or args.fused_recurrence else 2, length, 128)
            )
            for dtype, tolerance in ((jnp.float32, 1e-4), (jnp.bfloat16, 2e-2)):
                if length == 4096 and dtype == jnp.float32:
                    continue
                rng = np.random.default_rng(seed)
                q, k, v = (jnp.asarray(rng.normal(size=shape), dtype) for _ in range(3))
                gate_scale = 5 if case == "strong_decay" else 0.001 if case == "weak_decay" else 0.1
                decay = jnp.asarray(-gate_scale * rng.random(shape), jnp.float32)
                beta = jnp.asarray(rng.random(shape[:-1]), jnp.float32)
                initial = jnp.asarray(0.01 * rng.normal(size=(*shape[:2], 128, 128)), jnp.float32)
                erase = jnp.asarray(0.5 + rng.random(shape), jnp.float32)
                if case in ("long", "target", "weak_decay"):
                    initial, erase = jnp.zeros_like(initial), jnp.ones_like(erase)
                ids = np.searchsorted([7, 16, 17, 33, 56], np.arange(length), side="right")
                if case in ("long", "weak_decay"):
                    ids = np.zeros(length, np.int32)
                segments = jnp.asarray(ids[None, None, :], jnp.int32)
                inputs = (q, k, v, decay, beta, initial, erase)
                cotangents = (
                    jnp.asarray(rng.normal(size=shape), jnp.float32),
                    jnp.asarray(rng.normal(size=initial.shape), jnp.float32),
                )
                expected = None
                head_reference = None
                separate_reference = None
                oracle = (
                    jax.jit(functools.partial(evaluate, recurrent_kda))(inputs, segments, cotangents)
                    if dtype == jnp.float32
                    else None
                )
                precision = jax.lax.Precision.HIGHEST if dtype == jnp.float32 else jax.lax.Precision.HIGH
                for chunk_size, solve_groups, fused_prep, state_head_group, fused_recurrence in variants:
                    kernel = functools.partial(
                        chunk_kda_tpu,
                        matmul_dtype=dtype,
                        state_precision=precision,
                        chunk_size=chunk_size,
                        solve_groups=solve_groups,
                        fused_prep=fused_prep,
                        fused_recurrence=fused_recurrence,
                        state_head_group=state_head_group,
                    )
                    function = jax.jit(functools.partial(evaluate, kernel))
                    start = time.perf_counter()
                    compiled = function.lower(inputs, segments, cotangents).compile()
                    compile_seconds = time.perf_counter() - start
                    actual = jax.block_until_ready(compiled(inputs, segments, cotangents))
                    if expected is None:
                        expected = actual
                    if state_head_group == 1:
                        head_reference = actual
                    head_errors = measure_errors(actual, head_reference, tolerance) if args.state_heads else None
                    if not fused_recurrence:
                        separate_reference = actual
                    separate_errors = (
                        measure_errors(actual, separate_reference, tolerance) if args.fused_recurrence else None
                    )
                    deviations = measure_errors(actual, expected, tolerance)
                    oracle_errors = measure_errors(actual, oracle, tolerance) if oracle is not None else None
                    passed = all(
                        np.isfinite(row["normalized_max"]) and row["normalized_max"] <= 1
                        for row in [*deviations, *(oracle_errors or []), *(head_errors or []), *(separate_errors or [])]
                    )
                    if not passed:
                        failures.append(
                            (case, str(dtype), chunk_size, solve_groups, fused_prep, state_head_group, fused_recurrence)
                        )
                    samples = []
                    for _ in range(9):
                        start = time.perf_counter()
                        jax.block_until_ready(compiled(inputs, segments, cotangents))
                        samples.append(time.perf_counter() - start)
                    row = {
                        "stage": "kda_chunk_comparison",
                        "kernel": "chunk_kda_tpu",
                        "implementation": (
                            f"chunk_{chunk_size}_solve_{solve_groups}"
                            + ("_fused" if fused_prep else "")
                            + (f"_heads_{state_head_group}" if args.state_heads or args.fused_recurrence else "")
                            + ("_combined" if fused_recurrence else "")
                            if args.saved_solve
                            else f"chunk_{chunk_size}"
                        ),
                        "solve_groups": solve_groups,
                        "fused_prep": fused_prep,
                        "fused_recurrence": fused_recurrence,
                        "state_head_group": state_head_group,
                        "state_precision": precision.name,
                        "case": case,
                        "shape": shape,
                        "dtype": str(dtype),
                        "backend": jax.default_backend(),
                        "device_type": jax.local_devices()[0].device_kind,
                        "device_count": 1,
                        "rank": jax.process_index(),
                        "operator_topology": operator_topology,
                        "reference_topology": reference_topology,
                        "block_sizes": {
                            "chunk": chunk_size,
                            "state_tile": 1,
                            "decay_tile": 16 if chunk_size == 16 or solve_groups else 32,
                        },
                        "source_parity_pass": passed,
                        "errors": deviations,
                        "oracle_errors": oracle_errors,
                        "one_head_errors": head_errors,
                        "separate_preparation_errors": separate_errors,
                        "tolerance": tolerance,
                        "compile_time": compile_seconds,
                        "steady_state_time": float(np.median(samples[2:])),
                        "samples": samples,
                        "memory": str(compiled.memory_analysis()),
                        "git_sha": os.environ["BELAY_SOURCE_COMMIT"],
                        "xla_flags": os.environ.get("XLA_FLAGS", ""),
                        "backend_env": {"LIBTPU_INIT_ARGS": os.environ.get("LIBTPU_INIT_ARGS", "")},
                    }
                    print(json.dumps(row), flush=True)
                    with (output / "kda-chunk-check.jsonl").open("a") as file:
                        file.write(json.dumps(row) + "\n")
    tracker.log_summary({"kernel_check/kda_chunk_pass": not failures})
    tracker.current_tracker().finish()
    if failures:
        raise AssertionError(f"KDA chunk failures: {failures}")


if __name__ == "__main__":
    main()
