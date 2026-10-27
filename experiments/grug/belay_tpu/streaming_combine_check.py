# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Measure direct route accumulation and expert-order pullbacks on TPU."""

import argparse
import functools
import json
import os
import time

import jax
import jax.numpy as jnp
import levanter.tracker as tracker
import numpy as np

from experiments.grug.belay_tpu.kernel_check import measure_errors
from experiments.grug.belay_tpu.local_experts import combine_local
from experiments.grug.belay_tpu.local_movement import combine_expert_order
from experiments.grug.belay_tpu.operator_check import initialize_operator_check
from experiments.grug.belay_tpu.streaming_combine import combine_streaming, sum_routes, sum_routes_blocked


def evaluate(
    implementation, mode, rows, order, inverse, weights, gradient, *, token_tile=0, backward_tile=0, output_dtype=None
):
    if mode == "dispatch_backward":
        positions = inverse.reshape(weights.shape)
        if token_tile:
            return (sum_routes_blocked(rows, positions, tile_tokens=token_tile),)
        if implementation == "source":
            return (rows[positions].astype(jnp.float32).sum(axis=1).astype(rows.dtype),)
        return (sum_routes(rows, positions, implementation=implementation),)
    if implementation == "source":
        combine = combine_local
    elif implementation == "scan_sorted":
        combine = functools.partial(combine_streaming, implementation="scan", scalar_permutation="sort")
    elif implementation == "expert_order":
        combine = combine_expert_order
    else:
        combine = functools.partial(
            combine_streaming,
            implementation=implementation,
            token_tile=token_tile,
            backward_tile=backward_tile,
            output_dtype=output_dtype,
        )
    if mode == "combine_forward":
        return (combine(rows, order, inverse, weights),)
    output, backward = jax.vjp(lambda x, w: combine(x, order, inverse, w), rows, weights)
    return output, *backward(gradient)


def routing_weight_gradient_reference(rows, order, inverse, gradient, *, topk):
    """Compute the combine weight derivative in host FP64, then restore slots."""
    result = np.empty(rows.shape[0], np.float64)
    # Limit the oracle's working storage while checking the full target shape.
    for start in range(0, rows.shape[0], 4096):
        stop = min(start + 4096, rows.shape[0])
        upstream = np.asarray(gradient[order[start:stop] // topk], np.float64)
        expert_rows = np.asarray(rows[start:stop], np.float64)
        result[start:stop] = np.sum(upstream * expert_rows, axis=1, dtype=np.float64)
    return result[inverse].reshape(gradient.shape[0], topk)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    choices = parser.add_mutually_exclusive_group()
    choices.add_argument("--scalar-permutations", action="store_true")
    choices.add_argument("--token-tiles", type=int, nargs="+")
    choices.add_argument("--backward-tiles", type=int, nargs="+")
    parser.add_argument("--paired-shape", action="store_true", help="Use 65536 tokens and FP32 output cotangents")
    args = parser.parse_args()
    if args.paired_shape and not args.backward_tiles:
        parser.error("The paired shape is a backward-tile comparison")
    output, operator_topology, reference_topology = initialize_operator_check()
    tracker.log_summary(
        {
            "kernel_check/input_seed": 424,
            "kernel_check/token_tiles": args.token_tiles or [],
            "kernel_check/backward_tiles": args.backward_tiles or [],
            "kernel_check/paired_shape": args.paired_shape,
            "kernel_check/scalar_permutations": args.scalar_permutations,
        }
    )
    failures = []
    with jax.default_device(jax.local_devices()[0]):
        tokens, topk, width = 65536 if args.paired_shape else 32768, 10, 256
        output_dtype = jnp.float32 if args.paired_shape else None
        gradient_dtype = jnp.float32 if args.paired_shape else jnp.bfloat16
        for case in ("balanced", "skewed"):
            rng = np.random.default_rng(424)
            labels = rng.integers(384, size=tokens * topk)
            if case == "skewed":
                labels[rng.random(labels.size) < 0.9] = 0
            order = jnp.asarray(np.argsort(labels, kind="stable"), jnp.int32)
            inverse = jnp.argsort(order)
            rows = jnp.asarray(rng.standard_normal((tokens * topk, width), dtype=np.float32), jnp.bfloat16)
            rows = rows.at[9 * rows.shape[0] // 10 :].set(0)
            weights = jnp.asarray(rng.random((tokens, topk), dtype=np.float32)).at[:, 0].set(0)
            if args.scalar_permutations or args.token_tiles or args.backward_tiles:
                weights = weights.astype(jnp.bfloat16)
            gradient = jnp.asarray(rng.standard_normal((tokens, width), dtype=np.float32), gradient_dtype)
            inputs = (rows, order, inverse, weights, gradient)
            reference_gradient = None
            if args.scalar_permutations or args.backward_tiles:
                reference_gradient = routing_weight_gradient_reference(
                    np.asarray(rows), np.asarray(order), np.asarray(inverse), np.asarray(gradient), topk=topk
                )
            modes = (
                ("combine_vjp",)
                if args.scalar_permutations or args.backward_tiles
                else ("combine_forward", "combine_vjp", "dispatch_backward")
            )
            for mode in modes:
                variants = ("source", "unrolled", "scan", "flat")
                if mode != "dispatch_backward":
                    variants = ("source", "expert_order", "unrolled", "scan", "flat")
                if args.scalar_permutations:
                    variants = ("scan", "scan_sorted")
                configurations = [(variant, 0, 0) for variant in variants]
                if args.token_tiles:
                    configurations = [("scan", 0, 0), *[("scan", tile, 0) for tile in args.token_tiles]]
                if args.backward_tiles:
                    configurations = [("scan", 0, 0), *[("scan", 0, tile) for tile in args.backward_tiles]]
                expected = None
                for implementation, token_tile, backward_tile in configurations:
                    function = jax.jit(
                        functools.partial(
                            evaluate,
                            implementation,
                            mode,
                            token_tile=token_tile,
                            backward_tile=backward_tile,
                            output_dtype=output_dtype,
                        )
                    )
                    start = time.perf_counter()
                    compiled = function.lower(*inputs).compile()
                    compile_seconds = time.perf_counter() - start
                    actual = jax.block_until_ready(compiled(*inputs))
                    if expected is None:
                        expected = actual
                    deviations = measure_errors(actual, expected, 2e-2)
                    oracle_errors = (
                        measure_errors((actual[-1],), (reference_gradient,), 2e-2)
                        if reference_gradient is not None
                        else None
                    )
                    oracle_passed = (
                        all(np.isfinite(row["normalized_max"]) and row["normalized_max"] <= 1 for row in oracle_errors)
                        if oracle_errors is not None
                        else None
                    )
                    if oracle_passed is False:
                        failures.append(
                            (case, mode, implementation, token_tile, backward_tile, "routing_gradient_oracle")
                        )
                    passed = all(np.isfinite(row["normalized_max"]) and row["normalized_max"] <= 1 for row in deviations)
                    if not passed:
                        failures.append((case, mode, implementation, token_tile, backward_tile))
                    samples = []
                    for _ in range(9):
                        start = time.perf_counter()
                        jax.block_until_ready(compiled(*inputs))
                        samples.append(time.perf_counter() - start)
                    record = {
                        "stage": "streaming_combine_comparison",
                        "kernel": mode,
                        "implementation": implementation,
                        "reference": configurations[0][0],
                        "case": case,
                        "shape": [tokens, topk, width],
                        "dtype": str(rows.dtype),
                        "weight_dtype": str(weights.dtype),
                        "backend": jax.default_backend(),
                        "device_type": jax.local_devices()[0].device_kind,
                        "device_count": 1,
                        "rank": jax.process_index(),
                        "operator_topology": operator_topology,
                        "reference_topology": reference_topology,
                        "block_sizes": {"routes": topk, "tokens": token_tile, "backward_rows": backward_tile},
                        "output_dtype": str(actual[0].dtype),
                        "gradient_dtype": str(gradient.dtype),
                        "passed": passed and oracle_passed is not False,
                        "source_parity_pass": passed,
                        "errors": deviations,
                        "routing_gradient_oracle": "host_fp64_analytical" if oracle_errors is not None else None,
                        "routing_gradient_oracle_errors": oracle_errors,
                        "routing_gradient_oracle_pass": oracle_passed,
                        "compile_time": compile_seconds,
                        "steady_state_time": float(np.median(samples[2:])),
                        "samples": samples,
                        "memory": str(compiled.memory_analysis()),
                        "temporary_bytes": compiled.memory_analysis().temp_size_in_bytes,
                        "git_sha": os.environ["BELAY_SOURCE_COMMIT"],
                        "xla_flags": os.environ.get("XLA_FLAGS", ""),
                        "backend_env": {"LIBTPU_INIT_ARGS": os.environ.get("LIBTPU_INIT_ARGS", "")},
                    }
                    print(json.dumps(record), flush=True)
                    with (output / "streaming-combine-check.jsonl").open("a") as file:
                        file.write(json.dumps(record) + "\n")
    tracker.log_summary({"kernel_check/streaming_combine_pass": not failures})
    tracker.current_tracker().finish()
    if failures:
        raise AssertionError(f"Streaming combine failures: {failures}")


if __name__ == "__main__":
    main()
