# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Strict one-H100 numerical gate for the complete local Sonic/QuACK routed MLP.

Run inside a one-H100 GPU environment:

    CUDA_VISIBLE_DEVICES=0 TF_GPU_ALLOCATOR=cuda_malloc_async \
      XLA_PYTHON_CLIENT_MEM_FRACTION=.70 \
      uv run python experiments/grug/moe/repro_quack_grouped_mlp_numerics.py
"""

import argparse
import json
import math
import os
import statistics
import sys
import time
from collections.abc import Callable
from functools import partial
from importlib.metadata import version
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from haliax.nn.ragged_dot import ragged_dot
from levanter.grug._moe.common import _prepare_moe_dispatch
from levanter.grug._moe.sonic import _moe_mlp_local_sonic
from levanter.grug._moe.sonic_quack import _require_quack

TOKENS = 16_384
TOP_K = 4
ASSIGNMENTS = TOKENS * TOP_K
HIDDEN_DIM = 2_560
INTERMEDIATE_DIM = 1_280
NUM_EXPERTS = 64
ROWS_PER_EXPERT = ASSIGNMENTS // NUM_EXPERTS
QUACK_VERSION = "0.5.0"
RELATIVE_L2_LIMIT = 0.002
DEFAULT_WARMUP = 3
DEFAULT_ITERATIONS = 10
ARM_QUACK = "sonic_quack"
ARM_REFERENCE = "pallas_scatter"
ARMS = (ARM_QUACK, ARM_REFERENCE)
REQUIRED_TENSORS = (
    "loss",
    "gradient.tokens",
    "gradient.routing_weights",
    "gradient.w13",
    "gradient.w2",
)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    args = parser.parse_args(argv)
    if args.warmup < 1:
        parser.error("--warmup must be at least 1")
    if args.iterations < 2:
        parser.error("--iterations must be at least 2")
    return args


def balanced_route_table() -> np.ndarray:
    """Return the fixed exactly balanced local e64/top-k4 route table."""
    assignments = np.arange(ASSIGNMENTS, dtype=np.int64)
    return (assignments % NUM_EXPERTS).reshape(TOKENS, TOP_K).astype(np.int32)


def route_report(routes: np.ndarray) -> dict[str, Any]:
    """Validate that routes retain the fixed balanced admission-gate contract."""
    if routes.shape != (TOKENS, TOP_K):
        raise ValueError(f"route shape must be {(TOKENS, TOP_K)}, got {routes.shape}")
    if routes.dtype != np.int32:
        raise TypeError(f"route dtype must be int32, got {routes.dtype}")
    expected = balanced_route_table()
    if not np.array_equal(routes, expected):
        raise ValueError("routes must equal the fixed balanced route table")
    counts = np.bincount(routes.reshape(-1), minlength=NUM_EXPERTS)
    if not np.all(counts == ROWS_PER_EXPERT):
        raise ValueError(f"routes must assign {ROWS_PER_EXPERT} rows per expert, got {counts.tolist()}")
    return {
        "kind": "fixed_balanced",
        "preserved_exactly": True,
        "expert_count_min": int(counts.min()),
        "expert_count_max": int(counts.max()),
    }


def timing_orders(iterations: int) -> list[tuple[str, str]]:
    """Alternate which arm runs first while preserving equal work counts."""
    if iterations < 1:
        raise ValueError(f"iterations must be positive, got {iterations}")
    return [(ARM_QUACK, ARM_REFERENCE) if index % 2 == 0 else (ARM_REFERENCE, ARM_QUACK) for index in range(iterations)]


def summarize_times(times: list[float]) -> dict[str, float | int]:
    """Summarize positive seconds-per-call timing samples."""
    if not times or any(not math.isfinite(value) or value <= 0.0 for value in times):
        raise ValueError(f"timings must be nonempty, positive, and finite, got {times}")
    ordered = np.asarray(sorted(times), dtype=np.float64)
    return {
        "iterations": len(times),
        "median_ms": statistics.median(times) * 1_000.0,
        "p10_ms": float(np.percentile(ordered, 10)) * 1_000.0,
        "p90_ms": float(np.percentile(ordered, 90)) * 1_000.0,
    }


def summarize_benchmark_samples(samples: dict[str, list[float]]) -> dict[str, Any]:
    """Summarize both timing arms and their median throughput ratio."""
    if set(samples) != set(ARMS):
        raise ValueError(f"timing samples must contain exactly {ARMS}, got {tuple(samples)}")
    summaries = {arm: summarize_times(values) for arm, values in samples.items()}
    return {
        "arms": summaries,
        "sonic_quack_speedup_over_pallas_scatter": (
            float(summaries[ARM_REFERENCE]["median_ms"]) / float(summaries[ARM_QUACK]["median_ms"])
        ),
    }


def tensor_metrics(actual: jax.Array, reference: jax.Array) -> dict[str, float | int | bool]:
    """Return finite, norm, cosine, and pointwise mismatch metrics."""
    if actual.shape != reference.shape:
        raise ValueError(f"metric shapes differ: actual={actual.shape}, reference={reference.shape}")

    def reductions(candidate: jax.Array, expected: jax.Array) -> tuple[jax.Array, ...]:
        candidate_f32 = candidate.astype(jnp.float32)
        expected_f32 = expected.astype(jnp.float32)
        difference = candidate_f32 - expected_f32
        return (
            jnp.linalg.norm(candidate_f32.reshape(-1)),
            jnp.linalg.norm(expected_f32.reshape(-1)),
            jnp.linalg.norm(difference.reshape(-1)),
            jnp.vdot(candidate_f32.reshape(-1), expected_f32.reshape(-1)).real,
            jnp.max(jnp.abs(difference)),
            jnp.mean(jnp.abs(difference)),
            jnp.all(jnp.isfinite(candidate_f32)),
            jnp.all(jnp.isfinite(expected_f32)),
        )

    values = jax.device_get(jax.jit(reductions)(actual, reference))
    actual_norm = float(np.asarray(values[0]))
    reference_norm = float(np.asarray(values[1]))
    error_norm = float(np.asarray(values[2]))
    inner_product = float(np.asarray(values[3]))
    if actual_norm == 0.0 or reference_norm == 0.0:
        cosine = 1.0 if actual_norm == reference_norm else 0.0
    else:
        cosine = max(-1.0, min(1.0, inner_product / (actual_norm * reference_norm)))
    if reference_norm == 0.0:
        relative_l2 = 0.0 if error_norm == 0.0 else float("inf")
    else:
        relative_l2 = error_norm / reference_norm
    return {
        "element_count": int(actual.size),
        "actual_finite": bool(np.asarray(values[6])),
        "reference_finite": bool(np.asarray(values[7])),
        "actual_l2_norm": actual_norm,
        "reference_l2_norm": reference_norm,
        "error_l2_norm": error_norm,
        "relative_l2_error": relative_l2,
        "cosine_similarity": cosine,
        "max_abs_error": float(np.asarray(values[4])),
        "mean_abs_error": float(np.asarray(values[5])),
    }


def admission_report(tensors: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Apply the strict relative-L2 criterion to loss and every gradient leaf."""
    missing = [name for name in (*REQUIRED_TENSORS, "output") if name not in tensors]
    if missing:
        raise ValueError(f"missing parity tensors: {missing}")
    failures = []
    observed = {}
    for name in REQUIRED_TENSORS:
        metrics = tensors[name]
        relative_l2 = float(metrics["relative_l2_error"])
        finite = bool(metrics["actual_finite"]) and bool(metrics["reference_finite"])
        observed[name] = relative_l2
        if not finite or not math.isfinite(relative_l2) or relative_l2 > RELATIVE_L2_LIMIT:
            failures.append(name)
    return {
        "maximum_relative_l2_error": RELATIVE_L2_LIMIT,
        "required_tensors": list(REQUIRED_TENSORS),
        "output_is_diagnostic_only": True,
        "observed": observed,
        "failures": failures,
        "passed": not failures,
    }


def _reference_forward(
    tokens: jax.Array,
    routes: jax.Array,
    routing_weights: jax.Array,
    w13: jax.Array,
    w2: jax.Array,
) -> jax.Array:
    dispatched, dispatch_weights, token_indices, group_sizes = _prepare_moe_dispatch(
        tokens,
        routes,
        routing_weights,
        num_experts=NUM_EXPERTS,
    )
    preactivation = ragged_dot(dispatched, w13, group_sizes, implementation="triton")
    gate, up = jnp.split(preactivation, [INTERMEDIATE_DIM], axis=-1)
    dispatch_output = ragged_dot(jax.nn.silu(gate) * up, w2, group_sizes, implementation="triton")
    return (
        jnp.zeros_like(tokens)
        .at[token_indices]
        .add(
            dispatch_output * dispatch_weights[:, None],
            mode="drop",
        )
    )


def _quack_forward(
    tokens: jax.Array,
    routes: jax.Array,
    routing_weights: jax.Array,
    w13: jax.Array,
    w2: jax.Array,
) -> jax.Array:
    output, dropped = _moe_mlp_local_sonic(
        tokens,
        routes,
        routing_weights,
        w13,
        w2,
        activation_fn=jax.nn.silu,
        num_experts=NUM_EXPERTS,
    )
    del dropped
    return output


def _loss_with_aux(forward: Callable[..., jax.Array], *inputs: jax.Array) -> tuple[jax.Array, jax.Array]:
    output = forward(*inputs)
    loss = jnp.mean(jnp.sum(jnp.square(output.astype(jnp.float32)), axis=-1))
    return loss, output


def _make_inputs(seed: int) -> tuple[jax.Array, ...]:
    key_tokens, key_w13, key_w2 = jax.random.split(jax.random.key(seed), 3)
    tokens = jax.random.normal(key_tokens, (TOKENS, HIDDEN_DIM), dtype=jnp.bfloat16)
    weight_scale = jnp.asarray(0.02, dtype=jnp.bfloat16)
    w13 = weight_scale * jax.random.normal(
        key_w13,
        (NUM_EXPERTS, HIDDEN_DIM, 2 * INTERMEDIATE_DIM),
        dtype=jnp.bfloat16,
    )
    w2 = weight_scale * jax.random.normal(
        key_w2,
        (NUM_EXPERTS, INTERMEDIATE_DIM, HIDDEN_DIM),
        dtype=jnp.bfloat16,
    )
    routes = jnp.asarray(balanced_route_table())
    routing_weights = jnp.full((TOKENS, TOP_K), 1.0 / TOP_K, dtype=jnp.float32)
    return tokens, routes, routing_weights, w13, w2


def _parity_report(
    quack_result: tuple[tuple[jax.Array, jax.Array], tuple[jax.Array, ...]],
    reference_result: tuple[tuple[jax.Array, jax.Array], tuple[jax.Array, ...]],
) -> dict[str, Any]:
    (quack_loss, quack_output), quack_gradients = quack_result
    (reference_loss, reference_output), reference_gradients = reference_result
    tensors = {
        "loss": tensor_metrics(quack_loss, reference_loss),
        "output": tensor_metrics(quack_output, reference_output),
        "gradient.tokens": tensor_metrics(quack_gradients[0], reference_gradients[0]),
        "gradient.routing_weights": tensor_metrics(quack_gradients[1], reference_gradients[1]),
        "gradient.w13": tensor_metrics(quack_gradients[2], reference_gradients[2]),
        "gradient.w2": tensor_metrics(quack_gradients[3], reference_gradients[3]),
    }
    return {
        "tensors": tensors,
        "admission_criterion": admission_report(tensors),
    }


def _benchmark_alternating(
    compiled: dict[str, Callable[..., Any]],
    inputs: tuple[jax.Array, ...],
    *,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    for order in timing_orders(warmup):
        for arm in order:
            jax.block_until_ready(compiled[arm](*inputs))

    samples = {arm: [] for arm in ARMS}
    for order in timing_orders(iterations):
        for arm in order:
            start = time.perf_counter()
            jax.block_until_ready(compiled[arm](*inputs))
            samples[arm].append(time.perf_counter() - start)
    return summarize_benchmark_samples(samples)


def _assert_runtime() -> dict[str, Any]:
    devices = jax.devices()
    if jax.process_count() != 1 or len(devices) != 1 or len(jax.local_devices()) != 1:
        raise RuntimeError(
            "gate requires one JAX process with exactly one visible GPU; "
            f"process_count={jax.process_count()}, devices={devices}, local_devices={jax.local_devices()}"
        )
    device = devices[0]
    if device.platform != "gpu" or "H100" not in device.device_kind:
        raise RuntimeError(f"gate requires one H100, found {device}")
    installed_quack = version("quack-kernels")
    if installed_quack != QUACK_VERSION:
        raise RuntimeError(f"gate requires quack-kernels=={QUACK_VERSION}, found {installed_quack}")
    _require_quack()
    return {
        "topology": "one_process_x_one_h100",
        "device_kind": device.device_kind,
        "jax_version": jax.__version__,
        "quack_kernels_version": installed_quack,
        "tf_gpu_allocator": os.environ.get("TF_GPU_ALLOCATOR", ""),
        "xla_python_client_mem_fraction": os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION", ""),
        "fsdp": False,
        "jaxpp": False,
        "expert_parallelism": False,
        "timing_order": "alternating_quack_first_then_reference_first",
    }


def run_gate(args: argparse.Namespace) -> dict[str, Any]:
    runtime = _assert_runtime()
    routes = balanced_route_table()
    routing = route_report(routes)
    inputs = _make_inputs(args.seed)
    forwards = {
        ARM_QUACK: jax.jit(_quack_forward),
        ARM_REFERENCE: jax.jit(_reference_forward),
    }
    value_and_grads = {
        arm: jax.jit(
            jax.value_and_grad(
                partial(_loss_with_aux, forward),
                argnums=(0, 2, 3, 4),
                has_aux=True,
            )
        )
        for arm, forward in forwards.items()
    }
    compiled_forwards = {arm: forward.lower(*inputs).compile() for arm, forward in forwards.items()}
    compiled_value_and_grads = {
        arm: value_and_grad.lower(*inputs).compile() for arm, value_and_grad in value_and_grads.items()
    }

    validation_results = {
        arm: jax.block_until_ready(compiled_value_and_grad(*inputs))
        for arm, compiled_value_and_grad in compiled_value_and_grads.items()
    }
    parity = _parity_report(validation_results[ARM_QUACK], validation_results[ARM_REFERENCE])
    passed = bool(parity["admission_criterion"]["passed"])
    timings = None
    if passed:
        timings = {
            "forward": _benchmark_alternating(
                compiled_forwards,
                inputs,
                warmup=args.warmup,
                iterations=args.iterations,
            ),
            "value_and_grad": _benchmark_alternating(
                compiled_value_and_grads,
                inputs,
                warmup=args.warmup,
                iterations=args.iterations,
            ),
        }

    return {
        "event": "sonic_quack_no_ep_numerical_admission",
        "schema_version": 1,
        "status": "promote" if passed else "stop",
        "stop_reason": None if passed else "required_relative_l2_failed",
        "shape": {
            "tokens": TOKENS,
            "assignments": ASSIGNMENTS,
            "hidden_dim": HIDDEN_DIM,
            "intermediate_dim": INTERMEDIATE_DIM,
            "num_experts": NUM_EXPERTS,
            "top_k": TOP_K,
            "token_dtype": "bfloat16",
            "weight_dtype": "bfloat16",
            "routing_weight_dtype": "float32",
        },
        "routing": routing,
        "runtime": runtime,
        "loss_contract": "mean_tokens(sum_hidden(square(output.astype(float32))))",
        "reference": "Pallas Triton ragged_dot expert MLP plus JAX scatter-add combine",
        "parity": parity,
        "timings": timings,
        "promotion_criterion": {
            "maximum_required_relative_l2_error": RELATIVE_L2_LIMIT,
            "required_tensors": list(REQUIRED_TENSORS),
            "output_is_diagnostic_only": True,
            "timing_is_promotional": False,
            "passed": passed,
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    try:
        result = run_gate(args)
    except Exception as error:
        result = {
            "event": "sonic_quack_no_ep_numerical_admission",
            "schema_version": 1,
            "status": "error",
            "error_type": type(error).__name__,
            "error": str(error),
        }
        print(json.dumps(result, sort_keys=True), flush=True)
        return 2
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0 if result["status"] == "promote" else 1


if __name__ == "__main__":
    raise SystemExit(main())
