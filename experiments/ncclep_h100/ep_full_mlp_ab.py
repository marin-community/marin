# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""One-node H100x8 full routed-MLP A/B: Marin bulk ring versus TE NCCL_EP."""

import argparse
import importlib
import json
import math
import os
import re
import statistics
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from functools import partial
from typing import Any

import numpy as np

EP_SIZE = 8
TOKENS_PER_RANK = 16_384
HIDDEN_DIM = 2_560
INTERMEDIATE_DIM = 1_280
TOP_K = 4
NUM_EXPERTS = 64
LOCAL_EXPERTS = NUM_EXPERTS // EP_SIZE
CAPACITY_FACTOR = 1.25
ASSIGNMENTS_PER_DESTINATION = TOKENS_PER_RANK * TOP_K
RECV_CAPACITY_PER_RANK = int(CAPACITY_FACTOR * ASSIGNMENTS_PER_DESTINATION)
DISPATCH_ALIGNMENT = 16
BF16_RTOL = 0.1
BF16_ATOL = 2e-4
PROMOTION_SPEEDUP = 1.10
DEFAULT_WARMUP = 6
DEFAULT_ITERATIONS = 20
SUMMARY_EVENT = "ncclep_h100_full_mlp_ab"
ARM_RING = "marin_bulk_ring"
ARM_TE = "transformer_engine_nccl_ep"


@dataclass(frozen=True)
class TimingSummary:
    iterations: int
    median_ms: float
    p10_ms: float
    p90_ms: float


def percentile(values: list[float], quantile: float) -> float:
    """Return a linearly interpolated percentile for a nonempty sample."""
    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError(f"quantile must be in [0, 1], got {quantile}")
    ordered = sorted(values)
    position = quantile * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    upper_weight = position - lower
    return ordered[lower] * (1.0 - upper_weight) + ordered[upper] * upper_weight


def summarize_times(times: list[float]) -> TimingSummary:
    """Summarize positive seconds-per-call samples."""
    if any(not math.isfinite(value) or value <= 0.0 for value in times):
        raise ValueError(f"timings must be positive and finite, got {times}")
    return TimingSummary(
        iterations=len(times),
        median_ms=statistics.median(times) * 1_000.0,
        p10_ms=percentile(times, 0.10) * 1_000.0,
        p90_ms=percentile(times, 0.90) * 1_000.0,
    )


def timing_orders(iterations: int) -> list[tuple[str, str]]:
    """Alternate which arm runs first while preserving identical work counts."""
    if iterations < 1:
        raise ValueError(f"iterations must be positive, got {iterations}")
    return [(ARM_RING, ARM_TE) if index % 2 == 0 else (ARM_TE, ARM_RING) for index in range(iterations)]


def balanced_route_table() -> np.ndarray:
    """Return the fixed globally balanced e64/top-k4 route table."""
    global_tokens = TOKENS_PER_RANK * EP_SIZE
    assignments = np.arange(global_tokens * TOP_K, dtype=np.int64)
    return (assignments % NUM_EXPERTS).reshape(global_tokens, TOP_K).astype(np.int32)


def routing_capacity_report(routes: np.ndarray) -> dict[str, Any]:
    """Validate that the fixed routing is balanced and fits both implementations."""
    expected_shape = (TOKENS_PER_RANK * EP_SIZE, TOP_K)
    if routes.shape != expected_shape:
        raise ValueError(f"route table shape must be {expected_shape}, got {routes.shape}")
    if routes.dtype != np.int32:
        raise TypeError(f"route table dtype must be int32, got {routes.dtype}")
    if routes.size == 0 or int(routes.min()) < 0 or int(routes.max()) >= NUM_EXPERTS:
        raise ValueError(f"route IDs must be in [0, {NUM_EXPERTS})")

    counts = np.bincount(routes.reshape(-1), minlength=NUM_EXPERTS)
    aligned_counts = (counts + DISPATCH_ALIGNMENT - 1) // DISPATCH_ALIGNMENT * DISPATCH_ALIGNMENT
    destination_counts = counts.reshape(EP_SIZE, LOCAL_EXPERTS).sum(axis=1)
    aligned_destination_counts = aligned_counts.reshape(EP_SIZE, LOCAL_EXPERTS).sum(axis=1)
    if np.any(aligned_destination_counts > RECV_CAPACITY_PER_RANK):
        raise ValueError(
            "routing exceeds NCCL_EP receive capacity: "
            f"capacity={RECV_CAPACITY_PER_RANK}, counts={aligned_destination_counts.tolist()}"
        )
    if not np.all(destination_counts == ASSIGNMENTS_PER_DESTINATION):
        raise ValueError(f"routing must be exactly balanced, got destination counts {destination_counts.tolist()}")

    return {
        "expert_count_min": int(counts.min()),
        "expert_count_max": int(counts.max()),
        "destination_counts": destination_counts.tolist(),
        "aligned_destination_counts": aligned_destination_counts.tolist(),
        "recv_capacity_per_rank": RECV_CAPACITY_PER_RANK,
        "capacity_padding_rows_per_rank": (RECV_CAPACITY_PER_RANK - aligned_destination_counts).tolist(),
        "dispatch_alignment": DISPATCH_ALIGNMENT,
        "validated_before_dispatch": True,
    }


_STABLEHLO_OPS = (
    "custom_call",
    "all_gather",
    "reduce_scatter",
    "all_reduce",
    "collective_permute",
    "send",
    "recv",
)


def count_stablehlo_operations(stablehlo_text: str) -> dict[str, Any]:
    """Count communication operations and custom-call targets in StableHLO text."""
    operation_counts = {
        operation: len(re.findall(rf"\bstablehlo\.{re.escape(operation)}\b", stablehlo_text))
        for operation in _STABLEHLO_OPS
    }
    targets: dict[str, int] = {}
    for target in re.findall(r'call_target_name\s*=\s*"([^"]+)"', stablehlo_text):
        targets[target] = targets.get(target, 0) + 1
    return {
        "operations": operation_counts,
        "custom_call_targets": dict(sorted(targets.items())),
    }


def build_summary(
    *,
    timings: dict[str, TimingSummary],
    parity: dict[str, Any],
    finite: dict[str, Any],
    runtime: dict[str, Any],
    routing: dict[str, Any],
    stablehlo: dict[str, Any],
) -> dict[str, Any]:
    """Build the stable rank-0 JSON result and promotion decision."""
    ring_median = timings[ARM_RING].median_ms
    te_median = timings[ARM_TE].median_ms
    speedup = ring_median / te_median
    parity_passed = bool(parity["passed"])
    finite_passed = all(all(bool(value) for value in arm_checks.values()) for arm_checks in finite.values())
    promoted = parity_passed and finite_passed and speedup >= PROMOTION_SPEEDUP
    return {
        "event": SUMMARY_EVENT,
        "schema_version": 1,
        "status": "promote" if promoted else "stop",
        "shape": {
            "topology": "one_node_8_processes_x_1_h100",
            "ep": EP_SIZE,
            "tokens_per_rank": TOKENS_PER_RANK,
            "global_tokens": TOKENS_PER_RANK * EP_SIZE,
            "hidden_dim": HIDDEN_DIM,
            "intermediate_dim": INTERMEDIATE_DIM,
            "num_experts": NUM_EXPERTS,
            "local_experts": LOCAL_EXPERTS,
            "top_k": TOP_K,
            "capacity_factor": CAPACITY_FACTOR,
            "token_dtype": "bfloat16",
            "weight_dtype": "bfloat16",
            "routing_weight_dtype": "float32",
            "routing": "uniform_balanced",
        },
        "runtime": runtime,
        "routing_capacity": routing,
        "finite": finite,
        "parity": parity,
        "stablehlo": stablehlo,
        "timings": {arm: asdict(summary) for arm, summary in timings.items()},
        "comparison": {
            "kind": "paired_same_run_same_inputs_same_process_topology",
            "ring_over_te_speedup": speedup,
            "te_latency_reduction_fraction": 1.0 - te_median / ring_median,
        },
        "promotion_criterion": {
            "minimum_ring_over_te_speedup": PROMOTION_SPEEDUP,
            "all_parity_checks_must_pass": True,
            "all_finite_checks_must_pass": True,
            "passed": promoted,
        },
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument(
        "--parity-mode",
        choices=("strict", "diagnostic"),
        default="strict",
        help="strict stops before timing on parity failure; diagnostic times but remains non-promotable",
    )
    args = parser.parse_args(argv)
    if args.warmup < 1:
        parser.error("--warmup must be at least 1")
    if args.iterations < 2:
        parser.error("--iterations must be at least 2")
    return args


def _assert_topology(jax: Any) -> tuple[int, int]:
    rank = jax.process_index()
    world = jax.process_count()
    local_devices = jax.local_devices()
    expected_rank = int(os.environ.get("IRIS_MULTIGPU_PROCESS_INDEX", "-1"))
    expected_world = int(os.environ.get("IRIS_MULTIGPU_PROCESS_COUNT", "-1"))
    errors = []
    if world != EP_SIZE:
        errors.append(f"JAX process_count={world}, expected {EP_SIZE}")
    if expected_world != EP_SIZE:
        errors.append(f"IRIS_MULTIGPU_PROCESS_COUNT={expected_world}, expected {EP_SIZE}")
    if rank != expected_rank:
        errors.append(f"JAX process_index={rank}, supervisor rank={expected_rank}")
    if len(local_devices) != 1:
        errors.append(f"rank {rank} sees {len(local_devices)} local devices, expected 1")
    if len(jax.devices()) != EP_SIZE:
        errors.append(f"global device_count={len(jax.devices())}, expected {EP_SIZE}")
    if local_devices and local_devices[0].platform != "gpu":
        errors.append(f"rank {rank} device platform={local_devices[0].platform!r}, expected 'gpu'")
    if errors:
        raise RuntimeError("process topology mismatch: " + "; ".join(errors))
    return rank, world


def _make_array(
    jax: Any,
    sharding: Any,
    shape: tuple[int, ...],
    generator: Callable[[tuple[slice, ...], tuple[int, ...]], np.ndarray],
) -> Any:
    def callback(index: tuple[slice, ...]) -> np.ndarray:
        local_shape = tuple((part.stop or shape[axis]) - (part.start or 0) for axis, part in enumerate(index))
        return generator(index, local_shape)

    return jax.make_array_from_callback(shape, sharding, callback)


def _make_inputs(jax: Any, jnp: Any, mesh: Any, routes: np.ndarray) -> tuple[Any, ...]:
    NamedSharding = jax.sharding.NamedSharding
    P = jax.sharding.PartitionSpec
    batch_sharding = NamedSharding(mesh, P(("replica_dcn", "data", "expert"), None))
    expert_sharding = NamedSharding(mesh, P("expert", None, None))
    global_tokens = TOKENS_PER_RANK * EP_SIZE

    def bf16_normal(seed: int, scale: float = 1.0) -> Callable[[tuple[slice, ...], tuple[int, ...]], np.ndarray]:
        def generate(index: tuple[slice, ...], shape: tuple[int, ...]) -> np.ndarray:
            first = index[0].start or 0
            values = np.random.default_rng(seed + first).standard_normal(shape, dtype=np.float32)
            return (values * scale).astype(jnp.bfloat16)

        return generate

    tokens = _make_array(
        jax,
        batch_sharding,
        (global_tokens, HIDDEN_DIM),
        bf16_normal(10_000),
    )
    topk_indices = _make_array(
        jax,
        batch_sharding,
        (global_tokens, TOP_K),
        lambda index, shape: routes[index[0].start or 0 : index[0].stop or global_tokens].reshape(shape),
    )
    topk_weights = _make_array(
        jax,
        batch_sharding,
        (global_tokens, TOP_K),
        lambda _index, shape: np.full(shape, 1.0 / TOP_K, dtype=np.float32),
    )
    w13 = _make_array(
        jax,
        expert_sharding,
        (NUM_EXPERTS, HIDDEN_DIM, 2 * INTERMEDIATE_DIM),
        bf16_normal(20_000, 0.02),
    )
    w2 = _make_array(
        jax,
        expert_sharding,
        (NUM_EXPERTS, INTERMEDIATE_DIM, HIDDEN_DIM),
        bf16_normal(30_000, 0.02),
    )
    return tokens, topk_indices, topk_weights, w13, w2


def _compiled_ring_forward(jax: Any, mesh: Any, ring_local: Callable[..., Any]) -> Callable[..., Any]:
    P = jax.sharding.PartitionSpec
    batch_spec = P(("replica_dcn", "data", "expert"), None)
    expert_spec = P("expert", None, None)
    mapped = jax.shard_map(
        partial(
            ring_local,
            activation_fn=jax.nn.silu,
            num_experts=NUM_EXPERTS,
            capacity_factor=CAPACITY_FACTOR,
        ),
        mesh=mesh,
        in_specs=(batch_spec, batch_spec, batch_spec, expert_spec, expert_spec),
        out_specs=(batch_spec, P()),
        check_vma=False,
    )
    return jax.jit(mapped)


def _compiled_te_forward(
    jax: Any,
    jnp: Any,
    mesh: Any,
    *,
    layer_config: Any,
    ep_dispatch: Callable[..., Any],
    ep_combine: Callable[..., Any],
    ragged_dot: Callable[..., Any],
) -> Callable[..., Any]:
    P = jax.sharding.PartitionSpec
    batch_spec = P(("replica_dcn", "data", "expert"), None)
    lead = P(batch_spec[0], None, None)
    lead2 = P(batch_spec[0], None)
    expert_spec = P("expert", None, None)

    def local_ffn(recv_tokens: Any, token_counts: Any, w13: Any, w2: Any) -> Any:
        dispatched = recv_tokens.reshape(recv_tokens.shape[-2], recv_tokens.shape[-1])
        group_sizes = token_counts.reshape(-1).astype(jnp.int32)
        valid_rows = jnp.arange(dispatched.shape[0], dtype=jnp.int32) < jnp.sum(group_sizes, dtype=jnp.int32)
        dispatched = jnp.where(valid_rows[:, None], dispatched, jnp.zeros((), dtype=dispatched.dtype))
        group_sizes = group_sizes.at[-1].add(dispatched.shape[0] - jnp.sum(group_sizes, dtype=jnp.int32))
        w13_out = ragged_dot(dispatched, w13, group_sizes, implementation="triton")
        gate, up = jnp.split(w13_out, [INTERMEDIATE_DIM], axis=-1)
        expert_out = ragged_dot(jax.nn.silu(gate) * up, w2, group_sizes, implementation="triton")
        return expert_out.reshape(recv_tokens.shape)

    def body(tokens: Any, routes: Any, weights: Any, w13: Any, w2: Any) -> Any:
        recv_tokens, recv_weights, handle_memory, token_counts = ep_dispatch(
            layer_config,
            routes.astype(jnp.int32),
            tokens,
            weights.astype(jnp.float32),
            RECV_CAPACITY_PER_RANK,
        )
        recv_tokens = jax.lax.with_sharding_constraint(recv_tokens, lead)
        recv_weights = jax.lax.with_sharding_constraint(recv_weights, lead2)
        token_counts = jax.lax.with_sharding_constraint(token_counts, lead2)
        ffn = jax.shard_map(
            local_ffn,
            mesh=jax.sharding.get_abstract_mesh(),
            in_specs=(lead, lead2, expert_spec, expert_spec),
            out_specs=lead,
            check_vma=False,
        )
        expert_out = ffn(recv_tokens, token_counts, w13, w2)
        slot_weights = recv_weights[..., None].astype(expert_out.dtype)
        weighted = jnp.where(
            slot_weights != 0,
            expert_out * slot_weights,
            jnp.zeros((), dtype=expert_out.dtype),
        )
        weighted = jax.lax.with_sharding_constraint(weighted, lead)
        return ep_combine(
            layer_config,
            handle_memory,
            token_counts,
            weighted,
            tuple(tokens.shape[:-1]),
        ).astype(tokens.dtype)

    def forward(tokens: Any, routes: Any, weights: Any, w13: Any, w2: Any) -> tuple[Any, Any]:
        output = jax.sharding.auto_axes(
            body,
            axes=tuple(mesh.axis_names),
            out_sharding=batch_spec,
        )(tokens, routes, weights, w13, w2)
        return output, jnp.zeros((), dtype=jnp.int32)

    return jax.jit(forward)


def _loss_with_aux(forward: Callable[..., Any], *inputs: Any) -> tuple[Any, tuple[Any, Any]]:
    output, dropped = forward(*inputs)
    loss = importlib.import_module("jax.numpy").mean(
        importlib.import_module("jax.numpy").square(output.astype(np.float32))
    )
    return loss, (output, dropped)


def _finite_report(result: Any, jax: Any, jnp: Any) -> dict[str, bool]:
    (loss, (output, _dropped)), gradients = result

    def finite(value: Any) -> bool:
        return bool(np.asarray(jax.device_get(jnp.all(jnp.isfinite(value)))))

    return {
        "loss": finite(loss),
        "output": finite(output),
        "token_gradients": finite(gradients[0]),
        "routing_weight_gradients": finite(gradients[1]),
        "w13_gradients": finite(gradients[2]),
        "w2_gradients": finite(gradients[3]),
    }


def _array_parity(actual: Any, expected: Any, jax: Any, jnp: Any) -> dict[str, Any]:
    def reductions(candidate: Any, reference: Any) -> tuple[Any, ...]:
        candidate_f32 = candidate.astype(jnp.float32)
        reference_f32 = reference.astype(jnp.float32)
        difference = jnp.abs(candidate_f32 - reference_f32)
        tolerance = BF16_ATOL + BF16_RTOL * jnp.abs(reference_f32)
        error_l2 = jnp.linalg.norm(difference.reshape(-1))
        reference_l2 = jnp.linalg.norm(reference_f32.reshape(-1))
        return (
            jnp.max(difference),
            jnp.mean(difference),
            jnp.sum(difference > tolerance, dtype=jnp.int32),
            error_l2,
            reference_l2,
            jnp.all(jnp.isfinite(candidate_f32)),
            jnp.all(jnp.isfinite(reference_f32)),
        )

    values = jax.device_get(jax.jit(reductions)(actual, expected))
    mismatch_count = int(np.asarray(values[2]))
    error_l2 = float(np.asarray(values[3]))
    reference_l2 = float(np.asarray(values[4]))
    return {
        "rtol": BF16_RTOL,
        "atol": BF16_ATOL,
        "allclose": mismatch_count == 0,
        "mismatch_count": mismatch_count,
        "element_count": int(actual.size),
        "mismatch_fraction": mismatch_count / int(actual.size),
        "max_abs": float(np.asarray(values[0])),
        "mean_abs": float(np.asarray(values[1])),
        "relative_l2_error": error_l2 / reference_l2 if reference_l2 else (0.0 if error_l2 == 0.0 else float("inf")),
        "candidate_finite": bool(np.asarray(values[5])),
        "reference_finite": bool(np.asarray(values[6])),
    }


def _parity_report(te_result: Any, ring_result: Any, jax: Any, jnp: Any) -> dict[str, Any]:
    (te_loss, (te_output, te_dropped)), te_gradients = te_result
    (ring_loss, (ring_output, ring_dropped)), ring_gradients = ring_result
    tensors = {
        "loss": _array_parity(te_loss, ring_loss, jax, jnp),
        "output": _array_parity(te_output, ring_output, jax, jnp),
        "gradient.tokens": _array_parity(te_gradients[0], ring_gradients[0], jax, jnp),
        "gradient.routing_weights": _array_parity(te_gradients[1], ring_gradients[1], jax, jnp),
        "gradient.w13": _array_parity(te_gradients[2], ring_gradients[2], jax, jnp),
        "gradient.w2": _array_parity(te_gradients[3], ring_gradients[3], jax, jnp),
    }
    dropped = {
        ARM_RING: int(np.asarray(jax.device_get(ring_dropped))),
        ARM_TE: int(np.asarray(jax.device_get(te_dropped))),
    }
    failures = [name for name, metrics in tensors.items() if not metrics["allclose"]]
    if dropped[ARM_RING] != 0 or dropped[ARM_TE] != 0:
        failures.append("dropped_assignments")
    return {
        "passed": not failures,
        "failures": failures,
        "tolerances": {
            "rtol": BF16_RTOL,
            "atol": BF16_ATOL,
            "source": "experiments/grug/moe/benchmark_ep_ring.py strict BF16 parity",
        },
        "dropped_assignments": dropped,
        "tensors": tensors,
    }


def _slowest_rank_times(local_times: list[float], jmu: Any) -> list[float]:
    gathered = np.asarray(jmu.process_allgather(np.asarray(local_times, dtype=np.float64), tiled=False))
    if gathered.shape != (EP_SIZE, len(local_times)):
        raise RuntimeError(
            f"timing allgather shape mismatch: got {gathered.shape}, expected {(EP_SIZE, len(local_times))}"
        )
    return np.max(gathered, axis=0).tolist()


def _benchmark_interleaved(
    compiled: dict[str, Callable[..., Any]],
    inputs: tuple[Any, ...],
    *,
    warmup: int,
    iterations: int,
    jax: Any,
    jmu: Any,
) -> dict[str, list[float]]:
    for order in timing_orders(warmup):
        for arm in order:
            jax.block_until_ready(compiled[arm](*inputs))
    jmu.sync_global_devices("ncclep-full-mlp-ab-timed")

    local_times = {ARM_RING: [], ARM_TE: []}
    for order in timing_orders(iterations):
        for arm in order:
            start = time.perf_counter()
            jax.block_until_ready(compiled[arm](*inputs))
            local_times[arm].append(time.perf_counter() - start)
    jmu.sync_global_devices("ncclep-full-mlp-ab-timing-complete")
    return {arm: _slowest_rank_times(samples, jmu) for arm, samples in local_times.items()}


def run_ab(args: argparse.Namespace) -> int:
    # TE must register NCCL_EP FFI handlers before initialize_jax creates a CUDA
    # client. Dynamic imports preserve that ordering.
    transformer_engine = importlib.import_module("transformer_engine")
    te_ep = importlib.import_module("transformer_engine.jax.ep")
    te_sharding = importlib.import_module("transformer_engine.jax.sharding")

    jax = importlib.import_module("jax")
    jnp = importlib.import_module("jax.numpy")
    jmu = importlib.import_module("jax.experimental.multihost_utils")
    initialize_jax = importlib.import_module("iris.runtime.jax_init").initialize_jax

    initialize_jax()
    rank, world = _assert_topology(jax)

    compact_grug_mesh = importlib.import_module("levanter.grug.sharding").compact_grug_mesh
    ring_local = importlib.import_module("levanter.grug._moe.ep_ring")._moe_mlp_ep_ring_local
    ragged_dot = importlib.import_module("haliax.nn.ragged_dot").ragged_dot

    routes = balanced_route_table()
    routing = routing_capacity_report(routes)
    mesh = compact_grug_mesh(expert_axis_size=EP_SIZE, replica_axis_size=1)
    MeshResource = te_sharding.MeshResource
    global_shard_guard = te_sharding.global_shard_guard

    with jax.set_mesh(mesh), global_shard_guard(MeshResource(dp_resource="data", ep_resource="expert")):
        te_ep.ep_bootstrap(
            world_size=world,
            rank=rank,
            num_experts=NUM_EXPERTS,
            max_tokens_per_rank=TOKENS_PER_RANK,
            recv_capacity_per_rank=RECV_CAPACITY_PER_RANK,
            hidden_dim=HIDDEN_DIM,
        )
        layer_config = te_ep.EpLayerConfig(
            top_k=TOP_K,
            dispatch_output_per_expert_alignment=DISPATCH_ALIGNMENT,
        )
        inputs = _make_inputs(jax, jnp, mesh, routes)
        ring_forward = _compiled_ring_forward(jax, mesh, ring_local)
        te_forward = _compiled_te_forward(
            jax,
            jnp,
            mesh,
            layer_config=layer_config,
            ep_dispatch=te_ep.ep_dispatch,
            ep_combine=te_ep.ep_combine,
            ragged_dot=ragged_dot,
        )
        value_and_grads = {
            ARM_RING: jax.jit(
                jax.value_and_grad(partial(_loss_with_aux, ring_forward), argnums=(0, 2, 3, 4), has_aux=True)
            ),
            ARM_TE: jax.jit(jax.value_and_grad(partial(_loss_with_aux, te_forward), argnums=(0, 2, 3, 4), has_aux=True)),
        }
        lowered = {arm: fn.lower(*inputs) for arm, fn in value_and_grads.items()}
        stablehlo = {
            arm: count_stablehlo_operations(str(lowered_fn.compiler_ir(dialect="stablehlo")))
            for arm, lowered_fn in lowered.items()
        }
        compiled = {arm: lowered_fn.compile() for arm, lowered_fn in lowered.items()}

        validation_results = {arm: jax.block_until_ready(compiled_fn(*inputs)) for arm, compiled_fn in compiled.items()}
        finite = {arm: _finite_report(result, jax, jnp) for arm, result in validation_results.items()}
        parity = _parity_report(validation_results[ARM_TE], validation_results[ARM_RING], jax, jnp)
        finite_passed = all(all(checks.values()) for checks in finite.values())
        if not finite_passed or (not parity["passed"] and args.parity_mode == "strict"):
            validation_summary = {
                "event": SUMMARY_EVENT,
                "schema_version": 1,
                "status": "stop",
                "stop_reason": "strict_parity_failed" if not parity["passed"] else "non_finite_values",
                "finite": finite,
                "parity": parity,
                "routing_capacity": routing,
                "stablehlo": stablehlo,
                "timings": None,
                "promotion_criterion": {
                    "minimum_ring_over_te_speedup": PROMOTION_SPEEDUP,
                    "all_parity_checks_must_pass": True,
                    "all_finite_checks_must_pass": True,
                    "passed": False,
                },
            }
            if rank == 0:
                print(json.dumps(validation_summary, sort_keys=True), flush=True)
            jmu.sync_global_devices("ncclep-full-mlp-ab-validation-failed")
            return 0

        samples = _benchmark_interleaved(
            compiled,
            inputs,
            warmup=args.warmup,
            iterations=args.iterations,
            jax=jax,
            jmu=jmu,
        )

    timings = {arm: summarize_times(arm_samples) for arm, arm_samples in samples.items()}
    runtime = {
        "rank_count": world,
        "local_devices_per_rank": 1,
        "gpu": str(jax.local_devices()[0].device_kind),
        "jax_version": jax.__version__,
        "transformer_engine_version": transformer_engine.__version__,
        "te_sha": os.environ["NCCLEP_TE_SHA"],
        "nccl_runtime_version": os.environ["NCCLEP_NCCL_RUNTIME_VERSION"],
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "xla_preallocation_fraction": float(os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]),
        "ragged_dot_implementation": os.environ.get("RAGGED_DOT_IMPL", ""),
        "ragged_dot_triton_block_k": int(os.environ["HALIAX_RAGGED_DOT_TRITON_BLOCK_K"]),
        "ragged_dot_triton_num_warps": int(os.environ["HALIAX_RAGGED_DOT_TRITON_NUM_WARPS"]),
        "timing_order": "alternating_ring_first_then_te_first",
        "sample_aggregation": "slowest_rank_per_sample",
        "warmup_pairs": args.warmup,
        "measured_pairs": args.iterations,
        "parity_mode": args.parity_mode,
    }
    summary = build_summary(
        timings=timings,
        parity=parity,
        finite=finite,
        runtime=runtime,
        routing=routing,
        stablehlo=stablehlo,
    )
    if rank == 0:
        print(json.dumps(summary, sort_keys=True), flush=True)
    jmu.sync_global_devices("ncclep-full-mlp-ab-summary-emitted")
    return 0


def main(argv: list[str]) -> int:
    try:
        return run_ab(parse_args(argv))
    except Exception as error:
        if os.environ.get("IRIS_MULTIGPU_PROCESS_INDEX", "0") == "0":
            print(
                json.dumps(
                    {
                        "event": SUMMARY_EVENT,
                        "schema_version": 1,
                        "status": "error",
                        "error_type": type(error).__name__,
                        "error": str(error),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        raise


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
