# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""One-node H100x8 Transformer Engine NCCL_EP transport decision gate."""

import argparse
import importlib
import json
import math
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

EP_SIZE = 8
TOKENS_PER_RANK = 16_384
HIDDEN_DIM = 2_560
TOP_K = 4
NUM_EXPERTS = 64
CAPACITY_FACTOR = 1.25
RECV_CAPACITY_PER_RANK = 81_920
DISPATCH_ALIGNMENT = 16
BF16_BYTES = 2
ROUTING_WEIGHT_BYTES = 4
HISTORICAL_RING_FULL_ROUTED_MLP_MEDIAN_MS = 22.9143
MATERIAL_HEADROOM_RATIO = 0.8
DEFAULT_WARMUP = 8
DEFAULT_ITERATIONS = 30
SUMMARY_EVENT = "ncclep_h100_transport_gate"


@dataclass(frozen=True)
class TimingSummary:
    iterations: int
    median_ms: float
    p10_ms: float
    p90_ms: float
    remote_wire_bytes_per_rank: int
    effective_wire_gbps: float


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
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def remote_wire_bytes(round_trip_count: int) -> int:
    """Return token and routing-weight bytes sent off-rank by uniform EP8."""
    assignments = TOKENS_PER_RANK * TOP_K
    bytes_per_round_trip = assignments * (2 * HIDDEN_DIM * BF16_BYTES + ROUTING_WEIGHT_BYTES)
    remote_fraction = (EP_SIZE - 1) / EP_SIZE
    return int(round_trip_count * bytes_per_round_trip * remote_fraction)


def summarize_times(times: list[float], round_trip_count: int) -> TimingSummary:
    """Summarize seconds-per-call samples and their effective remote wire rate."""
    if any(not math.isfinite(value) or value <= 0.0 for value in times):
        raise ValueError(f"timings must be positive and finite, got {times}")
    median = statistics.median(times)
    wire_bytes = remote_wire_bytes(round_trip_count)
    return TimingSummary(
        iterations=len(times),
        median_ms=median * 1_000.0,
        p10_ms=percentile(times, 0.10) * 1_000.0,
        p90_ms=percentile(times, 0.90) * 1_000.0,
        remote_wire_bytes_per_rank=wire_bytes,
        effective_wire_gbps=wire_bytes / median / 1e9,
    )


def gate_threshold_ms() -> float:
    return HISTORICAL_RING_FULL_ROUTED_MLP_MEDIAN_MS * MATERIAL_HEADROOM_RATIO


def balanced_route_table() -> np.ndarray:
    """Build the exact fixed-shape uniform int32 route table."""
    global_tokens = TOKENS_PER_RANK * EP_SIZE
    assignments = np.arange(global_tokens * TOP_K, dtype=np.int64)
    return (assignments % NUM_EXPERTS).reshape(global_tokens, TOP_K).astype(np.int32)


def validate_route_capacity(routes: np.ndarray) -> dict[str, Any]:
    """Validate the controlled routes fit TE's fixed receive capacity."""
    expected_shape = (TOKENS_PER_RANK * EP_SIZE, TOP_K)
    if routes.shape != expected_shape:
        raise ValueError(f"route table shape must be {expected_shape}, got {routes.shape}")
    if routes.dtype != np.int32:
        raise TypeError(f"route table dtype must be int32, got {routes.dtype}")
    if routes.size == 0 or int(routes.min()) < 0 or int(routes.max()) >= NUM_EXPERTS:
        raise ValueError(f"route IDs must be in [0, {NUM_EXPERTS})")

    expert_counts = np.bincount(routes.reshape(-1), minlength=NUM_EXPERTS)
    aligned_expert_counts = (expert_counts + DISPATCH_ALIGNMENT - 1) // DISPATCH_ALIGNMENT * DISPATCH_ALIGNMENT
    local_experts = NUM_EXPERTS // EP_SIZE
    destination_counts = expert_counts.reshape(EP_SIZE, local_experts).sum(axis=1)
    aligned_destination_counts = aligned_expert_counts.reshape(EP_SIZE, local_experts).sum(axis=1)
    overflowing = np.flatnonzero(aligned_destination_counts > RECV_CAPACITY_PER_RANK)
    if overflowing.size:
        details = {int(rank): int(aligned_destination_counts[rank]) for rank in overflowing}
        raise ValueError(
            "balanced routing exceeds NCCL_EP receive capacity before dispatch: "
            f"capacity={RECV_CAPACITY_PER_RANK}, aligned_counts={details}"
        )

    return {
        "recv_capacity_per_rank": RECV_CAPACITY_PER_RANK,
        "dispatch_alignment": DISPATCH_ALIGNMENT,
        "expert_count_min": int(expert_counts.min()),
        "expert_count_max": int(expert_counts.max()),
        "destination_counts": destination_counts.tolist(),
        "aligned_destination_counts": aligned_destination_counts.tolist(),
        "maximum_aligned_destination_count": int(aligned_destination_counts.max()),
        "headroom_rows": int(RECV_CAPACITY_PER_RANK - aligned_destination_counts.max()),
        "validated_before_dispatch": True,
    }


def build_summary(
    forward: TimingSummary,
    forward_backward: TimingSummary,
    *,
    runtime: dict[str, Any],
    routing_capacity: dict[str, Any],
) -> dict[str, Any]:
    """Build the stable rank-0 JSON result."""
    passed = forward_backward.median_ms <= gate_threshold_ms()
    return {
        "event": SUMMARY_EVENT,
        "schema_version": 1,
        "status": "pass" if passed else "stop",
        "shape": {
            "ep": EP_SIZE,
            "tokens_per_rank": TOKENS_PER_RANK,
            "hidden_dim": HIDDEN_DIM,
            "top_k": TOP_K,
            "num_experts": NUM_EXPERTS,
            "token_dtype": "bfloat16",
            "routing_weight_dtype": "float32",
            "routing": "uniform",
            "capacity_factor": CAPACITY_FACTOR,
        },
        "routing_capacity": routing_capacity,
        "runtime": runtime,
        "validation": {
            "output_finite": True,
            "loss_finite": True,
            "token_gradients_finite": True,
            "routing_weight_gradients_finite": True,
        },
        "benchmarks": {
            "forward": asdict(forward),
            "forward_backward": asdict(forward_backward),
        },
        "decision_gate": {
            "comparison_kind": "unpaired_historical_hard_sanity_bound",
            "not_apples_to_apples": True,
            "historical_single_process_ring_full_routed_mlp_median_ms": HISTORICAL_RING_FULL_ROUTED_MLP_MEDIAN_MS,
            "material_headroom_ratio": MATERIAL_HEADROOM_RATIO,
            "maximum_transport_forward_backward_median_ms": gate_threshold_ms(),
            "passed": passed,
        },
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
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


def _local_arrays_finite(tree: Any, jax: Any) -> bool:
    for leaf in jax.tree.leaves(tree):
        shards = getattr(leaf, "addressable_shards", None)
        if shards is None:
            if not np.isfinite(np.asarray(leaf)).all():
                return False
            continue
        for shard in shards:
            if not np.isfinite(np.asarray(shard.data)).all():
                return False
    return True


def _assert_all_ranks_finite(label: str, tree: Any, jax: Any, jmu: Any) -> None:
    local_finite = _local_arrays_finite(tree, jax)
    gathered = np.asarray(jmu.process_allgather(np.asarray([int(local_finite)], dtype=np.int32), tiled=False)).reshape(
        -1
    )
    if gathered.shape != (EP_SIZE,):
        raise RuntimeError(f"{label} finite-check allgather shape mismatch: got {gathered.shape}")
    failing_ranks = np.flatnonzero(gathered == 0).tolist()
    if failing_ranks:
        raise FloatingPointError(f"non-finite {label} on ranks {failing_ranks}")


def _slowest_rank_times(local_times: list[float], jmu: Any) -> list[float]:
    gathered = np.asarray(jmu.process_allgather(np.asarray(local_times, dtype=np.float32), tiled=False))
    if gathered.shape != (EP_SIZE, len(local_times)):
        raise RuntimeError(
            f"timing allgather shape mismatch: got {gathered.shape}, expected {(EP_SIZE, len(local_times))}"
        )
    return np.max(gathered, axis=0).tolist()


def run_gate(args: argparse.Namespace) -> int:
    # TE must register its EP FFI handlers before initialize_jax creates a CUDA
    # client. Dynamic imports preserve that ordering despite import sorters.
    transformer_engine = importlib.import_module("transformer_engine")
    te_ep = importlib.import_module("transformer_engine.jax.ep")
    te_sharding = importlib.import_module("transformer_engine.jax.sharding")

    import jax  # noqa: PLC0415
    import jax.experimental.multihost_utils as jmu  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415
    from iris.runtime.jax_init import initialize_jax  # noqa: PLC0415
    from jax.sharding import Mesh, NamedSharding, PartitionSpec  # noqa: PLC0415

    EpLayerConfig = te_ep.EpLayerConfig
    ep_bootstrap = te_ep.ep_bootstrap
    ep_combine = te_ep.ep_combine
    ep_dispatch = te_ep.ep_dispatch
    MeshResource = te_sharding.MeshResource
    global_shard_guard = te_sharding.global_shard_guard

    initialize_jax()
    rank, world = _assert_topology(jax)

    route_table = balanced_route_table()
    routing_capacity = validate_route_capacity(route_table)
    devices = np.asarray(jax.devices()).reshape(1, EP_SIZE)
    mesh = Mesh(devices, ("dp", "ep"))
    global_tokens = TOKENS_PER_RANK * world

    with mesh, global_shard_guard(MeshResource(dp_resource="dp", ep_resource="ep")):
        ep_bootstrap(
            world_size=world,
            rank=rank,
            num_experts=NUM_EXPERTS,
            max_tokens_per_rank=TOKENS_PER_RANK,
            recv_capacity_per_rank=RECV_CAPACITY_PER_RANK,
            hidden_dim=HIDDEN_DIM,
        )
        layer_config = EpLayerConfig(
            top_k=TOP_K,
            dispatch_output_per_expert_alignment=DISPATCH_ALIGNMENT,
        )
        sharding = NamedSharding(mesh, PartitionSpec(("dp", "ep")))
        rng = np.random.default_rng(12_345 + rank)

        def make_array(shape: tuple[int, ...], dtype: Any, generator: Any) -> Any:
            def callback(index: tuple[slice, ...]) -> Any:
                start = index[0].start or 0
                stop = index[0].stop or shape[0]
                local_shape = (stop - start, *shape[1:])
                return generator(start, local_shape).astype(dtype)

            return jax.make_array_from_callback(shape, sharding, callback)

        topk_indices = make_array(
            (global_tokens, TOP_K),
            jnp.int32,
            lambda start, shape: route_table[start : start + shape[0]],
        )
        tokens = make_array(
            (global_tokens, HIDDEN_DIM),
            jnp.bfloat16,
            lambda _start, shape: rng.standard_normal(shape, dtype=np.float32),
        )
        topk_weights = make_array(
            (global_tokens, TOP_K),
            jnp.float32,
            lambda _start, shape: np.full(shape, 1.0 / TOP_K, dtype=np.float32),
        )

        def round_trip(token_values: Any, routes: Any, weights: Any) -> Any:
            recv_tokens, recv_weights, handle_memory, token_counts = ep_dispatch(
                layer_config,
                routes,
                token_values,
                weights,
                RECV_CAPACITY_PER_RANK,
            )
            valid = recv_weights != 0
            weighted = jnp.where(
                valid[..., None],
                recv_tokens * recv_weights[..., None].astype(recv_tokens.dtype),
                jnp.zeros((), dtype=recv_tokens.dtype),
            )
            return ep_combine(
                layer_config,
                handle_memory,
                token_counts,
                weighted,
                tuple(token_values.shape[:-1]),
            )

        def loss(token_values: Any, routes: Any, weights: Any) -> Any:
            output = round_trip(token_values, routes, weights)
            return jnp.mean(jnp.square(output.astype(jnp.float32)))

        forward_fn = jax.jit(round_trip)
        forward_backward_fn = jax.jit(jax.value_and_grad(loss, argnums=(0, 2)))

        output = forward_fn(tokens, topk_indices, topk_weights)
        jax.block_until_ready(output)
        _assert_all_ranks_finite("forward output", output, jax, jmu)

        loss_value, gradients = forward_backward_fn(tokens, topk_indices, topk_weights)
        jax.block_until_ready((loss_value, gradients))
        _assert_all_ranks_finite("loss", loss_value, jax, jmu)
        _assert_all_ranks_finite("token gradients", gradients[0], jax, jmu)
        _assert_all_ranks_finite("routing-weight gradients", gradients[1], jax, jmu)

        def benchmark(fn: Any, label: str) -> list[float]:
            for _ in range(args.warmup):
                jax.block_until_ready(fn(tokens, topk_indices, topk_weights))
            jmu.sync_global_devices(f"ncclep-h100-{label}-timed")
            local_times = []
            for _ in range(args.iterations):
                start = time.perf_counter()
                jax.block_until_ready(fn(tokens, topk_indices, topk_weights))
                local_times.append(time.perf_counter() - start)
            return _slowest_rank_times(local_times, jmu)

        forward_times = benchmark(forward_fn, "forward")
        forward_backward_times = benchmark(forward_backward_fn, "forward-backward")

    forward_summary = summarize_times(forward_times, round_trip_count=1)
    forward_backward_summary = summarize_times(forward_backward_times, round_trip_count=2)
    runtime = {
        "rank_count": world,
        "local_devices_per_rank": 1,
        "jax_version": jax.__version__,
        "transformer_engine_version": transformer_engine.__version__,
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "xla_preallocation_fraction": float(os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]),
        "nccl_runtime_version": os.environ["NCCLEP_NCCL_RUNTIME_VERSION"],
        "te_sha": os.environ["NCCLEP_TE_SHA"],
        "gpu": str(jax.local_devices()[0].device_kind),
    }
    summary = build_summary(
        forward_summary,
        forward_backward_summary,
        runtime=runtime,
        routing_capacity=routing_capacity,
    )
    if rank == 0:
        print(json.dumps(summary, sort_keys=True), flush=True)
    jmu.sync_global_devices("ncclep-h100-summary-emitted")
    return 0 if summary["decision_gate"]["passed"] else 2


def main(argv: list[str]) -> int:
    try:
        return run_gate(parse_args(argv))
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
