# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Benchmark the existing generic JAX expert-parallel path at the MoK shape.

This is deliberately not a wrapper around Mixture-of-Kittens.  It composes the
ordinary shared MLP with Levanter's generic routed-MoE implementation:

    route relation -> dispatch -> segmented GEMMs -> weighted combine

Run one implementation per process.  DeepEP initializes process-global
transport state, while the XLA ragged-all-to-all flags are fixed before JAX is
imported, so separate processes also make the comparison easier to reproduce.
"""

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from haliax.nn.ragged_dot import ragged_dot
from jax import shard_map
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from levanter.grug._moe.ep_common import (
    _clip_receiver_group_sizes,
    _compact_by_keep_mask,
    _expand_from_keep_mask,
    _expert_prefix_keep_mask,
    _local_permute_from_counts,
    _permute_by_global_expert,
    _shard_a2a_params,
    _sort_activations,
    _unpermute_from_global_expert,
)
from levanter.grug.grug_moe import moe_mlp, split_moe_w13_output
from levanter.kernels.deepep import deepep_combine_intranode, deepep_dispatch_intranode, deepep_get_dispatch_layout

DEFAULT_LOCAL_TOKENS = 2_048
DEFAULT_HIDDEN_SIZE = 7_168
DEFAULT_INTERMEDIATE_SIZE = 3_072
DEFAULT_GLOBAL_EXPERTS = 384
DEFAULT_TOP_K = 6
DEFAULT_EXPERT_PARALLEL_SIZE = 4
BYTES_PER_BF16 = 2


@dataclass(frozen=True)
class BenchmarkShape:
    """Static shape and capacity choices for one generic EP benchmark."""

    local_tokens: int
    hidden_size: int
    intermediate_size: int
    global_experts: int
    top_k: int
    expert_parallel_size: int
    capacity_factor: float

    @property
    def global_tokens(self) -> int:
        return self.local_tokens * self.expert_parallel_size

    @property
    def local_experts(self) -> int:
        return self.global_experts // self.expert_parallel_size

    @property
    def assignments_per_rank(self) -> int:
        return self.local_tokens * self.top_k

    @property
    def ragged_receive_capacity(self) -> int:
        return math.ceil(self.capacity_factor * self.assignments_per_rank)

    @property
    def deepep_receive_capacity(self) -> int:
        return self.local_tokens * self.expert_parallel_size

    @property
    def deepep_candidate_assignment_capacity(self) -> int:
        return self.deepep_receive_capacity * self.top_k

    @property
    def deepep_assignment_capacity(self) -> int:
        requested_capacity = math.ceil(self.capacity_factor * self.assignments_per_rank)
        return min(requested_capacity, self.deepep_candidate_assignment_capacity)

    @property
    def logical_flops_per_rank(self) -> int:
        shared_and_routed_rows = self.local_tokens * (1 + self.top_k)
        return 6 * shared_and_routed_rows * self.hidden_size * self.intermediate_size

    def component_flops_per_rank(self, component: str) -> int:
        if component == "shared":
            rows = self.local_tokens
        elif component == "routed":
            rows = self.local_tokens * self.top_k
        elif component == "local_segmented":
            rows = self.ragged_receive_capacity
        elif component in ("transport", "deepep_transport"):
            rows = 0
        else:
            rows = self.local_tokens * (1 + self.top_k)
        return 6 * rows * self.hidden_size * self.intermediate_size

    @property
    def routed_weight_bytes_per_rank(self) -> int:
        elements = self.local_experts * self.hidden_size * self.intermediate_size * 3
        return elements * BYTES_PER_BF16

    @property
    def shared_weight_bytes_per_rank(self) -> int:
        elements = self.hidden_size * self.intermediate_size * 3
        return elements * BYTES_PER_BF16

    def validate(self) -> None:
        dimensions = {
            "local_tokens": self.local_tokens,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "global_experts": self.global_experts,
            "top_k": self.top_k,
            "expert_parallel_size": self.expert_parallel_size,
        }
        nonpositive = [name for name, value in dimensions.items() if value <= 0]
        if nonpositive:
            raise ValueError(f"shape values must be positive: {', '.join(nonpositive)}")
        if self.global_experts % self.expert_parallel_size:
            raise ValueError(
                f"global_experts={self.global_experts} must be divisible by "
                f"expert_parallel_size={self.expert_parallel_size}"
            )
        if self.top_k > self.global_experts:
            raise ValueError(f"top_k={self.top_k} must not exceed global_experts={self.global_experts}")
        if self.hidden_size % 8:
            raise ValueError(f"hidden_size={self.hidden_size} must be divisible by 8 for DeepEP transport")
        if not np.isfinite(self.capacity_factor) or self.capacity_factor < 1.0:
            raise ValueError(f"capacity_factor must be finite and at least 1.0, got {self.capacity_factor}")


@dataclass(frozen=True)
class TimingSummary:
    median_ms: float
    mean_ms: float
    minimum_ms: float
    maximum_ms: float
    local_tokens_per_second: float
    logical_tflops_per_rank: float


@dataclass(frozen=True)
class RouteStatistics:
    """Traffic and balance checks for the deterministic routing fixture."""

    minimum_assignments_per_expert: int
    maximum_assignments_per_expert: int
    minimum_unique_experts_per_token: int
    maximum_unique_experts_per_token: int
    mean_distinct_destination_ranks_per_token: float


@dataclass(frozen=True)
class RouteFixture:
    """Host routing relation and weights copied to the expert-sharded mesh."""

    selected_experts: np.ndarray
    combine_weights: np.ndarray
    source: str
    sha256: str | None


def _balanced_routes_numpy(shape: BenchmarkShape) -> np.ndarray:
    token_ids = np.arange(shape.global_tokens, dtype=np.int64)[:, None]
    offsets = np.arange(shape.top_k, dtype=np.int64)[None, :]
    owners = (token_ids + offsets) % shape.expert_parallel_size
    local_experts = ((token_ids // shape.expert_parallel_size) * shape.top_k + offsets) % shape.local_experts
    return owners * shape.local_experts + local_experts


def _route_statistics(routes: np.ndarray, shape: BenchmarkShape) -> RouteStatistics:
    if routes.shape != (shape.global_tokens, shape.top_k):
        raise ValueError(f"selected_experts must have shape {(shape.global_tokens, shape.top_k)}, got {routes.shape}")
    if routes.min() < 0 or routes.max() >= shape.global_experts:
        raise ValueError(f"selected_experts must be in [0, {shape.global_experts})")

    counts = np.bincount(routes.ravel(), minlength=shape.global_experts)
    unique_experts = np.asarray([np.unique(row).size for row in routes])
    owners = routes // shape.local_experts
    unique_owners = np.asarray([np.unique(row).size for row in owners])
    statistics = RouteStatistics(
        minimum_assignments_per_expert=int(counts.min()),
        maximum_assignments_per_expert=int(counts.max()),
        minimum_unique_experts_per_token=int(unique_experts.min()),
        maximum_unique_experts_per_token=int(unique_experts.max()),
        mean_distinct_destination_ranks_per_token=float(unique_owners.mean()),
    )
    if statistics.minimum_unique_experts_per_token != shape.top_k:
        raise ValueError("routing fixture produced duplicate experts for a token")
    return statistics


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--implementation", choices=("ragged_all_to_all", "deepep"), required=True)
    parser.add_argument(
        "--component",
        choices=("full", "shared", "routed", "transport", "deepep_transport", "local_segmented"),
        default="full",
    )
    parser.add_argument("--local-tokens", type=int, default=DEFAULT_LOCAL_TOKENS)
    parser.add_argument("--hidden-size", type=int, default=DEFAULT_HIDDEN_SIZE)
    parser.add_argument("--intermediate-size", type=int, default=DEFAULT_INTERMEDIATE_SIZE)
    parser.add_argument("--global-experts", type=int, default=DEFAULT_GLOBAL_EXPERTS)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--expert-parallel-size", type=int, default=DEFAULT_EXPERT_PARALLEL_SIZE)
    parser.add_argument("--capacity-factor", type=float, default=1.25)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument(
        "--route-mode",
        choices=("mok_fixture", "balanced_diagnostic"),
        default="mok_fixture",
        help="Use saved official MoK random routes, or an explicitly diagnostic balanced fixture.",
    )
    parser.add_argument("--route-fixture", type=Path)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--json-output", type=Path)
    return parser


def _shape(args: argparse.Namespace) -> BenchmarkShape:
    shape = BenchmarkShape(
        local_tokens=args.local_tokens,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        global_experts=args.global_experts,
        top_k=args.top_k,
        expert_parallel_size=args.expert_parallel_size,
        capacity_factor=args.capacity_factor,
    )
    shape.validate()
    if args.warmup < 0:
        raise ValueError(f"warmup must be non-negative, got {args.warmup}")
    if args.iterations <= 0:
        raise ValueError(f"iterations must be positive, got {args.iterations}")
    return shape


def _routes(args: argparse.Namespace, shape: BenchmarkShape) -> RouteFixture:
    if args.route_mode == "balanced_diagnostic":
        selected_experts = _balanced_routes_numpy(shape)
        route_statistics = _route_statistics(selected_experts, shape)
        if route_statistics.maximum_assignments_per_expert - route_statistics.minimum_assignments_per_expert > 1:
            raise ValueError(
                "balanced diagnostic routing is not balanced for this shape: "
                f"min={route_statistics.minimum_assignments_per_expert}, "
                f"max={route_statistics.maximum_assignments_per_expert}"
            )
        combine_weights = np.full(selected_experts.shape, 1.0 / shape.top_k, dtype=np.float32)
        return RouteFixture(
            selected_experts=selected_experts.astype(np.int32),
            combine_weights=combine_weights,
            source="balanced_diagnostic",
            sha256=None,
        )

    if args.route_fixture is None:
        raise ValueError("--route-fixture is required when --route-mode=mok_fixture")
    with np.load(args.route_fixture) as fixture:
        selected_experts = np.asarray(fixture["selected_experts"], dtype=np.int32)
        combine_weights = np.asarray(fixture["combine_weights"], dtype=np.float32)
    _route_statistics(selected_experts, shape)
    if combine_weights.shape != selected_experts.shape:
        raise ValueError(f"combine_weights must have shape {selected_experts.shape}, got {combine_weights.shape}")
    if not np.isfinite(combine_weights).all():
        raise ValueError("combine_weights must be finite")
    if not np.allclose(combine_weights.sum(axis=1), 1.0, rtol=1e-6, atol=1e-6):
        raise ValueError("combine_weights must sum to one for every token")
    digest = hashlib.sha256(args.route_fixture.read_bytes()).hexdigest()
    return RouteFixture(
        selected_experts=selected_experts,
        combine_weights=combine_weights,
        source=str(args.route_fixture),
        sha256=digest,
    )


def _mesh(shape: BenchmarkShape) -> Mesh:
    devices = np.asarray(jax.devices(), dtype=object)
    if devices.size != shape.expert_parallel_size:
        raise RuntimeError(
            "the benchmark requires the expert group to span every visible device; "
            f"expected {shape.expert_parallel_size}, found {devices.size}. "
            "Set CUDA_VISIBLE_DEVICES explicitly."
        )
    return Mesh(
        devices.reshape(1, 1, shape.expert_parallel_size, 1),
        axis_names=("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit,) * 4,
    )


def _full(shape: tuple[int, ...], value: float, sharding: NamedSharding) -> jax.Array:
    materialize = jax.jit(
        lambda: jnp.full(shape, value, dtype=jnp.bfloat16),
        out_shardings=sharding,
    )
    return materialize()


def _inputs(shape: BenchmarkShape, mesh: Mesh, routes: RouteFixture) -> tuple[jax.Array, ...]:
    batch = NamedSharding(mesh, P(("replica_dcn", "data", "expert"), None))
    routed_weights = NamedSharding(mesh, P("expert", None, None))
    replicated = NamedSharding(mesh, P(None, None))

    x = _full((shape.global_tokens, shape.hidden_size), 0.01, batch)
    selected_experts = jax.device_put(routes.selected_experts, batch)
    combine_weights = jax.device_put(routes.combine_weights, batch)
    shared_w13 = _full((shape.hidden_size, 2 * shape.intermediate_size), 0.001, replicated)
    shared_w2 = _full((shape.intermediate_size, shape.hidden_size), 0.001, replicated)
    routed_w13 = _full(
        (shape.global_experts, shape.hidden_size, 2 * shape.intermediate_size),
        0.001,
        routed_weights,
    )
    routed_w2 = _full(
        (shape.global_experts, shape.intermediate_size, shape.hidden_size),
        0.001,
        routed_weights,
    )
    inputs = (x, selected_experts, combine_weights, shared_w13, shared_w2, routed_w13, routed_w2)
    jax.block_until_ready(inputs)
    return inputs


def _transport_roundtrip_local(
    x_local: jax.Array,
    selected_experts_local: jax.Array,
    combine_weights_local: jax.Array,
    *,
    num_experts: int,
    expert_parallel_size: int,
    capacity_factor: float,
) -> tuple[jax.Array, jax.Array]:
    local_experts = num_experts // expert_parallel_size
    shard_id = jax.lax.axis_index("expert")
    local_tokens = x_local.shape[0]
    top_k = selected_experts_local.shape[1]
    assignments = local_tokens * top_k
    capacity = max(local_experts, math.ceil(capacity_factor * assignments))

    sorted_x, sorted_indices, group_sizes = _permute_by_global_expert(
        x_local,
        selected_experts_local,
        num_experts=num_experts,
    )
    all_group_sizes = jax.lax.all_gather(group_sizes.astype(jnp.int32), "expert")
    clipped_group_sizes = _clip_receiver_group_sizes(
        all_group_sizes,
        local_expert_size=local_experts,
        receiver_capacity=capacity,
    )
    sender_group_sizes = clipped_group_sizes[shard_id]
    keep_mask = _expert_prefix_keep_mask(group_sizes, sender_group_sizes, total_size=assignments)
    sorted_x = _compact_by_keep_mask(sorted_x, keep_mask)
    all_shard_counts = jnp.sum(
        clipped_group_sizes.reshape(expert_parallel_size, expert_parallel_size, local_experts),
        axis=2,
    )
    input_offsets, send_sizes, output_offsets, recv_sizes = _shard_a2a_params(all_shard_counts, shard_id)
    dispatched = jax.lax.ragged_all_to_all(
        sorted_x,
        jnp.zeros((capacity, x_local.shape[1]), dtype=x_local.dtype),
        input_offsets,
        send_sizes,
        output_offsets,
        recv_sizes,
        axis_name="expert",
    )
    dispatched, local_sorted_indices, _ = _local_permute_from_counts(
        dispatched,
        clipped_group_sizes,
        local_expert_size=local_experts,
        shard_index=shard_id,
    )

    local_output = _sort_activations(dispatched, jnp.argsort(local_sorted_indices))
    return_parameters = _shard_a2a_params(all_shard_counts.T, shard_id)
    returned = jax.lax.ragged_all_to_all(
        local_output,
        jnp.zeros((assignments, x_local.shape[1]), dtype=x_local.dtype),
        *return_parameters,
        axis_name="expert",
    )
    returned = _expand_from_keep_mask(returned, keep_mask)
    output = _unpermute_from_global_expert(
        returned,
        sorted_indices,
        combine_weights_local,
        tokens_per_shard=local_tokens,
        topk=top_k,
    )
    dropped_local = jnp.sum(group_sizes, dtype=jnp.int32) - jnp.sum(sender_group_sizes, dtype=jnp.int32)
    return output.astype(x_local.dtype), jax.lax.psum(dropped_local, "expert")


def _deepep_transport_roundtrip_local(
    x_local: jax.Array,
    selected_experts_local: jax.Array,
    combine_weights_local: jax.Array,
    *,
    num_experts: int,
    expert_parallel_size: int,
) -> tuple[jax.Array, jax.Array]:
    num_tokens_per_rank, num_tokens_per_expert, is_token_in_rank = deepep_get_dispatch_layout(
        selected_experts_local,
        num_ranks=expert_parallel_size,
        num_experts=num_experts,
    )
    (
        recv_x,
        _recv_topk_idx,
        recv_topk_weights,
        recv_src_idx,
        rank_prefix_matrix,
        channel_prefix_matrix,
        recv_channel_prefix_matrix,
        send_head,
        _local_expert_counts,
        num_recv_tokens,
    ) = deepep_dispatch_intranode(
        x_local,
        selected_experts_local,
        combine_weights_local,
        num_tokens_per_rank,
        num_tokens_per_expert,
        is_token_in_rank,
        num_experts=num_experts,
        max_recv_tokens=x_local.shape[0] * expert_parallel_size,
    )
    output, _ = deepep_combine_intranode(
        recv_x,
        recv_topk_weights,
        recv_src_idx,
        rank_prefix_matrix,
        channel_prefix_matrix,
        recv_channel_prefix_matrix,
        send_head,
        num_recv_tokens,
        is_token_in_rank,
    )
    return output.astype(x_local.dtype), jnp.array(0, dtype=jnp.int32)


def _local_segmented_mlp_local(
    x_local: jax.Array,
    selected_experts_local: jax.Array,
    routed_w13_local: jax.Array,
    routed_w2_local: jax.Array,
    *,
    num_experts: int,
    capacity: int,
) -> tuple[jax.Array, jax.Array]:
    local_experts = routed_w13_local.shape[0]
    shard_id = jax.lax.axis_index("expert")
    all_selected_experts = jax.lax.all_gather(selected_experts_local, "expert")
    global_counts = jnp.bincount(all_selected_experts.reshape(-1), length=num_experts).astype(jnp.int32)
    local_counts = jax.lax.dynamic_slice_in_dim(
        global_counts,
        shard_id * local_experts,
        local_experts,
    )
    repeats = math.ceil(capacity / x_local.shape[0])
    dispatched = jnp.tile(x_local, (repeats, 1))[:capacity]
    group_sizes = local_counts.at[-1].add(capacity - jnp.sum(local_counts, dtype=jnp.int32))
    gate_up = ragged_dot(dispatched, routed_w13_local, group_sizes)
    gate, up = split_moe_w13_output(
        gate_up,
        intermediate_dim=routed_w2_local.shape[1],
        interleaved=False,
    )
    output = ragged_dot(jax.nn.silu(gate) * up, routed_w2_local, group_sizes)
    return output, jnp.array(0, dtype=jnp.int32)


def _forward(implementation: str, mesh: Mesh, shape: BenchmarkShape, component: str):
    def run(x, selected_experts, combine_weights, shared_w13, shared_w2, routed_w13, routed_w2):
        batch_spec = P(("replica_dcn", "data", "expert"), None)
        if component == "deepep_transport":
            if implementation != "deepep":
                raise ValueError("the deepep_transport component requires --implementation deepep")
            transport = shard_map(
                partial(
                    _deepep_transport_roundtrip_local,
                    num_experts=shape.global_experts,
                    expert_parallel_size=shape.expert_parallel_size,
                ),
                mesh=mesh,
                in_specs=(batch_spec, batch_spec, batch_spec),
                out_specs=(batch_spec, P()),
                check_vma=False,
            )
            return transport(x, selected_experts, combine_weights)
        if component == "transport":
            transport = shard_map(
                partial(
                    _transport_roundtrip_local,
                    num_experts=shape.global_experts,
                    expert_parallel_size=shape.expert_parallel_size,
                    capacity_factor=shape.capacity_factor,
                ),
                mesh=mesh,
                in_specs=(batch_spec, batch_spec, batch_spec),
                out_specs=(batch_spec, P()),
                check_vma=False,
            )
            return transport(x, selected_experts, combine_weights)
        if component == "local_segmented":
            local_segmented = shard_map(
                partial(
                    _local_segmented_mlp_local,
                    num_experts=shape.global_experts,
                    capacity=shape.ragged_receive_capacity,
                ),
                mesh=mesh,
                in_specs=(batch_spec, batch_spec, P("expert", None, None), P("expert", None, None)),
                out_specs=(P("expert", None), P()),
                check_vma=False,
            )
            return local_segmented(x, selected_experts, routed_w13, routed_w2)
        if component != "routed":
            with jax.named_scope("shared_expert"):
                shared_gate_up = x @ shared_w13
                shared_gate, shared_up = split_moe_w13_output(
                    shared_gate_up,
                    intermediate_dim=shape.intermediate_size,
                    interleaved=False,
                )
                shared = (jax.nn.silu(shared_gate) * shared_up) @ shared_w2
            if component == "shared":
                return shared.astype(jnp.bfloat16), jnp.array(0, dtype=jnp.int32)

        routed, dropped = moe_mlp(
            x,
            selected_experts,
            combine_weights,
            routed_w13,
            routed_w2,
            activation=jax.nn.silu,
            implementation=implementation,
            mesh=mesh,
            capacity_factor=shape.capacity_factor,
            report_capacity_overflow=True,
        )
        if component == "routed":
            return routed.astype(jnp.bfloat16), dropped
        return (shared + routed).astype(jnp.bfloat16), dropped

    return jax.jit(run)


def _timings(
    compiled,
    inputs: tuple[jax.Array, ...],
    shape: BenchmarkShape,
    *,
    component: str,
    warmup: int,
    iterations: int,
):
    latest = None
    for _ in range(warmup):
        latest = compiled(*inputs)
        jax.block_until_ready(latest)

    durations: list[float] = []
    for _ in range(iterations):
        started = time.perf_counter()
        latest = compiled(*inputs)
        jax.block_until_ready(latest)
        durations.append(time.perf_counter() - started)

    assert latest is not None
    median = statistics.median(durations)
    summary = TimingSummary(
        median_ms=median * 1_000,
        mean_ms=statistics.fmean(durations) * 1_000,
        minimum_ms=min(durations) * 1_000,
        maximum_ms=max(durations) * 1_000,
        local_tokens_per_second=shape.local_tokens / median,
        logical_tflops_per_rank=shape.component_flops_per_rank(component) / median / 1e12,
    )
    return latest, summary


def _git_revision() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _result(
    args: argparse.Namespace,
    shape: BenchmarkShape,
    compile_seconds: float,
    output: tuple[jax.Array, jax.Array],
    timing: TimingSummary,
    routes: RouteFixture,
) -> dict[str, object]:
    values, dropped = output
    route_statistics = _route_statistics(routes.selected_experts, shape)
    summarize = jax.jit(
        lambda value: (
            jnp.mean(value.astype(jnp.float32)),
            jnp.max(jnp.abs(value.astype(jnp.float32))),
            jnp.all(jnp.isfinite(value)),
        )
    )
    output_mean, output_max_abs, output_finite = map(np.asarray, jax.device_get(summarize(values)))
    return {
        "implementation": args.implementation,
        "component": args.component,
        "shape": asdict(shape),
        "route_fixture": {
            "mode": args.route_mode,
            "source": routes.source,
            "sha256": routes.sha256,
        },
        "derived": {
            "global_tokens": shape.global_tokens,
            "local_experts": shape.local_experts,
            "assignments_per_rank": shape.assignments_per_rank,
            "ragged_receive_capacity": shape.ragged_receive_capacity,
            "deepep_receive_capacity": shape.deepep_receive_capacity,
            "deepep_candidate_assignment_capacity": shape.deepep_candidate_assignment_capacity,
            "deepep_assignment_capacity": shape.deepep_assignment_capacity,
            "deepep_assignment_overprovision": shape.deepep_assignment_capacity / shape.assignments_per_rank,
            "logical_flops_per_rank": shape.component_flops_per_rank(args.component),
            "routed_weight_gib_per_rank": shape.routed_weight_bytes_per_rank / 2**30,
            "shared_weight_gib_per_rank": shape.shared_weight_bytes_per_rank / 2**30,
            "routes": asdict(route_statistics),
        },
        "compile_seconds": compile_seconds,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "timing": asdict(timing),
        "correctness_smoke": {
            "dropped_assignments": int(np.asarray(jax.device_get(dropped))),
            "output_mean": float(output_mean),
            "output_max_abs": float(output_max_abs),
            "output_finite": bool(output_finite),
        },
        "environment": {
            "git_revision": _git_revision(),
            "python": platform.python_version(),
            "jax": jax.__version__,
            "jaxlib": _package_version("jaxlib"),
            "jax_triton": _package_version("jax-triton"),
            "triton": _package_version("triton"),
            "device_kind": jax.devices()[0].device_kind,
            "device_count": len(jax.devices()),
            "xla_flags": os.environ.get("XLA_FLAGS", ""),
            "ragged_dot_impl": os.environ.get("RAGGED_DOT_IMPL", "auto"),
            "deepep_source_root": os.environ.get("DEEPEP_SRC_ROOT"),
            "deepep_cuda_arch": os.environ.get("DEEPEP_CUDA_ARCH"),
        },
    }


def main() -> None:
    args = _parser().parse_args()
    shape = _shape(args)
    routes = _routes(args, shape)
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"GB200 timing requires the GPU backend, got {jax.default_backend()}")

    mesh = _mesh(shape)
    with jax.set_mesh(mesh):
        inputs = _inputs(shape, mesh, routes)
        forward = _forward(args.implementation, mesh, shape, args.component)
        started = time.perf_counter()
        compiled = forward.lower(*inputs).compile()
        compile_seconds = time.perf_counter() - started
        if args.compile_only:
            result = {
                "implementation": args.implementation,
                "shape": asdict(shape),
                "route_fixture": {
                    "mode": args.route_mode,
                    "source": routes.source,
                    "sha256": routes.sha256,
                },
                "compile_seconds": compile_seconds,
                "compile_only": True,
            }
            rendered = json.dumps(result, indent=2, sort_keys=True)
            print(rendered)
            if args.json_output is not None:
                args.json_output.parent.mkdir(parents=True, exist_ok=True)
                args.json_output.write_text(rendered + "\n")
            return
        output, timing = _timings(
            compiled,
            inputs,
            shape,
            component=args.component,
            warmup=args.warmup,
            iterations=args.iterations,
        )
        result = _result(args, shape, compile_seconds, output, timing, routes)

    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
