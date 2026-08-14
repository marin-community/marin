# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Four-GPU numerical and multi-process failure gates for the supported mok_like API."""

import json
import math
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from enum import StrEnum
from functools import partial
from threading import Barrier

import click
import jax
import jax.numpy as jnp
import numpy as np
from iris.runtime.jax_init import initialize_jax
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from levanter.kernels.mixture_of_kittens import (
    MOK_CONTEXT_CHECKPOINT_NAME,
    MokLikeBackwardPeerStorage,
    MokLikeBuildConfig,
    MokLikeConfig,
    MokLikeForwardXStorage,
    MokLikeWorkspaceTransport,
    initialize_mok_like_runtime,
    mok_like_mlp,
    mok_like_reference,
)
from levanter.kernels.mixture_of_kittens.api import _fused_backward, _fused_forward_with_context
from levanter.kernels.mixture_of_kittens.runtime import MokLikeTestFailurePhase, MokLikeTestFailurePoint
from levanter.kernels.mixture_of_kittens.schedule import schedule_capacity

WORLD_SIZE = 4
TOP_K = 4
NUM_LOCAL_EXPERTS = 2
NUM_EXPERTS = WORLD_SIZE * NUM_LOCAL_EXPERTS
BF16_ATOL = 0.5
BF16_RTOL = 0.01
SHARED_GRADIENT_ATOL = 1.0
ROUTER_GRADIENT_RELATIVE_L2_TOLERANCE = 0.01
MOK_LIKE_SOURCE_ROOT = "/tmp/marin-mok-like/source"
MOK_LIKE_BUILD_ROOT = "/tmp/marin-mok-like/build"

type FailureResult = jax.Array | tuple[jax.Array, tuple[jax.Array, ...]]
type CompiledFailure = Callable[..., FailureResult]


class RouteScenario(StrEnum):
    BALANCED = "balanced"
    ZERO_TOKEN_EXPERT = "zero_token_expert"
    SKEWED = "skewed"
    ALL_TO_ONE = "all_to_one"


def _routes(num_tokens: int, scenario: RouteScenario) -> np.ndarray:
    source_ranks = np.arange(WORLD_SIZE, dtype=np.int32)[:, None, None]
    token_indices = np.arange(num_tokens, dtype=np.int32)[None, :, None]
    route_indices = np.arange(TOP_K, dtype=np.int32)[None, None, :]
    destination_ranks = (source_ranks + route_indices) % WORLD_SIZE
    if scenario is RouteScenario.ALL_TO_ONE:
        destination_ranks = np.zeros((WORLD_SIZE, num_tokens, TOP_K), dtype=np.int32)
        local_experts = np.broadcast_to(route_indices % NUM_LOCAL_EXPERTS, (WORLD_SIZE, num_tokens, TOP_K))
    elif scenario is RouteScenario.BALANCED:
        local_experts = ((source_ranks + token_indices + route_indices) % NUM_LOCAL_EXPERTS).astype(np.int32)
    elif scenario is RouteScenario.ZERO_TOKEN_EXPERT:
        local_experts = np.zeros((WORLD_SIZE, num_tokens, TOP_K), dtype=np.int32)
    elif scenario is RouteScenario.SKEWED:
        local_experts = np.broadcast_to(
            ((source_ranks + token_indices) % 4 == 0).astype(np.int32),
            (WORLD_SIZE, num_tokens, TOP_K),
        )
    else:
        raise AssertionError(f"unhandled route scenario {scenario}")
    return destination_ranks * NUM_LOCAL_EXPERTS + local_experts


def _canonical_inputs(
    *,
    num_tokens: int,
    hidden_dim: int,
    intermediate_dim: int,
) -> tuple[np.ndarray, ...]:
    random = np.random.default_rng(1234)
    global_tokens = WORLD_SIZE * num_tokens
    x = random.normal(size=(global_tokens, hidden_dim)).astype(np.float32)
    combine_weights = np.ones((global_tokens, TOP_K), dtype=np.float32)
    w_gate = (random.normal(size=(NUM_EXPERTS, hidden_dim, intermediate_dim)) / hidden_dim**0.5).astype(np.float32)
    w_up = (random.normal(size=(NUM_EXPERTS, hidden_dim, intermediate_dim)) / hidden_dim**0.5).astype(np.float32)
    w_down = (random.normal(size=(NUM_EXPERTS, intermediate_dim, hidden_dim)) / intermediate_dim**0.5).astype(np.float32)
    shared_gate = (random.normal(size=(hidden_dim, intermediate_dim)) / hidden_dim**0.5).astype(np.float32)
    shared_up = (random.normal(size=(hidden_dim, intermediate_dim)) / hidden_dim**0.5).astype(np.float32)
    shared_down = (random.normal(size=(intermediate_dim, hidden_dim)) / intermediate_dim**0.5).astype(np.float32)
    return x, combine_weights, w_gate, w_up, w_down, shared_gate, shared_up, shared_down


def _real_macrobuffers(routes: np.ndarray, *, capacity: int, macrobatch_size: int) -> int:
    per_rank: list[int] = []
    for rank in range(WORLD_SIZE):
        first_expert = rank * NUM_LOCAL_EXPERTS
        padded = 0
        for local_expert in range(NUM_LOCAL_EXPERTS):
            assignments = int(np.count_nonzero(routes == first_expert + local_expert))
            padded += math.ceil(assignments / 256) * 256
        per_rank.append(math.ceil(min(padded, capacity) / macrobatch_size))
    return max(per_rank)


def _required_schedule_capacity(routes: np.ndarray) -> int:
    """Return the largest destination's independently padded assignment count."""

    required_per_rank: list[int] = []
    for rank in range(WORLD_SIZE):
        first_expert = rank * NUM_LOCAL_EXPERTS
        required = 0
        for local_expert in range(NUM_LOCAL_EXPERTS):
            assignments = int(np.count_nonzero(routes == first_expert + local_expert))
            required += math.ceil(assignments / 256) * 256
        required_per_rank.append(required)
    return max(required_per_rank)


def _error_metrics(
    actual: jax.Array,
    expected: jax.Array,
    *,
    absolute_tolerance: float,
) -> dict[str, float | bool]:
    actual_float = np.asarray(jax.device_get(actual), dtype=np.float32)
    expected_float = np.asarray(jax.device_get(expected), dtype=np.float32)
    absolute_error = np.abs(actual_float - expected_float)
    close = np.isclose(actual_float, expected_float, atol=absolute_tolerance, rtol=BF16_RTOL)
    worst_flat_index = int(np.argmax(absolute_error))
    reference_l2 = float(np.linalg.norm(expected_float))
    return {
        "allclose": bool(np.all(close)),
        "absolute_tolerance": absolute_tolerance,
        "max_absolute_error": float(np.max(absolute_error)),
        "mean_absolute_error": float(np.mean(absolute_error)),
        "mismatch_fraction": float(np.mean(~close)),
        "relative_l2_error": float(np.linalg.norm(actual_float - expected_float) / max(reference_l2, 1e-12)),
        "worst_actual": float(actual_float.reshape(-1)[worst_flat_index]),
        "worst_expected": float(expected_float.reshape(-1)[worst_flat_index]),
    }


@click.command()
@click.option(
    "--workspace-transport",
    type=click.Choice([value.value for value in MokLikeWorkspaceTransport]),
    default=MokLikeWorkspaceTransport.IN_PROCESS_PEER.value,
    show_default=True,
    help="peer-workspace transport; fabric_symmetric runs one rank per process over imported arenas",
)
@click.option(
    "--forward-only",
    is_flag=True,
    help="run only the forward comparison; the rest of the matrix cannot run cross-process yet",
)
@click.option("--num-tokens", type=click.IntRange(min=256), default=512, show_default=True)
@click.option("--hidden-dim", type=click.IntRange(min=256), default=512, show_default=True)
@click.option("--intermediate-dim", type=click.IntRange(min=256), default=512, show_default=True)
@click.option("--minibatch-size", type=click.IntRange(min=256), default=256, show_default=True)
@click.option("--macrobatch-size", type=click.IntRange(min=256), default=256, show_default=True)
@click.option(
    "--schedule-capacity-factor",
    type=click.FloatRange(min=1.0),
    default=1.1,
    show_default=True,
)
@click.option(
    "--scenario",
    type=click.Choice(RouteScenario, case_sensitive=False),
    default=RouteScenario.BALANCED,
    show_default=True,
)
@click.option(
    "--workspace-slots",
    type=click.IntRange(min=1, max=2),
    default=2,
    show_default=True,
    help="Keep two slots for concurrent-call coverage; use one only for serialized-runtime checks.",
)
@click.option(
    "--forward-x-storage",
    type=click.Choice(MokLikeForwardXStorage, case_sensitive=False),
    default=MokLikeForwardXStorage.RUNTIME_STAGED,
    show_default=True,
    help="Choose staged x storage or experimental direct peer reads from XLA-owned x buffers.",
)
@click.option(
    "--backward-peer-storage",
    type=click.Choice(MokLikeBackwardPeerStorage, case_sensitive=False),
    default=MokLikeBackwardPeerStorage.RUNTIME_STAGED,
    show_default=True,
    help="Choose staged backward peer buffers or experimental direct access to XLA-owned buffers.",
)
@click.option("--remat", is_flag=True, help="Checkpoint the loss and require one forward plus one backward per GPU.")
@click.option("--offload", is_flag=True, help="Offload the saved native context to pinned host memory.")
@click.option(
    "--expected-real-macrobuffers",
    type=click.IntRange(min=1),
    help="Require the route schedule to exercise this many non-padding macrobuffers.",
)
@click.option("--back-to-back", is_flag=True, help="Run two dependent forward calls in one compiled executable.")
@click.option("--concurrent-calls", is_flag=True, help="Invoke one compiled forward executable from two host threads.")
@click.option("--corrupt-stamp", is_flag=True, help="Deliberately corrupt the saved runtime epoch; execution must fail.")
@click.option(
    "--failure-gate",
    type=click.Choice(
        (
            "forward_before_input_ready",
            "forward_before_completion",
            "backward_before_input_ready",
            "backward_before_completion",
        )
    ),
    help="Inject one synchronous rank-local handler failure and require an all-rank typed-FFI error.",
)
@click.option(
    "--failure-rank",
    type=click.IntRange(min=0, max=WORLD_SIZE - 1),
    default=0,
    show_default=True,
)
@click.option(
    "--failure-concurrent-control",
    is_flag=True,
    help="Overlap a second RunId and require cancellation to remain slot-local.",
)
@click.option(
    "--output-gradient-scale",
    type=click.FloatRange(min=0.0, min_open=True),
    default=1.0,
    show_default=True,
    help="Scale the independent output cotangent used by gradient parity checks.",
)
def main(
    forward_only: bool,
    workspace_transport: str,
    num_tokens: int,
    hidden_dim: int,
    intermediate_dim: int,
    minibatch_size: int,
    macrobatch_size: int,
    schedule_capacity_factor: float,
    scenario: RouteScenario,
    workspace_slots: int,
    forward_x_storage: MokLikeForwardXStorage,
    backward_peer_storage: MokLikeBackwardPeerStorage,
    remat: bool,
    offload: bool,
    expected_real_macrobuffers: int | None,
    back_to_back: bool,
    concurrent_calls: bool,
    corrupt_stamp: bool,
    failure_gate: str | None,
    failure_rank: int,
    failure_concurrent_control: bool,
    output_gradient_scale: float,
) -> None:
    """Compare every differentiable leaf with the ordinary Grug EP reference."""
    initialize_jax()

    for name, value in (
        ("num_tokens", num_tokens),
        ("hidden_dim", hidden_dim),
        ("intermediate_dim", intermediate_dim),
        ("minibatch_size", minibatch_size),
        ("macrobatch_size", macrobatch_size),
    ):
        if value % 256 != 0:
            raise click.BadParameter(f"{name} must be divisible by 256, got {value}")
    if macrobatch_size % minibatch_size != 0:
        raise click.BadParameter("macrobatch_size must be divisible by minibatch_size")
    if remat and offload:
        raise click.BadParameter("choose either --remat or --offload")
    if corrupt_stamp and (back_to_back or concurrent_calls):
        raise click.BadParameter("--corrupt-stamp must run as an isolated expected-failure gate")
    if failure_gate is not None and (corrupt_stamp or back_to_back or concurrent_calls or remat or offload):
        raise click.BadParameter("--failure-gate must run as an isolated non-rematerialized gate")
    if failure_concurrent_control and failure_gate is None:
        raise click.BadParameter("--failure-concurrent-control requires --failure-gate")
    if failure_concurrent_control and workspace_slots != 2:
        raise click.BadParameter("--failure-concurrent-control requires --workspace-slots=2")
    if failure_concurrent_control and jax.process_count() != 1:
        raise click.BadParameter("--failure-concurrent-control is a process-local two-slot gate")
    if concurrent_calls and workspace_slots != 2:
        raise click.BadParameter("--concurrent-calls requires --workspace-slots=2")

    devices = jax.devices()
    process_count = jax.process_count()
    transport = MokLikeWorkspaceTransport(workspace_transport)
    if any(device.platform != "gpu" for device in devices):
        raise RuntimeError(f"The correctness gate requires CUDA GPUs, got {devices}")

    if transport.crosses_processes:
        # One rank per process, the layout the fabric transport requires. The expert axis spans
        # processes rather than one process's local devices, and there is no data replication:
        # this gate exists to exercise the kernel over imported peer arenas, nothing else.
        if process_count != WORLD_SIZE or jax.local_device_count() != 1:
            raise RuntimeError(
                f"{transport.value} needs {WORLD_SIZE} processes with one GPU each, got "
                f"{process_count} processes and {jax.local_device_count()} local devices"
            )
        replicas = 1
        mesh = Mesh(
            np.asarray(devices).reshape(1, 1, WORLD_SIZE, 1),
            ("replica_dcn", "data", "expert", "model"),
            axis_types=(AxisType.Explicit,) * 4,
        )
    else:
        if jax.local_device_count() != WORLD_SIZE or len(devices) != process_count * WORLD_SIZE:
            raise RuntimeError(f"The correctness gate requires four visible GPUs per process, got {devices}")
        if process_count != 1 and failure_gate is None:
            raise click.BadParameter("multi-process execution is supported only for --failure-gate")
        replicas = process_count
        mesh = Mesh(
            np.asarray(devices).reshape(process_count, 1, WORLD_SIZE, 1),
            ("replica_dcn", "data", "expert", "model"),
            axis_types=(AxisType.Explicit,) * 4,
        )
    batch_sharding = NamedSharding(mesh, P(("replica_dcn", "data", "expert")))
    expert_sharding = NamedSharding(mesh, P("expert"))
    shared_sharding = NamedSharding(mesh, P(("data", "expert"), "model"))
    router_sharding = NamedSharding(mesh, P(None, None))
    arrays = _canonical_inputs(
        num_tokens=replicas * num_tokens,
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
    )
    x, combine_weights, w_gate, w_up, w_down, shared_gate, shared_up, shared_down = arrays
    differentiable = (
        jax.device_put(jnp.asarray(x, dtype=jnp.bfloat16), batch_sharding),
        jax.device_put(jnp.asarray(combine_weights), batch_sharding),
        jax.device_put(jnp.asarray(w_gate, dtype=jnp.bfloat16), expert_sharding),
        jax.device_put(jnp.asarray(w_up, dtype=jnp.bfloat16), expert_sharding),
        jax.device_put(jnp.asarray(w_down, dtype=jnp.bfloat16), expert_sharding),
        jax.device_put(jnp.asarray(shared_gate, dtype=jnp.bfloat16), shared_sharding),
        jax.device_put(jnp.asarray(shared_up, dtype=jnp.bfloat16), shared_sharding),
        jax.device_put(jnp.asarray(shared_down, dtype=jnp.bfloat16), shared_sharding),
    )
    process_routes = _routes(num_tokens, scenario)
    routes = np.tile(process_routes, (replicas, 1, 1))
    selected_experts = jax.device_put(
        jnp.asarray(routes.reshape(-1, TOP_K)),
        batch_sharding,
    )
    router = jax.device_put(
        jnp.asarray(
            np.random.default_rng(9876).normal(size=(hidden_dim, NUM_EXPERTS)) / hidden_dim**0.5,
            dtype=jnp.float32,
        ),
        router_sharding,
    )
    output_gradient = jax.device_put(
        jnp.asarray(
            np.random.default_rng(4321).normal(size=(replicas * WORLD_SIZE * num_tokens, hidden_dim)),
            dtype=jnp.bfloat16,
        ),
        batch_sharding,
    ) * jnp.asarray(output_gradient_scale, dtype=jnp.bfloat16)
    config = MokLikeConfig(
        minibatch_size=minibatch_size,
        macrobatch_size=macrobatch_size,
        schedule_capacity_factor=schedule_capacity_factor,
        workspace_slots=workspace_slots,
        forward_x_storage=forward_x_storage,
        backward_peer_storage=backward_peer_storage,
        workspace_transport=transport,
    )
    capacity = schedule_capacity(num_tokens, TOP_K, NUM_LOCAL_EXPERTS, config)
    required_capacity = _required_schedule_capacity(process_routes)
    if capacity < required_capacity:
        raise click.BadParameter(
            f"scenario {scenario.value} requires padded schedule capacity {required_capacity}, "
            f"but the configured capacity is {capacity}; numerical parity requires zero overflow"
        )
    real_macrobuffers = _real_macrobuffers(
        process_routes,
        capacity=capacity,
        macrobatch_size=macrobatch_size,
    )
    build_config = MokLikeBuildConfig(
        source_root=MOK_LIKE_SOURCE_ROOT,
        cache_root=MOK_LIKE_BUILD_ROOT,
        cuda_arch="sm_100a",
        clone_if_missing=True,
    )
    gradient_names = (
        "x",
        "combine_weights",
        "routed_gate",
        "routed_up",
        "routed_down",
        "shared_gate",
        "shared_up",
        "shared_down",
    )

    with initialize_mok_like_runtime(
        build_config=build_config,
        num_tokens=num_tokens,
        hidden_dim=hidden_dim,
        top_k=TOP_K,
        workspace_slots=config.workspace_slots,
        mesh=mesh,
        workspace_transport=transport,
    ) as runtime:

        def fused_loss(*arguments: jax.Array) -> jax.Array:
            output, _ = mok_like_mlp(
                arguments[0],
                selected_experts,
                *arguments[1:],
                mesh=mesh,
                runtime=runtime,
                config=config,
                collective_id=0,
            )
            return jnp.sum(output.astype(jnp.float32) * output_gradient.astype(jnp.float32))

        def reference_loss(*arguments: jax.Array) -> jax.Array:
            output = mok_like_reference(
                arguments[0],
                selected_experts,
                *arguments[1:],
                mesh=mesh,
                config=config,
                fallback_implementation="ring",
            )
            return jnp.sum(output.astype(jnp.float32) * output_gradient.astype(jnp.float32))

        def fused_output(*arguments: jax.Array, collective_id: int = 0) -> jax.Array:
            return mok_like_mlp(
                arguments[0],
                selected_experts,
                *arguments[1:],
                mesh=mesh,
                runtime=runtime,
                config=config,
                collective_id=collective_id,
            )[0]

        def fused_output_and_drops(routing: jax.Array, *arguments: jax.Array) -> tuple[jax.Array, jax.Array]:
            return mok_like_mlp(
                arguments[0],
                routing,
                *arguments[1:],
                mesh=mesh,
                runtime=runtime,
                config=config,
                collective_id=0,
            )

        def reference_output(routing: jax.Array, *arguments: jax.Array) -> jax.Array:
            return mok_like_reference(
                arguments[0],
                routing,
                *arguments[1:],
                mesh=mesh,
                config=config,
                fallback_implementation="ring",
            )

        if failure_gate is not None:
            phase = (
                MokLikeTestFailurePhase.FORWARD
                if failure_gate.startswith("forward_")
                else MokLikeTestFailurePhase.BACKWARD
            )
            point = (
                MokLikeTestFailurePoint.BEFORE_INPUT_READY
                if failure_gate.endswith("input_ready")
                else MokLikeTestFailurePoint.BEFORE_COMPLETION
            )

            def failure_output(
                failure_routes: jax.Array,
                *arguments: jax.Array,
                collective_id: int,
            ) -> jax.Array:
                return mok_like_mlp(
                    arguments[0],
                    failure_routes,
                    *arguments[1:],
                    mesh=mesh,
                    runtime=runtime,
                    config=config,
                    collective_id=collective_id,
                )[0]

            def failure_loss(
                failure_routes: jax.Array,
                failure_output_gradient: jax.Array,
                *arguments: jax.Array,
                collective_id: int,
            ) -> jax.Array:
                output = failure_output(failure_routes, *arguments, collective_id=collective_id)
                return jnp.sum(output.astype(jnp.float32) * failure_output_gradient.astype(jnp.float32))

            failure_collective_ids = (0, 1) if failure_concurrent_control else (0,)
            if phase is MokLikeTestFailurePhase.FORWARD:
                differentiable_offset = 1
                failure_arguments = (selected_experts, *differentiable)
                compiled_failures = tuple(
                    jax.jit(partial(failure_output, collective_id=collective_id)).lower(*failure_arguments).compile()
                    for collective_id in failure_collective_ids
                )
            else:
                differentiable_offset = 2
                failure_arguments = (selected_experts, output_gradient, *differentiable)
                compiled_failures = tuple(
                    jax.jit(
                        jax.value_and_grad(
                            partial(failure_loss, collective_id=collective_id),
                            argnums=tuple(range(2, 10)),
                        )
                    )
                    .lower(*failure_arguments)
                    .compile()
                    for collective_id in failure_collective_ids
                )
            runtime.reset_call_counts()
            runtime.reset_debug_counters()
            # Only process zero injects. Every other process must still take the
            # same failure branch through the process-spanning status pmax.
            if jax.process_index() == 0:
                runtime.arm_test_failure(
                    rank=failure_rank,
                    phase=phase,
                    point=point,
                    require_two_active_slots=failure_concurrent_control,
                )
            failure_started = time.monotonic()
            if failure_concurrent_control:
                scaled_x_arguments = (
                    *failure_arguments[:differentiable_offset],
                    failure_arguments[differentiable_offset] * jnp.asarray(0.5, dtype=jnp.bfloat16),
                    *failure_arguments[differentiable_offset + 1 :],
                )
                concurrent_arguments = (
                    failure_arguments,
                    scaled_x_arguments,
                )
                concurrent_executions = tuple(zip(compiled_failures, concurrent_arguments, strict=True))
                start = Barrier(2)

                def run_failure_control(
                    execution: tuple[CompiledFailure, tuple[jax.Array, ...]],
                ) -> tuple[FailureResult | None, Exception | None]:
                    compiled_failure, arguments = execution
                    start.wait()
                    try:
                        result = compiled_failure(*arguments)
                        jax.block_until_ready(result)
                        return result, None
                    except Exception as failure:
                        return None, failure

                with ThreadPoolExecutor(max_workers=2) as executor:
                    results = tuple(executor.map(run_failure_control, concurrent_executions))
                errors = tuple(error for _, error in results if error is not None)
                successes = tuple((index, result) for index, (result, error) in enumerate(results) if error is None)
                if len(errors) != 1 or len(successes) != 1:
                    raise AssertionError(f"failure cancellation crossed RunIds: {results}")
                captured_failure = errors[0]
                elapsed = time.monotonic() - failure_started
                success_index, successful_result = successes[0]
                assert successful_result is not None
                successful_arguments = concurrent_arguments[success_index]
                successful_differentiable = successful_arguments[differentiable_offset:]
                if phase is MokLikeTestFailurePhase.FORWARD:
                    expected_success = jax.jit(reference_output)(*successful_differentiable)
                    jax.block_until_ready(expected_success)
                    success_metrics = _error_metrics(
                        successful_result,
                        expected_success,
                        absolute_tolerance=BF16_ATOL,
                    )
                    success_matches = bool(success_metrics["allclose"])
                else:
                    expected_success = jax.jit(jax.value_and_grad(reference_loss, argnums=tuple(range(8))))(
                        *successful_differentiable
                    )
                    jax.block_until_ready(expected_success)
                    gradient_metrics = tuple(
                        _error_metrics(
                            actual,
                            expected,
                            absolute_tolerance=(SHARED_GRADIENT_ATOL if name.startswith("shared_") else BF16_ATOL),
                        )
                        for name, actual, expected in zip(
                            gradient_names,
                            successful_result[1],
                            expected_success[1],
                            strict=True,
                        )
                    )
                    success_metrics = {
                        "loss_absolute_error": float(abs(successful_result[0] - expected_success[0])),
                        "gradients": gradient_metrics,
                    }
                    loss_matches = bool(
                        np.isclose(
                            float(successful_result[0]),
                            float(expected_success[0]),
                            atol=BF16_ATOL,
                            rtol=BF16_RTOL,
                        )
                    )
                    success_metrics["loss_allclose"] = loss_matches
                    success_matches = loss_matches and all(bool(metric["allclose"]) for metric in gradient_metrics)
                if not success_matches:
                    raise AssertionError(f"concurrent control result did not match the reference: {success_metrics}")
            else:
                try:
                    failed_result = compiled_failures[0](*failure_arguments)
                    jax.block_until_ready(failed_result)
                except Exception as failure:
                    captured_failure = failure
                else:
                    raise AssertionError("mok_like accepted an injected rank-local failure")
                success_metrics = None
                elapsed = time.monotonic() - failure_started
            if elapsed >= 60:
                raise AssertionError(f"failure gate exceeded the 60-second bound: {elapsed:.3f}s")
            expected_calls_per_execution = (
                (WORLD_SIZE, 0) if phase is MokLikeTestFailurePhase.FORWARD else (WORLD_SIZE, WORLD_SIZE)
            )
            execution_count = 2 if failure_concurrent_control else 1
            expected_call_counts = tuple(execution_count * count for count in expected_calls_per_execution)
            call_counts = runtime.call_counts()
            debug_counters = runtime.debug_counters()
            if call_counts != expected_call_counts:
                raise AssertionError(
                    f"failure gate had unexpected FFI calls: {call_counts}, expected {expected_call_counts}"
                )
            if any(debug_counters.generation_mismatches) or any(debug_counters.slot_reuse_failures):
                raise AssertionError(f"failure gate violated generation/slot isolation: {debug_counters}")
            phase_count = 1 if phase is MokLikeTestFailurePhase.FORWARD else 2
            expected_acquisitions = execution_count * phase_count
            if any(sum(acquisitions) != expected_acquisitions for acquisitions in debug_counters.slot_acquisitions):
                raise AssertionError(f"failure gate had unexpected slot acquisitions: {debug_counters}")
            if failure_concurrent_control:
                expected_per_slot = (phase_count, phase_count)
                if any(acquisitions != expected_per_slot for acquisitions in debug_counters.slot_acquisitions):
                    raise AssertionError(f"concurrent failure gate did not isolate both slots: {debug_counters}")
                if any(maximum < 2 for maximum in debug_counters.max_active_slots):
                    raise AssertionError(f"concurrent failure gate did not overlap both slots: {debug_counters}")
            elif any(maximum != 1 for maximum in debug_counters.max_active_slots):
                raise AssertionError(f"isolated failure gate had unexpected slot overlap: {debug_counters}")
            if "Mixture-of-Kittens failed on at least one mesh rank" in str(captured_failure):
                metrics = {
                    "failure_gate": failure_gate,
                    "failure_rank": failure_rank,
                    "process_count": process_count,
                    "process_index": jax.process_index(),
                    "concurrent_control": failure_concurrent_control,
                    "ffi_call_counts": call_counts,
                    "slot_acquisitions": debug_counters.slot_acquisitions,
                    "max_active_slots": debug_counters.max_active_slots,
                    "generation_mismatches": debug_counters.generation_mismatches,
                    "slot_reuse_failures": debug_counters.slot_reuse_failures,
                    "successful_control": success_metrics,
                    "wall_time_seconds": elapsed,
                }
                # Shutdown is part of the negative gate: failure closure is not
                # complete if the runtime cannot retire every leased slot.
                runtime.close()
                print(json.dumps(metrics, sort_keys=True))
                return
            raise AssertionError(
                f"all-rank failure gate raised the wrong error: {captured_failure}"
            ) from captured_failure

        if corrupt_stamp:

            def corrupted_stamp_execution(*arguments: jax.Array) -> jax.Array:
                _, _, context = _fused_forward_with_context(
                    arguments[0],
                    selected_experts,
                    *arguments[1:],
                    mesh=mesh,
                    runtime=runtime,
                    config=config,
                    collective_id=20,
                )
                corrupt_context = context._replace(stamp_runtime_epoch=context.stamp_runtime_epoch + 1)
                gradients = _fused_backward(
                    output_gradient,
                    arguments[0],
                    selected_experts,
                    arguments[1],
                    arguments[2],
                    arguments[3],
                    arguments[4],
                    arguments[5],
                    arguments[6],
                    arguments[7],
                    corrupt_context,
                    mesh=mesh,
                    runtime=runtime,
                    config=config,
                    collective_id=20,
                )
                return sum(
                    (jnp.sum(gradient.astype(jnp.float32)) for gradient in gradients),
                    start=jnp.asarray(0.0, dtype=jnp.float32),
                )

            corrupted_result = jax.jit(corrupted_stamp_execution)(*differentiable)
            corrupted_result.block_until_ready()
            raise AssertionError("mok_like accepted a corrupt saved runtime stamp")

        def combine_weights_from_router(router_matrix: jax.Array) -> jax.Array:
            logits = jnp.einsum("td,de->te", differentiable[0], router_matrix).astype(jnp.float32)
            selected_logits = jnp.take_along_axis(logits, selected_experts, axis=-1)
            weights = jax.nn.sigmoid(selected_logits)
            return weights * (2.5 / (jnp.sum(weights, axis=-1, keepdims=True) + 1e-9))

        def fused_router_loss(router_matrix: jax.Array) -> jax.Array:
            arguments = (differentiable[0], combine_weights_from_router(router_matrix), *differentiable[2:])
            return fused_loss(*arguments)

        def reference_router_loss(router_matrix: jax.Array) -> jax.Array:
            arguments = (differentiable[0], combine_weights_from_router(router_matrix), *differentiable[2:])
            return reference_loss(*arguments)

        actual_output, fused_dropped_assignments = jax.jit(fused_output_and_drops)(
            selected_experts, *differentiable
        )
        expected_output = jax.jit(reference_output)(selected_experts, *differentiable)
        actual_output.block_until_ready()
        expected_output.block_until_ready()
        if forward_only:
            # The forward comparison alone answers whether the kernel executes and agrees with the
            # reference. The rest of the matrix closes over sharded arrays, which is legal only
            # when one process owns every device, so it cannot run under the fabric transport yet.
            difference = float(
                jnp.max(jnp.abs(actual_output.astype(jnp.float32) - expected_output.astype(jnp.float32)))
            )
            dropped = int(jnp.sum(fused_dropped_assignments))
            print(f"FORWARD max_abs_difference={difference:.6f} dropped_assignments={dropped}", flush=True)
            if difference > BF16_ATOL:
                raise RuntimeError(f"forward output differs from the reference by {difference}")
            print("FORWARD PASS", flush=True)
            return
        if offload:
            differentiated_fused_loss = jax.checkpoint(
                fused_loss,
                policy=jax.checkpoint_policies.save_and_offload_only_these_names(
                    names_which_can_be_saved=(),
                    names_which_can_be_offloaded=(MOK_CONTEXT_CHECKPOINT_NAME,),
                    offload_src="device",
                    offload_dst="pinned_host",
                ),
            )
        elif remat:
            differentiated_fused_loss = jax.checkpoint(
                fused_loss,
                policy=jax.checkpoint_policies.save_only_these_names(MOK_CONTEXT_CHECKPOINT_NAME),
            )
        else:
            differentiated_fused_loss = fused_loss
        runtime.reset_call_counts()
        runtime.reset_debug_counters()
        fused_value, fused_gradients = jax.jit(jax.value_and_grad(differentiated_fused_loss, argnums=tuple(range(8))))(
            *differentiable
        )
        fused_value.block_until_ready()
        call_counts = runtime.call_counts()
        debug_counters = runtime.debug_counters()
        reference_value, reference_gradients = jax.jit(jax.value_and_grad(reference_loss, argnums=tuple(range(8))))(
            *differentiable
        )
        reference_value.block_until_ready()
        fused_router_gradient = jax.jit(jax.grad(fused_router_loss))(router)
        reference_router_gradient = jax.jit(jax.grad(reference_router_loss))(router)
        fused_router_gradient.block_until_ready()
        reference_router_gradient.block_until_ready()

        stress_metrics: dict[str, object] = {}
        if back_to_back:

            def fused_chained_loss(x_first: jax.Array, x_second: jax.Array, *arguments: jax.Array) -> jax.Array:
                first = fused_output(x_first, *arguments, collective_id=10)
                chained_x = x_second + jnp.asarray(1 / 128, dtype=jnp.bfloat16) * jnp.tanh(first)
                second = fused_output(chained_x, *arguments, collective_id=11)
                return jnp.asarray(0.5, dtype=jnp.float32) * jnp.sum(
                    (first + second).astype(jnp.float32) * output_gradient.astype(jnp.float32)
                )

            def reference_chained_loss(x_first: jax.Array, x_second: jax.Array, *arguments: jax.Array) -> jax.Array:
                first = reference_output(x_first, *arguments)
                chained_x = x_second + jnp.asarray(1 / 128, dtype=jnp.bfloat16) * jnp.tanh(first)
                second = reference_output(chained_x, *arguments)
                return jnp.asarray(0.5, dtype=jnp.float32) * jnp.sum(
                    (first + second).astype(jnp.float32) * output_gradient.astype(jnp.float32)
                )

            chained_arguments = (
                differentiable[0],
                differentiable[0] + jnp.asarray(0.125, dtype=jnp.bfloat16),
                *differentiable[1:],
            )
            compiled_chained = (
                jax.jit(jax.value_and_grad(fused_chained_loss, argnums=tuple(range(9))))
                .lower(*chained_arguments)
                .compile()
            )
            runtime.reset_call_counts()
            runtime.reset_debug_counters()
            actual_twice = compiled_chained(*chained_arguments)
            jax.block_until_ready(actual_twice)
            twice_call_counts = runtime.call_counts()
            twice_counters = runtime.debug_counters()
            expected_twice = jax.jit(jax.value_and_grad(reference_chained_loss, argnums=tuple(range(9))))(
                *chained_arguments
            )
            jax.block_until_ready(expected_twice)
            chained_gradient_names = ("x_first", "x_second", *gradient_names[1:])
            twice_gradient_metrics = {
                name: _error_metrics(
                    actual,
                    expected,
                    absolute_tolerance=SHARED_GRADIENT_ATOL if name.startswith("shared_") else BF16_ATOL,
                )
                for name, actual, expected in zip(
                    chained_gradient_names,
                    actual_twice[1],
                    expected_twice[1],
                    strict=True,
                )
            }
            slot_acquisitions_per_rank = tuple(sum(acquisitions) for acquisitions in twice_counters.slot_acquisitions)
            stress_metrics["back_to_back"] = {
                "loss_absolute_error": float(abs(actual_twice[0] - expected_twice[0])),
                "gradients": twice_gradient_metrics,
                "ffi_call_counts": twice_call_counts,
                "slot_acquisitions": twice_counters.slot_acquisitions,
                "max_active_slots": twice_counters.max_active_slots,
                "generation_mismatches": twice_counters.generation_mismatches,
                "slot_reuse_failures": twice_counters.slot_reuse_failures,
            }
            print(json.dumps({"stress_probe": {"back_to_back": stress_metrics["back_to_back"]}}, sort_keys=True))
            if twice_call_counts != (2 * WORLD_SIZE, 2 * WORLD_SIZE):
                raise AssertionError(
                    f"back-to-back forward/backward calls were replayed or eliminated: {twice_call_counts}"
                )
            if not all(metric["allclose"] for metric in twice_gradient_metrics.values()):
                raise AssertionError("back-to-back mok_like gradients do not match the ordinary Grug EP reference")
            if slot_acquisitions_per_rank != (4,) * WORLD_SIZE:
                raise AssertionError(
                    f"back-to-back calls did not acquire four phase-local slots per rank: {twice_counters}"
                )
            if any(twice_counters.generation_mismatches) or any(twice_counters.slot_reuse_failures):
                raise AssertionError(f"back-to-back slot protocol failed: {twice_counters}")

        if concurrent_calls:
            compiled_fused_loss = (
                jax.jit(jax.value_and_grad(fused_loss, argnums=tuple(range(8)))).lower(*differentiable).compile()
            )
            concurrent_arguments = (
                differentiable,
                (differentiable[0] * jnp.asarray(0.5, dtype=jnp.bfloat16), *differentiable[1:]),
            )
            compiled_reference_loss = (
                jax.jit(jax.value_and_grad(reference_loss, argnums=tuple(range(8)))).lower(*differentiable).compile()
            )
            expected_concurrent = tuple(compiled_reference_loss(*arguments) for arguments in concurrent_arguments)
            jax.block_until_ready(expected_concurrent)
            warmup = compiled_fused_loss(*differentiable)
            jax.block_until_ready(warmup)
            runtime.reset_call_counts()
            runtime.reset_debug_counters()
            start = Barrier(2)

            def run_compiled(arguments: tuple[jax.Array, ...]) -> tuple[jax.Array, tuple[jax.Array, ...]]:
                start.wait()
                result = compiled_fused_loss(*arguments)
                jax.block_until_ready(result)
                return result

            with ThreadPoolExecutor(max_workers=2) as executor:
                actual_concurrent = tuple(executor.map(run_compiled, concurrent_arguments))
            concurrent_call_counts = runtime.call_counts()
            concurrent_counters = runtime.debug_counters()
            concurrent_gradient_metrics = tuple(
                {
                    name: _error_metrics(
                        actual,
                        expected,
                        absolute_tolerance=(SHARED_GRADIENT_ATOL if name.startswith("shared_") else BF16_ATOL),
                    )
                    for name, actual, expected in zip(
                        gradient_names,
                        actual_result[1],
                        expected_result[1],
                        strict=True,
                    )
                }
                for actual_result, expected_result in zip(actual_concurrent, expected_concurrent, strict=True)
            )
            stress_metrics["concurrent_calls"] = {
                "loss_absolute_errors": tuple(
                    float(abs(actual[0] - expected[0]))
                    for actual, expected in zip(actual_concurrent, expected_concurrent, strict=True)
                ),
                "gradients": concurrent_gradient_metrics,
                "ffi_call_counts": concurrent_call_counts,
                "slot_acquisitions": concurrent_counters.slot_acquisitions,
                "max_active_slots": concurrent_counters.max_active_slots,
                "generation_mismatches": concurrent_counters.generation_mismatches,
                "slot_reuse_failures": concurrent_counters.slot_reuse_failures,
            }
            print(
                json.dumps(
                    {"stress_probe": {"concurrent_calls": stress_metrics["concurrent_calls"]}},
                    sort_keys=True,
                )
            )
            if concurrent_call_counts != (2 * WORLD_SIZE, 2 * WORLD_SIZE):
                raise AssertionError(
                    f"concurrent forward/backward calls were replayed or eliminated: {concurrent_call_counts}"
                )
            if not all(
                metric["allclose"]
                for result_metrics in concurrent_gradient_metrics
                for metric in result_metrics.values()
            ):
                raise AssertionError("concurrent mok_like gradients do not match the ordinary Grug EP reference")
            if any(concurrent_counters.generation_mismatches) or any(concurrent_counters.slot_reuse_failures):
                raise AssertionError(f"concurrent slot protocol failed: {concurrent_counters}")
            if any(sum(acquisitions) != 4 for acquisitions in concurrent_counters.slot_acquisitions):
                raise AssertionError(f"concurrent calls had the wrong slot acquisition count: {concurrent_counters}")
            if any(not all(acquisitions) for acquisitions in concurrent_counters.slot_acquisitions):
                raise AssertionError(f"concurrent calls did not use both runtime slots: {concurrent_counters}")
            if any(maximum < 2 for maximum in concurrent_counters.max_active_slots):
                raise AssertionError(f"concurrent calls did not occupy both runtime slots: {concurrent_counters}")

    gradient_metrics = {
        name: _error_metrics(
            actual,
            expected,
            absolute_tolerance=SHARED_GRADIENT_ATOL if name.startswith("shared_") else BF16_ATOL,
        )
        for name, actual, expected in zip(gradient_names, fused_gradients, reference_gradients, strict=True)
    }
    router_gradient_metrics = _error_metrics(
        fused_router_gradient,
        reference_router_gradient,
        absolute_tolerance=BF16_ATOL,
    )
    router_gradient_metrics["allclose"] = (
        float(router_gradient_metrics["relative_l2_error"]) <= ROUTER_GRADIENT_RELATIVE_L2_TOLERANCE
    )
    gradient_metrics["router"] = router_gradient_metrics
    inactive_routed_gradient_maxima: dict[str, float] = {}
    if scenario is RouteScenario.ZERO_TOKEN_EXPERT:
        for name, gradient in zip(gradient_names[2:5], fused_gradients[2:5], strict=True):
            inactive = np.asarray(jax.device_get(gradient), dtype=np.float32)[1::NUM_LOCAL_EXPERTS]
            inactive_routed_gradient_maxima[name] = float(np.max(np.abs(inactive)))
    forward_metrics = _error_metrics(actual_output, expected_output, absolute_tolerance=BF16_ATOL)
    fused_dropped_assignments_value = int(jax.device_get(fused_dropped_assignments))
    if call_counts[0] % WORLD_SIZE != 0 or call_counts[1] % WORLD_SIZE != 0:
        raise AssertionError(f"FFI call counts do not divide evenly across ranks: {call_counts}")
    forward_invocations = call_counts[0] // WORLD_SIZE
    backward_invocations = call_counts[1] // WORLD_SIZE
    activation_bytes = num_tokens * hidden_dim * jnp.dtype(jnp.bfloat16).itemsize
    router_bytes = num_tokens * TOP_K * jnp.dtype(jnp.float32).itemsize
    expected_forward_staging = (
        (forward_invocations, forward_invocations * activation_bytes)
        if forward_x_storage is MokLikeForwardXStorage.RUNTIME_STAGED
        else (0, 0)
    )
    if backward_peer_storage is MokLikeBackwardPeerStorage.RUNTIME_STAGED:
        expected_backward_staging = (
            4 * backward_invocations,
            backward_invocations * (2 * activation_bytes + 2 * router_bytes),
        )
    elif backward_peer_storage is MokLikeBackwardPeerStorage.XLA_PEER_INPUTS_EXPERIMENTAL:
        expected_backward_staging = (backward_invocations, backward_invocations * router_bytes)
    else:
        expected_backward_staging = (0, 0)
    expected_staging_copy_calls = (expected_forward_staging[0], expected_backward_staging[0])
    expected_staging_copy_bytes = (expected_forward_staging[1], expected_backward_staging[1])
    metrics = {
        "backend": "mok_like",
        "scenario": scenario.value,
        "forward_x_storage": forward_x_storage.value,
        "backward_peer_storage": backward_peer_storage.value,
        "workspace_slots": workspace_slots,
        "remat": "offload" if offload else "save" if remat else "none",
        "output_gradient_scale": output_gradient_scale,
        "ffi_call_counts": {"forward": call_counts[0], "backward": call_counts[1]},
        "runtime_debug_counters": {
            "peer_ready_waits": debug_counters.peer_ready_waits,
            "completion_waits": debug_counters.completion_waits,
            "generation_mismatches": debug_counters.generation_mismatches,
            "slot_reuse_failures": debug_counters.slot_reuse_failures,
            "slot_acquisitions": debug_counters.slot_acquisitions,
            "max_active_slots": debug_counters.max_active_slots,
            "peer_wait_events": debug_counters.peer_wait_events,
            "peer_wait_cycles": debug_counters.peer_wait_cycles,
            "peer_wait_max_cycles": debug_counters.peer_wait_max_cycles,
            "staging_copy_calls": debug_counters.staging_copy_calls,
            "staging_copy_bytes": debug_counters.staging_copy_bytes,
        },
        "schedule_capacity": capacity,
        "required_schedule_capacity": required_capacity,
        "dropped_assignments": fused_dropped_assignments_value,
        "real_macrobuffers": real_macrobuffers,
        "forward": forward_metrics,
        "loss_absolute_error": float(abs(fused_value - reference_value)),
        "gradients": gradient_metrics,
        "inactive_routed_gradient_maxima": inactive_routed_gradient_maxima,
        "stress": stress_metrics,
    }
    print(json.dumps(metrics, sort_keys=True))
    if not forward_metrics["allclose"] or not all(metric["allclose"] for metric in gradient_metrics.values()):
        raise AssertionError("mok_like gradients do not match the ordinary Grug EP reference")
    if fused_dropped_assignments_value != 0:
        raise AssertionError(f"numerical parity gate dropped {fused_dropped_assignments_value} assignments")
    if any(maximum != 0.0 for maximum in inactive_routed_gradient_maxima.values()):
        raise AssertionError(f"zero-token expert received a weight gradient: {inactive_routed_gradient_maxima}")
    if (remat or offload) and call_counts != (WORLD_SIZE, WORLD_SIZE):
        raise AssertionError(
            f"saved context replayed native work: expected {(WORLD_SIZE, WORLD_SIZE)}, got {call_counts}"
        )
    if any(debug_counters.generation_mismatches):
        raise AssertionError(f"peer readiness accepted a future generation: {debug_counters.generation_mismatches}")
    if any(debug_counters.slot_reuse_failures):
        raise AssertionError(f"runtime workspace was reused while active: {debug_counters.slot_reuse_failures}")
    if any(calls != expected_staging_copy_calls for calls in debug_counters.staging_copy_calls):
        raise AssertionError(
            f"unexpected staging copy calls: expected {expected_staging_copy_calls} per rank, "
            f"got {debug_counters.staging_copy_calls}"
        )
    if any(bytes_ != expected_staging_copy_bytes for bytes_ in debug_counters.staging_copy_bytes):
        raise AssertionError(
            f"unexpected staging copy bytes: expected {expected_staging_copy_bytes} per rank, "
            f"got {debug_counters.staging_copy_bytes}"
        )
    for rank in range(WORLD_SIZE):
        for phase in range(4):
            for peer in range(WORLD_SIZE):
                events = debug_counters.peer_wait_events[rank][phase][peer]
                cycles = debug_counters.peer_wait_cycles[rank][phase][peer]
                maximum = debug_counters.peer_wait_max_cycles[rank][phase][peer]
                if (events == 0) != (cycles == 0 and maximum == 0) or maximum > cycles:
                    raise AssertionError(
                        f"invalid peer-wait counters at rank={rank}, phase={phase}, peer={peer}: "
                        f"events={events}, cycles={cycles}, max={maximum}"
                    )
                if rank == peer and (events != 0 or cycles != 0 or maximum != 0):
                    raise AssertionError(
                        f"rank {rank} waited on itself during phase {phase}: "
                        f"events={events}, cycles={cycles}, max={maximum}"
                    )
        ready_events = sum(
            debug_counters.peer_wait_events[rank][phase][peer] for phase in (0, 2) for peer in range(WORLD_SIZE)
        )
        completion_events = sum(
            debug_counters.peer_wait_events[rank][phase][peer] for phase in (1, 3) for peer in range(WORLD_SIZE)
        )
        if ready_events != debug_counters.peer_ready_waits[rank]:
            raise AssertionError(
                f"rank {rank} ready-wait cells total {ready_events}, aggregate is "
                f"{debug_counters.peer_ready_waits[rank]}"
            )
        if completion_events != debug_counters.completion_waits[rank]:
            raise AssertionError(
                f"rank {rank} completion-wait cells total {completion_events}, aggregate is "
                f"{debug_counters.completion_waits[rank]}"
            )
    if expected_real_macrobuffers is not None and real_macrobuffers != expected_real_macrobuffers:
        raise AssertionError(f"expected {expected_real_macrobuffers} real macrobuffers, got {real_macrobuffers}")


def _run_cli() -> None:
    primary_failure: BaseException | None = None
    try:
        main()
    except BaseException as failure:
        primary_failure = failure
        raise
    finally:
        if jax.distributed.is_initialized():
            try:
                jax.distributed.shutdown()
            except BaseException as shutdown_failure:
                if primary_failure is None:
                    raise
                primary_failure.add_note(f"JAX distributed shutdown also failed: {shutdown_failure}")


if __name__ == "__main__":
    _run_cli()
