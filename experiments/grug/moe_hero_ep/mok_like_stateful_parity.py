# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Long-running stateful numerical parity gate for the supported mok_like API."""

import json
import os
import re
from collections.abc import Mapping, Sequence
from typing import NamedTuple

import click
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from levanter.kernels.mixture_of_kittens import (
    MokLikeBackwardPeerStorage,
    MokLikeBuildConfig,
    MokLikeConfig,
    MokLikeForwardXStorage,
    initialize_mok_like_runtime,
    mok_like_mlp,
    mok_like_reference,
)
from levanter.kernels.mixture_of_kittens.schedule import schedule_capacity

from experiments.grug.moe_hero_ep.mok_like_correctness import (
    BF16_ATOL,
    BF16_RTOL,
    DEFAULT_TOP_K,
    MOK_LIKE_BUILD_ROOT,
    MOK_LIKE_SOURCE_ROOT,
    NUM_EXPERTS,
    NUM_LOCAL_EXPERTS,
    SHARED_GRADIENT_ATOL,
    WORLD_SIZE,
    RouteScenario,
    _required_schedule_capacity,
    _routes,
)

# This harness pins one routing shape across every update, so it takes the correctness gate's
# default rather than exposing a knob of its own.
TOP_K = DEFAULT_TOP_K

DEFAULT_NUM_UPDATES = 64
DEFAULT_SEED = 20260812
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_MOMENTUM = 0.9
PARAMETER_NAMES = (
    "routed_gate",
    "routed_up",
    "routed_down",
    "shared_gate",
    "shared_up",
    "shared_down",
)
GRADIENT_NAMES = ("x", "combine_weights", *PARAMETER_NAMES)
_IRIS_TASK_INDEX = re.compile(r"/(\d+)(?::\d+)?$")


class StepInputs(NamedTuple):
    x: np.ndarray
    combine_weights: np.ndarray
    target: np.ndarray


class ReducedErrorValues(NamedTuple):
    allclose: jax.Array
    max_absolute_error: jax.Array
    mean_absolute_error: jax.Array
    mismatch_fraction: jax.Array
    relative_l2_error: jax.Array


class FusedStepResult(NamedTuple):
    parameters: tuple[jax.Array, ...]
    optimizer_state: tuple[jax.Array, ...]
    loss: jax.Array
    output: jax.Array
    dropped_assignments: jax.Array
    gradients: tuple[jax.Array, ...]


class ReferenceStepResult(NamedTuple):
    parameters: tuple[jax.Array, ...]
    optimizer_state: tuple[jax.Array, ...]
    loss: jax.Array
    output: jax.Array
    gradients: tuple[jax.Array, ...]


def _stateful_route_plan(num_updates: int) -> tuple[RouteScenario, ...]:
    """Alternate maximum-capacity routes with distinct imbalance patterns."""

    if num_updates < 1:
        raise ValueError("num_updates must be positive")
    plan = [RouteScenario.BALANCED]
    for step in range(1, num_updates):
        if step % 2 == 0:
            plan.append(RouteScenario.ALL_TO_ONE)
        elif step % 4 == 1:
            plan.append(RouteScenario.ZERO_TOKEN_EXPERT)
        else:
            plan.append(RouteScenario.SKEWED)
    return tuple(plan)


def _replica_identity(
    replica_index: int | None,
    replica_count: int | None,
    *,
    environ: Mapping[str, str] = os.environ,
) -> tuple[int, int]:
    """Resolve an independent parity replica from explicit flags or Iris task metadata."""

    count = replica_count if replica_count is not None else int(environ.get("IRIS_NUM_TASKS", "1"))
    if replica_index is None:
        task_id = environ.get("IRIS_TASK_ID", "")
        match = _IRIS_TASK_INDEX.search(task_id)
        replica_index = int(match.group(1)) if match else 0
    if count < 1:
        raise ValueError("replica_count must be positive")
    if replica_index < 0 or replica_index >= count:
        raise ValueError(f"replica_index must be in [0, {count}), got {replica_index}")
    return replica_index, count


def _step_seed(seed: int, replica_index: int, step: int) -> np.random.SeedSequence:
    return np.random.SeedSequence((seed, replica_index, step))


def _step_inputs(
    *,
    seed: int,
    replica_index: int,
    step: int,
    num_tokens: int,
    hidden_dim: int,
) -> StepInputs:
    random = np.random.default_rng(_step_seed(seed, replica_index, step))
    global_tokens = WORLD_SIZE * num_tokens
    x = random.normal(size=(global_tokens, hidden_dim)).astype(np.float32)
    logits = random.normal(size=(global_tokens, TOP_K)).astype(np.float32)
    weights = 1.0 / (1.0 + np.exp(-logits))
    combine_weights = weights * (2.5 / np.sum(weights, axis=-1, keepdims=True))
    target = random.normal(size=(global_tokens, hidden_dim)).astype(np.float32)
    return StepInputs(x=x, combine_weights=combine_weights.astype(np.float32), target=target)


def _initial_parameters(
    *, seed: int, replica_index: int, hidden_dim: int, intermediate_dim: int
) -> tuple[np.ndarray, ...]:
    random = np.random.default_rng(np.random.SeedSequence((seed, replica_index, 0x4D4F4B)))
    routed_scale = hidden_dim**-0.5
    down_scale = intermediate_dim**-0.5
    return (
        (random.normal(size=(NUM_EXPERTS, hidden_dim, intermediate_dim)) * routed_scale).astype(np.float32),
        (random.normal(size=(NUM_EXPERTS, hidden_dim, intermediate_dim)) * routed_scale).astype(np.float32),
        (random.normal(size=(NUM_EXPERTS, intermediate_dim, hidden_dim)) * down_scale).astype(np.float32),
        (random.normal(size=(hidden_dim, intermediate_dim)) * routed_scale).astype(np.float32),
        (random.normal(size=(hidden_dim, intermediate_dim)) * routed_scale).astype(np.float32),
        (random.normal(size=(intermediate_dim, hidden_dim)) * down_scale).astype(np.float32),
    )


def _route_metrics(routes: np.ndarray, *, capacity: int) -> dict[str, object]:
    assignment_counts = np.bincount(routes.reshape(-1), minlength=NUM_EXPERTS)
    required_capacity = _required_schedule_capacity(routes)
    return {
        "assignment_counts": assignment_counts.tolist(),
        "zero_token_experts": np.flatnonzero(assignment_counts == 0).tolist(),
        "required_schedule_capacity": required_capacity,
        "schedule_capacity": capacity,
        "at_capacity_boundary": required_capacity == capacity,
    }


@jax.jit
def _reduced_error_values(
    actual: jax.Array,
    expected: jax.Array,
    absolute_tolerance: jax.Array,
) -> ReducedErrorValues:
    actual_float = actual.astype(jnp.float32)
    expected_float = expected.astype(jnp.float32)
    absolute_error = jnp.abs(actual_float - expected_float)
    close = jnp.isclose(actual_float, expected_float, atol=absolute_tolerance, rtol=BF16_RTOL)
    reference_l2 = jnp.linalg.norm(expected_float)
    return ReducedErrorValues(
        allclose=jnp.all(close),
        max_absolute_error=jnp.max(absolute_error),
        mean_absolute_error=jnp.mean(absolute_error),
        mismatch_fraction=jnp.mean(~close),
        relative_l2_error=jnp.linalg.norm(actual_float - expected_float) / jnp.maximum(reference_l2, 1e-12),
    )


def _reduced_error_metrics(
    actual: jax.Array,
    expected: jax.Array,
    *,
    absolute_tolerance: float,
) -> dict[str, float | bool]:
    values = jax.device_get(_reduced_error_values(actual, expected, jnp.asarray(absolute_tolerance, dtype=jnp.float32)))
    return {
        "allclose": bool(values.allclose),
        "absolute_tolerance": absolute_tolerance,
        "max_absolute_error": float(values.max_absolute_error),
        "mean_absolute_error": float(values.mean_absolute_error),
        "mismatch_fraction": float(values.mismatch_fraction),
        "relative_l2_error": float(values.relative_l2_error),
    }


def _tree_error_metrics(
    names: Sequence[str],
    actual: Sequence[jax.Array],
    expected: Sequence[jax.Array],
) -> dict[str, dict[str, float | bool]]:
    return {
        name: _reduced_error_metrics(
            actual_leaf,
            expected_leaf,
            absolute_tolerance=SHARED_GRADIENT_ATOL if name.startswith("shared_") else BF16_ATOL,
        )
        for name, actual_leaf, expected_leaf in zip(names, actual, expected, strict=True)
    }


def _assert_step_parity(step_metrics: dict[str, object]) -> None:
    metric_groups = (
        step_metrics["output"],
        step_metrics["loss"],
        *step_metrics["gradients"].values(),
        *step_metrics["parameters"].values(),
        *step_metrics["optimizer_state"].values(),
    )
    failed = [metric for metric in metric_groups if not metric["allclose"]]
    if failed:
        raise AssertionError(f"stateful mok_like parity diverged at step {step_metrics['step']}: {failed}")
    if step_metrics["dropped_assignments"] != 0:
        raise AssertionError(
            f"stateful mok_like parity dropped assignments at step {step_metrics['step']}: "
            f"{step_metrics['dropped_assignments']}"
        )


def _validated_inactive_expert_gradient_maxima(
    routes: np.ndarray,
    gradients: Sequence[jax.Array],
) -> dict[str, float]:
    active_experts = np.unique(routes)
    inactive_experts = np.setdiff1d(np.arange(NUM_EXPERTS), active_experts)
    if not len(inactive_experts):
        return {}
    inactive_mask = jnp.asarray(np.isin(np.arange(NUM_EXPERTS), inactive_experts))[:, None, None]
    maxima: dict[str, float] = {}
    for name, gradient in zip(PARAMETER_NAMES[:3], gradients[:3], strict=True):
        inactive = jnp.where(inactive_mask, jnp.abs(gradient.astype(jnp.float32)), 0)
        maximum = float(jax.device_get(jnp.max(inactive)))
        maxima[name] = maximum
    if any(maximum != 0.0 for maximum in maxima.values()):
        raise AssertionError(f"inactive routed expert received a gradient: {maxima}")
    return maxima


def _optimizer_update(
    parameters: tuple[jax.Array, ...],
    momentum_state: tuple[jax.Array, ...],
    gradients: tuple[jax.Array, ...],
    *,
    learning_rate: float,
    momentum: float,
) -> tuple[tuple[jax.Array, ...], tuple[jax.Array, ...]]:
    next_momentum = jax.tree.map(
        lambda state, gradient: momentum * state + gradient.astype(jnp.float32),
        momentum_state,
        gradients,
    )
    next_parameters = jax.tree.map(
        lambda parameter, state: parameter.astype(jnp.float32) - learning_rate * state,
        parameters,
        next_momentum,
    )
    return next_parameters, next_momentum


@click.command()
@click.option("--num-updates", type=click.IntRange(min=4), default=DEFAULT_NUM_UPDATES, show_default=True)
@click.option("--seed", type=int, default=DEFAULT_SEED, show_default=True)
@click.option("--replica-index", type=click.IntRange(min=0))
@click.option("--replica-count", type=click.IntRange(min=1))
@click.option("--num-tokens", type=click.IntRange(min=256), default=512, show_default=True)
@click.option("--hidden-dim", type=click.IntRange(min=256), default=512, show_default=True)
@click.option("--intermediate-dim", type=click.IntRange(min=256), default=3072, show_default=True)
@click.option("--minibatch-size", type=click.IntRange(min=256), default=256, show_default=True)
@click.option("--macrobatch-size", type=click.IntRange(min=256), default=256, show_default=True)
@click.option("--schedule-capacity-factor", type=click.FloatRange(min=1.0), default=3.75, show_default=True)
@click.option("--workspace-slots", type=click.IntRange(min=1, max=2), default=1, show_default=True)
@click.option(
    "--forward-x-storage",
    type=click.Choice(MokLikeForwardXStorage, case_sensitive=False),
    default=MokLikeForwardXStorage.XLA_PEER_EXPERIMENTAL,
    show_default=True,
)
@click.option(
    "--backward-peer-storage",
    type=click.Choice(MokLikeBackwardPeerStorage, case_sensitive=False),
    default=MokLikeBackwardPeerStorage.RUNTIME_STAGED,
    show_default=True,
)
@click.option("--learning-rate", type=click.FloatRange(min=0.0, min_open=True), default=DEFAULT_LEARNING_RATE)
@click.option("--momentum", type=click.FloatRange(min=0.0, max=1.0, max_open=True), default=DEFAULT_MOMENTUM)
def main(
    num_updates: int,
    seed: int,
    replica_index: int | None,
    replica_count: int | None,
    num_tokens: int,
    hidden_dim: int,
    intermediate_dim: int,
    minibatch_size: int,
    macrobatch_size: int,
    schedule_capacity_factor: float,
    workspace_slots: int,
    forward_x_storage: MokLikeForwardXStorage,
    backward_peer_storage: MokLikeBackwardPeerStorage,
    learning_rate: float,
    momentum: float,
) -> None:
    """Compare sequential MoK updates with the independent ring reference."""

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
    try:
        resolved_replica_index, resolved_replica_count = _replica_identity(replica_index, replica_count)
    except ValueError as error:
        raise click.BadParameter(str(error)) from error

    devices = jax.devices()
    if len(devices) != WORLD_SIZE or any(device.platform != "gpu" for device in devices):
        raise RuntimeError(f"The stateful parity gate requires four visible GPUs, got {devices}")
    mesh = Mesh(
        np.asarray(devices).reshape(1, 1, WORLD_SIZE, 1),
        ("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit,) * 4,
    )
    batch_sharding = NamedSharding(mesh, P(("replica_dcn", "data", "expert")))
    expert_sharding = NamedSharding(mesh, P("expert"))
    shared_sharding = NamedSharding(mesh, P(("data", "expert"), "model"))
    config = MokLikeConfig(
        minibatch_size=minibatch_size,
        macrobatch_size=macrobatch_size,
        schedule_capacity_factor=schedule_capacity_factor,
        workspace_slots=workspace_slots,
        forward_x_storage=forward_x_storage,
        backward_peer_storage=backward_peer_storage,
    )
    capacity = schedule_capacity(num_tokens, TOP_K, NUM_LOCAL_EXPERTS, config)
    route_plan = _stateful_route_plan(num_updates)
    route_metrics_by_scenario = {
        scenario: _route_metrics(_routes(num_tokens, scenario, TOP_K), capacity=capacity) for scenario in set(route_plan)
    }
    overflowing = {
        scenario.value: metrics["required_schedule_capacity"]
        for scenario, metrics in route_metrics_by_scenario.items()
        if metrics["required_schedule_capacity"] > capacity
    }
    if overflowing:
        raise click.BadParameter(f"stateful route plan exceeds schedule capacity {capacity}: {overflowing}")
    if not route_metrics_by_scenario[RouteScenario.ALL_TO_ONE]["at_capacity_boundary"]:
        required = route_metrics_by_scenario[RouteScenario.ALL_TO_ONE]["required_schedule_capacity"]
        raise click.BadParameter(
            f"all_to_one must exercise the exact schedule-capacity boundary; required={required}, capacity={capacity}"
        )

    parameter_arrays = _initial_parameters(
        seed=seed,
        replica_index=resolved_replica_index,
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
    )
    parameter_shardings = (
        expert_sharding,
        expert_sharding,
        expert_sharding,
        shared_sharding,
        shared_sharding,
        shared_sharding,
    )
    fused_parameters = tuple(
        jax.device_put(jnp.asarray(array, dtype=jnp.float32), sharding)
        for array, sharding in zip(parameter_arrays, parameter_shardings, strict=True)
    )
    reference_parameters = jax.tree.map(lambda value: jnp.array(value), fused_parameters)
    fused_momentum = jax.tree.map(lambda value: jnp.zeros_like(value, dtype=jnp.float32), fused_parameters)
    reference_momentum = jax.tree.map(lambda value: jnp.zeros_like(value, dtype=jnp.float32), reference_parameters)
    build_config = MokLikeBuildConfig(
        source_root=MOK_LIKE_SOURCE_ROOT,
        cache_root=MOK_LIKE_BUILD_ROOT,
        cuda_arch="sm_100a",
        clone_if_missing=True,
    )
    initial_x, initial_combine_weights, initial_target = _step_inputs(
        seed=seed,
        replica_index=resolved_replica_index,
        step=0,
        num_tokens=num_tokens,
        hidden_dim=hidden_dim,
    )
    initial_routes = _routes(num_tokens, route_plan[0], TOP_K).reshape(-1, TOP_K)
    example_arguments = (
        fused_parameters,
        fused_momentum,
        jax.device_put(jnp.asarray(initial_x, dtype=jnp.bfloat16), batch_sharding),
        jax.device_put(jnp.asarray(initial_routes, dtype=jnp.int32), batch_sharding),
        jax.device_put(jnp.asarray(initial_combine_weights, dtype=jnp.float32), batch_sharding),
        jax.device_put(jnp.asarray(initial_target, dtype=jnp.bfloat16), batch_sharding),
    )

    with initialize_mok_like_runtime(
        build_config=build_config,
        num_tokens=num_tokens,
        hidden_dim=hidden_dim,
        top_k=TOP_K,
        workspace_slots=workspace_slots,
        mesh=mesh,
    ) as runtime:

        def fused_step(parameters, momentum_state, x, selected_experts, combine_weights, target):
            def loss_fn(current_parameters, current_x, current_combine_weights):
                output, dropped_assignments = mok_like_mlp(
                    current_x,
                    selected_experts,
                    current_combine_weights,
                    *(parameter.astype(jnp.bfloat16) for parameter in current_parameters),
                    mesh=mesh,
                    runtime=runtime,
                    config=config,
                    collective_id=30,
                )
                loss = jnp.mean(jnp.square(output.astype(jnp.float32) - target.astype(jnp.float32)))
                return loss, (output, dropped_assignments)

            (loss, (output, dropped_assignments)), gradients = jax.value_and_grad(
                loss_fn,
                argnums=(0, 1, 2),
                has_aux=True,
            )(parameters, x, combine_weights)
            parameter_gradients, x_gradient, combine_weights_gradient = gradients
            next_parameters, next_momentum = _optimizer_update(
                parameters,
                momentum_state,
                parameter_gradients,
                learning_rate=learning_rate,
                momentum=momentum,
            )
            return FusedStepResult(
                parameters=next_parameters,
                optimizer_state=next_momentum,
                loss=loss,
                output=output,
                dropped_assignments=dropped_assignments,
                gradients=(x_gradient, combine_weights_gradient, *parameter_gradients),
            )

        def reference_step(parameters, momentum_state, x, selected_experts, combine_weights, target):
            def loss_fn(current_parameters, current_x, current_combine_weights):
                output = mok_like_reference(
                    current_x,
                    selected_experts,
                    current_combine_weights,
                    *(parameter.astype(jnp.bfloat16) for parameter in current_parameters),
                    mesh=mesh,
                    config=config,
                    fallback_implementation="ring",
                )
                loss = jnp.mean(jnp.square(output.astype(jnp.float32) - target.astype(jnp.float32)))
                return loss, output

            (loss, output), gradients = jax.value_and_grad(
                loss_fn,
                argnums=(0, 1, 2),
                has_aux=True,
            )(parameters, x, combine_weights)
            parameter_gradients, x_gradient, combine_weights_gradient = gradients
            next_parameters, next_momentum = _optimizer_update(
                parameters,
                momentum_state,
                parameter_gradients,
                learning_rate=learning_rate,
                momentum=momentum,
            )
            return ReferenceStepResult(
                parameters=next_parameters,
                optimizer_state=next_momentum,
                loss=loss,
                output=output,
                gradients=(x_gradient, combine_weights_gradient, *parameter_gradients),
            )

        compiled_fused_step = jax.jit(fused_step).lower(*example_arguments).compile()
        reference_example_arguments = (reference_parameters, reference_momentum, *example_arguments[2:])
        compiled_reference_step = jax.jit(reference_step).lower(*reference_example_arguments).compile()
        runtime.reset_call_counts()
        runtime.reset_debug_counters()
        worst_output_relative_l2 = 0.0
        worst_gradient_relative_l2 = 0.0
        worst_optimizer_relative_l2 = 0.0

        for step, scenario in enumerate(route_plan):
            x, combine_weights, target = _step_inputs(
                seed=seed,
                replica_index=resolved_replica_index,
                step=step,
                num_tokens=num_tokens,
                hidden_dim=hidden_dim,
            )
            routes = _routes(num_tokens, scenario, TOP_K)
            step_arguments = (
                jax.device_put(jnp.asarray(x, dtype=jnp.bfloat16), batch_sharding),
                jax.device_put(jnp.asarray(routes.reshape(-1, TOP_K), dtype=jnp.int32), batch_sharding),
                jax.device_put(jnp.asarray(combine_weights, dtype=jnp.float32), batch_sharding),
                jax.device_put(jnp.asarray(target, dtype=jnp.bfloat16), batch_sharding),
            )
            fused_result = compiled_fused_step(fused_parameters, fused_momentum, *step_arguments)
            reference_result = compiled_reference_step(reference_parameters, reference_momentum, *step_arguments)
            jax.block_until_ready((fused_result, reference_result))
            fused_parameters = fused_result.parameters
            fused_momentum = fused_result.optimizer_state
            fused_loss = fused_result.loss
            fused_output = fused_result.output
            dropped_assignments = fused_result.dropped_assignments
            fused_gradients = fused_result.gradients
            reference_parameters = reference_result.parameters
            reference_momentum = reference_result.optimizer_state
            reference_loss = reference_result.loss
            reference_output = reference_result.output
            reference_gradients = reference_result.gradients
            output_metrics = _reduced_error_metrics(fused_output, reference_output, absolute_tolerance=BF16_ATOL)
            loss_metrics = _reduced_error_metrics(fused_loss, reference_loss, absolute_tolerance=BF16_ATOL)
            gradient_metrics = _tree_error_metrics(GRADIENT_NAMES, fused_gradients, reference_gradients)
            parameter_metrics = _tree_error_metrics(PARAMETER_NAMES, fused_parameters, reference_parameters)
            optimizer_metrics = _tree_error_metrics(PARAMETER_NAMES, fused_momentum, reference_momentum)
            inactive_expert_gradient_maxima = _validated_inactive_expert_gradient_maxima(routes, fused_gradients[2:])
            call_counts = runtime.call_counts()
            debug_counters = runtime.debug_counters()
            step_metrics = {
                "replica_index": resolved_replica_index,
                "replica_count": resolved_replica_count,
                "step": step,
                "scenario": scenario.value,
                "routes": route_metrics_by_scenario[scenario],
                "fused_loss": float(fused_loss),
                "reference_loss": float(reference_loss),
                "loss": loss_metrics,
                "output": output_metrics,
                "gradients": gradient_metrics,
                "parameters": parameter_metrics,
                "optimizer_state": optimizer_metrics,
                "inactive_expert_gradient_maxima": inactive_expert_gradient_maxima,
                "dropped_assignments": int(dropped_assignments),
                "runtime": {
                    "ffi_call_counts": {"forward": call_counts[0], "backward": call_counts[1]},
                    "generation_mismatches": debug_counters.generation_mismatches,
                    "slot_reuse_failures": debug_counters.slot_reuse_failures,
                    "slot_acquisitions": debug_counters.slot_acquisitions,
                    "max_active_slots": debug_counters.max_active_slots,
                },
            }
            print(json.dumps({"stateful_parity_step": step_metrics}, sort_keys=True), flush=True)
            _assert_step_parity(step_metrics)
            expected_calls = (step + 1) * WORLD_SIZE
            if call_counts != (expected_calls, expected_calls):
                raise AssertionError(f"stateful parity replayed or eliminated native work at step {step}: {call_counts}")
            if any(debug_counters.generation_mismatches) or any(debug_counters.slot_reuse_failures):
                raise AssertionError(f"stateful workspace reuse failed at step {step}: {debug_counters}")
            worst_output_relative_l2 = max(worst_output_relative_l2, float(output_metrics["relative_l2_error"]))
            worst_gradient_relative_l2 = max(
                worst_gradient_relative_l2,
                *(float(metric["relative_l2_error"]) for metric in gradient_metrics.values()),
            )
            worst_optimizer_relative_l2 = max(
                worst_optimizer_relative_l2,
                *(float(metric["relative_l2_error"]) for metric in optimizer_metrics.values()),
            )

        call_counts = runtime.call_counts()
        debug_counters = runtime.debug_counters()

    expected_calls = num_updates * WORLD_SIZE
    if call_counts != (expected_calls, expected_calls):
        raise AssertionError(f"wrong terminal FFI call counts: expected {expected_calls} per phase, got {call_counts}")
    expected_slot_acquisitions = (2 * num_updates, 0)
    if any(acquisitions != expected_slot_acquisitions for acquisitions in debug_counters.slot_acquisitions):
        raise AssertionError(
            f"serialized stateful calls did not reuse only workspace slot zero: {debug_counters.slot_acquisitions}"
        )
    if any(maximum != 1 for maximum in debug_counters.max_active_slots):
        raise AssertionError(f"serialized stateful calls overlapped workspace slots: {debug_counters.max_active_slots}")
    activation_bytes = num_tokens * hidden_dim * jnp.dtype(jnp.bfloat16).itemsize
    router_bytes = num_tokens * TOP_K * jnp.dtype(jnp.float32).itemsize
    expected_forward_staging = (
        (num_updates, num_updates * activation_bytes)
        if forward_x_storage is MokLikeForwardXStorage.RUNTIME_STAGED
        else (0, 0)
    )
    if backward_peer_storage is MokLikeBackwardPeerStorage.RUNTIME_STAGED:
        expected_backward_staging = (4 * num_updates, num_updates * (2 * activation_bytes + 2 * router_bytes))
    elif backward_peer_storage is MokLikeBackwardPeerStorage.XLA_PEER_INPUTS_EXPERIMENTAL:
        expected_backward_staging = (num_updates, num_updates * router_bytes)
    else:
        expected_backward_staging = (0, 0)
    expected_staging_calls = (expected_forward_staging[0], expected_backward_staging[0])
    expected_staging_bytes = (expected_forward_staging[1], expected_backward_staging[1])
    if any(calls != expected_staging_calls for calls in debug_counters.staging_copy_calls):
        raise AssertionError(
            f"stateful parity observed unexpected staging calls: expected {expected_staging_calls}, "
            f"got {debug_counters.staging_copy_calls}"
        )
    if any(bytes_ != expected_staging_bytes for bytes_ in debug_counters.staging_copy_bytes):
        raise AssertionError(
            f"stateful parity observed unexpected staging bytes: expected {expected_staging_bytes}, "
            f"got {debug_counters.staging_copy_bytes}"
        )
    summary = {
        "replica_index": resolved_replica_index,
        "replica_count": resolved_replica_count,
        "seed": seed,
        "num_updates": num_updates,
        "scenarios": [scenario.value for scenario in route_plan],
        "schedule_capacity": capacity,
        "workspace_slots": workspace_slots,
        "forward_x_storage": forward_x_storage.value,
        "backward_peer_storage": backward_peer_storage.value,
        "optimizer": {"name": "sgd_momentum", "learning_rate": learning_rate, "momentum": momentum},
        "ffi_call_counts": {"forward": call_counts[0], "backward": call_counts[1]},
        "runtime_debug_counters": {
            "generation_mismatches": debug_counters.generation_mismatches,
            "slot_reuse_failures": debug_counters.slot_reuse_failures,
            "slot_acquisitions": debug_counters.slot_acquisitions,
            "max_active_slots": debug_counters.max_active_slots,
            "staging_copy_calls": debug_counters.staging_copy_calls,
            "staging_copy_bytes": debug_counters.staging_copy_bytes,
        },
        "worst_output_relative_l2_error": worst_output_relative_l2,
        "worst_gradient_relative_l2_error": worst_gradient_relative_l2,
        "worst_optimizer_state_relative_l2_error": worst_optimizer_relative_l2,
    }
    print(json.dumps({"stateful_parity_summary": summary}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
