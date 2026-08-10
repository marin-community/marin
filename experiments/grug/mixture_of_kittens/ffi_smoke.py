# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run a four-GPU correctness check for the Mixture-of-Kittens JAX FFI."""

import json

import click
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from levanter.kernels.mixture_of_kittens.forward_ffi import (
    MoKForwardConfig,
    ensure_runtime,
    forward_bf16_local,
    schedule_capacity,
)

from experiments.grug.mixture_of_kittens.fused_moe import (
    mixture_of_kittens_mlp,
    mixture_of_kittens_reference,
)
from experiments.grug.mixture_of_kittens.schedule import build_schedule

WORLD_SIZE = 4
DEFAULT_NUM_TOKENS = 512
DEFAULT_HIDDEN_DIM = 512
DEFAULT_INTERMEDIATE_DIM = 512
TOP_K = 4
NUM_LOCAL_EXPERTS = 2
BF16_ATOL = 0.5
BF16_RTOL = 0.01
SHARED_GRADIENT_BF16_ATOL = 1.0


def _random_inputs(*, num_tokens: int, hidden_dim: int, intermediate_dim: int) -> tuple[np.ndarray, ...]:
    random = np.random.default_rng(1234)
    x = random.normal(size=(WORLD_SIZE, num_tokens, hidden_dim)).astype(np.float32)
    router_weights = np.ones((WORLD_SIZE, num_tokens, TOP_K), dtype=np.float32)
    shared_gate = (random.normal(size=(intermediate_dim, hidden_dim)) / hidden_dim**0.5).astype(np.float32)
    shared_up = (random.normal(size=(intermediate_dim, hidden_dim)) / hidden_dim**0.5).astype(np.float32)
    shared_down = (random.normal(size=(hidden_dim, intermediate_dim)) / intermediate_dim**0.5).astype(np.float32)
    routed_gate = (
        random.normal(size=(WORLD_SIZE, NUM_LOCAL_EXPERTS, intermediate_dim, hidden_dim)) / hidden_dim**0.5
    ).astype(np.float32)
    routed_up = (
        random.normal(size=(WORLD_SIZE, NUM_LOCAL_EXPERTS, intermediate_dim, hidden_dim)) / hidden_dim**0.5
    ).astype(np.float32)
    routed_down = (
        random.normal(size=(WORLD_SIZE, NUM_LOCAL_EXPERTS, hidden_dim, intermediate_dim)) / intermediate_dim**0.5
    ).astype(np.float32)
    return x, router_weights, shared_gate, shared_up, shared_down, routed_gate, routed_up, routed_down


def _routes(num_tokens: int) -> np.ndarray:
    source_ranks = np.arange(WORLD_SIZE, dtype=np.int32)[:, None, None]
    token_indices = np.arange(num_tokens, dtype=np.int32)[None, :, None]
    route_indices = np.arange(TOP_K, dtype=np.int32)[None, None, :]
    destination_ranks = (source_ranks + route_indices) % WORLD_SIZE
    local_experts = ((source_ranks + token_indices) % 4 == 0).astype(np.int32)
    return destination_ranks * NUM_LOCAL_EXPERTS + local_experts


def _schedules(
    top_experts: np.ndarray,
    config: MoKForwardConfig,
    *,
    num_tokens: int,
) -> tuple[np.ndarray, ...]:
    capacity = schedule_capacity(num_tokens, TOP_K, NUM_LOCAL_EXPERTS, config)
    schedules = [
        build_schedule(
            jnp.asarray(top_experts),
            num_local_experts=NUM_LOCAL_EXPERTS,
            schedule_capacity=capacity,
            rank=jnp.asarray(rank, dtype=jnp.int32),
        )
        for rank in range(WORLD_SIZE)
    ]
    if any(bool(jax.device_get(schedule.overflow)) for schedule in schedules):
        raise RuntimeError("The smoke-test schedule capacity is too small")
    return tuple(
        np.stack([np.asarray(jax.device_get(getattr(schedule, field))) for schedule in schedules])
        for field in ("peer_rank", "peer_token_idx", "num_tokens", "tokens_per_expert")
    )


def _reference(
    x: jax.Array,
    router_weights: jax.Array,
    shared_gate: jax.Array,
    shared_up: jax.Array,
    shared_down: jax.Array,
    routed_gate: jax.Array,
    routed_up: jax.Array,
    routed_down: jax.Array,
    top_experts: jax.Array,
) -> jax.Array:
    shared_gate_values = jnp.einsum("wth,ih->wti", x, shared_gate)
    shared_up_values = jnp.einsum("wth,ih->wti", x, shared_up)
    shared_hidden = jax.nn.silu(shared_gate_values) * shared_up_values
    shared_output = jnp.einsum("wti,hi->wth", shared_hidden, shared_down)

    routed_output = jnp.zeros_like(shared_output, dtype=jnp.float32)
    intermediate_dim, hidden_dim = shared_gate.shape
    flattened_gate = routed_gate.reshape(-1, intermediate_dim, hidden_dim)
    flattened_up = routed_up.reshape(-1, intermediate_dim, hidden_dim)
    flattened_down = routed_down.reshape(-1, hidden_dim, intermediate_dim)
    for expert_index in range(WORLD_SIZE * NUM_LOCAL_EXPERTS):
        routed_gate_values = jnp.einsum("wth,ih->wti", x, flattened_gate[expert_index])
        routed_up_values = jnp.einsum("wth,ih->wti", x, flattened_up[expert_index])
        routed_hidden = jax.nn.silu(routed_gate_values) * routed_up_values
        expert_output = jnp.einsum("wti,hi->wth", routed_hidden, flattened_down[expert_index])
        expert_weights = jnp.sum(
            jnp.where(top_experts == expert_index, router_weights, 0.0),
            axis=2,
        )
        routed_output += expert_output.astype(jnp.float32) * expert_weights[..., None]
    return (shared_output.astype(jnp.float32) + routed_output).astype(jnp.bfloat16)


def _error_metrics(
    actual: jax.Array,
    expected: jax.Array,
    *,
    absolute_tolerance: float = BF16_ATOL,
) -> dict[str, float | bool]:
    actual_float = np.asarray(jax.device_get(actual), dtype=np.float32)
    expected_float = np.asarray(jax.device_get(expected), dtype=np.float32)
    absolute_error = np.abs(actual_float - expected_float)
    close = np.isclose(actual_float, expected_float, atol=absolute_tolerance, rtol=BF16_RTOL)
    return {
        "allclose": bool(np.all(close)),
        "absolute_tolerance": absolute_tolerance,
        "max_absolute_error": float(np.max(absolute_error)),
        "mean_absolute_error": float(np.mean(absolute_error)),
        "mismatch_fraction": float(np.mean(~close)),
    }


@click.command()
@click.option("--num-tokens", type=click.IntRange(min=256), default=DEFAULT_NUM_TOKENS, show_default=True)
@click.option("--hidden-dim", type=click.IntRange(min=256), default=DEFAULT_HIDDEN_DIM, show_default=True)
@click.option(
    "--intermediate-dim",
    type=click.IntRange(min=256),
    default=DEFAULT_INTERMEDIATE_DIM,
    show_default=True,
)
def main(num_tokens: int, hidden_dim: int, intermediate_dim: int) -> None:
    """Run the fused correctness gate at the selected model dimensions."""
    for name, value in (
        ("num_tokens", num_tokens),
        ("hidden_dim", hidden_dim),
        ("intermediate_dim", intermediate_dim),
    ):
        if value % 256 != 0:
            raise click.BadParameter(f"{name} must be divisible by 256, got {value}")
    devices = jax.devices()
    if len(devices) != WORLD_SIZE:
        raise RuntimeError(f"The smoke test requires {WORLD_SIZE} visible GPUs, got {len(devices)}")

    config = MoKForwardConfig(minibatch_size=256, macrobatch_size=256)
    inputs = _random_inputs(num_tokens=num_tokens, hidden_dim=hidden_dim, intermediate_dim=intermediate_dim)
    top_experts = _routes(num_tokens)
    peer_rank, peer_token_idx, num_scheduled_tokens, tokens_per_expert = _schedules(
        top_experts,
        config,
        num_tokens=num_tokens,
    )
    ensure_runtime(num_tokens=num_tokens, hidden_dim=hidden_dim, top_k=TOP_K)

    mesh = Mesh(
        np.asarray(devices).reshape(1, 1, WORLD_SIZE, 1),
        ("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit,) * 4,
    )
    expert_sharded = NamedSharding(mesh, P("expert"))
    batch_sharded = NamedSharding(mesh, P(("replica_dcn", "data", "expert")))
    shared_model_sharded = NamedSharding(mesh, P(("data", "expert"), "model"))
    replicated = NamedSharding(mesh, P())
    x, router_weights, shared_gate, shared_up, shared_down, routed_gate, routed_up, routed_down = inputs
    arguments = (
        jax.device_put(jnp.asarray(x, dtype=jnp.bfloat16), expert_sharded),
        jax.device_put(jnp.asarray(router_weights), expert_sharded),
        jax.device_put(jnp.asarray(shared_gate, dtype=jnp.bfloat16), replicated),
        jax.device_put(jnp.asarray(routed_gate, dtype=jnp.bfloat16), expert_sharded),
        jax.device_put(jnp.asarray(shared_up, dtype=jnp.bfloat16), replicated),
        jax.device_put(jnp.asarray(routed_up, dtype=jnp.bfloat16), expert_sharded),
        jax.device_put(jnp.asarray(shared_down, dtype=jnp.bfloat16), replicated),
        jax.device_put(jnp.asarray(routed_down, dtype=jnp.bfloat16), expert_sharded),
        jax.device_put(jnp.asarray(peer_rank), expert_sharded),
        jax.device_put(jnp.asarray(peer_token_idx), expert_sharded),
        jax.device_put(jnp.asarray(num_scheduled_tokens), expert_sharded),
        jax.device_put(jnp.asarray(tokens_per_expert), expert_sharded),
    )

    def local_forward(
        local_x: jax.Array,
        local_router_weights: jax.Array,
        local_shared_gate: jax.Array,
        local_routed_gate: jax.Array,
        local_shared_up: jax.Array,
        local_routed_up: jax.Array,
        local_shared_down: jax.Array,
        local_routed_down: jax.Array,
        local_peer_rank: jax.Array,
        local_peer_token_idx: jax.Array,
        local_num_scheduled_tokens: jax.Array,
        local_tokens_per_expert: jax.Array,
    ) -> jax.Array:
        output, _ = forward_bf16_local(
            local_x[0],
            local_router_weights[0],
            local_shared_gate,
            local_routed_gate[0],
            local_shared_up,
            local_routed_up[0],
            local_shared_down,
            local_routed_down[0],
            local_peer_rank[0],
            local_peer_token_idx[0],
            local_num_scheduled_tokens[0],
            local_tokens_per_expert[0],
            config=config,
        )
        return output[None]

    fused_forward = jax.jit(
        jax.shard_map(
            local_forward,
            mesh=mesh,
            in_specs=(
                P("expert", None, None),
                P("expert", None, None),
                P(None, None),
                P("expert", None, None, None),
                P(None, None),
                P("expert", None, None, None),
                P(None, None),
                P("expert", None, None, None),
                P("expert", None),
                P("expert", None),
                P("expert"),
                P("expert", None),
            ),
            out_specs=P("expert", None, None),
            check_vma=False,
        )
    )
    actual = fused_forward(*arguments)
    actual.block_until_ready()
    print("Fused forward complete", flush=True)

    with jax.default_device(devices[0]):
        reference = jax.jit(_reference)(
            jnp.asarray(x, dtype=jnp.bfloat16),
            jnp.asarray(router_weights),
            jnp.asarray(shared_gate, dtype=jnp.bfloat16),
            jnp.asarray(shared_up, dtype=jnp.bfloat16),
            jnp.asarray(shared_down, dtype=jnp.bfloat16),
            jnp.asarray(routed_gate, dtype=jnp.bfloat16),
            jnp.asarray(routed_up, dtype=jnp.bfloat16),
            jnp.asarray(routed_down, dtype=jnp.bfloat16),
            jnp.asarray(top_experts),
        )
        reference.block_until_ready()
    print("Independent forward complete", flush=True)

    forward_metrics = _error_metrics(actual, reference)

    output_gradient = np.random.default_rng(4321).normal(size=x.shape).astype(np.float32)
    current_layout_arguments = (
        jax.device_put(jnp.asarray(x.reshape(-1, hidden_dim), dtype=jnp.bfloat16), batch_sharded),
        jax.device_put(jnp.asarray(router_weights.reshape(-1, TOP_K)), batch_sharded),
        jax.device_put(
            jnp.asarray(
                np.transpose(routed_gate, (0, 1, 3, 2)).reshape(-1, hidden_dim, intermediate_dim), dtype=jnp.bfloat16
            ),
            expert_sharded,
        ),
        jax.device_put(
            jnp.asarray(
                np.transpose(routed_up, (0, 1, 3, 2)).reshape(-1, hidden_dim, intermediate_dim), dtype=jnp.bfloat16
            ),
            expert_sharded,
        ),
        jax.device_put(
            jnp.asarray(
                np.transpose(routed_down, (0, 1, 3, 2)).reshape(-1, intermediate_dim, hidden_dim), dtype=jnp.bfloat16
            ),
            expert_sharded,
        ),
        jax.device_put(jnp.asarray(shared_gate.T, dtype=jnp.bfloat16), shared_model_sharded),
        jax.device_put(jnp.asarray(shared_up.T, dtype=jnp.bfloat16), shared_model_sharded),
        jax.device_put(jnp.asarray(shared_down.T, dtype=jnp.bfloat16), shared_model_sharded),
    )
    selected_experts = jax.device_put(jnp.asarray(top_experts.reshape(-1, TOP_K)), batch_sharded)
    sharded_output_gradient = jax.device_put(
        jnp.asarray(output_gradient.reshape(-1, hidden_dim), dtype=jnp.bfloat16), batch_sharded
    )

    def integrated_loss(*differentiable: jax.Array) -> jax.Array:
        (
            integrated_x,
            integrated_combine_weights,
            integrated_w_gate,
            integrated_w_up,
            integrated_w_down,
            integrated_shared_gate,
            integrated_shared_up,
            integrated_shared_down,
        ) = differentiable
        output, _ = mixture_of_kittens_mlp(
            integrated_x,
            selected_experts,
            integrated_combine_weights,
            integrated_w_gate,
            integrated_w_up,
            integrated_w_down,
            integrated_shared_gate,
            integrated_shared_up,
            integrated_shared_down,
            mesh=mesh,
            config=config,
        )
        return jnp.sum(output.astype(jnp.float32) * sharded_output_gradient.astype(jnp.float32))

    integrated_value, integrated_gradients = jax.jit(jax.value_and_grad(integrated_loss, argnums=tuple(range(8))))(
        *current_layout_arguments
    )
    integrated_value.block_until_ready()
    gradient_names = (
        "x",
        "router_weights",
        "routed_gate",
        "routed_up",
        "routed_down",
        "shared_gate",
        "shared_up",
        "shared_down",
    )
    for name, gradient in zip(gradient_names, integrated_gradients, strict=True):
        gradient.block_until_ready()
        print(f"Fused gradient complete: {name}", flush=True)

    def fallback_loss(*differentiable: jax.Array) -> jax.Array:
        (
            fallback_x,
            fallback_combine_weights,
            fallback_w_gate,
            fallback_w_up,
            fallback_w_down,
            fallback_shared_gate,
            fallback_shared_up,
            fallback_shared_down,
        ) = differentiable
        output = mixture_of_kittens_reference(
            fallback_x,
            selected_experts,
            fallback_combine_weights,
            fallback_w_gate,
            fallback_w_up,
            fallback_w_down,
            fallback_shared_gate,
            fallback_shared_up,
            fallback_shared_down,
            mesh=mesh,
            config=config,
            fallback_implementation="ring",
            ragged_all_to_all_splits_per_peer=1,
        )
        return jnp.sum(output.astype(jnp.float32) * sharded_output_gradient.astype(jnp.float32))

    reference_value, reference_gradients = jax.jit(jax.value_and_grad(fallback_loss, argnums=tuple(range(8))))(
        *current_layout_arguments
    )
    reference_value.block_until_ready()
    for name, gradient in zip(gradient_names, reference_gradients, strict=True):
        gradient.block_until_ready()
        print(f"JAX gradient complete: {name}", flush=True)
    loss_absolute_error = float(
        np.abs(np.asarray(jax.device_get(integrated_value)) - np.asarray(jax.device_get(reference_value)))
    )
    gradient_metrics = {
        name: _error_metrics(
            integrated_gradient,
            reference_gradient,
            absolute_tolerance=SHARED_GRADIENT_BF16_ATOL if name.startswith("shared_") else BF16_ATOL,
        )
        for name, integrated_gradient, reference_gradient in zip(
            gradient_names,
            integrated_gradients,
            reference_gradients,
            strict=True,
        )
    }
    metrics = {
        "forward": forward_metrics,
        "loss_absolute_error": loss_absolute_error,
        "gradients": gradient_metrics,
    }
    print(json.dumps(metrics, sort_keys=True))
    checks = [forward_metrics["allclose"]]
    checks.extend(result["allclose"] for result in gradient_metrics.values())
    if not all(checks):
        raise AssertionError("The fused output or fallback gradient does not match the JAX BF16 reference")


if __name__ == "__main__":
    main()
