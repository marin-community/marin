# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Ordinary JAX forward and reverse program for routed-MoE parity work."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from tile_lifetime.moe_reference import MoEDebugConfig, _linear, _silu, _softmax


@dataclass(frozen=True)
class MoETrainingBoundaryConfig:
    """Shapes shared by the natural program and the expert comparison."""

    tokens: int
    hidden: int
    intermediate: int
    experts: int
    top_k: int
    expert_parallel_size: int

    def __post_init__(self) -> None:
        if min(self.tokens, self.hidden, self.intermediate, self.experts, self.expert_parallel_size) <= 0:
            raise ValueError("MoE training dimensions and expert-parallel size must be positive")
        if not 0 < self.top_k <= self.experts:
            raise ValueError("top-k must be positive and no larger than the expert count")
        if self.experts % self.expert_parallel_size:
            raise ValueError("expert count must be divisible by expert-parallel size")

    @property
    def local_experts(self) -> int:
        """Return the contiguous expert count owned by one rank."""
        return self.experts // self.expert_parallel_size


PRIMARY_MOK_BF16_TRAINING_CONFIG = MoETrainingBoundaryConfig(
    tokens=2_048,
    hidden=7_168,
    intermediate=3_072,
    experts=384,
    top_k=6,
    expert_parallel_size=4,
)


def moe_training_boundary(config: MoETrainingBoundaryConfig):
    """Return a natural JAX program with JAX-owned reverse-mode AD.

    The function exposes the post-selection route-weight cotangent because that
    is the common semantic boundary with the pinned expert implementation. JAX
    then propagates that cotangent through normalized top-k weights and the
    router projection; Shuttle does not define a model-specific VJP.
    """

    def expert_body(
        x,
        expert_indices,
        route_weights,
        shared_gate_weight,
        shared_up_weight,
        shared_down_weight,
        routed_gate_weight,
        routed_up_weight,
        routed_down_weight,
    ):
        shared_gate = _linear(x, shared_gate_weight, "th,ih->ti")
        shared_up = _linear(x, shared_up_weight, "th,ih->ti")
        shared_hidden = _silu(shared_gate) * shared_up
        shared_output = _linear(shared_hidden, shared_down_weight, "ti,hi->th")

        selected_gate_weight = routed_gate_weight[expert_indices]
        selected_up_weight = routed_up_weight[expert_indices]
        selected_down_weight = routed_down_weight[expert_indices]
        routed_gate = _linear(x, selected_gate_weight, "th,tkih->tki")
        routed_up = _linear(x, selected_up_weight, "th,tkih->tki")
        routed_hidden = _silu(routed_gate) * routed_up
        routed_output = _linear(routed_hidden, selected_down_weight, "tki,tkhi->tkh")
        weighted = routed_output.astype(jnp.float32) * route_weights[..., None]
        return shared_output + jnp.sum(weighted, axis=1).astype(jnp.bfloat16)

    def training(
        x,
        router_weight,
        shared_gate_weight,
        shared_up_weight,
        shared_down_weight,
        routed_gate_weight,
        routed_up_weight,
        routed_down_weight,
        output_cotangent,
    ):
        def route(source, weight):
            logits = _linear(source, weight, "th,he->te")
            top_values, indices = jax.lax.top_k(logits, config.top_k)
            return _softmax(top_values.astype(jnp.float32)), indices

        route_weights, expert_indices = route(x, router_weight)
        differentiable_expert_inputs = (
            x,
            route_weights,
            shared_gate_weight,
            shared_up_weight,
            shared_down_weight,
            routed_gate_weight,
            routed_up_weight,
            routed_down_weight,
        )

        def selected_expert_body(*values):
            return expert_body(values[0], expert_indices, *values[1:])

        output, expert_pullback = jax.vjp(selected_expert_body, *differentiable_expert_inputs)
        expert_gradients = expert_pullback(output_cotangent)
        expert_input_gradient, route_weight_gradient, *weight_gradients = expert_gradients

        def route_weights_only(source, weight):
            return route(source, weight)[0]

        _, router_pullback = jax.vjp(route_weights_only, x, router_weight)
        router_input_gradient, router_weight_gradient = router_pullback(route_weight_gradient)
        input_gradient = expert_input_gradient + router_input_gradient
        return (
            output,
            expert_indices,
            route_weights,
            input_gradient,
            route_weight_gradient,
            router_weight_gradient,
            *weight_gradients,
        )

    return training


def _export_debug_moe_training_boundary(config: MoEDebugConfig):
    boundary_config = MoETrainingBoundaryConfig(
        tokens=config.tokens,
        hidden=config.hidden,
        intermediate=config.intermediate,
        experts=config.experts,
        top_k=config.top_k,
        expert_parallel_size=1,
    )
    bf16 = jnp.bfloat16
    specifications = (
        jax.ShapeDtypeStruct((config.tokens, config.hidden), bf16),
        jax.ShapeDtypeStruct((config.hidden, config.experts), bf16),
        jax.ShapeDtypeStruct((config.intermediate, config.hidden), bf16),
        jax.ShapeDtypeStruct((config.intermediate, config.hidden), bf16),
        jax.ShapeDtypeStruct((config.hidden, config.intermediate), bf16),
        jax.ShapeDtypeStruct((config.experts, config.intermediate, config.hidden), bf16),
        jax.ShapeDtypeStruct((config.experts, config.intermediate, config.hidden), bf16),
        jax.ShapeDtypeStruct((config.experts, config.hidden, config.intermediate), bf16),
        jax.ShapeDtypeStruct((config.tokens, config.hidden), bf16),
    )
    return jax.export.export(jax.jit(moe_training_boundary(boundary_config)))(*specifications)


def export_debug_moe_training_boundary(config: MoEDebugConfig = MoEDebugConfig()) -> bytes:
    """Export a small ordinary training boundary as portable StableHLO."""
    return _export_debug_moe_training_boundary(config).mlir_module_serialized


def export_debug_moe_training_boundary_text(config: MoEDebugConfig = MoEDebugConfig()) -> str:
    """Render the small ordinary training boundary for source-lineage audits."""
    return _export_debug_moe_training_boundary(config).mlir_module()
