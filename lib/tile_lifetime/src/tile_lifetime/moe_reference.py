# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Ordinary JAX reference/export for the first shared-plus-routed MoE region."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp

MOE_REGION_INPUT_NAMES = (
    "x",
    "router_weight",
    "shared_gate_weight",
    "shared_up_weight",
    "shared_down_weight",
    "routed_gate_weight",
    "routed_up_weight",
    "routed_down_weight",
)

# JAX emits top-k as a public StableHLO composite and expert indexing as gather.
# Its private top-k decomposition contains stablehlo.sort; the importer deliberately
# ignores that private implementation and validates the versioned composite boundary.
MOE_PUBLIC_STABLEHLO_EXTENSIONS = ("stablehlo.composite[chlo.top_k]", "stablehlo.gather")
MOE_UNIMPORTED_PRIVATE_OPERATIONS = ("stablehlo.sort",)


@dataclass(frozen=True)
class MoEDebugConfig:
    """Static dimensions for a small single-device semantic MoE fixture."""

    tokens: int = 8
    hidden: int = 16
    intermediate: int = 32
    experts: int = 4
    top_k: int = 2

    def __post_init__(self) -> None:
        if min(self.tokens, self.hidden, self.intermediate, self.experts) <= 0:
            raise ValueError("MoE dimensions must be positive")
        if not 0 < self.top_k <= self.experts:
            raise ValueError("top-k must be positive and no larger than the expert count")


def export_debug_moe_region(config: MoEDebugConfig = MoEDebugConfig()) -> bytes:
    """Export the ordinary shared-plus-routed JAX region as portable StableHLO."""
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
    )
    exported = jax.export.export(jax.jit(moe_region(config)))(*specifications)
    return exported.mlir_module_serialized


def moe_region(config: MoEDebugConfig):
    """Return the ordinary JAX function used to create the semantic fixture."""

    def region(
        x,
        router_weight,
        shared_gate_weight,
        shared_up_weight,
        shared_down_weight,
        routed_gate_weight,
        routed_up_weight,
        routed_down_weight,
    ):
        router_logits = _linear(x, router_weight, "th,he->te")
        top_values, expert_indices = jax.lax.top_k(router_logits, config.top_k)
        route_weights = _softmax(top_values.astype(jnp.float32))

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
        routed_sum = jnp.sum(weighted, axis=1).astype(jnp.bfloat16)
        return shared_output + routed_sum, expert_indices, route_weights

    return region


def _linear(left, right, specification: str):
    return jnp.einsum(specification, left, right, preferred_element_type=jnp.float32).astype(jnp.bfloat16)


def _softmax(value):
    shifted = value - jnp.max(value, axis=-1, keepdims=True)
    exponential = jnp.exp(shifted)
    return exponential / jnp.sum(exponential, axis=-1, keepdims=True)


def _silu(value):
    one = jnp.asarray(1, dtype=value.dtype)
    return value / (one + jnp.exp(-value))
