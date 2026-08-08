# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Ordinary JAX affine-state scan used by StableHLO recovery tests."""

from dataclasses import dataclass
from enum import StrEnum

import jax
import jax.numpy as jnp


class ScanDecayAxes(StrEnum):
    """Logical domain of the multiplicative state decay."""

    SCALAR = "scalar"
    KEY = "key"


@dataclass(frozen=True)
class StatefulScanDebugConfig:
    """Small static shape for an ordinary paper-style affine recurrence."""

    batch: int = 1
    sequence: int = 4
    heads: int = 2
    key_dimension: int = 8
    value_dimension: int = 12
    update_rank: int = 1
    decay_axes: ScanDecayAxes = ScanDecayAxes.SCALAR

    def __post_init__(self) -> None:
        dimensions = (
            self.batch,
            self.sequence,
            self.heads,
            self.key_dimension,
            self.value_dimension,
            self.update_rank,
        )
        if any(dimension <= 0 for dimension in dimensions):
            raise ValueError("stateful-scan dimensions must be positive")


STATEFUL_SCAN_INPUT_NAMES = ("query", "key", "value", "log_decay", "beta", "initial_state")


def stateful_scan_region(config: StatefulScanDebugConfig):
    """Return natural JAX tensor/state math for a bounded-rank affine scan."""

    def region(query, key, value, log_decay, beta, initial_state):
        def step(state, inputs):
            query_token, key_token, value_token, log_decay_token, beta_token = inputs
            decay = jnp.exp(log_decay_token)
            if config.decay_axes is ScanDecayAxes.SCALAR:
                decay = decay[..., None, None]
            else:
                decay = decay[..., :, None]
            decayed_state = state * decay
            prediction = jnp.einsum(
                "bhkv,bhrk->bhrv",
                decayed_state,
                key_token,
                preferred_element_type=jnp.float32,
            )
            delta = beta_token[..., None] * (value_token.astype(jnp.float32) - prediction)
            state_next = decayed_state + jnp.einsum(
                "bhrk,bhrv->bhkv",
                key_token,
                delta,
                preferred_element_type=jnp.float32,
            )
            output = jnp.einsum(
                "bhkv,bhk->bhv",
                state_next,
                query_token,
                preferred_element_type=jnp.float32,
            )
            return state_next, output

        scan_inputs = tuple(jnp.swapaxes(value, 0, 1) for value in (query, key, value, log_decay, beta))
        final_state, output = jax.lax.scan(step, initial_state, scan_inputs)
        return jnp.swapaxes(output, 0, 1), final_state

    return region


def export_debug_stateful_scan(config: StatefulScanDebugConfig = StatefulScanDebugConfig()) -> bytes:
    """Export the natural recurrence as portable StableHLO containing ``while``."""
    bf16 = jnp.bfloat16
    decay_shape = (
        (config.batch, config.sequence, config.heads)
        if config.decay_axes is ScanDecayAxes.SCALAR
        else (config.batch, config.sequence, config.heads, config.key_dimension)
    )
    specifications = (
        jax.ShapeDtypeStruct((config.batch, config.sequence, config.heads, config.key_dimension), bf16),
        jax.ShapeDtypeStruct(
            (config.batch, config.sequence, config.heads, config.update_rank, config.key_dimension),
            bf16,
        ),
        jax.ShapeDtypeStruct(
            (config.batch, config.sequence, config.heads, config.update_rank, config.value_dimension),
            bf16,
        ),
        jax.ShapeDtypeStruct(decay_shape, jnp.float32),
        jax.ShapeDtypeStruct(
            (config.batch, config.sequence, config.heads, config.update_rank),
            jnp.float32,
        ),
        jax.ShapeDtypeStruct(
            (config.batch, config.heads, config.key_dimension, config.value_dimension),
            jnp.float32,
        ),
    )
    exported = jax.export.export(jax.jit(stateful_scan_region(config)))(*specifications)
    return exported.mlir_module_serialized
