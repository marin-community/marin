# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import pytest

from levanter.kernels.ubx.transport_ffi import (
    UbxRuntimeConfig,
    pool_layout,
    validate_combine_inputs,
    validate_dispatch_inputs,
    validate_local_hardware_ordinals,
)


def _config() -> UbxRuntimeConfig:
    return UbxRuntimeConfig(
        num_ranks=8,
        max_tokens_per_rank=65_536,
        max_local_tokens=16_384,
        hidden_size=2560,
        top_k=4,
        experts_per_rank=8,
    )


def _spec(shape: tuple[int, ...], dtype: jnp.dtype) -> jax.ShapeDtypeStruct:
    return jax.ShapeDtypeStruct(shape, dtype)


def test_pool_layout_sizes_transport_by_compact_capacity() -> None:
    config = _config()
    layout = pool_layout(config)

    assert layout.dispatch_bytes == config.max_tokens_per_rank * config.hidden_size * 2
    assert layout.combine_bytes == config.max_local_tokens * config.top_k * config.hidden_size * 2
    assert layout.dispatch_offsets[0] >= layout.reg0_bytes
    assert layout.dispatch_offsets[1] >= layout.dispatch_offsets[0] + layout.dispatch_bytes
    assert layout.combine_offsets[0] >= layout.dispatch_offsets[1] + layout.dispatch_bytes
    assert layout.combine_offsets[1] >= layout.combine_offsets[0] + layout.combine_bytes
    assert layout.pool_bytes % (2 * 1024 * 1024) == 0


def test_dispatch_contract_accepts_compact_capacity_layout() -> None:
    config = _config()

    validate_dispatch_inputs(
        _spec((config.max_local_tokens, config.hidden_size), jnp.bfloat16),
        _spec((config.max_local_tokens, config.top_k), jnp.int32),
        _spec((config.max_local_tokens, config.top_k), jnp.int32),
        _spec((config.max_tokens_per_rank,), jnp.bool_),
        config,
    )


def test_combine_contract_requires_capacity_sized_inverse_and_expert_rows() -> None:
    config = _config()

    validate_combine_inputs(
        _spec((config.max_tokens_per_rank, config.hidden_size), jnp.bfloat16),
        _spec((config.max_tokens_per_rank, 4), jnp.int32),
        _spec((config.max_local_tokens, config.top_k), jnp.int32),
        _spec((config.max_local_tokens, config.total_experts), jnp.float32),
        config,
    )

    upstream_skew_capacity = config.max_tokens_per_rank * config.experts_per_rank
    with pytest.raises(ValueError, match="expert_outputs"):
        validate_combine_inputs(
            _spec((upstream_skew_capacity, config.hidden_size), jnp.bfloat16),
            _spec((config.max_tokens_per_rank, 4), jnp.int32),
            _spec((config.max_local_tokens, config.top_k), jnp.int32),
            _spec((config.max_local_tokens, config.total_experts), jnp.float32),
            config,
        )


def test_runtime_contract_rejects_non_eight_rank_group() -> None:
    config = UbxRuntimeConfig(
        num_ranks=4,
        max_tokens_per_rank=32,
        max_local_tokens=8,
        hidden_size=64,
        top_k=2,
        experts_per_rank=2,
    )

    with pytest.raises(ValueError, match="exactly 8 local ranks"):
        pool_layout(config)


@dataclass(frozen=True)
class _FakeDevice:
    id: int
    local_hardware_id: int


def test_local_device_contract_ignores_global_jax_ids() -> None:
    devices = [_FakeDevice(id=24 + local_ordinal, local_hardware_id=local_ordinal) for local_ordinal in range(8)]

    validate_local_hardware_ordinals(devices, num_ranks=8)


def test_local_device_contract_rejects_missing_cuda_ordinal() -> None:
    devices = [_FakeDevice(id=24 + local_ordinal, local_hardware_id=local_ordinal) for local_ordinal in range(7)]

    with pytest.raises(RuntimeError, match="local_hardware_id"):
        validate_local_hardware_ordinals(devices, num_ranks=8)
