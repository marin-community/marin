# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import pytest

from tile_lifetime.jax_streaming_attention_training_frontend import (
    JaxAutomaticDifferentiationOwner,
    recover_jax_vjp_streaming_attention_training,
)
from tile_lifetime.streaming_attention import StreamingTileSchedule
from tile_lifetime.streaming_attention_backward import StreamingAttentionBackwardProvenance
from tile_lifetime.streaming_attention_backward_reference import (
    STREAMING_ATTENTION_BACKWARD_INPUT_NAMES,
    StreamingAttentionBackwardDebugConfig,
    causal_gqa_attention_training,
    causal_gqa_attention_with_log_sum_exp,
    streaming_attention_training_input_specifications,
)


def _config() -> StreamingAttentionBackwardDebugConfig:
    return StreamingAttentionBackwardDebugConfig(
        batch=1,
        query_length=4,
        key_length=4,
        query_heads=4,
        key_value_heads=2,
        head_dimension=4,
        scale=0.5,
    )


def test_natural_jax_vjp_frontend_records_recovered_algebra_provenance() -> None:
    config = _config()
    result = recover_jax_vjp_streaming_attention_training(
        causal_gqa_attention_training(config),
        streaming_attention_training_input_specifications(config),
        input_names=STREAMING_ATTENTION_BACKWARD_INPUT_NAMES,
        schedule=StreamingTileSchedule(query_tile_size=2, key_value_tile_size=2, pipeline_depth=2),
    )

    assert result.audit.source_kind == "ordinary_jax_tensor_program"
    assert result.audit.automatic_differentiation_owner is JaxAutomaticDifferentiationOwner.JAX_VJP
    assert result.audit.recovered_provenance is StreamingAttentionBackwardProvenance.JAX_VJP_GENERIC_ALGEBRA_IMPORT
    assert result.audit.workload_dispatch_key is None
    assert result.audit.opaque_frontend_primitives == ()
    assert result.audit.source_operation_ids
    assert result.audit.generic_algebra_operation_ids == result.audit.source_operation_ids
    assert result.audit.contract_operation_ids
    assert result.audit.fold_operation_ids
    assert result.audit.domain_restriction_operation_ids
    assert result.audit.cast_and_view_operation_ids
    assert len(result.audit.jaxpr_sha256) == 64
    assert len(result.audit.stablehlo_sha256) == 64
    assert result.recovered.forward_output is not None


def test_named_opaque_kernel_cannot_substitute_for_frontend_algebra() -> None:
    config = _config()
    specifications = streaming_attention_training_input_specifications(config)

    def opaque_training(query, key, value, output_cotangent):
        output_specifications = tuple(jax.ShapeDtypeStruct(array.shape, array.dtype) for array in (query, key, value))
        return jax.ffi.ffi_call(
            "flash_attention_4",
            output_specifications,
            vmap_method="broadcast_all",
        )(query, key, value, output_cotangent)

    with pytest.raises(ValueError, match=r"must expose tensor algebra.*opaque primitives.*ffi_call"):
        recover_jax_vjp_streaming_attention_training(
            opaque_training,
            specifications,
            input_names=STREAMING_ATTENTION_BACKWARD_INPUT_NAMES,
            schedule=StreamingTileSchedule(query_tile_size=2, key_value_tile_size=2, pipeline_depth=2),
        )


def test_workload_name_cannot_substitute_for_natural_jax_source() -> None:
    config = StreamingAttentionBackwardDebugConfig()

    with pytest.raises(TypeError, match="requires a callable tensor program"):
        recover_jax_vjp_streaming_attention_training(
            "gated_attention_kernel",  # type: ignore[arg-type]
            streaming_attention_training_input_specifications(config),
            input_names=STREAMING_ATTENTION_BACKWARD_INPUT_NAMES,
            schedule=StreamingTileSchedule(query_tile_size=32, key_value_tile_size=32, pipeline_depth=3),
        )


def test_forward_saved_state_is_natural_log_sum_exp_in_bhs_layout() -> None:
    config = _config()
    query = jnp.arange(1 * 4 * 4 * 4, dtype=jnp.float32).reshape(1, 4, 4, 4).astype(jnp.bfloat16) / 64
    key = jnp.flip(query[:, :, :2], axis=1)
    value = query[:, :, :2]

    _output, log_sum_exp = causal_gqa_attention_with_log_sum_exp(config)(query, key, value)
    grouped_query = query.reshape(1, 4, 2, 2, 4)
    scores = (
        jnp.einsum(
            "bqhgd,bkhd->bhgqk",
            grouped_query.astype(jnp.float32),
            key.astype(jnp.float32),
        )
        * config.scale
    )
    causal = jnp.arange(4)[None, :] <= jnp.arange(4)[:, None]
    expected = jax.nn.logsumexp(jnp.where(causal[None, None, None], scores, -jnp.inf), axis=-1).reshape(1, 4, 4)

    assert log_sum_exp.shape == (1, 4, 4)
    assert jnp.allclose(log_sum_exp, expected, atol=1e-6, rtol=1e-6)
