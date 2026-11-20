# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import jmp
import numpy as np
import pytest
from haliax.nn.scan import ScanCheckpointPolicy
from jax.sharding import AxisType, Mesh

from levanter.grug.attention import AttentionMask as RawAttentionMask
from levanter.layers.attention import AttentionBackend, AttentionMask
from levanter.layers.scan_checkpoint import ScanCheckpointPolicy as RawScanCheckpointPolicy
from levanter.models.llama import LlamaConfig, LlamaLMHeadModel
from levanter.models.llama_native import NativeLlamaAttention, NativeLlamaConfig, NativeLlamaSharding
from levanter.models.lm_model import LmExample


pytestmark = pytest.mark.tpu


@pytest.mark.parametrize("kv_heads", [2, 4])
@pytest.mark.parametrize("compute_dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("upcast_attn", [False, True])
def test_native_model_matches_named_hf_weights_loss_and_all_gradients(kv_heads, compute_dtype, upcast_attn):
    _assert_named_native_loss_and_gradients(
        kv_heads, gradient_checkpointing=False, compute_dtype=compute_dtype, upcast_attn=upcast_attn
    )


def test_native_bfloat16_reference_attention_upcasts_before_query_scaling():
    # An eight-wide head exposes BF16 rounding in the inverse square root scale.
    config = NativeLlamaConfig(vocab_size=32, hidden_dim=8, num_heads=1, num_kv_heads=1, upcast_attn=True)
    layout = NativeLlamaSharding()
    mesh = Mesh(np.asarray(jax.devices()[:1]), ("data",), axis_types=(AxisType.Explicit,))
    with jax.set_mesh(mesh), jax.default_matmul_precision("highest"):
        attention = NativeLlamaAttention.init(config, layout, key=jax.random.PRNGKey(0))
        policy = jmp.Policy(param_dtype=jnp.float32, compute_dtype=jnp.bfloat16, output_dtype=jnp.float32)
        attention = policy.cast_to_compute(attention)
        x = jax.random.normal(jax.random.PRNGKey(1), (1, 8, config.hidden_dim), dtype=jnp.bfloat16)
        positions = jnp.arange(x.shape[1])[None, :]
        mask = positions[:, :, None] >= positions[:, None, :]

        q, k, v = attention.project_qkv(x, positions, sharding=layout)
        q, k = q[:, :, 0].astype(jnp.float32), k[:, :, 0].astype(jnp.float32)
        scores = (q @ k.swapaxes(-1, -2)) * (config.actual_head_dim**-0.5)
        weights = jax.nn.softmax(jnp.where(mask, scores, -1e9), axis=-1).astype(v.dtype)
        expected = attention.project_output((weights @ v[:, :, 0])[:, :, None], dtype=x.dtype, sharding=layout)
        actual = eqx.filter_jit(attention, compiler_options={"xla_allow_excess_precision": False})(x, mask, positions)
        np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_native_gqa_checkpointed_scan_matches_named_loss_and_all_gradients():
    _assert_named_native_loss_and_gradients(2, gradient_checkpointing=True)


def test_native_gqa_full_named_checkpoint_policy_matches_loss_and_all_gradients():
    policy = ScanCheckpointPolicy(save_carries=False, save_inputs=True, prevent_cse=True, nested=2)
    _assert_named_native_loss_and_gradients(2, gradient_checkpointing=policy)


def _assert_named_native_loss_and_gradients(
    kv_heads, *, gradient_checkpointing, compute_dtype=jnp.float32, upcast_attn=False
):
    config = LlamaConfig(
        hidden_dim=16,
        intermediate_dim=24,
        num_layers=2,
        num_heads=4,
        num_kv_heads=kv_heads,
        max_seq_len=8,
        attn_backend=AttentionBackend.VANILLA,
        gradient_checkpointing=gradient_checkpointing,
        upcast_attn=upcast_attn,
    )
    raw_checkpointing = gradient_checkpointing
    if isinstance(gradient_checkpointing, ScanCheckpointPolicy):
        raw_checkpointing = RawScanCheckpointPolicy(**dataclasses.asdict(gradient_checkpointing))
    native_config = NativeLlamaConfig(
        vocab_size=32,
        hidden_dim=config.hidden_dim,
        intermediate_dim=config.intermediate_dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        num_kv_heads=config.num_kv_heads,
        max_seq_len=config.max_seq_len,
        gradient_checkpointing=raw_checkpointing,
        upcast_attn=upcast_attn,
    )
    Vocab, Batch, Pos = hax.Axis("vocab", 32), hax.Axis("batch", 2), config.max_Pos
    tokens = hax.named((jnp.arange(16).reshape(2, 8) * 3) % 32, (Batch, Pos))
    segments = hax.named(jnp.array([[0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 1, 1, 1, 2, 2, 2]]), (Batch, Pos))
    weights = hax.named(jnp.ones((2, 8)).at[:, -1].set(0).at[0, 2].set(0.5), (Batch, Pos))
    example = LmExample(tokens, weights, AttentionMask.causal().with_segment_ids(segments))
    mesh = Mesh(np.asarray(jax.devices()[:1]), ("data",), axis_types=(AxisType.Explicit,))
    with jax.set_mesh(mesh), hax.axis_mapping({}):
        named = LlamaLMHeadModel.init(Vocab, config, key=jax.random.PRNGKey(0))
        source = named.to_state_dict()
        native = native_config.build(key=jax.random.PRNGKey(1)).from_state_dict(source)
        for name, value in native.to_state_dict().items():
            np.testing.assert_array_equal(value, source[name].reshape(value.shape))
        policy = jmp.Policy(param_dtype=jnp.float32, compute_dtype=compute_dtype, output_dtype=jnp.float32)
        raw_mask = RawAttentionMask.causal().with_segment_ids(segments.array)
        # Preserve the BF16 rounding stages used by the training runtime.
        compiler_options = {"xla_allow_excess_precision": False}

        @eqx.filter_jit(compiler_options=compiler_options)
        @eqx.filter_value_and_grad
        def named_loss(model):
            return policy.cast_to_compute(model).compute_next_token_loss(example).scalar()

        @eqx.filter_jit(compiler_options=compiler_options)
        @eqx.filter_value_and_grad
        def native_loss(model):
            return policy.cast_to_compute(model).next_token_loss(tokens.array, weights.array, mask=raw_mask)

        reference_loss, reference_grad = named_loss(named)
        actual_loss, actual_grad = native_loss(native)
        np.testing.assert_allclose(actual_loss, reference_loss, atol=1e-5, rtol=1e-5)
        gradients = reference_grad.to_state_dict(), actual_grad.to_state_dict()
        assert gradients[0].keys() == gradients[1].keys()
        for name, value in gradients[1].items():
            np.testing.assert_allclose(
                value, gradients[0][name].reshape(value.shape), atol=1e-5, rtol=1e-5, err_msg=name
            )
