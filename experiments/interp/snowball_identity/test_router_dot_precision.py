"""Router compute precision tests using the real positions-writer witness."""

from dataclasses import replace

import equinox as eqx
import jax
import jax.numpy as jnp
import jmp
import numpy as np
import pytest
from haliax import Axis
from levanter.grug.sharding import compact_grug_mesh
from levanter.models.snowball import SnowballConfig, SnowballLMHeadModel

from experiments.interp.snowball_identity import levanter_positions

CONFIG = SnowballConfig(
    vocab_size=64,
    hidden_dim=32,
    intermediate_dim=64,
    shared_expert_intermediate_dim=48,
    num_experts=8,
    num_experts_per_token=2,
    num_layers=5,
    num_heads=4,
    num_kv_heads=2,
    head_dim=12,
    max_seq_len=32,
    sliding_window=4,
    qk_mult=1.37,
    initializer_std=0.02,
    attention_implementation="reference",
)


@pytest.mark.parametrize("enabled", [False, True])
def test_router_dot_exception_uses_quantized_operands_and_preserves_observer(enabled):
    config = replace(CONFIG, num_layers=1, router_dot_fp32=enabled)
    mesh = compact_grug_mesh(expert_axis_size=1)
    with jax.set_mesh(mesh):
        model = SnowballLMHeadModel.init(Axis("vocab", config.vocab_size), config, key=jax.random.key(13))
        bias = jnp.linspace(-0.01234567, 0.01234567, config.num_experts, dtype=jnp.float32)
        model = eqx.tree_at(lambda value: value.transformer.blocks[0].mlp.router_bias, model, bias)
        view = levanter_positions.cast_model_for_compute(
            model, jmp.get_policy("p=f32,c=bfloat16"), {"router_bias": "float32"}
        )
        mlp = view.transformer.blocks[0].mlp
        assert mlp.router.dtype == jnp.bfloat16
        np.testing.assert_array_equal(mlp.router_bias, bias)
        assert np.any(np.asarray(bias) != np.asarray(bias.astype(jnp.bfloat16).astype(jnp.float32)))
        witness = levanter_positions.router_dot_dtypes(view, jnp.bfloat16)
        expected_dtype = "float32" if enabled else "bfloat16"
        assert witness == [{"lhs": expected_dtype, "rhs": expected_dtype, "out": expected_dtype}]
        assert levanter_positions.router_dot_is_fp32(witness) == enabled

        inputs = jax.random.normal(jax.random.key(17), (1, 8, config.hidden_dim)).astype(jnp.bfloat16)
        flat = inputs.reshape(-1, config.hidden_dim)
        router_input = flat.astype(jnp.float32) if enabled else flat
        weight = mlp.router.astype(jnp.float32) if enabled else mlp.router
        logits = jnp.einsum("td,de->te", router_input, weight)
        expected_routes = jax.lax.top_k(logits.astype(jnp.float32) + bias, config.num_experts_per_token + 1)[1][:, :-1]
        output, routes = eqx.filter_jit(lambda layer, values: layer.with_routes(values))(mlp, inputs)
        np.testing.assert_array_equal(routes.reshape(expected_routes.shape), expected_routes)
        np.testing.assert_array_equal(output, eqx.filter_jit(lambda layer, values: layer(values))(mlp, inputs))
        assert output.dtype == jnp.bfloat16
        assert "router_dot_fp32" not in config.to_hf_config(config.vocab_size).to_dict()
