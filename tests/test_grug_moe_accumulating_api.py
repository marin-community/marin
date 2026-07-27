# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import importlib
import os
import subprocess
import sys
import textwrap

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from experiments.grug.moe.model import GrugModelConfig, Transformer
from experiments.grug.moe.train import (
    _stage_expert_gradient_accumulators,
    _stage_with_expert_gradients,
    _stage_without_expert_gradients,
)


def _fake_triton_ragged_dot(monkeypatch) -> None:
    ragged_dot_module = importlib.import_module("haliax.nn.ragged_dot")

    def fake_triton_pallas_call(
        lhs,
        rhs,
        group_sizes,
        ragged_dot_dimension_numbers=ragged_dot_module._DEFAULT_DIM_NUMS,
        *,
        output_dtype=None,
    ):
        output = jax.lax.ragged_dot_general(
            lhs=lhs,
            rhs=rhs,
            group_sizes=group_sizes,
            ragged_dot_dimension_numbers=ragged_dot_dimension_numbers,
        )
        return output if output_dtype is None else output.astype(output_dtype)

    def fake_accumulating_pallas_call(lhs, rhs, group_sizes, accumulator, accumulation_scale):
        fresh_gradient = fake_triton_pallas_call(
            lhs,
            rhs,
            group_sizes,
            ragged_dot_module._DRHS_DIM_NUMS,
            output_dtype=jnp.float32,
        )
        return fresh_gradient + accumulation_scale * accumulator

    monkeypatch.setattr(ragged_dot_module, "_has_pallas_triton", True)
    monkeypatch.setattr(ragged_dot_module, "_triton_pallas_call", fake_triton_pallas_call)
    monkeypatch.setattr(
        ragged_dot_module,
        "_triton_ragged_contracting_dim_accumulating_pallas_call",
        fake_accumulating_pallas_call,
    )


def _bfloat16_arrays(tree):
    return jax.tree.map(
        lambda value: value.astype(jnp.bfloat16) if eqx.is_inexact_array(value) else value,
        tree,
    )


def test_pipeline_stage_accumulating_weight_gradient_threads_every_block_token(monkeypatch):
    _fake_triton_ragged_dot(monkeypatch)
    mesh = Mesh(
        np.asarray(jax.devices()[:1], dtype=object).reshape((1, 1, 1, 1)),
        ("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit,) * 4,
    )
    config = GrugModelConfig(
        vocab_size=16,
        hidden_dim=8,
        intermediate_dim=4,
        shared_expert_intermediate_dim=0,
        num_experts=2,
        num_experts_per_token=1,
        num_layers=2,
        num_heads=2,
        num_kv_heads=2,
        max_seq_len=4,
        sliding_window=4,
        attention_implementation="reference",
        moe_implementation="ring",
        loss_implementation="reference",
        remat_mode="save_moe",
    )
    with jax.set_mesh(mesh):
        stage = Transformer.init(config, key=jax.random.key(0)).split_for_pipeline(1)[0]
        stage = _bfloat16_arrays(stage)

    hidden = jax.random.normal(jax.random.key(1), (1, 4, config.hidden_dim), dtype=jnp.bfloat16)
    output_cotangent = jax.random.normal(jax.random.key(2), hidden.shape, dtype=jnp.float32)
    zero_w13 = tuple(
        jnp.zeros(
            (
                config.num_experts,
                config.hidden_dim,
                2 * config.intermediate_dim,
            ),
            dtype=jnp.float32,
        )
        for _ in stage.blocks
    )
    zero_w2 = tuple(
        jnp.zeros(
            (
                config.num_experts,
                config.intermediate_dim,
                config.hidden_dim,
            ),
            dtype=jnp.float32,
        )
        for _ in stage.blocks
    )
    keys = jax.random.split(jax.random.key(3), 2 * len(stage.blocks))
    prior_w13 = tuple(jax.random.normal(keys[i], value.shape, dtype=jnp.float32) for i, value in enumerate(zero_w13))
    prior_w2 = tuple(
        jax.random.normal(keys[len(stage.blocks) + i], value.shape, dtype=jnp.float32) for i, value in enumerate(zero_w2)
    )
    expert_sharding = NamedSharding(mesh, P("expert", None, None))
    zero_w13 = tuple(jax.device_put(value, expert_sharding) for value in zero_w13)
    zero_w2 = tuple(jax.device_put(value, expert_sharding) for value in zero_w2)
    prior_w13 = tuple(jax.device_put(value, expert_sharding) for value in prior_w13)
    prior_w2 = tuple(jax.device_put(value, expert_sharding) for value in prior_w2)

    def stage_loss(stage, w13_accumulators, w2_accumulators):
        output, router_metrics, token = stage.block_range_accumulating_weight_gradient(
            hidden,
            w13_accumulators,
            w2_accumulators,
        )
        normalized_output_loss = jnp.mean(output.astype(jnp.float32) * output_cotangent)
        return normalized_output_loss + token, (output, router_metrics, token)

    value_and_grad = eqx.filter_value_and_grad(stage_loss, has_aux=True)
    with jax.set_mesh(mesh):
        (zero_value, zero_aux), zero_gradient = value_and_grad(stage, zero_w13, zero_w2)
        (prior_value, prior_aux), prior_gradient = value_and_grad(stage, prior_w13, prior_w2)

    np.testing.assert_array_equal(np.asarray(prior_aux[0]), np.asarray(zero_aux[0]))
    np.testing.assert_array_equal(
        np.asarray(prior_aux[1]["capacity_overflow_per_layer"]),
        np.asarray(zero_aux[1]["capacity_overflow_per_layer"]),
    )
    assert float(prior_aux[2]) == 0.0
    assert float(prior_value) == float(zero_value)

    for block_index, (w13_prior, w2_prior) in enumerate(zip(prior_w13, prior_w2, strict=True)):
        zero_expert_gradient = zero_gradient.blocks[block_index].mlp.expert_mlp
        prior_expert_gradient = prior_gradient.blocks[block_index].mlp.expert_mlp
        zero_w13_gradient = jnp.concatenate(
            (zero_expert_gradient.w_gate, zero_expert_gradient.w_up),
            axis=-1,
        )
        prior_w13_gradient = jnp.concatenate(
            (prior_expert_gradient.w_gate, prior_expert_gradient.w_up),
            axis=-1,
        )
        np.testing.assert_allclose(
            np.asarray(prior_w13_gradient),
            np.asarray(zero_w13_gradient + w13_prior),
            rtol=1e-5,
            atol=1e-5,
        )
        np.testing.assert_allclose(
            np.asarray(prior_expert_gradient.w_down),
            np.asarray(zero_expert_gradient.w_down + w2_prior),
            rtol=1e-5,
            atol=1e-5,
        )

    ordinary_gradient = _stage_without_expert_gradients(prior_gradient)
    expert_gradient = _stage_expert_gradient_accumulators(prior_gradient)
    restored_gradient = _stage_with_expert_gradients(ordinary_gradient, expert_gradient)
    for expected_block, restored_block in zip(prior_gradient.blocks, restored_gradient.blocks, strict=True):
        expected_expert = expected_block.mlp.expert_mlp
        restored_expert = restored_block.mlp.expert_mlp
        np.testing.assert_array_equal(np.asarray(restored_expert.w_gate), np.asarray(expected_expert.w_gate))
        np.testing.assert_array_equal(np.asarray(restored_expert.w_up), np.asarray(expected_expert.w_up))
        np.testing.assert_array_equal(np.asarray(restored_expert.w_down), np.asarray(expected_expert.w_down))


def test_data_local_expert_gradients_sync_only_at_step_boundary():
    script = textwrap.dedent(
        """
        import importlib
        import re

        import jax
        import jax.numpy as jnp
        import numpy as np
        from jax.sharding import AxisType, Mesh, NamedSharding
        from jax.sharding import PartitionSpec as P
        from levanter.grug.grug_moe import moe_mlp, moe_mlp_accumulating_weight_gradient

        from experiments.grug.moe.train import (
            StageExpertGradientAccumulators,
            _sync_expert_gradient_accumulators,
        )

        ragged_dot_module = importlib.import_module("haliax.nn.ragged_dot")
        ring_module = importlib.import_module("levanter.grug._moe.ep_ring")

        def fake_ragged_dot(
            lhs,
            rhs,
            group_sizes,
            ragged_dot_dimension_numbers=ragged_dot_module._DEFAULT_DIM_NUMS,
            *,
            output_dtype=None,
            **_,
        ):
            output = jax.lax.ragged_dot_general(
                lhs=lhs,
                rhs=rhs,
                group_sizes=group_sizes,
                ragged_dot_dimension_numbers=ragged_dot_dimension_numbers,
            )
            return output if output_dtype is None else output.astype(output_dtype)

        def fake_accumulating_ragged_dot(lhs, rhs, group_sizes, accumulator, accumulation_scale):
            fresh_gradient = fake_ragged_dot(
                lhs,
                rhs,
                group_sizes,
                ragged_dot_module._DRHS_DIM_NUMS,
                output_dtype=jnp.float32,
            )
            return fresh_gradient + accumulation_scale * accumulator

        ragged_dot_module._has_pallas_triton = True
        ragged_dot_module._triton_pallas_call = fake_ragged_dot
        ragged_dot_module._triton_ragged_contracting_dim_accumulating_pallas_call = (
            fake_accumulating_ragged_dot
        )
        ring_module.ragged_dot = fake_ragged_dot

        mesh = Mesh(
            np.asarray(jax.devices(), dtype=object).reshape((2, 2, 1)),
            ("data", "expert", "model"),
            axis_types=(AxisType.Explicit,) * 3,
        )
        batch_sharding = NamedSharding(mesh, P(("data", "expert"), None))
        expert_sharding = NamedSharding(mesh, P("expert", None, None))
        w13_gradient_sharding = NamedSharding(mesh, P("expert", "data", None))
        w2_gradient_sharding = NamedSharding(mesh, P("expert", None, "data"))

        keys = jax.random.split(jax.random.key(0), 6)
        x = jax.device_put(
            jax.random.normal(keys[0], (8, 4), dtype=jnp.bfloat16),
            batch_sharding,
        )
        selected_experts = jax.device_put(
            jnp.arange(16, dtype=jnp.int32).reshape(8, 2) % 4,
            batch_sharding,
        )
        combine_weights = jax.device_put(
            jax.nn.softmax(jax.random.normal(keys[1], (8, 2)), axis=-1).astype(jnp.bfloat16),
            batch_sharding,
        )
        w13 = jax.device_put(
            jax.random.normal(keys[2], (4, 4, 6), dtype=jnp.bfloat16),
            expert_sharding,
        )
        w2 = jax.device_put(
            jax.random.normal(keys[3], (4, 3, 4), dtype=jnp.bfloat16),
            expert_sharding,
        )
        zero_w13 = jax.device_put(jnp.zeros(w13.shape, dtype=jnp.float32), expert_sharding)
        zero_w2 = jax.device_put(jnp.zeros(w2.shape, dtype=jnp.float32), expert_sharding)
        output_cotangent = jax.device_put(
            jax.random.normal(keys[4], (8, 4), dtype=jnp.float32),
            batch_sharding,
        )

        def fused_loss(w13, w2):
            output, _, token = moe_mlp_accumulating_weight_gradient(
                x,
                selected_experts,
                combine_weights,
                w13,
                w2,
                zero_w13,
                zero_w2,
                implementation="ring",
                mesh=mesh,
                capacity_factor=1.0,
            )
            return jnp.sum(output.astype(jnp.float32) * output_cotangent) + token

        local_backward = jax.jit(
            jax.grad(fused_loss, argnums=(0, 1)),
            out_shardings=(expert_sharding, expert_sharding),
        )
        local_backward_compiled = local_backward.lower(w13, w2).compile()
        local_backward_hlo = local_backward_compiled.as_text()
        assert not re.findall(r" = [^\\n]* all-reduce\\(", local_backward_hlo, re.IGNORECASE)
        assert not re.findall(r" = [^\\n]* reduce-scatter\\(", local_backward_hlo, re.IGNORECASE)

        def sync(w13_gradient, w2_gradient):
            return _sync_expert_gradient_accumulators(
                StageExpertGradientAccumulators((w13_gradient,), (w2_gradient,))
            )

        sync_output_shardings = StageExpertGradientAccumulators(
            (w13_gradient_sharding,),
            (w2_gradient_sharding,),
        )
        with jax.set_mesh(mesh):
            sync_compiled = jax.jit(sync, out_shardings=sync_output_shardings).lower(
                zero_w13,
                zero_w2,
            ).compile()
        sync_hlo = sync_compiled.as_text()
        assert len(re.findall(r" = [^\\n]* reduce-scatter\\(", sync_hlo, re.IGNORECASE)) == 2
        assert not re.findall(r" = [^\\n]* all-reduce\\(", sync_hlo, re.IGNORECASE)

        local_w13, local_w2 = local_backward_compiled(w13, w2)
        synced = sync_compiled(local_w13, local_w2)

        def ordinary_loss(w13, w2):
            output = moe_mlp(
                x,
                selected_experts,
                combine_weights,
                w13,
                w2,
                implementation="ring",
                mesh=mesh,
                capacity_factor=1.0,
            )
            return jnp.sum(output.astype(jnp.float32) * output_cotangent)

        ordinary_w13, ordinary_w2 = jax.jit(
            jax.grad(ordinary_loss, argnums=(0, 1)),
            out_shardings=(w13_gradient_sharding, w2_gradient_sharding),
        )(w13, w2)

        def fused_input_loss(x, combine_weights):
            output, _, token = moe_mlp_accumulating_weight_gradient(
                x,
                selected_experts,
                combine_weights,
                w13,
                w2,
                zero_w13,
                zero_w2,
                implementation="ring",
                mesh=mesh,
                capacity_factor=1.0,
            )
            return jnp.sum(output.astype(jnp.float32) * output_cotangent) + token

        def ordinary_input_loss(x, combine_weights):
            output = moe_mlp(
                x,
                selected_experts,
                combine_weights,
                w13,
                w2,
                implementation="ring",
                mesh=mesh,
                capacity_factor=1.0,
            )
            return jnp.sum(output.astype(jnp.float32) * output_cotangent)

        fused_x, fused_combine_weights = jax.jit(
            jax.grad(fused_input_loss, argnums=(0, 1)),
            out_shardings=(batch_sharding, batch_sharding),
        )(x, combine_weights)
        ordinary_x, ordinary_combine_weights = jax.jit(
            jax.grad(ordinary_input_loss, argnums=(0, 1)),
            out_shardings=(batch_sharding, batch_sharding),
        )(x, combine_weights)

        for actual, expected in (
            (fused_x, ordinary_x),
            (fused_combine_weights, ordinary_combine_weights),
            (synced.w13[0], ordinary_w13),
            (synced.w2[0], ordinary_w2),
        ):
            actual = np.asarray(actual)
            expected = np.asarray(expected, dtype=np.float32)
            relative_l2 = np.linalg.norm(actual - expected) / np.linalg.norm(expected)
            assert relative_l2 <= 0.002

        def two_block_fused_loss(first_w13, first_w2, second_w13, second_w2):
            hidden, _, first_token = moe_mlp_accumulating_weight_gradient(
                x,
                selected_experts,
                combine_weights,
                first_w13,
                first_w2,
                zero_w13,
                zero_w2,
                implementation="ring",
                mesh=mesh,
                capacity_factor=1.0,
            )
            output, _, second_token = moe_mlp_accumulating_weight_gradient(
                hidden,
                selected_experts,
                combine_weights,
                second_w13,
                second_w2,
                zero_w13,
                zero_w2,
                implementation="ring",
                mesh=mesh,
                capacity_factor=1.0,
            )
            return jnp.sum(output.astype(jnp.float32) * output_cotangent) + first_token + second_token

        two_block_local_backward = jax.jit(
            jax.grad(two_block_fused_loss, argnums=(0, 1, 2, 3)),
            out_shardings=(expert_sharding,) * 4,
        ).lower(w13, w2, w13, w2).compile()
        two_block_local_gradients = two_block_local_backward(w13, w2, w13, w2)
        first_two_block_synced = sync_compiled(*two_block_local_gradients[:2])
        second_two_block_synced = sync_compiled(*two_block_local_gradients[2:])

        def two_block_ordinary_loss(first_w13, first_w2, second_w13, second_w2):
            hidden = moe_mlp(
                x,
                selected_experts,
                combine_weights,
                first_w13,
                first_w2,
                implementation="ring",
                mesh=mesh,
                capacity_factor=1.0,
            )
            output = moe_mlp(
                hidden,
                selected_experts,
                combine_weights,
                second_w13,
                second_w2,
                implementation="ring",
                mesh=mesh,
                capacity_factor=1.0,
            )
            return jnp.sum(output.astype(jnp.float32) * output_cotangent)

        two_block_ordinary_gradients = jax.jit(
            jax.grad(two_block_ordinary_loss, argnums=(0, 1, 2, 3)),
            out_shardings=(w13_gradient_sharding, w2_gradient_sharding) * 2,
        )(w13, w2, w13, w2)
        for actual, expected in (
            (first_two_block_synced.w13[0], two_block_ordinary_gradients[0]),
            (first_two_block_synced.w2[0], two_block_ordinary_gradients[1]),
            (second_two_block_synced.w13[0], two_block_ordinary_gradients[2]),
            (second_two_block_synced.w2[0], two_block_ordinary_gradients[3]),
        ):
            actual = np.asarray(actual)
            expected = np.asarray(expected, dtype=np.float32)
            relative_l2 = np.linalg.norm(actual - expected) / np.linalg.norm(expected)
            assert relative_l2 <= 0.002

        fused_output, fused_dropped, token = moe_mlp_accumulating_weight_gradient(
            x,
            selected_experts,
            combine_weights,
            w13,
            w2,
            zero_w13,
            zero_w2,
            implementation="ring",
            mesh=mesh,
            capacity_factor=1.0,
        )
        ordinary_output, ordinary_dropped = moe_mlp(
            x,
            selected_experts,
            combine_weights,
            w13,
            w2,
            implementation="ring",
            mesh=mesh,
            capacity_factor=1.0,
            report_capacity_overflow=True,
        )
        np.testing.assert_array_equal(np.asarray(fused_output), np.asarray(ordinary_output))
        assert int(fused_dropped) == int(ordinary_dropped)
        assert float(token) == 0.0
        """
    )
    result = subprocess.run(
        (sys.executable, "-c", script),
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "XLA_FLAGS": "--xla_force_host_platform_device_count=4"},
    )
    assert result.returncode == 0, result.stderr
