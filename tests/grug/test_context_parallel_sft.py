# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys
import textwrap


def test_context_parallel_packed_model_matches_unsharded_loss_and_gradients():
    # Device count is process-global. A fresh CPU process exercises real collectives
    # even on developer machines with one accelerator or no accelerator.
    script = textwrap.dedent(
        """
        import jax
        import jax.numpy as jnp
        import numpy as np
        from levanter.grug.sharding import compact_grug_mesh
        from levanter.data.text.examples import GrugLmExample
        from experiments.june_tpu_67b_a2b.moe.model import GrugModelConfig, Transformer

        config = GrugModelConfig(
            vocab_size=16, hidden_dim=8, intermediate_dim=8,
            shared_expert_intermediate_dim=8, num_experts=2, num_experts_per_token=1,
            num_layers=2, num_heads=2, num_kv_heads=2, max_seq_len=8, sliding_window=4,
            attention_implementation="reference", moe_implementation="scatter",
            disable_pko=True, disable_long_rope=True, use_array_stacked_blocks=True,
        )
        tokens = jnp.array([[1, 2, 3, 4, 5, 6, 0, 0], [6, 5, 4, 3, 2, 1, 7, 0]], jnp.int32)
        example = jax.vmap(GrugLmExample.causal)(
            tokens,
            segment_ids=jnp.array([[0, 0, 0, 1, 1, 1, -1, -1], [0, 0, 0, 0, 1, 1, 1, -1]], jnp.int32),
            loss_weight=jnp.array([[1, 1, 0, 1, 1, 0, 0, 0], [1, 1, 1, 0, 1, 1, 0, 0]], jnp.float32),
        )
        results = []
        for context in (1, 2):
            mesh = compact_grug_mesh(replica_axis_size=1, context_axis_size=context)
            with jax.set_mesh(mesh):
                model = Transformer.init(config, key=jax.random.PRNGKey(0))
                hidden, _ = jax.jit(lambda m: m(tokens, mask=example.attn_mask))(model)
                assert hidden.sharding.spec[1] == "context"
                assert hidden.addressable_shards[0].data.shape[1] == 8 // context
                value, gradients = jax.jit(jax.value_and_grad(
                    lambda m: m.next_token_loss(tokens, example.loss_weight, mask=example.attn_mask)
                ))(model)
                results.append((value, gradients))
                # Changing a previous conversation must not change the next one.
                changed = tokens.at[0, :3].set(jnp.array([7, 8, 9]))
                changed_hidden, _ = jax.jit(lambda m: m(changed, mask=example.attn_mask))(model)
                np.testing.assert_allclose(
                    np.asarray(hidden)[:, 3:6], np.asarray(changed_hidden)[:, 3:6], rtol=1e-4, atol=1e-4
                )
        for expected, actual in zip(jax.tree.leaves(results[0]), jax.tree.leaves(results[1]), strict=True):
            np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)
        """
    )
    env = {**os.environ, "JAX_PLATFORMS": "cpu", "JAX_NUM_CPU_DEVICES": "2"}
    result = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True)
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"


def test_context_parallel_splash_matches_reference_loss_and_gradients():
    script = textwrap.dedent(
        """
        import jax
        import jax.numpy as jnp
        import numpy as np
        from jax.experimental.pallas import tpu as pltpu
        from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P
        from levanter.grug.attention import attention, AttentionMask

        mesh = Mesh(np.array(jax.devices()).reshape(1, 2, 1), ("data", "context", "model"),
                    axis_types=(AxisType.Explicit,) * 3)
        q_sharding = NamedSharding(mesh, P("data", "context", "model", None))
        kv_sharding = NamedSharding(mesh, P("data", None, "model", None))
        q, k, v = [
            jax.device_put(jax.random.normal(key, (1, 256, 2, 128)) * 0.02, sharding)
            for key, sharding in zip(jax.random.split(jax.random.PRNGKey(42), 3),
                                     (q_sharding, kv_sharding, kv_sharding), strict=True)
        ]
        # A conversation crosses the context-shard boundary at position 128.
        # Segment IDs arrive replicated, as they do from a packed data loader.
        segments = jnp.array([[0] * 160 + [1] * 64 + [-1] * 32], jnp.int32)
        mask = AttentionMask(is_causal=True, sliding_window=128, segment_ids=(segments, segments))
        with jax.set_mesh(mesh):
            def loss(q, k, v, implementation):
                output = attention(q, k, v, mask, implementation=implementation)
                weights = jax.sharding.reshard(
                    (jnp.arange(256) < 224)[None, :, None, None], P(None, "context", None, None)
                )
                return jnp.sum(output * weights)

            expected = jax.jit(jax.value_and_grad(
                lambda q, k, v: loss(q, k, v, "reference"), argnums=(0, 1, 2)
            ))(q, k, v)
            with pltpu.force_tpu_interpret_mode():
                actual = jax.jit(jax.value_and_grad(
                    lambda q, k, v: loss(q, k, v, "tpu_splash"), argnums=(0, 1, 2)
                ))(q, k, v)
            for observed, reference in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
                np.testing.assert_allclose(observed, reference, rtol=1e-4, atol=1e-4)
        """
    )
    env = {**os.environ, "JAX_PLATFORMS": "cpu", "JAX_NUM_CPU_DEVICES": "2"}
    result = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True)
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
