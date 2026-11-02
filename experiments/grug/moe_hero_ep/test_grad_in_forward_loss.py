# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from levanter.testing.cpu_devices import run_on_cpu_devices


@pytest.mark.timeout(300)
def test_ce_grad_in_forward_matches_default_loss_and_parameter_grads():
    # The grad-in-forward cross-entropy must leave the training loss and every parameter
    # gradient unchanged. Eight tokens per expert shard and a 3-row block exercise padding.
    script = """
        import dataclasses
        import math

        import equinox as eqx
        import jax
        import jax.numpy as jnp
        import numpy as np
        from jax.sharding import AxisType, Mesh

        from experiments.grug.moe_hero_ep import model

        config = model.GrugModelConfig(
            vocab_size=128, hidden_dim=32, intermediate_dim=16, shared_expert_intermediate_dim=16,
            num_shared_experts=1, num_experts=4, num_experts_per_token=2, num_layers=2, num_heads=4,
            num_kv_heads=2, local_kv_heads=2, global_kv_heads=2, head_dim=8, max_seq_len=8,
            sliding_window=4, global_every=2, capacity_factor=4.0, initializer_std=0.5 / math.sqrt(32),
            qk_mult=1.3, attention_implementation="reference", moe_implementation="fixed_all_to_all",
        )
        rng = np.random.default_rng(0)
        tokens = jnp.asarray(rng.integers(0, config.vocab_size, size=(4, 8), dtype=np.int32))
        loss_weight = jnp.asarray(rng.uniform(0.0, 2.0, size=(4, 8)).astype(np.float32)).at[:, -1].set(0.0)
        mesh = Mesh(
            np.asarray(jax.devices()).reshape(1, 1, 1, 4, 1),
            ("replica_dcn", "data", "context", "expert", "model"),
            axis_types=(AxisType.Explicit,) * 5,
        )

        def loss(transformer):
            return transformer.next_token_loss(tokens, loss_weight, logsumexp_weight=1e-4)

        results = {}
        with jax.set_mesh(mesh):
            for block in (None, 3):
                block_config = dataclasses.replace(config, ce_grad_in_forward_token_block_size=block)
                transformer = model.Transformer.init(block_config, key=jax.random.key(0))
                value, grads = eqx.filter_jit(eqx.filter_value_and_grad(loss))(transformer)
                results[block] = (np.asarray(value), [np.asarray(g) for g in jax.tree.leaves(grads)])

        np.testing.assert_allclose(results[3][0], results[None][0], rtol=1e-6, atol=1e-6)
        assert len(results[3][1]) == len(results[None][1]) > 0
        for actual, expected in zip(results[3][1], results[None][1], strict=True):
            np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
    """

    run_on_cpu_devices(script, device_count=4)
