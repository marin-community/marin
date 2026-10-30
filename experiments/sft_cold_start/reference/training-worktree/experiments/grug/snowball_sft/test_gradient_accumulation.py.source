# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from levanter.testing.cpu_devices import run_on_cpu_devices


@pytest.mark.timeout(300)
def test_accumulated_update_matches_full_batch_with_unequal_loss_weights():
    run_on_cpu_devices(
        """
        import jax
        import jax.numpy as jnp
        import jmp
        import numpy as np
        import optax
        from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P
        from levanter.data.text.examples import GrugLmExample
        from levanter.grug.attention import AttentionMask
        from experiments.grug.snowball_sft.model import GrugModelConfig
        from experiments.grug.snowball_sft.train import initial_state, _make_train_step

        config = GrugModelConfig(
            vocab_size=32, hidden_dim=16, intermediate_dim=8, shared_expert_intermediate_dim=8,
            num_shared_experts=1, num_experts=4, num_experts_per_token=2, num_layers=2, num_heads=2,
            num_kv_heads=1, local_kv_heads=1, global_kv_heads=1, head_dim=8, max_seq_len=8,
            sliding_window=4, global_every=2, capacity_factor=4.0,
            attention_implementation="reference", moe_implementation="fixed_all_to_all",
        )
        mesh = Mesh(np.asarray(jax.devices()).reshape(1,2,2,1,1),
                    ("replica_dcn", "data", "context", "expert", "model"),
                    axis_types=(AxisType.Explicit,) * 5)
        with jax.set_mesh(mesh):
            optimizer = optax.adam(0.001)
            mp = jmp.get_policy("float32")
            full = initial_state(config, optimizer=optimizer, mp=mp, key=jax.random.key(1), ema_beta=None)
            micro = initial_state(config, optimizer=optimizer, mp=mp, key=jax.random.key(1), ema_beta=None)
            tokens = (np.arange(32).reshape(4,8) % 29 + 1).astype(np.int32)
            weights = np.array([[0]*8, [0]*8, [1,0,0,0,0,0,0,0], [1,1,1,1,1,1,1,0]], np.float32)
            segments = np.tile(np.array([0,0,0,0,1,1,1,1], np.int32), (4,1))
            # Match the loader: labels shift before the model partitions context.
            sharding = NamedSharding(mesh, P("data", None))
            batch = GrugLmExample(
                jax.device_put(tokens, sharding), jax.device_put(weights, sharding),
                AttentionMask.causal().with_segment_ids(jax.device_put(segments, sharding)),
            )
            kwargs = dict(z_loss_weight=0, ema_beta=None, special_token_ids=(30,31),
                          special_token_lr_multiplier=4)
            full_step = _make_train_step(optimizer, mp, **kwargs)
            micro_step = _make_train_step(optimizer, mp, accumulation_steps=2, **kwargs)
            for iteration in range(2):
                full, full_metrics, _ = full_step(full, batch)
                micro, micro_metrics, _ = micro_step(micro, batch)
                assert int(full.step) == int(micro.step) == iteration + 1
                np.testing.assert_allclose(micro_metrics['train/loss'], full_metrics['train/loss'],
                                           rtol=2e-5, atol=2e-6)
                np.testing.assert_allclose(micro_metrics['train/cross_entropy_loss'], full_metrics['train/loss'],
                                           rtol=2e-5, atol=2e-6)
                for actual, expected in zip(jax.tree.leaves(micro), jax.tree.leaves(full), strict=True):
                    np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-6)
                np.testing.assert_array_equal(micro_metrics['train/router/routing_counts_per_layer'],
                                              full_metrics['train/router/routing_counts_per_layer'])
        """,
        device_count=4,
    )
