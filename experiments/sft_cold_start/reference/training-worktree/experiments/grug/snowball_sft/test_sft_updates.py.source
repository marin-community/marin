# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from levanter.testing.cpu_devices import run_on_cpu_devices


@pytest.mark.timeout(300)
def test_sft_update_preserves_router_and_scales_selected_rows():
    run_on_cpu_devices(
        """
        import dataclasses
        import tempfile
        import equinox as eqx
        import jax
        import jax.numpy as jnp
        import jmp
        import numpy as np
        import optax
        from jax.sharding import AxisType, Mesh
        from levanter.data.text.examples import GrugLmExample
        from levanter.checkpoint import save_checkpoint
        from experiments.grug.snowball_sft.model import GrugModelConfig
        from experiments.grug.snowball_sft.train import (
            GrugTrainerConfig, initial_state, initialize_snowball_sft, reseed_lm_head,
            _make_train_step, _loss_and_grads,
        )

        config = GrugModelConfig(
            vocab_size=32, hidden_dim=16, intermediate_dim=8, shared_expert_intermediate_dim=8,
            num_shared_experts=1, num_experts=4, num_experts_per_token=2, num_layers=2, num_heads=2,
            num_kv_heads=1, local_kv_heads=1, global_kv_heads=1, head_dim=8, max_seq_len=8,
            sliding_window=4, global_every=2, capacity_factor=4.0,
            attention_implementation="reference", moe_implementation="fixed_all_to_all",
        )
        mesh = Mesh(np.asarray(jax.devices()).reshape(1,1,1,1,1),
                    ("replica_dcn", "data", "context", "expert", "model"),
                    axis_types=(AxisType.Explicit,) * 5)
        with jax.set_mesh(mesh):
            optimizer = optax.adam(0.001)
            mp = jmp.get_policy("float32")
            state = initial_state(config, optimizer=optimizer, mp=mp, key=jax.random.key(1), ema_beta=None)
            before_embed = np.asarray(state.params.token_embed).copy()
            before_head = np.asarray(state.params.output_proj).copy()
            seeded = reseed_lm_head(state.params, (30, 31), (2, 3))
            np.testing.assert_array_equal(seeded.token_embed, before_embed)
            np.testing.assert_array_equal(seeded.output_proj[:, 30:32], before_head[:, 2:4])
            np.testing.assert_array_equal(seeded.output_proj[:, :30], before_head[:, :30])
            legacy = eqx.tree_at(lambda m: m.stacked_blocks.stacked.shared,
                                 state.params, state.params.stacked_blocks.stacked.shared[0])
            with tempfile.TemporaryDirectory() as directory:
                save_checkpoint({'params': legacy, 'pending_qb_betas': state.pending_qb_betas}, 0, directory)
                fresh = initial_state(config, optimizer=optimizer, mp=mp, key=jax.random.key(8), ema_beta=None)
                loaded = initialize_snowball_sft(fresh, directory, mesh,
                           GrugTrainerConfig(special_token_ids=(30,31), anchor_token_ids=(2,3)))
                for actual, expected in zip(jax.tree.leaves(loaded.params), jax.tree.leaves(seeded), strict=True):
                    np.testing.assert_array_equal(actual, expected)
                assert int(loaded.step) == 0
                for actual, expected in zip(
                    jax.tree.leaves(loaded.opt_state), jax.tree.leaves(fresh.opt_state), strict=True
                ):
                    np.testing.assert_array_equal(actual, expected)
            # Nonzero optimizer momentum would move routing biases even with zero current gradient.
            seeded = eqx.tree_at(lambda m: m.stacked_blocks.stacked.mlp.router_bias, seeded,
                                 jnp.array([[.7,-.4,.2,-.5],[-.3,.1,.8,-.6]]))
            _, opt_state = optimizer.update(jax.tree.map(jnp.ones_like, seeded), state.opt_state, seeded)
            state = dataclasses.replace(state, params=seeded, opt_state=opt_state,
                                        pending_qb_betas=jnp.tile(jnp.array([.1,.2,.3,.4]), (2,1)))
            tokens = jnp.array([[1,2,30,4,5,31,7,8]], dtype=jnp.int32)
            batch = GrugLmExample(tokens, jnp.ones_like(tokens, dtype=jnp.float32).at[:, -1].set(0))
            # The HF checkpoint already contains the effective routing bias. Pending
            # pretraining QB state must not replace it during frozen-bias SFT.
            qb = state.params
            (_, _), grads = _loss_and_grads(qb, batch, mp, None)
            ordinary_updates, _ = optimizer.update(grads, state.opt_state, qb)
            expected_embed = np.asarray(qb.token_embed + ordinary_updates.token_embed).copy()
            expected_selected = np.asarray(qb.token_embed[30:] + 4 * ordinary_updates.token_embed[30:]).copy()
            expected_head = np.asarray(qb.output_proj[:, 30:] + 4 * ordinary_updates.output_proj[:, 30:]).copy()
            expected_router = np.asarray(qb.stacked_blocks.stacked.mlp.router_bias).copy()
            expected_pending = np.asarray(state.pending_qb_betas).copy()
            step = _make_train_step(optimizer, mp, z_loss_weight=0, ema_beta=None,
                                    special_token_ids=(30,31), special_token_lr_multiplier=4)
            result, metrics, _ = step(state, batch)
            assert np.isfinite(float(metrics['train/loss']))
            np.testing.assert_allclose(result.params.token_embed[:30], expected_embed[:30], rtol=1e-6, atol=1e-7)
            np.testing.assert_allclose(result.params.token_embed[30:], expected_selected, rtol=1e-6, atol=1e-7)
            np.testing.assert_allclose(result.params.output_proj[:,30:], expected_head, rtol=1e-6, atol=1e-7)
            np.testing.assert_array_equal(result.params.stacked_blocks.stacked.mlp.router_bias, expected_router)
            np.testing.assert_array_equal(result.pending_qb_betas, expected_pending)
        """,
        device_count=1,
    )
