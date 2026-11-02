# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys
import textwrap


def test_reset_rows_copy_anchors_or_take_the_mean_and_clear_moments():
    # A context-parallel mesh shards the LM head the way the replay runs; device count is process-global.
    script = textwrap.dedent(
        """
        import jax
        import jax.numpy as jnp
        import numpy as np
        import optax
        from levanter.grug.sharding import compact_grug_mesh
        from experiments.grug_sft.special_token_train import GrugTrainState, reinitialize_token_rows
        from experiments.june_tpu_67b_a2b.moe.model import GrugModelConfig, Transformer

        config = GrugModelConfig(
            vocab_size=16, hidden_dim=8, intermediate_dim=8,
            shared_expert_intermediate_dim=8, num_experts=2, num_experts_per_token=1,
            num_layers=2, num_heads=2, num_kv_heads=2, max_seq_len=8, sliding_window=4,
            attention_implementation="reference", moe_implementation="scatter",
            disable_pko=True, disable_long_rope=True, use_array_stacked_blocks=True,
        )
        with jax.set_mesh(compact_grug_mesh(replica_axis_size=1, context_axis_size=2)):
            params = Transformer.init(config, key=jax.random.PRNGKey(0))
            optimizer = optax.adam(1e-2)
            opt_state = optimizer.init(params)
            grads = jax.tree.map(jnp.ones_like, params)
            _, opt_state = optimizer.update(grads, opt_state, params)
            state = GrugTrainState(
                step=jnp.array(0, jnp.int32), params=params, opt_state=opt_state, ema_params=None,
                pending_qb_betas=jnp.zeros((config.num_layers, config.num_experts)),
            )
            new = reinitialize_token_rows(state, (14, 15), ((3,), (4,)), mean_seeded=(15,))
        before = np.asarray(params.output_proj)
        after = np.asarray(new.params.output_proj)
        np.testing.assert_allclose(after[:, 14], before[:, 3], rtol=1e-6)
        np.testing.assert_allclose(after[:, 15], before[:, :14].mean(axis=1), rtol=1e-5, atol=1e-7)
        np.testing.assert_array_equal(after[:, :14], before[:, :14])
        mu = np.asarray(new.opt_state[0].mu.output_proj)
        assert (mu[:, 14:] == 0).all() and (mu[:, :14] != 0).all()
        """
    )
    env = {**os.environ, "JAX_PLATFORMS": "cpu", "JAX_NUM_CPU_DEVICES": "2"}
    result = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True)
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
