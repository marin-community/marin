# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from levanter.testing.cpu_devices import run_on_cpu_devices

# The test imports the model only in the subprocess. This import lets the CI test selector see that dependency.
import experiments.grug.moe_hero_ep.model  # noqa: F401


def test_context_parallel_logits_match_unsharded_sequence():
    # The MLPs flatten each device's own (rows, sequence shard) block, so under context parallelism
    # the experts see tokens in a different order. With capacity for every token nothing drops, and
    # the logits must equal the unsharded-sequence run on the same parameters.
    script = """
        import math

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
        tokens = np.random.default_rng(0).integers(0, config.vocab_size, size=(4, 8), dtype=np.int32)
        logits = {}
        for context, expert in [(1, 4), (2, 2)]:
            mesh = Mesh(
                np.asarray(jax.devices()).reshape(1, 1, context, expert, 1),
                ("replica_dcn", "data", "context", "expert", "model"),
                axis_types=(AxisType.Explicit,) * 5,
            )
            with jax.set_mesh(mesh):
                transformer = model.Transformer.init(config, key=jax.random.key(0))
                logits[context] = np.asarray(jax.jit(lambda m, ids: m.logits(ids))(transformer, jnp.asarray(tokens)))
        np.testing.assert_allclose(logits[2], logits[1], rtol=1e-5, atol=1e-5)
    """

    run_on_cpu_devices(script, device_count=4)
