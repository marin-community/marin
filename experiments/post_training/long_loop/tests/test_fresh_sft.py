# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

import equinox as eqx
import jax
import jmp
import numpy as np
import optax
import yaml
from jax.sharding import AxisType, Mesh
from levanter.checkpoint import save_checkpoint

from experiments.grug.checkpointing import restore_grug_state_from_checkpoint
from experiments.post_training.long_loop.sft import SMOKE_CONFIG, fresh_sft_config
from experiments.post_training.long_loop.sft_runtime.train import initial_state, initialize_snowball_sft


def test_fresh_sft_loads_base_without_resume_manifest_lookup(tmp_path):
    recipe = yaml.safe_load(SMOKE_CONFIG.read_text())
    recipe["trainer"]["trainer"]["load_checkpoint"] = True
    config = fresh_sft_config(
        output=str(tmp_path / "fresh"), initializer=str(tmp_path / "base"), recipe=recipe, run_id="cpu-init"
    )
    model = replace(
        config.model,
        vocab_size=32,
        hidden_dim=16,
        intermediate_dim=8,
        shared_expert_intermediate_dim=8,
        num_experts=4,
        num_experts_per_token=2,
        num_layers=2,
        num_heads=2,
        num_kv_heads=1,
        local_kv_heads=1,
        global_kv_heads=1,
        head_dim=8,
        max_seq_len=8,
        sliding_window=4,
        global_every=2,
        attention_implementation="reference",
        moe_implementation="fixed_all_to_all",
    )
    trainer = replace(config.trainer, special_token_ids=(30, 31), anchor_token_ids=(2, 3))
    mesh = Mesh(
        np.asarray(jax.devices("cpu")[:1]).reshape(1, 1, 1, 1, 1),
        ("replica_dcn", "data", "context", "expert", "model"),
        axis_types=(AxisType.Explicit,) * 5,
    )
    with jax.set_mesh(mesh):
        kwargs = dict(optimizer=optax.adam(0.001), mp=jmp.get_policy("float32"), ema_beta=None)
        base = initial_state(model, key=jax.random.key(1), **kwargs)
        fresh = initial_state(model, key=jax.random.key(2), **kwargs)
        legacy = eqx.tree_at(
            lambda p: p.stacked_blocks.stacked.shared, base.params, base.params.stacked_blocks.stacked.shared[0]
        )
        save_checkpoint(
            {"params": legacy, "pending_qb_betas": base.pending_qb_betas},
            0,
            config.trainer.trainer.initialize_from,
            is_temporary=False,
        )

        def forbidden_manifest_lookup(candidate):
            raise AssertionError(f"fresh start attempted resume manifest lookup: {candidate}")

        restored = restore_grug_state_from_checkpoint(
            fresh,
            checkpoint_search_paths=[str(tmp_path / "fresh/checkpoints")],
            load_checkpoint_setting=config.trainer.trainer.load_checkpoint,
            mesh=mesh,
            allow_partial=False,
            template_for_candidate=forbidden_manifest_lookup,
        )
        actual = initialize_snowball_sft(restored, config.trainer.trainer.initialize_from, mesh, trainer)
        np.testing.assert_array_equal(actual.params.token_embed, base.params.token_embed)
        np.testing.assert_array_equal(actual.params.output_proj[:, 30], base.params.output_proj[:, 2])
        for a, b in zip(jax.tree.leaves(actual.opt_state), jax.tree.leaves(fresh.opt_state), strict=True):
            np.testing.assert_array_equal(a, b)
        assert int(actual.step) == 0
