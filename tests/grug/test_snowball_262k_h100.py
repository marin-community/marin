# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import jax
import jax.numpy as jnp
import jmp
import numpy as np
import optax
from haliax.partitioning import set_mesh
from levanter.checkpoint import save_checkpoint
from levanter.data.text.datasets import LmDataConfig
from levanter.grug.sharding import compact_grug_mesh

from experiments.grug.moe_hero_ep import train as hero_train
from experiments.grug.moe_hero_ep.model import GrugModelConfig as CurrentConfig
from experiments.grug.moe_hero_ep.train import WeightInitialization, initial_state
from experiments.grug_sft import snowball_262k_h100
from experiments.june_tpu_67b_a2b.moe.model import GrugModelConfig as LegacyConfig
from experiments.june_tpu_67b_a2b.moe.model import Transformer as LegacyTransformer


def test_snowball_run_uses_one_h100_node_for_one_context_sharded_example():
    data = LmDataConfig(tokenizer="test", components={}, train_weights={})
    config = snowball_262k_h100.run_config("test-run", 7, data, (11,), ((12,),))

    assert config.resources.device.variant == "H100"
    assert config.resources.device.count == 8
    assert config.resources.replicas == 1
    assert config.processes_per_task == 8
    assert config.model.max_seq_len == 262_144
    assert config.model.attention_implementation == "gpu_fa4_cute"
    assert config.model.moe_implementation == "sonic"
    assert config.trainer.trainer.train_batch_size == 1
    assert config.trainer.context_axis_size == 8
    assert config.trainer.expert_axis_size == 1
    assert config.trainer.offload_opt_state
    assert config.trainer.weight_initialization == WeightInitialization.LEGACY_SINGLE_SHARED_EXPERT
    assert config.trainer.trainer.initialize_from == snowball_262k_h100.BASE_CHECKPOINT
    checkpoint_paths = config.trainer.trainer.load_checkpoint_path
    assert isinstance(checkpoint_paths, list)
    assert checkpoint_paths[0].endswith("/test-run/checkpoints")


def test_legacy_initializer_remaps_the_single_shared_expert(tmp_path: Path):
    common = dict(
        vocab_size=24,
        hidden_dim=12,
        intermediate_dim=16,
        shared_expert_intermediate_dim=12,
        num_experts=4,
        num_experts_per_token=2,
        num_layers=2,
        num_heads=2,
        num_kv_heads=1,
        head_dim=8,
        max_seq_len=16,
        sliding_window=4,
        moe_implementation="ring",
    )
    legacy_config = LegacyConfig(
        **common,
        use_array_stacked_blocks=True,
        disable_pko=True,
        disable_long_rope=True,
    )
    current_config = CurrentConfig(**common)
    mesh = compact_grug_mesh(expert_axis_size=1, replica_axis_size=1)
    pending_qb_betas = jnp.ones((2, 4))

    with set_mesh(mesh):
        legacy = LegacyTransformer.init(legacy_config, key=jax.random.key(0))
        fresh = initial_state(
            current_config,
            optimizer=optax.sgd(0.1),
            mp=jmp.get_policy("params=float32,compute=float32,output=float32"),
            key=jax.random.key(1),
            ema_beta=None,
        )
        save_checkpoint(
            {"params": legacy, "pending_qb_betas": pending_qb_betas},
            step=0,
            checkpoint_path=str(tmp_path),
        )
        initialized = hero_train.initialize_legacy_single_shared_expert_weights(
            fresh,
            str(tmp_path),
            mesh=mesh,
        )

    np.testing.assert_array_equal(initialized.params.token_embed, legacy.token_embed)
    initialized_shared = initialized.params.stacked_blocks.stacked.shared
    assert initialized_shared is not None
    np.testing.assert_array_equal(
        initialized_shared[0].w_gate,
        legacy.stacked_blocks.stacked.shared.w_gate,
    )
    np.testing.assert_array_equal(initialized.pending_qb_betas, pending_qb_betas)
    assert int(initialized.step) == 0
