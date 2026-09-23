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
from levanter.grug.sharding import compact_grug_mesh
from levanter.models.snowball import SnowballConfig, SnowballTransformer

from experiments.grug.checkpointing import init_weights_only_from_checkpoint
from experiments.grug.moe_hero_ep.model import GrugModelConfig as CurrentConfig
from experiments.grug.moe_hero_ep.train import initial_state
from experiments.grug_sft.snowball_hf_import import import_snowball_hf_weights


def test_snowball_checkpoint_initializes_current_trainer_weights(tmp_path: Path):
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
    snowball_config = SnowballConfig(**common)
    current_config = CurrentConfig(**common)
    mesh = compact_grug_mesh(expert_axis_size=1, replica_axis_size=1)
    pending_qb_betas = jnp.ones((2, 4))

    with set_mesh(mesh):
        snowball = SnowballTransformer.init(snowball_config, key=jax.random.key(0))
        converted, _ = import_snowball_hf_weights(
            snowball_config,
            current_config,
            snowball.to_state_dict(),
            key=jax.random.key(2),
        )
        fresh = initial_state(
            current_config,
            optimizer=optax.sgd(0.1),
            mp=jmp.get_policy("params=float32,compute=float32,output=float32"),
            key=jax.random.key(1),
            ema_beta=None,
        )
        save_checkpoint(
            {"params": converted, "pending_qb_betas": pending_qb_betas},
            step=0,
            checkpoint_path=str(tmp_path),
        )
        initialized = init_weights_only_from_checkpoint(
            fresh,
            str(tmp_path),
            mesh=mesh,
            allow_partial=False,
            additional_weight_fields=("pending_qb_betas",),
        )

    np.testing.assert_array_equal(initialized.params.token_embed, snowball.token_embed)
    initialized_shared = initialized.params.stacked_blocks.stacked.shared
    assert initialized_shared is not None
    np.testing.assert_array_equal(
        initialized_shared[0].w_gate,
        converted.stacked_blocks.stacked.shared[0].w_gate,
    )
    np.testing.assert_array_equal(initialized.pending_qb_betas, pending_qb_betas)
    assert int(initialized.step) == 0
