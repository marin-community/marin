# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import equinox as eqx
import jax
import jmp
import numpy as np
import optax
from haliax.partitioning import set_mesh
from levanter.checkpoint import save_checkpoint
from levanter.grug.sharding import compact_grug_mesh
from levanter.models.snowball import SnowballConfig, SnowballTransformer

from experiments.grug.checkpointing import init_weights_only_from_checkpoint
from experiments.grug.moe_hero_ep.model import GrugModelConfig as CurrentConfig
from experiments.grug.moe_hero_ep.model import Transformer
from experiments.grug.moe_hero_ep.train import initial_state


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
    current_config = CurrentConfig(**common, qk_mult=snowball_config.qk_mult)
    mesh = compact_grug_mesh(expert_axis_size=1, replica_axis_size=1)
    with set_mesh(mesh):
        snowball = SnowballTransformer.init(snowball_config, key=jax.random.key(0))
        target = eqx.filter_eval_shape(Transformer.init, current_config, key=jax.random.key(2))
        converted, pending_qb_betas = target.with_snowball_weights(snowball)
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
