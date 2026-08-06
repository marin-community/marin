# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from haliax.nn import ArrayStacked
from levanter.checkpoint import save_checkpoint
from levanter.grug.grug_moe import MoEExpertMlp
from levanter.grug.sharding import compact_grug_mesh

from experiments.june_tpu_67b_a2b.checkpointing import load_june_checkpoint
from experiments.june_tpu_67b_a2b.moe.model import GrugModelConfig, Transformer
from experiments.june_tpu_67b_a2b.moe.train import GrugTrainState, init_weights_only_from_checkpoint


class _LegacyMoEMLP(eqx.Module):
    router: jax.Array
    router_bias: jax.Array
    expert_mlp: MoEExpertMlp
    cfg: GrugModelConfig = eqx.field(static=True)


class _LegacyBlock(eqx.Module):
    rms_attn: Any
    attn_gated_norm: Any
    attn: Any
    rms_mlp: Any
    mlp_gated_norm: Any
    mlp: _LegacyMoEMLP
    shared: Any


class _LegacyTransformer(eqx.Module):
    token_embed: jax.Array
    embed_norm: Any
    embed_gated_norm: Any
    output_proj: jax.Array
    blocks: None
    stacked_blocks: ArrayStacked[_LegacyBlock]
    final_norm: Any
    final_gated_norm: Any
    config: GrugModelConfig = eqx.field(static=True)


def _tiny_stacked_config() -> GrugModelConfig:
    return GrugModelConfig(
        vocab_size=32,
        hidden_dim=16,
        intermediate_dim=16,
        shared_expert_intermediate_dim=16,
        num_experts=4,
        num_experts_per_token=2,
        num_layers=3,
        num_heads=2,
        num_kv_heads=1,
        max_seq_len=8,
        sliding_window=4,
        moe_implementation="ring",
        disable_pko=True,
        use_array_stacked_blocks=True,
    )


def _legacy_model_from_explicit(model: Transformer) -> _LegacyTransformer:
    assert model.stacked_blocks is not None
    assert isinstance(model.expert_banks, ArrayStacked)
    block = model.stacked_blocks.stacked
    legacy_mlp = _LegacyMoEMLP(
        router=block.mlp.router,
        router_bias=block.mlp.router_bias,
        expert_mlp=model.expert_banks.stacked,
        cfg=block.mlp.cfg,
    )
    legacy_block = _LegacyBlock(
        rms_attn=block.rms_attn,
        attn_gated_norm=block.attn_gated_norm,
        attn=block.attn,
        rms_mlp=block.rms_mlp,
        mlp_gated_norm=block.mlp_gated_norm,
        mlp=legacy_mlp,
        shared=block.shared,
    )
    legacy_stacked_blocks = dataclasses.replace(model.stacked_blocks, stacked=legacy_block)
    return _LegacyTransformer(
        token_embed=model.token_embed,
        embed_norm=model.embed_norm,
        embed_gated_norm=model.embed_gated_norm,
        output_proj=model.output_proj,
        blocks=None,
        stacked_blocks=legacy_stacked_blocks,
        final_norm=model.final_norm,
        final_gated_norm=model.final_gated_norm,
        config=model.config,
    )


def _array_leaves(tree) -> list[jax.Array]:
    return jax.tree_util.tree_leaves(eqx.filter(tree, eqx.is_array))


def _assert_array_trees_equal(actual, expected) -> None:
    for actual_leaf, expected_leaf in zip(_array_leaves(actual), _array_leaves(expected), strict=True):
        np.testing.assert_array_equal(actual_leaf, expected_leaf)


def _save_legacy_checkpoint(path, source: Transformer, pending_qb_betas: jax.Array) -> None:
    save_checkpoint(
        {
            "params": _legacy_model_from_explicit(source),
            "pending_qb_betas": pending_qb_betas,
        },
        step=7,
        checkpoint_path=path,
    )


def test_load_june_checkpoint_adapts_legacy_stacked_experts_and_preserves_qb(tmp_path):
    config = _tiny_stacked_config()
    mesh = compact_grug_mesh(expert_axis_size=1)
    with jax.set_mesh(mesh):
        source = Transformer.init(config, key=jax.random.key(0))
        target = Transformer.init(config, key=jax.random.key(1))
        pending_qb_betas = jnp.arange(config.num_layers * config.num_experts, dtype=jnp.float32).reshape(
            config.num_layers, config.num_experts
        )
        _save_legacy_checkpoint(tmp_path, source, pending_qb_betas)

        loaded = load_june_checkpoint(
            {
                "params": target,
                "pending_qb_betas": jnp.zeros_like(pending_qb_betas),
            },
            str(tmp_path),
            mesh=mesh,
            allow_partial=True,
        )

    np.testing.assert_array_equal(loaded["pending_qb_betas"], pending_qb_betas)
    _assert_array_trees_equal(loaded["params"], source)


def test_load_june_checkpoint_rejects_legacy_optimizer_state(tmp_path):
    config = _tiny_stacked_config()
    mesh = compact_grug_mesh(expert_axis_size=1)
    with jax.set_mesh(mesh):
        source = Transformer.init(config, key=jax.random.key(2))
        target = Transformer.init(config, key=jax.random.key(3))
        pending_qb_betas = jnp.arange(config.num_layers * config.num_experts, dtype=jnp.float32).reshape(
            config.num_layers, config.num_experts
        )
        _save_legacy_checkpoint(tmp_path, source, pending_qb_betas)

        with pytest.raises(ValueError, match="optimizer-state migration is not supported"):
            load_june_checkpoint(
                {
                    "params": target,
                    "opt_state": {"fresh": jnp.array(0)},
                    "pending_qb_betas": jnp.zeros_like(pending_qb_betas),
                },
                str(tmp_path),
                mesh=mesh,
                allow_partial=True,
            )


def test_load_june_checkpoint_keeps_current_explicit_bank_format(tmp_path):
    config = _tiny_stacked_config()
    mesh = compact_grug_mesh(expert_axis_size=1)
    with jax.set_mesh(mesh):
        source = Transformer.init(config, key=jax.random.key(6))
        target = Transformer.init(config, key=jax.random.key(7))
        pending_qb_betas = jnp.arange(config.num_layers * config.num_experts, dtype=jnp.float32).reshape(
            config.num_layers, config.num_experts
        )
        save_checkpoint(
            {"params": source, "pending_qb_betas": pending_qb_betas},
            step=8,
            checkpoint_path=tmp_path,
        )

        loaded = load_june_checkpoint(
            {"params": target, "pending_qb_betas": jnp.zeros_like(pending_qb_betas)},
            str(tmp_path),
            mesh=mesh,
        )

    np.testing.assert_array_equal(loaded["pending_qb_betas"], pending_qb_betas)
    _assert_array_trees_equal(loaded["params"], source)


def test_weights_only_init_uses_legacy_expert_adapter_and_keeps_fresh_training_state(tmp_path):
    config = _tiny_stacked_config()
    mesh = compact_grug_mesh(expert_axis_size=1)
    with jax.set_mesh(mesh):
        source = Transformer.init(config, key=jax.random.key(4))
        target = Transformer.init(config, key=jax.random.key(5))
        pending_qb_betas = jnp.arange(config.num_layers * config.num_experts, dtype=jnp.float32).reshape(
            config.num_layers, config.num_experts
        )
        _save_legacy_checkpoint(tmp_path, source, pending_qb_betas)
        fresh_opt_state = {"fresh": jnp.array([13.0])}
        state = GrugTrainState(
            step=jnp.array(0, dtype=jnp.int32),
            params=target,
            opt_state=fresh_opt_state,
            ema_params=target,
            pending_qb_betas=jnp.zeros_like(pending_qb_betas),
        )

        loaded = init_weights_only_from_checkpoint(
            state,
            str(tmp_path),
            mesh=mesh,
            load_ema=True,
        )

    np.testing.assert_array_equal(loaded.pending_qb_betas, pending_qb_betas)
    np.testing.assert_array_equal(loaded.step, state.step)
    assert loaded.opt_state is fresh_opt_state
    _assert_array_trees_equal(loaded.params, source)
    _assert_array_trees_equal(loaded.ema_params, source)
