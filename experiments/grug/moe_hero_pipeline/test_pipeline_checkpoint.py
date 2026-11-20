# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import jmp
import numpy as np
import optax
import pytest
from levanter.checkpoint import load_checkpoint

from experiments.grug.moe_hero_ep.model import QbEstimator
from experiments.grug.moe_hero_ep.optimizer import GrugMoeMuonHConfig
from experiments.grug.moe_hero_pipeline.checkpoint import checkpoint_state, restore_checkpoint, save_checkpoint
from experiments.grug.moe_hero_pipeline.pipeline import (
    _apply_qb_betas,
    updated_pipeline_state,
)
from experiments.grug.moe_hero_pipeline.tiny_hero import packed_batch, pipeline_state, tiny_hero


@eqx.filter_jit
def _serial_training_step(state, batch, optimizer):
    # Exercise real Hero gradients and delayed QB updates without CUDA/JAXPP.
    def loss_fn(params):
        hidden = params[0].embed(batch.tokens)
        next_betas = []
        for stage, betas in zip(params, state.pending_qb_betas, strict=True):
            stage = _apply_qb_betas(stage, betas)
            hidden, metrics = stage.run_blocks(hidden, batch.attn_mask)
            next_betas.append(metrics["qb_beta_per_layer"])
        loss = stage.cross_entropy_loss(stage.finish(hidden), batch.tokens, batch.loss_weight, logsumexp_weight=0.01)
        return loss, tuple(next_betas)

    (loss, next_betas), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.trainable_params)
    next_state = updated_pipeline_state(
        state,
        grads,
        next_betas,
        optimizer,
        jmp.get_policy("params=float32,compute=float32,output=float32"),
        microbatches=1,
    )
    return next_state, loss


@pytest.mark.parametrize("optimizer_name", ["adamw", "muonh"])
def test_hero_checkpoint_restores_complete_state_with_different_stage_split(tmp_path, optimizer_name):
    mesh, model = tiny_hero(QbEstimator.HIST, devices=jax.local_devices())
    optimizer = (
        optax.adamw(1e-4)
        if optimizer_name == "adamw"
        else GrugMoeMuonHConfig(learning_rate=1e-4, adam_lr=1e-4, warmup=0).build(3)
    )

    with jax.set_mesh(mesh):
        state = pipeline_state(model, optimizer, (2, 3))
        if optimizer_name == "adamw":
            gradients = jax.tree.map(jnp.ones_like, state.trainable_params)
            updates = tuple(
                optimizer.update(gradient, opt_state, params)
                for gradient, opt_state, params in zip(gradients, state.opt_state, state.trainable_params, strict=True)
            )
            state = dataclasses.replace(
                state,
                trainable_params=tuple(
                    optax.apply_updates(params, update)
                    for params, (update, _) in zip(state.trainable_params, updates, strict=True)
                ),
                opt_state=tuple(next_state for _, next_state in updates),
            )
        else:
            # MuonH's sharded Newton-Schulz update is GPU-only; give each state
            # array a distinct nonzero value to test its nested checkpoint tree.
            state = dataclasses.replace(
                state,
                opt_state=jax.tree.map(lambda value: value + jnp.ones_like(value), state.opt_state),
            )
        canonical = checkpoint_state(state)
        root = str(tmp_path)
        contract = {"model": "tiny-hero", "optimizer": optimizer_name, "data_schedule_steps": 100}
        path = save_checkpoint(root, state, step=1, contract=contract)
        loaded = load_checkpoint(jax.tree.map(jnp.zeros_like, canonical), path)
        assert jax.tree.structure(loaded) == jax.tree.structure(canonical)
        for actual, expected in zip(jax.tree.leaves(loaded), jax.tree.leaves(canonical), strict=True):
            np.testing.assert_array_equal(actual, expected)

        destination = pipeline_state(model, optimizer, (1, 1, 1, 1, 1))
        destination = jax.tree.map(jnp.zeros_like, destination)
        shardings = jax.tree.map(lambda value: value.sharding, destination)
        # A changed mixture horizon changes random-access data order after resume.
        with pytest.raises(ValueError):
            restore_checkpoint(root, destination, shardings, contract={**contract, "data_schedule_steps": 99})
        restored, completed = restore_checkpoint(root, destination, shardings, contract=contract)
        assert completed == 1
        restored_canonical = checkpoint_state(restored)
        assert jax.tree.structure(restored_canonical) == jax.tree.structure(canonical)
        for actual, expected in zip(jax.tree.leaves(restored_canonical), jax.tree.leaves(canonical), strict=True):
            np.testing.assert_array_equal(actual, expected)


@pytest.mark.timeout(180)
@pytest.mark.parametrize("restored_layer_counts", [(2, 3), (1, 1, 1, 1, 1)])
def test_checkpoint_next_two_training_steps_match_uninterrupted_run(tmp_path, restored_layer_counts):
    mesh, model = tiny_hero(QbEstimator.HIST, devices=jax.local_devices())
    # Changing learning rates expose a reset schedule counter as well as Adam's
    # bias-correction counter. Every restart uses the full four-step schedule.
    optimizer = optax.adamw(optax.linear_schedule(1e-3, 1e-4, transition_steps=4))
    contract = {"model": "tiny-hero", "optimizer": "adamw", "training_steps": 4}
    batch = packed_batch()

    def advance(state, completed_step):
        next_batch = dataclasses.replace(batch, tokens=(batch.tokens + completed_step) % model.config.vocab_size)
        return _serial_training_step(state, next_batch, optimizer)

    with jax.set_mesh(mesh):
        uninterrupted = pipeline_state(model, optimizer, (2, 3))
        for completed_step in (1, 2):
            uninterrupted, _ = advance(uninterrupted, completed_step)
        save_checkpoint(str(tmp_path), uninterrupted, step=2, contract=contract)

        destination = pipeline_state(model, optimizer, restored_layer_counts)
        shardings = jax.tree.map(lambda value: value.sharding, destination)
        resumed, completed = restore_checkpoint(str(tmp_path), destination, shardings, contract=contract)
        assert completed == 2

        resumed_steps = []
        for completed_step in range(completed + 1, contract["training_steps"] + 1):
            uninterrupted, expected_loss = advance(uninterrupted, completed_step)
            resumed, actual_loss = advance(resumed, completed_step)
            resumed_steps.append(completed_step)
            np.testing.assert_array_equal(actual_loss, expected_loss)
            expected = checkpoint_state(uninterrupted)
            actual = checkpoint_state(resumed)
            for actual_value, expected_value in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
                np.testing.assert_array_equal(actual_value, expected_value)
        assert resumed_steps == [3, 4]
