# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
import optax
import jmp
import pytest

from levanter.tracker.json_file import JsonFileTrackerConfig
from levanter.trainer import Trainer, TrainerConfig
from levanter.trainer_state import TrainerState
from levanter.utils.mesh import MeshConfig

BF16_POLICY = "p=bf16,c=bf16"


@pytest.mark.parametrize("policy", [BF16_POLICY, "f32"])
def test_optimizer_precision_does_not_change_native_step_input_dtypes(tmp_path, policy):
    device_count = jax.device_count()
    config = TrainerConfig(
        id="stable-parameter-dtypes",
        train_batch_size=device_count,
        per_device_parallelism=1,
        mp=jmp.get_policy(policy),
        mesh=MeshConfig(axes={"data": -1, "model": 1, "replica": 1, "context": 1}),
        tracker=JsonFileTrackerConfig(output_path=str(tmp_path)),
        require_accelerator=False,
        log_jaxprs=False,
        log_xla_hlo=False,
    )
    optimizer = optax.adam(0.1, mu_dtype=jnp.float32)
    batch = hax.named(jnp.ones(device_count), hax.Axis("batch", device_count))

    def loss(model, batch, *, key=None):
        return hax.mean((model - batch) ** 2).scalar()

    def signature(state):
        return [(value.shape, value.dtype, jax.typeof(value).weak_type) for value in jax.tree.leaves(state)]

    with Trainer(config, optimizer, loss, add_default_hooks=False) as trainer:
        state = TrainerState.init(optimizer, jnp.asarray(2.0), key=jax.random.key(0), mp=config.mp)
        expected = signature(state)
        first_loss = None
        for _ in range(3):
            info = trainer.train_step(state, batch)
            state = info.state
            if first_loss is None:
                first_loss = info.loss
            assert signature(state) == expected
        assert int(state.step) == 3
        assert info.loss < first_loss


def test_fp32_master_parameters_accumulate_updates_below_bfloat16_resolution():
    optimizer = optax.sgd(1e-4)
    policy = jmp.get_policy("params=float32,compute=bfloat16,output=bfloat16")
    state = TrainerState.init(optimizer, jnp.asarray(1.0, jnp.bfloat16), key=jax.random.key(0), mp=policy)

    @jax.jit
    def update(state):
        return state.take_step(jnp.asarray(1.0, jnp.float32), key=state.training_key)[0]

    for _ in range(100):
        state = update(state)
    assert state.model.dtype == jnp.float32
    np.testing.assert_allclose(state.model, 0.99, atol=2e-6)


def test_skipped_optimizer_update_keeps_the_parameter():
    def skip(_grads, state, params=None, **kwargs):
        return None, state

    optimizer = optax.GradientTransformation(lambda _params: (), skip)
    policy = jmp.get_policy(BF16_POLICY)
    state = TrainerState.init(optimizer, jnp.asarray(2.0), key=jax.random.key(0), mp=policy)
    updated, updates = state.take_step(jnp.asarray(1.0), key=state.training_key)
    assert updates is None
    assert updated.model.dtype == jnp.bfloat16
    np.testing.assert_array_equal(updated.model, state.model)


class WeightedBatch(eqx.Module):
    values: hax.NamedArray
    weights: hax.NamedArray


def test_token_weighted_microbatches_match_full_weighted_update(tmp_path):
    Batch = hax.Axis("batch", 4)
    values = jnp.asarray([1.0, 3.0, 5.0, 7.0])
    weights = jnp.asarray([1.0, 0.0, 2.0, 8.0])
    batch = WeightedBatch(hax.named(values, Batch), hax.named(weights, Batch))

    def loss(model, batch, *, key=None):
        del key
        return hax.sum((model - batch.values) ** 2 * batch.weights).scalar()

    def weight(batch, *, key=None):
        del key
        return hax.sum(batch.weights).scalar()

    config = TrainerConfig(
        id="weighted-microbatch-regression",
        train_batch_size=4,
        per_device_parallelism=1,
        mesh=MeshConfig(axes={"data": 1, "model": 1, "replica": 1, "context": 1}),
        tracker=JsonFileTrackerConfig(output_path=str(tmp_path)),
        require_accelerator=False,
        log_jaxprs=False,
        log_xla_hlo=False,
    )
    optimizer = optax.sgd(0.1)
    initial = jnp.asarray(2.0)
    reference_loss = jnp.sum((initial - values) ** 2 * weights) / weights.sum()
    reference_update = initial - 0.1 * (2 * (initial - values) * weights).sum() / weights.sum()
    with Trainer(config, optimizer, loss, add_default_hooks=False, loss_weight_fn=weight) as trainer:
        state = TrainerState.init(optimizer, initial, key=jax.random.key(0), mp=config.mp)
        actual = trainer.train_step(state, batch)
    np.testing.assert_allclose(actual.loss, reference_loss, rtol=1e-6)
    np.testing.assert_allclose(actual.state.model, reference_update, rtol=1e-6)
