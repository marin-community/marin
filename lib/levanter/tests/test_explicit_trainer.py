# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P

from levanter.checkpoint import CheckpointerConfig, resolve_checkpoint
from levanter.tracker import NoopTracker, current_tracker
from levanter.trainer import ExplicitTrainingLayout, Trainer, TrainerConfig


class Regression(eqx.Module):
    weight: jax.Array

    def checkpoint_metadata(self):
        return {"outputs": self.weight.shape[1]}

    def __call__(self, inputs):
        return jnp.einsum("bi,io->bo", inputs, self.weight, out_sharding=P("data", "model"))


@pytest.mark.multidevice
def test_explicit_trainer_restores_sharded_model_and_optimizer(tmp_path):
    if len(jax.devices()) < 4:
        pytest.skip("Requires two batch shards and two output shards")
    devices = np.asarray(jax.devices()[:4]).reshape(2, 2)
    mesh = Mesh(devices, ("data", "model"), axis_types=(AxisType.Explicit,) * 2)
    layout = ExplicitTrainingLayout(mesh, P("data", None))
    inputs = jax.device_put(np.arange(16, dtype=np.float32).reshape(4, 4) / 16, NamedSharding(mesh, layout.batch_spec))
    config = TrainerConfig(
        id="explicit-trainer",
        train_batch_size=4,
        num_train_steps=1,
        per_device_parallelism=1,
        load_checkpoint=False,
        require_accelerator=False,
        checkpointer=CheckpointerConfig(base_path=str(tmp_path), append_run_id_to_base_path=False),
        log_dir=tmp_path,
        log_jaxprs=False,
        log_xla_hlo=False,
    )

    def init():
        return Regression(jnp.zeros((4, 2), out_sharding=P(None, "model")))

    def loss(model, batch, *, key):
        return jnp.mean(jnp.square(model(batch) - 1))

    optimizer = optax.adam(0.01)
    key = jax.random.PRNGKey(0)
    with current_tracker(NoopTracker()), Trainer(config, optimizer, loss, layout=layout) as trainer:
        state = trainer.initial_state(key, model_init=init)
        after = trainer.train(state, [inputs]).state
        assert float(loss(after.model, inputs, key=key)) < 1.0

    checkpoint = resolve_checkpoint(str(tmp_path))
    assert checkpoint.metadata["outputs"] == 2
    restore = dataclasses.replace(config, load_checkpoint=True, load_checkpoint_path=checkpoint.path)
    with (
        current_tracker(NoopTracker()),
        Trainer(restore, optimizer, loss, layout=layout, add_default_hooks=False) as trainer,
    ):
        restored = trainer.initial_state(jax.random.PRNGKey(1), model_init=init)
        for actual, expected in zip(jax.tree.leaves(restored), jax.tree.leaves(after), strict=True):
            assert actual.sharding == expected.sharding
            np.testing.assert_array_equal(actual, expected)
        assert int(restored.step) == 1
        resumed = trainer.train_step(restored, inputs).state
        uninterrupted = trainer.train_step(after, inputs).state
        np.testing.assert_array_equal(resumed.model.weight, uninterrupted.model.weight)
        assert int(resumed.step) == 2
