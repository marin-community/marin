# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh
from levanter.checkpoint import Checkpointer

from experiments.post_training.long_loop.sft_runtime.train import restore_first_checkpoint


class State(NamedTuple):
    step: jax.Array
    params: jax.Array
    optimizer: jax.Array


def test_first_step_restores_full_state_and_releases_old_buffers(tmp_path):
    mesh = Mesh(np.asarray(jax.devices('cpu')[:1]), ('data',))
    with jax.set_mesh(mesh):
        state = State(jnp.array(1), jnp.arange(32, dtype=jnp.float32), jnp.full(32, 0.125))
        expected = [np.asarray(x).copy() for x in state]
        checkpointer = Checkpointer(tmp_path, save_interval=None, step_policies=[])
        restored = restore_first_checkpoint(state, checkpointer, mesh)
        for original, actual, wanted in zip(state, restored, expected, strict=True):
            assert original.is_deleted()
            np.testing.assert_array_equal(actual, wanted)
        assert (tmp_path / 'step-1' / 'metadata.json').is_file()
