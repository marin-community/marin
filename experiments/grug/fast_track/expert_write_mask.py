# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The fixed random output-block pattern of ``expert_write_blocks`` (shared by the model and the optimizer)."""

import jax
import jax.numpy as jnp
from jax import random

_EXPERT_WRITE_BLOCKS_SALT = 7311


def expert_write_block_ids(neurons: int, blocks: int, keep: int) -> jax.Array:
    """``[I, keep]`` sorted output-block ids each neuron writes: a fixed random subset per neuron."""
    keys = random.split(random.PRNGKey(_EXPERT_WRITE_BLOCKS_SALT), neurons)
    perms = jax.vmap(lambda k: random.permutation(k, blocks))(keys)
    return jnp.sort(perms[:, :keep], axis=-1)


def expert_write_mask(neurons: int, out: int, blocks: int, keep: int) -> jax.Array:
    """``[I, out]`` 0/1 mask: neuron i writes the ``out / blocks``-wide output blocks ``expert_write_block_ids[i]``."""
    ids = expert_write_block_ids(neurons, blocks, keep)
    per_block = jnp.max(jax.nn.one_hot(ids, blocks, dtype=jnp.float32), axis=1)  # [I, B]
    return jnp.repeat(per_block, out // blocks, axis=-1)
