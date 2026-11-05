# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Training and verification functions shared by the bounded drafter runs."""

import jax
import jax.numpy as jnp
import jmp
import optax

from experiments.grug.fast_track.model import Transformer
from experiments.grug.fast_track.train import GrugTrainState, _flat_train_step, _make_train_step


def draft_training_state(params: Transformer, *, learning_rate: float, z_loss_weight: float):
    """Reset draft Adam moments for a snapshot episode; retain the snapshot's router bias."""
    mp = jmp.get_policy("params=float32,compute=bfloat16,output=bfloat16")
    optimizer = optax.adamw(learning_rate, b1=0.9, b2=0.95, eps=1e-8, weight_decay=0.1)
    # The native step writes centered -pending_qb_betas into router_bias.
    biases = jnp.stack([layer.mlp.router_bias for layer in params.layers()])
    state = GrugTrainState(
        step=jnp.array(0, dtype=jnp.int32),
        params=params,
        master_params=None,
        ema_params=None,
        opt_state=optimizer.init(params),
        pending_qb_betas=-biases,
    )
    update = _make_train_step(optimizer, mp, z_loss_weight=z_loss_weight)
    leaves, structure = jax.tree.flatten(state)
    return leaves, structure, _flat_train_step(update, structure), mp


def sequence_losses(params: Transformer, batch, mp: jmp.Policy):
    """Return per-sequence plain cross-entropy and scored-token counts."""
    model = mp.cast_to_compute(params)
    loss = model.next_token_loss(batch.tokens, batch.loss_weight, mask=batch.attn_mask, reduction="none")
    count = jnp.sum(batch.loss_weight, axis=-1)
    return jnp.sum(loss, axis=-1) / jnp.maximum(count, 1), count
