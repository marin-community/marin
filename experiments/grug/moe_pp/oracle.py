# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Non-pipelined oracle for the grug-MoE PP de-risk.

Runs the UNMODIFIED production ``Transformer.next_token_loss`` sequentially on a
``stage=1`` compact mesh -- same params, same tokens -- so the pipelined loss /
grads can be compared against it. The mesh keeps the real FSDP ``data`` + ring-EP
``expert`` + vocab-TP ``model`` sharding; only ``stage`` is absent (size 1), so
the model runs exactly as in production.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from experiments.grug.moe.model import Transformer


def oracle_loss(transformer: Transformer, token_ids: jax.Array, loss_weight: jax.Array) -> jax.Array:
    """Production next-token loss, non-pipelined, on the current (stage=1) mesh."""
    return transformer.next_token_loss(
        token_ids,
        loss_weight.astype(jnp.float32),
        reduction="mean",
        loss_dtype=jnp.float32,
    )
