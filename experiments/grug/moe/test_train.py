# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import jmp
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from experiments.grug.moe.train import _BATCH_AXES, _tagged_eval_loss_fn


class _DtypeProbeModel(eqx.Module):
    """Model whose per-position loss echoes the dtype of its weights at forward time.

    `next_token_loss` returns zeros in `weight.dtype`, so the loss dtype reports
    whether the eval path cast the model to the compute dtype before the forward.
    """

    weight: jax.Array

    def next_token_loss(self, tokens, loss_weight, *, mask, reduction, logsumexp_weight):
        _ = (mask, reduction, logsumexp_weight)
        return jnp.zeros(tokens.shape, dtype=self.weight.dtype) + self.weight[0]


@dataclasses.dataclass(frozen=True)
class _FakeBatch:
    tokens: jax.Array
    loss_weight: jax.Array
    attn_mask: jax.Array | None


def test_tagged_eval_runs_forward_in_compute_dtype():
    # Regression for the tagged-eval mixed-precision bug: evaluation ran the model
    # on stored float32 parameters, so bfloat16-only kernels rejected them. The
    # eval loss must apply the training compute cast (mp.cast_to_compute).
    mp = jmp.get_policy("params=float32,compute=bfloat16,output=bfloat16")
    model = _DtypeProbeModel(weight=jnp.ones((2,), dtype=jnp.float32))
    assert model.weight.dtype == jnp.float32

    mesh = jax.make_mesh((1, 1, 1), _BATCH_AXES)
    per_pos_sharding = NamedSharding(mesh, P(_BATCH_AXES, None))
    batch = _FakeBatch(
        tokens=jnp.arange(6, dtype=jnp.int32).reshape(2, 3),
        loss_weight=jnp.ones((2, 3), dtype=jnp.float32),
        attn_mask=None,
    )

    per_pos_loss, per_pos_weight, per_pos_token_id = _tagged_eval_loss_fn(
        model, batch, mp=mp, per_pos_sharding=per_pos_sharding
    )

    assert per_pos_loss.dtype == jnp.bfloat16
    assert per_pos_loss.shape == (2, 3)
    assert per_pos_weight.shape == (2, 3)
    assert per_pos_token_id.shape == (2, 3)
