# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Fixed-row log-normalization shared by sampling and unfused policy losses."""

import math

import jax
import jax.numpy as jnp
from jax.custom_batching import custom_vmap
from jax.sharding import NamedSharding, PartitionSpec as P, reshard

_ROW_TILE = 8


def accurate_logsumexp(logits: jax.Array) -> jax.Array:
    """Reduce the final axis with TPU's highest accuracy exp and log."""
    maximum = jnp.max(logits, axis=-1)
    maximum = jax.lax.stop_gradient(jnp.where(jnp.isfinite(maximum), maximum, 0))
    total = jnp.sum(jax.lax.exp(logits - maximum[..., None], accuracy=jax.lax.AccuracyMode.HIGHEST), axis=-1)
    return maximum + jax.lax.log(total, accuracy=jax.lax.AccuracyMode.HIGHEST)


def _local_logsumexp(logits: jax.Array) -> jax.Array:
    rows = math.prod(logits.shape[:-1])
    flat = logits.reshape((rows, logits.shape[-1]))
    padded = jnp.pad(flat, ((0, (-rows) % _ROW_TILE), (0, 0)))

    normalizers = jax.lax.map(accurate_logsumexp, padded.reshape((-1, _ROW_TILE, flat.shape[-1])))
    return normalizers.reshape((-1,))[:rows].reshape(logits.shape[:-1])


@custom_vmap
def _mapped_logsumexp(logits: jax.Array) -> jax.Array:
    logits = logits.astype(jnp.float32)
    sharding = logits.sharding if not isinstance(logits, jax.core.Tracer) else jax.typeof(logits).sharding
    if isinstance(sharding, NamedSharding) and not sharding.mesh.empty:
        spec = tuple(sharding.spec) + (None,) * (logits.ndim - len(sharding.spec))
        local_spec = P(*spec[:-1], None)
        normalizers = jax.shard_map(
            _local_logsumexp,
            mesh=sharding.mesh,
            in_specs=local_spec,
            out_specs=P(*spec[:-1]),
            check_vma=False,
        )(reshard(logits, local_spec))
    else:
        normalizers = _local_logsumexp(logits)
    # Keep the completed normalizer separate from caller logit subtraction.
    return jax.lax.optimization_barrier(normalizers)


@_mapped_logsumexp.def_vmap
def _logsumexp_vmap(_axis_size, in_batched, logits):
    # Re-enter with the physical batch axes visible before choosing row tiles.
    # The engine vmaps its sampler; a default rule would add axes outside them.
    return _mapped_logsumexp(logits), in_batched[0]


@jax.custom_jvp
def batch_invariant_logsumexp(logits: jax.Array) -> jax.Array:
    """Reduce the final vocabulary axis using the same FP32 schedule for every row.

    The reduction order is independent of the leading dimensions. The
    differentiable result preserves leading-axis sharding and accepts masked
    negative infinities in the vocabulary.
    """
    return _mapped_logsumexp(logits)


@batch_invariant_logsumexp.defjvp
def _logsumexp_jvp(primals, tangents):
    (logits,), (tangent,) = primals, tangents
    normalizers = batch_invariant_logsumexp(logits)
    probabilities = jax.lax.exp(
        logits.astype(jnp.float32) - normalizers[..., None], accuracy=jax.lax.AccuracyMode.HIGHEST
    )
    return normalizers, jnp.sum(probabilities * tangent.astype(jnp.float32), axis=-1)


def batch_invariant_log_softmax(logits: jax.Array) -> jax.Array:
    """Return FP32 policy logprobs over the final vocabulary axis."""
    logits = logits.astype(jnp.float32)
    return logits - batch_invariant_logsumexp(logits)[..., None]
