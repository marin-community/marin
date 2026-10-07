# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Loss helpers for ``lm_head_prototypes``: the targets' per-prototype logits and the fused CE kernel's reduction."""

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Float, Int


def prototype_target_logits(
    head_in: Float[Array, "B S E"],
    lm_head: Float[Array, "E KV"],
    labels: Int[Array, "B S"],
    *,
    vocab_size: int,
    prototypes: int,
    cap: float | tuple[float, float, float] | None,
    batch_axes: tuple[str, ...],
) -> Float[Array, "B S K"]:
    """The soft-capped logits of each label's ``lm_head_prototypes`` columns (``k V + label``)."""
    columns = labels[..., None] + vocab_size * jnp.arange(prototypes)  # [B, S, K]
    head_t = jax.sharding.reshard(lm_head.T, P(None, None))
    rows = head_t.at[columns].get(out_sharding=P(batch_axes, None, None, None))  # [B, S, K, E]
    logits = jnp.einsum("bse,bske->bsk", head_in.astype(jnp.float32), rows.astype(jnp.float32))
    if cap is None:
        return logits
    if isinstance(cap, tuple):
        a, b, c = cap
        return a * jax.nn.sigmoid((logits + b) / c)
    return jnp.tanh(logits / cap) * cap


def reduce_token_loss(loss: jax.Array, reduction: str | None, weight: jax.Array | None) -> jax.Array:
    """The fused CE kernel's reduction (weighted mean / sum / none) for an externally adjusted per-token loss."""
    if weight is not None:
        loss = loss * weight.astype(loss.dtype)
    if reduction in (None, "none"):
        return loss
    if reduction == "sum":
        return jnp.sum(loss)
    if reduction != "mean":
        raise ValueError(f"Unsupported reduction: {reduction}")
    if weight is None:
        return jnp.mean(loss)
    denom = jnp.sum(weight.astype(loss.dtype))
    return jnp.where(denom != 0, jnp.sum(loss) / denom, jnp.zeros_like(denom))
