# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Fused linear softmax cross-entropy for grug.

This wraps the shared fused kernel API for TPU and falls back to a full-logits
reference implementation on non-TPU backends.
"""

import functools
import os

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P, get_abstract_mesh, get_mesh, reshard

from haliax.jax_utils import named_call
from levanter.kernels.pallas.fused_cross_entropy_loss import (
    fused_cross_entropy_loss_and_logsumexp_penalty,
)


# Liger-style CE: chunk the tokens, do all matmuls with cuBLAS, no custom kernel.
# Peak memory is bounded by one [chunk, V] logit tile; backward recomputes logits.
# Enabled per-shard via CE_IMPL=liger (chunk from CE_LIGER_CHUNK, default 8192).


def _liger_ce_forward(hidden, lm_head, labels, weight, logsumexp_weight, chunk_size):
    n, h = hidden.shape
    npad = -(-n // chunk_size) * chunk_size
    hp = jnp.pad(hidden, ((0, npad - n), (0, 0)))
    lp = jnp.pad(labels.astype(jnp.int32), (0, npad - n))
    wp = jnp.pad(weight, (0, npad - n))
    nc = npad // chunk_size

    def chunk_loss(a):
        h_c, l_c, w_c = a
        logits = (h_c @ lm_head).astype(jnp.float32)
        lse = jax.scipy.special.logsumexp(logits, axis=-1)
        tgt = jnp.take_along_axis(logits, l_c[:, None], axis=1)[:, 0]
        loss = lse - tgt
        if logsumexp_weight:
            loss = loss + logsumexp_weight * lse * lse
        return w_c * loss

    out = jax.lax.map(
        chunk_loss, (hp.reshape(nc, chunk_size, h), lp.reshape(nc, chunk_size), wp.reshape(nc, chunk_size))
    )
    return out.reshape(npad)[:n]


@functools.partial(jax.custom_vjp, nondiff_argnums=(4, 5))
def _liger_weighted_ce(hidden, lm_head, labels, weight, logsumexp_weight, chunk_size):
    """Per-token weight*(CE + z-loss), Liger-chunked. custom_vjp over (hidden, lm_head)."""
    return _liger_ce_forward(hidden, lm_head, labels, weight, logsumexp_weight, chunk_size)


def _liger_ce_vjp_fwd(hidden, lm_head, labels, weight, logsumexp_weight, chunk_size):
    out = _liger_ce_forward(hidden, lm_head, labels, weight, logsumexp_weight, chunk_size)
    return out, (hidden, lm_head, labels, weight)


def _liger_ce_vjp_bwd(logsumexp_weight, chunk_size, residual, g):
    hidden, lm_head, labels, weight = residual
    n, h = hidden.shape
    v = lm_head.shape[1]
    npad = -(-n // chunk_size) * chunk_size
    hp = jnp.pad(hidden, ((0, npad - n), (0, 0)))
    lp = jnp.pad(labels.astype(jnp.int32), (0, npad - n))
    wp = jnp.pad(weight, (0, npad - n))
    gp = jnp.pad(g.astype(jnp.float32), (0, npad - n))
    nc = npad // chunk_size

    def body(dw, a):
        h_c, l_c, w_c, g_c = a
        logits = (h_c @ lm_head).astype(jnp.float32)
        lse = jax.scipy.special.logsumexp(logits, axis=-1)
        probs = jnp.exp(logits - lse[:, None])
        onehot = jax.nn.one_hot(l_c, v, dtype=jnp.float32)
        coef = (1.0 + 2.0 * logsumexp_weight * lse)[:, None] if logsumexp_weight else 1.0
        scale = (w_c * g_c)[:, None]  # weight (inside loss) * incoming cotangent
        d_logits = (scale * (probs * coef - onehot)).astype(hidden.dtype)
        dh_c = d_logits @ lm_head.T
        dw_c = (h_c.T @ d_logits).astype(jnp.float32)
        return dw + dw_c, dh_c

    unroll = int(os.environ.get("CE_LIGER_UNROLL", "1"))
    dw, dh = jax.lax.scan(
        body,
        jnp.zeros((h, v), jnp.float32),
        (
            hp.reshape(nc, chunk_size, h),
            lp.reshape(nc, chunk_size),
            wp.reshape(nc, chunk_size),
            gp.reshape(nc, chunk_size),
        ),
        unroll=unroll,
    )
    # cotangents for (hidden, lm_head, labels, weight); labels/weight are not differentiated.
    return dh.reshape(npad, h)[:n].astype(hidden.dtype), dw.astype(lm_head.dtype), None, None


_liger_weighted_ce.defvjp(_liger_ce_vjp_fwd, _liger_ce_vjp_bwd)


def _batch_axis_spec(x: jax.Array):
    x_type = jax.typeof(x)
    sharding = getattr(x_type, "sharding", None)
    spec = getattr(sharding, "spec", None)
    if spec is not None and len(spec) > 0 and spec[0] is not None:
        return spec[0]
    sharding = getattr(x, "sharding", None)
    spec = getattr(sharding, "spec", None)
    if spec is not None and len(spec) > 0 and spec[0] is not None:
        return spec[0]
    return ("data",)


def _axis_names_from_spec(axis_spec) -> tuple[str, ...]:
    if axis_spec is None:
        return ()
    if isinstance(axis_spec, tuple):
        return tuple(str(name) for name in axis_spec)
    return (str(axis_spec),)


def _psum_over_axes(x: jax.Array, axis_names: tuple[str, ...]) -> jax.Array:
    if len(axis_names) == 0:
        return x
    if len(axis_names) == 1:
        return jax.lax.psum(x, axis_names[0])
    return jax.lax.psum(x, axis_names)


def _current_mesh() -> Mesh | jax.sharding.AbstractMesh:
    try:
        mesh = get_mesh()
    except ValueError:
        mesh = None
    if mesh is not None and not mesh.empty:
        return mesh
    return get_abstract_mesh()


def _reshard_for_shard_map(
    x: jax.Array,
    mesh: Mesh | jax.sharding.AbstractMesh | None,
    spec: P,
) -> jax.Array:
    if mesh is not None and not mesh.empty:
        return reshard(x, NamedSharding(mesh, spec))
    return x


@named_call
def fused_linear_softmax_cross_entropy_loss(
    hidden: jax.Array,
    lm_head: jax.Array,
    labels: jax.Array,
    *,
    weight: jax.Array | None = None,
    reduction: str = "mean",
    logsumexp_weight: float | None = None,
    dtype: jnp.dtype = jnp.float32,
    precision: jax.lax.PrecisionLike = None,
    implementation: str | tuple[str, ...] | None = None,
) -> jax.Array:
    """Compute cross-entropy loss via the fused kernel path.

    Args:
        hidden: Array with shape (..., hidden_dim).
        lm_head: Array with shape (hidden_dim, vocab_size).
        labels: Integer array with shape (...,).
        weight: Optional per-example weights with shape matching labels.
        reduction: One of {"mean", "sum", "none"}.
        logsumexp_weight: Optional z-loss weight (logsumexp^2 term).
        dtype: Accumulator dtype for logits/logsumexp.
        precision: Optional matmul precision override for XLA/reference paths.
        implementation: Optional fused CE backend selection override.

    Returns:
        If reduction=="none": array with shape labels.shape.
        Else: scalar array.
    """
    if lm_head.ndim != 2:
        raise ValueError(f"lm_head must be 2D (hidden_dim, vocab), got shape={lm_head.shape}")
    hidden_dim = hidden.shape[-1]
    if lm_head.shape[0] != hidden_dim:
        raise ValueError(f"hidden_dim mismatch: hidden={hidden_dim}, lm_head={lm_head.shape[0]}")

    reduction_mode: str | None
    if reduction == "none":
        reduction_mode = None
    elif reduction in ("sum", "mean"):
        reduction_mode = reduction
    else:
        raise ValueError(f"Unknown reduction: {reduction}")

    mesh = _current_mesh()
    has_mesh = mesh is not None and not mesh.empty
    weight_array = weight if weight is not None else jnp.ones_like(labels, dtype=dtype)
    batch_axis_spec = _batch_axis_spec(hidden) if has_mesh else None
    batch_axis_names = _axis_names_from_spec(batch_axis_spec) if has_mesh else ()

    def _loss_shard(
        shard_hidden: jax.Array,
        shard_lm_head: jax.Array,
        shard_labels: jax.Array,
        shard_weight: jax.Array,
    ) -> jax.Array:
        flat_hidden = shard_hidden.reshape((-1, hidden_dim))
        flat_labels = shard_labels.reshape((-1,)).astype(jnp.int32)
        flat_weight = shard_weight.reshape((-1,))

        if os.environ.get("CE_IMPL") == "liger":
            loss = _liger_weighted_ce(
                flat_hidden,
                shard_lm_head,
                flat_labels,
                flat_weight.astype(dtype),
                float(logsumexp_weight or 0.0),
                int(os.environ.get("CE_LIGER_CHUNK", "8192")),
            )
        else:
            loss = fused_cross_entropy_loss_and_logsumexp_penalty(
                flat_hidden,
                flat_labels,
                shard_lm_head,
                reduction=None,
                weight=flat_weight,
                logsumexp_weight=logsumexp_weight,
                dtype=dtype,
                logit_soft_cap=None,
                precision=precision,
                implementation=implementation,
            )

        if reduction_mode is None:
            return loss.reshape(shard_labels.shape)

        local_sum = jnp.sum(loss)
        local_denom = jnp.sum(flat_weight)
        total_sum = _psum_over_axes(local_sum, batch_axis_names)
        if reduction_mode == "sum":
            return total_sum
        total_denom = _psum_over_axes(local_denom, batch_axis_names)
        return jnp.where(total_denom != 0, total_sum / total_denom, jnp.zeros_like(total_denom))

    if not has_mesh:
        return _loss_shard(hidden, lm_head, labels, weight_array)

    hidden_spec = P(batch_axis_spec)
    lm_head_spec = P(None, None)
    label_spec = P(batch_axis_spec)
    hidden = _reshard_for_shard_map(hidden, mesh, hidden_spec)
    lm_head = _reshard_for_shard_map(lm_head, mesh, lm_head_spec)
    labels = _reshard_for_shard_map(labels, mesh, label_spec)
    weight_array = _reshard_for_shard_map(weight_array, mesh, label_spec)

    out_specs = hidden_spec if reduction_mode is None else P()
    return jax.shard_map(
        _loss_shard,
        mesh=mesh,
        in_specs=(hidden_spec, lm_head_spec, label_spec, label_spec),
        out_specs=out_specs,
        check_vma=False,
    )(hidden, lm_head, labels, weight_array)


__all__ = [
    "fused_linear_softmax_cross_entropy_loss",
]
