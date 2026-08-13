# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""The RMSNorm-GatedNorm boundary: forward, reverse-mode algebra, and the fused SM100 path.

Grug applies RMSNorm immediately before a rank-128 GatedNorm. The boundary lives here rather
than in a variant's ``model.py`` because every variant shares it and because the fused reverse
is mechanism -- kernels, a custom VJP, and batch-axis collectives -- rather than architecture.
Variants own the module classes and call :func:`rms_gated_norm`.
"""

from functools import partial
from typing import Literal, NamedTuple

import jax
import jax.numpy as jnp
from jax import shard_map
from jax.sharding import PartitionSpec as P, get_abstract_mesh, reshard

from levanter.grug.sharding import _batch_axes

RmsGatedNormImplementation = Literal["xla", "quack_coda_backward"]


class RmsGatedNormResiduals(NamedTuple):
    """Stock BF16 forward values retained by the fused reverse."""

    x: jax.Array
    norm_weight: jax.Array
    w_down: jax.Array
    w_up: jax.Array
    inverse_rms: jax.Array
    normalized: jax.Array
    gate_preactivation: jax.Array
    gate_hidden: jax.Array
    gate: jax.Array


class GatedNormUpCotangents(NamedTuple):
    """Cotangents emitted by the output-gate reverse."""

    direct: jax.Array
    gate_accumulator: jax.Array
    w_up: jax.Array


def exact_gated_norm_up_reverse(
    output_cotangent: jax.Array,
    residuals: RmsGatedNormResiduals,
) -> GatedNormUpCotangents:
    """Reverse the stock BF16 output gate without changing rounding sites."""
    output_cotangent = output_cotangent.reshape(residuals.normalized.shape)
    direct_cotangent = output_cotangent * residuals.gate
    gate_cotangent = output_cotangent * residuals.normalized
    gate_accumulator_cotangent = gate_cotangent * (residuals.gate * (1 - residuals.gate))
    w_up_cotangent = jnp.einsum("tr,td->rd", residuals.gate_hidden, gate_accumulator_cotangent)
    return GatedNormUpCotangents(direct_cotangent, gate_accumulator_cotangent, w_up_cotangent)


def exact_silu_backward_reference(
    output_cotangent: jax.Array,
    w_up: jax.Array,
    preactivation: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Reference for the fused SiLU-input cotangent GEMM.

    Mirrors ``quack_silu_backward_gemm``: returns the cotangent of the SiLU input and the
    recomputed SiLU output.
    """
    postactivation, silu_pullback = jax.vjp(jax.nn.silu, preactivation)
    preactivation_cotangent = silu_pullback(jnp.einsum("td,rd->tr", output_cotangent, w_up))[0]
    return preactivation_cotangent, postactivation


def exact_rms_backward_producer_reference(
    gate_preactivation_cotangent: jax.Array,
    w_down: jax.Array,
    direct_cotangent: jax.Array,
    x: jax.Array,
    norm_weight: jax.Array,
    inverse_rms: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Reference for the BF16 producer GEMM and its RMS row partials.

    Matches ``quack_coda_rms_backward_producer``, including its output contract: the row dot
    arrives as ``[M, tiles]`` partials for the caller to reduce. The kernel emits one column per
    N tile of the GEMM; this reference reduces in one pass and so emits a single column.
    """
    gate_cotangent = jnp.einsum("tr,dr->td", gate_preactivation_cotangent, w_down)
    unweighted_cotangent = gate_cotangent + direct_cotangent
    normalized_x = x.astype(jnp.float32) * inverse_rms[:, None]
    weighted_cotangent = unweighted_cotangent.astype(jnp.float32) * norm_weight
    row_dot_partial = jnp.sum(weighted_cotangent * normalized_x, axis=-1, keepdims=True)
    return unweighted_cotangent, row_dot_partial


def exact_rms_backward_consumer(
    unweighted_cotangent: jax.Array,
    row_dot: jax.Array,
    x: jax.Array,
    norm_weight: jax.Array,
    inverse_rms: jax.Array,
) -> jax.Array:
    """Apply the norm gain and reduced row scalar while emitting ``dx``."""
    normalized_x = x.astype(jnp.float32) * inverse_rms[:, None]
    weighted_cotangent = unweighted_cotangent.astype(jnp.float32) * norm_weight
    row_mean = row_dot / x.shape[-1]
    return ((weighted_cotangent - normalized_x * row_mean[:, None]) * inverse_rms[:, None]).astype(x.dtype)


def exact_rms_backward_partials_reference(
    gate_preactivation_cotangent: jax.Array,
    w_down: jax.Array,
    output_cotangent: jax.Array,
    gate: jax.Array,
    x: jax.Array,
    norm_weight: jax.Array,
    inverse_rms: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Reference for row-dot and norm-gain partials without a full-width output."""
    direct_cotangent = output_cotangent * gate
    unweighted_cotangent, row_dot_partial = exact_rms_backward_producer_reference(
        gate_preactivation_cotangent, w_down, direct_cotangent, x, norm_weight, inverse_rms
    )
    normalized_x = x.astype(jnp.float32) * inverse_rms[:, None]
    norm_weight_partial = jnp.sum(unweighted_cotangent.astype(jnp.float32) * normalized_x, axis=0, keepdims=True)
    return row_dot_partial, norm_weight_partial


def exact_rms_backward_recompute_consumer_reference(
    gate_preactivation_cotangent: jax.Array,
    w_down: jax.Array,
    output_cotangent: jax.Array,
    gate: jax.Array,
    row_dot: jax.Array,
    x: jax.Array,
    norm_weight: jax.Array,
    inverse_rms: jax.Array,
) -> jax.Array:
    """Reference for the recomputing consumer that emits final ``dx`` directly."""
    direct_cotangent = output_cotangent * gate
    unweighted_cotangent, _ = exact_rms_backward_producer_reference(
        gate_preactivation_cotangent, w_down, direct_cotangent, x, norm_weight, inverse_rms
    )
    return exact_rms_backward_consumer(unweighted_cotangent, row_dot, x, norm_weight, inverse_rms)


def rms_norm_with_inverse(x: jax.Array, weight: jax.Array, eps: float) -> tuple[jax.Array, jax.Array]:
    """Apply RMSNorm and return the per-row inverse RMS its reverse needs."""
    dtype = x.dtype
    x = x.astype(jnp.float32)
    inverse_rms = jax.lax.rsqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + eps)
    return (x * inverse_rms * weight).astype(dtype), inverse_rms[..., 0]


def gated_norm_with_residuals(
    x: jax.Array, w_down: jax.Array, w_up: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Apply GatedNorm and return the values its fused reverse needs."""
    gate_preactivation = jnp.einsum("...d,dr->...r", x, w_down)
    # TODO: silu activation here isn't explored, just cargo-culted from Qwen. Likely low-hanging
    # ablation fruit (e.g. compare no activation, relu, etc.). Note that the quack_coda_backward
    # reverse hardcodes the matching dSiLU epilogue, so an ablation has to change both.
    gate_hidden = jax.nn.silu(gate_preactivation)
    gate = jax.nn.sigmoid(jnp.einsum("...r,rd->...d", gate_hidden, w_up))
    return x * gate.astype(x.dtype), gate_preactivation, gate_hidden, gate


def _exact_forward(x, norm_weight, w_down, w_up, eps):
    """Run the stock BF16 forward while retaining exact reverse-mode residuals."""
    x_flat = x.reshape((-1, x.shape[-1]))
    normalized, inverse_rms = rms_norm_with_inverse(x_flat, norm_weight, eps)
    output, gate_preactivation, gate_hidden, gate = gated_norm_with_residuals(normalized, w_down, w_up)
    residuals = RmsGatedNormResiduals(
        x=x,
        norm_weight=norm_weight,
        w_down=w_down,
        w_up=w_up,
        inverse_rms=inverse_rms,
        normalized=normalized,
        gate_preactivation=gate_preactivation,
        gate_hidden=gate_hidden,
        gate=gate,
    )
    return output.reshape(x.shape), residuals


def _backward_kernels():
    """Return the ``(silu_backward, rms_backward_producer)`` pair driving the fused reverse.

    Imported lazily so the default XLA path never pulls in the optional SM100 dependency, and
    indirected through one function so CPU tests can substitute the pure-JAX references.
    """
    from levanter.grug._moe.quack_rms_cute import (  # noqa: PLC0415
        quack_coda_rms_backward_consumer,
        quack_coda_rms_backward_producer,
        quack_silu_backward_gemm,
    )

    return (
        quack_silu_backward_gemm,
        quack_coda_rms_backward_producer,
        quack_coda_rms_backward_consumer,
    )


@partial(jax.custom_vjp, nondiff_argnums=(4, 5))
def _fused(x, norm_weight, w_down, w_up, eps, batch_axes):
    del batch_axes
    return _exact_forward(x, norm_weight, w_down, w_up, eps)[0]


def _fused_fwd(x, norm_weight, w_down, w_up, eps, batch_axes):
    del batch_axes
    return _exact_forward(x, norm_weight, w_down, w_up, eps)


def _fused_bwd(eps, batch_axes, residuals, output_cotangent):
    silu_backward, rms_backward_producer, rms_backward_consumer = _backward_kernels()

    del eps
    x = residuals.x
    x_flat = x.reshape((-1, x.shape[-1]))
    output_cotangent = output_cotangent.reshape(residuals.normalized.shape)
    gate_cotangent = output_cotangent * residuals.normalized
    gate_accumulator = gate_cotangent * (residuals.gate * (1 - residuals.gate))
    w_up_cotangent = jnp.einsum("tr,td->rd", residuals.gate_hidden, gate_accumulator)
    gate_preactivation_cotangent, _ = silu_backward(gate_accumulator, residuals.w_up, residuals.gate_preactivation)
    w_down_cotangent = jnp.einsum("td,tr->dr", residuals.normalized, gate_preactivation_cotangent)
    row_dot_partial, norm_weight_partial = rms_backward_producer(
        gate_preactivation_cotangent,
        residuals.w_down,
        output_cotangent,
        residuals.gate,
        x_flat,
        residuals.norm_weight,
        residuals.inverse_rms,
    )
    row_dot = jnp.sum(row_dot_partial, axis=-1)
    norm_weight_cotangent = jnp.sum(norm_weight_partial, axis=0).astype(residuals.norm_weight.dtype)
    x_cotangent = rms_backward_consumer(
        gate_preactivation_cotangent,
        residuals.w_down,
        output_cotangent,
        residuals.gate,
        row_dot,
        x_flat,
        residuals.norm_weight,
        residuals.inverse_rms,
    ).reshape(x.shape)
    # The parameters enter replicated, so their cotangents must be reduced over the axes the
    # tokens are sharded across. shard_map defaults to check_vma=True, which suppresses the
    # transpose's own defensive psum and requires the reduction here.
    norm_weight_cotangent = jax.lax.psum(norm_weight_cotangent, axis_name=batch_axes)
    w_down_cotangent = jax.lax.psum(w_down_cotangent, axis_name=batch_axes)
    w_up_cotangent = jax.lax.psum(w_up_cotangent, axis_name=batch_axes)
    return x_cotangent, norm_weight_cotangent, w_down_cotangent, w_up_cotangent


_fused.defvjp(_fused_fwd, _fused_bwd)


def rms_gated_norm(
    x: jax.Array,
    *,
    norm_weight: jax.Array,
    w_down: jax.Array,
    w_up: jax.Array,
    eps: float,
    implementation: RmsGatedNormImplementation,
) -> jax.Array:
    """Apply the RMSNorm-GatedNorm boundary with an explicit implementation.

    ``xla`` is the stock path. ``quack_coda_backward`` keeps that forward bit-identical and
    replaces the reverse with the fused SM100 kernels.
    """
    if implementation not in ("xla", "quack_coda_backward"):
        raise ValueError(f"unsupported RMS-GatedNorm implementation {implementation!r}")

    norm_weight = reshard(norm_weight, P(None))
    w_down = reshard(w_down, P(None, None))
    w_up = reshard(w_up, P(None, None))
    if implementation == "xla":
        normalized, _ = rms_norm_with_inverse(x, norm_weight, eps)
        return gated_norm_with_residuals(normalized, w_down, w_up)[0]

    batch_axes = _batch_axes(get_abstract_mesh())
    x = reshard(x, P(batch_axes))

    def _local(local_x, local_norm_weight, local_w_down, local_w_up):
        return _fused(local_x, local_norm_weight, local_w_down, local_w_up, eps, batch_axes)

    return shard_map(
        _local,
        mesh=get_abstract_mesh(),
        in_specs=(P(batch_axes, None, None), P(None), P(None, None), P(None, None)),
        out_specs=P(batch_axes, None, None),
        # Pallas clears the VMA annotation on aliased outputs. This boundary is device-local and
        # all cross-device parameter reductions are explicit below, so disabling the redundant
        # nested check avoids a false custom-VJP type mismatch without changing collectives.
        check_vma=False,
    )(x, norm_weight, w_down, w_up)
