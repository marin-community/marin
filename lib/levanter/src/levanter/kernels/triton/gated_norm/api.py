# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Public API and backend selection for the fused ``GatedNorm(RMSNorm(x))``.

With ``xhat = x * rstd``, ``n = xhat * w``, ``a = n @ W_down`` and ``y = n * sigmoid(silu(a) @ W_up)``,
the op is a rank-``R`` bottleneck around an RMSNorm and is bandwidth-bound: its cost is how many
times ``[T, D]`` crosses HBM. Autodiff of the two modules writes and re-reads ``n``, the gate, and
their cotangents. This ``custom_vjp`` computes the same function with three identities that keep
``n`` out of HBM:

* ``a = rstd * (x @ (w * W_down))``, because ``rstd`` is per row and ``w`` per column;
* ``dW_down = w * (x^T @ (rstd * da))`` for the same reason;
* the RMSNorm backward's row sum ``c = sum_d dn_d * n_d``, with ``dn = dy * sigma + da @ W_down^T``,
  is ``sum_d dy_d * sigma_d * n_d + sum_r da_r * a_r``. The first term comes out of the sweep that
  writes ``dg`` and the second is ``[T, R]``, so ``dx`` needs one more sweep and no other.

Every rank-``R`` GEMM runs on XLA with an activation-dtype output. The backward saves ``a``
``[T, R]`` and ``rstd`` ``[T]`` and reruns the gate GEMM. Its two ``[T, D]`` elementwise steps run on
the Triton kernels on GPU and on XLA elsewhere.
"""

import functools
import warnings
from typing import Literal, TypeAlias

import jax
import jax.numpy as jnp
from jax import shard_map
from jax.sharding import PartitionSpec as P
from jax.sharding import get_abstract_mesh, reshard
from jaxtyping import Array, Float

from levanter.sharding import partition_spec_of, partitioned_dims

from .triton_gpu import dg_triton, dx_triton, triton_gated_norm_available, triton_gated_norm_shapes_supported
from .xla import dg_xla, dx_xla

Implementation: TypeAlias = Literal["xla", "triton_gpu"]

_ELEMENTWISE = {"xla": (dg_xla, dx_xla), "triton_gpu": (dg_triton, dx_triton)}


def _matmul_f32(a: jax.Array, b: jax.Array, contract: tuple[int, int]) -> jax.Array:
    return jax.lax.dot_general(a, b, (((contract[0],), (contract[1],)), ((), ())), preferred_element_type=jnp.float32)


def _matmul(a: jax.Array, b: jax.Array, contract: tuple[int, int]) -> jax.Array:
    """A GEMM whose output keeps the operands' dtype, so a ``[T, D]`` result is one bf16 pass."""
    return jax.lax.dot_general(a, b, (((contract[0],), (contract[1],)), ((), ())))


def _gate_logits(a, w_up, dtype):
    s = jax.nn.silu(a).astype(dtype)
    return s, _matmul(s, w_up.astype(dtype), (1, 0))


@functools.partial(jax.custom_vjp, nondiff_argnums=(4, 5))
def _gated_rms_norm_2d(
    x: Float[Array, "T D"],
    rms_weight: Float[Array, " D"],
    w_down: Float[Array, "D R"],
    w_up: Float[Array, "R D"],
    eps: float,
    implementation: Implementation,
) -> Float[Array, "T D"]:
    return _gated_rms_norm_2d_fwd(x, rms_weight, w_down, w_up, eps, implementation)[0]


def _gated_rms_norm_2d_fwd(x, rms_weight, w_down, w_up, eps, implementation):
    del implementation
    x32 = x.astype(jnp.float32)
    rstd = jax.lax.rsqrt(jnp.mean(x32 * x32, axis=-1) + eps)
    folded_w_down = (rms_weight.astype(jnp.float32)[:, None] * w_down.astype(jnp.float32)).astype(x.dtype)
    a = _matmul_f32(x, folded_w_down, (1, 0)) * rstd[:, None]
    _, g = _gate_logits(a, w_up, x.dtype)
    n = x32 * rstd[:, None] * rms_weight.astype(jnp.float32)
    y = (n * jax.nn.sigmoid(g.astype(jnp.float32))).astype(x.dtype)
    return y, (x, rms_weight, w_down, w_up, a, rstd)


def _gated_rms_norm_2d_bwd(eps, implementation, residuals, dy):
    del eps
    x, rms_weight, w_down, w_up, a, rstd = residuals
    dg_step, dx_step = _ELEMENTWISE[implementation]
    s, g = _gate_logits(a, w_up, x.dtype)
    dg, dy_gate_n = dg_step(x, dy, g, rstd, rms_weight)
    ds = _matmul_f32(dg, w_up.astype(x.dtype), (1, 1))
    d_w_up = _matmul_f32(s, dg, (0, 0))
    sig_a = jax.nn.sigmoid(a)
    da = ds * (sig_a * (1.0 + a * (1.0 - sig_a)))
    c = dy_gate_n + jnp.sum(da * a, axis=-1)
    dn_bottleneck = _matmul(da.astype(x.dtype), w_down.astype(x.dtype), (1, 1))
    dx, d_rms_weight = dx_step(x, dy, g, dn_bottleneck, rstd, c, rms_weight)
    d_w_down = rms_weight.astype(jnp.float32)[:, None] * _matmul_f32(x, (rstd[:, None] * da).astype(x.dtype), (0, 0))
    return (
        dx.astype(x.dtype),
        d_rms_weight.astype(rms_weight.dtype),
        d_w_down.astype(w_down.dtype),
        d_w_up.astype(w_up.dtype),
    )


_gated_rms_norm_2d.defvjp(_gated_rms_norm_2d_fwd, _gated_rms_norm_2d_bwd)


def _resolve(implementation: Implementation | None, tokens: int, hidden: int) -> Implementation:
    if implementation == "xla":
        return "xla"
    if implementation not in (None, "triton_gpu"):
        raise ValueError(f"Unknown gated_rms_norm implementation {implementation!r}")
    available = triton_gated_norm_available()
    reason = triton_gated_norm_shapes_supported(tokens, hidden) if available else "Triton unavailable or not on a GPU"
    if reason is None:
        return "triton_gpu"
    if implementation == "triton_gpu":
        raise RuntimeError(f"gated_rms_norm implementation 'triton_gpu' is unusable: {reason}")
    if available:
        warnings.warn(f"gated_rms_norm falling back from 'triton_gpu' to 'xla' ({reason})", stacklevel=3)
    return "xla"


def _local(x, rms_weight, w_down, w_up, *, eps, implementation):
    hidden = x.shape[-1]
    x_2d = x.reshape(-1, hidden)
    resolved = _resolve(implementation, x_2d.shape[0], hidden)
    return _gated_rms_norm_2d(x_2d, rms_weight, w_down, w_up, eps, resolved).reshape(x.shape)


def gated_rms_norm(
    x: Float[Array, "... D"],
    rms_weight: Float[Array, " D"],
    w_down: Float[Array, "D R"],
    w_up: Float[Array, "R D"],
    *,
    eps: float,
    implementation: Implementation | None = None,
) -> Float[Array, "... D"]:
    """``GatedNorm(RMSNorm(x))``: ``n * sigmoid(silu(n @ w_down) @ w_up)`` with ``n = RMSNorm(x)``.

    The same function as ``gated_rms_norm_reference``, which rounds ``n`` to ``x``'s dtype before the
    bottleneck; this op keeps ``n`` in fp32, so it is closer to an fp32 evaluation.

    Under a mesh the op runs shard-local inside an explicit ``shard_map`` over ``x``'s own partition
    spec, so it adds no communication beyond summing the weight gradients. The hidden axis must be
    unsharded. The weights are replicated and their gradients are summed in fp32.

    Args:
      x: activations; every axis but the last is a token axis.
      rms_weight: the RMSNorm scale ``w``.
      w_down: the bottleneck down projection.
      w_up: the bottleneck up projection.
      eps: the RMSNorm epsilon.
      implementation: "triton_gpu" (raises where it cannot run) or "xla". ``None`` takes the Triton
        backward kernels on GPU and XLA elsewhere; on GPU it warns and takes XLA when the local token
        count is not a multiple of 64 or the hidden size not a multiple of 128.
    """
    # fp32 weights inside the shard_map make the cross-shard gradient sums fp32.
    weights = tuple(w.astype(jnp.float32) for w in (rms_weight, w_down, w_up))
    local = functools.partial(_local, eps=eps, implementation=implementation)

    mesh = get_abstract_mesh()
    if mesh is None or mesh.empty:
        return local(x, *weights)
    if partitioned_dims(x, mesh)[-1]:
        raise ValueError(f"gated_rms_norm requires an unsharded hidden axis; got {partition_spec_of(x)}.")
    spec = tuple(partition_spec_of(x) or ())
    x_spec = P(*spec, *(None,) * (x.ndim - len(spec)))
    weights = tuple(reshard(w, P(*(None,) * w.ndim)) for w in weights)
    return shard_map(
        local,
        mesh=mesh,
        in_specs=(x_spec, P(None), P(None, None), P(None, None)),
        out_specs=x_spec,
        check_vma=False,
    )(x, *weights)
