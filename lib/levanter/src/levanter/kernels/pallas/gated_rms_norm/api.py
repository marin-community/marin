# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Public API and backend selection for fused RMSNorm + GatedNorm."""

import functools
import warnings
from collections.abc import Sequence
from typing import Literal, TypeAlias

import jax
import jax.numpy as jnp
from jax import shard_map
from jax.sharding import PartitionSpec as P
from jax.sharding import get_abstract_mesh, reshard
from jaxtyping import Array, Float

from levanter.sharding import partition_spec_of, partitioning_axes

from .config import GatedRmsNormBlockSizes
from .pallas_gpu import (
    gated_rms_norm_pallas_fwd_local,
    gated_rms_norm_shapes_supported,
    pallas_gated_rms_norm_available,
)
from .reference import gated_rms_norm_reference

Implementation: TypeAlias = Literal["reference", "pallas_gpu"]


def _default_implementations() -> tuple[Implementation, ...]:
    if pallas_gated_rms_norm_available():
        return ("pallas_gpu", "reference")
    return ("reference",)


def _as_sequence(implementation: Implementation | Sequence[Implementation] | None) -> tuple[Implementation, ...]:
    if implementation is None:
        return _default_implementations()
    if isinstance(implementation, str):
        return (implementation,)  # explicit single choice: fail fast, never silently fall back
    return tuple(implementation)


def gated_rms_norm_bwd(
    x: Float[Array, "T D"],
    norm_weight: Float[Array, " D"],
    w_down: Float[Array, "D R"],
    w_up: Float[Array, "R D"],
    gate: Float[Array, "T D"],
    gate_hidden: Float[Array, "T R"],
    rstd: Float[Array, " T"],
    dout: Float[Array, "T D"],
) -> tuple[Float[Array, "T D"], Float[Array, " D"], Float[Array, "D R"], Float[Array, "R D"]]:
    """Backward of ``gated_rms_norm_reference`` from the forward's saved gate, projection and row scale.

    ``y`` is recomputed elementwise from ``x`` rather than saved. The gate and SiLU steps use
    the same JAX ops the reference's autodiff would emit, so they round at the same points;
    the RMSNorm step is written out with the saved ``rstd`` in f32.
    """
    dtype = x.dtype
    xf = x.astype(jnp.float32)
    weight = norm_weight.astype(jnp.float32)
    normed = xf * rstd[:, None]
    y = (normed * weight).astype(dtype)
    # out = y * gate
    dy_direct = dout * gate
    dgate = dout * y
    # gate = logistic(logits): JAX's rule is t * ans * (1 - ans).
    dlogits = dgate * (gate * (1 - gate))
    silu, silu_vjp = jax.vjp(jax.nn.silu, gate_hidden)
    dsilu = jnp.einsum("td,rd->tr", dlogits, w_up)
    dw_up = jnp.einsum("tr,td->rd", silu, dlogits)
    (dgate_hidden,) = silu_vjp(dsilu)
    dw_down = jnp.einsum("td,tr->dr", y, dgate_hidden)
    dy = (dy_direct + jnp.einsum("tr,dr->td", dgate_hidden, w_down)).astype(jnp.float32)
    # y = (x * rstd) * w, rstd = (mean(x^2) + eps)^-1/2
    dnormed = dy * weight
    drstd = jnp.sum(dnormed * xf, axis=-1, keepdims=True)
    rstd_col = rstd[:, None]
    dx = rstd_col * dnormed - xf * (rstd_col * rstd_col * rstd_col) * drstd * (1.0 / x.shape[-1])
    dnorm_weight = jnp.sum(dy * normed, axis=0)
    return dx.astype(dtype), dnorm_weight.astype(norm_weight.dtype), dw_down.astype(w_down.dtype), dw_up.astype(w_up.dtype)


# JAX must not differentiate through a pallas_call; the backward is written out above.
@functools.partial(jax.custom_vjp, nondiff_argnums=(4, 5))
def _gated_rms_norm_pallas_local(x, norm_weight, w_down, w_up, eps, block_sizes):
    out, _, _, _ = gated_rms_norm_pallas_fwd_local(x, norm_weight, w_down, w_up, eps=eps, block_sizes=block_sizes)
    return out


def _gated_rms_norm_pallas_local_fwd(x, norm_weight, w_down, w_up, eps, block_sizes):
    out, gate, gate_hidden, rstd = gated_rms_norm_pallas_fwd_local(
        x, norm_weight, w_down, w_up, eps=eps, block_sizes=block_sizes
    )
    return out, (x, norm_weight, w_down, w_up, gate, gate_hidden, rstd)


def _gated_rms_norm_pallas_local_bwd(eps, block_sizes, residuals, dout):
    return gated_rms_norm_bwd(*residuals, dout)


_gated_rms_norm_pallas_local.defvjp(_gated_rms_norm_pallas_local_fwd, _gated_rms_norm_pallas_local_bwd)


def _pallas_rows(x2d, norm_weight, w_down, w_up, *, eps, block_sizes):
    """Pad the token axis to the tile size, run the kernels, drop the padding."""
    tokens = x2d.shape[0]
    pad = -tokens % block_sizes.t_block_size
    if pad:
        x2d = jnp.pad(x2d, ((0, pad), (0, 0)))
    out = _gated_rms_norm_pallas_local(x2d, norm_weight, w_down, w_up, eps, block_sizes)
    return out[:tokens] if pad else out


def _local_call(x, norm_weight, w_down, w_up, *, eps, block_sizes):
    shape = x.shape
    out = _pallas_rows(x.reshape(-1, shape[-1]), norm_weight, w_down, w_up, eps=eps, block_sizes=block_sizes)
    return out.reshape(shape)


def _sharded(x, norm_weight, w_down, w_up, *, eps, block_sizes):
    """Run the kernels inside an explicit ``shard_map``; every token is independent."""
    mesh = get_abstract_mesh()
    call = functools.partial(_local_call, eps=eps, block_sizes=block_sizes)
    if mesh is None or mesh.empty:
        return call(x, norm_weight, w_down, w_up)
    spec = partition_spec_of(x)
    entries = tuple(spec) + (None,) * (x.ndim - len(spec)) if spec is not None else (None,) * x.ndim
    if partitioning_axes(entries[-1], mesh):
        raise ValueError(f"gated_rms_norm requires an unsharded hidden axis; got {spec}")
    x_spec = P(*entries[:-1], None)
    x = reshard(x, x_spec)
    weights = [reshard(w, P(*(None,) * w.ndim)) for w in (norm_weight, w_down, w_up)]

    @functools.partial(
        shard_map,
        mesh=mesh,
        in_specs=(x_spec, P(None), P(None, None), P(None, None)),
        out_specs=x_spec,
        check_vma=False,
    )
    def _local(x_local, norm_weight_local, w_down_local, w_up_local):
        return call(x_local, norm_weight_local, w_down_local, w_up_local)

    # pyrefly: ignore[bad-argument-count]  # jax.shard_map decorator erases _local's real signature
    return _local(x, *weights)


def gated_rms_norm(
    x: Float[Array, "... D"],
    norm_weight: Float[Array, " D"],
    w_down: Float[Array, "D R"],
    w_up: Float[Array, "R D"],
    *,
    eps: float,
    implementation: Implementation | Sequence[Implementation] | None = None,
    block_sizes: GatedRmsNormBlockSizes | None = None,
) -> Float[Array, "... D"]:
    """RMSNorm followed by a low-rank sigmoid gate: ``y * sigmoid(silu(y @ w_down) @ w_up)``.

    ``y = rms_norm(x) * norm_weight``. Leading axes are tokens and may be sharded over any mesh
    axes; the hidden axis must be whole. The Pallas path runs the forward as two kernels with a
    hand-written backward (see ``pallas_gpu``) and agrees with the reference to bf16 rounding.

    Args:
      x: activations; the kernels flatten the leading axes into tokens.
      norm_weight: RMSNorm scale, any float dtype.
      w_down: ``[D, R]`` gate down projection, in ``x.dtype``.
      w_up: ``[R, D]`` gate up projection, in ``x.dtype``.
      eps: RMSNorm epsilon.
      implementation: a single name (fail fast if unsupported) or an ordered sequence to try in
        turn. Defaults to the Pallas kernels on GPU, the reference elsewhere.
      block_sizes: GPU tile configuration.
    """
    if w_down.dtype != x.dtype or w_up.dtype != x.dtype:
        raise ValueError(
            f"gated_rms_norm requires the gate weights in x's dtype; got x={x.dtype}, "
            f"w_down={w_down.dtype}, w_up={w_up.dtype}. Cast before calling."
        )
    block_sizes = block_sizes or GatedRmsNormBlockSizes.get_default()
    requested = _as_sequence(implementation)
    explicit_single = isinstance(implementation, str)
    errors: list[str] = []
    for name in requested:
        if name == "reference":
            return gated_rms_norm_reference(x, norm_weight, w_down, w_up, eps=eps)
        if name != "pallas_gpu":
            raise ValueError(f"Unknown gated_rms_norm implementation {name!r}")
        if not pallas_gated_rms_norm_available():
            reason = "Pallas Triton backend unavailable or not running on a GPU"
        else:
            reason = gated_rms_norm_shapes_supported((1, x.shape[-1]), w_down.shape, block_sizes)
        if reason is not None:
            if explicit_single:
                raise RuntimeError(f"gated_rms_norm implementation 'pallas_gpu' is unusable: {reason}")
            errors.append(f"pallas_gpu: {reason}")
            warnings.warn(f"gated_rms_norm falling back from 'pallas_gpu' ({reason})", stacklevel=2)
            continue
        return _sharded(x, norm_weight, w_down, w_up, eps=eps, block_sizes=block_sizes)
    raise RuntimeError("No usable gated_rms_norm implementation: " + "; ".join(errors))
