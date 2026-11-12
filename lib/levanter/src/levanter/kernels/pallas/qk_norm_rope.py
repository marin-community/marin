# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Fused q/k head normalization: a weightless RMS norm over ``head_dim``, half-RoPE and the q scale, in one pass.

The June attention normalizes each q and k head over its ``head_dim`` channels (float32 statistic, no weight), rotates
the first half of the channels (rotary pairs ``(i, i + D/4)`` for ``i < D/4``) and leaves the second half as it is,
then multiplies q by ``qk_mult``. Written as separate JAX ops, XLA runs this as several kernels per head tensor in each
direction (the statistic, the normalization and the rotation do not fuse on GPU), each reading the whole tensor again.
This module computes it in one kernel per direction:

- forward: ``y = rope(x * rsqrt(mean(x^2) + eps)) * scale...`` from ``x``;
- backward: ``dx`` from ``x`` and ``dy``, recomputing the statistic from ``x`` (only ``x`` is saved).

Both implementations round where the separate ops do, so their forward matches them up to the float32 rounding
of the statistic (one bfloat16 ulp in a few elements): the normalized ``x`` is rounded to ``x.dtype``, then the
rotated value, then the product with each scale in turn (a scale of 1 is skipped, as XLA drops it); the backward
rounds the scaled cotangent after each scale, the rotated-back cotangent and ``dx``.

``impl``:

- ``"reference"``: the separate ops, differentiated by JAX (the June model's own formulation).
- ``"pallas"``: Pallas-Triton kernels for both directions (GPU). On CPU they run in the Pallas interpreter, for
  tests; TPU is rejected.
"""

import functools
from typing import Literal

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu
from jaxtyping import Array, Float

from levanter.kernels.pallas.cost_estimate_utils import with_io_bytes_accessed

QkNormRopeImpl = Literal["reference", "pallas"]

# Rows (one head of one token each) per program and warps: the best of a sweep at [8, 4096, 20|5, 128] bf16 on MI355X
# (q and k, forward 0.101 ms and forward + backward 0.215 ms, against 0.385 and 0.655 ms for the separate XLA ops).
_BLOCK_ROWS = 32
_NUM_WARPS = 4
_MIN_QUARTER = 16


def rope_tables(
    positions: Float[Array, "Bt S"], quarter: int, theta: float
) -> tuple[Float[Array, "Bt S Q"], Float[Array, "Bt S Q"]]:
    """``cos`` and ``sin`` of ``positions * inv_freq`` for the ``quarter = head_dim // 4`` rotary frequencies."""
    inv_freq = 1.0 / (theta ** (jnp.arange(0, quarter, dtype=jnp.float32) / quarter))
    angles = positions.astype(jnp.float32)[..., None] * inv_freq
    return jnp.cos(angles), jnp.sin(angles)


def _check_shapes(x: jax.Array, cos: jax.Array, sin: jax.Array) -> None:
    if x.ndim != 4:
        raise ValueError(f"x must be [batch, seq, heads, head_dim], got shape={x.shape}")
    head_dim = x.shape[-1]
    if head_dim % 4:
        raise ValueError(f"head_dim must be a multiple of 4, got {head_dim}")
    want = (x.shape[1], head_dim // 4)
    for name, t in (("cos", cos), ("sin", sin)):
        if t.ndim != 3 or t.shape[1:] != want or t.shape[0] not in (1, x.shape[0]):
            raise ValueError(f"{name} must be [1 or batch, seq, head_dim // 4] = [., {want}], got {t.shape}")


def _round(x: jax.Array, dtype) -> jax.Array:
    return x.astype(dtype).astype(jnp.float32)


def _scale_chain(y: jax.Array, scales: tuple[float, ...], dtype) -> jax.Array:
    """``y * s`` for each scale in turn, each product rounded to ``dtype`` (scales of 1 skipped)."""
    for s in scales:
        if s != 1.0:
            y = _round(y * _round(jnp.float32(s), dtype), dtype)
    return y


# --------------------------------------------------------------------------------------------------------------------
# JAX formulations.


def _rotate_blocks(blocks: jax.Array, cos: jax.Array, sin: jax.Array, *, transpose: bool) -> jax.Array:
    """Half-RoPE on a ``[..., 2 (half), 2 (part), quarter]`` float32 view: ``x*c + flip(x)*s`` with ``c = 1`` and
    ``s = 0`` on the rope-free half (exactly ``x`` there). ``transpose`` applies the transposed rotation
    (``x*c + flip(x*s)``), the rotation's VJP. ``cos``/``sin`` are ``[Bt, S, Q]``, broadcast over heads."""
    cos = cos[:, :, None]
    sin = sin[:, :, None]
    one, zero = jnp.ones_like(cos), jnp.zeros_like(sin)
    c = jnp.stack([jnp.stack([cos, cos], axis=-2), jnp.stack([one, one], axis=-2)], axis=-3)
    s = jnp.stack([jnp.stack([-sin, sin], axis=-2), jnp.stack([zero, zero], axis=-2)], axis=-3)
    if transpose:
        return blocks * c + jnp.flip(blocks * s, axis=-2)
    return blocks * c + jnp.flip(blocks, axis=-2) * s


def _inverse_rms(x_f32: jax.Array, eps: float) -> jax.Array:
    return jax.lax.rsqrt(jnp.mean(jnp.square(x_f32), axis=-1, keepdims=True) + eps)


def _fwd_jax(x, cos, sin, *, eps: float, rope: bool, scales: tuple[float, ...]):
    dtype = x.dtype
    x_f32 = x.astype(jnp.float32)
    y = _round(x_f32 * _inverse_rms(x_f32, eps), dtype)
    if rope:
        quarter = x.shape[-1] // 4
        blocks = y.reshape(*x.shape[:-1], 2, 2, quarter)
        y = _round(_rotate_blocks(blocks, cos, sin, transpose=False).reshape(x.shape), dtype)
    return _scale_chain(y, scales, dtype).astype(dtype)


def _bwd_jax(x, cos, sin, g, *, eps: float, rope: bool, scales: tuple[float, ...]):
    dtype = x.dtype
    g = _scale_chain(g.astype(jnp.float32), tuple(reversed(scales)), dtype)
    if rope:
        quarter = x.shape[-1] // 4
        blocks = g.reshape(*x.shape[:-1], 2, 2, quarter)
        g = _round(_rotate_blocks(blocks, cos, sin, transpose=True).reshape(x.shape), dtype)
    x_f32 = x.astype(jnp.float32)
    inv_rms = _inverse_rms(x_f32, eps)
    projection = jnp.mean(g * x_f32, axis=-1, keepdims=True)
    return (inv_rms * (g - x_f32 * (inv_rms * inv_rms) * projection)).astype(dtype)


def qk_norm_rope_reference(
    x: Float[Array, "B S H D"],
    cos: Float[Array, "Bt S Q"],
    sin: Float[Array, "Bt S Q"],
    *,
    eps: float,
    rope: bool = True,
    scales: tuple[float, ...] = (),
) -> Float[Array, "B S H D"]:
    """The separate ops, as the June model writes them: ``rms_norm`` (float32 statistic, result in ``x.dtype``), the
    half-RoPE rotation (float32, result in ``x.dtype``), then ``* s`` for each scale in ``x.dtype``."""
    _check_shapes(x, cos, sin)
    dtype = x.dtype
    x_f32 = x.astype(jnp.float32)
    y = (x_f32 * _inverse_rms(x_f32, eps)).astype(dtype)
    if rope:
        blocks = y.reshape(*x.shape[:-1], 2, 2, x.shape[-1] // 4)
        y = _rotate_blocks(blocks.astype(jnp.float32), cos, sin, transpose=False).reshape(x.shape).astype(dtype)
    for s in scales:
        y = y * s
    return y


# --------------------------------------------------------------------------------------------------------------------
# Pallas-Triton kernels. ``x`` is viewed as [rows = B * S * H, D]; each program takes ``block_rows`` rows and loads
# their three channel ranges [0, Q), [Q, 2Q) (the rotary pairs) and [2Q, D) (rope-free) as separate tiles, so the
# rotation needs no shuffles. Row r is token r // H, position (r // H) % S, batch r // (H * S).


def _table_rows(rows: jax.Array, heads: int, seq: int, table_batch: int) -> jax.Array:
    token = rows // heads
    if table_batch == 1:
        return token % seq
    return token


def _load_rows(ref, row0, block_rows, col0, cols, in_bounds):
    return plgpu.load(ref.at[pl.ds(row0, block_rows), pl.ds(col0, cols)], mask=in_bounds[:, None], other=0).astype(
        jnp.float32
    )


def _store_rows(ref, row0, block_rows, col0, value, in_bounds):
    cols = value.shape[1]
    plgpu.store(ref.at[pl.ds(row0, block_rows), pl.ds(col0, cols)], value.astype(ref.dtype), mask=in_bounds[:, None])


def _load_tables(cos_ref, sin_ref, rows, in_bounds, *, heads, seq, table_batch, quarter):
    t = jnp.where(in_bounds, _table_rows(rows, heads, seq, table_batch), 0)
    q = jnp.arange(quarter)
    cos = plgpu.load(cos_ref.at[t[:, None], q[None, :]], mask=in_bounds[:, None], other=0)
    sin = plgpu.load(sin_ref.at[t[:, None], q[None, :]], mask=in_bounds[:, None], other=0)
    return cos, sin


def _fwd_kernel(x_ref, cos_ref, sin_ref, y_ref, *, block_rows, heads, seq, table_batch, eps, rope, scales):
    num_rows, head_dim = x_ref.shape
    quarter = head_dim // 4
    dtype = y_ref.dtype
    row0 = pl.program_id(0) * block_rows
    rows = row0 + jnp.arange(block_rows)
    in_bounds = rows < num_rows
    a = _load_rows(x_ref, row0, block_rows, 0, quarter, in_bounds)
    b = _load_rows(x_ref, row0, block_rows, quarter, quarter, in_bounds)
    c = _load_rows(x_ref, row0, block_rows, 2 * quarter, 2 * quarter, in_bounds)
    total = jnp.sum(a * a, axis=1) + jnp.sum(b * b, axis=1) + jnp.sum(c * c, axis=1)
    inv_rms = jax.lax.rsqrt(total / head_dim + eps)[:, None]
    a, b, c = (_round(v * inv_rms, dtype) for v in (a, b, c))
    if rope:
        cos, sin = _load_tables(
            cos_ref, sin_ref, rows, in_bounds, heads=heads, seq=seq, table_batch=table_batch, quarter=quarter
        )
        a, b = _round(a * cos + b * -sin, dtype), _round(b * cos + a * sin, dtype)
    a, b, c = (_scale_chain(v, scales, dtype) for v in (a, b, c))
    _store_rows(y_ref, row0, block_rows, 0, a, in_bounds)
    _store_rows(y_ref, row0, block_rows, quarter, b, in_bounds)
    _store_rows(y_ref, row0, block_rows, 2 * quarter, c, in_bounds)


def _bwd_kernel(x_ref, cos_ref, sin_ref, g_ref, dx_ref, *, block_rows, heads, seq, table_batch, eps, rope, scales):
    num_rows, head_dim = x_ref.shape
    quarter = head_dim // 4
    dtype = dx_ref.dtype
    row0 = pl.program_id(0) * block_rows
    rows = row0 + jnp.arange(block_rows)
    in_bounds = rows < num_rows
    ga = _load_rows(g_ref, row0, block_rows, 0, quarter, in_bounds)
    gb = _load_rows(g_ref, row0, block_rows, quarter, quarter, in_bounds)
    gc = _load_rows(g_ref, row0, block_rows, 2 * quarter, 2 * quarter, in_bounds)
    ga, gb, gc = (_scale_chain(v, tuple(reversed(scales)), dtype) for v in (ga, gb, gc))
    if rope:
        cos, sin = _load_tables(
            cos_ref, sin_ref, rows, in_bounds, heads=heads, seq=seq, table_batch=table_batch, quarter=quarter
        )
        ga, gb = _round(ga * cos + gb * sin, dtype), _round(gb * cos + ga * -sin, dtype)
    xa = _load_rows(x_ref, row0, block_rows, 0, quarter, in_bounds)
    xb = _load_rows(x_ref, row0, block_rows, quarter, quarter, in_bounds)
    xc = _load_rows(x_ref, row0, block_rows, 2 * quarter, 2 * quarter, in_bounds)
    total = jnp.sum(xa * xa, axis=1) + jnp.sum(xb * xb, axis=1) + jnp.sum(xc * xc, axis=1)
    inv_rms = jax.lax.rsqrt(total / head_dim + eps)[:, None]
    projection = ((jnp.sum(ga * xa, axis=1) + jnp.sum(gb * xb, axis=1) + jnp.sum(gc * xc, axis=1)) / head_dim)[:, None]
    factor = inv_rms * inv_rms * projection
    for col0, g, xv in ((0, ga, xa), (quarter, gb, xb), (2 * quarter, gc, xc)):
        _store_rows(dx_ref, row0, block_rows, col0, inv_rms * (g - xv * factor), in_bounds)


def _cost_estimate(body, operands, flat_inputs, out_shape) -> pl.CostEstimate:
    """FLOPs and transcendentals of the JAX formulation of the kernel's direction, bytes of its inputs and output."""
    specs = [jax.ShapeDtypeStruct(o.shape, o.dtype) for o in operands]
    return with_io_bytes_accessed(
        pl.estimate_cost(body, *specs), kernel_inputs_specs=flat_inputs, kernel_outputs_specs=out_shape
    )


def _pallas_call(kernel, x, cos, sin, *extra, eps, rope, scales, interpret, name, block_rows, num_warps):
    batch, seq, heads, head_dim = x.shape
    rows = batch * seq * heads
    table_batch = cos.shape[0]
    if interpret is None:
        interpret = jax.default_backend() != "gpu"
    jax_kwargs = dict(eps=eps, rope=rope, scales=scales)
    body = functools.partial(_fwd_jax if kernel is _fwd_kernel else _bwd_jax, **jax_kwargs)
    kernel = functools.partial(
        kernel,
        block_rows=block_rows,
        heads=heads,
        seq=seq,
        table_batch=table_batch,
        eps=eps,
        rope=rope,
        scales=scales,
    )
    operands = (x, cos, sin, *extra)
    flat = [x.reshape(rows, head_dim)]
    flat += [t.reshape(table_batch * seq, head_dim // 4).astype(jnp.float32) for t in (cos, sin)]
    flat += [e.reshape(rows, head_dim) for e in extra]
    out_shape = jax.ShapeDtypeStruct((rows, head_dim), x.dtype)
    out = pl.pallas_call(
        kernel,
        grid=(pl.cdiv(rows, block_rows),),
        in_specs=[pl.no_block_spec] * len(operands),
        out_specs=pl.no_block_spec,
        out_shape=out_shape,
        compiler_params=plgpu.CompilerParams(num_warps=num_warps, num_stages=1),
        cost_estimate=_cost_estimate(body, operands, flat, out_shape),
        interpret=interpret,
        name=name,
    )(*flat)
    return out.reshape(x.shape)


@functools.partial(jax.custom_vjp, nondiff_argnums=(3, 4, 5, 6, 7, 8))
def _qk_norm_rope_pallas(x, cos, sin, eps, rope, scales, interpret, block_rows, num_warps):
    return _pallas_call(
        _fwd_kernel,
        x,
        cos,
        sin,
        eps=eps,
        rope=rope,
        scales=scales,
        interpret=interpret,
        name="qk_norm_rope_fwd",
        block_rows=block_rows,
        num_warps=num_warps,
    )


def _pallas_fwd(x, cos, sin, eps, rope, scales, interpret, block_rows, num_warps):
    y = _qk_norm_rope_pallas(x, cos, sin, eps, rope, scales, interpret, block_rows, num_warps)
    return y, (x, cos, sin)


def _pallas_bwd(eps, rope, scales, interpret, block_rows, num_warps, residuals, g):
    x, cos, sin = residuals
    dx = _pallas_call(
        _bwd_kernel,
        x,
        cos,
        sin,
        g.astype(x.dtype),
        eps=eps,
        rope=rope,
        scales=scales,
        interpret=interpret,
        name="qk_norm_rope_bwd",
        block_rows=block_rows,
        num_warps=num_warps,
    )
    return dx, jnp.zeros_like(cos), jnp.zeros_like(sin)


_qk_norm_rope_pallas.defvjp(_pallas_fwd, _pallas_bwd)


def pallas_qk_norm_rope_supported(head_dim: int) -> bool:
    """Whether the kernels can tile a head: ``head_dim // 4`` must be a power of two of at least 16."""
    quarter = head_dim // 4
    return head_dim % 4 == 0 and quarter >= _MIN_QUARTER and quarter == 1 << (quarter.bit_length() - 1)


def qk_norm_rope(
    x: Float[Array, "B S H D"],
    cos: Float[Array, "Bt S Q"],
    sin: Float[Array, "Bt S Q"],
    *,
    eps: float,
    rope: bool = True,
    scales: tuple[float, ...] = (),
    impl: QkNormRopeImpl = "reference",
    interpret: bool | None = None,
    block_rows: int = _BLOCK_ROWS,
    num_warps: int = _NUM_WARPS,
) -> Float[Array, "B S H D"]:
    """``rope(rms_norm(x)) * scales[0] * scales[1] ...`` per head (see the module docstring).

    ``cos``/``sin`` (:func:`rope_tables`) are ``[1 or B, S, D // 4]`` float32 and get no gradient. The tensors are
    per device: under a mesh call this inside ``shard_map`` (the Pallas kernels have no partitioning rule).
    ``interpret=None`` runs the Pallas kernels in the interpreter on CPU. ``"pallas"`` on TPU raises ``ValueError``
    (the kernels are Pallas-Triton).
    """
    _check_shapes(x, cos, sin)
    cos = jax.lax.stop_gradient(cos.astype(jnp.float32))
    sin = jax.lax.stop_gradient(sin.astype(jnp.float32))
    scales = tuple(float(s) for s in scales)
    if impl == "reference":
        return qk_norm_rope_reference(x, cos, sin, eps=eps, rope=rope, scales=scales)
    if impl == "pallas":
        if jax.default_backend() == "tpu" and not interpret:
            raise ValueError(
                "the Pallas q/k norm kernels are Pallas-Triton (GPU) kernels; use impl='reference' on TPU"
            )
        if not pallas_qk_norm_rope_supported(x.shape[-1]):
            raise ValueError(
                f"the Pallas q/k norm needs head_dim // 4 a power of two >= {_MIN_QUARTER}, got {x.shape}"
            )
        return _qk_norm_rope_pallas(x, cos, sin, float(eps), rope, scales, interpret, block_rows, num_warps)
    raise ValueError(f"unknown q/k norm implementation {impl!r}")
