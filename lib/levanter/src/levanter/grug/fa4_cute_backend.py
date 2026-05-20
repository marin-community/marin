# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""JAX/CuTe backend boundary for Grug packed-segment attention.

The production attention kernel is intentionally isolated here so the high-level Grug
attention code stays independent of optional CUDA-only dependencies. The first kernel
target is BF16/FP16 BSHD causal self-attention with dynamic per-token lower bounds:

    valid[b, q] and lower_bounds[b, q] <= k <= q

This avoids both THD compaction and materialized [B, S, S] masks.
"""

import importlib
from dataclasses import dataclass
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp

from levanter.grug.fa4_cute_kernels import (
    segmented_flash_attention_backward_launcher,
    segmented_flash_attention_forward_launcher,
)


@dataclass(frozen=True)
class _CutlassCuteModules:
    cute: Any
    cjax: Any
    cuda: Any


def _import_cutlass_cute() -> _CutlassCuteModules:
    cute = importlib.import_module("cutlass.cute")
    cjax = importlib.import_module("cutlass.jax")
    cuda = importlib.import_module("cuda.bindings.driver")
    return _CutlassCuteModules(cute=cute, cjax=cjax, cuda=cuda)


def _optional_dependency_error() -> RuntimeError:
    return RuntimeError(
        "gpu_fa4_cute_attention requires nvidia-cutlass-dsl with JAX support, and backward requires "
        "flash-attn-4. Install the CUDA 13 CUTLASS DSL extra, for example "
        "`nvidia-cutlass-dsl[cu13]>=4.4`, plus `flash-attn-4`."
    )


def cutlass_cute_available() -> bool:
    """Return whether the optional CuTe/JAX CUTLASS modules are importable."""
    try:
        _import_cutlass_cute()
    except Exception:
        return False
    return True


def require_cutlass_cute() -> None:
    """Raise a clear error if nvidia-cutlass-dsl with JAX support is unavailable."""
    try:
        _import_cutlass_cute()
    except Exception as exc:
        raise _optional_dependency_error() from exc


def _vector_add_launcher(modules: _CutlassCuteModules) -> Any:
    cute = modules.cute
    cuda = modules.cuda

    @cute.kernel
    def _vector_add_kernel(a: cute.Tensor, b: cute.Tensor, out: cute.Tensor):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

        frag_a = cute.make_rmem_tensor(cute.size(a, mode=[0]), a.element_type)
        frag_b = cute.make_rmem_tensor(cute.size(b, mode=[0]), b.element_type)
        frag_out = cute.make_rmem_tensor(cute.size(out, mode=[0]), out.element_type)

        cute.autovec_copy(a[None, tidx, bidx], frag_a)
        cute.autovec_copy(b[None, tidx, bidx], frag_b)
        frag_out.store(frag_a.load() + frag_b.load())
        cute.autovec_copy(frag_out, out[None, tidx, bidx])

    @cute.jit
    def _launch_vector_add(stream: cuda.CUstream, a: cute.Tensor, b: cute.Tensor, out: cute.Tensor):
        _vector_add_kernel(a, b, out).launch(
            grid=[a.shape[-1], 1, 1],
            block=[a.shape[-2], 1, 1],
            stream=stream,
        )

    return _launch_vector_add


def cute_vector_add(a: jax.Array, b: jax.Array, *, block_size: int = 256) -> jax.Array:
    """Small CuTe/JAX launch proof used to validate the optional backend environment."""
    if a.shape != b.shape:
        raise ValueError(f"a and b must have matching shapes, got {a.shape} and {b.shape}")
    if a.dtype != b.dtype:
        raise ValueError(f"a and b must have matching dtypes, got {a.dtype} and {b.dtype}")
    if a.ndim != 1:
        raise ValueError(f"cute_vector_add expects 1D inputs, got shape={a.shape}")
    if a.dtype not in (jnp.bfloat16, jnp.float16):
        raise ValueError(f"cute_vector_add expects BF16/FP16 inputs, got {a.dtype}")
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    if a.shape[0] == 0:
        raise ValueError("cute_vector_add expects a non-empty input")

    try:
        modules = _import_cutlass_cute()
    except Exception as exc:
        raise _optional_dependency_error() from exc

    padded = ((a.shape[0] + block_size - 1) // block_size) * block_size
    a_padded = jnp.pad(a, (0, padded - a.shape[0])).reshape(1, block_size, padded // block_size)
    b_padded = jnp.pad(b, (0, padded - b.shape[0])).reshape(1, block_size, padded // block_size)
    call = modules.cjax.cutlass_call(
        _vector_add_launcher(modules),
        output_shape_dtype=jax.ShapeDtypeStruct(a_padded.shape, a_padded.dtype),
        use_static_tensors=True,
    )
    out = call(a_padded, b_padded)
    return out.reshape(-1)[: a.shape[0]]


def segmented_flash_attention_forward(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    lower_bounds: jax.Array,
    valid: jax.Array,
    *,
    softmax_scale: float,
    kernel_config: Any,
) -> tuple[jax.Array, jax.Array]:
    """FA4/CuTe segmented attention forward entry point.

    Args:
        q: Query tensor with shape [B, S, Hq, D].
        k: Key tensor with shape [B, S, Hkv, D].
        v: Value tensor with shape [B, S, Hkv, Dv].
        lower_bounds: Inclusive per-token key lower bound, shape [B, S].
        valid: Per-token query validity mask, shape [B, S].
        softmax_scale: QK softmax scale.
        kernel_config: Architecture-specific tile/config object selected by attention.py.

    Returns:
        ``(out, lse)`` where ``out`` has shape [B, S, Hq, Dv] and ``lse`` has
        shape [B, Hq, S]. The backward kernel consumes both tensors.
    """
    _validate_forward_inputs(q, k, v, lower_bounds, valid, softmax_scale=softmax_scale)
    try:
        modules = _import_cutlass_cute()
    except Exception as exc:
        raise _optional_dependency_error() from exc

    forward_tile = getattr(kernel_config, "forward_tile", (128, 64))
    num_threads = getattr(kernel_config, "num_threads", 128)
    launcher = segmented_flash_attention_forward_launcher(
        modules,
        head_dim=q.shape[-1],
        head_dim_v=v.shape[-1],
        qhead_per_kvhead=q.shape[2] // k.shape[2],
        tile_m=forward_tile[0],
        tile_n=forward_tile[1],
        num_threads=num_threads,
    )
    input_spec, output_spec = _cutlass_attention_forward_specs(modules, vector_elems=8)
    out_shape_dtype = jax.ShapeDtypeStruct((*q.shape[:3], v.shape[-1]), q.dtype)
    lse_shape_dtype = jax.ShapeDtypeStruct((q.shape[0], q.shape[2], q.shape[1]), jnp.float32)
    call = modules.cjax.cutlass_call(
        launcher,
        output_shape_dtype=(out_shape_dtype, lse_shape_dtype),
        input_spec=input_spec,
        output_spec=output_spec,
        use_static_tensors=True,
        softmax_scale=softmax_scale,
    )
    return call(q, k, v, lower_bounds, valid.astype(jnp.int32))


def segmented_flash_attention_backward(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    out: jax.Array,
    dout: jax.Array,
    lse: jax.Array,
    lower_bounds: jax.Array,
    valid: jax.Array,
    *,
    softmax_scale: float,
    kernel_config: Any,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """FA4/CuTe segmented attention backward boundary.

    This is intentionally a separate CUTLASS call from the forward path because
    upstream FA4 backward is a preprocess + main-kernel + postprocess pipeline.
    The current launcher raises until those kernels are ported, while tests can
    fake ``cutlass_call`` to lock down the JAX custom-VJP contract.
    """
    _validate_forward_inputs(q, k, v, lower_bounds, valid, softmax_scale=softmax_scale)
    _validate_backward_inputs(q, k, v, out, dout, lse)
    try:
        modules = _import_cutlass_cute()
    except Exception as exc:
        raise _optional_dependency_error() from exc

    backward_tile = getattr(kernel_config, "backward_tile", (128, 64))
    num_threads = getattr(kernel_config, "num_threads", 128)
    launcher = segmented_flash_attention_backward_launcher(
        modules,
        dtype=q.dtype,
        head_dim=q.shape[-1],
        head_dim_v=v.shape[-1],
        qhead_per_kvhead=q.shape[2] // k.shape[2],
        tile_m=backward_tile[0],
        tile_n=backward_tile[1],
        num_threads=num_threads,
    )
    input_spec, output_spec = _cutlass_attention_backward_specs(modules, vector_elems=8)
    output_shape_dtype = _cutlass_attention_backward_output_shapes(q, k, v, backward_tile)
    call = modules.cjax.cutlass_call(
        launcher,
        output_shape_dtype=output_shape_dtype,
        input_spec=input_spec,
        output_spec=output_spec,
        use_static_tensors=True,
        softmax_scale=softmax_scale,
    )
    dq, dk, dv, *_scratch = call(q, k, v, out, dout, lse, lower_bounds, valid.astype(jnp.int32))
    return dq, dk, dv


def _cutlass_attention_forward_specs(
    modules: _CutlassCuteModules, *, vector_elems: int
) -> tuple[tuple[Any, ...], Any]:
    tensor_spec = modules.cjax.TensorSpec
    qkv_spec = tensor_spec(mode=(1, 3, 2, 0), divisibility=(1, 1, 1, vector_elems), static=True)
    lse_spec = tensor_spec(divisibility=(1, 1, 1), static=True)
    metadata_spec = tensor_spec(static=True)
    return (qkv_spec, qkv_spec, qkv_spec, metadata_spec, metadata_spec), (qkv_spec, lse_spec)


def _cutlass_attention_backward_specs(
    modules: _CutlassCuteModules, *, vector_elems: int
) -> tuple[tuple[Any, ...], Any]:
    tensor_spec = modules.cjax.TensorSpec
    qkv_spec = tensor_spec(mode=(0, 1, 2, 3), divisibility=(1, 1, 1, vector_elems), static=True)
    lse_spec = tensor_spec(mode=(0, 1, 2), divisibility=(1, 1, 1), static=True)
    metadata_spec = tensor_spec(mode=(0, 1), static=True)
    scratch_spec = tensor_spec(mode=(0, 1, 2), static=True)
    input_spec = (
        qkv_spec,
        qkv_spec,
        qkv_spec,
        qkv_spec,
        qkv_spec,
        lse_spec,
        metadata_spec,
        metadata_spec,
    )
    return input_spec, (
        qkv_spec,
        qkv_spec,
        qkv_spec,
        scratch_spec,
        scratch_spec,
        scratch_spec,
        scratch_spec,
        scratch_spec,
    )


def _cutlass_attention_backward_output_shapes(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    backward_tile: tuple[int, int],
) -> tuple[jax.ShapeDtypeStruct, ...]:
    batch, seq_len, q_heads, head_dim = q.shape
    kv_heads = k.shape[2]
    tile_m, tile_n = backward_tile
    seq_q_rounded = ((seq_len + tile_m - 1) // tile_m) * tile_m
    seq_k_rounded = ((seq_len + tile_n - 1) // tile_n) * tile_n
    head_dim_rounded = ((head_dim + 31) // 32) * 32
    head_dim_v_rounded = ((v.shape[-1] + 31) // 32) * 32
    return (
        jax.ShapeDtypeStruct(q.shape, q.dtype),
        jax.ShapeDtypeStruct(k.shape, k.dtype),
        jax.ShapeDtypeStruct(v.shape, v.dtype),
        jax.ShapeDtypeStruct((batch, q_heads, seq_q_rounded), jnp.float32),
        jax.ShapeDtypeStruct((batch, q_heads, seq_q_rounded), jnp.float32),
        jax.ShapeDtypeStruct((batch, q_heads, seq_q_rounded * head_dim_rounded), jnp.float32),
        jax.ShapeDtypeStruct((batch, kv_heads, seq_k_rounded * head_dim_rounded), jnp.float32),
        jax.ShapeDtypeStruct((batch, kv_heads, seq_k_rounded * head_dim_v_rounded), jnp.float32),
    )


def fa4_cute_attention_forward(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    lower_bounds: jax.Array,
    valid: jax.Array,
    *,
    sm_scale: float | None = None,
    kernel_config: Any | None = None,
) -> jax.Array:
    """FA4/CuTe attention boundary with packed causal metadata.

    Forward uses the CUTLASS/CuTe JAX FFI path. Backward is routed through a custom VJP so JAX does not
    attempt to autodiff through ``cutlass_call``; it currently raises a targeted implementation error.
    """
    if sm_scale is None:
        sm_scale = float(q.shape[-1] ** -0.5)
    return _segmented_flash_attention_custom_vjp(
        q,
        k,
        v,
        lower_bounds,
        valid,
        sm_scale,
        kernel_config,
    )


@partial(jax.custom_vjp, nondiff_argnums=(5, 6))
def _segmented_flash_attention_custom_vjp(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    lower_bounds: jax.Array,
    valid: jax.Array,
    softmax_scale: float,
    kernel_config: Any,
) -> jax.Array:
    out, _ = segmented_flash_attention_forward(
        q,
        k,
        v,
        lower_bounds,
        valid,
        softmax_scale=softmax_scale,
        kernel_config=kernel_config,
    )
    return out


def _segmented_flash_attention_custom_vjp_fwd(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    lower_bounds: jax.Array,
    valid: jax.Array,
    softmax_scale: float,
    kernel_config: Any,
) -> tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]]:
    out, lse = segmented_flash_attention_forward(
        q,
        k,
        v,
        lower_bounds,
        valid,
        softmax_scale=softmax_scale,
        kernel_config=kernel_config,
    )
    return out, (q, k, v, out, lse, lower_bounds, valid)


def _segmented_flash_attention_custom_vjp_bwd(
    softmax_scale: float,
    kernel_config: Any,
    residuals: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
    cotangent: jax.Array | jax.custom_derivatives.SymbolicZero,
) -> tuple[jax.Array | None, jax.Array | None, jax.Array | None, None, None]:
    q, k, v, out, lse, lower_bounds, valid = residuals
    if isinstance(cotangent, jax.custom_derivatives.SymbolicZero):
        return jnp.zeros_like(q), jnp.zeros_like(k), jnp.zeros_like(v), None, None
    dq, dk, dv = segmented_flash_attention_backward(
        q,
        k,
        v,
        out,
        cotangent.astype(q.dtype),
        lse,
        lower_bounds,
        valid,
        softmax_scale=softmax_scale,
        kernel_config=kernel_config,
    )
    return dq, dk, dv, None, None


_segmented_flash_attention_custom_vjp.defvjp(
    _segmented_flash_attention_custom_vjp_fwd,
    _segmented_flash_attention_custom_vjp_bwd,
)


def _validate_forward_inputs(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    lower_bounds: jax.Array,
    valid: jax.Array,
    *,
    softmax_scale: float,
) -> None:
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError(f"q/k/v must be BSHD tensors, got q={q.shape}, k={k.shape}, v={v.shape}")
    if q.shape[0] != k.shape[0] or q.shape[0] != v.shape[0]:
        raise ValueError(f"q/k/v batch sizes must match, got q={q.shape}, k={k.shape}, v={v.shape}")
    if q.shape[1] != k.shape[1] or q.shape[1] != v.shape[1]:
        raise ValueError(f"q/k/v sequence lengths must match, got q={q.shape}, k={k.shape}, v={v.shape}")
    if q.shape[-1] != k.shape[-1]:
        raise ValueError(f"q/k head dimensions must match, got q={q.shape}, k={k.shape}")
    if k.shape[2] != v.shape[2]:
        raise ValueError(f"k/v head counts must match, got k={k.shape}, v={v.shape}")
    if v.shape[-1] != q.shape[-1]:
        raise NotImplementedError(f"gpu_fa4_cute_attention currently requires Dv == D, got q={q.shape}, v={v.shape}")
    if q.shape[2] % k.shape[2] != 0:
        raise ValueError(f"Hq must be divisible by Hkv for GQA, got q={q.shape}, k={k.shape}")
    if lower_bounds.shape != q.shape[:2]:
        raise ValueError(f"lower_bounds must have shape [B, S]={q.shape[:2]}, got {lower_bounds.shape}")
    if valid.shape != q.shape[:2]:
        raise ValueError(f"valid must have shape [B, S]={q.shape[:2]}, got {valid.shape}")
    if lower_bounds.dtype != jnp.int32:
        raise ValueError(f"lower_bounds must be int32, got {lower_bounds.dtype}")
    if valid.dtype != jnp.bool_:
        raise ValueError(f"valid must be bool, got {valid.dtype}")
    if q.dtype not in (jnp.bfloat16, jnp.float16):
        raise TypeError(f"gpu_fa4_cute_attention currently supports only bf16/fp16, got {q.dtype}")
    if k.dtype != q.dtype or v.dtype != q.dtype:
        raise TypeError(f"q/k/v dtypes must match, got q={q.dtype}, k={k.dtype}, v={v.dtype}")
    if not isinstance(softmax_scale, float):
        raise TypeError(f"softmax_scale must be a Python float, got {type(softmax_scale).__name__}")
    if softmax_scale <= 0.0:
        raise ValueError(f"softmax_scale must be positive, got {softmax_scale}")


def _validate_backward_inputs(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    out: jax.Array,
    dout: jax.Array,
    lse: jax.Array,
) -> None:
    expected_out_shape = (*q.shape[:3], v.shape[-1])
    if out.shape != expected_out_shape:
        raise ValueError(f"out must have shape {expected_out_shape}, got {out.shape}")
    if dout.shape != expected_out_shape:
        raise ValueError(f"dout must have shape {expected_out_shape}, got {dout.shape}")
    if out.dtype != q.dtype or dout.dtype != q.dtype:
        raise TypeError(f"out/dout dtypes must match q dtype {q.dtype}, got out={out.dtype}, dout={dout.dtype}")
    expected_lse_shape = (q.shape[0], q.shape[2], q.shape[1])
    if lse.shape != expected_lse_shape:
        raise ValueError(f"lse must have shape [B, Hq, S]={expected_lse_shape}, got {lse.shape}")
    if lse.dtype != jnp.float32:
        raise TypeError(f"lse must be float32, got {lse.dtype}")


__all__ = [
    "cutlass_cute_available",
    "cute_vector_add",
    "fa4_cute_attention_forward",
    "require_cutlass_cute",
    "segmented_flash_attention_backward",
    "segmented_flash_attention_forward",
]
