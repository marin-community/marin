# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Upstream FlashAttention-4 THD/varlen wrapper for Grug attention."""

import importlib
import math
from dataclasses import dataclass
from functools import partial
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from jax.sharding import reshard
from jaxtyping import Array, Bool, Float, Int

from levanter.grug.attention._core import AttentionMask
from levanter.grug.attention._fa4_cute import _gpu_compute_arch
from levanter.grug.attention._fa4_cute_config import Flash4CuteKernelConfig, flash4_cute_kernel_config


@dataclass(frozen=True)
class _UpstreamFa4CuteModules:
    arch: int
    cutlass: Any
    cute: Any
    cjax: Any
    cuda: Any
    FlashAttentionForward: Any
    FlashAttentionBackward: Any
    FlashAttentionBackwardPreprocess: Any
    FlashAttentionBackwardPostprocess: Any


_SM90_BACKWARD_TILE = (64, 128)
_SM90_BACKWARD_NUM_THREADS = 384
_SM90_BACKWARD_PDS_STAGE = 1
_SM90_BACKWARD_SDP_SWAP_AB = True
_SM90_BACKWARD_ATOM_LAYOUT_N_DKV = 2
_HOPPER_ARCH_FAMILY = 9
_BLACKWELL_ARCH_FAMILY = 10
_BLACKWELL_NEXT_ARCH_FAMILY = 11
_SUPPORTED_ARCH_FAMILIES = (_HOPPER_ARCH_FAMILY, _BLACKWELL_ARCH_FAMILY, _BLACKWELL_NEXT_ARCH_FAMILY)


def _sm90_backward_kernel_options() -> dict[str, int | bool]:
    return {
        "PdS_stage": _SM90_BACKWARD_PDS_STAGE,
        "SdP_swapAB": _SM90_BACKWARD_SDP_SWAP_AB,
        "AtomLayoutNdKV": _SM90_BACKWARD_ATOM_LAYOUT_N_DKV,
    }


def _import_upstream_fa4_cute() -> _UpstreamFa4CuteModules:
    try:
        arch = _gpu_compute_arch()
        arch_family = arch // 10
        cutlass = importlib.import_module("cutlass")
        cute = importlib.import_module("cutlass.cute")
        cjax = importlib.import_module("cutlass.jax")
        cuda = importlib.import_module("cuda.bindings.driver")
        if arch_family == _HOPPER_ARCH_FAMILY:
            flash_fwd = importlib.import_module("flash_attn.cute.flash_fwd_sm90")
            flash_bwd = importlib.import_module("flash_attn.cute.flash_bwd_sm90")
            flash_fwd_cls = flash_fwd.FlashAttentionForwardSm90
            flash_bwd_cls = flash_bwd.FlashAttentionBackwardSm90
        elif arch_family in (_BLACKWELL_ARCH_FAMILY, _BLACKWELL_NEXT_ARCH_FAMILY):
            flash_fwd = importlib.import_module("flash_attn.cute.flash_fwd_sm100")
            flash_bwd = importlib.import_module("flash_attn.cute.flash_bwd_sm100")
            flash_fwd_cls = flash_fwd.FlashAttentionForwardSm100
            flash_bwd_cls = flash_bwd.FlashAttentionBackwardSm100
        else:
            raise NotImplementedError(f"gpu_fa4_thd_attention supports SM90/SM100/SM110, got SM{arch}.")
        flash_bwd_preprocess = importlib.import_module("flash_attn.cute.flash_bwd_preprocess")
        flash_bwd_postprocess = importlib.import_module("flash_attn.cute.flash_bwd_postprocess")
    except NotImplementedError:
        raise
    except Exception as exc:
        raise RuntimeError(
            "gpu_fa4_thd_attention requires upstream flash-attn-4 CuTe internals and "
            "nvidia-cutlass-dsl with JAX support. Install flash-attn-4 and "
            "`nvidia-cutlass-dsl[cu13]>=4.4` in the GPU environment."
        ) from exc

    return _UpstreamFa4CuteModules(
        arch=arch,
        cutlass=cutlass,
        cute=cute,
        cjax=cjax,
        cuda=cuda,
        FlashAttentionForward=flash_fwd_cls,
        FlashAttentionBackward=flash_bwd_cls,
        FlashAttentionBackwardPreprocess=flash_bwd_preprocess.FlashAttentionBackwardPreprocess,
        FlashAttentionBackwardPostprocess=flash_bwd_postprocess.FlashAttentionBackwardPostprocess,
    )


def _optional_dependency_error() -> RuntimeError:
    return RuntimeError(
        "gpu_fa4_thd_attention requires upstream flash-attn-4 CuTe internals and "
        "nvidia-cutlass-dsl with JAX support in the GPU environment."
    )


def _cutlass_dtype(cutlass: Any, dtype: Any) -> Any:
    if str(dtype) == "bfloat16":
        return cutlass.BFloat16
    if str(dtype) == "float16":
        return cutlass.Float16
    raise TypeError(f"gpu_fa4_thd_attention expects bf16/fp16, got {dtype}")


def _validate_simple_causal_self_attention(
    q: Float[Array, "B Q Hq D"],
    k: Float[Array, "B K Hkv D"],
    v: Float[Array, "B K Hkv D"],
    mask: AttentionMask | Bool[Array, "B Q K"] | Float[Array, "B Q K"] | None,
    *,
    backend_name: str,
) -> None:
    if isinstance(mask, jax.Array):
        raise NotImplementedError(f"{backend_name} does not support dense masks.")
    if not isinstance(mask, AttentionMask):
        raise NotImplementedError(f"{backend_name} requires an AttentionMask with packed segment_ids.")
    if not mask.is_causal:
        raise NotImplementedError(f"{backend_name} supports only causal self-attention.")
    if mask.sliding_window is not None:
        if mask.sliding_window <= 0:
            raise ValueError(f"sliding_window must be positive, got {mask.sliding_window}")

    if len(q.shape) != 4 or len(k.shape) != 4 or len(v.shape) != 4:
        raise ValueError(
            f"{backend_name} expects q/k/v with shape [B,S,H,D], got q={q.shape}, k={k.shape}, v={v.shape}"
        )
    if q.shape[0] != k.shape[0] or q.shape[0] != v.shape[0]:
        raise ValueError(f"{backend_name} requires matching batch sizes, got q={q.shape}, k={k.shape}, v={v.shape}")
    if q.shape[1] != k.shape[1] or q.shape[1] != v.shape[1]:
        raise NotImplementedError(f"{backend_name} supports only self-attention with q_len == kv_len.")
    if q.shape[-1] != k.shape[-1] or q.shape[-1] != v.shape[-1]:
        raise ValueError(f"{backend_name} requires Dq == Dk == Dv, got q={q.shape}, k={k.shape}, v={v.shape}")
    if q.shape[2] % k.shape[2] != 0:
        raise ValueError(f"{backend_name} requires Hq divisible by Hkv, got q={q.shape}, k={k.shape}")
    if q.shape[2] == k.shape[2]:
        raise NotImplementedError(
            f"{backend_name} currently supports only GQA with Hq > Hkv, got q={q.shape}, k={k.shape}"
        )
    if k.shape[2] != v.shape[2]:
        raise ValueError(f"{backend_name} requires matching K/V heads, got k={k.shape}, v={v.shape}")

    if mask.segment_ids is None and mask.thd_segment_metadata is None:
        raise NotImplementedError(f"{backend_name} requires packed segment_ids or precomputed THD segment metadata.")


def _thd_kernel_config(head_dim: int) -> Flash4CuteKernelConfig:
    arch = _gpu_compute_arch()
    arch_family = arch // 10
    if arch_family not in _SUPPORTED_ARCH_FAMILIES:
        raise NotImplementedError(f"gpu_fa4_thd_attention currently supports only SM90/SM100/SM110, got SM{arch}.")
    if head_dim != 128:
        raise NotImplementedError(f"gpu_fa4_thd_attention is only wired for head_dim=128, got {head_dim}.")
    if arch_family == _HOPPER_ARCH_FAMILY:
        base = flash4_cute_kernel_config(head_dim, arch=arch)
        return Flash4CuteKernelConfig(
            forward_tile=base.forward_tile,
            backward_tile=_SM90_BACKWARD_TILE,
            num_threads=_SM90_BACKWARD_NUM_THREADS,
        )
    return Flash4CuteKernelConfig(forward_tile=(128, 128), backward_tile=(128, 128), num_threads=384)


def _thd_cu_seqlens_from_segment_lengths(
    segment_lengths: Int[Array, "... M"],
    num_segments: Int[Array, "..."],
    *,
    batch_size: int,
    total_tokens: int,
    token_reference: Float[Array, "B S H D"] | None,
) -> Int[Array, "N"]:
    if segment_lengths.ndim == 1:
        segment_lengths = jnp.broadcast_to(segment_lengths[None, :], (batch_size, segment_lengths.shape[0]))
    elif segment_lengths.ndim == 2:
        if segment_lengths.shape[0] not in (1, batch_size):
            raise ValueError(
                f"segment_lengths must have one row or one row per batch item, got {segment_lengths.shape}"
            )
        if segment_lengths.shape[0] == 1 and batch_size != 1:
            segment_lengths = jnp.broadcast_to(segment_lengths, (batch_size, segment_lengths.shape[1]))
    else:
        raise ValueError(f"segment_lengths must have shape [M] or [B,M], got {segment_lengths.shape}")

    num_segments = jnp.reshape(num_segments, (-1,))
    if num_segments.shape[0] == 1 and batch_size != 1:
        num_segments = jnp.broadcast_to(num_segments, (batch_size,))
    elif num_segments.shape[0] != batch_size:
        raise ValueError(f"num_segments must have shape [B] or scalar, got {num_segments.shape}")

    max_segments = segment_lengths.shape[1]
    if max_segments == 1:
        if total_tokens % batch_size != 0:
            raise ValueError(f"total_tokens={total_tokens} is not divisible by batch_size={batch_size}.")
        tokens_per_row = total_tokens // batch_size
        return jnp.arange(batch_size + 1, dtype=jnp.int32) * tokens_per_row
    else:
        segment_index = jnp.arange(max_segments, dtype=jnp.int32)
        keep = segment_index[None, :] < num_segments[:, None]
        sharding = _segment_lengths_sharding(segment_lengths, num_segments, token_reference)
        if sharding is not None:
            keep = reshard(keep, sharding)
        lengths = jnp.where(keep, segment_lengths.astype(jnp.int32), jnp.zeros_like(segment_lengths, dtype=jnp.int32))
    lengths = eqx.error_if(
        lengths,
        jnp.any((lengths <= 0) & keep),
        "THD segment metadata contains a non-positive active segment length.",
    )
    lengths = _replicate_for_global_prefix_sum(lengths)
    cu_seqlens = jnp.concatenate(
        [
            jnp.zeros((1,), dtype=jnp.int32),
            jnp.cumsum(jnp.reshape(lengths, (-1,)), dtype=jnp.int32),
        ],
        axis=0,
    )
    cu_seqlens = eqx.error_if(
        cu_seqlens,
        cu_seqlens[-1] != total_tokens,
        "THD segment metadata does not cover the q/k/v token count.",
    )
    return cu_seqlens


def _replicate_for_global_prefix_sum(x: Int[Array, "..."]) -> Int[Array, "..."]:
    sharding = _sharding_of(x)
    if isinstance(sharding, NamedSharding):
        return reshard(x, NamedSharding(sharding.mesh, P(*([None] * x.ndim))))
    return x


def _segment_lengths_sharding(
    segment_lengths: Int[Array, "... M"],
    num_segments: Int[Array, "..."],
    token_reference: Float[Array, "B S H D"] | None,
) -> jax.sharding.Sharding | None:
    token_sharding = _sharding_of(token_reference)
    if isinstance(token_sharding, NamedSharding) and len(token_sharding.spec) >= 1:
        return NamedSharding(token_sharding.mesh, P(token_sharding.spec[0], None))
    sharding = _sharding_of(segment_lengths)
    if sharding is not None:
        return sharding
    num_segments_sharding = _sharding_of(num_segments)
    if isinstance(num_segments_sharding, NamedSharding) and len(num_segments_sharding.spec) == 1:
        return NamedSharding(num_segments_sharding.mesh, P(num_segments_sharding.spec[0], None))
    return None


def _sharding_of(x: Array | None) -> jax.sharding.Sharding | None:
    if x is None:
        return None
    sharding = getattr(x, "sharding", None)
    if isinstance(sharding, NamedSharding) and not getattr(sharding.mesh, "empty", False):
        return sharding
    aval = getattr(x, "aval", None)
    sharding = getattr(aval, "sharding", None) if aval is not None else None
    if isinstance(sharding, NamedSharding) and not getattr(sharding.mesh, "empty", False):
        return sharding
    return None


def _upstream_fa4_thd_forward_launcher(
    modules: _UpstreamFa4CuteModules,
    *,
    dtype: Any,
    head_dim: int,
    head_dim_v: int,
    qhead_per_kvhead: int,
    kernel_config: Flash4CuteKernelConfig,
    sliding_window: int | None,
) -> Any:
    cutlass = modules.cutlass
    cute = modules.cute
    cuda = modules.cuda
    cute_dtype = _cutlass_dtype(cutlass, dtype)
    if modules.arch // 10 == _HOPPER_ARCH_FAMILY:
        if sliding_window is not None:
            raise NotImplementedError("gpu_fa4_thd_attention does not support sliding-window attention on SM90.")
        flash_fwd = modules.FlashAttentionForward(
            cute_dtype,
            head_dim,
            head_dim_v,
            qhead_per_kvhead=qhead_per_kvhead,
            is_causal=True,
            is_local=False,
            pack_gqa=qhead_per_kvhead > 1,
            tile_m=kernel_config.forward_tile[0],
            tile_n=kernel_config.forward_tile[1],
            num_stages=2,
            num_threads=kernel_config.num_threads,
            score_mod=None,
            mask_mod=None,
            has_aux_tensors=False,
            q_subtile_factor=None,
            paged_kv_non_tma=False,
        )
    else:
        flash_fwd = modules.FlashAttentionForward(
            head_dim,
            head_dim_v,
            qhead_per_kvhead=qhead_per_kvhead,
            is_causal=sliding_window is None,
            is_local=sliding_window is not None,
            is_split_kv=False,
            pack_gqa=qhead_per_kvhead > 1,
            m_block_size=kernel_config.forward_tile[0],
            n_block_size=kernel_config.forward_tile[1],
            q_stage=2,
            is_persistent=False,
            score_mod=None,
            mask_mod=None,
            has_aux_tensors=False,
            paged_kv_non_tma=False,
            is_varlen_q=True,
            q_subtile_factor=None,
            use_2cta_instrs=False,
            use_clc_scheduler=False,
        )

    @cute.jit
    def _launch_upstream_fa4_thd_forward(
        stream: cuda.CUstream,
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        cu_seqlens: cute.Tensor,
        out: cute.Tensor,
        lse: cute.Tensor,
        *,
        softmax_scale: cutlass.Float32,
    ):
        if sliding_window is None:
            flash_fwd(
                q,
                k,
                v,
                out,
                lse,
                softmax_scale,
                cu_seqlens,
                cu_seqlens,
                stream=stream,
            )
        else:
            flash_fwd(
                q,
                k,
                v,
                out,
                lse,
                softmax_scale,
                cu_seqlens,
                cu_seqlens,
                None,
                None,
                None,
                sliding_window - 1,
                0,
                None,
                None,
                None,
                None,
                stream,
            )

    return _launch_upstream_fa4_thd_forward


def _upstream_fa4_thd_backward_launcher(
    modules: _UpstreamFa4CuteModules,
    *,
    dtype: Any,
    head_dim: int,
    head_dim_v: int,
    qhead_per_kvhead: int,
    kernel_config: Flash4CuteKernelConfig,
    sliding_window: int | None,
) -> Any:
    cutlass = modules.cutlass
    cute = modules.cute
    cuda = modules.cuda
    cute_dtype = _cutlass_dtype(cutlass, dtype)

    tile_m, tile_n = kernel_config.backward_tile
    preprocess = modules.FlashAttentionBackwardPreprocess(
        cute_dtype,
        head_dim,
        head_dim_v,
        tile_m,
        num_threads=128,
        use_padded_offsets=False,
    )
    if modules.arch // 10 == _HOPPER_ARCH_FAMILY:
        if sliding_window is not None:
            raise NotImplementedError("gpu_fa4_thd_attention does not support sliding-window attention on SM90.")
        backward = modules.FlashAttentionBackward(
            cute_dtype,
            head_dim,
            head_dim_v,
            qhead_per_kvhead=qhead_per_kvhead,
            is_causal=True,
            is_local=False,
            deterministic=False,
            tile_m=tile_m,
            tile_n=tile_n,
            **_sm90_backward_kernel_options(),
            num_threads=kernel_config.num_threads,
            score_mod=None,
            score_mod_bwd=None,
            mask_mod=None,
            has_aux_tensors=False,
            subtile_factor=1,
        )
        cluster_size = 1
        use_2cta_instrs = False
    else:
        backward = modules.FlashAttentionBackward(
            head_dim,
            head_dim_v,
            is_causal=sliding_window is None,
            is_local=sliding_window is not None,
            qhead_per_kvhead=qhead_per_kvhead,
            tile_m=tile_m,
            tile_n=tile_n,
            is_persistent=False,
            deterministic=False,
            spt=False,
            cluster_size=2,
            use_2cta_instrs=True,
            score_mod=None,
            score_mod_bwd=None,
            mask_mod=None,
            has_aux_tensors=False,
            subtile_factor=1,
        )
        cluster_size = 2
        use_2cta_instrs = True
    dq_postprocess_tile_m = _sm90_postprocess_tile_m(modules.arch, tile_m)
    dkv_postprocess_tile_m = _sm90_postprocess_tile_m(modules.arch, tile_n)
    dq_postprocess = modules.FlashAttentionBackwardPostprocess(
        cute_dtype,
        head_dim,
        modules.arch,
        dq_postprocess_tile_m,
        num_threads=128,
        AtomLayoutMdQ=1,
        use_2cta_instrs=use_2cta_instrs,
    )
    dk_postprocess = modules.FlashAttentionBackwardPostprocess(
        cute_dtype,
        head_dim,
        modules.arch,
        dkv_postprocess_tile_m,
        num_threads=128,
        AtomLayoutMdQ=1,
        cluster_size=cluster_size,
    )
    dv_postprocess = modules.FlashAttentionBackwardPostprocess(
        cute_dtype,
        head_dim_v,
        modules.arch,
        dkv_postprocess_tile_m,
        num_threads=128,
        AtomLayoutMdQ=1,
        cluster_size=cluster_size,
    )

    class _Float32ZeroFill:
        def __init__(self, num_threads: int):
            self._num_threads = num_threads

        @cute.jit
        def __call__(self, tensor: cute.Tensor, stream: cuda.CUstream):
            self.kernel(tensor).launch(
                grid=[cute.ceil_div(cute.size(tensor), self._num_threads), 1, 1],
                block=[self._num_threads, 1, 1],
                stream=stream,
            )

        @cute.kernel
        def kernel(self, tensor: cute.Tensor):
            tidx, _, _ = cute.arch.thread_idx()
            bidx, _, _ = cute.arch.block_idx()
            flat = cute.make_tensor(tensor.iterator, cute.make_layout(cute.size(tensor)))
            idx = bidx * self._num_threads + tidx
            if idx < cute.size(flat):
                flat[idx] = cutlass.Float32(0.0)

    zero_fill = _Float32ZeroFill(128)

    @cute.jit
    def _launch_upstream_fa4_thd_backward(
        stream: cuda.CUstream,
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        out: cute.Tensor,
        dout: cute.Tensor,
        lse: cute.Tensor,
        cu_seqlens: cute.Tensor,
        dq: cute.Tensor,
        dk: cute.Tensor,
        dv: cute.Tensor,
        dpsum: cute.Tensor,
        lse_log2: cute.Tensor,
        dq_accum: cute.Tensor,
        dk_accum: cute.Tensor,
        dv_accum: cute.Tensor,
        *,
        softmax_scale: cutlass.Float32,
    ):
        preprocess(out, dout, dpsum, lse, lse_log2, dq_accum, cu_seqlens, None, None, stream)
        zero_fill(dk_accum, stream)
        zero_fill(dv_accum, stream)
        if sliding_window is None:
            backward(
                q,
                k,
                v,
                dout,
                lse_log2,
                dpsum,
                dq_accum,
                dk_accum,
                dv_accum,
                softmax_scale,
                cu_seqlens,
                cu_seqlens,
                stream=stream,
            )
        else:
            backward(
                q,
                k,
                v,
                dout,
                lse_log2,
                dpsum,
                dq_accum,
                dk_accum,
                dv_accum,
                softmax_scale,
                cu_seqlens,
                cu_seqlens,
                None,
                None,
                sliding_window - 1,
                0,
                None,
                None,
                None,
                None,
                None,
                stream,
            )
        dq_postprocess(dq_accum, dq, softmax_scale, cu_seqlens, None, stream)
        dk_postprocess(dk_accum, dk, softmax_scale, cu_seqlens, None, stream)
        dv_postprocess(dv_accum, dv, cutlass.Float32(1.0), cu_seqlens, None, stream)

    return _launch_upstream_fa4_thd_backward


def _sm90_postprocess_tile_m(arch: int, tile_m: int) -> int:
    if arch // 10 == _HOPPER_ARCH_FAMILY:
        return 64
    return tile_m


def fa4_thd_attention_forward(
    q: Float[Array, "T Hq D"],
    k: Float[Array, "T Hkv D"],
    v: Float[Array, "T Hkv D"],
    cu_seqlens: Int[Array, "N"],
    *,
    softmax_scale: float,
    kernel_config: Flash4CuteKernelConfig,
    sliding_window: int | None,
) -> tuple[Float[Array, "T Hq D"], Float[Array, "Hq T"]]:
    _validate_thd_inputs(q, k, v, cu_seqlens, softmax_scale=softmax_scale)
    try:
        modules = _import_upstream_fa4_cute()
    except Exception as exc:
        raise _optional_dependency_error() from exc

    launcher = _upstream_fa4_thd_forward_launcher(
        modules,
        dtype=q.dtype,
        head_dim=q.shape[-1],
        head_dim_v=v.shape[-1],
        qhead_per_kvhead=q.shape[1] // k.shape[1],
        kernel_config=kernel_config,
        sliding_window=sliding_window,
    )
    input_spec, output_spec = _cutlass_thd_forward_specs(modules)
    out_shape_dtype = jax.ShapeDtypeStruct(q.shape, q.dtype)
    lse_shape_dtype = jax.ShapeDtypeStruct((q.shape[1], q.shape[0]), jnp.float32)
    call = modules.cjax.cutlass_call(
        launcher,
        output_shape_dtype=(out_shape_dtype, lse_shape_dtype),
        input_spec=input_spec,
        output_spec=output_spec,
        use_static_tensors=True,
        softmax_scale=softmax_scale,
    )
    return call(q, k, v, cu_seqlens)


def fa4_thd_attention_backward(
    q: Float[Array, "T Hq D"],
    k: Float[Array, "T Hkv D"],
    v: Float[Array, "T Hkv D"],
    out: Float[Array, "T Hq D"],
    dout: Float[Array, "T Hq D"],
    lse: Float[Array, "Hq T"],
    cu_seqlens: Int[Array, "N"],
    *,
    softmax_scale: float,
    kernel_config: Flash4CuteKernelConfig,
    sliding_window: int | None,
) -> tuple[Float[Array, "T Hq D"], Float[Array, "T Hkv D"], Float[Array, "T Hkv D"]]:
    _validate_thd_inputs(q, k, v, cu_seqlens, softmax_scale=softmax_scale)
    if q.shape[1] == k.shape[1]:
        raise NotImplementedError("gpu_fa4_thd_attention backward is currently wired for GQA only.")
    try:
        modules = _import_upstream_fa4_cute()
    except Exception as exc:
        raise _optional_dependency_error() from exc

    launcher = _upstream_fa4_thd_backward_launcher(
        modules,
        dtype=q.dtype,
        head_dim=q.shape[-1],
        head_dim_v=v.shape[-1],
        qhead_per_kvhead=q.shape[1] // k.shape[1],
        kernel_config=kernel_config,
        sliding_window=sliding_window,
    )
    input_spec, output_spec = _cutlass_thd_backward_specs(modules)
    output_shape_dtype = _cutlass_thd_backward_output_shapes(q, k, v, cu_seqlens, kernel_config.backward_tile)
    call = modules.cjax.cutlass_call(
        launcher,
        output_shape_dtype=output_shape_dtype,
        input_spec=input_spec,
        output_spec=output_spec,
        use_static_tensors=True,
        softmax_scale=softmax_scale,
    )
    dq, dk, dv, *_scratch = call(q, k, v, out, dout, lse, cu_seqlens)
    return dq, dk, dv


def _cutlass_thd_forward_specs(modules: _UpstreamFa4CuteModules) -> tuple[tuple[Any, ...], Any]:
    tensor_spec = modules.cjax.TensorSpec
    qkv_spec = tensor_spec(mode=(0, 1, 2), divisibility=(1, 1, 8), static=True)
    cu_spec = tensor_spec(mode=(0,), divisibility=(1,), static=True)
    lse_spec = tensor_spec(mode=(0, 1), divisibility=(1, 1), static=True)
    return (qkv_spec, qkv_spec, qkv_spec, cu_spec), (qkv_spec, lse_spec)


def _cutlass_thd_backward_specs(modules: _UpstreamFa4CuteModules) -> tuple[tuple[Any, ...], Any]:
    tensor_spec = modules.cjax.TensorSpec
    qkv_spec = tensor_spec(mode=(0, 1, 2), divisibility=(1, 1, 8), static=True)
    cu_spec = tensor_spec(mode=(0,), divisibility=(1,), static=True)
    lse_spec = tensor_spec(mode=(0, 1), divisibility=(1, 1), static=True)
    scratch_spec = tensor_spec(mode=(0, 1), divisibility=(1, 1), static=True)
    input_spec = (qkv_spec, qkv_spec, qkv_spec, qkv_spec, qkv_spec, lse_spec, cu_spec)
    output_spec = (qkv_spec, qkv_spec, qkv_spec, scratch_spec, scratch_spec, scratch_spec, scratch_spec, scratch_spec)
    return input_spec, output_spec


def _cutlass_thd_backward_output_shapes(
    q: Float[Array, "T Hq D"],
    k: Float[Array, "T Hkv D"],
    v: Float[Array, "T Hkv D"],
    cu_seqlens: Int[Array, "N"],
    backward_tile: tuple[int, int],
) -> tuple[jax.ShapeDtypeStruct, ...]:
    total_q, q_heads, head_dim = q.shape
    total_k, kv_heads, _ = k.shape
    tile_m, tile_n = backward_tile
    num_sequences = _num_thd_sequences(cu_seqlens_shape=cu_seqlens.shape[0], cu_seqlens_rank=cu_seqlens.ndim)
    total_q_rounded = ((total_q + (num_sequences + 1) * tile_m - 1) // tile_m) * tile_m
    total_k_rounded = ((total_k + (num_sequences + 1) * tile_n * 2 - 1) // (tile_n * 2)) * (tile_n * 2)
    head_dim_rounded = ((head_dim + 31) // 32) * 32
    head_dim_v_rounded = ((v.shape[-1] + 31) // 32) * 32
    return (
        jax.ShapeDtypeStruct(q.shape, q.dtype),
        jax.ShapeDtypeStruct(k.shape, k.dtype),
        jax.ShapeDtypeStruct(v.shape, v.dtype),
        jax.ShapeDtypeStruct((q_heads, total_q_rounded), jnp.float32),
        jax.ShapeDtypeStruct((q_heads, total_q_rounded), jnp.float32),
        jax.ShapeDtypeStruct((q_heads, total_q_rounded * head_dim_rounded), jnp.float32),
        jax.ShapeDtypeStruct((kv_heads, total_k_rounded * head_dim_rounded), jnp.float32),
        jax.ShapeDtypeStruct((kv_heads, total_k_rounded * head_dim_v_rounded), jnp.float32),
    )


def _num_thd_sequences(*, cu_seqlens_shape: int, cu_seqlens_rank: int) -> int:
    if cu_seqlens_rank != 1:
        raise ValueError("cu_seqlens must be rank 1.")
    return cu_seqlens_shape - 1


def _effective_sliding_window(sliding_window: int | None, seq_len: int) -> int | None:
    """Return the local window for FA4, or None when full causal attention is equivalent."""
    if sliding_window is None or sliding_window >= seq_len:
        return None
    return sliding_window


@partial(jax.custom_vjp, nondiff_argnums=(4, 5, 6))
def _jax_fa4_thd_attention(
    q: Float[Array, "T Hq D"],
    k: Float[Array, "T Hkv D"],
    v: Float[Array, "T Hkv D"],
    cu_seqlens: Int[Array, "N"],
    softmax_scale: float,
    kernel_config: Flash4CuteKernelConfig,
    sliding_window: int | None,
) -> Float[Array, "T Hq D"]:
    out, _ = fa4_thd_attention_forward(
        q,
        k,
        v,
        cu_seqlens,
        softmax_scale=softmax_scale,
        kernel_config=kernel_config,
        sliding_window=sliding_window,
    )
    return out


def _jax_fa4_thd_attention_fwd(
    q: Float[Array, "T Hq D"],
    k: Float[Array, "T Hkv D"],
    v: Float[Array, "T Hkv D"],
    cu_seqlens: Int[Array, "N"],
    softmax_scale: float,
    kernel_config: Flash4CuteKernelConfig,
    sliding_window: int | None,
) -> tuple[
    Float[Array, "T Hq D"],
    tuple[
        Float[Array, "T Hq D"],
        Float[Array, "T Hkv D"],
        Float[Array, "T Hkv D"],
        Float[Array, "T Hq D"],
        Float[Array, "Hq T"],
        Int[Array, "N"],
    ],
]:
    out, lse = fa4_thd_attention_forward(
        q,
        k,
        v,
        cu_seqlens,
        softmax_scale=softmax_scale,
        kernel_config=kernel_config,
        sliding_window=sliding_window,
    )
    return out, (q, k, v, out, lse, cu_seqlens)


def _jax_fa4_thd_attention_bwd(
    softmax_scale: float,
    kernel_config: Flash4CuteKernelConfig,
    sliding_window: int | None,
    residuals: tuple[
        Float[Array, "T Hq D"],
        Float[Array, "T Hkv D"],
        Float[Array, "T Hkv D"],
        Float[Array, "T Hq D"],
        Float[Array, "Hq T"],
        Int[Array, "N"],
    ],
    cotangent: Float[Array, "T Hq D"] | jax.custom_derivatives.SymbolicZero,
) -> tuple[Float[Array, "T Hq D"] | None, Float[Array, "T Hkv D"] | None, Float[Array, "T Hkv D"] | None, None]:
    q, k, v, out, lse, cu_seqlens = residuals
    if isinstance(cotangent, jax.custom_derivatives.SymbolicZero):
        return jnp.zeros_like(q), jnp.zeros_like(k), jnp.zeros_like(v), None
    dq, dk, dv = fa4_thd_attention_backward(
        q,
        k,
        v,
        out,
        cotangent.astype(q.dtype),
        lse,
        cu_seqlens,
        softmax_scale=softmax_scale,
        kernel_config=kernel_config,
        sliding_window=sliding_window,
    )
    return dq, dk, dv, None


_jax_fa4_thd_attention.defvjp(_jax_fa4_thd_attention_fwd, _jax_fa4_thd_attention_bwd)


def _validate_thd_inputs(
    q: Float[Array, "T Hq D"],
    k: Float[Array, "T Hkv D"],
    v: Float[Array, "T Hkv D"],
    cu_seqlens: Int[Array, "N"],
    *,
    softmax_scale: float,
) -> None:
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError(f"q/k/v must be THD tensors, got q={q.shape}, k={k.shape}, v={v.shape}")
    if q.shape[0] != k.shape[0] or q.shape[0] != v.shape[0]:
        raise ValueError(f"q/k/v token counts must match, got q={q.shape}, k={k.shape}, v={v.shape}")
    if q.shape[-1] != k.shape[-1] or q.shape[-1] != v.shape[-1]:
        raise ValueError(f"q/k/v head dimensions must match, got q={q.shape}, k={k.shape}, v={v.shape}")
    if q.shape[1] % k.shape[1] != 0 or k.shape[1] != v.shape[1]:
        raise ValueError(f"Hq must be divisible by Hkv and K/V heads must match, got q={q.shape}, k={k.shape}")
    if cu_seqlens.ndim != 1 or cu_seqlens.dtype != jnp.int32:
        raise ValueError(f"cu_seqlens must be rank-1 int32, got shape={cu_seqlens.shape}, dtype={cu_seqlens.dtype}")
    if q.dtype not in (jnp.bfloat16, jnp.float16):
        raise TypeError(f"gpu_fa4_thd_attention currently supports only bf16/fp16, got {q.dtype}")
    if k.dtype != q.dtype or v.dtype != q.dtype:
        raise TypeError(f"q/k/v dtypes must match, got q={q.dtype}, k={k.dtype}, v={v.dtype}")
    if not isinstance(softmax_scale, float):
        raise TypeError(f"softmax_scale must be a Python float, got {type(softmax_scale).__name__}")
    if softmax_scale <= 0.0:
        raise ValueError(f"softmax_scale must be positive, got {softmax_scale}")


def gpu_fa4_thd_attention(
    q: Float[Array, "B Q Hq D"],
    k: Float[Array, "B K Hkv D"],
    v: Float[Array, "B K Hkv D"],
    mask: AttentionMask | Bool[Array, "B Q K"] | Float[Array, "B Q K"] | None,
) -> Float[Array, "B Q Hq D"]:
    """FA4 THD/varlen backend for simple causal self-attention.

    The JAX path uses upstream FA4 CuTe internals through `cutlass.jax.cutlass_call`.
    It requires fixed-shape THD segment metadata so `cu_seqlens` has a static
    shape under JIT.
    """
    if jax.default_backend() != "gpu":
        raise RuntimeError("gpu_fa4_thd_attention requires the JAX GPU backend.")
    _validate_simple_causal_self_attention(q, k, v, mask, backend_name="gpu_fa4_thd_attention")
    assert isinstance(mask, AttentionMask)
    if mask.thd_segment_metadata is None:
        raise NotImplementedError("gpu_fa4_thd_attention requires fixed-shape THD segment metadata under JAX.")
    metadata = mask.thd_segment_metadata
    batch_size, seq_len, _, head_dim = q.shape
    cu_seqlens = _thd_cu_seqlens_from_segment_lengths(
        metadata.segment_lengths,
        metadata.num_segments,
        batch_size=batch_size,
        total_tokens=batch_size * seq_len,
        token_reference=q,
    )
    kernel_config = _thd_kernel_config(head_dim)
    sliding_window = _effective_sliding_window(mask.sliding_window, seq_len)
    out = _jax_fa4_thd_attention(
        q.reshape(batch_size * seq_len, q.shape[2], head_dim),
        k.reshape(batch_size * seq_len, k.shape[2], head_dim),
        v.reshape(batch_size * seq_len, v.shape[2], head_dim),
        cu_seqlens,
        1.0 / math.sqrt(head_dim),
        kernel_config,
        sliding_window,
    )
    return out.reshape(batch_size, seq_len, q.shape[2], head_dim)


__all__ = [
    "gpu_fa4_thd_attention",
    "fa4_thd_attention_forward",
    "fa4_thd_attention_backward",
]
