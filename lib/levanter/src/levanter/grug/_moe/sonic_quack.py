# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""JAX TVM-FFI bindings for the QuACK GEMMs used by SonicMoE."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

try:
    import cutlass.cute as cute  # pyrefly: ignore[missing-import]  # optional GPU dependency
    from cuda.bindings import driver as cuda  # pyrefly: ignore[missing-import]  # optional GPU dependency
    from cutlass import BFloat16, Float32, Int32  # pyrefly: ignore[missing-import]  # optional GPU dependency
    from quack.activation import gate_fn_map  # pyrefly: ignore[missing-import]  # optional GPU dependency
    from quack.compile_utils import make_fake_tensor  # pyrefly: ignore[missing-import]  # optional GPU dependency
    from quack.cute_dsl_utils import (  # pyrefly: ignore[missing-import]  # optional GPU dependency
        get_max_active_clusters,
    )
    from quack.gemm_act import (  # pyrefly: ignore[missing-import]  # optional GPU dependency
        GemmActMixin,
        GemmGatedSm90,
    )
    from quack.gemm_default_epi import (  # pyrefly: ignore[missing-import]  # optional GPU dependency
        GemmDefaultEpiMixin,
        GemmDefaultSm90,
    )
    from quack.jax_utils import TvmFfiKernel  # pyrefly: ignore[missing-import]  # optional GPU dependency
    from quack.tile_scheduler import (  # pyrefly: ignore[missing-import]  # optional GPU dependency
        TileSchedulerOptions,
    )
    from quack.varlen_utils import (  # pyrefly: ignore[missing-import]  # optional GPU dependency
        VarlenArguments,
    )
except ModuleNotFoundError:
    cute = None


_TILE_SHAPE = (128, 192)
_CLUSTER_SHAPE = (2, 1, 1)
_ALIGNMENT = 8


def _weight_view(weight_storage: Any) -> Any:
    return cute.make_tensor(
        weight_storage.iterator,
        cute.make_layout(
            (weight_storage.shape[1], weight_storage.shape[2], weight_storage.shape[0]),
            stride=(
                weight_storage.shape[2],
                1,
                weight_storage.shape[1] * weight_storage.shape[2],
            ),
        ),
    )


def _transposed_matrix_view(storage: Any) -> Any:
    return cute.make_tensor(
        storage.iterator,
        cute.make_layout(
            (storage.shape[1], storage.shape[0]),
            stride=(1, storage.shape[1]),
        ),
    )


def _weight_grad_view(storage: Any) -> Any:
    return cute.make_tensor(
        storage.iterator,
        cute.make_layout(
            (storage.shape[1], storage.shape[2], storage.shape[0]),
            stride=(storage.shape[2], 1, storage.shape[1] * storage.shape[2]),
        ),
    )


if cute is not None:

    class _GatedVarlenFfi:
        def __init__(self) -> None:
            self.gemm = GemmGatedSm90(
                Float32,
                BFloat16,
                _TILE_SHAPE,
                _CLUSTER_SHAPE,
                pingpong=True,
                is_persistent=True,
                concat_layout=("B",),
            )
            self.max_active_clusters = get_max_active_clusters(_CLUSTER_SHAPE[0] * _CLUSTER_SHAPE[1])

        @cute.jit
        def __call__(
            self,
            x: cute.Tensor,
            weight_storage: cute.Tensor,
            offsets: cute.Tensor,
            preact: cute.Tensor,
            postact: cute.Tensor,
            stream: cuda.CUstream,
        ) -> None:
            epilogue = GemmActMixin.EpilogueArguments(postact, gate_fn_map["swiglu"])
            scheduler = TileSchedulerOptions(
                max_active_clusters=Int32(self.max_active_clusters),
                max_swizzle_size=Int32(8),
            )
            varlen = VarlenArguments(mCuSeqlensM=offsets)
            self.gemm(x, _weight_view(weight_storage), preact, None, epilogue, scheduler, varlen, stream)

    class _GroupedVarlenFfi:
        def __init__(self) -> None:
            self.gemm = GemmDefaultSm90(
                Float32,
                BFloat16,
                _TILE_SHAPE,
                _CLUSTER_SHAPE,
                pingpong=True,
                is_persistent=True,
            )
            self.max_active_clusters = get_max_active_clusters(_CLUSTER_SHAPE[0] * _CLUSTER_SHAPE[1])

        @cute.jit
        def __call__(
            self,
            x: cute.Tensor,
            weight_storage: cute.Tensor,
            offsets: cute.Tensor,
            output: cute.Tensor,
            stream: cuda.CUstream,
        ) -> None:
            epilogue = GemmDefaultEpiMixin.EpilogueArguments()
            scheduler = TileSchedulerOptions(
                max_active_clusters=Int32(self.max_active_clusters),
                max_swizzle_size=Int32(8),
            )
            varlen = VarlenArguments(mCuSeqlensM=offsets)
            self.gemm(x, _weight_view(weight_storage), output, None, epilogue, scheduler, varlen, stream)

    class _GroupedVarlenWeightGradFfi:
        def __init__(self) -> None:
            self.gemm = GemmDefaultSm90(
                Float32,
                BFloat16,
                _TILE_SHAPE,
                _CLUSTER_SHAPE,
                pingpong=True,
                is_persistent=True,
            )
            self.max_active_clusters = get_max_active_clusters(_CLUSTER_SHAPE[0] * _CLUSTER_SHAPE[1])

        @cute.jit
        def __call__(
            self,
            x: cute.Tensor,
            dout: cute.Tensor,
            offsets: cute.Tensor,
            dweights: cute.Tensor,
            stream: cuda.CUstream,
        ) -> None:
            epilogue = GemmDefaultEpiMixin.EpilogueArguments()
            scheduler = TileSchedulerOptions(
                max_active_clusters=Int32(self.max_active_clusters),
                max_swizzle_size=Int32(8),
            )
            varlen = VarlenArguments(mCuSeqlensK=offsets)
            self.gemm(
                _transposed_matrix_view(x),
                _transposed_matrix_view(dout),
                _weight_grad_view(dweights),
                None,
                epilogue,
                scheduler,
                varlen,
                stream,
            )

    def _compile_gated_varlen() -> Any:
        total_tokens = cute.sym_int()
        hidden_dim = cute.sym_int()
        output_dim = cute.sym_int()
        postact_dim = cute.sym_int()
        num_experts = cute.sym_int()
        num_offsets = cute.sym_int()
        x = make_fake_tensor(BFloat16, (total_tokens, hidden_dim), leading_dim=1, divisibility=_ALIGNMENT)
        weights = make_fake_tensor(
            BFloat16,
            (num_experts, output_dim, hidden_dim),
            leading_dim=2,
            divisibility=_ALIGNMENT,
        )
        offsets = make_fake_tensor(Int32, (num_offsets,), leading_dim=0, divisibility=4)
        preact = make_fake_tensor(
            BFloat16,
            (total_tokens, output_dim),
            leading_dim=1,
            divisibility=_ALIGNMENT,
        )
        postact = make_fake_tensor(
            BFloat16,
            (total_tokens, postact_dim),
            leading_dim=1,
            divisibility=_ALIGNMENT,
        )
        return cute.compile(
            _GatedVarlenFfi(),
            x,
            weights,
            offsets,
            preact,
            postact,
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )

    def _compile_grouped_varlen() -> Any:
        total_tokens = cute.sym_int()
        input_dim = cute.sym_int()
        output_dim = cute.sym_int()
        num_experts = cute.sym_int()
        num_offsets = cute.sym_int()
        x = make_fake_tensor(BFloat16, (total_tokens, input_dim), leading_dim=1, divisibility=_ALIGNMENT)
        weights = make_fake_tensor(
            BFloat16,
            (num_experts, output_dim, input_dim),
            leading_dim=2,
            divisibility=_ALIGNMENT,
        )
        offsets = make_fake_tensor(Int32, (num_offsets,), leading_dim=0, divisibility=4)
        output = make_fake_tensor(
            BFloat16,
            (total_tokens, output_dim),
            leading_dim=1,
            divisibility=_ALIGNMENT,
        )
        return cute.compile(
            _GroupedVarlenFfi(),
            x,
            weights,
            offsets,
            output,
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )

    def _compile_grouped_varlen_weight_grad() -> Any:
        total_tokens = cute.sym_int()
        input_dim = cute.sym_int()
        output_dim = cute.sym_int()
        num_experts = cute.sym_int()
        num_offsets = cute.sym_int()
        x = make_fake_tensor(BFloat16, (total_tokens, input_dim), leading_dim=1, divisibility=_ALIGNMENT)
        dout = make_fake_tensor(BFloat16, (total_tokens, output_dim), leading_dim=1, divisibility=_ALIGNMENT)
        offsets = make_fake_tensor(Int32, (num_offsets,), leading_dim=0, divisibility=4)
        dweights = make_fake_tensor(
            BFloat16,
            (num_experts, input_dim, output_dim),
            leading_dim=2,
            divisibility=_ALIGNMENT,
        )
        return cute.compile(
            _GroupedVarlenWeightGradFfi(),
            x,
            dout,
            offsets,
            dweights,
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )

    _GATED_VARLEN = TvmFfiKernel(
        "levanter_sonic_quack_gated_varlen_fwd",
        _compile_gated_varlen,
        allow_cuda_graph=True,
    )
    _GROUPED_VARLEN = TvmFfiKernel(
        "levanter_sonic_quack_grouped_varlen_fwd",
        _compile_grouped_varlen,
        allow_cuda_graph=True,
    )
    _GROUPED_VARLEN_WEIGHT_GRAD = TvmFfiKernel(
        "levanter_sonic_quack_grouped_varlen_weight_grad",
        _compile_grouped_varlen_weight_grad,
        allow_cuda_graph=True,
    )
else:
    _GATED_VARLEN = None
    _GROUPED_VARLEN = None
    _GROUPED_VARLEN_WEIGHT_GRAD = None


def _require_quack() -> None:
    if _GATED_VARLEN is None or _GROUPED_VARLEN is None or _GROUPED_VARLEN_WEIGHT_GRAD is None:
        raise ImportError(
            "implementation='sonic' requires quack-kernels, jax-tvm-ffi, and the NVIDIA Cutlass DSL; "
            "install the gpu extra for marin-levanter or marin."
        )


def _expert_offsets(group_sizes: Int[Array, "E"]) -> Int[Array, "Ep1"]:
    return jnp.concatenate((jnp.zeros((1,), dtype=jnp.int32), jnp.cumsum(group_sizes, dtype=jnp.int32)))


def _quack_gated_impl(
    x: Float[Array, "M H"],
    weights: Float[Array, "E H I2"],
    group_sizes: Int[Array, "E"],
) -> tuple[Float[Array, "M I2"], Float[Array, "M I"]]:
    _require_quack()
    assert _GATED_VARLEN is not None
    if x.dtype != jnp.bfloat16 or weights.dtype != jnp.bfloat16:
        raise TypeError("SonicMoE QuACK kernels require bfloat16 activations and weights")
    weight_storage = jnp.swapaxes(weights, 1, 2)
    preact_shape = jax.ShapeDtypeStruct((x.shape[0], weights.shape[2]), x.dtype)
    postact_shape = jax.ShapeDtypeStruct((x.shape[0], weights.shape[2] // 2), x.dtype)
    return _GATED_VARLEN(
        x,
        weight_storage,
        _expert_offsets(group_sizes),
        key=(),
        output_shape_dtype=(preact_shape, postact_shape),
    )


def _quack_grouped_impl(
    x: Float[Array, "M I"],
    weights: Float[Array, "E I H"],
    group_sizes: Int[Array, "E"],
) -> Float[Array, "M H"]:
    _require_quack()
    assert _GROUPED_VARLEN is not None
    if x.dtype != jnp.bfloat16 or weights.dtype != jnp.bfloat16:
        raise TypeError("SonicMoE QuACK kernels require bfloat16 activations and weights")
    weight_storage = jnp.swapaxes(weights, 1, 2)
    output_shape = jax.ShapeDtypeStruct((x.shape[0], weights.shape[2]), x.dtype)
    return _GROUPED_VARLEN(
        x,
        weight_storage,
        _expert_offsets(group_sizes),
        key=(),
        output_shape_dtype=output_shape,
    )


def _quack_grouped_input_grad_impl(
    dout: Float[Array, "M O"],
    weights: Float[Array, "E N O"],
    group_sizes: Int[Array, "E"],
) -> Float[Array, "M N"]:
    _require_quack()
    assert _GROUPED_VARLEN is not None
    if dout.dtype != jnp.bfloat16 or weights.dtype != jnp.bfloat16:
        raise TypeError("SonicMoE QuACK kernels require bfloat16 activations and weights")
    output_shape = jax.ShapeDtypeStruct((dout.shape[0], weights.shape[1]), dout.dtype)
    return _GROUPED_VARLEN(
        dout,
        weights,
        _expert_offsets(group_sizes),
        key=(),
        output_shape_dtype=output_shape,
    )


def _quack_grouped_weight_grad_impl(
    x: Float[Array, "M I"],
    dout: Float[Array, "M O"],
    group_sizes: Int[Array, "E"],
) -> Float[Array, "E I O"]:
    _require_quack()
    assert _GROUPED_VARLEN_WEIGHT_GRAD is not None
    if x.dtype != jnp.bfloat16 or dout.dtype != jnp.bfloat16:
        raise TypeError("SonicMoE QuACK kernels require bfloat16 activations and weights")
    output_shape = jax.ShapeDtypeStruct((group_sizes.shape[0], x.shape[1], dout.shape[1]), x.dtype)
    return _GROUPED_VARLEN_WEIGHT_GRAD(
        x,
        dout,
        _expert_offsets(group_sizes),
        key=(),
        output_shape_dtype=output_shape,
    )


@jax.custom_vjp
def quack_gated_varlen(
    x: Float[Array, "M H"],
    weights: Float[Array, "E H I2"],
    group_sizes: Int[Array, "E"],
) -> Float[Array, "M I"]:
    """Apply SonicMoE's QuACK fused grouped up projection and SwiGLU."""
    return _quack_gated_impl(x, weights, group_sizes)[1]


def _quack_gated_fwd(x: jax.Array, weights: jax.Array, group_sizes: jax.Array):
    preact, postact = _quack_gated_impl(x, weights, group_sizes)
    return postact, (x, weights, group_sizes, preact)


def _quack_gated_bwd(residuals: tuple[jax.Array, jax.Array, jax.Array, jax.Array], dpostact: jax.Array):
    x, weights, group_sizes, preact = residuals

    def activation(preactivation):
        gate, up = jnp.split(preactivation, 2, axis=1)
        return jax.nn.silu(gate) * up

    _, activation_pullback = jax.vjp(activation, preact)
    (dpreact,) = activation_pullback(dpostact)
    dx = _quack_grouped_input_grad_impl(dpreact, weights, group_sizes)
    dweights = _quack_grouped_weight_grad_impl(x, dpreact, group_sizes)
    return dx, dweights, None


quack_gated_varlen.defvjp(_quack_gated_fwd, _quack_gated_bwd)


@jax.custom_vjp
def quack_grouped_varlen(
    x: Float[Array, "M I"],
    weights: Float[Array, "E I H"],
    group_sizes: Int[Array, "E"],
) -> Float[Array, "M H"]:
    """Apply SonicMoE's QuACK variable-length grouped down projection."""
    return _quack_grouped_impl(x, weights, group_sizes)


def _quack_grouped_fwd(x: jax.Array, weights: jax.Array, group_sizes: jax.Array):
    output = _quack_grouped_impl(x, weights, group_sizes)
    return output, (x, weights, group_sizes)


def _quack_grouped_bwd(residuals: tuple[jax.Array, jax.Array, jax.Array], doutput: jax.Array):
    x, weights, group_sizes = residuals
    dx = _quack_grouped_input_grad_impl(doutput, weights, group_sizes)
    dweights = _quack_grouped_weight_grad_impl(x, doutput, group_sizes)
    return dx, dweights, None


quack_grouped_varlen.defvjp(_quack_grouped_fwd, _quack_grouped_bwd)


__all__ = ["quack_gated_varlen", "quack_grouped_varlen"]
