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
    from quack.activation import (  # pyrefly: ignore[missing-import]  # optional GPU dependency
        dgate_fn_map,
        gate_fn_map,
    )
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
    from quack.gemm_dact import (  # pyrefly: ignore[missing-import]  # optional GPU dependency
        GemmDGatedMixin,
        GemmDGatedSm90,
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

    class _GroupedVarlenConcatFfi:
        def __init__(self) -> None:
            self.gemm = GemmDefaultSm90(
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

    class _GroupedVarlenWeightGradConcatFfi:
        def __init__(self) -> None:
            self.gemm = GemmDefaultSm90(
                Float32,
                BFloat16,
                _TILE_SHAPE,
                _CLUSTER_SHAPE,
                pingpong=True,
                is_persistent=True,
                concat_layout=("out",),
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

    class _GroupedVarlenDGatedFfi:
        def __init__(self) -> None:
            self.gemm = GemmDGatedSm90(
                Float32,
                BFloat16,
                _TILE_SHAPE,
                _CLUSTER_SHAPE,
                pingpong=True,
                is_persistent=True,
            )
            self.gemm.implicit_dtype = BFloat16
            self.max_active_clusters = get_max_active_clusters(_CLUSTER_SHAPE[0] * _CLUSTER_SHAPE[1])

        @cute.jit
        def __call__(
            self,
            dout: cute.Tensor,
            weight_storage: cute.Tensor,
            preact: cute.Tensor,
            offsets: cute.Tensor,
            dpreact: cute.Tensor,
            postact: cute.Tensor,
            stream: cuda.CUstream,
        ) -> None:
            epilogue = GemmDGatedMixin.EpilogueArguments(postact, dgate_fn_map["swiglu"])
            scheduler = TileSchedulerOptions(
                max_active_clusters=Int32(self.max_active_clusters),
                max_swizzle_size=Int32(8),
            )
            varlen = VarlenArguments(mCuSeqlensM=offsets)
            self.gemm(
                dout,
                _weight_view(weight_storage),
                cute.recast_tensor(dpreact, Float32),
                cute.recast_tensor(preact, Float32),
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

    def _compile_grouped_varlen_concat() -> Any:
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
            _GroupedVarlenConcatFfi(),
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

    def _compile_grouped_varlen_weight_grad_concat() -> Any:
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
            _GroupedVarlenWeightGradConcatFfi(),
            x,
            dout,
            offsets,
            dweights,
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )

    def _compile_grouped_varlen_dgated() -> Any:
        total_tokens = cute.sym_int()
        hidden_dim = cute.sym_int()
        intermediate_dim = cute.sym_int()
        preact_dim = cute.sym_int()
        num_experts = cute.sym_int()
        num_offsets = cute.sym_int()
        dout = make_fake_tensor(BFloat16, (total_tokens, hidden_dim), leading_dim=1, divisibility=_ALIGNMENT)
        weights = make_fake_tensor(
            BFloat16,
            (num_experts, intermediate_dim, hidden_dim),
            leading_dim=2,
            divisibility=_ALIGNMENT,
        )
        preact = make_fake_tensor(
            BFloat16,
            (total_tokens, preact_dim),
            leading_dim=1,
            divisibility=_ALIGNMENT,
        )
        offsets = make_fake_tensor(Int32, (num_offsets,), leading_dim=0, divisibility=4)
        dpreact = make_fake_tensor(
            BFloat16,
            (total_tokens, preact_dim),
            leading_dim=1,
            divisibility=_ALIGNMENT,
        )
        postact = make_fake_tensor(
            BFloat16,
            (total_tokens, intermediate_dim),
            leading_dim=1,
            divisibility=_ALIGNMENT,
        )
        return cute.compile(
            _GroupedVarlenDGatedFfi(),
            dout,
            weights,
            preact,
            offsets,
            dpreact,
            postact,
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
    _GROUPED_VARLEN_CONCAT = TvmFfiKernel(
        "levanter_sonic_quack_grouped_varlen_concat",
        _compile_grouped_varlen_concat,
        allow_cuda_graph=True,
    )
    _GROUPED_VARLEN_WEIGHT_GRAD = TvmFfiKernel(
        "levanter_sonic_quack_grouped_varlen_weight_grad",
        _compile_grouped_varlen_weight_grad,
        allow_cuda_graph=True,
    )
    _GROUPED_VARLEN_WEIGHT_GRAD_CONCAT = TvmFfiKernel(
        "levanter_sonic_quack_grouped_varlen_weight_grad_concat",
        _compile_grouped_varlen_weight_grad_concat,
        allow_cuda_graph=True,
    )
    _GROUPED_VARLEN_DGATED = TvmFfiKernel(
        "levanter_sonic_quack_grouped_varlen_dgated",
        _compile_grouped_varlen_dgated,
        allow_cuda_graph=True,
    )
else:
    _GATED_VARLEN = None
    _GROUPED_VARLEN = None
    _GROUPED_VARLEN_CONCAT = None
    _GROUPED_VARLEN_WEIGHT_GRAD = None
    _GROUPED_VARLEN_WEIGHT_GRAD_CONCAT = None
    _GROUPED_VARLEN_DGATED = None


def _require_quack() -> None:
    if (
        _GATED_VARLEN is None
        or _GROUPED_VARLEN is None
        or _GROUPED_VARLEN_CONCAT is None
        or _GROUPED_VARLEN_WEIGHT_GRAD is None
        or _GROUPED_VARLEN_WEIGHT_GRAD_CONCAT is None
        or _GROUPED_VARLEN_DGATED is None
    ):
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


def _quack_grouped_concat_impl(
    x: Float[Array, "M I"],
    weights: Float[Array, "E I O"],
    group_sizes: Int[Array, "E"],
) -> Float[Array, "M O"]:
    _require_quack()
    assert _GROUPED_VARLEN_CONCAT is not None
    if x.dtype != jnp.bfloat16 or weights.dtype != jnp.bfloat16:
        raise TypeError("SonicMoE QuACK kernels require bfloat16 activations and weights")
    weight_storage = jnp.swapaxes(weights, 1, 2)
    output_shape = jax.ShapeDtypeStruct((x.shape[0], weights.shape[2]), x.dtype)
    return _GROUPED_VARLEN_CONCAT(
        x,
        weight_storage,
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


def _quack_grouped_weight_grad_concat_impl(
    x: Float[Array, "M I"],
    dout: Float[Array, "M O"],
    group_sizes: Int[Array, "E"],
) -> Float[Array, "E I O"]:
    _require_quack()
    assert _GROUPED_VARLEN_WEIGHT_GRAD_CONCAT is not None
    if x.dtype != jnp.bfloat16 or dout.dtype != jnp.bfloat16:
        raise TypeError("SonicMoE QuACK kernels require bfloat16 activations and weights")
    output_shape = jax.ShapeDtypeStruct((group_sizes.shape[0], x.shape[1], dout.shape[1]), x.dtype)
    return _GROUPED_VARLEN_WEIGHT_GRAD_CONCAT(
        x,
        dout,
        _expert_offsets(group_sizes),
        key=(),
        output_shape_dtype=output_shape,
    )


def _quack_grouped_dswiglu_impl(
    dout: Float[Array, "M H"],
    weights: Float[Array, "E I H"],
    preact: Float[Array, "M I2"],
    group_sizes: Int[Array, "E"],
) -> tuple[Float[Array, "M I2"], Float[Array, "M I"]]:
    _require_quack()
    assert _GROUPED_VARLEN_DGATED is not None
    if dout.dtype != jnp.bfloat16 or weights.dtype != jnp.bfloat16 or preact.dtype != jnp.bfloat16:
        raise TypeError("SonicMoE QuACK kernels require bfloat16 activations and weights")
    if preact.shape[1] != 2 * weights.shape[1]:
        raise ValueError("SonicMoE SwiGLU preactivation must be twice the intermediate dimension")
    dpreact_shape = jax.ShapeDtypeStruct(preact.shape, preact.dtype)
    postact_shape = jax.ShapeDtypeStruct((preact.shape[0], weights.shape[1]), preact.dtype)
    return _GROUPED_VARLEN_DGATED(
        dout,
        weights,
        preact,
        _expert_offsets(group_sizes),
        key=(),
        output_shape_dtype=(dpreact_shape, postact_shape),
    )


@jax.custom_vjp
def quack_mlp_varlen(
    x: Float[Array, "M H"],
    up_weights: Float[Array, "E H I2"],
    down_weights: Float[Array, "E I H"],
    group_sizes: Int[Array, "E"],
) -> Float[Array, "M H"]:
    """Apply the memory-efficient SonicMoE grouped expert MLP."""
    hidden = _quack_gated_impl(x, up_weights, group_sizes)[1]
    return _quack_grouped_impl(hidden, down_weights, group_sizes)


def _quack_mlp_fwd(x: jax.Array, up_weights: jax.Array, down_weights: jax.Array, group_sizes: jax.Array):
    hidden = _quack_gated_impl(x, up_weights, group_sizes)[1]
    output = _quack_grouped_impl(hidden, down_weights, group_sizes)
    return output, (x, up_weights, down_weights, group_sizes)


def _quack_mlp_bwd(residuals: tuple[jax.Array, jax.Array, jax.Array, jax.Array], doutput: jax.Array):
    x, up_weights, down_weights, group_sizes = residuals
    preact = _quack_grouped_concat_impl(x, up_weights, group_sizes)
    dpreact, hidden = _quack_grouped_dswiglu_impl(doutput, down_weights, preact, group_sizes)
    dx = _quack_grouped_concat_impl(dpreact, jnp.swapaxes(up_weights, 1, 2), group_sizes)
    dup_weights = _quack_grouped_weight_grad_concat_impl(x, dpreact, group_sizes)
    ddown_weights = _quack_grouped_weight_grad_impl(hidden, doutput, group_sizes)
    return dx, dup_weights, ddown_weights, None


quack_mlp_varlen.defvjp(_quack_mlp_fwd, _quack_mlp_bwd)


__all__ = ["quack_mlp_varlen"]
