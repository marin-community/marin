# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""QuACK symmetric GEMM for Grug Muon Newton-Schulz on Hopper and Blackwell GPUs."""

from __future__ import annotations

import jax
import jax.numpy as jnp

import cutlass
import cutlass.cute as cute
import cutlass.jax as cjax
from cutlass import Float32
from levanter.cutlass_kernel_cache import cute_launcher_factory, cutlass_call, gpu_compute_capability
from quack.cute_dsl_utils import get_max_active_clusters
from quack.gemm_tvm_ffi_utils import make_scheduler_args
from quack.gemm_symmetric import GemmSymmetricMixin, GemmSymmetricSm90, GemmSymmetricSm100

_ACC = cutlass.Float32
_JAX_TO_CUTE = {
    jnp.dtype(jnp.bfloat16): cutlass.BFloat16,
    jnp.dtype(jnp.float16): cutlass.Float16,
    jnp.dtype(jnp.float32): cutlass.Float32,
}

# QuACK's symmetric configs (_symmetric_gemm_config): SM90 uses tile (128, 256), SM100 uses
# tile (256, 256); both use cluster_M 2 and static persistence.
# NOTE: (256, 128) benchmarks 1.62x faster on the isolated [96, 2560, 5120] expert gram, but it
# BREAKS live training (loss stuck at init, run never progresses) -- the NS also orthogonalizes
# non-expert matrices whose shapes (256, 128) does not handle correctly, and the isolated gram
# microbench does not cover them. (256, 256) is validated clean end-to-end (30-step GB200 run,
# loss 11.8 -> 5.94). Do not narrow tile_N without validating on the full set of live NS shapes.
_SM90_MMA_TILER = (128, 256)
_SM100_MMA_TILER = (256, 256)
_HOPPER_ARCH_FAMILY = 9
_BLACKWELL_ARCH_FAMILIES = (10, 11)
_DEFAULT_CLUSTER = (2, 1, 1)
_DEFAULT_SWIZZLE = 8


def _cute_dtype(dt):
    return _JAX_TO_CUTE[jnp.dtype(dt)]


def _transpose_mn(mD):
    """aux = D^T (same storage): swap matrix axes of batch-first [L, M, M]."""
    return cute.make_tensor(mD.iterator, cute.select(mD.layout, mode=[0, 2, 1]))


def _symmetric_gemm_config(arch: int) -> tuple[int, tuple[int, int]]:
    arch_family = arch // 10
    if arch_family == _HOPPER_ARCH_FAMILY:
        return arch_family, _SM90_MMA_TILER
    if arch_family in _BLACKWELL_ARCH_FAMILIES:
        return arch_family, _SM100_MMA_TILER
    raise NotImplementedError(f"QuACK symmetric GEMM does not support CUDA compute capability {arch}.")


@cute_launcher_factory
def _build_launcher(*, arch_family, a_dtype, mma_tiler_mnk, cluster_mnk, mac, max_swizzle):
    gemm_type = GemmSymmetricSm90 if arch_family == _HOPPER_ARCH_FAMILY else GemmSymmetricSm100

    @cute.jit
    def launcher(stream, mA, mB, mD):
        # Static persistence (use_clc_persistence=False): no tile-count semaphore, deterministic
        # tile scheduling. The dynamic CLC path raced and corrupted tiny/square outputs.
        gemm = gemm_type(_ACC, a_dtype, mma_tiler_mnk, cluster_mnk, use_clc_persistence=False)
        # aux = D.mT (transposed view of the single output): the kernel writes each lower tile to D
        # and its mirror to D.mT, so D is fully symmetric. alpha/beta must be set (D = a*acc + b*C).
        epi_args = GemmSymmetricMixin.EpilogueArguments(_transpose_mn(mD), alpha=Float32(1.0), beta=Float32(1.0))
        scheduler_args = make_scheduler_args(mac, max_swizzle, None)
        gemm(mA, mB, mD, None, epi_args, scheduler_args, None, stream)

    return launcher


def quack_symmetric_gemm(
    X: jax.Array,
    *,
    mma_tiler_mnk: tuple[int, int] | None = None,
    cluster_mnk: tuple[int, int, int] = _DEFAULT_CLUSTER,
    max_swizzle: int = _DEFAULT_SWIZZLE,
) -> jax.Array:
    """Batched symmetric GEMM: ``X[L, M, K] -> X @ X^T [L, M, M]`` (full symmetric, bit-exact).

    ``X`` must be device-local (no cross-device sharding on any axis) — call inside a shard_map.
    The kernel computes ``A @ B^T``, so both operands are ``X``, k-major ([L, M, K]).
    """
    L, M, K = X.shape
    arch_family, default_mma_tiler = _symmetric_gemm_config(gpu_compute_capability())
    if mma_tiler_mnk is None:
        mma_tiler_mnk = default_mma_tiler
    a_dtype = _cute_dtype(X.dtype)
    mac = get_max_active_clusters(cluster_mnk[0] * cluster_mnk[1])
    launcher = _build_launcher(
        arch_family=arch_family,
        a_dtype=a_dtype,
        mma_tiler_mnk=mma_tiler_mnk,
        cluster_mnk=cluster_mnk,
        mac=mac,
        max_swizzle=max_swizzle,
    )
    ts = cjax.TensorSpec
    spec = ts(divisibility=(1, 1, 8), static=False)
    call = cutlass_call(
        launcher,
        output_shape_dtype=(jax.ShapeDtypeStruct((L, M, M), X.dtype),),
        input_spec=(spec, spec),
        output_spec=(spec,),
        use_static_tensors=False,
    )
    (d,) = call(X, X)
    return d
