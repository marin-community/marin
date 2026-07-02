# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import traceback

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax, shard_map
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as mgpu
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


AXIS = "dev"
M = 64
K = 64
N = 128
WGMMA_SWIZZLE_BYTES = 128
WGMMA_TILE_M = 8


def wgmma_smem(shape: tuple[int, int], dtype, lowering_semantics: mgpu.LoweringSemantics):
    if lowering_semantics != mgpu.LoweringSemantics.Lane:
        return mgpu.SMEM(shape, dtype=dtype)
    swizzle_elems = WGMMA_SWIZZLE_BYTES // jnp.dtype(dtype).itemsize
    return mgpu.SMEM(
        shape,
        dtype=dtype,
        transforms=(
            mgpu.TilingTransform((WGMMA_TILE_M, swizzle_elems)),
            mgpu.SwizzleTransform(WGMMA_SWIZZLE_BYTES),
        ),
    )


def make_copy_kernel(lowering_semantics: mgpu.LoweringSemantics):
    def body(x_ref, y_ref):
        rank = lax.axis_index(AXIS)
        peer = (rank + 1) % 2
        remote_x_ref = mgpu.remote_ref(
            x_ref,
            peer,
            device_id_type=pl.DeviceIdType.LOGICAL,
        )

        def smem_scope(lhs_smem, barrier):
            mgpu.copy_gmem_to_smem(
                remote_x_ref.at[pl.ds(0, M), pl.ds(0, K)],
                lhs_smem,
                barrier,
            )
            mgpu.barrier_wait(barrier)
            mgpu.commit_smem()
            y_ref[0, 0] = lhs_smem[0, 0]

        pl.run_scoped(
            smem_scope,
            lhs_smem=mgpu.SMEM((M, K), dtype=x_ref.dtype),
            barrier=mgpu.Barrier(num_arrivals=1),
        )

    return mgpu.kernel(
        body,
        out_shape=jax.ShapeDtypeStruct((M, K), jnp.bfloat16),
        grid=(1,),
        grid_names=("sm",),
        compiler_params=mgpu.CompilerParams(lowering_semantics=lowering_semantics),
    )


def make_wgmma_kernel(lowering_semantics: mgpu.LoweringSemantics):
    def body(x_ref, w_ref, y_ref):
        rank = lax.axis_index(AXIS)
        peer = (rank + 1) % 2
        remote_x_ref = mgpu.remote_ref(
            x_ref,
            peer,
            device_id_type=pl.DeviceIdType.LOGICAL,
        )

        def acc_scope(acc_ref):
            def smem_scope(lhs_smem, rhs_smem, barrier):
                mgpu.copy_gmem_to_smem(
                    remote_x_ref.at[pl.ds(0, M), pl.ds(0, K)],
                    lhs_smem,
                    barrier,
                )
                mgpu.copy_gmem_to_smem(
                    w_ref.at[pl.ds(0, K), pl.ds(0, N)],
                    rhs_smem,
                    barrier,
                )
                mgpu.barrier_wait(barrier)
                mgpu.commit_smem()
                mgpu.wgmma(acc_ref, lhs_smem, rhs_smem)
                mgpu.wgmma_wait(0)

            pl.run_scoped(
                smem_scope,
                lhs_smem=wgmma_smem((M, K), x_ref.dtype, lowering_semantics),
                rhs_smem=wgmma_smem((K, N), w_ref.dtype, lowering_semantics),
                barrier=mgpu.Barrier(num_arrivals=2),
            )
            return acc_ref[...]

        y = pl.run_scoped(acc_scope, acc_ref=mgpu.ACC((M, N)))
        y_ref[:, :] = y.astype(y_ref.dtype)

    return mgpu.kernel(
        body,
        out_shape=jax.ShapeDtypeStruct((M, N), jnp.bfloat16),
        grid=(1,),
        grid_names=("sm",),
        compiler_params=mgpu.CompilerParams(lowering_semantics=lowering_semantics),
    )


def run_copy_one(lowering_semantics: mgpu.LoweringSemantics, x):
    kernel = make_copy_kernel(lowering_semantics)

    def local_fn(x_local):
        x_local = x_local[0]
        y = kernel(x_local)
        return y[None, ...]

    fn = jax.jit(
        shard_map(
            local_fn,
            mesh=x.sharding.mesh,
            in_specs=P(AXIS, None, None),
            out_specs=P(AXIS, None, None),
            check_vma=False,
        )
    )
    out = fn(x)
    out.block_until_ready()
    observed = np.asarray(out[:, 0, 0], dtype=np.float32)
    if not np.allclose(observed, np.asarray([2.0, 1.0], dtype=np.float32)):
        raise AssertionError(f"copy observed {observed}, expected [2, 1]")
    print(f"copy {lowering_semantics.name}: success")
    print("out[:, 0, 0] =", np.asarray(out[:, 0, 0]))


def run_wgmma_one(lowering_semantics: mgpu.LoweringSemantics, x, w):
    kernel = make_wgmma_kernel(lowering_semantics)

    def local_fn(x_local, w_local):
        x_local = x_local[0]
        w_local = w_local[0]
        y = kernel(x_local, w_local)
        return y[None, ...]

    fn = jax.jit(
        shard_map(
            local_fn,
            mesh=x.sharding.mesh,
            in_specs=(P(AXIS, None, None), P(AXIS, None, None)),
            out_specs=P(AXIS, None, None),
            check_vma=False,
        )
    )
    out = fn(x, w)
    out.block_until_ready()
    observed = np.asarray(out[:, 0, 0], dtype=np.float32)
    expected = np.asarray([2.0 * K, 1.0 * K], dtype=np.float32)
    if not np.allclose(observed, expected):
        raise AssertionError(f"wgmma observed {observed}, expected {expected}")
    print(f"wgmma {lowering_semantics.name}: success")
    print("out[:, 0, 0] =", np.asarray(out[:, 0, 0]))


def main():
    devices = np.asarray(jax.devices()[:2])
    if devices.size < 2:
        raise RuntimeError(f"Need at least 2 visible JAX devices, got {devices.size}")
    mesh = Mesh(devices, (AXIS,))
    x_host = jnp.stack(
        [
            jnp.full((M, K), 1.0, dtype=jnp.bfloat16),
            jnp.full((M, K), 2.0, dtype=jnp.bfloat16),
        ],
        axis=0,
    )
    x = jax.device_put(x_host, NamedSharding(mesh, P(AXIS, None, None)))
    w_host = jnp.ones((2, K, N), dtype=jnp.bfloat16)
    w = jax.device_put(w_host, NamedSharding(mesh, P(AXIS, None, None)))

    for lowering_semantics in (mgpu.LoweringSemantics.Warpgroup, mgpu.LoweringSemantics.Lane):
        print(f"=== copy {lowering_semantics.name} ===", flush=True)
        try:
            run_copy_one(lowering_semantics, x)
        except Exception as exc:
            print(f"copy {lowering_semantics.name}: {type(exc).__name__}: {exc}")
            traceback.print_exc()
        finally:
            jax.clear_caches()

    for lowering_semantics in (mgpu.LoweringSemantics.Warpgroup, mgpu.LoweringSemantics.Lane):
        print(f"=== wgmma {lowering_semantics.name} ===", flush=True)
        try:
            run_wgmma_one(lowering_semantics, x, w)
        except Exception as exc:
            print(f"wgmma {lowering_semantics.name}: {type(exc).__name__}: {exc}")
            traceback.print_exc()
        finally:
            jax.clear_caches()


if __name__ == "__main__":
    main()
