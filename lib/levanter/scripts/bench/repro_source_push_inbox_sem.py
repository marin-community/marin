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
K = 128


def make_kernel(lowering_semantics: mgpu.LoweringSemantics):
    def body(x_ref, inbox_ref, meta_ref):
        full_sem = pl.get_global(mgpu.SemaphoreType.REGULAR)
        rank = lax.axis_index(AXIS)
        peer = (rank + 1) % 2
        remote_inbox_ref = mgpu.remote_ref(
            inbox_ref,
            peer,
            device_id_type=pl.DeviceIdType.LOGICAL,
        )
        remote_meta_ref = mgpu.remote_ref(
            meta_ref,
            peer,
            device_id_type=pl.DeviceIdType.LOGICAL,
        )

        def smem_scope(tile_smem):
            tile_smem[:, :] = x_ref[pl.ds(0, M), pl.ds(0, K)]
            mgpu.commit_smem()
            mgpu.copy_smem_to_gmem(
                tile_smem,
                remote_inbox_ref.at[pl.ds(0, M), pl.ds(0, K)],
            )
            mgpu.wait_smem_to_gmem(0, wait_read_only=False)

        pl.run_scoped(
            smem_scope,
            tile_smem=mgpu.SMEM((M, K), dtype=x_ref.dtype),
        )
        remote_meta_ref[0] = rank
        remote_meta_ref[1] = jnp.int32(M)
        pl.semaphore_signal(full_sem, device_id=peer, device_id_type=pl.DeviceIdType.LOGICAL)
        pl.semaphore_wait(full_sem)

    return mgpu.kernel(
        body,
        out_shape=[
            jax.ShapeDtypeStruct((M, K), jnp.bfloat16),
            jax.ShapeDtypeStruct((2,), jnp.int32),
        ],
        grid=(1,),
        grid_names=("sm",),
        compiler_params=mgpu.CompilerParams(lowering_semantics=lowering_semantics),
    )


def run_one(lowering_semantics: mgpu.LoweringSemantics, x):
    kernel = make_kernel(lowering_semantics)

    def local_fn(x_local):
        x_local = x_local[0]
        inbox, meta = kernel(x_local)
        return inbox[None, ...], meta[None, ...]

    fn = jax.jit(
        shard_map(
            local_fn,
            mesh=x.sharding.mesh,
            in_specs=P(AXIS, None, None),
            out_specs=(P(AXIS, None, None), P(AXIS, None)),
            check_vma=False,
        )
    )
    inbox, meta = fn(x)
    inbox.block_until_ready()
    meta.block_until_ready()
    observed = np.asarray(inbox[:, 0, 0], dtype=np.float32)
    expected = np.asarray([2.0, 1.0], dtype=np.float32)
    if not np.allclose(observed, expected):
        raise AssertionError(f"inbox observed {observed}, expected {expected}")
    observed_meta = np.asarray(meta, dtype=np.int32)
    expected_meta = np.asarray([[1, M], [0, M]], dtype=np.int32)
    if not np.array_equal(observed_meta, expected_meta):
        raise AssertionError(f"meta observed {observed_meta}, expected {expected_meta}")
    print(f"{lowering_semantics.name}: success")
    print("inbox[:, 0, 0] =", np.asarray(inbox[:, 0, 0]))
    print("meta =", observed_meta)


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

    for lowering_semantics in (mgpu.LoweringSemantics.Warpgroup, mgpu.LoweringSemantics.Lane):
        print(f"=== {lowering_semantics.name} ===", flush=True)
        try:
            run_one(lowering_semantics, x)
        except Exception as exc:
            print(f"{lowering_semantics.name}: {type(exc).__name__}: {exc}")
            traceback.print_exc()
        finally:
            jax.clear_caches()


if __name__ == "__main__":
    main()
