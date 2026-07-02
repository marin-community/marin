# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import argparse
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
LOWERING_SEMANTICS = {
    "warpgroup": mgpu.LoweringSemantics.Warpgroup,
    "lane": mgpu.LoweringSemantics.Lane,
}


def make_kernel(ep_size: int, lowering_semantics: mgpu.LoweringSemantics):
    def body(x_ref, y_ref):
        rank = lax.axis_index(AXIS)
        peer_phase = pl.program_id(0)
        dst = (rank + peer_phase) % ep_size

        def _copy_scope(tile_smem):
            tile_smem[:, :] = x_ref[pl.ds(0, M), pl.ds(0, K)]
            mgpu.commit_smem()

            @pl.when(peer_phase == 0)
            def _copy_local() -> None:
                mgpu.copy_smem_to_gmem(
                    tile_smem,
                    y_ref.at[rank, pl.ds(0, M), pl.ds(0, K)],
                )

            @pl.when(peer_phase != 0)
            def _copy_remote() -> None:
                remote_y_ref = mgpu.remote_ref(
                    y_ref,
                    dst,
                    device_id_type=pl.DeviceIdType.LOGICAL,
                )
                mgpu.copy_smem_to_gmem(
                    tile_smem,
                    remote_y_ref.at[rank, pl.ds(0, M), pl.ds(0, K)],
                )

            mgpu.wait_smem_to_gmem(0, wait_read_only=False)

        pl.run_scoped(
            _copy_scope,
            tile_smem=mgpu.SMEM((M, K), dtype=x_ref.dtype),
        )

    return mgpu.kernel(
        body,
        out_shape=jax.ShapeDtypeStruct((ep_size, M, K), jnp.bfloat16),
        grid=(ep_size,),
        grid_names=("peer_phase",),
        compiler_params=mgpu.CompilerParams(lowering_semantics=lowering_semantics),
    )


def run_one(ep_size: int, lowering_semantics: mgpu.LoweringSemantics, x):
    kernel = make_kernel(ep_size, lowering_semantics)

    def local_fn(x_local):
        x_local = x_local[0]
        y = kernel(x_local)
        return y[None, ...]

    fn = jax.jit(
        shard_map(
            local_fn,
            mesh=x.sharding.mesh,
            in_specs=P(AXIS, None, None),
            out_specs=P(AXIS, None, None, None),
            check_vma=False,
        )
    )
    out = fn(x)
    out.block_until_ready()
    observed = np.asarray(out[:, :, 0, 0], dtype=np.float32)
    expected = np.broadcast_to(
        np.arange(1, ep_size + 1, dtype=np.float32)[None, :],
        (ep_size, ep_size),
    )
    if not np.allclose(observed, expected):
        raise AssertionError(f"observed\n{observed}\nexpected\n{expected}")
    print(f"{lowering_semantics.name}: success")
    print("out[:, :, 0, 0] =")
    print(np.asarray(out[:, :, 0, 0]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ep-size", type=int, default=8)
    parser.add_argument("--lowering-semantics", choices=tuple(LOWERING_SEMANTICS) + ("all",), default="all")
    args = parser.parse_args()

    devices = np.asarray(jax.devices()[: args.ep_size])
    if devices.size < args.ep_size:
        raise RuntimeError(f"Need at least {args.ep_size} visible JAX devices, got {devices.size}")
    mesh = Mesh(devices, (AXIS,))
    x_host = jnp.stack(
        [jnp.full((M, K), float(rank + 1), dtype=jnp.bfloat16) for rank in range(args.ep_size)],
        axis=0,
    )
    x = jax.device_put(x_host, NamedSharding(mesh, P(AXIS, None, None)))

    if args.lowering_semantics == "all":
        lowering_semantics_to_run = tuple(LOWERING_SEMANTICS.values())
    else:
        lowering_semantics_to_run = (LOWERING_SEMANTICS[args.lowering_semantics],)

    for lowering_semantics in lowering_semantics_to_run:
        print(f"=== {lowering_semantics.name} ===", flush=True)
        try:
            run_one(args.ep_size, lowering_semantics, x)
        except Exception as exc:
            print(f"{lowering_semantics.name}: {type(exc).__name__}: {exc}")
            traceback.print_exc()
        finally:
            jax.clear_caches()


if __name__ == "__main__":
    main()
