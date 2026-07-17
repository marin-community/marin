# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Probe Blackwell Warpgroup support for source-push peer refs.

This is intentionally a small capability probe, not a production kernel. It
checks whether a Warpgroup-lowered Mosaic GPU kernel can:

1. address a peer rank with ``mgpu.remote_ref``;
2. copy local SMEM to peer GMEM;
3. copy peer GMEM to local SMEM; and
4. issue one ``tcgen05_mma`` using the peer-loaded LHS.

Run this on a Blackwell multi-GPU allocation before tuning a fused source-push
W13 kernel.
"""

from __future__ import annotations

import argparse
import functools
import json
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax import Ref, lax, shard_map
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as mgpu
from jax.sharding import Mesh, PartitionSpec as P
from jaxtyping import Array, Float


AXIS = "expert"
COPY_SENTINEL_SCALE = 0.25


@dataclass(frozen=True)
class BlackwellPeerProbeConfig:
    ep_size: int = 2
    block_m: int = 64
    block_n: int = 64
    block_k: int = 64
    warmup: int = 1
    steps: int = 5
    check: bool = True

    def validate(self) -> None:
        if self.ep_size <= 1:
            raise ValueError(f"ep_size must be greater than 1, got {self.ep_size}")
        if self.block_m % 64:
            raise ValueError(f"block_m must be a multiple of 64 for tcgen05_mma, got {self.block_m}")
        if self.block_n % 8:
            raise ValueError(f"block_n must be a multiple of 8, got {self.block_n}")
        if self.block_k % 64:
            raise ValueError(f"block_k must be a multiple of 64, got {self.block_k}")
        if self.warmup < 0:
            raise ValueError(f"warmup must be non-negative, got {self.warmup}")
        if self.steps <= 0:
            raise ValueError(f"steps must be positive, got {self.steps}")


def _require_blackwell_gpu() -> str:
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"Blackwell peer probe requires a GPU backend, got {jax.default_backend()!r}")
    devices = jax.devices("gpu")
    if not devices:
        raise RuntimeError("Blackwell peer probe requires visible GPU devices")
    device_kind = getattr(devices[0], "device_kind", "")
    compute_capability = getattr(devices[0], "compute_capability", None)
    if compute_capability is not None:
        try:
            if float(compute_capability) >= 10.0:
                return device_kind
        except (TypeError, ValueError):
            pass
    if any(name in device_kind for name in ("B200", "B300", "GB200", "GB300")):
        return device_kind
    raise RuntimeError(f"Blackwell peer probe requires Blackwell GPUs, got {device_kind!r}")


def _make_mesh(ep_size: int) -> Mesh:
    devices = np.asarray(jax.devices("gpu")[:ep_size])
    if devices.size < ep_size:
        raise RuntimeError(f"Need {ep_size} visible GPU devices, got {devices.size}")
    return Mesh(devices, (AXIS,))


def _make_probe_kernel(config: BlackwellPeerProbeConfig):
    dtype = jnp.bfloat16
    swizzle = mgpu.find_swizzle(config.block_k * jnp.dtype(dtype).itemsize * 8)
    swizzle_elems = swizzle // jnp.dtype(dtype).itemsize
    transforms = (
        mgpu.TilingTransform((8, swizzle_elems)),
        mgpu.SwizzleTransform(swizzle),
    )

    def body(
        x_ref: Float[Ref, "M K"],
        w_ref: Float[Ref, "K N"],
        copy_out_ref: Float[Ref, "SRC M K"],
        matmul_out_ref: Float[Ref, "M N"],
    ) -> None:
        rank = lax.axis_index(AXIS)
        peer = lax.rem(rank + jnp.int32(1), jnp.int32(config.ep_size))
        wg_idx = lax.axis_index("wg")
        compute_wg = 0
        store_wg = 1

        @functools.partial(
            pl.run_scoped,
            copy_smem=mgpu.SMEM((config.block_m, config.block_k), dtype),
            a_smem=mgpu.SMEM((config.block_m, config.block_k), dtype, transforms=transforms),
            b_smem=mgpu.SMEM((config.block_k, config.block_n), dtype, transforms=transforms),
            acc_smem=mgpu.SMEM((config.block_m, config.block_n), dtype),
            copy_barrier=mgpu.Barrier(num_arrivals=1),
            a_barrier=mgpu.Barrier(num_arrivals=1),
            b_barrier=mgpu.Barrier(num_arrivals=1),
            consumed_barrier=mgpu.Barrier(num_arrivals=1, orders_tensor_core=True),
            mma_done_barrier=mgpu.Barrier(num_arrivals=1, orders_tensor_core=True),
            acc_tmem=mgpu.TMEM((config.block_m, config.block_n), jnp.float32),
            collective_axes=("wg",),
        )
        def _scope(
            copy_smem,
            a_smem,
            b_smem,
            acc_smem,
            copy_barrier,
            a_barrier,
            b_barrier,
            consumed_barrier,
            mma_done_barrier,
            acc_tmem,
        ) -> None:
            @pl.when(wg_idx == compute_wg)
            def _compute() -> None:
                remote_copy_out_ref = mgpu.remote_ref(copy_out_ref, peer, device_id_type=pl.DeviceIdType.LOGICAL)
                mgpu.copy_gmem_to_smem(
                    x_ref.at[pl.ds(0, config.block_m), pl.ds(0, config.block_k)],
                    copy_smem,
                    copy_barrier,
                )
                mgpu.barrier_wait(copy_barrier)
                mgpu.commit_smem()
                mgpu.copy_smem_to_gmem(
                    copy_smem,
                    remote_copy_out_ref.at[
                        rank,
                        pl.ds(0, config.block_m),
                        pl.ds(0, config.block_k),
                    ],
                )
                mgpu.wait_smem_to_gmem(0, wait_read_only=False)

                remote_x_ref = mgpu.remote_ref(x_ref, peer, device_id_type=pl.DeviceIdType.LOGICAL)
                mgpu.copy_gmem_to_smem(
                    remote_x_ref.at[pl.ds(0, config.block_m), pl.ds(0, config.block_k)],
                    a_smem,
                    a_barrier,
                )
                mgpu.copy_gmem_to_smem(
                    w_ref.at[pl.ds(0, config.block_k), pl.ds(0, config.block_n)],
                    b_smem,
                    b_barrier,
                )
                mgpu.barrier_wait(a_barrier)
                mgpu.barrier_wait(b_barrier)
                mgpu.tcgen05_mma(
                    acc_tmem,
                    a_smem,
                    b_smem,
                    consumed_barrier,
                    accumulate=False,
                )
                mgpu.tcgen05_commit_arrive(mma_done_barrier)

            @pl.when(wg_idx == store_wg)
            def _store() -> None:
                mgpu.barrier_wait(mma_done_barrier)
                acc_smem[:, :] = mgpu.async_load_tmem(acc_tmem, layout=mgpu.Layout.TCGEN05).astype(dtype)
                mgpu.commit_smem()
                mgpu.copy_smem_to_gmem(
                    acc_smem,
                    matmul_out_ref.at[
                        pl.ds(0, config.block_m),
                        pl.ds(0, config.block_n),
                    ],
                )
                mgpu.wait_smem_to_gmem(0, wait_read_only=True)
                mgpu.wait_load_tmem()

    return mgpu.kernel(
        body,
        out_shape=(
            jax.ShapeDtypeStruct((config.ep_size, config.block_m, config.block_k), dtype),
            jax.ShapeDtypeStruct((config.block_m, config.block_n), dtype),
        ),
        compiler_params=mgpu.CompilerParams(lowering_semantics=mgpu.LoweringSemantics.Warpgroup),
        kernel_name="blackwell_peer_ref_warpgroup_probe",
        grid=(1,),
        grid_names=("program",),
        num_threads=2,
        thread_name="wg",
    )


def _sharded_probe(mesh: Mesh, config: BlackwellPeerProbeConfig):
    kernel = _make_probe_kernel(config)

    def local_fn(
        x_local: Float[Array, "1 M K"],
        w_local: Float[Array, "1 K N"],
    ):
        copy_out, matmul_out = kernel(x_local[0], w_local[0])
        return copy_out[None, ...], matmul_out[None, ...]

    return shard_map(
        local_fn,
        mesh=mesh,
        in_specs=(
            P(AXIS, None, None),
            P(AXIS, None, None),
        ),
        out_specs=(
            P(AXIS, None, None, None),
            P(AXIS, None, None),
        ),
        check_vma=False,
    )


def _make_inputs(config: BlackwellPeerProbeConfig) -> tuple[jax.Array, jax.Array]:
    x = np.arange(config.ep_size * config.block_m * config.block_k, dtype=np.float32)
    x = (x.reshape(config.ep_size, config.block_m, config.block_k) * COPY_SENTINEL_SCALE).astype(np.float32)
    w = np.arange(config.ep_size * config.block_k * config.block_n, dtype=np.float32)
    w = (w.reshape(config.ep_size, config.block_k, config.block_n) / max(config.block_k, 1)).astype(np.float32)
    return jnp.asarray(x, dtype=jnp.bfloat16), jnp.asarray(w, dtype=jnp.bfloat16)


def _check_outputs(config: BlackwellPeerProbeConfig, x: jax.Array, w: jax.Array, copy_out: jax.Array, out: jax.Array):
    copy_out_host = np.asarray(copy_out)
    out_host = np.asarray(out)
    x_host = np.asarray(x)
    w_host = np.asarray(w)
    max_copy_abs_diff = 0.0
    max_matmul_abs_diff = 0.0
    for rank in range(config.ep_size):
        src = (rank - 1) % config.ep_size
        copied = copy_out_host[rank, src]
        max_copy_abs_diff = max(max_copy_abs_diff, float(np.max(np.abs(copied - x_host[src]))))

        peer = (rank + 1) % config.ep_size
        expected = x_host[peer].astype(np.float32) @ w_host[rank].astype(np.float32)
        max_matmul_abs_diff = max(
            max_matmul_abs_diff, float(np.max(np.abs(out_host[rank].astype(np.float32) - expected)))
        )
    return {
        "max_copy_abs_diff": max_copy_abs_diff,
        "max_matmul_abs_diff": max_matmul_abs_diff,
    }


def run_probe(config: BlackwellPeerProbeConfig) -> dict[str, object]:
    config.validate()
    device_kind = _require_blackwell_gpu()
    mesh = _make_mesh(config.ep_size)
    x, w = _make_inputs(config)
    sharded_probe = jax.jit(_sharded_probe(mesh, config))

    compile_start = time.perf_counter()
    copy_out, out = sharded_probe(x, w)
    jax.block_until_ready(out)
    compile_time = time.perf_counter() - compile_start

    step_times = []
    for _ in range(config.warmup):
        copy_out, out = sharded_probe(x, w)
        jax.block_until_ready(out)
    for _ in range(config.steps):
        start = time.perf_counter()
        copy_out, out = sharded_probe(x, w)
        jax.block_until_ready(out)
        step_times.append(time.perf_counter() - start)

    row: dict[str, object] = {
        "probe": "blackwell_peer_ref_warpgroup",
        "device_kind": device_kind,
        "jax_version": jax.__version__,
        "compile_time": compile_time,
        "steady_state_median": float(np.median(step_times)),
        "steady_state_min": float(np.min(step_times)),
        "steady_state_max": float(np.max(step_times)),
        **asdict(config),
    }
    if config.check:
        row.update(_check_outputs(config, x, w, copy_out, out))
    return row


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ep-size", type=int, default=2)
    parser.add_argument("--block-m", type=int, default=64)
    parser.add_argument("--block-n", type=int, default=64)
    parser.add_argument("--block-k", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--check", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    row = run_probe(
        BlackwellPeerProbeConfig(
            ep_size=args.ep_size,
            block_m=args.block_m,
            block_n=args.block_n,
            block_k=args.block_k,
            warmup=args.warmup,
            steps=args.steps,
            check=args.check,
        )
    )
    print(json.dumps(row, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
