# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, PartitionSpec as P

from levanter.kernels.pallas.fused_cross_entropy_loss import (
    BlockSizes,
    fused_cross_entropy_loss_and_logsumexp_penalty,
)

_V5P_TFLOPS_BF16_PER_CHIP = 459e12
_V5P_HBM_BW_BYTES_PER_S_PER_CHIP = 2.765e12


def _estimate_v5p_roofline(
    *,
    batch: int,
    embed: int,
    vocab: int,
    dtype: jnp.dtype,
    num_devices: int,
) -> dict[str, float] | None:
    device_kind = jax.devices()[0].device_kind.lower() if jax.devices() else ""
    if not device_kind:
        return None
    if "v5p" not in device_kind and "v5" not in device_kind:
        return None

    # v5p JAX devices are chips (megacore), so use num_devices directly.
    chips = max(1, num_devices)
    peak_tflops = _V5P_TFLOPS_BF16_PER_CHIP * chips
    # TPU matmuls accumulate in fp32 by default; no extra scaling for fp32 accum.

    flop_count = 2.0 * batch * embed * vocab
    compute_time_s = flop_count / peak_tflops

    bytes_per_elem = jnp.dtype(dtype).itemsize
    memory_bytes = (batch * embed + embed * vocab + batch) * bytes_per_elem
    peak_bw = _V5P_HBM_BW_BYTES_PER_S_PER_CHIP * chips
    memory_time_s = memory_bytes / peak_bw

    return {
        "device_kind": device_kind,
        "chips": float(chips),
        "flops": float(flop_count),
        "compute_time_s": float(compute_time_s),
        "memory_time_s": float(memory_time_s),
        "tokens_per_s": float(batch / max(compute_time_s, memory_time_s)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--pos", type=int, default=2048)
    parser.add_argument("--embed", type=int, default=1024)
    parser.add_argument("--vocab", type=int, default=128256)
    parser.add_argument("--input-dtype", type=str, default="bfloat16")
    parser.add_argument("--accum-dtype", type=str, default="float32")
    parser.add_argument("--implementation", type=str, default="pallas_tpu")
    parser.add_argument("--block-sizes", type=str, choices=("default", "infer"), default="default")
    parser.add_argument("--shard-map", action="store_true")
    parser.add_argument("--data-shards", type=int, default=0)
    args = parser.parse_args()

    print("devices:", jax.devices())

    batch = args.batch
    pos = args.pos
    embed = args.embed
    vocab = args.vocab
    input_dtype = jnp.dtype(args.input_dtype)
    accum_dtype = jnp.dtype(args.accum_dtype)
    implementation = args.implementation
    use_shard_map = args.shard_map
    data_shards = args.data_shards or len(jax.devices())
    block_sizes = BlockSizes.get_default() if args.block_sizes == "default" else None

    key = jax.random.PRNGKey(0)
    key_x, key_w, key_y = jax.random.split(key, 3)

    x_raw = jax.random.normal(key_x, (batch * pos, embed), dtype=input_dtype)
    w_raw = jax.random.normal(key_w, (embed, vocab), dtype=input_dtype)
    y_raw = jax.random.randint(key_y, (batch * pos,), 0, vocab, dtype=jnp.int32)

    roofline = _estimate_v5p_roofline(
        batch=batch * pos,
        embed=embed,
        vocab=vocab,
        dtype=accum_dtype,
        num_devices=len(jax.devices()),
    )
    if roofline is not None:
        print("roofline.device_kind", roofline["device_kind"])
        print("roofline.chips", roofline["chips"])
        print("roofline.flops", roofline["flops"])
        print("roofline.compute_time_s", roofline["compute_time_s"])
        print("roofline.memory_time_s", roofline["memory_time_s"])
        print("roofline.tokens_per_s", roofline["tokens_per_s"])

    def loss_fn(x_in, w_in, y_in):
        return fused_cross_entropy_loss_and_logsumexp_penalty(
            x_in,
            y_in,
            w_in,
            reduction="mean",
            logsumexp_weight=0.0,
            block_sizes=block_sizes,
            dtype=accum_dtype,
            logit_soft_cap=None,
            implementation=implementation,
        )

    def grad_fn(x_in, w_in, y_in):
        return jax.grad(loss_fn, argnums=(0, 1))(x_in, w_in, y_in)

    def _with_mesh(mesh):
        set_mesh = getattr(jax, "set_mesh", None)
        if set_mesh is None:
            return mesh
        set_mesh(mesh)
        return None

    def _clear_mesh(mesh):
        del mesh

    if use_shard_map:
        if data_shards <= 0:
            raise ValueError("data_shards must be positive when using --shard-map.")
        devices = jax.devices()[:data_shards]
        mesh = Mesh(np.array(devices), ("data",))

        def shard_loss(x_in, w_in, y_in):
            loss = fused_cross_entropy_loss_and_logsumexp_penalty(
                x_in,
                y_in,
                w_in,
                reduction=None,
                logsumexp_weight=0.0,
                block_sizes=block_sizes,
                dtype=accum_dtype,
                logit_soft_cap=None,
                implementation=implementation,
            )
            local_sum = jnp.sum(loss)
            total_sum = jax.lax.psum(local_sum, "data")
            total_denom = jax.lax.psum(loss.shape[0], "data")
            return total_sum / total_denom

        def shard_grad(x_in, w_in, y_in):
            def loss_inner(x_inner, w_inner, y_inner):
                loss = fused_cross_entropy_loss_and_logsumexp_penalty(
                    x_inner,
                    y_inner,
                    w_inner,
                    reduction=None,
                    logsumexp_weight=0.0,
                    block_sizes=block_sizes,
                    dtype=accum_dtype,
                    logit_soft_cap=None,
                    implementation=implementation,
                )
                local_sum = jnp.sum(loss)
                total_sum = jax.lax.psum(local_sum, "data")
                total_denom = jax.lax.psum(loss.shape[0], "data")
                return total_sum / total_denom

            return jax.grad(loss_inner, argnums=(0, 1))(x_in, w_in, y_in)

        shard_map_fn = jax.shard_map(
            shard_loss,
            in_specs=(P("data", None), P(None, None), P("data")),
            out_specs=P(),
            check_vma=False,
        )
        shard_grad_fn = jax.shard_map(
            shard_grad,
            in_specs=(P("data", None), P(None, None), P("data")),
            out_specs=(P("data", None), P(None, None)),
            check_vma=False,
        )
        loss_jit = jax.jit(shard_map_fn)
        grad_jit = jax.jit(shard_grad_fn)
    else:
        loss_jit = jax.jit(loss_fn)
        grad_jit = jax.jit(grad_fn)

    if use_shard_map:
        token = _with_mesh(mesh)
        try:
            if token is None:
                start = time.perf_counter()
                out = loss_jit(x_raw, w_raw, y_raw)
                out.block_until_ready()
                compile_time = time.perf_counter() - start

                steps = 5
                start = time.perf_counter()
                for _ in range(steps):
                    out = loss_jit(x_raw, w_raw, y_raw)
                    out.block_until_ready()
                steady_time = (time.perf_counter() - start) / steps

                start = time.perf_counter()
                grad_x, grad_w = grad_jit(x_raw, w_raw, y_raw)
                grad_x.block_until_ready()
                grad_w.block_until_ready()
                bwd_compile_time = time.perf_counter() - start

                start = time.perf_counter()
                for _ in range(steps):
                    grad_x, grad_w = grad_jit(x_raw, w_raw, y_raw)
                    grad_x.block_until_ready()
                    grad_w.block_until_ready()
                bwd_steady_time = (time.perf_counter() - start) / steps
            else:
                with token:
                    start = time.perf_counter()
                    out = loss_jit(x_raw, w_raw, y_raw)
                    out.block_until_ready()
                    compile_time = time.perf_counter() - start

                    steps = 5
                    start = time.perf_counter()
                    for _ in range(steps):
                        out = loss_jit(x_raw, w_raw, y_raw)
                        out.block_until_ready()
                    steady_time = (time.perf_counter() - start) / steps

                    start = time.perf_counter()
                    grad_x, grad_w = grad_jit(x_raw, w_raw, y_raw)
                    grad_x.block_until_ready()
                    grad_w.block_until_ready()
                    bwd_compile_time = time.perf_counter() - start

                    start = time.perf_counter()
                    for _ in range(steps):
                        grad_x, grad_w = grad_jit(x_raw, w_raw, y_raw)
                        grad_x.block_until_ready()
                        grad_w.block_until_ready()
                    bwd_steady_time = (time.perf_counter() - start) / steps
        finally:
            _clear_mesh(mesh)
    else:
        start = time.perf_counter()
        out = loss_jit(x_raw, w_raw, y_raw)
        out.block_until_ready()
        compile_time = time.perf_counter() - start

        steps = 5
        start = time.perf_counter()
        for _ in range(steps):
            out = loss_jit(x_raw, w_raw, y_raw)
            out.block_until_ready()
        steady_time = (time.perf_counter() - start) / steps

        start = time.perf_counter()
        grad_x, grad_w = grad_jit(x_raw, w_raw, y_raw)
        grad_x.block_until_ready()
        grad_w.block_until_ready()
        bwd_compile_time = time.perf_counter() - start

        start = time.perf_counter()
        for _ in range(steps):
            grad_x, grad_w = grad_jit(x_raw, w_raw, y_raw)
            grad_x.block_until_ready()
            grad_w.block_until_ready()
        bwd_steady_time = (time.perf_counter() - start) / steps

    tokens = batch * pos
    print("loss", float(out))
    print("batch", batch)
    print("pos", pos)
    print("embed", embed)
    print("vocab", vocab)
    print("input_dtype", input_dtype)
    print("accum_dtype", accum_dtype)
    print("compile_time_s", compile_time)
    print("steady_time_s", steady_time)
    print("tokens_per_s", tokens / steady_time)
    print("bwd_compile_time_s", bwd_compile_time)
    print("bwd_steady_time_s", bwd_steady_time)
    print("bwd_tokens_per_s", tokens / bwd_steady_time)


if __name__ == "__main__":
    main()
