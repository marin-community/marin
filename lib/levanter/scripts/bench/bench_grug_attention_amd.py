# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0
"""Time one Grug attention layer, forward and forward+backward, on one or more AMD GPUs.

AMD (ROCm) only: the script exits on any other platform. It benchmarks Grug's AMD attention kernel,
``gpu_pallas_triton_flash``, whose tile defaults are tuned for AMD Instinct GPUs. The NVIDIA kernels
(``gpu_fa4_cute*``) need packed segment IDs, which these inputs do not have.

Defaults are the June Snowball per-GPU shape at global batch 64 on 8 GPUs: 8 sequences of 4096 tokens, 20 query
heads, 5 KV heads, head_dim 128, bf16. ``--num-devices N`` runs on the first N GPUs, with the batch sharded over an
explicit ``data`` mesh axis as in Grug training; ``--batch`` stays per GPU, so the global batch is ``N * --batch``.
Each mask (sliding window 2048, global causal) runs for each
implementation. TFLOP/s counts only the FLOPs the mask needs: 2 matmuls forward and 5 backward (QK^T recompute,
dP, dV, dQ, dK) over the allowed (query, key) pairs. Reported TFLOP/s cover all N GPUs; ``*_per_device`` divides
by N.

With ``--sweep``, the script instead times a grid of tile sizes, warps and stages for one of the AMD Pallas-Triton
flash kernels in ``levanter.grug.attention._amd_pallas_triton_flash`` (forward, dQ or dK/dV, chosen by
``--sweep-part``), keeping the defaults for the other two. ``--block-q``, ``--block-k``, ``--num-warps`` and
``--num-stages`` replace that part's default values for one axis of the grid.

    python lib/levanter/scripts/bench/bench_grug_attention_amd.py --impl reference xla_flash gpu_pallas_triton_flash
    python lib/levanter/scripts/bench/bench_grug_attention_amd.py --sweep --sweep-part dkv --block-q 64 128 --num-stages 1
    python lib/levanter/scripts/bench/bench_grug_attention_amd.py --num-devices 8 --impl gpu_pallas_triton_flash
"""

import argparse
import itertools
import json
import statistics
import time

import jax
import jax.numpy as jnp
from jax.extend.backend import get_backend
from jax.sharding import AxisType
from jax.sharding import PartitionSpec as P

from levanter.grug.attention import AttentionMask, attention
from levanter.grug.attention._amd_pallas_triton_flash import TritonFlashBlockSizes, pallas_triton_flash_attention

WARMUP_SECONDS = 1.0
WINDOWS = 10
WINDOW_SECONDS = 0.2
DATA_AXIS = "data"

# Default sweep grid per kernel; each axis can be replaced from the command line.
_FWD_AND_DQ_GRID = {"block_q": (64, 128, 256), "block_k": (32, 64, 128), "num_warps": (4, 8), "num_stages": (1, 2)}
SWEEP_GRIDS = {
    "fwd": _FWD_AND_DQ_GRID,
    "dq": _FWD_AND_DQ_GRID,
    "dkv": {"block_q": (32, 64, 128), "block_k": (32, 64, 128), "num_warps": (4, 8), "num_stages": (1, 2)},
}
# Suffix of each kernel's fields in TritonFlashBlockSizes.
SWEEP_FIELD_SUFFIXES = {"fwd": "", "dq": "_dq", "dkv": "_dkv"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--num-devices", type=int, default=1, help="Shard the batch over the first N GPUs.")
    parser.add_argument("--batch", type=int, default=8, help="Sequences per GPU.")
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--q-heads", type=int, default=20)
    parser.add_argument("--kv-heads", type=int, default=5)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--window", type=int, default=2048)
    parser.add_argument("--impl", nargs="+", default=["reference", "xla_flash", "gpu_pallas_triton_flash"])
    parser.add_argument("--sweep", action="store_true", help="Sweep Triton tile sizes instead of comparing impls.")
    parser.add_argument("--sweep-part", choices=list(SWEEP_GRIDS), default="fwd")
    for axis in SWEEP_GRIDS["fwd"]:
        parser.add_argument(
            f"--{axis.replace('_', '-')}", type=int, nargs="+", help="--sweep values; default depends on --sweep-part"
        )
    parser.add_argument("--masks", nargs="+", choices=["window", "causal"], default=["window", "causal"])
    return parser.parse_args()


def sweep_overrides(args: argparse.Namespace) -> list[dict[str, int]]:
    """The TritonFlashBlockSizes overrides for every point of the swept kernel's grid."""
    axes = {axis: getattr(args, axis) or values for axis, values in SWEEP_GRIDS[args.sweep_part].items()}
    suffix = SWEEP_FIELD_SUFFIXES[args.sweep_part]
    return [
        {f"{axis}{suffix}": value for axis, value in zip(axes, point, strict=True)}
        for point in itertools.product(*axes.values())
    ]


def median_seconds(fn, *args) -> float:
    """Median of ten windows of back-to-back calls, after a warmup."""
    jax.block_until_ready(fn(*args))
    start = time.perf_counter()
    while time.perf_counter() - start < WARMUP_SECONDS:
        jax.block_until_ready(fn(*args))
    per_call = []
    for _ in range(WINDOWS):
        calls = 0
        start = time.perf_counter()
        while time.perf_counter() - start < WINDOW_SECONDS:
            out = fn(*args)
            calls += 1
        jax.block_until_ready(out)
        per_call.append((time.perf_counter() - start) / calls)
    return statistics.median(per_call)


def make_fns(attend):
    def forward(q, k, v):
        return attend(q, k, v)

    def loss(q, k, v, cot):
        return jnp.sum(attend(q, k, v).astype(jnp.float32) * cot)

    return jax.jit(forward), jax.jit(jax.grad(loss, argnums=(0, 1, 2)))


def global_batch(args) -> int:
    return args.num_devices * args.batch


def inputs(args):
    """Random q, k, v and output cotangent, batch-sharded over the ``data`` axis of the current mesh."""
    kq, kk, kv, kc = jax.random.split(jax.random.key(0), 4)
    shape_q = (global_batch(args), args.seq_len, args.q_heads, args.head_dim)
    shape_kv = (global_batch(args), args.seq_len, args.kv_heads, args.head_dim)
    batch_sharded = P(DATA_AXIS)
    q = jax.random.normal(kq, shape_q, jnp.bfloat16, out_sharding=batch_sharded)
    k = jax.random.normal(kk, shape_kv, jnp.bfloat16, out_sharding=batch_sharded)
    v = jax.random.normal(kv, shape_kv, jnp.bfloat16, out_sharding=batch_sharded)
    cot = jax.random.normal(kc, shape_q, jnp.float32, out_sharding=batch_sharded)
    return q, k, v, cot


def needed_flops(args, window: int | None) -> tuple[float, float]:
    """Forward and forward+backward FLOPs over the global batch."""
    if window is None or window >= args.seq_len:
        pairs = args.seq_len * (args.seq_len + 1) // 2
    else:
        pairs = window * (window + 1) // 2 + (args.seq_len - window) * window
    per_matmul = 2 * global_batch(args) * args.q_heads * pairs * args.head_dim
    return 2 * per_matmul, 7 * per_matmul


def tflops(args, name: str, flops: float, seconds: float) -> dict[str, float]:
    total = flops / seconds / 1e12
    return {name: total, f"{name}_per_device": total / args.num_devices}


def report(record: dict) -> None:
    print("RESULT " + json.dumps(record), flush=True)


def compare(args) -> None:
    q, k, v, cot = inputs(args)
    for mask_name in args.masks:
        window = args.window if mask_name == "window" else None
        mask = AttentionMask.causal(sliding_window=window)
        fwd_flops, total_flops = needed_flops(args, window)
        reference_out = None
        for impl in args.impl:
            forward, grad = make_fns(lambda q, k, v, impl=impl: attention(q, k, v, mask, implementation=impl))
            try:
                out = forward(q, k, v)
                if reference_out is None:
                    reference_out = out
                max_diff = float(jnp.max(jnp.abs(out.astype(jnp.float32) - reference_out.astype(jnp.float32))))
                t_fwd = median_seconds(forward, q, k, v)
                t_all = median_seconds(grad, q, k, v, cot)
            except Exception as e:  # report and continue with the other implementations
                report({"mask": mask_name, "impl": impl, "num_devices": args.num_devices, "error": repr(e)[:500]})
                continue
            report(
                {
                    "mask": mask_name,
                    "impl": impl,
                    "num_devices": args.num_devices,
                    "fwd_ms": t_fwd * 1e3,
                    "fwd_bwd_ms": t_all * 1e3,
                    **tflops(args, "fwd_tflops", fwd_flops, t_fwd),
                    **tflops(args, "fwd_bwd_tflops", total_flops, t_all),
                    "max_abs_diff_vs_first": max_diff,
                }
            )


def sweep(args) -> None:
    """Time each tile/launch config for one kernel; the other kernels keep their defaults."""
    q, k, v, cot = inputs(args)
    grid = sweep_overrides(args)
    for mask_name in args.masks:
        window = args.window if mask_name == "window" else None
        mask = AttentionMask.causal(sliding_window=window)
        fwd_flops, total_flops = needed_flops(args, window)
        for overrides in grid:
            config = {"mask": mask_name, "part": args.sweep_part, "num_devices": args.num_devices, **overrides}
            blocks = TritonFlashBlockSizes(**overrides)
            forward, grad = make_fns(
                lambda q, k, v, blocks=blocks: pallas_triton_flash_attention(q, k, v, mask, block_sizes=blocks)
            )
            try:
                if args.sweep_part == "fwd":
                    t = median_seconds(forward, q, k, v)
                    flops = fwd_flops
                else:
                    t = median_seconds(grad, q, k, v, cot)
                    flops = total_flops
            except Exception as e:
                report({**config, "error": repr(e)[:300]})
                continue
            report({**config, "ms": t * 1e3, **tflops(args, "tflops", flops, t)})


def main() -> None:
    args = parse_args()
    platform_version = get_backend().platform_version
    if "rocm" not in platform_version:
        raise SystemExit(f"This benchmark needs an AMD (ROCm) GPU; got platform {platform_version!r}.")
    devices = jax.devices()
    if not 1 <= args.num_devices <= len(devices):
        raise SystemExit(f"--num-devices must be between 1 and {len(devices)}, got {args.num_devices}.")
    devices = devices[: args.num_devices]
    mesh = jax.make_mesh((args.num_devices,), (DATA_AXIS,), (AxisType.Explicit,), devices=devices)
    print(f"devices={devices} global_batch={global_batch(args)} default_blocks={TritonFlashBlockSizes()}", flush=True)
    with jax.set_mesh(mesh):
        if args.sweep:
            sweep(args)
        else:
            compare(args)


if __name__ == "__main__":
    main()
