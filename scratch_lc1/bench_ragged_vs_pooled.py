# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Time one d512 fast_track MoE layer (forward + backward) on 8 GPUs: pooled-wave vs ragged all-to-all.

Shapes follow the d512 recipe: 65536 tokens per GPU, LatentMoE width 256, 384 experts (48 per GPU) with top-8,
ungated ReLU^2 experts of width 384, capacity 1.15. Every variant runs in its own child process because the
ragged transport kernel is an XLA flag fixed at backend start. The runtime XLA flags are fast_track's own
(``experiments.grug.fast_track.train._apply_runtime_defaults`` with the inline watch on, as the ladder runs).

Run from the repo root in ONE process that owns all 8 GPUs (the device and one-shot kernels need peer access):

    python -m scratch_lc1.bench_ragged_vs_pooled
    python -m scratch_lc1.bench_ragged_vs_pooled --variants pooled,ragged-device --profile-dir /tmp/moe_prof

Prints one row per variant with median / p10 / p90 ms and the dropped-assignment count.
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from levanter.grug.grug_moe import moe_mlp
from levanter.kernels.pallas.relu2_mlp import fused_relu2
from levanter.utils.activation import ActivationFunctionEnum

from experiments.grug.fast_track.train import RaggedTransport, _apply_runtime_defaults

TOKENS_PER_GPU = 65536
HIDDEN = 256  # LatentMoE latent width: the experts' input/output dim
INTERMEDIATE = 384
NUM_EXPERTS = 384
TOPK = 8
CAPACITY_FACTOR = 1.15
RESULT_PREFIX = "BENCH_RESULT "
MODULE = "scratch_lc1.bench_ragged_vs_pooled"

# variant -> (moe implementation, ragged transport or None, pooled-wave waves)
VARIANTS = {
    "pooled": ("fixed_pooled_wave_all_to_all", None, 1),
    "pooled-w3": ("fixed_pooled_wave_all_to_all", None, 3),
    "ragged-device": ("ragged_all_to_all", "device", 1),
    "ragged-one_shot": ("ragged_all_to_all", "one_shot", 1),
    "ragged-nccl": ("ragged_all_to_all", "nccl", 1),
}


def _child_env(transport: str | None) -> dict[str, str]:
    """fast_track's runtime env and XLA flags for this variant, without touching this process's env."""
    saved = dict(os.environ)
    try:
        _apply_runtime_defaults(
            inline_watch_enabled=True, ragged_transport=None if transport is None else RaggedTransport(transport)
        )
        return dict(os.environ)
    finally:
        os.environ.clear()
        os.environ.update(saved)


def run_child(args: argparse.Namespace) -> None:
    implementation, _, waves = VARIANTS[args.child]
    devices = jax.devices()
    num_devices = len(devices)
    if jax.process_count() != 1:
        raise RuntimeError(f"run as one process owning every GPU, got {jax.process_count()} processes")
    print(f"[{args.child}] {num_devices} x {devices[0].device_kind}; XLA_FLAGS={os.environ.get('XLA_FLAGS')}")

    mesh = Mesh(
        np.asarray(devices).reshape(1, 1, num_devices, 1),
        ("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit,) * 4,
    )
    batch_axes = ("replica_dcn", "data", "expert")
    tokens = args.tokens_per_gpu * num_devices
    token_sharding = NamedSharding(mesh, P(batch_axes, None))
    expert_sharding = NamedSharding(mesh, P("expert", None, None))
    keys = jax.random.split(jax.random.key(args.seed), 6)

    with jax.set_mesh(mesh):
        x = jax.device_put(jax.random.normal(keys[0], (tokens, HIDDEN), jnp.bfloat16), token_sharding)
        # Router-like selection: top-k of noisy logits plus a per-expert skew (0: balanced as under QB).
        logits = jax.random.normal(keys[1], (tokens, NUM_EXPERTS), jnp.float32)
        logits = logits + args.expert_skew * jax.random.normal(keys[2], (NUM_EXPERTS,), jnp.float32)
        weights, selected = jax.lax.top_k(logits, TOPK)
        selected = jax.device_put(selected.astype(jnp.int32), token_sharding)
        combine = jax.device_put(jax.nn.softmax(weights, axis=-1).astype(jnp.bfloat16), token_sharding)
        w_up = jax.device_put(
            (jax.random.normal(keys[3], (NUM_EXPERTS, HIDDEN, INTERMEDIATE)) * HIDDEN**-0.5).astype(jnp.bfloat16),
            expert_sharding,
        )
        w_down = jax.device_put(
            (jax.random.normal(keys[4], (NUM_EXPERTS, INTERMEDIATE, HIDDEN)) * INTERMEDIATE**-0.5).astype(jnp.bfloat16),
            expert_sharding,
        )
        cotangent = jax.device_put(jax.random.normal(keys[5], (tokens, HIDDEN), jnp.bfloat16), token_sharding)
        activation = fused_relu2 if args.fused_relu2 else ActivationFunctionEnum.relu2

        def layer(x, w_up, w_down):
            return moe_mlp(
                x,
                selected,
                combine,
                w_up,
                w_down,
                activation=activation,
                implementation=implementation,
                mesh=mesh,
                capacity_factor=CAPACITY_FACTOR,
                pooled_transport_capacity_factor=CAPACITY_FACTOR,
                report_capacity_overflow=True,
                num_expert_waves=waves,
            )

        @jax.jit
        def forward(x, w_up, w_down):
            out, counts = layer(x, w_up, w_down)
            return out, counts.dropped

        @jax.jit
        def forward_backward(x, w_up, w_down):
            def loss(x, w_up, w_down):
                out, counts = layer(x, w_up, w_down)
                return jnp.sum((out * cotangent).astype(jnp.float32)), counts.dropped

            (value, dropped), grads = jax.value_and_grad(loss, argnums=(0, 1, 2), has_aux=True)(x, w_up, w_down)
            return value, dropped, grads

        def time_ms(fn) -> list[float]:
            for _ in range(args.warmup):
                jax.block_until_ready(fn(x, w_up, w_down))
            samples = []
            for _ in range(args.iters):
                start = time.perf_counter()
                jax.block_until_ready(fn(x, w_up, w_down))
                samples.append((time.perf_counter() - start) * 1e3)
            return samples

        fwd = time_ms(forward)
        fwd_bwd = time_ms(forward_backward)
        _, dropped, _ = forward_backward(x, w_up, w_down)
        if args.profile_dir:
            jax.profiler.start_trace(os.path.join(args.profile_dir, args.child))
            for _ in range(3):
                jax.block_until_ready(forward_backward(x, w_up, w_down))
            jax.profiler.stop_trace()

    def summary(samples: list[float]) -> dict[str, float]:
        ordered = sorted(samples)
        return {
            "median": statistics.median(ordered),
            "p10": ordered[len(ordered) // 10],
            "p90": ordered[(len(ordered) * 9) // 10],
        }

    result = {
        "variant": args.child,
        "fwd": summary(fwd),
        "fwd_bwd": summary(fwd_bwd),
        "dropped": int(dropped),
        "assignments": tokens * TOPK,
    }
    print(RESULT_PREFIX + json.dumps(result), flush=True)


def run_parent(args: argparse.Namespace) -> None:
    rows = []
    for variant in args.variants.split(","):
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}; choose from {sorted(VARIANTS)}")
        env = _child_env(VARIANTS[variant][1])
        if args.ragged_dot_impl:
            env["RAGGED_DOT_IMPL"] = args.ragged_dot_impl
        cmd = [sys.executable, "-m", MODULE, "--child", variant, *_forwarded(args)]
        print(f"=== {variant}: {' '.join(cmd)}", flush=True)
        env["NCCL_DEBUG"] = "WARN"
        log_path = f"/tmp/bench_{variant}.log"
        with open(log_path, "w") as log:
            proc = subprocess.run(cmd, env=env, text=True, stdout=log, stderr=subprocess.STDOUT, check=False)
        output = Path(log_path).read_text()
        sys.stdout.write(output[-4000:])
        results = [line for line in output.splitlines() if line.startswith(RESULT_PREFIX)]
        if proc.returncode != 0 or not results:
            # Keep going: one transport failing to lower (e.g. the device kernel) must not hide the others.
            print(f"!!! {variant} failed (exit {proc.returncode}):\n{output[-6000:]}", flush=True)
            rows.append((variant, None))
            continue
        rows.append((variant, json.loads(results[-1][len(RESULT_PREFIX) :])))

    print("\nvariant            fwd+bwd ms (p10-p90)      fwd ms (p10-p90)       dropped")
    for variant, result in rows:
        if result is None:
            print(f"{variant:<18} FAILED")
            continue
        fb, f = result["fwd_bwd"], result["fwd"]
        dropped = f"{result['dropped']} ({100 * result['dropped'] / result['assignments']:.3f}%)"
        print(
            f"{variant:<18} {fb['median']:8.2f} ({fb['p10']:.2f}-{fb['p90']:.2f})    "
            f"{f['median']:7.2f} ({f['p10']:.2f}-{f['p90']:.2f})    {dropped}"
        )


def _forwarded(args: argparse.Namespace) -> list[str]:
    forwarded = ["--iters", str(args.iters), "--warmup", str(args.warmup), "--seed", str(args.seed)]
    forwarded += ["--tokens-per-gpu", str(args.tokens_per_gpu)]
    forwarded += ["--expert-skew", str(args.expert_skew)]
    if args.fused_relu2:
        forwarded.append("--fused-relu2")
    if args.profile_dir:
        forwarded += ["--profile-dir", args.profile_dir]
    return forwarded


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variants", default="pooled,ragged-device,ragged-one_shot,ragged-nccl")
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tokens-per-gpu", type=int, default=TOKENS_PER_GPU)
    parser.add_argument(
        "--expert-skew", type=float, default=0.0, help="Std of a per-expert logit bias (0: QB-balanced load)."
    )
    parser.add_argument(
        "--fused-relu2",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pass fused_relu2 (pooled-wave runs its fused Pallas kernel; ragged applies it elementwise).",
    )
    parser.add_argument("--ragged-dot-impl", default=None, help="RAGGED_DOT_IMPL for haliax ragged_dot (triton/xla).")
    parser.add_argument("--profile-dir", default=None, help="Also capture a 3-step jax profile per variant here.")
    parser.add_argument("--child", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.child is None:
        run_parent(args)
    else:
        run_child(args)


if __name__ == "__main__":
    main()
