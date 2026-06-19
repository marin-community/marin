# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from levanter.grug.attention._fa4_cute_backend import fa4_cute_attention_forward
from levanter.grug.attention._fa4_cute_config import Flash4CuteKernelConfig, flash4_cute_kernel_config

TARGET_SEQ_LEN = 4096
TARGET_SLIDING_WINDOW = 2048
TARGET_HEAD_DIM = 128
TARGET_Q_HEADS = 20
TARGET_KV_HEADS = 5
EXPECTED_GPU_COUNT = 8
PRODUCTION_CONFIG_LABEL = "prod"


@dataclass(frozen=True)
class BenchConfig:
    label: str
    forward_tile: tuple[int, int]
    backward_tile: tuple[int, int]
    num_threads: int

    @property
    def kernel_config(self) -> Flash4CuteKernelConfig:
        return Flash4CuteKernelConfig(self.forward_tile, self.backward_tile, self.num_threads)


@dataclass(frozen=True)
class BenchResult:
    batch: int
    label: str
    forward_tile: tuple[int, int]
    backward_tile: tuple[int, int]
    num_threads: int
    compile_seconds: float | None
    times: list[float]
    median_seconds: float | None
    min_seconds: float | None
    mean_seconds: float | None
    stdev_seconds: float | None
    cv: float | None
    loss: float | None
    status: str
    error: str | None


def parse_tile(raw: str) -> tuple[int, int]:
    first, second = raw.lower().split("x", maxsplit=1)
    return int(first), int(second)


def format_tile(tile: tuple[int, int]) -> str:
    return f"{tile[0]}x{tile[1]}"


def parse_config(raw: str) -> BenchConfig:
    parts = raw.split("/")
    if len(parts) == 3:
        label = f"f{parts[0]}_b{parts[1]}_t{parts[2]}"
        forward_tile, backward_tile, threads = parts
    elif len(parts) == 4:
        label, forward_tile, backward_tile, threads = parts
    else:
        config_format = "FORWARD/BACKWARD/THREADS or LABEL/FORWARD/BACKWARD/THREADS"
        message = f"--config must be {config_format}, got {raw!r}"
        raise ValueError(message)
    return BenchConfig(label, parse_tile(forward_tile), parse_tile(backward_tile), int(threads))


def default_configs(production_config: Flash4CuteKernelConfig) -> list[BenchConfig]:
    configs = [
        BenchConfig(
            PRODUCTION_CONFIG_LABEL,
            production_config.forward_tile,
            production_config.backward_tile,
            production_config.num_threads,
        ),
        BenchConfig("f128x64_b64x64_t128", (128, 64), (64, 64), 128),
        BenchConfig("f64x128_b64x64_t128", (64, 128), (64, 64), 128),
        BenchConfig("f64x64_b64x64_t128", (64, 64), (64, 64), 128),
        BenchConfig("f128x128_b64x64_t128", (128, 128), (64, 64), 128),
        BenchConfig("f256x32_b64x64_t128", (256, 32), (64, 64), 128),
        BenchConfig("f128x64_b128x64_t128", (128, 64), (128, 64), 128),
        BenchConfig("f128x64_b64x128_t128", (128, 64), (64, 128), 128),
        BenchConfig("f128x64_b128x128_t128", (128, 64), (128, 128), 128),
    ]
    seen: set[tuple[tuple[int, int], tuple[int, int], int]] = set()
    deduplicated = []
    for config in configs:
        key = (config.forward_tile, config.backward_tile, config.num_threads)
        if key in seen:
            continue
        seen.add(key)
        deduplicated.append(config)
    return deduplicated


def thread64_configs() -> list[BenchConfig]:
    return [
        BenchConfig("f128x64_b64x64_t64", (128, 64), (64, 64), 64),
        BenchConfig("f64x128_b64x64_t64", (64, 128), (64, 64), 64),
    ]


def thread_sweep_configs() -> list[BenchConfig]:
    return [BenchConfig(f"f64x128_b64x64_t{threads}", (64, 128), (64, 64), threads) for threads in (64, 256, 384)] + [
        BenchConfig(f"f128x64_b64x64_t{threads}", (128, 64), (64, 64), threads) for threads in (64, 256, 384)
    ]


def host_bf16_normal(shape: tuple[int, ...], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(shape).astype(np.float32)


def causal_lower_bounds(batch: int, seq_len: int, sliding_window: int) -> np.ndarray:
    positions = np.arange(seq_len, dtype=np.int32)
    row = np.maximum(positions - (sliding_window - 1), 0)
    return np.broadcast_to(row[None, :], (batch, seq_len)).copy()


def assert_single_h100_node(devices: np.ndarray) -> None:
    if devices.size != EXPECTED_GPU_COUNT:
        raise RuntimeError(f"Expected one 8-GPU H100 node, got {devices.size} GPU devices: {devices}")
    device_kinds = {getattr(device, "device_kind", "") for device in devices.flat}
    if not all("H100" in kind for kind in device_kinds):
        raise RuntimeError(f"Expected H100 devices, got device kinds: {sorted(device_kinds)}")


def result_stats(times: list[float]) -> tuple[float | None, float | None, float | None, float | None, float | None]:
    if not times:
        return None, None, None, None, None
    median = float(statistics.median(times))
    minimum = float(min(times))
    mean = float(statistics.mean(times))
    stdev = float(statistics.stdev(times)) if len(times) > 1 else 0.0
    cv = stdev / mean if mean > 0 else None
    return median, minimum, mean, stdev, cv


def emit_jsonl(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, sort_keys=True), flush=True)


def make_step(
    *,
    kernel_config: Flash4CuteKernelConfig,
    lower_bounds: jax.Array,
    head_dim: int,
) -> Any:
    def loss_fn(q_arg: jax.Array, k_arg: jax.Array, v_arg: jax.Array, dout_arg: jax.Array) -> jax.Array:
        out = fa4_cute_attention_forward(
            q_arg,
            k_arg,
            v_arg,
            lower_bounds,
            sm_scale=1.0 / np.sqrt(head_dim),
            kernel_config=kernel_config,
        )
        return jnp.sum(out.astype(jnp.float32) * dout_arg.astype(jnp.float32))

    return jax.jit(jax.value_and_grad(loss_fn, argnums=(0, 1, 2)))


def run_batch(
    *,
    mesh: Mesh,
    configs: list[BenchConfig],
    batch: int,
    seq_len: int,
    sliding_window: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    warmup: int,
    iters: int,
    rounds: int,
    seed_offset: int,
    rotate_order: bool,
) -> list[BenchResult]:
    batch_sharding = NamedSharding(mesh, P("expert", None, None, None))
    metadata_sharding = NamedSharding(mesh, P("expert", None))

    q_shape = (batch, seq_len, q_heads, head_dim)
    kv_shape = (batch, seq_len, kv_heads, head_dim)
    q_host = host_bf16_normal(q_shape, seed_offset)
    k_host = host_bf16_normal(kv_shape, seed_offset + 1)
    v_host = host_bf16_normal(kv_shape, seed_offset + 2)
    dout_host = host_bf16_normal(q_shape, seed_offset + 3)
    lower_bounds_host = causal_lower_bounds(batch, seq_len, sliding_window)

    with mesh:
        q = jax.device_put(q_host.astype(jnp.bfloat16), batch_sharding)
        k = jax.device_put(k_host.astype(jnp.bfloat16), batch_sharding)
        v = jax.device_put(v_host.astype(jnp.bfloat16), batch_sharding)
        dout = jax.device_put(dout_host.astype(jnp.bfloat16), batch_sharding)
        lower_bounds = jax.device_put(lower_bounds_host, metadata_sharding)

        compiled_steps: dict[str, Any] = {}
        results: dict[str, BenchResult] = {}
        latest_loss: dict[str, float] = {}

        for config in configs:
            step = make_step(kernel_config=config.kernel_config, lower_bounds=lower_bounds, head_dim=head_dim)
            try:
                compile_start = time.perf_counter()
                value, grads = step(q, k, v, dout)
                jax.block_until_ready((value, grads))
                compile_seconds = time.perf_counter() - compile_start
                for _ in range(warmup):
                    value, grads = step(q, k, v, dout)
                    jax.block_until_ready((value, grads))
                compiled_steps[config.label] = step
                latest_loss[config.label] = float(value)
                emit_jsonl(
                    {
                        "event": "compiled",
                        "batch": batch,
                        "label": config.label,
                        "forward_tile": format_tile(config.forward_tile),
                        "backward_tile": format_tile(config.backward_tile),
                        "num_threads": config.num_threads,
                        "compile_seconds": compile_seconds,
                    }
                )
                results[config.label] = BenchResult(
                    batch=batch,
                    label=config.label,
                    forward_tile=config.forward_tile,
                    backward_tile=config.backward_tile,
                    num_threads=config.num_threads,
                    compile_seconds=compile_seconds,
                    times=[],
                    median_seconds=None,
                    min_seconds=None,
                    mean_seconds=None,
                    stdev_seconds=None,
                    cv=None,
                    loss=None,
                    status="compiled",
                    error=None,
                )
            except Exception as exc:
                result = BenchResult(
                    batch=batch,
                    label=config.label,
                    forward_tile=config.forward_tile,
                    backward_tile=config.backward_tile,
                    num_threads=config.num_threads,
                    compile_seconds=None,
                    times=[],
                    median_seconds=None,
                    min_seconds=None,
                    mean_seconds=None,
                    stdev_seconds=None,
                    cv=None,
                    loss=None,
                    status="failed",
                    error=repr(exc),
                )
                results[config.label] = result
                emit_jsonl({"event": "result", **asdict(result)})

        for round_index in range(rounds):
            round_configs = configs
            if rotate_order and configs:
                offset = round_index % len(configs)
                round_configs = configs[offset:] + configs[:offset]
            for config in round_configs:
                step = compiled_steps.get(config.label)
                if step is None:
                    continue
                for iter_index in range(iters):
                    start = time.perf_counter()
                    value, grads = step(q, k, v, dout)
                    jax.block_until_ready((value, grads))
                    elapsed = time.perf_counter() - start
                    results[config.label].times.append(elapsed)
                    latest_loss[config.label] = float(value)
                    emit_jsonl(
                        {
                            "event": "timing",
                            "batch": batch,
                            "label": config.label,
                            "round": round_index,
                            "iter": iter_index,
                            "seconds": elapsed,
                        }
                    )

        completed_results = []
        for config in configs:
            result = results[config.label]
            if result.status != "failed":
                median, minimum, mean, stdev, cv = result_stats(result.times)
                result = BenchResult(
                    batch=result.batch,
                    label=result.label,
                    forward_tile=result.forward_tile,
                    backward_tile=result.backward_tile,
                    num_threads=result.num_threads,
                    compile_seconds=result.compile_seconds,
                    times=result.times,
                    median_seconds=median,
                    min_seconds=minimum,
                    mean_seconds=mean,
                    stdev_seconds=stdev,
                    cv=cv,
                    loss=latest_loss.get(config.label),
                    status="ok",
                    error=None,
                )
            completed_results.append(result)
            emit_jsonl({"event": "result", **asdict(result)})
        return completed_results


def summarize(results: list[BenchResult]) -> dict[str, Any]:
    by_batch: dict[int, list[BenchResult]] = {}
    for result in results:
        by_batch.setdefault(result.batch, []).append(result)

    summary: dict[str, Any] = {}
    for batch, batch_results in by_batch.items():
        prod = next((result for result in batch_results if result.label == PRODUCTION_CONFIG_LABEL), None)
        prod_median = prod.median_seconds if prod is not None else None
        rows = []
        for result in sorted(
            batch_results,
            key=lambda item: (float("inf") if item.median_seconds is None else item.median_seconds, item.label),
        ):
            speedup = None
            if prod_median is not None and result.median_seconds is not None:
                speedup = prod_median / result.median_seconds
            rows.append({**asdict(result), "speedup_vs_prod": speedup})
        summary[str(batch)] = rows
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="FA4/CuTe single-node sweep for CoreWeave Grug MoE d2560.")
    parser.add_argument("--batch", type=int, action="append", dest="batches")
    parser.add_argument("--seq-len", type=int, default=TARGET_SEQ_LEN)
    parser.add_argument("--sliding-window", type=int, default=TARGET_SLIDING_WINDOW)
    parser.add_argument("--q-heads", type=int, default=TARGET_Q_HEADS)
    parser.add_argument("--kv-heads", type=int, default=TARGET_KV_HEADS)
    parser.add_argument("--head-dim", type=int, default=TARGET_HEAD_DIM)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--config", action="append", default=[])
    parser.add_argument("--include-thread-64", action="store_true")
    parser.add_argument("--include-thread-sweep", action="store_true")
    parser.add_argument("--rotate-order", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--skip-h100-check", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    devices = np.array(jax.local_devices(backend="gpu"))
    if not args.skip_h100_check:
        assert_single_h100_node(devices)

    production_config = flash4_cute_kernel_config(args.head_dim, arch=90)
    configs = [parse_config(config) for config in args.config] if args.config else default_configs(production_config)
    if args.include_thread_64:
        configs.extend(thread64_configs())
    if args.include_thread_sweep:
        configs.extend(thread_sweep_configs())
        seen: set[tuple[tuple[int, int], tuple[int, int], int]] = set()
        deduplicated = []
        for config in configs:
            key = (config.forward_tile, config.backward_tile, config.num_threads)
            if key in seen:
                continue
            seen.add(key)
            deduplicated.append(config)
        configs = deduplicated
    batches = args.batches or [8, 16]

    metadata = {
        "device_count": int(devices.size),
        "device_kinds": sorted({getattr(device, "device_kind", "") for device in devices.flat}),
        "batches": batches,
        "seq_len": args.seq_len,
        "sliding_window": args.sliding_window,
        "q_heads": args.q_heads,
        "kv_heads": args.kv_heads,
        "head_dim": args.head_dim,
        "warmup": args.warmup,
        "iters": args.iters,
        "rounds": args.rounds,
        "rotate_order": args.rotate_order,
        "configs": [asdict(config) for config in configs],
    }
    emit_jsonl({"event": "metadata", **metadata})

    mesh = Mesh(devices.reshape((1, 1, EXPECTED_GPU_COUNT, 1)), ("replica_dcn", "data", "expert", "model"))
    all_results: list[BenchResult] = []
    for batch_index, batch in enumerate(batches):
        try:
            all_results.extend(
                run_batch(
                    mesh=mesh,
                    configs=configs,
                    batch=batch,
                    seq_len=args.seq_len,
                    sliding_window=args.sliding_window,
                    q_heads=args.q_heads,
                    kv_heads=args.kv_heads,
                    head_dim=args.head_dim,
                    warmup=args.warmup,
                    iters=args.iters,
                    rounds=args.rounds,
                    seed_offset=args.seed + batch_index * 10,
                    rotate_order=args.rotate_order,
                )
            )
        except Exception as exc:
            emit_jsonl({"event": "batch_failed", "batch": batch, "error": repr(exc)})

    payload = {
        "metadata": metadata,
        "results": [asdict(result) for result in all_results],
        "summary": summarize(all_results),
    }
    emit_jsonl({"event": "summary", "summary": payload["summary"]})
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    failed_batches = set(batches) - {result.batch for result in all_results}
    if failed_batches:
        sys.exit(2)


if __name__ == "__main__":
    main()
