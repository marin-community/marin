# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import importlib
import json
import os
import subprocess
import time
from collections.abc import Sequence
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

HERO_BASELINE_BATCH = 1024
HERO_BASELINE_SEQ_LEN = 4096
HERO_TOKENS_PER_BATCH = HERO_BASELINE_BATCH * HERO_BASELINE_SEQ_LEN
HERO_QUERY_HEADS = 48
HERO_LOCAL_KV_HEADS = 12
HERO_GLOBAL_KV_HEADS = 6
HERO_HEAD_DIM = 128
HERO_LOCAL_WINDOW = 512

AttentionCase = Literal["local", "global"]
ContextParallelStrategy = Literal["ring", "all_gather"]


@dataclass(frozen=True, slots=True)
class Shape:
    batch: int
    seq_len: int
    query_heads: int
    kv_heads: int
    head_dim: int
    window_size: tuple[int, int]
    segments_per_sequence: int


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    kernel: str
    implementation: str
    shape: dict[str, Any]
    dtype: str
    backend: str
    device_type: str
    device_count: int
    block_sizes: dict[str, Any]
    compile_time: float | None
    steady_state_time: float | None
    error: str | None
    git_sha: str
    xla_flags: str
    backend_env: dict[str, str]
    estimated_forward_backward_tflops: float | None
    estimated_forward_communication_bytes_per_device: int


@dataclass(frozen=True, slots=True)
class TransformerEngineApi:
    te: Any
    AttnBiasType: Any
    AttnMaskType: Any
    AttnSoftmaxType: Any
    CPStrategy: Any
    QKVLayout: Any
    ReorderStrategy: Any
    SequenceDescriptor: Any
    fused_attn: Any
    is_fused_attn_kernel_available: Any
    reorder_causal_load_balancing: Any
    MeshResource: Any


def _load_transformer_engine() -> TransformerEngineApi:
    try:
        te = importlib.import_module("transformer_engine.jax")
        attention = importlib.import_module("transformer_engine.jax.attention")
        sharding = importlib.import_module("transformer_engine.jax.sharding")
    except ImportError as exc:
        raise ImportError(
            "This benchmark requires Transformer Engine with JAX support. "
            "Install a CUDA-compatible transformer_engine[jax] build."
        ) from exc

    return TransformerEngineApi(
        te=te,
        AttnBiasType=attention.AttnBiasType,
        AttnMaskType=attention.AttnMaskType,
        AttnSoftmaxType=attention.AttnSoftmaxType,
        CPStrategy=attention.CPStrategy,
        QKVLayout=attention.QKVLayout,
        ReorderStrategy=attention.ReorderStrategy,
        SequenceDescriptor=attention.SequenceDescriptor,
        fused_attn=attention.fused_attn,
        is_fused_attn_kernel_available=attention.is_fused_attn_kernel_available,
        reorder_causal_load_balancing=attention.reorder_causal_load_balancing,
        MeshResource=sharding.MeshResource,
    )


def _git_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _parse_csv(value: str) -> tuple[str, ...]:
    items = tuple(item.strip() for item in value.split(",") if item.strip())
    if not items:
        raise ValueError("Expected at least one comma-separated value.")
    return items


def _shape(seq_len: int, case: AttentionCase, segments_per_sequence: int) -> Shape:
    if HERO_TOKENS_PER_BATCH % seq_len != 0:
        raise ValueError(f"seq_len={seq_len} must divide the fixed hero token batch {HERO_TOKENS_PER_BATCH}.")
    if seq_len % segments_per_sequence != 0:
        raise ValueError(f"seq_len={seq_len} must be divisible by segments_per_sequence={segments_per_sequence}.")

    if case == "local":
        kv_heads = HERO_LOCAL_KV_HEADS
        window_size = (HERO_LOCAL_WINDOW, 0)
    else:
        kv_heads = HERO_GLOBAL_KV_HEADS
        window_size = (-1, -1)

    return Shape(
        batch=HERO_TOKENS_PER_BATCH // seq_len,
        seq_len=seq_len,
        query_heads=HERO_QUERY_HEADS,
        kv_heads=kv_heads,
        head_dim=HERO_HEAD_DIM,
        window_size=window_size,
        segments_per_sequence=segments_per_sequence,
    )


def _mesh(device_count: int, context_parallel_size: int) -> tuple[Mesh, int]:
    if context_parallel_size <= 1:
        raise ValueError("context_parallel_size must be greater than one.")
    if device_count % context_parallel_size != 0:
        raise ValueError(
            f"device_count={device_count} must be divisible by context_parallel_size={context_parallel_size}."
        )
    if device_count > len(jax.devices()):
        raise ValueError(f"Requested {device_count} devices, but JAX exposes {len(jax.devices())}.")

    data_size = device_count // context_parallel_size
    devices = np.asarray(jax.devices()[:device_count], dtype=object).reshape(data_size, context_parallel_size)
    return Mesh(devices, axis_names=("data", "context")), data_size


def _reordered_random(
    te_api: TransformerEngineApi,
    *,
    key: jax.Array,
    shape: tuple[int, ...],
    dtype: jnp.dtype,
    sharding: NamedSharding,
    context_parallel_size: int,
    stripe_size: int,
) -> jax.Array:
    @jax.jit(out_shardings=sharding)
    def build(random_key: jax.Array) -> jax.Array:
        value = jax.random.normal(random_key, shape, dtype=dtype)
        return te_api.reorder_causal_load_balancing(
            value,
            strategy=te_api.ReorderStrategy.Striped,
            cp_size=context_parallel_size,
            seq_dim=1,
            stripe_size=stripe_size,
        )

    return build(key)


def _sequence_descriptor(
    te_api: TransformerEngineApi,
    *,
    shape: Shape,
    sharding: NamedSharding,
    context_parallel_size: int,
    stripe_size: int,
) -> Any:
    segment_len = shape.seq_len // shape.segments_per_sequence

    @jax.jit(out_shardings=(sharding, sharding))
    def build() -> tuple[jax.Array, jax.Array]:
        token_index = jnp.arange(shape.seq_len, dtype=jnp.int32)
        segment_ids = token_index // segment_len + 1
        segment_positions = token_index % segment_len
        segment_ids = jnp.broadcast_to(segment_ids[None, :], (shape.batch, shape.seq_len))
        segment_positions = jnp.broadcast_to(segment_positions[None, :], (shape.batch, shape.seq_len))
        reordered_ids = te_api.reorder_causal_load_balancing(
            segment_ids,
            strategy=te_api.ReorderStrategy.Striped,
            cp_size=context_parallel_size,
            seq_dim=1,
            stripe_size=stripe_size,
        )
        reordered_positions = te_api.reorder_causal_load_balancing(
            segment_positions,
            strategy=te_api.ReorderStrategy.Striped,
            cp_size=context_parallel_size,
            seq_dim=1,
            stripe_size=stripe_size,
        )
        return reordered_ids, reordered_positions

    segment_ids, segment_positions = build()
    return te_api.SequenceDescriptor.from_segment_ids_and_pos(segment_ids, segment_positions)


def _strategy(te_api: TransformerEngineApi, strategy: ContextParallelStrategy) -> Any:
    if strategy == "ring":
        return te_api.CPStrategy.RING
    if strategy == "all_gather":
        return te_api.CPStrategy.ALL_GATHER
    raise ValueError(f"Unsupported strategy: {strategy}.")


def _stripe_size(strategy: ContextParallelStrategy, all_gather_stripe_size: int) -> int:
    return 1 if strategy == "ring" else all_gather_stripe_size


def _kernel_is_available(te_api: TransformerEngineApi, shape: Shape) -> bool:
    return bool(
        te_api.is_fused_attn_kernel_available(
            True,
            jnp.bfloat16,
            jnp.bfloat16,
            te_api.QKVLayout.THD_THD_THD,
            te_api.AttnBiasType.NO_BIAS,
            te_api.AttnMaskType.PADDING_CAUSAL_MASK,
            te_api.AttnSoftmaxType.VANILLA_SOFTMAX,
            0.0,
            shape.query_heads,
            shape.kv_heads,
            shape.seq_len,
            shape.seq_len,
            shape.head_dim,
            shape.head_dim,
            shape.window_size,
        )
    )


def _time_jitted(function, *args, steps: int, warmup: int) -> tuple[float, float]:
    start = time.perf_counter()
    output = function(*args)
    jax.block_until_ready(output)
    compile_time = time.perf_counter() - start

    for _ in range(warmup):
        output = function(*args)
        jax.block_until_ready(output)

    start = time.perf_counter()
    for _ in range(steps):
        output = function(*args)
        jax.block_until_ready(output)
    steady_state_time = (time.perf_counter() - start) / steps
    return compile_time, steady_state_time


def _attention_pairs(shape: Shape) -> int:
    if shape.window_size == (-1, -1):
        return shape.seq_len * (shape.seq_len + 1) // 2
    window = shape.window_size[0]
    return shape.seq_len * window - window * (window - 1) // 2


def _estimated_forward_backward_flops(shape: Shape) -> float:
    # QK and PV each cost two FLOPs per element in the forward pass. Approximate
    # forward+backward as three forward passes, matching the repository's LM FLOP accounting.
    forward = 4.0 * shape.batch * shape.query_heads * shape.head_dim * _attention_pairs(shape)
    return 3.0 * forward


def _estimated_forward_communication_bytes(
    shape: Shape,
    *,
    data_size: int,
    context_parallel_size: int,
    dtype: jnp.dtype,
) -> int:
    local_batch = shape.batch // data_size
    local_seq = shape.seq_len // context_parallel_size
    local_kv_bytes = 2 * local_batch * local_seq * shape.kv_heads * shape.head_dim * jnp.dtype(dtype).itemsize
    return local_kv_bytes * (context_parallel_size - 1)


def _backend_environment() -> dict[str, str]:
    names = (
        "NVTE_FUSED_ATTN",
        "NVTE_FUSED_RING_ATTENTION_USE_SCAN",
        "XLA_PYTHON_CLIENT_ALLOCATOR",
    )
    return {name: os.environ[name] for name in names if name in os.environ}


def _failure_result(
    shape: Shape,
    *,
    strategy: ContextParallelStrategy,
    stripe_size: int,
    device_count: int,
    error: str,
) -> BenchmarkResult:
    return BenchmarkResult(
        kernel="grug_context_parallel_attention",
        implementation=f"transformer_engine_{strategy}",
        shape=asdict(shape),
        dtype="bfloat16",
        backend=jax.default_backend(),
        device_type=jax.devices()[0].device_kind if jax.devices() else "unknown",
        device_count=device_count,
        block_sizes={"stripe_size": stripe_size},
        compile_time=None,
        steady_state_time=None,
        error=error,
        git_sha=_git_sha(),
        xla_flags=os.environ.get("XLA_FLAGS", ""),
        backend_env=_backend_environment(),
        estimated_forward_backward_tflops=None,
        estimated_forward_communication_bytes_per_device=0,
    )


def _benchmark(
    te_api: TransformerEngineApi,
    shape: Shape,
    *,
    strategy: ContextParallelStrategy,
    mesh: Mesh,
    data_size: int,
    context_parallel_size: int,
    all_gather_stripe_size: int,
    steps: int,
    warmup: int,
) -> BenchmarkResult:
    stripe_size = _stripe_size(strategy, all_gather_stripe_size)
    if shape.batch % data_size != 0:
        return _failure_result(
            shape,
            strategy=strategy,
            stripe_size=stripe_size,
            device_count=mesh.size,
            error=f"batch={shape.batch} is not divisible by data_size={data_size}",
        )
    if shape.seq_len % (2 * context_parallel_size * stripe_size) != 0:
        return _failure_result(
            shape,
            strategy=strategy,
            stripe_size=stripe_size,
            device_count=mesh.size,
            error=(
                f"seq_len={shape.seq_len} is not divisible by the striped causal load-balancing factor "
                f"{2 * context_parallel_size * stripe_size}"
            ),
        )
    if not _kernel_is_available(te_api, shape):
        return _failure_result(
            shape,
            strategy=strategy,
            stripe_size=stripe_size,
            device_count=mesh.size,
            error="Transformer Engine reports no fused-attention backend for this shape",
        )

    q_sharding = NamedSharding(mesh, P("data", "context", None, None))
    kv_sharding = NamedSharding(mesh, P("data", "context", None, None))
    metadata_sharding = NamedSharding(mesh, P("data", "context"))
    q_shape = (shape.batch, shape.seq_len, shape.query_heads, shape.head_dim)
    kv_shape = (shape.batch, shape.seq_len, shape.kv_heads, shape.head_dim)
    q_key, k_key, v_key, dout_key = jax.random.split(jax.random.PRNGKey(0), 4)

    q = _reordered_random(
        te_api,
        key=q_key,
        shape=q_shape,
        dtype=jnp.bfloat16,
        sharding=q_sharding,
        context_parallel_size=context_parallel_size,
        stripe_size=stripe_size,
    )
    k = _reordered_random(
        te_api,
        key=k_key,
        shape=kv_shape,
        dtype=jnp.bfloat16,
        sharding=kv_sharding,
        context_parallel_size=context_parallel_size,
        stripe_size=stripe_size,
    )
    v = _reordered_random(
        te_api,
        key=v_key,
        shape=kv_shape,
        dtype=jnp.bfloat16,
        sharding=kv_sharding,
        context_parallel_size=context_parallel_size,
        stripe_size=stripe_size,
    )
    dout = _reordered_random(
        te_api,
        key=dout_key,
        shape=q_shape,
        dtype=jnp.bfloat16,
        sharding=q_sharding,
        context_parallel_size=context_parallel_size,
        stripe_size=stripe_size,
    )
    sequence_descriptor = _sequence_descriptor(
        te_api,
        shape=shape,
        sharding=metadata_sharding,
        context_parallel_size=context_parallel_size,
        stripe_size=stripe_size,
    )
    sequence_descriptor_shardings = jax.tree.map(lambda value: value.sharding, sequence_descriptor)
    qkv = (q, k, v)
    qkv_shardings = (q_sharding, kv_sharding, kv_sharding)

    def loss_fn(qkv_arg, descriptor_arg, dout_arg):
        output = te_api.fused_attn(
            qkv_arg,
            None,
            descriptor_arg,
            None,
            attn_bias_type=te_api.AttnBiasType.NO_BIAS,
            attn_mask_type=te_api.AttnMaskType.PADDING_CAUSAL_MASK,
            qkv_layout=te_api.QKVLayout.THD_THD_THD,
            softmax_type=te_api.AttnSoftmaxType.VANILLA_SOFTMAX,
            scaling_factor=shape.head_dim**-0.5,
            dropout_probability=0.0,
            is_training=True,
            max_segments_per_seq=shape.segments_per_sequence,
            window_size=shape.window_size,
            context_parallel_strategy=_strategy(te_api, strategy),
            context_parallel_causal_load_balanced=True,
            context_parallel_axis="context",
            stripe_size=stripe_size,
        )
        return jnp.vdot(output.astype(jnp.float32), dout_arg.astype(jnp.float32))

    function = jax.jit(
        jax.value_and_grad(loss_fn),
        in_shardings=(qkv_shardings, sequence_descriptor_shardings, q_sharding),
        out_shardings=(None, qkv_shardings),
    )
    mesh_resource = te_api.MeshResource(dp_resource="data", cp_resource="context")
    with jax.set_mesh(mesh), te_api.te.autocast(mesh_resource=mesh_resource):
        compile_time, steady_state_time = _time_jitted(
            function,
            qkv,
            sequence_descriptor,
            dout,
            steps=steps,
            warmup=warmup,
        )

    estimated_flops = _estimated_forward_backward_flops(shape)
    return BenchmarkResult(
        kernel="grug_context_parallel_attention",
        implementation=f"transformer_engine_{strategy}",
        shape=asdict(shape),
        dtype="bfloat16",
        backend=jax.default_backend(),
        device_type=jax.devices()[0].device_kind,
        device_count=mesh.size,
        block_sizes={"stripe_size": stripe_size},
        compile_time=compile_time,
        steady_state_time=steady_state_time,
        error=None,
        git_sha=_git_sha(),
        xla_flags=os.environ.get("XLA_FLAGS", ""),
        backend_env=_backend_environment(),
        estimated_forward_backward_tflops=estimated_flops / steady_state_time / 1e12,
        estimated_forward_communication_bytes_per_device=_estimated_forward_communication_bytes(
            shape,
            data_size=data_size,
            context_parallel_size=context_parallel_size,
            dtype=jnp.bfloat16,
        ),
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark Transformer Engine context-parallel attention at Grug hero shapes."
    )
    parser.add_argument("--seq-lens", default="4096")
    parser.add_argument("--cases", default="local,global", help="Comma-separated: local,global")
    parser.add_argument("--strategies", default="ring,all_gather", help="Comma-separated: ring,all_gather")
    parser.add_argument("--context-parallel-size", type=int, default=4)
    parser.add_argument("--device-count", type=int, default=0, help="Zero uses every visible device.")
    parser.add_argument("--segments-per-sequence", type=int, default=1)
    parser.add_argument("--all-gather-stripe-size", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=5)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    os.environ.setdefault("NVTE_FUSED_RING_ATTENTION_USE_SCAN", "0")
    te_api = _load_transformer_engine()
    device_count = args.device_count or len(jax.devices())
    mesh, data_size = _mesh(device_count, args.context_parallel_size)

    cases = _parse_csv(args.cases)
    unknown_cases = set(cases) - {"local", "global"}
    if unknown_cases:
        raise ValueError(f"Unknown attention cases: {sorted(unknown_cases)}")
    strategies = _parse_csv(args.strategies)
    unknown_strategies = set(strategies) - {"ring", "all_gather"}
    if unknown_strategies:
        raise ValueError(f"Unknown context-parallel strategies: {sorted(unknown_strategies)}")

    seq_lens = tuple(int(value) for value in _parse_csv(args.seq_lens))
    for seq_len in seq_lens:
        for case in cases:
            shape = _shape(seq_len, case, args.segments_per_sequence)
            for strategy in strategies:
                try:
                    result = _benchmark(
                        te_api,
                        shape,
                        strategy=strategy,
                        mesh=mesh,
                        data_size=data_size,
                        context_parallel_size=args.context_parallel_size,
                        all_gather_stripe_size=args.all_gather_stripe_size,
                        steps=args.steps,
                        warmup=args.warmup,
                    )
                except Exception as exc:  # benchmark rows must retain unsupported compile/runtime failures
                    result = _failure_result(
                        shape,
                        strategy=strategy,
                        stripe_size=_stripe_size(strategy, args.all_gather_stripe_size),
                        device_count=device_count,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                if jax.process_index() == 0:
                    print(json.dumps(asdict(result), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
