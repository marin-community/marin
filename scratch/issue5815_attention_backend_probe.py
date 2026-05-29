# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import argparse
import importlib.metadata
import math
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import levanter.grug.attention._fa4_cute as fa4_cute_module
import levanter.grug.attention._fa4_thd as fa4_thd_module
from jax._src.cudnn import fused_attention_stablehlo as cudnn_fa
from levanter.grug.attention import AttentionMask
from levanter.grug.attention._fa4_cute import gpu_fa4_cute_attention
from levanter.grug.attention._fa4_cute_backend import fa4_cute_attention_forward
from levanter.grug.attention._fa4_cute_config import Flash4CuteKernelConfig
from levanter.grug.attention._fa4_thd import torch_fa4_thd_attention

ORIGINAL_FA4_CUTE_CONFIG = fa4_cute_module.flash4_cute_kernel_config


@dataclass(frozen=True)
class ProbeConfig:
    hidden_dim: int
    batch: int
    seq_len: int
    q_heads: int
    kv_heads: int
    head_dim: int
    segment_len: int
    segments_per_row: int | None
    warmup: int
    iters: int
    fa4_forward_tile: tuple[int, int] | None
    fa4_backward_tile: tuple[int, int] | None
    fa4_num_threads: int | None


@dataclass(frozen=True)
class BenchmarkResult:
    name: str
    times: tuple[float, ...]
    error: str | None = None

    @property
    def median(self) -> float:
        return statistics.median(self.times)


@dataclass(frozen=True)
class SegmentLayout:
    ids: tuple[tuple[int, ...], ...]
    gather_indices: tuple[tuple[int, ...], ...]
    row_lengths: tuple[tuple[int, ...], ...]
    lengths: tuple[int, ...]
    max_len: int

    @property
    def num_segments(self) -> int:
        return len(self.lengths)


@dataclass(frozen=True)
class ShapeSpec:
    hidden_dim: int
    q_heads: int
    kv_heads: int
    head_dim: int

    @property
    def label(self) -> str:
        return f"d{self.hidden_dim}"


def _block(value):
    return jax.block_until_ready(value)


def _benchmark_jax(name: str, fn: Callable[[], object], *, warmup: int, iters: int) -> BenchmarkResult:
    try:
        for _ in range(warmup):
            _block(fn())
        times = []
        for _ in range(iters):
            start = time.perf_counter()
            _block(fn())
            times.append(time.perf_counter() - start)
        return BenchmarkResult(name=name, times=tuple(times))
    except Exception as exc:
        return BenchmarkResult(name=name, times=(), error=f"{type(exc).__name__}: {exc}")


def _benchmark_torch(name: str, fn: Callable[[], object], *, warmup: int, iters: int) -> BenchmarkResult:
    try:
        import torch

        for _ in range(warmup):
            fn()
            torch.cuda.synchronize()
        times = []
        for _ in range(iters):
            start = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
        return BenchmarkResult(name=name, times=tuple(times))
    except Exception as exc:
        return BenchmarkResult(name=name, times=(), error=f"{type(exc).__name__}: {exc}")


def _flash_attn_dense_func():
    try:
        import flash_attn.cute as flash_attn_cute

        return flash_attn_cute.flash_attn_func
    except (ImportError, AttributeError):
        try:
            from flash_attn.cute import interface as flash_attn_cute_interface

            return flash_attn_cute_interface.flash_attn_func
        except (ImportError, AttributeError) as exc:
            raise RuntimeError(
                "upstream_fa4_dense_core requires flash-attn-4 with " "`flash_attn.cute.flash_attn_func`."
            ) from exc


def _flash_attn_internal_funcs():
    try:
        import flash_attn.cute as flash_attn_cute

        return flash_attn_cute._flash_attn_fwd, flash_attn_cute._flash_attn_bwd
    except (ImportError, AttributeError):
        try:
            from flash_attn.cute import interface as flash_attn_cute_interface

            return flash_attn_cute_interface._flash_attn_fwd, flash_attn_cute_interface._flash_attn_bwd
        except (ImportError, AttributeError) as exc:
            raise RuntimeError(
                "upstream FA4 internal fwd+bwd benchmarking requires "
                "`flash_attn.cute.interface._flash_attn_fwd/_flash_attn_bwd`."
            ) from exc


def _jax_inputs(config: ProbeConfig):
    key = jax.random.PRNGKey(5815)
    q_key, k_key, v_key, do_key = jax.random.split(key, 4)
    q_shape = (config.batch, config.seq_len, config.q_heads, config.head_dim)
    kv_shape = (config.batch, config.seq_len, config.kv_heads, config.head_dim)
    q = jax.random.normal(q_key, q_shape, dtype=jnp.bfloat16)
    k = jax.random.normal(k_key, kv_shape, dtype=jnp.bfloat16)
    v = jax.random.normal(v_key, kv_shape, dtype=jnp.bfloat16)
    dout = jax.random.normal(do_key, q_shape, dtype=jnp.bfloat16)
    segment_ids = _jax_segment_ids(config)
    mask = AttentionMask.causal().with_segment_ids(segment_ids)
    return q, k, v, dout, mask


def _torch_inputs(config: ProbeConfig):
    import torch

    device = torch.device("cuda")
    generator = torch.Generator(device=device)
    generator.manual_seed(5815)
    q_shape = (config.batch, config.seq_len, config.q_heads, config.head_dim)
    kv_shape = (config.batch, config.seq_len, config.kv_heads, config.head_dim)
    q = torch.randn(q_shape, device=device, dtype=torch.bfloat16, generator=generator, requires_grad=True)
    k = torch.randn(kv_shape, device=device, dtype=torch.bfloat16, generator=generator, requires_grad=True)
    v = torch.randn(kv_shape, device=device, dtype=torch.bfloat16, generator=generator, requires_grad=True)
    dout = torch.randn(q_shape, device=device, dtype=torch.bfloat16, generator=generator)
    segment_ids = _torch_segment_ids(config)
    return q, k, v, dout, segment_ids


def _jax_segment_ids(config: ProbeConfig):
    return jnp.array(_segment_layout(config).ids, dtype=jnp.int32)


def _torch_segment_ids(config: ProbeConfig):
    import torch

    return torch.tensor(_segment_layout(config).ids, device="cuda", dtype=torch.int32)


def _torch_segment_length_metadata(config: ProbeConfig):
    import torch

    layout = _segment_layout(config)
    max_segments = max(len(row) for row in layout.row_lengths)
    segment_lengths = [list(row) + [0] * (max_segments - len(row)) for row in layout.row_lengths]
    num_segments = [len(row) for row in layout.row_lengths]
    return (
        torch.tensor(segment_lengths, device="cuda", dtype=torch.int32),
        torch.tensor(num_segments, device="cuda", dtype=torch.int32),
    )


def _segment_lengths(config: ProbeConfig, *, row: int) -> list[int]:
    if config.segments_per_row is None:
        if config.seq_len % config.segment_len != 0:
            raise ValueError("seq_len must be divisible by segment_len for equal segment layout.")
        return [config.segment_len] * (config.seq_len // config.segment_len)

    segments = config.segments_per_row
    if segments <= 0:
        raise ValueError(f"segments_per_row must be positive, got {segments}.")
    weights = list(range(1, segments + 1))
    shift = row % segments
    weights = weights[shift:] + weights[:shift]
    total_weight = sum(weights)
    lengths = [max(1, config.seq_len * weight // total_weight) for weight in weights[:-1]]
    lengths.append(config.seq_len - sum(lengths))
    if lengths[-1] <= 0:
        raise ValueError(f"segments_per_row={segments} creates a non-positive final segment for S={config.seq_len}.")
    return lengths


def _segment_layout(config: ProbeConfig) -> SegmentLayout:
    ids: list[list[int]] = []
    gather_indices: list[list[int]] = []
    row_lengths: list[tuple[int, ...]] = []
    lengths: list[int] = []
    segment_id = 0
    for batch_index in range(config.batch):
        row_ids: list[int] = []
        cursor = 0
        segment_lengths = _segment_lengths(config, row=batch_index)
        row_lengths.append(tuple(segment_lengths))
        for length in segment_lengths:
            row_ids.extend([segment_id] * length)
            gather_indices.append([batch_index * config.seq_len + cursor + offset for offset in range(length)])
            lengths.append(length)
            cursor += length
            segment_id += 1
        if cursor != config.seq_len:
            raise ValueError(f"segment lengths must sum to S={config.seq_len}, got {cursor}.")
        ids.append(row_ids)

    max_len = max(lengths)
    padded_indices = tuple(tuple(segment + [-1] * (max_len - len(segment))) for segment in gather_indices)
    return SegmentLayout(
        ids=tuple(tuple(row) for row in ids),
        gather_indices=padded_indices,
        row_lengths=tuple(row_lengths),
        lengths=tuple(lengths),
        max_len=max_len,
    )


def _pack_segments_jax(x: jax.Array, layout: SegmentLayout) -> jax.Array:
    indices = jnp.array(layout.gather_indices, dtype=jnp.int32)
    valid = indices >= 0
    flat = x.reshape(configured_flat_tokens(x), x.shape[2], x.shape[3])
    packed = flat[jnp.maximum(indices, 0)]
    return jnp.where(valid[..., None, None], packed, jnp.zeros((), dtype=x.dtype))


def configured_flat_tokens(x: jax.Array) -> int:
    return x.shape[0] * x.shape[1]


def _jax_cudnn_lengths_and_offsets(layout: SegmentLayout) -> tuple[jax.Array, jax.Array]:
    max_segments = max(len(row) for row in layout.row_lengths)
    seqlens = []
    offsets = []
    for row in layout.row_lengths:
        row_seqlens = list(row) + [-1] * (max_segments - len(row))
        row_offsets = []
        cursor = 0
        for length in row:
            row_offsets.append(cursor)
            cursor += length
        row_offsets.extend([-1] * (max_segments + 1 - len(row_offsets)))
        seqlens.append(row_seqlens)
        offsets.append(row_offsets)

    return jnp.array(seqlens, dtype=jnp.int32), jnp.array(offsets, dtype=jnp.int32)


def _jax_cudnn_core(config: ProbeConfig) -> BenchmarkResult:
    q, k, v, dout, _ = _jax_inputs(config)
    layout = _segment_layout(config)
    lengths = jnp.array(layout.lengths, dtype=jnp.int32)
    dout = _pack_segments_jax(dout, layout)

    def loss(q_arg, k_arg, v_arg):
        q_packed = _pack_segments_jax(q_arg, layout)
        k_packed = _pack_segments_jax(k_arg, layout)
        v_packed = _pack_segments_jax(v_arg, layout)
        out = jax.nn.dot_product_attention(
            q_packed,
            k_packed,
            v_packed,
            is_causal=True,
            query_seq_lengths=lengths,
            key_value_seq_lengths=lengths,
            implementation="cudnn",
        )
        return jnp.sum(out.astype(jnp.float32) * dout.astype(jnp.float32))

    grad_fn = jax.jit(jax.grad(loss, argnums=(0, 1, 2)))
    return _benchmark_jax("jax_cudnn_core", lambda: grad_fn(q, k, v), warmup=config.warmup, iters=config.iters)


def _jax_cudnn_dense_core(config: ProbeConfig) -> BenchmarkResult:
    q, k, v, dout, _ = _jax_inputs(config)

    def loss(q_arg, k_arg, v_arg):
        out = jax.nn.dot_product_attention(
            q_arg,
            k_arg,
            v_arg,
            is_causal=True,
            implementation="cudnn",
        )
        return jnp.sum(out.astype(jnp.float32) * dout.astype(jnp.float32))

    grad_fn = jax.jit(jax.grad(loss, argnums=(0, 1, 2)))
    return _benchmark_jax("jax_cudnn_dense_core", lambda: grad_fn(q, k, v), warmup=config.warmup, iters=config.iters)


def _jax_cudnn_offsets_core(config: ProbeConfig) -> BenchmarkResult:
    q, k, v, dout, _ = _jax_inputs(config)
    layout = _segment_layout(config)
    seqlens, offsets = _jax_cudnn_lengths_and_offsets(layout)

    def loss(q_arg, k_arg, v_arg):
        out = cudnn_fa.dot_product_attention(
            q_arg,
            k_arg,
            v_arg,
            q_seqlen=seqlens,
            kv_seqlen=seqlens,
            q_offsets=offsets,
            kv_offsets=offsets,
            scale=1.0 / math.sqrt(config.head_dim),
            mask_type=cudnn_fa.MaskType.PADDING_CAUSAL,
            qkv_layout="BTNH",
        )
        return jnp.sum(out.astype(jnp.float32) * dout.astype(jnp.float32))

    grad_fn = jax.jit(jax.grad(loss, argnums=(0, 1, 2)))
    return _benchmark_jax("jax_cudnn_offsets_core", lambda: grad_fn(q, k, v), warmup=config.warmup, iters=config.iters)


def _fa4_cute_core(config: ProbeConfig) -> BenchmarkResult:
    q, k, v, dout, mask = _jax_inputs(config)

    def loss(q_arg, k_arg, v_arg):
        out = gpu_fa4_cute_attention(q_arg, k_arg, v_arg, mask)
        return jnp.sum(out.astype(jnp.float32) * dout.astype(jnp.float32))

    grad_fn = jax.jit(jax.grad(loss, argnums=(0, 1, 2)))
    return _benchmark_jax("fa4_cute_core", lambda: grad_fn(q, k, v), warmup=config.warmup, iters=config.iters)


def _fa4_cute_precomputed_core(config: ProbeConfig) -> BenchmarkResult:
    q, k, v, dout, mask = _jax_inputs(config)
    assert isinstance(mask, AttentionMask)
    q_segment_ids = fa4_cute_module._packed_self_attention_segment_ids(
        q,
        k,
        mask,
        backend_name="fa4_cute_precomputed_core",
    )
    lower_bounds, valid = fa4_cute_module._packed_segment_causal_lower_bounds(
        q_segment_ids,
        batch_size=q.shape[0],
        seq_len=q.shape[1],
        sliding_window=mask.sliding_window,
    )
    kernel_config = fa4_cute_module.flash4_cute_kernel_config(q.shape[-1], arch=fa4_cute_module._gpu_compute_arch())

    def loss(q_arg, k_arg, v_arg):
        out = fa4_cute_attention_forward(
            q_arg,
            k_arg,
            v_arg,
            lower_bounds,
            valid,
            sm_scale=1.0 / math.sqrt(q.shape[-1]),
            kernel_config=kernel_config,
        )
        return jnp.sum(out.astype(jnp.float32) * dout.astype(jnp.float32))

    grad_fn = jax.jit(jax.grad(loss, argnums=(0, 1, 2)))
    return _benchmark_jax(
        "fa4_cute_precomputed_core", lambda: grad_fn(q, k, v), warmup=config.warmup, iters=config.iters
    )


def _install_fa4_cute_override(config: ProbeConfig) -> None:
    if config.fa4_forward_tile is None and config.fa4_backward_tile is None and config.fa4_num_threads is None:
        return

    def override(head_dim: int, *, arch: int) -> Flash4CuteKernelConfig:
        base = ORIGINAL_FA4_CUTE_CONFIG(head_dim, arch=arch)
        return Flash4CuteKernelConfig(
            forward_tile=config.fa4_forward_tile or base.forward_tile,
            backward_tile=config.fa4_backward_tile or base.backward_tile,
            num_threads=config.fa4_num_threads or base.num_threads,
        )

    fa4_cute_module.flash4_cute_kernel_config = override


def _fa4_thd_core(config: ProbeConfig) -> BenchmarkResult:
    q, k, v, dout, segment_ids = _torch_inputs(config)

    def step():
        q.grad = None
        k.grad = None
        v.grad = None
        out = torch_fa4_thd_attention(q, k, v, (segment_ids, segment_ids))
        torch_loss = (out.float() * dout.float()).sum()
        torch_loss.backward()
        return q.grad, k.grad, v.grad

    return _benchmark_torch("fa4_thd_core", step, warmup=config.warmup, iters=config.iters)


def _fa4_thd_internal_core(config: ProbeConfig) -> BenchmarkResult:
    q, k, v, dout, segment_ids = _torch_inputs(config)

    def step():
        return fa4_thd_module.torch_fa4_thd_attention_internal_fwd_bwd(
            q,
            k,
            v,
            dout,
            (segment_ids, segment_ids),
        )

    return _benchmark_torch("fa4_thd_internal_core", step, warmup=config.warmup, iters=config.iters)


def _fa4_thd_cu_lens_core(config: ProbeConfig) -> BenchmarkResult:
    q, k, v, dout, segment_ids = _torch_inputs(config)
    batch_size, seq_len, _, _ = q.shape
    cu_seqlens, max_seqlen = fa4_thd_module._torch_thd_reshape_metadata(
        segment_ids,
        segment_ids,
        batch_size=batch_size,
        seq_len=seq_len,
    )

    def step():
        return fa4_thd_module.torch_fa4_thd_cu_lens_fwd_bwd(
            q,
            k,
            v,
            dout,
            cu_seqlens,
            max_seqlen,
        )

    return _benchmark_torch("fa4_thd_cu_lens_core", step, warmup=config.warmup, iters=config.iters)


def _fa4_thd_lengths_core(config: ProbeConfig) -> BenchmarkResult:
    q, k, v, dout, _ = _torch_inputs(config)
    segment_lengths, num_segments = _torch_segment_length_metadata(config)

    def step():
        return fa4_thd_module.torch_fa4_thd_attention_internal_fwd_bwd_from_lengths(
            q,
            k,
            v,
            dout,
            segment_lengths,
            num_segments,
        )

    return _benchmark_torch("fa4_thd_lengths_core", step, warmup=config.warmup, iters=config.iters)


def _upstream_fa4_dense_core(config: ProbeConfig) -> BenchmarkResult:
    q, k, v, dout, _ = _torch_inputs(config)
    flash_attn_func = _flash_attn_dense_func()

    def step():
        q.grad = None
        k.grad = None
        v.grad = None
        out = flash_attn_func(
            q,
            k,
            v,
            softmax_scale=1.0 / math.sqrt(config.head_dim),
            causal=True,
        )
        if isinstance(out, tuple):
            out = out[0]
        torch_loss = (out.float() * dout.float()).sum()
        torch_loss.backward()
        return q.grad, k.grad, v.grad

    return _benchmark_torch("upstream_fa4_dense_core", step, warmup=config.warmup, iters=config.iters)


def _upstream_fa4_dense_internal_core(config: ProbeConfig) -> BenchmarkResult:
    q, k, v, dout, _ = _torch_inputs(config)
    flash_attn_fwd, flash_attn_bwd = _flash_attn_internal_funcs()

    def step():
        out, lse = flash_attn_fwd(
            q,
            k,
            v,
            softmax_scale=1.0 / math.sqrt(config.head_dim),
            causal=True,
            return_lse=True,
        )
        return flash_attn_bwd(
            q,
            k,
            v,
            out,
            dout,
            lse,
            softmax_scale=1.0 / math.sqrt(config.head_dim),
            causal=True,
        )

    return _benchmark_torch("upstream_fa4_dense_internal_core", step, warmup=config.warmup, iters=config.iters)


def _print_environment(config: ProbeConfig) -> None:
    layout = _segment_layout(config)
    unique_lengths = sorted(set(layout.lengths))
    print(
        "shape "
        f"B={config.batch} S={config.seq_len} Hq={config.q_heads} Hkv={config.kv_heads} D={config.head_dim} "
        f"d_model={config.hidden_dim} segment_len={config.segment_len} "
        f"segments={layout.num_segments} max_segment_len={layout.max_len} unique_segment_lengths={unique_lengths}"
    )
    if config.fa4_forward_tile is not None or config.fa4_backward_tile is not None or config.fa4_num_threads is not None:
        print(
            "fa4_cute_override "
            f"forward_tile={config.fa4_forward_tile} backward_tile={config.fa4_backward_tile} "
            f"num_threads={config.fa4_num_threads}"
        )
    print(f"jax={jax.__version__} backend={jax.default_backend()}")
    try:
        print(f"flash-attn-4={importlib.metadata.version('flash-attn-4')}")
    except importlib.metadata.PackageNotFoundError:
        print("flash-attn-4=not-installed")
    try:
        import torch

        device_name = torch.cuda.get_device_name() if torch.cuda.is_available() else "cuda-unavailable"
        print(f"torch={torch.__version__} device={device_name}")
    except Exception as exc:
        print(f"torch=unavailable ({type(exc).__name__}: {exc})")
    print("jax_cudnn_offsets_api=jax._src.cudnn.fused_attention_stablehlo.dot_product_attention")
    print("upstream_fa4_dense_api=flash_attn.cute.flash_attn_func")
    print("upstream_fa4_dense_internal_api=flash_attn.cute.interface._flash_attn_fwd/_flash_attn_bwd")
    print("fa4_thd_api=flash_attn.cute.flash_attn_varlen_func")


def _print_result(result: BenchmarkResult) -> None:
    if result.error is not None:
        print(f"{result.name}: ERROR {result.error}", flush=True)
        return
    times = ", ".join(f"{t:.6f}" for t in result.times)
    print(f"{result.name}: median={result.median:.6f}s times=[{times}]", flush=True)


def _print_goal_comparison(results: list[BenchmarkResult]) -> None:
    by_name = {result.name: result for result in results}
    baseline_names = ("upstream_fa4_dense_internal_core", "upstream_fa4_dense_core", "jax_cudnn_dense_core")
    candidate_names = (
        "fa4_cute_core",
        "fa4_thd_core",
        "fa4_thd_internal_core",
        "fa4_thd_cu_lens_core",
        "fa4_thd_lengths_core",
    )
    baselines = [
        result
        for name in baseline_names
        if (result := by_name.get(name)) is not None and result.error is None and result.times
    ]
    candidates = [
        result
        for name in candidate_names
        if (result := by_name.get(name)) is not None and result.error is None and result.times
    ]
    if not baselines or not candidates:
        return

    baseline = min(baselines, key=lambda result: result.median)
    target = baseline.median * 1.10
    print(
        f"goal_baseline: {baseline.name} median={baseline.median:.6f}s target_10pct={target:.6f}s",
        flush=True,
    )
    for candidate in candidates:
        ratio = candidate.median / baseline.median
        status = "PASS" if candidate.median <= target else "MISS"
        print(
            f"goal_candidate: {candidate.name} median={candidate.median:.6f}s " f"ratio={ratio:.3f} status={status}",
            flush=True,
        )


def _parse_tile(value: str | None) -> tuple[int, int] | None:
    if value is None:
        return None
    parts = value.lower().replace("x", ",").split(",")
    if len(parts) != 2:
        raise ValueError(f"Tile must be formatted as M,N or MxN, got {value!r}")
    return int(parts[0]), int(parts[1])


BACKENDS: dict[str, Callable[[ProbeConfig], BenchmarkResult]] = {
    "upstream_fa4_dense_core": _upstream_fa4_dense_core,
    "upstream_fa4_dense_internal_core": _upstream_fa4_dense_internal_core,
    "jax_cudnn_dense_core": _jax_cudnn_dense_core,
    "jax_cudnn_core": _jax_cudnn_core,
    "jax_cudnn_offsets_core": _jax_cudnn_offsets_core,
    "fa4_cute_core": _fa4_cute_core,
    "fa4_cute_precomputed_core": _fa4_cute_precomputed_core,
    "fa4_thd_core": _fa4_thd_core,
    "fa4_thd_internal_core": _fa4_thd_internal_core,
    "fa4_thd_cu_lens_core": _fa4_thd_cu_lens_core,
    "fa4_thd_lengths_core": _fa4_thd_lengths_core,
}


def _compute_kv_heads(q_heads: int, gqa_ratio: int | None) -> int:
    if gqa_ratio is None:
        return q_heads
    target = q_heads // gqa_ratio
    for kv_heads in range(target, 0, -1):
        if q_heads % kv_heads == 0:
            return kv_heads
    return 1


def _shape_specs(
    *,
    shape_specs: str | None,
    model_dims: str | None,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    gqa_ratio: int | None,
) -> list[ShapeSpec]:
    if shape_specs is not None:
        specs = []
        for part in shape_specs.split(","):
            fields = part.strip().removeprefix("d").split(":")
            if len(fields) not in (3, 4):
                raise ValueError(
                    "--shape-specs entries must be hidden_dim:q_heads:kv_heads[:head_dim], " f"got {part!r}."
                )
            spec_hidden_dim = int(fields[0])
            spec_q_heads = int(fields[1])
            spec_kv_heads = int(fields[2])
            spec_head_dim = int(fields[3]) if len(fields) == 4 else head_dim
            if spec_q_heads * spec_head_dim != spec_hidden_dim:
                raise ValueError(
                    f"shape spec {part!r} is inconsistent: q_heads * head_dim = "
                    f"{spec_q_heads * spec_head_dim}, expected {spec_hidden_dim}."
                )
            specs.append(
                ShapeSpec(
                    hidden_dim=spec_hidden_dim,
                    q_heads=spec_q_heads,
                    kv_heads=spec_kv_heads,
                    head_dim=spec_head_dim,
                )
            )
        return specs

    if model_dims is None:
        return [
            ShapeSpec(
                hidden_dim=q_heads * head_dim,
                q_heads=q_heads,
                kv_heads=kv_heads,
                head_dim=head_dim,
            )
        ]

    specs = []
    for part in model_dims.split(","):
        hidden_dim = int(part.strip().removeprefix("d"))
        if hidden_dim % head_dim != 0:
            raise ValueError(f"model dim {hidden_dim} must be divisible by head_dim={head_dim}.")
        dim_q_heads = hidden_dim // head_dim
        dim_kv_heads = _compute_kv_heads(dim_q_heads, gqa_ratio)
        specs.append(
            ShapeSpec(
                hidden_dim=hidden_dim,
                q_heads=dim_q_heads,
                kv_heads=dim_kv_heads,
                head_dim=head_dim,
            )
        )
    return specs


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe issue5815 attention backend fwd+bwd timings.")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--batches", type=str, default=None, help="Comma-separated batch sizes for a sweep.")
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument(
        "--shape-specs",
        type=str,
        default=None,
        help=(
            "Comma-separated explicit specs hidden_dim:q_heads:kv_heads[:head_dim], "
            "for example 1024:8:2,2560:20:4,5120:40:8."
        ),
    )
    parser.add_argument(
        "--model-dims",
        type=str,
        default=None,
        help=(
            "Comma-separated hidden dims such as 1024,2560,5120. "
            "When set, q_heads=hidden_dim/head_dim and kv_heads is derived from --gqa-ratio."
        ),
    )
    parser.add_argument(
        "--gqa-ratio",
        type=int,
        default=5,
        help="GQA ratio used with --model-dims. Default 5 matches the exact d5120 40q/8kv shape.",
    )
    parser.add_argument("--q-heads", type=int, default=40)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--segment-len", type=int, default=1024)
    parser.add_argument(
        "--segments-per-row",
        type=int,
        default=None,
        help="Use variable contiguous segment lengths with this many segments per row.",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--fa4-forward-tile", type=str, default=None)
    parser.add_argument("--fa4-backward-tile", type=str, default=None)
    parser.add_argument("--fa4-num-threads", type=int, default=None)
    parser.add_argument(
        "--backends",
        type=str,
        default=(
            "upstream_fa4_dense_internal_core,upstream_fa4_dense_core,jax_cudnn_dense_core,"
            "jax_cudnn_core,fa4_cute_core,fa4_thd_lengths_core,fa4_thd_cu_lens_core,fa4_thd_internal_core,fa4_thd_core"
        ),
        help=f"Comma-separated backend list. Available: {','.join(BACKENDS)}",
    )
    args = parser.parse_args()
    batches = [args.batch] if args.batches is None else [int(part) for part in args.batches.split(",")]
    shape_specs = _shape_specs(
        shape_specs=args.shape_specs,
        model_dims=args.model_dims,
        q_heads=args.q_heads,
        kv_heads=args.kv_heads,
        head_dim=args.head_dim,
        gqa_ratio=args.gqa_ratio,
    )
    backends = [part.strip() for part in args.backends.split(",") if part.strip()]
    unknown_backends = sorted(set(backends) - set(BACKENDS))
    if unknown_backends:
        raise ValueError(f"Unknown backends: {unknown_backends}. Available: {sorted(BACKENDS)}")

    for shape_spec in shape_specs:
        for batch in batches:
            config = ProbeConfig(
                hidden_dim=shape_spec.hidden_dim,
                batch=batch,
                seq_len=args.seq_len,
                q_heads=shape_spec.q_heads,
                kv_heads=shape_spec.kv_heads,
                head_dim=shape_spec.head_dim,
                segment_len=args.segment_len,
                segments_per_row=args.segments_per_row,
                warmup=args.warmup,
                iters=args.iters,
                fa4_forward_tile=_parse_tile(args.fa4_forward_tile),
                fa4_backward_tile=_parse_tile(args.fa4_backward_tile),
                fa4_num_threads=args.fa4_num_threads,
            )
            _install_fa4_cute_override(config)
            _print_environment(config)
            results = []
            for backend in backends:
                result = BACKENDS[backend](config)
                _print_result(result)
                results.append(result)
            _print_goal_comparison(results)


if __name__ == "__main__":
    main()
