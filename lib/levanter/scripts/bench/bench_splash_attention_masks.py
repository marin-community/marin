# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Benchmark Splash Attention structured mask lowering on TPU.

This harness is intentionally small and focused on the mask variants used by
packed prefix-LM and packed segment-ID work. It reports compile-including and
steady-state forward timings for long sequence lengths where block skipping
should matter.
"""

import argparse
import json
import math
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from math import gcd

import equinox as eqx
import haliax as hax
import haliax.nn as hnn
import jax
import jax.numpy as jnp
import jax.random as jrandom

from levanter.kernels.pallas.splash_attention import DEFAULT_SPLASH_BLOCK_SIZE
from levanter.layers.attention import (
    AttentionMask,
    _prepare_splash_invocation_plan,
    _prepare_splash_layout,
    _run_prepared_tpu_splash_attention,
    _tpu_splash_attention,
)
from levanter.utils.mesh import create_mesh_from_axis_specs

STATIC_CAUSAL_SPLASH_VARIANT = "static_causal_splash"
STATIC_PREFIX_LM_SPLASH_VARIANT = "static_prefix_lm_splash"
PACKED_CAUSAL_SEGMENT_SPLASH_VARIANT = "packed_causal_segment_splash"
PACKED_CAUSAL_SEGMENT_RUNS_SPLASH_VARIANT = "packed_causal_segment_runs_splash"
PACKED_PREFIX_LM_SPLASH_VARIANT = "packed_prefix_lm_splash"
INPUT_SCALE = 0.02
RESIDUAL_SCALE = 0.01
DOC_LENGTH_PROFILE_EQUAL = "equal"
DOC_LENGTH_PROFILE_STAGGERED = "staggered"
DOC_LENGTH_PROFILE_LONG_TAIL = "long-tail"
DOC_LENGTH_PROFILES = (
    DOC_LENGTH_PROFILE_EQUAL,
    DOC_LENGTH_PROFILE_STAGGERED,
    DOC_LENGTH_PROFILE_LONG_TAIL,
)
DTYPE_OPTIONS = (
    ("bf16", jnp.bfloat16),
    ("fp32", jnp.float32),
)
DTYPE_NAMES = tuple(name for name, _ in DTYPE_OPTIONS)
DEFAULT_DTYPE_NAME = DTYPE_NAMES[0]


@dataclass(frozen=True, slots=True)
class BenchShape:
    batch: int
    seq_len: int
    heads: int
    head_dim: int
    block_size: int
    docs_per_sequence: int
    prefix_tokens_per_doc: int
    doc_length_profile: str
    doc_lengths: tuple[tuple[int, ...], ...] | None
    dtype: jnp.dtype


@dataclass(frozen=True, slots=True)
class BenchResult:
    variant: str
    layers: int
    batch: int
    seq_len: int
    heads: int
    head_dim: int
    block_size: int
    docs_per_sequence: int
    prefix_tokens_per_doc: int
    doc_length_profile: str
    doc_lengths: tuple[int, ...]
    doc_lengths_by_batch: tuple[tuple[int, ...], ...]
    dtype: str
    device_kind: str
    num_devices: int
    total_blocks: int
    visited_blocks: int
    visited_block_fraction: float
    skipped_block_fraction: float
    visited_blocks_vs_static_causal: float
    compile_including: float
    compile_including_per_layer: float
    steady_median: float
    steady_median_per_layer: float
    steady_min: float
    steady_min_per_layer: float
    steady_max: float
    steady_max_per_layer: float
    steady_times: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class _BlockStats:
    total_blocks: int
    visited_blocks: int
    static_causal_visited_blocks: int


def main() -> None:
    args = _parse_args()
    dtype = _parse_dtype(args.dtype)
    doc_lengths = _parse_doc_length_batches(args.doc_lengths)
    shape = BenchShape(
        batch=args.batch,
        seq_len=args.seq_len,
        heads=args.heads,
        head_dim=args.head_dim,
        block_size=args.block_size,
        docs_per_sequence=(
            max(len(lengths) for lengths in doc_lengths) if doc_lengths is not None else args.docs_per_sequence
        ),
        prefix_tokens_per_doc=args.prefix_tokens_per_doc,
        doc_length_profile=args.doc_length_profile,
        doc_lengths=doc_lengths,
        dtype=dtype,
    )
    if jax.default_backend() != "tpu" and not args.allow_non_tpu:
        raise RuntimeError("This benchmark expects a TPU backend. Pass --allow-non-tpu only for dry-run debugging.")

    results = run_benchmarks(
        shape,
        warmup=args.warmup,
        iterations=args.iterations,
        include_dense=args.include_dense,
        layers=args.layers,
    )
    for result in results:
        print(json.dumps(asdict(result)))


def run_benchmarks(
    shape: BenchShape,
    *,
    warmup: int,
    iterations: int,
    include_dense: bool,
    layers: int,
) -> list[BenchResult]:
    if layers <= 0:
        raise ValueError("layers must be positive.")

    Batch = hax.Axis("batch", shape.batch)
    Heads = hax.Axis("heads", shape.heads)
    Pos = hax.Axis("position", shape.seq_len)
    KPos = Pos.alias("key_position")
    Key = hax.Axis("key", shape.head_dim)

    rng = jrandom.PRNGKey(0)
    q = hax.random.normal(rng, (Batch, Pos, Heads, Key)).astype(shape.dtype) * INPUT_SCALE
    rng, key_rng = jrandom.split(rng)
    k = hax.random.normal(key_rng, (Batch, KPos, Heads, Key)).astype(shape.dtype) * INPUT_SCALE
    rng, value_rng = jrandom.split(rng)
    v = hax.random.normal(value_rng, (Batch, KPos, Heads, Key)).astype(shape.dtype) * INPUT_SCALE

    segment_ids = _packed_segment_ids(shape, Batch, Pos, KPos)
    packed_segment_run_mask = AttentionMask.causal().with_segment_runs(
        segment_ids.q,
        kv_segment_ids=segment_ids.kv,
        max_segments=shape.docs_per_sequence,
    )
    assert packed_segment_run_mask.segment_run_metadata is not None
    prefix_lengths_per_segment = _packed_prefix_lengths_per_segment(
        shape,
        Batch,
        packed_segment_run_mask.segment_run_metadata.segment_lengths.axes[-1],
    )
    masks = {
        STATIC_CAUSAL_SPLASH_VARIANT: AttentionMask.causal(),
        STATIC_PREFIX_LM_SPLASH_VARIANT: AttentionMask.prefix_lm(prefix_length=shape.prefix_tokens_per_doc),
        PACKED_CAUSAL_SEGMENT_SPLASH_VARIANT: AttentionMask.causal(segment_ids=(segment_ids.q, segment_ids.kv)),
        PACKED_CAUSAL_SEGMENT_RUNS_SPLASH_VARIANT: packed_segment_run_mask,
        PACKED_PREFIX_LM_SPLASH_VARIANT: packed_segment_run_mask.with_prefix_lengths_per_segment(
            prefix_lengths_per_segment
        ),
    }

    data_axis = gcd(len(jax.devices()), shape.batch)
    mesh = create_mesh_from_axis_specs(
        ici_axes={"replica": len(jax.devices()) // data_axis, "data": data_axis, "model": 1},
        dcn_axes={},
    )
    mapping = {"batch": "data", "heads": "model"}
    results = []
    with hax.partitioning.set_mesh(mesh), hax.axis_mapping(mapping):
        q = hax.shard(q)
        k = hax.shard(k)
        v = hax.shard(v)
        for variant, mask in masks.items():
            if layers == 1:
                results.append(_time_splash_variant(variant, shape, Pos, KPos, Key, q, k, v, mask, warmup, iterations))
            else:
                results.append(
                    _time_prepared_splash_stack_variant(
                        variant, shape, Pos, KPos, Key, q, k, v, mask, warmup, iterations, layers
                    )
                )
        if include_dense:
            for variant, mask in masks.items():
                if variant == STATIC_CAUSAL_SPLASH_VARIANT:
                    continue
                dense_variant = variant.replace("_splash", "_dense_reference")
                results.append(
                    _time_dense_variant(
                        dense_variant, shape, Pos, KPos, Key, q, k, v, mask, warmup, iterations, layers
                    )
                )
    return results


def _time_splash_variant(
    variant: str,
    shape: BenchShape,
    Pos: hax.Axis,
    KPos: hax.Axis,
    Key: hax.Axis,
    q: hax.NamedArray,
    k: hax.NamedArray,
    v: hax.NamedArray,
    mask: AttentionMask,
    warmup: int,
    iterations: int,
) -> BenchResult:
    @eqx.filter_jit
    def run(q, k, v):
        return _tpu_splash_attention(
            Pos,
            KPos,
            Key,
            q,
            k,
            v,
            inference=True,
            mask=mask,
            block_size=shape.block_size,
            scaling_factor=1 / math.sqrt(Key.size),
        )

    return _time_variant(variant, shape, lambda: run(q, k, v), warmup=warmup, iterations=iterations, layers=1)


def _time_prepared_splash_stack_variant(
    variant: str,
    shape: BenchShape,
    Pos: hax.Axis,
    KPos: hax.Axis,
    Key: hax.Axis,
    q: hax.NamedArray,
    k: hax.NamedArray,
    v: hax.NamedArray,
    mask: AttentionMask,
    warmup: int,
    iterations: int,
    layers: int,
) -> BenchResult:
    layout = _prepare_splash_layout(
        Pos,
        KPos,
        Key,
        q,
        k,
        v,
        scaling_factor=1 / math.sqrt(Key.size),
        block_size=shape.block_size,
    )
    invocation = _prepare_splash_invocation_plan(
        layout=layout,
        mask=mask,
        block_size=shape.block_size,
        logits_soft_cap=None,
        attn_sink=None,
    )

    @eqx.filter_jit
    def run(q, k, v):
        x = q
        for _ in range(layers):
            attn_out = _run_prepared_tpu_splash_attention(
                Pos,
                KPos,
                Key,
                x,
                k,
                v,
                inference=True,
                mask=mask,
                scaling_factor=1 / math.sqrt(Key.size),
                layout=layout,
                invocation=invocation,
            )
            x = x + attn_out * RESIDUAL_SCALE
        return x

    return _time_variant(variant, shape, lambda: run(q, k, v), warmup=warmup, iterations=iterations, layers=layers)


def _time_dense_variant(
    variant: str,
    shape: BenchShape,
    Pos: hax.Axis,
    KPos: hax.Axis,
    Key: hax.Axis,
    q: hax.NamedArray,
    k: hax.NamedArray,
    v: hax.NamedArray,
    mask: AttentionMask,
    warmup: int,
    iterations: int,
    layers: int,
) -> BenchResult:
    materialized_mask = mask.materialize(Pos, KPos)

    @eqx.filter_jit
    def run(q, k, v):
        x = q
        for _ in range(layers):
            attn_out = hnn.attention.dot_product_attention(KPos, Key, x, k, v, mask=materialized_mask)
            x = x + attn_out * RESIDUAL_SCALE
        return x

    return _time_variant(variant, shape, lambda: run(q, k, v), warmup=warmup, iterations=iterations, layers=layers)


def _time_variant(
    variant: str,
    shape: BenchShape,
    run: Callable[[], hax.NamedArray],
    *,
    warmup: int,
    iterations: int,
    layers: int,
) -> BenchResult:
    start = time.perf_counter()
    output = run()
    output.array.block_until_ready()
    compile_including = time.perf_counter() - start

    for _ in range(warmup):
        output = run()
        output.array.block_until_ready()

    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        output = run()
        output.array.block_until_ready()
        times.append(time.perf_counter() - start)

    device_kind = jax.devices()[0].device_kind if jax.devices() else "unknown"
    times_array = jnp.asarray(times)
    block_stats = _block_stats_for_variant(shape, variant)
    return BenchResult(
        variant=variant,
        layers=layers,
        batch=shape.batch,
        seq_len=shape.seq_len,
        heads=shape.heads,
        head_dim=shape.head_dim,
        block_size=shape.block_size,
        docs_per_sequence=shape.docs_per_sequence,
        prefix_tokens_per_doc=shape.prefix_tokens_per_doc,
        doc_length_profile=shape.doc_length_profile,
        doc_lengths=_doc_lengths(shape),
        doc_lengths_by_batch=_doc_length_batches(shape),
        dtype=str(shape.dtype),
        device_kind=device_kind,
        num_devices=len(jax.devices()),
        total_blocks=block_stats.total_blocks,
        visited_blocks=block_stats.visited_blocks,
        visited_block_fraction=block_stats.visited_blocks / block_stats.total_blocks,
        skipped_block_fraction=1 - block_stats.visited_blocks / block_stats.total_blocks,
        visited_blocks_vs_static_causal=block_stats.visited_blocks / block_stats.static_causal_visited_blocks,
        compile_including=compile_including,
        compile_including_per_layer=compile_including / layers,
        steady_median=float(jnp.median(times_array)),
        steady_median_per_layer=float(jnp.median(times_array)) / layers,
        steady_min=float(jnp.min(times_array)),
        steady_min_per_layer=float(jnp.min(times_array)) / layers,
        steady_max=float(jnp.max(times_array)),
        steady_max_per_layer=float(jnp.max(times_array)) / layers,
        steady_times=tuple(float(t) for t in times),
    )


@dataclass(frozen=True, slots=True)
class _PackedSegmentIds:
    q: hax.NamedArray
    kv: hax.NamedArray


def _block_stats_for_variant(shape: BenchShape, variant: str) -> _BlockStats:
    splash_variant = variant.replace("_dense_reference", "_splash")
    total_blocks = shape.batch * _blocks_per_sequence(shape) ** 2
    static_causal_blocks = _count_static_causal_blocks(shape) * shape.batch

    if splash_variant == STATIC_CAUSAL_SPLASH_VARIANT:
        visited_blocks = static_causal_blocks
    elif splash_variant == STATIC_PREFIX_LM_SPLASH_VARIANT:
        visited_blocks = _count_static_prefix_lm_blocks(shape) * shape.batch
    elif splash_variant in {PACKED_CAUSAL_SEGMENT_SPLASH_VARIANT, PACKED_CAUSAL_SEGMENT_RUNS_SPLASH_VARIANT}:
        visited_blocks = _count_packed_blocks(shape, _packed_causal_block_has_attention)
    elif splash_variant == PACKED_PREFIX_LM_SPLASH_VARIANT:
        visited_blocks = _count_packed_blocks(shape, _packed_prefix_lm_block_has_attention)
    else:
        raise ValueError(f"Unknown benchmark variant {variant!r}.")

    return _BlockStats(
        total_blocks=total_blocks,
        visited_blocks=visited_blocks,
        static_causal_visited_blocks=static_causal_blocks,
    )


def _blocks_per_sequence(shape: BenchShape) -> int:
    return math.ceil(shape.seq_len / shape.block_size)


def _count_static_causal_blocks(shape: BenchShape) -> int:
    blocks = _blocks_per_sequence(shape)
    return blocks * (blocks + 1) // 2


def _count_static_prefix_lm_blocks(shape: BenchShape) -> int:
    return _count_blocks(
        shape,
        lambda _doc_lengths, _q_start, q_end, kv_start, _kv_end: kv_start < q_end
        or kv_start < shape.prefix_tokens_per_doc,
    )


def _count_packed_blocks(
    shape: BenchShape,
    block_has_attention: Callable[[BenchShape, tuple[int, ...], int, int, int, int], bool],
) -> int:
    # Count block reachability without constructing block_q x block_kv payloads.
    return sum(
        _count_blocks(
            shape,
            lambda doc_lengths, q_start, q_end, kv_start, kv_end: block_has_attention(
                shape, doc_lengths, q_start, q_end, kv_start, kv_end
            ),
            doc_lengths=doc_lengths,
        )
        for doc_lengths in _doc_length_batches(shape)
    )


def _count_blocks(
    shape: BenchShape,
    block_has_attention: Callable[[tuple[int, ...], int, int, int, int], bool],
    *,
    doc_lengths: tuple[int, ...] | None = None,
) -> int:
    blocks = _blocks_per_sequence(shape)
    count = 0
    lengths = _doc_lengths(shape) if doc_lengths is None else doc_lengths
    for q_block in range(blocks):
        q_start = q_block * shape.block_size
        q_end = min(q_start + shape.block_size, shape.seq_len)
        for kv_block in range(blocks):
            kv_start = kv_block * shape.block_size
            kv_end = min(kv_start + shape.block_size, shape.seq_len)
            if block_has_attention(lengths, q_start, q_end, kv_start, kv_end):
                count += 1
    return count


def _packed_causal_block_has_attention(
    _shape: BenchShape,
    doc_lengths: tuple[int, ...],
    q_start: int,
    q_end: int,
    kv_start: int,
    kv_end: int,
) -> bool:
    return _any_packed_doc_block_overlap(
        doc_lengths,
        q_start,
        q_end,
        kv_start,
        kv_end,
        lambda _doc_start, _doc_end, _q_doc_start, q_doc_end, kv_doc_start, _kv_doc_end: kv_doc_start < q_doc_end,
    )


def _packed_prefix_lm_block_has_attention(
    shape: BenchShape,
    doc_lengths: tuple[int, ...],
    q_start: int,
    q_end: int,
    kv_start: int,
    kv_end: int,
) -> bool:
    return _any_packed_doc_block_overlap(
        doc_lengths,
        q_start,
        q_end,
        kv_start,
        kv_end,
        lambda doc_start, doc_end, _q_doc_start, q_doc_end, kv_doc_start, _kv_doc_end: (
            kv_doc_start < min(doc_start + shape.prefix_tokens_per_doc, doc_end) or kv_doc_start < q_doc_end
        ),
    )


def _any_packed_doc_block_overlap(
    doc_lengths: tuple[int, ...],
    q_start: int,
    q_end: int,
    kv_start: int,
    kv_end: int,
    doc_block_has_attention: Callable[[int, int, int, int, int, int], bool],
) -> bool:
    for doc_start, doc_end in _doc_intervals(doc_lengths):
        q_doc_start = max(q_start, doc_start)
        q_doc_end = min(q_end, doc_end)
        kv_doc_start = max(kv_start, doc_start)
        kv_doc_end = min(kv_end, doc_end)
        if q_doc_start >= q_doc_end or kv_doc_start >= kv_doc_end:
            continue
        if doc_block_has_attention(doc_start, doc_end, q_doc_start, q_doc_end, kv_doc_start, kv_doc_end):
            return True
    return False


def _doc_intervals(doc_lengths: tuple[int, ...]) -> tuple[tuple[int, int], ...]:
    intervals = []
    start = 0
    for doc_len in doc_lengths:
        end = start + doc_len
        intervals.append((start, end))
        start = end
    return tuple(intervals)


def _packed_segment_ids(shape: BenchShape, Batch: hax.Axis, Pos: hax.Axis, KPos: hax.Axis) -> _PackedSegmentIds:
    segment_ids = jnp.stack(
        [
            jnp.concatenate(
                [jnp.full((doc_len,), doc_id, dtype=jnp.int32) for doc_id, doc_len in enumerate(doc_lengths)]
            )
            for doc_lengths in _doc_length_batches(shape)
        ],
        axis=0,
    )
    return _PackedSegmentIds(
        q=hax.named(segment_ids, (Batch, Pos)),
        kv=hax.named(segment_ids, (Batch, KPos)),
    )


def _packed_prefix_mask(shape: BenchShape, Batch: hax.Axis, Pos: hax.Axis) -> hax.NamedArray:
    prefix = jnp.stack(
        [
            jnp.concatenate(
                [
                    jnp.arange(doc_len, dtype=jnp.int32) < min(shape.prefix_tokens_per_doc, doc_len)
                    for doc_len in doc_lengths
                ]
            )
            for doc_lengths in _doc_length_batches(shape)
        ],
        axis=0,
    )
    return hax.named(prefix, (Batch, Pos))


def _packed_prefix_lengths_per_segment(
    shape: BenchShape,
    Batch: hax.Axis,
    SegmentRun: hax.Axis,
) -> hax.NamedArray:
    prefix_lengths = jnp.stack(
        [
            jnp.asarray(
                [min(shape.prefix_tokens_per_doc, doc_len) for doc_len in doc_lengths]
                + [0] * (SegmentRun.size - len(doc_lengths)),
                dtype=jnp.int32,
            )
            for doc_lengths in _doc_length_batches(shape)
        ],
        axis=0,
    )
    return hax.named(prefix_lengths, (Batch, SegmentRun))


def _doc_lengths(shape: BenchShape) -> tuple[int, ...]:
    return _doc_length_batches(shape)[0]


def _doc_length_batches(shape: BenchShape) -> tuple[tuple[int, ...], ...]:
    if shape.doc_lengths is not None:
        _validate_doc_length_batches(shape.doc_lengths, shape)
        if len(shape.doc_lengths) == 1:
            return (shape.doc_lengths[0],) * shape.batch
        return shape.doc_lengths

    if shape.docs_per_sequence <= 0:
        raise ValueError("docs_per_sequence must be positive.")
    if shape.docs_per_sequence > shape.seq_len:
        raise ValueError("docs_per_sequence must be <= seq_len.")

    if shape.doc_length_profile == DOC_LENGTH_PROFILE_EQUAL:
        if shape.seq_len % shape.docs_per_sequence != 0:
            raise ValueError("seq_len must be divisible by docs_per_sequence for equal doc lengths.")
        doc_len = shape.seq_len // shape.docs_per_sequence
        return ((doc_len,) * shape.docs_per_sequence,) * shape.batch

    if shape.doc_length_profile == DOC_LENGTH_PROFILE_STAGGERED:
        weights = tuple(range(1, shape.docs_per_sequence + 1))
    elif shape.doc_length_profile == DOC_LENGTH_PROFILE_LONG_TAIL:
        weights = tuple(2**i for i in reversed(range(shape.docs_per_sequence)))
    else:
        raise ValueError(f"Unsupported doc_length_profile {shape.doc_length_profile!r}.")

    return (_integer_partition_from_weights(shape.seq_len, weights),) * shape.batch


def _validate_doc_length_batches(doc_lengths: tuple[tuple[int, ...], ...], shape: BenchShape) -> None:
    if len(doc_lengths) == 0:
        raise ValueError("doc_lengths must not be empty.")
    if len(doc_lengths) not in {1, shape.batch}:
        raise ValueError(f"doc_lengths must provide either one layout or batch={shape.batch} layouts.")
    for batch_index, batch_lengths in enumerate(doc_lengths):
        if len(batch_lengths) == 0:
            raise ValueError(f"doc_lengths layout {batch_index} must not be empty.")
        if any(doc_len <= 0 for doc_len in batch_lengths):
            raise ValueError(f"doc_lengths layout {batch_index} must contain only positive lengths.")
        if sum(batch_lengths) != shape.seq_len:
            raise ValueError(
                f"doc_lengths layout {batch_index} must sum to seq_len={shape.seq_len}, got {sum(batch_lengths)}."
            )


def _integer_partition_from_weights(total: int, weights: tuple[int, ...]) -> tuple[int, ...]:
    min_lengths = len(weights)
    if total < min_lengths:
        raise ValueError("total must be at least the number of weights.")

    remaining = total - min_lengths
    weight_sum = sum(weights)
    raw_lengths = [remaining * weight / weight_sum for weight in weights]
    lengths = [1 + int(raw_length) for raw_length in raw_lengths]
    remainder = total - sum(lengths)
    fractional_order = sorted(range(len(weights)), key=lambda i: raw_lengths[i] - int(raw_lengths[i]), reverse=True)
    for index in fractional_order[:remainder]:
        lengths[index] += 1

    assert sum(lengths) == total
    return tuple(lengths)


def _parse_dtype(dtype: str) -> jnp.dtype:
    for name, jax_dtype in DTYPE_OPTIONS:
        if dtype == name:
            return jax_dtype
    raise ValueError(f"Unsupported dtype {dtype!r}.")


def _parse_doc_length_batches(doc_lengths: str | None) -> tuple[tuple[int, ...], ...] | None:
    if doc_lengths is None:
        return None
    parsed_batches = tuple(
        tuple(int(part.strip()) for part in batch_part.split(",") if part.strip())
        for batch_part in doc_lengths.split(";")
        if batch_part.strip()
    )
    if not parsed_batches:
        raise ValueError("--doc-lengths must contain at least one integer.")
    return parsed_batches


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=8192)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=DEFAULT_SPLASH_BLOCK_SIZE)
    parser.add_argument("--docs-per-sequence", type=int, default=4)
    parser.add_argument("--prefix-tokens-per-doc", type=int, default=256)
    parser.add_argument("--doc-length-profile", choices=DOC_LENGTH_PROFILES, default=DOC_LENGTH_PROFILE_EQUAL)
    parser.add_argument(
        "--doc-lengths",
        type=str,
        default=None,
        help="Comma-separated lengths for one packed layout, or semicolon-separated layouts for each batch row.",
    )
    parser.add_argument("--dtype", choices=DTYPE_NAMES, default=DEFAULT_DTYPE_NAME)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument(
        "--layers",
        type=int,
        default=1,
        help="Number of repeated attention calls in a fake residual stack. Values >1 reuse one prepared Splash plan.",
    )
    parser.add_argument("--include-dense", action="store_true")
    parser.add_argument("--allow-non-tpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
