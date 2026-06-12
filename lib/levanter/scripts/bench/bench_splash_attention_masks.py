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
from levanter.layers.attention import AttentionMask, _tpu_splash_attention
from levanter.utils.mesh import create_mesh_from_axis_specs

STATIC_CAUSAL_SPLASH_VARIANT = "static_causal_splash"
INPUT_SCALE = 0.02
DOC_LENGTH_PROFILE_EQUAL = "equal"
DOC_LENGTH_PROFILE_STAGGERED = "staggered"
DOC_LENGTH_PROFILE_LONG_TAIL = "long-tail"
DOC_LENGTH_PROFILES = (
    DOC_LENGTH_PROFILE_EQUAL,
    DOC_LENGTH_PROFILE_STAGGERED,
    DOC_LENGTH_PROFILE_LONG_TAIL,
)


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
    doc_lengths: tuple[int, ...] | None
    dtype: jnp.dtype


@dataclass(frozen=True, slots=True)
class BenchResult:
    variant: str
    batch: int
    seq_len: int
    heads: int
    head_dim: int
    block_size: int
    docs_per_sequence: int
    prefix_tokens_per_doc: int
    doc_length_profile: str
    doc_lengths: tuple[int, ...]
    dtype: str
    device_kind: str
    num_devices: int
    compile_including: float
    steady_median: float
    steady_min: float
    steady_max: float
    steady_times: tuple[float, ...]


def main() -> None:
    args = _parse_args()
    dtype = _parse_dtype(args.dtype)
    doc_lengths = _parse_doc_lengths(args.doc_lengths)
    shape = BenchShape(
        batch=args.batch,
        seq_len=args.seq_len,
        heads=args.heads,
        head_dim=args.head_dim,
        block_size=args.block_size,
        docs_per_sequence=len(doc_lengths) if doc_lengths is not None else args.docs_per_sequence,
        prefix_tokens_per_doc=args.prefix_tokens_per_doc,
        doc_length_profile=args.doc_length_profile,
        doc_lengths=doc_lengths,
        dtype=dtype,
    )
    if jax.default_backend() != "tpu" and not args.allow_non_tpu:
        raise RuntimeError("This benchmark expects a TPU backend. Pass --allow-non-tpu only for dry-run debugging.")

    results = run_benchmarks(shape, warmup=args.warmup, iterations=args.iterations, include_dense=args.include_dense)
    for result in results:
        print(json.dumps(asdict(result)))


def run_benchmarks(
    shape: BenchShape,
    *,
    warmup: int,
    iterations: int,
    include_dense: bool,
) -> list[BenchResult]:
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
    prefix_mask = _packed_prefix_mask(shape, Batch, Pos)
    masks = {
        STATIC_CAUSAL_SPLASH_VARIANT: AttentionMask.causal(),
        "packed_causal_segment_splash": AttentionMask.causal(segment_ids=(segment_ids.q, segment_ids.kv)),
        "packed_causal_segment_runs_splash": AttentionMask.causal().with_segment_runs(
            segment_ids.q,
            kv_segment_ids=segment_ids.kv,
            max_segments=shape.docs_per_sequence,
        ),
        "packed_prefix_lm_splash": AttentionMask.prefix_lm(
            prefix_mask=prefix_mask,
            segment_ids=(segment_ids.q, segment_ids.kv),
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
            results.append(_time_splash_variant(variant, shape, Pos, KPos, Key, q, k, v, mask, warmup, iterations))
        if include_dense:
            for variant, mask in masks.items():
                if variant == STATIC_CAUSAL_SPLASH_VARIANT:
                    continue
                dense_variant = variant.replace("_splash", "_dense_reference")
                results.append(
                    _time_dense_variant(dense_variant, shape, Pos, KPos, Key, q, k, v, mask, warmup, iterations)
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

    return _time_variant(variant, shape, lambda: run(q, k, v), warmup=warmup, iterations=iterations)


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
) -> BenchResult:
    materialized_mask = mask.materialize(Pos, KPos)

    @eqx.filter_jit
    def run(q, k, v):
        return hnn.attention.dot_product_attention(KPos, Key, q, k, v, mask=materialized_mask)

    return _time_variant(variant, shape, lambda: run(q, k, v), warmup=warmup, iterations=iterations)


def _time_variant(
    variant: str,
    shape: BenchShape,
    run: Callable[[], hax.NamedArray],
    *,
    warmup: int,
    iterations: int,
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
    return BenchResult(
        variant=variant,
        batch=shape.batch,
        seq_len=shape.seq_len,
        heads=shape.heads,
        head_dim=shape.head_dim,
        block_size=shape.block_size,
        docs_per_sequence=shape.docs_per_sequence,
        prefix_tokens_per_doc=shape.prefix_tokens_per_doc,
        doc_length_profile=shape.doc_length_profile,
        doc_lengths=_doc_lengths(shape),
        dtype=str(shape.dtype),
        device_kind=device_kind,
        num_devices=len(jax.devices()),
        compile_including=compile_including,
        steady_median=float(jnp.median(times_array)),
        steady_min=float(jnp.min(times_array)),
        steady_max=float(jnp.max(times_array)),
        steady_times=tuple(float(t) for t in times),
    )


@dataclass(frozen=True, slots=True)
class _PackedSegmentIds:
    q: hax.NamedArray
    kv: hax.NamedArray


def _packed_segment_ids(shape: BenchShape, Batch: hax.Axis, Pos: hax.Axis, KPos: hax.Axis) -> _PackedSegmentIds:
    doc_lengths = _doc_lengths(shape)
    one_sequence = jnp.concatenate(
        [jnp.full((doc_len,), doc_id, dtype=jnp.int32) for doc_id, doc_len in enumerate(doc_lengths)]
    )
    segment_ids = jnp.broadcast_to(one_sequence[None, :], (shape.batch, shape.seq_len))
    return _PackedSegmentIds(
        q=hax.named(segment_ids, (Batch, Pos)),
        kv=hax.named(segment_ids, (Batch, KPos)),
    )


def _packed_prefix_mask(shape: BenchShape, Batch: hax.Axis, Pos: hax.Axis) -> hax.NamedArray:
    doc_lengths = _doc_lengths(shape)
    prefix = jnp.concatenate(
        [jnp.arange(doc_len, dtype=jnp.int32) < min(shape.prefix_tokens_per_doc, doc_len) for doc_len in doc_lengths]
    )
    prefix = jnp.broadcast_to(prefix[None, :], (shape.batch, shape.seq_len))
    return hax.named(prefix, (Batch, Pos))


def _doc_lengths(shape: BenchShape) -> tuple[int, ...]:
    if shape.doc_lengths is not None:
        if len(shape.doc_lengths) == 0:
            raise ValueError("doc_lengths must not be empty.")
        if any(doc_len <= 0 for doc_len in shape.doc_lengths):
            raise ValueError("doc_lengths must all be positive.")
        if sum(shape.doc_lengths) != shape.seq_len:
            raise ValueError(f"doc_lengths must sum to seq_len={shape.seq_len}, got {sum(shape.doc_lengths)}.")
        return shape.doc_lengths

    if shape.docs_per_sequence <= 0:
        raise ValueError("docs_per_sequence must be positive.")
    if shape.docs_per_sequence > shape.seq_len:
        raise ValueError("docs_per_sequence must be <= seq_len.")

    if shape.doc_length_profile == DOC_LENGTH_PROFILE_EQUAL:
        if shape.seq_len % shape.docs_per_sequence != 0:
            raise ValueError("seq_len must be divisible by docs_per_sequence for equal doc lengths.")
        doc_len = shape.seq_len // shape.docs_per_sequence
        return (doc_len,) * shape.docs_per_sequence

    if shape.doc_length_profile == DOC_LENGTH_PROFILE_STAGGERED:
        weights = tuple(range(1, shape.docs_per_sequence + 1))
    elif shape.doc_length_profile == DOC_LENGTH_PROFILE_LONG_TAIL:
        weights = tuple(2**i for i in reversed(range(shape.docs_per_sequence)))
    else:
        raise ValueError(f"Unsupported doc_length_profile {shape.doc_length_profile!r}.")

    return _integer_partition_from_weights(shape.seq_len, weights)


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
    if dtype == "bf16":
        return jnp.bfloat16
    if dtype == "fp32":
        return jnp.float32
    raise ValueError(f"Unsupported dtype {dtype!r}.")


def _parse_doc_lengths(doc_lengths: str | None) -> tuple[int, ...] | None:
    if doc_lengths is None:
        return None
    parsed_lengths = tuple(int(part.strip()) for part in doc_lengths.split(",") if part.strip())
    if not parsed_lengths:
        raise ValueError("--doc-lengths must contain at least one integer.")
    return parsed_lengths


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
    parser.add_argument("--doc-lengths", type=str, default=None)
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--include-dense", action="store_true")
    parser.add_argument("--allow-non-tpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
