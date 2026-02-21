# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import argparse
import asyncio
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import jax
import numpy as np

from levanter.data import AsyncDataset
from levanter.data.text.datasets import TokenSeqDataset
from levanter.store.cache import SerialCacheWriter, TreeCache
from levanter.store.jagged_array import DEFAULT_CHUNK_SIZE


@dataclass(frozen=True)
class StrategyMetrics:
    strategy: str
    elapsed_s: float
    examples_requested: int
    tensorstore_reads_issued: int
    unique_examples_requested: int
    coalesced_ranges: int
    examples_per_s: float
    tokens_per_s: float
    reads_per_example: float
    estimated_old_reads: int
    estimated_read_reduction: float
    estimated_read_speedup: float


def _derive_chunk_aligned_block_size(seq_len: int) -> int:
    return max(1, DEFAULT_CHUNK_SIZE // seq_len)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark block-shuffle locality and coalesced reads.")
    parser.add_argument("--cache-dir", type=str, default=None, help="Existing TreeCache path to load.")
    parser.add_argument("--num-examples", type=int, default=16384, help="Rows to synthesize when building a cache.")
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=200, help="Number of benchmark windows.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--prefetch-batches", type=int, default=16)
    parser.add_argument(
        "--strategies",
        type=str,
        default="full,era,block",
        help="Comma-separated strategies from: none,full,era,block",
    )
    parser.add_argument("--perm-type", type=str, default="feistel", choices=("feistel", "linear"))
    parser.add_argument("--era-length", type=int, default=1024)
    parser.add_argument(
        "--io-block-size",
        type=int,
        default=0,
        help="Block size for block shuffle. 0 means chunk-aligned default.",
    )
    parser.add_argument("--window-blocks", type=int, default=8)
    parser.add_argument("--write-batch-size", type=int, default=1024)
    return parser.parse_args()


def _build_synthetic_cache(
    cache_dir: Path,
    *,
    seq_len: int,
    num_examples: int,
    vocab_size: int,
    write_batch_size: int,
    seed: int,
) -> TreeCache[dict]:
    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    exemplar = {"input_ids": np.zeros((seq_len,), dtype=np.int32)}
    rng = np.random.default_rng(seed)

    with SerialCacheWriter(str(cache_dir), exemplar) as writer:
        written = 0
        while written < num_examples:
            this_batch = min(write_batch_size, num_examples - written)
            tokens = rng.integers(0, vocab_size, size=(this_batch, seq_len), dtype=np.int32)
            writer.write_batch({"input_ids": tokens})
            written += this_batch
    return writer.result()


def _load_or_build_cache(args: argparse.Namespace) -> tuple[TreeCache[dict], Path, bool]:
    exemplar = {"input_ids": np.zeros((args.seq_len,), dtype=np.int32)}
    if args.cache_dir is not None:
        cache_path = Path(args.cache_dir)
        try:
            cache = TreeCache.load(str(cache_path), exemplar)
            return cache, cache_path, False
        except FileNotFoundError:
            cache = _build_synthetic_cache(
                cache_path,
                seq_len=args.seq_len,
                num_examples=args.num_examples,
                vocab_size=args.vocab_size,
                write_batch_size=args.write_batch_size,
                seed=args.seed,
            )
            return cache, cache_path, True

    temp_root = Path(tempfile.mkdtemp(prefix="levanter-bench-data-"))
    cache_path = temp_root / "cache"
    cache = _build_synthetic_cache(
        cache_path,
        seq_len=args.seq_len,
        num_examples=args.num_examples,
        vocab_size=args.vocab_size,
        write_batch_size=args.write_batch_size,
        seed=args.seed,
    )
    return cache, cache_path, True


def _make_strategy_dataset(
    strategy: str,
    dataset: TokenSeqDataset,
    *,
    seed: int,
    era_length: int,
    io_block_size: int,
    window_blocks: int,
    perm_type: str,
) -> AsyncDataset[np.ndarray]:
    key = jax.random.PRNGKey(seed)

    if strategy == "none":
        return dataset
    if strategy == "full":
        return dataset.shuffle(key, perm_type=perm_type)
    if strategy == "era":
        return dataset.era_shuffle(era_length, key=key, perm_type=perm_type)
    if strategy == "block":
        return dataset.block_shuffle(
            io_block_size=io_block_size,
            window_blocks=window_blocks,
            key=key,
            perm_type=perm_type,
        )

    raise ValueError(f"Unknown strategy: {strategy}")


async def _run_strategy(
    strategy: str,
    dataset: TokenSeqDataset,
    shuffled: AsyncDataset[np.ndarray],
    *,
    batch_size: int,
    prefetch_batches: int,
    steps: int,
    seq_len: int,
) -> StrategyMetrics:
    request_size = batch_size * prefetch_batches
    dataset_len = await shuffled.async_len()
    dataset.reset_read_stats()

    t0 = time.perf_counter()
    for step in range(steps):
        start = step * request_size
        logical_indices = [(start + i) % dataset_len for i in range(request_size)]
        batch = await shuffled.get_batch(logical_indices)
        if len(batch) != request_size:
            raise RuntimeError(f"Expected {request_size} examples from strategy={strategy}, received {len(batch)}.")
    elapsed = time.perf_counter() - t0

    stats = dataset.read_stats
    estimated_old_reads = stats.examples_requested
    reads_issued = stats.tensorstore_reads_issued
    examples_per_s = stats.examples_requested / elapsed if elapsed > 0 else float("inf")
    tokens_per_s = examples_per_s * seq_len
    reads_per_example = reads_issued / stats.examples_requested if stats.examples_requested > 0 else 0.0
    estimated_read_reduction = (
        (estimated_old_reads - reads_issued) / estimated_old_reads if estimated_old_reads > 0 else 0.0
    )
    estimated_read_speedup = estimated_old_reads / reads_issued if reads_issued > 0 else float("inf")

    return StrategyMetrics(
        strategy=strategy,
        elapsed_s=elapsed,
        examples_requested=stats.examples_requested,
        tensorstore_reads_issued=reads_issued,
        unique_examples_requested=stats.unique_examples_requested,
        coalesced_ranges=stats.coalesced_ranges,
        examples_per_s=examples_per_s,
        tokens_per_s=tokens_per_s,
        reads_per_example=reads_per_example,
        estimated_old_reads=estimated_old_reads,
        estimated_read_reduction=estimated_read_reduction,
        estimated_read_speedup=estimated_read_speedup,
    )


def _print_report(
    *,
    metrics: list[StrategyMetrics],
    io_block_size: int,
    window_blocks: int,
    era_length: int,
    seq_len: int,
    batch_size: int,
    prefetch_batches: int,
    steps: int,
    cache_path: Path,
    cache_was_built: bool,
) -> None:
    print("benchmark.cache_path", cache_path)
    print("benchmark.cache_generated", cache_was_built)
    print("benchmark.seq_len", seq_len)
    print("benchmark.batch_size", batch_size)
    print("benchmark.prefetch_batches", prefetch_batches)
    print("benchmark.steps", steps)
    print("benchmark.era_length", era_length)
    print("benchmark.block.io_block_size", io_block_size)
    print("benchmark.block.window_blocks", window_blocks)

    for m in metrics:
        prefix = f"strategy.{m.strategy}"
        print(f"{prefix}.elapsed_s", m.elapsed_s)
        print(f"{prefix}.examples_requested", m.examples_requested)
        print(f"{prefix}.tensorstore_reads_issued", m.tensorstore_reads_issued)
        print(f"{prefix}.unique_examples_requested", m.unique_examples_requested)
        print(f"{prefix}.coalesced_ranges", m.coalesced_ranges)
        print(f"{prefix}.examples_per_s", m.examples_per_s)
        print(f"{prefix}.tokens_per_s", m.tokens_per_s)
        print(f"{prefix}.reads_per_example", m.reads_per_example)
        print(f"{prefix}.estimated_old_reads", m.estimated_old_reads)
        print(f"{prefix}.estimated_read_reduction", m.estimated_read_reduction)
        print(f"{prefix}.estimated_read_speedup", m.estimated_read_speedup)


async def _run(args: argparse.Namespace) -> None:
    cache, cache_path, cache_was_built = _load_or_build_cache(args)
    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    allowed = {"none", "full", "era", "block"}
    invalid = [s for s in strategies if s not in allowed]
    if invalid:
        raise ValueError(f"Invalid strategies: {invalid}. Allowed: {sorted(allowed)}")
    if not strategies:
        raise ValueError("No strategies were provided.")

    io_block_size = args.io_block_size
    if io_block_size <= 0:
        io_block_size = _derive_chunk_aligned_block_size(args.seq_len)

    dataset = TokenSeqDataset(cache, seq_len=args.seq_len)
    metrics: list[StrategyMetrics] = []
    for strategy in strategies:
        shuffled = _make_strategy_dataset(
            strategy,
            dataset,
            seed=args.seed,
            era_length=args.era_length,
            io_block_size=io_block_size,
            window_blocks=args.window_blocks,
            perm_type=args.perm_type,
        )
        result = await _run_strategy(
            strategy,
            dataset,
            shuffled,
            batch_size=args.batch_size,
            prefetch_batches=args.prefetch_batches,
            steps=args.steps,
            seq_len=args.seq_len,
        )
        metrics.append(result)

    _print_report(
        metrics=metrics,
        io_block_size=io_block_size,
        window_blocks=args.window_blocks,
        era_length=args.era_length,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        prefetch_batches=args.prefetch_batches,
        steps=args.steps,
        cache_path=cache_path,
        cache_was_built=cache_was_built,
    )


def main() -> None:
    args = _parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
