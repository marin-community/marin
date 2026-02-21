# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import argparse
import asyncio
from dataclasses import dataclass

import jax
import numpy as np

from levanter.data import AsyncDataset
from levanter.data.dataset import ListAsyncDataset
from levanter.store.jagged_array import DEFAULT_CHUNK_SIZE


@dataclass(frozen=True)
class PermutationQualityMetrics:
    strategy: str
    seed: int
    n: int
    io_block_size: int
    window_blocks: int
    is_bijection: bool
    fixed_points: int
    fixed_points_rate: float
    mean_abs_displacement: float
    mean_abs_displacement_norm: float
    displacement_p50: float
    displacement_p90: float
    displacement_p99: float
    spearman_rho: float
    inversion_fraction: float | None
    same_block_transition_rate: float
    mean_block_jump: float
    mean_source_block_run_length: float


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Statistical characterization of permutation quality/locality.")
    parser.add_argument("--n", type=int, default=65536, help="Dataset length to probe.")
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=3, help="How many seeds to evaluate per strategy.")
    parser.add_argument("--strategies", type=str, default="full,era,block")
    parser.add_argument("--perm-type", type=str, default="feistel", choices=("feistel", "linear"))
    parser.add_argument("--era-length", type=int, default=1024)
    parser.add_argument("--io-block-size", type=int, default=0, help="0 => chunk-aligned default.")
    parser.add_argument("--window-blocks", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8192, help="Materialization chunk size.")
    parser.add_argument(
        "--max-n-for-inversions",
        type=int,
        default=200000,
        help="Skip inversion metric above this length to keep runtime bounded.",
    )
    return parser.parse_args()


def _derive_chunk_aligned_block_size(seq_len: int) -> int:
    return max(1, DEFAULT_CHUNK_SIZE // seq_len)


def _make_strategy_dataset(
    strategy: str,
    base: ListAsyncDataset[int],
    *,
    seed: int,
    perm_type: str,
    era_length: int,
    io_block_size: int,
    window_blocks: int,
) -> AsyncDataset[int]:
    key = jax.random.PRNGKey(seed)
    if strategy == "none":
        return base
    if strategy == "full":
        return base.shuffle(key, perm_type=perm_type)
    if strategy == "era":
        return base.era_shuffle(era_length, key=key, perm_type=perm_type)
    if strategy == "block":
        return base.block_shuffle(
            io_block_size=io_block_size,
            window_blocks=window_blocks,
            key=key,
            perm_type=perm_type,
        )
    raise ValueError(f"Unknown strategy: {strategy}")


async def _materialize_permutation(dataset: AsyncDataset[int], n: int, batch_size: int) -> np.ndarray:
    perm = np.empty(n, dtype=np.int64)
    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        idx = list(range(start, stop))
        out = await dataset.get_batch(idx)
        perm[start:stop] = np.asarray(out, dtype=np.int64)
    return perm


def _merge_count_inversions(a: np.ndarray) -> int:
    b = a.copy()
    temp = np.empty_like(b)

    def _count(lo: int, hi: int) -> int:
        if hi - lo <= 1:
            return 0
        mid = (lo + hi) // 2
        inv = _count(lo, mid) + _count(mid, hi)
        i = lo
        j = mid
        k = lo
        while i < mid and j < hi:
            if b[i] <= b[j]:
                temp[k] = b[i]
                i += 1
            else:
                temp[k] = b[j]
                inv += mid - i
                j += 1
            k += 1
        while i < mid:
            temp[k] = b[i]
            i += 1
            k += 1
        while j < hi:
            temp[k] = b[j]
            j += 1
            k += 1
        b[lo:hi] = temp[lo:hi]
        return inv

    return _count(0, len(b))


def _mean_source_block_run_length(block_ids: np.ndarray) -> float:
    if len(block_ids) == 0:
        return 0.0
    boundaries = np.count_nonzero(np.diff(block_ids) != 0)
    run_count = boundaries + 1
    return float(len(block_ids)) / float(run_count)


def _same_block_transition_expected_random(n: int, io_block_size: int) -> float:
    if n <= 1:
        return 0.0
    full_blocks = n // io_block_size
    tail = n % io_block_size
    block_sizes = [io_block_size] * full_blocks
    if tail > 0:
        block_sizes.append(tail)
    num = sum(s * (s - 1) for s in block_sizes)
    den = n * (n - 1)
    return num / den


def _same_block_transition_no_shuffle(n: int, io_block_size: int) -> float:
    if n <= 1:
        return 0.0
    full_blocks = n // io_block_size
    tail = n % io_block_size
    transitions = 0
    if full_blocks > 0:
        transitions += full_blocks * (io_block_size - 1)
    if tail > 0:
        transitions += tail - 1
    return transitions / (n - 1)


def _expected_mean_abs_displacement_random(n: int) -> float:
    if n <= 0:
        return 0.0
    return (n**2 - 1) / (3 * n)


def _compute_metrics(
    *,
    perm: np.ndarray,
    strategy: str,
    seed: int,
    n: int,
    io_block_size: int,
    window_blocks: int,
    max_n_for_inversions: int,
) -> PermutationQualityMetrics:
    idx = np.arange(n, dtype=np.int64)

    is_bijection = bool(np.array_equal(np.sort(perm), idx))
    fixed_points = int(np.count_nonzero(perm == idx))
    fixed_points_rate = fixed_points / n if n > 0 else 0.0

    displacement = np.abs(perm - idx)
    mean_abs_displacement = float(np.mean(displacement)) if n > 0 else 0.0
    mean_abs_displacement_norm = mean_abs_displacement / max(1, n - 1)
    displacement_p50 = float(np.quantile(displacement, 0.50)) if n > 0 else 0.0
    displacement_p90 = float(np.quantile(displacement, 0.90)) if n > 0 else 0.0
    displacement_p99 = float(np.quantile(displacement, 0.99)) if n > 0 else 0.0

    spearman_rho = float(np.corrcoef(idx, perm)[0, 1]) if n > 1 else 1.0

    inversion_fraction: float | None
    if n > 1 and n <= max_n_for_inversions:
        inv = _merge_count_inversions(perm)
        max_inv = n * (n - 1) // 2
        inversion_fraction = inv / max_inv
    else:
        inversion_fraction = None

    if n > 1:
        block_ids = perm // io_block_size
        same_block_transition_rate = float(np.mean(block_ids[1:] == block_ids[:-1]))
        mean_block_jump = float(np.mean(np.abs(np.diff(block_ids))))
        mean_source_block_run_length = _mean_source_block_run_length(block_ids)
    else:
        same_block_transition_rate = 0.0
        mean_block_jump = 0.0
        mean_source_block_run_length = 1.0

    return PermutationQualityMetrics(
        strategy=strategy,
        seed=seed,
        n=n,
        io_block_size=io_block_size,
        window_blocks=window_blocks,
        is_bijection=is_bijection,
        fixed_points=fixed_points,
        fixed_points_rate=fixed_points_rate,
        mean_abs_displacement=mean_abs_displacement,
        mean_abs_displacement_norm=mean_abs_displacement_norm,
        displacement_p50=displacement_p50,
        displacement_p90=displacement_p90,
        displacement_p99=displacement_p99,
        spearman_rho=spearman_rho,
        inversion_fraction=inversion_fraction,
        same_block_transition_rate=same_block_transition_rate,
        mean_block_jump=mean_block_jump,
        mean_source_block_run_length=mean_source_block_run_length,
    )


def _aggregate(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return values[0], 0.0
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=1))


def _print_run_metrics(m: PermutationQualityMetrics) -> None:
    p = f"strategy.{m.strategy}.seed{m.seed}"
    print(f"{p}.is_bijection", m.is_bijection)
    print(f"{p}.fixed_points", m.fixed_points)
    print(f"{p}.fixed_points_rate", m.fixed_points_rate)
    print(f"{p}.mean_abs_displacement", m.mean_abs_displacement)
    print(f"{p}.mean_abs_displacement_norm", m.mean_abs_displacement_norm)
    print(f"{p}.displacement_p50", m.displacement_p50)
    print(f"{p}.displacement_p90", m.displacement_p90)
    print(f"{p}.displacement_p99", m.displacement_p99)
    print(f"{p}.spearman_rho", m.spearman_rho)
    print(f"{p}.inversion_fraction", m.inversion_fraction)
    print(f"{p}.same_block_transition_rate", m.same_block_transition_rate)
    print(f"{p}.mean_block_jump", m.mean_block_jump)
    print(f"{p}.mean_source_block_run_length", m.mean_source_block_run_length)


def _print_aggregates(strategy: str, runs: list[PermutationQualityMetrics]) -> None:
    prefix = f"aggregate.{strategy}"
    assert runs

    fields = [
        ("fixed_points_rate", [r.fixed_points_rate for r in runs]),
        ("mean_abs_displacement_norm", [r.mean_abs_displacement_norm for r in runs]),
        ("spearman_rho", [r.spearman_rho for r in runs]),
        ("same_block_transition_rate", [r.same_block_transition_rate for r in runs]),
        ("mean_block_jump", [r.mean_block_jump for r in runs]),
        ("mean_source_block_run_length", [r.mean_source_block_run_length for r in runs]),
    ]

    inversion_values = [r.inversion_fraction for r in runs if r.inversion_fraction is not None]
    if inversion_values:
        fields.append(("inversion_fraction", [float(x) for x in inversion_values]))

    for name, vals in fields:
        mean, std = _aggregate(vals)
        print(f"{prefix}.{name}.mean", mean)
        print(f"{prefix}.{name}.std", std)


async def _run(args: argparse.Namespace) -> None:
    if args.n <= 0:
        raise ValueError("--n must be positive")
    if args.repeats <= 0:
        raise ValueError("--repeats must be positive")

    io_block_size = args.io_block_size
    if io_block_size <= 0:
        io_block_size = _derive_chunk_aligned_block_size(args.seq_len)

    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    allowed = {"none", "full", "era", "block"}
    bad = [s for s in strategies if s not in allowed]
    if bad:
        raise ValueError(f"Unknown strategies: {bad}; allowed: {sorted(allowed)}")
    if not strategies:
        raise ValueError("No strategies requested.")

    print("meta.n", args.n)
    print("meta.repeats", args.repeats)
    print("meta.seq_len", args.seq_len)
    print("meta.perm_type", args.perm_type)
    print("meta.era_length", args.era_length)
    print("meta.io_block_size", io_block_size)
    print("meta.window_blocks", args.window_blocks)
    print("meta.random_expected.fixed_points", 1.0)
    print("meta.random_expected.mean_abs_displacement", _expected_mean_abs_displacement_random(args.n))
    print(
        "meta.random_expected.mean_abs_displacement_norm",
        _expected_mean_abs_displacement_random(args.n) / (args.n - 1),
    )
    print("meta.random_expected.inversion_fraction", 0.5)
    print(
        "meta.random_expected.same_block_transition_rate",
        _same_block_transition_expected_random(args.n, io_block_size),
    )
    print("meta.no_shuffle.same_block_transition_rate", _same_block_transition_no_shuffle(args.n, io_block_size))

    base = ListAsyncDataset(list(range(args.n)))
    results_by_strategy: dict[str, list[PermutationQualityMetrics]] = {s: [] for s in strategies}

    for strategy in strategies:
        for repeat in range(args.repeats):
            seed = args.seed + repeat
            ds = _make_strategy_dataset(
                strategy,
                base,
                seed=seed,
                perm_type=args.perm_type,
                era_length=args.era_length,
                io_block_size=io_block_size,
                window_blocks=args.window_blocks,
            )
            perm = await _materialize_permutation(ds, args.n, args.batch_size)
            metrics = _compute_metrics(
                perm=perm,
                strategy=strategy,
                seed=seed,
                n=args.n,
                io_block_size=io_block_size,
                window_blocks=args.window_blocks,
                max_n_for_inversions=args.max_n_for_inversions,
            )
            _print_run_metrics(metrics)
            results_by_strategy[strategy].append(metrics)

    for strategy, runs in results_by_strategy.items():
        _print_aggregates(strategy, runs)


def main() -> None:
    args = _parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
