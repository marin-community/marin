# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Bounded CPU reproduction for DataLoader prefetch-buffer starvation.

The synthetic dataset deliberately returns examples in all-or-nothing bursts. The
default invocation compares a consumer above the producer's sustained rate with
one below it and asserts the expected queue-drain boundary.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
import json
import time
from typing import Sequence

import jax
from jax.sharding import Mesh
import numpy as np

from levanter.data.dataset import AsyncDataset
from levanter.data.loader import DataLoader
from levanter.utils.background_iterable import BackgroundIterator


@dataclass(frozen=True)
class TraceRow:
    scenario: str
    step: int
    wait_s: float
    buffered_after_next: int | None


class BurstyDataset(AsyncDataset[np.ndarray]):
    """An infinite dataset whose batch API completes in fixed-size bursts."""

    def __init__(self, fetch_delay_s: float):
        self.fetch_delay_s = fetch_delay_s
        self.fetches: list[tuple[int, ...]] = []

    async def async_len(self) -> int:
        raise ValueError("BurstyDataset is infinite")

    def is_finite(self) -> bool:
        return False

    async def getitem_async(self, index: int) -> np.ndarray:
        return np.asarray(index, dtype=np.int32)

    async def get_batch(self, indices: Sequence[int]) -> list[np.ndarray]:
        await asyncio.sleep(self.fetch_delay_s)
        self.fetches.append(tuple(indices))
        return [np.asarray(index, dtype=np.int32) for index in indices]


def _buffered_batch_count(iterator) -> int | None:
    # This is deliberately diagnostic code: DataLoaderIterator does not expose
    # queue occupancy, but its BackgroundIterator already has a public qsize().
    batches = iterator._batches
    if isinstance(batches, BackgroundIterator):
        return batches.qsize()
    return None


def run_scenario(
    *,
    name: str,
    consumer_delay_s: float,
    fetch_delay_s: float,
    prefetch_size: int,
    max_buffered_batches: int,
    warmup_s: float,
    steps: int,
) -> tuple[list[TraceRow], list[tuple[int, ...]], float]:
    dataset = BurstyDataset(fetch_delay_s)
    mesh = Mesh(np.asarray(jax.devices()[:1]), ("data",))
    loader = DataLoader(
        dataset,
        1,
        mesh=mesh,
        axis_resources={"batch": "data"},
        prefetch_size=prefetch_size,
        max_buffered_batches=max_buffered_batches,
    )
    iterator = iter(loader)
    time.sleep(warmup_s)

    rows = []
    scenario_started_at = time.perf_counter()
    try:
        for step in range(steps):
            started_at = time.perf_counter()
            next(iterator)
            rows.append(
                TraceRow(
                    scenario=name,
                    step=step,
                    wait_s=time.perf_counter() - started_at,
                    buffered_after_next=_buffered_batch_count(iterator),
                )
            )
            time.sleep(consumer_delay_s)
    finally:
        iterator._batches.stop()

    return rows, dataset.fetches, time.perf_counter() - scenario_started_at


def _validate(
    rows: list[TraceRow],
    fetches: list[tuple[int, ...]],
    *,
    expect_starvation: bool,
    prefetch_size: int,
    stall_threshold_s: float,
) -> None:
    if not fetches or any(len(indices) != prefetch_size for indices in fetches):
        raise AssertionError(f"expected all fetches to contain {prefetch_size} batches")

    stalled_rows = [row for row in rows if row.wait_s >= stall_threshold_s]
    if expect_starvation and not stalled_rows:
        raise AssertionError("expected the faster consumer to drain the buffer and block")
    if not expect_starvation and stalled_rows:
        raise AssertionError(f"expected the sustainable consumer not to block; stalls={stalled_rows}")

    occupancies = [row.buffered_after_next for row in rows]
    if expect_starvation and 0 not in occupancies:
        raise AssertionError("expected the faster consumer to observe an empty queue")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch-delay-s", type=float, default=0.4)
    parser.add_argument("--prefetch-size", type=int, default=4)
    parser.add_argument("--max-buffered-batches", type=int, default=8)
    parser.add_argument("--warmup-s", type=float, default=1.3)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--fast-consumer-delay-s", type=float, default=0.05)
    parser.add_argument("--sustainable-consumer-delay-s", type=float, default=0.125)
    parser.add_argument("--stall-threshold-s", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    producer_seconds_per_batch = args.fetch_delay_s / args.prefetch_size
    if not args.fast_consumer_delay_s < producer_seconds_per_batch < args.sustainable_consumer_delay_s:
        raise ValueError(
            "expected fast_consumer_delay_s < fetch_delay_s / prefetch_size " "< sustainable_consumer_delay_s"
        )

    config = {
        **vars(args),
        "producer_seconds_per_batch": producer_seconds_per_batch,
        "jax_platform": jax.default_backend(),
        "device_count": 1,
    }
    print("config " + json.dumps(config, sort_keys=True))
    print("scenario\tstep\twait_s\tbuffered_after_next")

    for scenario, consumer_delay_s, expect_starvation in (
        ("faster_than_producer", args.fast_consumer_delay_s, True),
        ("slower_than_producer", args.sustainable_consumer_delay_s, False),
    ):
        rows, fetches, elapsed_s = run_scenario(
            name=scenario,
            consumer_delay_s=consumer_delay_s,
            fetch_delay_s=args.fetch_delay_s,
            prefetch_size=args.prefetch_size,
            max_buffered_batches=args.max_buffered_batches,
            warmup_s=args.warmup_s,
            steps=args.steps,
        )
        for row in rows:
            print(f"{row.scenario}\t{row.step}\t{row.wait_s:.3f}\t{row.buffered_after_next}")
        _validate(
            rows,
            fetches,
            expect_starvation=expect_starvation,
            prefetch_size=args.prefetch_size,
            stall_threshold_s=args.stall_threshold_s,
        )
        stalled_steps = [row.step for row in rows if row.wait_s >= args.stall_threshold_s]
        print(
            "summary "
            + json.dumps(
                {
                    "scenario": scenario,
                    "elapsed_s": round(elapsed_s, 3),
                    "observed_batches_per_s": round(args.steps / elapsed_s, 3),
                    "total_next_wait_s": round(sum(row.wait_s for row in rows), 3),
                    "stalled_steps": stalled_steps,
                    "fetch_calls": len(fetches),
                    "first_fetch_windows": [[indices[0], indices[-1]] for indices in fetches[:3] if indices],
                },
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
