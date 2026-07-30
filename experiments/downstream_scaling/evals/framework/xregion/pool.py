# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Generic cross-region Zephyr worker pool orchestration."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from math import prod

from fray.cluster import ResourceConfig
from zephyr.dataset import Dataset, ShardInfo
from zephyr.execution import ZephyrContext
from zephyr.runners import InlineRunner

from experiments.downstream_scaling.evals.framework.xregion import ledger

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WorkerPoolConfig:
    pool_id: str
    num_workers: int
    worker_resources: ResourceConfig
    vm_count: int
    chips_per_vm: int


@dataclass(frozen=True)
class EnginePlacement:
    visible_chips: tuple[int, ...]
    chips_per_process_bounds: tuple[int, int, int]
    tensor_parallel_size: int

    def __post_init__(self) -> None:
        if not self.visible_chips:
            raise ValueError("visible_chips must be non-empty")
        if len(set(self.visible_chips)) != len(self.visible_chips):
            raise ValueError(f"visible_chips must be unique, got {self.visible_chips}")
        if any(chip < 0 for chip in self.visible_chips):
            raise ValueError(f"visible_chips must be non-negative, got {self.visible_chips}")
        if any(size <= 0 for size in self.chips_per_process_bounds):
            raise ValueError(f"chips_per_process_bounds must be positive, got {self.chips_per_process_bounds}")
        bounds_volume = prod(self.chips_per_process_bounds)
        if bounds_volume != len(self.visible_chips):
            raise ValueError(
                f"chips_per_process_bounds={self.chips_per_process_bounds} has volume {bounds_volume}, "
                f"expected {len(self.visible_chips)}"
            )
        if not 1 <= self.tensor_parallel_size <= len(self.visible_chips):
            raise ValueError(
                f"tensor_parallel_size must be in [1, {len(self.visible_chips)}], got {self.tensor_parallel_size}"
            )


def validate_pool_placements(pool: WorkerPoolConfig, placements: tuple[EnginePlacement, ...]) -> None:
    if pool.vm_count != 1:
        raise ValueError(f"xregion placements require vm_count=1, got {pool.vm_count}")
    if not placements:
        raise ValueError("placements must be non-empty")

    used_chips: set[int] = set()
    for placement in placements:
        out_of_range = [chip for chip in placement.visible_chips if chip >= pool.chips_per_vm]
        if out_of_range:
            raise ValueError(
                f"placement chips {out_of_range} are outside pool {pool.pool_id!r} with {pool.chips_per_vm} chips"
            )
        overlap = used_chips.intersection(placement.visible_chips)
        if overlap:
            raise ValueError(f"placements overlap on chips {sorted(overlap)} in pool {pool.pool_id!r}")
        used_chips.update(placement.visible_chips)


@dataclass
class PoolRun:
    pool: WorkerPoolConfig
    context: ZephyrContext
    future: Future[None] | None = None
    shutdown_requested: bool = False
    error_recorded: bool = False


ShardFn = Callable[[Iterator[int], ShardInfo], Iterator[dict[str, object]]]
MakeShardFn = Callable[[WorkerPoolConfig], ShardFn]


def _context_for_pool(pool: WorkerPoolConfig, heartbeat_timeout: float) -> ZephyrContext:
    return ZephyrContext(
        name=f"xregion-pool-{pool.pool_id}",
        max_workers=pool.num_workers,
        resources=pool.worker_resources,
        coordinator_resources=ResourceConfig(cpu=0.1, ram="1g", preemptible=False),
        stage_runner_factory=InlineRunner,
        heartbeat_timeout=heartbeat_timeout,
        max_execution_retries=0,
    )


def run_pool(
    run: PoolRun,
    *,
    ledger_path: str,
    make_process_shard: MakeShardFn,
) -> None:
    process_shard = make_process_shard(run.pool)
    pipeline = Dataset.from_list(list(range(run.pool.num_workers))).map_shard(process_shard)
    if _is_complete(ledger_path):
        return
    run.context.execute(pipeline)


def _is_complete(ledger_path: str) -> bool:
    summary = ledger.summarize(ledger_path)
    return summary.done == summary.total


def _shutdown_unfinished(runs: tuple[PoolRun, ...]) -> None:
    for run in runs:
        if run.future is None or run.future.done():
            continue
        if not run.shutdown_requested:
            logger.info("Shutting down unfinished xregion pool %s", run.pool.pool_id)
            run.shutdown_requested = True
        run.context.shutdown()


def _wait_for_unfinished(runs: tuple[PoolRun, ...]) -> None:
    for run in runs:
        if run.future is None or run.future.done():
            continue
        try:
            run.future.result()
        except Exception:
            if run.shutdown_requested:
                logger.info("Ignored shutdown error from xregion pool %s", run.pool.pool_id, exc_info=True)
                continue
            raise


def run_worker_pools(
    *,
    worker_pools: tuple[WorkerPoolConfig, ...],
    ledger_path: str,
    make_process_shard: MakeShardFn,
    poll_backoff: float,
    heartbeat_timeout: float,
) -> None:
    if not worker_pools:
        raise ValueError("xregion requires at least one worker pool")

    runs = tuple(
        PoolRun(
            pool=pool,
            context=_context_for_pool(pool, heartbeat_timeout),
        )
        for pool in worker_pools
    )

    first_error: Exception | None = None
    with ThreadPoolExecutor(max_workers=len(runs)) as executor:
        for run in runs:
            run.future = executor.submit(
                run_pool,
                run,
                ledger_path=ledger_path,
                make_process_shard=make_process_shard,
            )

        while True:
            if _is_complete(ledger_path):
                _shutdown_unfinished(runs)
                _wait_for_unfinished(runs)
                return

            all_done = True
            for run in runs:
                assert run.future is not None
                if not run.future.done():
                    all_done = False
                    continue
                if run.error_recorded:
                    continue
                try:
                    run.future.result()
                except Exception as error:
                    run.error_recorded = True
                    logger.warning("xregion pool %s failed before ledger completion", run.pool.pool_id, exc_info=True)
                    if first_error is None:
                        first_error = error

            if all_done:
                break

            time.sleep(poll_backoff)

    summary = ledger.summarize(ledger_path)
    error = RuntimeError(f"xregion incomplete: {summary.done}/{summary.total} chunks done")
    if first_error is not None:
        raise error from first_error
    raise error
