# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Memory-budget arithmetic for Zephyr's scatter/reduce shuffle.

The scatter write path (a flush) and the external-sort read path (a merge
pass) bound their peak memory the same way: an operation's peak RSS growth is
modeled as a single ratio, ``R``, times the estimated bytes it holds ("bytes
at risk"), and the operation is allowed to grow to ``safety_fraction`` of the
task's memory budget. The two ``R`` constants and two ``safety_fraction``
constants below are the only free variables; every function here is derived
from them plus already-available inputs (task memory/CPU, shard byte stats).

``R_WRITE`` comes from a direct measurement: building one scatter buffer of
1.5M rows / 150-byte payloads over 4096 target shards and comparing
``DataFrame.estimated_size()`` against process RSS before and after
``buffer.sort()`` (231.7 MiB estimated, 716.3 MiB RSS post-sort, 190.9 MiB
process baseline). See https://github.com/marin-community/marin/issues/7946.
``R_READ`` and both ``SAFETY_FRACTION`` constants are provisional
placeholders pending calibration against
``lib/zephyr/tests/benchmark_shuffle.py``; that same issue tracks the
calibration work.
"""

import math

# Peak RSS growth during a scatter flush (pl.concat + sort + serialize), per
# byte of buffered DataFrame.estimated_size(). Measured directly:
# https://github.com/marin-community/marin/issues/7946.
R_WRITE = 2.27

# Peak RSS growth during one external-sort merge pass, per byte of
# fan_in * STREAMING_CHUNK_SIZE_ROWS * avg_item_bytes held in flight.
# Placeholder pending calibration: reuses R_WRITE as the closest available
# estimate for a structurally similar Polars operation (sort/merge keeping
# input batches live while building output).
R_READ = R_WRITE

# Fraction of task memory a flush or merge pass is allowed to peak at.
# Placeholders pending calibration.
SAFETY_FRACTION_WRITE = 0.5
SAFETY_FRACTION_READ = 0.5

# Rows held per external-sort merge input at once; also Polars' streaming
# chunk size for both the in-memory and external-sort merge paths.
STREAMING_CHUNK_SIZE_ROWS = 10_000


def write_flush_threshold_bytes(task_memory_bytes: int) -> int:
    """Buffered ``DataFrame.estimated_size()`` at which a scatter writer should flush."""
    if task_memory_bytes <= 0:
        raise ValueError(f"task_memory_bytes must be positive, got {task_memory_bytes}")
    return int(task_memory_bytes * SAFETY_FRACTION_WRITE / R_WRITE)


def read_merge_fan_in(task_memory_bytes: int, avg_item_bytes: float) -> int:
    """Maximum LazyFrames one ``pl.merge_sorted`` call may combine at once.

    Comparing this to a shard's total chunk count tells the caller whether
    the whole shard merges in memory (``total_chunks <= fan_in``) or needs
    ``zephyr.shuffle._merge_sorted_frames`` to spill; the same value bounds
    every pass of that merge.
    """
    if task_memory_bytes <= 0:
        raise ValueError(f"task_memory_bytes must be positive, got {task_memory_bytes}")
    if avg_item_bytes <= 0:
        raise ValueError(f"avg_item_bytes must be positive, got {avg_item_bytes}")
    bytes_per_batch = STREAMING_CHUNK_SIZE_ROWS * avg_item_bytes
    fan_in = math.floor(task_memory_bytes * SAFETY_FRACTION_READ / (R_READ * bytes_per_batch))
    return max(2, fan_in)


def polars_thread_count(task_cpu: float) -> int:
    """Polars thread pool size for a task's CPU allocation."""
    return max(1, math.ceil(task_cpu))
