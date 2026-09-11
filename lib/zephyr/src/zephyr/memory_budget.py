# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Memory-budget arithmetic for Zephyr's scatter/reduce shuffle.

The scatter write path (a flush) and external-sort read path (a merge pass)
bound peak memory with additive models. A flush uses:

    peak RSS = baseline RSS + fixed operation overhead + R * bytes at risk

A merge separates memory associated with streaming rows, Polars workers
buffering input shards, and intermediate spill runs:

    peak RSS = baseline RSS + fixed operation overhead
             + expanded_streaming_batch_bytes
             + R_input * buffered_input_payload_outside_streaming_batch
             + R_thread * active_threads * mean_input_payload
             + R_spill * reducer_payload  # only when fan_in < total_chunks

``baseline RSS`` is sampled in the shard subprocess immediately before the
flush or merge call, so it captures whatever the process has already loaded
by then: Polars (its thread pool and lazy-scan machinery are touched while
building the input frames) plus the rest of the shard's normal working set.
Zephyr shard subprocesses never import JAX, so training-side memory is not a
factor here. The operation may use up to ``safety_fraction`` of the task's
memory budget.

The base coefficients below are rounded envelopes from 80 write measurements,
a 36-cell sufficient-shard read matrix, and eight held-out read cells on Iris.
The buffered-input coefficient comes from the lower bound imposed by an
incident-shaped OOM and a passing bounded-fan-in rerun. End-to-end cgroup,
wide-merge, and ferry results are recorded in
``.agents/projects/2026-08-11_zephyr_shuffle_memory_budget/results.md``.

Tasks whose baseline plus fixed overhead exceeds the safety budget use the
smallest executable flush or merge unit. That boundary cannot enforce the
safety fraction, but it avoids rejecting tiny shuffles that do not incur the
fitted large-operation overhead.
"""

import math

# Rounded upper envelope from 80 Iris scatter-flush measurements. The fixed
# term absorbs the scale-dependent growth in the zero-intercept ratio.
FIXED_OVERHEAD_WRITE_BYTES = 320 * 2**20
R_WRITE = 4.9

# Conservative upper envelope over the production-path cluster matrix. Small
# and medium rows expand by at most 8x. Wide rows instead follow 2.4 payload
# copies plus 5.5 KiB of row-local state, whichever estimate is smaller.
FIXED_OVERHEAD_READ_BYTES = 80 * 2**20
R_READ_MAX = 8.0
R_READ_PAYLOAD = 2.4
READ_ROW_OVERHEAD_BYTES = int(5.5 * 2**10)
# Rounded up from the 3.18 lower bound implied by the 191-input incident OOM.
R_READ_BUFFERED_INPUT_PAYLOAD = 3.2
R_READ_THREAD_SHARD = 1.1
R_READ_SPILL_PAYLOAD = 1.1

# A ceiling, not a target: the growth budget is derived from this fraction of
# task memory, so the operation may use up to but not necessarily this much.
# Both paths passed three end-to-end Iris repetitions run at 0.85, 0.90, and
# 0.95; 0.90 is kept for the predeclared 5% task-memory reserve.
SAFETY_FRACTION_WRITE = 0.9
SAFETY_FRACTION_READ = 0.9

# Rows held per external-sort merge input at once; also Polars' streaming
# chunk size for both the in-memory and external-sort merge paths.
STREAMING_CHUNK_SIZE_ROWS = 10_000
MIN_MERGE_FAN_IN = 2


def _growth_budget_bytes(
    task_memory_bytes: int,
    baseline_rss_bytes: int,
    safety_fraction: float,
    fixed_overhead_bytes: int,
) -> int:
    if task_memory_bytes <= 0:
        raise ValueError(f"task_memory_bytes must be positive, got {task_memory_bytes}")
    if baseline_rss_bytes < 0:
        raise ValueError(f"baseline_rss_bytes must be nonnegative, got {baseline_rss_bytes}")

    growth_budget = int(task_memory_bytes * safety_fraction) - baseline_rss_bytes - fixed_overhead_bytes
    # The writer cannot flush before receiving its first DataFrame. Saturating
    # at one byte makes tasks below the fitted envelope flush that frame
    # immediately instead of rejecting small shuffles outright.
    return max(1, growth_budget)


def write_flush_threshold_bytes(task_memory_bytes: int, baseline_rss_bytes: int) -> int:
    """Buffered ``DataFrame.estimated_size()`` at which a scatter writer should flush."""
    growth_budget = _growth_budget_bytes(
        task_memory_bytes,
        baseline_rss_bytes,
        SAFETY_FRACTION_WRITE,
        FIXED_OVERHEAD_WRITE_BYTES,
    )
    return max(1, int(growth_budget / R_WRITE))


def read_merge_growth_bytes(
    fan_in: int,
    avg_item_bytes: float,
    total_chunks: int,
    shard_payload_bytes: float,
    polars_threads: int,
    streaming_chunk_size_rows: int = STREAMING_CHUNK_SIZE_ROWS,
) -> float:
    """Predict RSS growth for one sorted-merge plan.

    ``streaming_chunk_size_rows`` defaults to the module constant that
    production also uses to configure Polars. Pass it explicitly only when
    the merge is actually configured with a different streaming chunk size,
    so the prediction matches the batch size the merge runs with.
    """
    active_rows = min(fan_in * streaming_chunk_size_rows, shard_payload_bytes / avg_item_bytes)
    batch_bytes = active_rows * avg_item_bytes
    expanded_batch_bytes = min(
        R_READ_MAX * batch_bytes,
        R_READ_PAYLOAD * batch_bytes + READ_ROW_OVERHEAD_BYTES * active_rows,
    )
    mean_chunk_payload_bytes = shard_payload_bytes / total_chunks
    active_input_payload_bytes = min(shard_payload_bytes, fan_in * mean_chunk_payload_bytes)
    buffered_input_payload_bytes = max(0.0, active_input_payload_bytes - batch_bytes)
    thread_shard_bytes = min(polars_threads, fan_in) * mean_chunk_payload_bytes
    spill_bytes = R_READ_SPILL_PAYLOAD * shard_payload_bytes if fan_in < total_chunks else 0.0
    return (
        FIXED_OVERHEAD_READ_BYTES
        + expanded_batch_bytes
        + R_READ_BUFFERED_INPUT_PAYLOAD * buffered_input_payload_bytes
        + R_READ_THREAD_SHARD * thread_shard_bytes
        + spill_bytes
    )


def read_merge_fan_in(
    task_memory_bytes: int,
    baseline_rss_bytes: int,
    avg_item_bytes: float,
    total_chunks: int,
    shard_payload_bytes: float,
    polars_threads: int,
    streaming_chunk_size_rows: int = STREAMING_CHUNK_SIZE_ROWS,
) -> int:
    """Return the maximum frames a merge may combine within the task budget."""
    if avg_item_bytes <= 0:
        raise ValueError(f"avg_item_bytes must be positive, got {avg_item_bytes}")
    if total_chunks <= 0:
        raise ValueError(f"total_chunks must be positive, got {total_chunks}")
    if shard_payload_bytes <= 0:
        raise ValueError(f"shard_payload_bytes must be positive, got {shard_payload_bytes}")
    if polars_threads <= 0:
        raise ValueError(f"polars_threads must be positive, got {polars_threads}")
    if task_memory_bytes <= 0:
        raise ValueError(f"task_memory_bytes must be positive, got {task_memory_bytes}")
    if baseline_rss_bytes < 0:
        raise ValueError(f"baseline_rss_bytes must be nonnegative, got {baseline_rss_bytes}")

    def predicted_growth_bytes(fan_in: int) -> float:
        return read_merge_growth_bytes(
            fan_in,
            avg_item_bytes,
            total_chunks,
            shard_payload_bytes,
            polars_threads,
            streaming_chunk_size_rows,
        )

    growth_budget = int(task_memory_bytes * SAFETY_FRACTION_READ) - baseline_rss_bytes
    direct_fan_in = max(MIN_MERGE_FAN_IN, total_chunks)
    if predicted_growth_bytes(direct_fan_in) <= growth_budget:
        return direct_fan_in

    if predicted_growth_bytes(MIN_MERGE_FAN_IN) > growth_budget:
        # A merge cannot use fewer than two inputs. Below the fitted envelope,
        # use that minimum so small shuffles still make progress.
        return MIN_MERGE_FAN_IN

    low = MIN_MERGE_FAN_IN
    high = max(MIN_MERGE_FAN_IN, total_chunks - 1)
    while low < high:
        candidate = (low + high + 1) // 2
        if predicted_growth_bytes(candidate) <= growth_budget:
            low = candidate
        else:
            high = candidate - 1
    return low


def polars_thread_count(task_cpu: float) -> int:
    """Return the Polars thread-pool size for a task's CPU allocation."""
    return max(1, math.ceil(task_cpu))
