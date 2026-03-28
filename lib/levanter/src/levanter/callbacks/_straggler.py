# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Rank-level straggler detection callback for distributed training.

Periodically gathers per-rank step durations across all processes, reports
min/median/max and identifies the top-k chronically slow ranks using an
exponentially weighted moving average (EWMA).
"""

import collections
import json
import logging
import statistics
from dataclasses import dataclass, field
from typing import Optional

import jax

import levanter.tracker
from levanter.callbacks._core import Callback, StepInfo

logger = logging.getLogger(__name__)

# Smoothing factor for EWMA.  0.1 keeps ~10-step memory, weighting recent
# observations heavily while still distinguishing chronic lag from one-off stalls.
DEFAULT_EWMA_ALPHA = 0.1
DEFAULT_HISTORY_LENGTH = 50
DEFAULT_TOP_K = 3


@dataclass
class StragglerReporter(Callback):
    """Callback that detects and reports rank-level stragglers.

    Every time ``on_step`` is called the reporter records the local rank's
    ``step_duration`` into a rolling window and broadcasts it to all ranks.
    Rank 0 aggregates the durations, computes summary statistics
    (min/median/max), maintains a per-rank EWMA, and logs the top-k slowest
    ranks to the standard tracker.

    This callback is designed to be registered with ``every=N`` on the
    trainer so that the cross-rank communication cost is amortised.
    """

    ewma_alpha: float = DEFAULT_EWMA_ALPHA
    history_length: int = DEFAULT_HISTORY_LENGTH
    top_k: int = DEFAULT_TOP_K
    prefix: str = "straggler"

    # Mutable state — not constructor args.
    _rank_ewma: dict[int, float] = field(default_factory=dict, init=False, repr=False)
    _local_durations: collections.deque = field(init=False, repr=False)

    def __post_init__(self):
        self._local_durations = collections.deque(maxlen=self.history_length)

    def on_step(self, info: StepInfo, force: bool = False):
        duration = info.step_duration
        self._local_durations.append(duration)

        rank = jax.process_index()
        process_count = jax.process_count()

        all_durations = _gather_durations(rank, duration, process_count)

        if rank == 0 and all_durations is not None:
            self._report(all_durations, info.step)

    def _report(self, all_durations: dict[int, float], step: int):
        """Aggregate durations and log straggler metrics (rank 0 only)."""
        durations = list(all_durations.values())

        min_dur = min(durations)
        max_dur = max(durations)
        med_dur = statistics.median(durations)

        # Update per-rank EWMA
        alpha = self.ewma_alpha
        for r, d in all_durations.items():
            prev = self._rank_ewma.get(r)
            if prev is None:
                self._rank_ewma[r] = d
            else:
                self._rank_ewma[r] = alpha * d + (1 - alpha) * prev

        # Identify top-k slowest by EWMA
        sorted_ranks = sorted(self._rank_ewma.items(), key=lambda kv: kv[1], reverse=True)
        effective_k = min(self.top_k, len(sorted_ranks))
        top_slow = sorted_ranks[:effective_k]

        metrics: dict[str, float] = {
            f"{self.prefix}/min_duration": min_dur,
            f"{self.prefix}/median_duration": med_dur,
            f"{self.prefix}/max_duration": max_dur,
            f"{self.prefix}/spread": max_dur - min_dur,
        }

        for i, (r, ewma_val) in enumerate(top_slow):
            metrics[f"{self.prefix}/slow_rank_{i}_id"] = r
            metrics[f"{self.prefix}/slow_rank_{i}_ewma"] = ewma_val

        levanter.tracker.log(metrics, step=step)

        if max_dur - min_dur > med_dur * 0.5:
            slow_summary = ", ".join(f"rank {r} ({ewma_val:.3f}s)" for r, ewma_val in top_slow)
            logger.warning(
                "Straggler detected at step %d: spread=%.3fs (min=%.3f, med=%.3f, max=%.3f). " "Slowest by EWMA: %s",
                step,
                max_dur - min_dur,
                min_dur,
                med_dur,
                max_dur,
                slow_summary,
            )


def _gather_durations(rank: int, duration: float, process_count: int) -> Optional[dict[int, float]]:
    """Gather per-rank durations to rank 0.

    In single-process mode, returns {0: duration} directly.
    In multi-process mode, uses the JAX distributed KV store: each rank
    writes its duration, a barrier synchronizes, then rank 0 reads all.
    Returns the merged dict on rank 0, None on other ranks.
    """
    if process_count == 1:
        return {0: duration}

    import jax._src.distributed as distributed

    client = distributed.global_state.client
    if client is None:
        return {0: duration}

    counter = _next_straggler_counter()
    key = f"LEVANTER_STRAGGLER_{counter}_RANK_{rank}"

    client.key_value_set(key, json.dumps(duration))
    client.wait_at_barrier(f"levanter_straggler_gather_{counter}", timeout_in_ms=60_000)

    if rank == 0:
        result: dict[int, float] = {}
        for r in range(process_count):
            rkey = f"LEVANTER_STRAGGLER_{counter}_RANK_{r}"
            val = client.blocking_key_value_get(rkey, timeout_in_ms=10_000)
            result[r] = json.loads(val)
        return result

    return None


_straggler_counter = 0


def _next_straggler_counter() -> int:
    global _straggler_counter
    _straggler_counter += 1
    return _straggler_counter


def straggler_reporter(
    ewma_alpha: float = DEFAULT_EWMA_ALPHA,
    history_length: int = DEFAULT_HISTORY_LENGTH,
    top_k: int = DEFAULT_TOP_K,
    prefix: str = "straggler",
) -> StragglerReporter:
    """Factory for creating a StragglerReporter callback.

    Usage::

        trainer.add_hook(straggler_reporter(), every=10)
    """
    return StragglerReporter(
        ewma_alpha=ewma_alpha,
        history_length=history_length,
        top_k=top_k,
        prefix=prefix,
    )
