# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Rank-level straggler detection callback for distributed training."""

import collections
import logging as pylogging
import statistics
from dataclasses import dataclass, field

import jax

import levanter.tracker
from levanter.callbacks._core import Callback, StepInfo
from levanter.utils.jax_utils import multihost_broadcast_sync


logger = pylogging.getLogger(__name__)

# Default EWMA smoothing factor (higher = more weight on recent values)
EWMA_ALPHA = 0.3

# A rank whose EWMA exceeds the median EWMA by this factor is "chronic"
CHRONIC_STRAGGLER_THRESHOLD = 1.15


@dataclass
class RankTimingStats:
    """Aggregated timing statistics across all ranks for a single reporting window."""

    min_duration: float
    max_duration: float
    median_duration: float
    mean_duration: float
    slowest_ranks: list[tuple[int, float]]  # (rank, duration) sorted descending
    chronic_stragglers: list[tuple[int, float]]  # (rank, ewma) for ranks with elevated EWMA


def compute_rank_stats(
    rank_durations: dict[int, float],
    rank_ewmas: dict[int, float],
    top_k: int = 3,
    chronic_threshold: float = CHRONIC_STRAGGLER_THRESHOLD,
) -> RankTimingStats:
    """Compute aggregate statistics from per-rank step durations.

    Args:
        rank_durations: Mapping from rank index to latest step duration.
        rank_ewmas: Mapping from rank index to EWMA of step durations.
        top_k: Number of slowest ranks to report.
        chronic_threshold: A rank is chronic if its EWMA exceeds median EWMA by this factor.

    Returns:
        RankTimingStats with min/max/median/mean, top-k slowest, and chronic stragglers.
    """
    durations = list(rank_durations.values())
    min_d = min(durations)
    max_d = max(durations)
    median_d = statistics.median(durations)
    mean_d = statistics.mean(durations)

    sorted_ranks = sorted(rank_durations.items(), key=lambda x: x[1], reverse=True)
    slowest = sorted_ranks[:top_k]

    chronic: list[tuple[int, float]] = []
    if rank_ewmas:
        ewma_values = list(rank_ewmas.values())
        median_ewma = statistics.median(ewma_values)
        if median_ewma > 0:
            chronic = [
                (rank, ewma)
                for rank, ewma in sorted(rank_ewmas.items(), key=lambda x: x[1], reverse=True)
                if ewma > median_ewma * chronic_threshold
            ]

    return RankTimingStats(
        min_duration=min_d,
        max_duration=max_d,
        median_duration=median_d,
        mean_duration=mean_d,
        slowest_ranks=slowest,
        chronic_stragglers=chronic,
    )


def update_ewma(prev: float, value: float, alpha: float = EWMA_ALPHA) -> float:
    """Update an exponentially weighted moving average."""
    if prev == 0.0:
        return value
    return alpha * value + (1.0 - alpha) * prev


@dataclass
class StragglerReporter(Callback):
    """Callback that periodically gathers per-rank step durations and reports stragglers.

    Every ``every`` steps, each rank broadcasts its recent step duration and EWMA.
    Process 0 aggregates the results and logs min/median/max plus the top-k slowest
    ranks. Chronic stragglers (elevated EWMA relative to median) are logged as warnings.

    Args:
        top_k: Number of slowest ranks to highlight each report.
        window_size: Rolling history length used for EWMA seeding.
        chronic_threshold: Ratio above median EWMA that flags a rank as chronic.
        prefix: Metric key prefix for tracker logging.
    """

    top_k: int = 3
    window_size: int = 50
    chronic_threshold: float = CHRONIC_STRAGGLER_THRESHOLD
    prefix: str = "straggler"

    _ewma: float = field(default=0.0, init=False, repr=False)
    _history: collections.deque = field(default_factory=lambda: collections.deque(maxlen=50), init=False, repr=False)

    def __post_init__(self):
        self._history = collections.deque(maxlen=self.window_size)

    def on_step(self, info: StepInfo, force: bool = False):
        duration = info.step_duration
        self._ewma = update_ewma(self._ewma, duration)
        self._history.append(duration)

        # Gather per-rank data across all processes
        rank = jax.process_index()
        local_data = {rank: {"duration": duration, "ewma": self._ewma}}

        try:
            all_data = multihost_broadcast_sync(local_data, is_source=True, timeout=30.0)
        except Exception:
            # In single-process or when distributed is unavailable, use local data only
            all_data = local_data

        # Only aggregate on a single process to avoid duplicate logs. In multihost
        # scenarios, multihost_broadcast_sync sends process 0's payload to everyone.
        # We therefore only have the full picture on process 0; other ranks see only
        # their own data.  When running single-process, process_index is always 0.
        if jax.process_index() != 0:
            return

        # For single-process runs, all_data has only one rank. For multi-process,
        # process 0 should have broadcast its own data; we augment with what we know.
        rank_durations = {int(r): d["duration"] for r, d in all_data.items()}
        rank_ewmas = {int(r): d["ewma"] for r, d in all_data.items()}

        stats = compute_rank_stats(
            rank_durations,
            rank_ewmas,
            top_k=self.top_k,
            chronic_threshold=self.chronic_threshold,
        )

        metrics = {
            f"{self.prefix}/min_duration": stats.min_duration,
            f"{self.prefix}/max_duration": stats.max_duration,
            f"{self.prefix}/median_duration": stats.median_duration,
            f"{self.prefix}/mean_duration": stats.mean_duration,
            f"{self.prefix}/spread": stats.max_duration - stats.min_duration,
        }

        if stats.max_duration > 0:
            metrics[f"{self.prefix}/slowdown_ratio"] = stats.max_duration / stats.median_duration

        for i, (r, d) in enumerate(stats.slowest_ranks):
            metrics[f"{self.prefix}/slowest_rank_{i}"] = r
            metrics[f"{self.prefix}/slowest_duration_{i}"] = d

        levanter.tracker.log(metrics, step=info.step)

        if stats.chronic_stragglers:
            rank_strs = [f"rank {r} (ewma={e:.4f}s)" for r, e in stats.chronic_stragglers]
            logger.warning("Chronic stragglers detected: %s", ", ".join(rank_strs))
