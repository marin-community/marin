# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared live progress reporting for alignment inference stages."""

from __future__ import annotations

import logging
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

_SAMPLE_WINDOW = 8


def progress_log_interval(total_items: int, batch_size: int) -> int:
    if total_items <= 0:
        return 1
    return max(batch_size, 1)


def _compact_number(value: float) -> str:
    abs_value = abs(value)
    if abs_value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if abs_value >= 1_000:
        return f"{value / 1_000:.1f}k"
    if abs_value >= 100:
        return f"{value:.0f}"
    if abs_value >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def _format_duration(seconds: float) -> str:
    total_seconds = max(round(seconds), 0)
    if total_seconds < 60:
        return f"{total_seconds}s"
    minutes, seconds = divmod(total_seconds, 60)
    if minutes < 60:
        return f"{minutes}m {seconds:02d}s"
    hours, minutes = divmod(minutes, 60)
    if hours < 24:
        return f"{hours}h {minutes:02d}m"
    days, hours = divmod(hours, 24)
    return f"{days}d {hours:02d}h"


def _items_per_second(rate: float | None) -> str | None:
    if rate is None or rate <= 0:
        return None
    if rate >= 10:
        return f"{rate:.1f} items/s"
    return f"{rate:.2f} items/s"


def _token_rate(label: str, rate: Any) -> str | None:
    if not isinstance(rate, int | float) or rate <= 0:
        return None
    return f"{label} {_compact_number(float(rate))} tok/s"


def vllm_stage_metrics_provider(session: Any, *, stage_name: str) -> Callable[[], dict[str, Any] | None]:
    """Return a callback that reads live metrics for one logical vLLM stage."""

    def load_metrics() -> dict[str, Any] | None:
        if not hasattr(session, "metrics_snapshot"):
            return None
        snapshot = session.metrics_snapshot()
        if not isinstance(snapshot, dict):
            return None
        stages = snapshot.get("stages")
        if not isinstance(stages, dict):
            return None
        stage_metrics = stages.get(stage_name)
        return stage_metrics if isinstance(stage_metrics, dict) else None

    return load_metrics


@dataclass
class LiveProgressReporter:
    """Log smoothed live progress with optional token-throughput metrics."""

    stage_name: str
    total_items: int
    batch_size: int
    metrics_provider: Callable[[], dict[str, Any] | None] | None = None
    initial_completed_items: int = 0
    _start_time: float = field(init=False, default_factory=time.perf_counter)
    _last_logged_items: int = field(init=False)
    _samples: deque[tuple[float, int]] = field(init=False, default_factory=lambda: deque(maxlen=_SAMPLE_WINDOW))

    def __post_init__(self) -> None:
        self._last_logged_items = self.initial_completed_items
        self._samples.append((self._start_time, self.initial_completed_items))

    def maybe_log(
        self,
        completed_items: int,
        *,
        details: Sequence[str] = (),
        force: bool = False,
    ) -> None:
        if self.total_items <= 0:
            return

        now = time.perf_counter()
        if completed_items != self._samples[-1][1]:
            self._samples.append((now, completed_items))

        interval = progress_log_interval(self.total_items, self.batch_size)
        should_log = (
            force or completed_items == self.total_items or completed_items - self._last_logged_items >= interval
        )
        if not should_log:
            return

        percent = 100.0 * completed_items / self.total_items
        extras = [detail for detail in details if detail]
        rate = self._smoothed_rate(now)
        if item_rate := _items_per_second(rate):
            extras.append(item_rate)
        if rate is not None and rate > 0 and completed_items < self.total_items:
            extras.append(f"ETA {_format_duration((self.total_items - completed_items) / rate)}")

        if self.metrics_provider is not None:
            metrics = self.metrics_provider() or {}
            if prompt_rate := _token_rate("prompt", metrics.get("input_tokens_per_second")):
                extras.append(prompt_rate)
            if completion_rate := _token_rate("completion", metrics.get("output_tokens_per_second")):
                extras.append(completion_rate)

        message = f"{self.stage_name} progress: {completed_items}/{self.total_items} ({percent:.1f}%)"
        if extras:
            message += f" [{', '.join(extras)}]"
        logger.info(message)
        self._last_logged_items = completed_items

    def _smoothed_rate(self, now: float) -> float | None:
        oldest_time = self._start_time
        oldest_completed = self.initial_completed_items
        for sample_time, sample_completed in self._samples:
            if sample_completed < self._samples[-1][1]:
                oldest_time = sample_time
                oldest_completed = sample_completed
                break

        newest_time, newest_completed = self._samples[-1]
        elapsed = newest_time - oldest_time
        delta_items = newest_completed - oldest_completed
        if elapsed > 0 and delta_items > 0:
            return delta_items / elapsed

        total_elapsed = now - self._start_time
        delta_from_start = newest_completed - self.initial_completed_items
        if total_elapsed <= 0 or delta_from_start <= 0:
            return None
        return delta_from_start / total_elapsed
