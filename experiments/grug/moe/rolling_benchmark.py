# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Small, deterministic primitives for a rolling closed-loop benchmark."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from prometheus_client.parser import text_string_to_metric_families


@dataclass(frozen=True)
class PlateauRequirements:
    target_concurrency: int
    minimum_seconds: float
    minimum_generated_tokens: int
    required_request_ids: frozenset[str]

    def __post_init__(self) -> None:
        if self.target_concurrency <= 0:
            raise ValueError("target concurrency must be positive")
        if self.minimum_seconds < 0:
            raise ValueError("minimum plateau seconds cannot be negative")
        if self.minimum_generated_tokens < 0:
            raise ValueError("minimum generated tokens cannot be negative")
        if not self.required_request_ids:
            raise ValueError("a plateau requires at least one manifest request")

    @property
    def minimum_in_flight(self) -> int:
        return math.ceil(0.95 * self.target_concurrency)


@dataclass
class PlateauWindow:
    """Track one valid window, discarding it after any load-floor breach."""

    requirements: PlateauRequirements
    opened_at: float | None = None
    generation_counter_start: float | None = None
    prompt_counter_start: float | None = None
    in_flight_samples: list[int] = field(default_factory=list)
    completed_request_ids: set[str] = field(default_factory=set)
    successful_requests: int = 0
    failed_requests: int = 0
    client_generated_tokens: int = 0
    cohort_completions: dict[str, int] = field(default_factory=dict)
    discarded: list[dict[str, Any]] = field(default_factory=list)

    @property
    def is_open(self) -> bool:
        return self.opened_at is not None

    def observe_in_flight(
        self,
        *,
        now: float,
        in_flight: int,
        generation_counter: float | None = None,
        prompt_counter: float | None = None,
    ) -> str | None:
        """Observe load and return ``opened`` or ``discarded`` on a transition."""
        if in_flight < 0 or in_flight > self.requirements.target_concurrency:
            raise ValueError("in-flight sample is outside [0, target_concurrency]")
        if self.is_open and in_flight < self.requirements.minimum_in_flight:
            assert self.opened_at is not None
            self.discarded.append(
                {
                    "opened_at": self.opened_at,
                    "discarded_at": now,
                    "elapsed_seconds": now - self.opened_at,
                    "minimum_observed_in_flight": min([*self.in_flight_samples, in_flight]),
                    "reason": "in_flight_below_95_percent",
                }
            )
            self._reset_active_window()
            return "discarded"
        if not self.is_open and in_flight >= self.requirements.minimum_in_flight:
            if generation_counter is None or prompt_counter is None:
                raise ValueError("opening a plateau requires generation and prompt counter snapshots")
            self.opened_at = now
            self.generation_counter_start = generation_counter
            self.prompt_counter_start = prompt_counter
            self.in_flight_samples.append(in_flight)
            return "opened"
        if self.is_open:
            self.in_flight_samples.append(in_flight)
        return None

    def record_completion(
        self,
        *,
        request_id: str,
        cohort: str,
        completion_tokens: int,
        succeeded: bool,
    ) -> None:
        if completion_tokens < 0:
            raise ValueError("completion tokens cannot be negative")
        if not self.is_open:
            return
        if succeeded:
            self.successful_requests += 1
            self.client_generated_tokens += completion_tokens
            self.completed_request_ids.add(request_id)
            self.cohort_completions[cohort] = self.cohort_completions.get(cohort, 0) + 1
        else:
            self.failed_requests += 1

    def ready_to_close(self, *, now: float, in_flight: int, generation_counter: float) -> bool:
        if not self.is_open:
            return False
        assert self.opened_at is not None
        assert self.generation_counter_start is not None
        return (
            in_flight == self.requirements.target_concurrency
            and now - self.opened_at >= self.requirements.minimum_seconds
            and generation_counter - self.generation_counter_start >= self.requirements.minimum_generated_tokens
            and self.completed_request_ids == set(self.requirements.required_request_ids)
            and self.failed_requests == 0
        )

    def close(
        self,
        *,
        now: float,
        in_flight: int,
        generation_counter: float,
        prompt_counter: float,
    ) -> dict[str, Any]:
        if not self.ready_to_close(now=now, in_flight=in_flight, generation_counter=generation_counter):
            raise ValueError("plateau floors have not all passed while the queue is full")
        assert self.opened_at is not None
        assert self.generation_counter_start is not None
        assert self.prompt_counter_start is not None
        elapsed = now - self.opened_at
        generated = round(generation_counter - self.generation_counter_start)
        prompt = round(prompt_counter - self.prompt_counter_start)
        samples = [*self.in_flight_samples, in_flight]
        return {
            "opened_at": self.opened_at,
            "closed_at": now,
            "elapsed_seconds": elapsed,
            "generated_tokens": generated,
            "processed_prompt_tokens": prompt,
            "generation_counter_start": self.generation_counter_start,
            "generation_counter_end": generation_counter,
            "prompt_counter_start": self.prompt_counter_start,
            "prompt_counter_end": prompt_counter,
            "target_concurrency": self.requirements.target_concurrency,
            "minimum_required_in_flight": self.requirements.minimum_in_flight,
            "in_flight": {
                "samples": len(samples),
                "min": min(samples),
                "mean": sum(samples) / len(samples),
                "max": max(samples),
                "at_close": in_flight,
            },
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "client_generated_tokens": self.client_generated_tokens,
            "cohort_completions": dict(sorted(self.cohort_completions.items())),
            "manifest": {
                "expected": len(self.requirements.required_request_ids),
                "observed": len(self.completed_request_ids),
                "passed": self.completed_request_ids == set(self.requirements.required_request_ids),
            },
            "discarded_windows": [*self.discarded],
        }

    def _reset_active_window(self) -> None:
        self.opened_at = None
        self.generation_counter_start = None
        self.prompt_counter_start = None
        self.in_flight_samples.clear()
        self.completed_request_ids.clear()
        self.successful_requests = 0
        self.failed_requests = 0
        self.client_generated_tokens = 0
        self.cohort_completions.clear()


@dataclass
class FrozenSlot:
    """One live slot whose request order is fixed independent of completion order."""

    slot_id: int
    cohort: str
    requests: tuple[dict[str, Any], ...]
    request_index: int
    stride: int

    def next_request(self) -> dict[str, Any]:
        request = self.requests[self.request_index]
        self.request_index = (self.request_index + self.stride) % len(self.requests)
        return request


def frozen_cohort_slots(requests: list[dict[str, Any]], *, target_concurrency: int) -> list[FrozenSlot]:
    """Assign equal live slots to cohorts and freeze each slot's cyclic order."""
    cohorts: dict[str, list[dict[str, Any]]] = {}
    for request in requests:
        cohort = str(request["cohort"])
        cohorts.setdefault(cohort, []).append(request)
    if not cohorts:
        raise ValueError("the workload has no requests")
    if target_concurrency <= 0 or target_concurrency % len(cohorts):
        raise ValueError("target concurrency must divide evenly across cohorts")
    slots_per_cohort = target_concurrency // len(cohorts)
    slots: list[FrozenSlot] = []
    for cohort, cohort_requests in cohorts.items():
        frozen_requests = tuple(cohort_requests)
        for cohort_slot in range(slots_per_cohort):
            slots.append(
                FrozenSlot(
                    slot_id=len(slots),
                    cohort=cohort,
                    requests=frozen_requests,
                    request_index=cohort_slot % len(frozen_requests),
                    stride=slots_per_cohort,
                )
            )
    return slots


@dataclass(frozen=True)
class PrometheusSample:
    """One Prometheus sample with its labels preserved."""

    name: str
    labels: tuple[tuple[str, str], ...]
    value: float

    def label(self, name: str) -> str | None:
        return dict(self.labels).get(name)

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "labels": dict(self.labels), "value": self.value}


def parse_labeled_prometheus(text: str) -> list[PrometheusSample]:
    """Parse an exposition while retaining the labels needed for per-rank evidence."""
    return [
        PrometheusSample(
            name=sample.name,
            labels=tuple(sorted((str(key), str(value)) for key, value in sample.labels.items())),
            value=float(sample.value),
        )
        for family in text_string_to_metric_families(text)
        for sample in family.samples
    ]


def prometheus_value(
    samples: list[PrometheusSample],
    metric: str,
    *,
    labels: dict[str, str] | None = None,
) -> float:
    """Sum one scalar metric, accepting Prometheus' automatic counter suffix."""
    required = labels or {}

    def matches(sample: PrometheusSample, name: str) -> bool:
        sample_labels = dict(sample.labels)
        return sample.name == name and all(sample_labels.get(key) == value for key, value in required.items())

    exact = [sample.value for sample in samples if matches(sample, metric)]
    if exact:
        return sum(exact)
    return sum(sample.value for sample in samples if matches(sample, f"{metric}_total"))


def prometheus_values_by_label(
    samples: list[PrometheusSample],
    metric: str,
    *,
    label: str,
) -> dict[str, float]:
    """Sum one metric independently for every value of one retained label."""
    names = (metric, f"{metric}_total")
    selected_name = next((name for name in names if any(sample.name == name for sample in samples)), names[-1])
    values: dict[str, float] = {}
    for sample in samples:
        if sample.name != selected_name:
            continue
        label_value = sample.label(label)
        if label_value is None:
            continue
        values[label_value] = values.get(label_value, 0.0) + sample.value
    return dict(sorted(values.items()))


def histogram_quantile_delta(
    before: list[PrometheusSample],
    after: list[PrometheusSample],
    metric: str,
    quantile: float,
) -> float | None:
    """Estimate a quantile from the aggregate histogram observations in one window."""
    if not 0 <= quantile <= 1:
        raise ValueError("histogram quantile must be in [0, 1]")

    def buckets(samples: list[PrometheusSample]) -> dict[float, float]:
        result: dict[float, float] = {}
        for sample in samples:
            if sample.name != f"{metric}_bucket":
                continue
            raw_bound = sample.label("le")
            if raw_bound is None:
                continue
            bound = math.inf if raw_bound == "+Inf" else float(raw_bound)
            result[bound] = result.get(bound, 0.0) + sample.value
        return result

    before_buckets = buckets(before)
    after_buckets = buckets(after)
    deltas = {bound: after_count - before_buckets.get(bound, 0.0) for bound, after_count in after_buckets.items()}
    if not deltas:
        return None
    ordered = sorted(deltas.items())
    total = next((count for bound, count in ordered if math.isinf(bound)), ordered[-1][1])
    if total <= 0:
        return None
    target = quantile * total
    previous_bound = 0.0
    previous_count = 0.0
    for bound, cumulative_count in ordered:
        if cumulative_count < target:
            if not math.isinf(bound):
                previous_bound = bound
            previous_count = cumulative_count
            continue
        if math.isinf(bound):
            return previous_bound
        bucket_count = cumulative_count - previous_count
        if bucket_count <= 0:
            return bound
        return previous_bound + (bound - previous_bound) * (target - previous_count) / bucket_count
    return None


__all__ = [
    "FrozenSlot",
    "PlateauRequirements",
    "PlateauWindow",
    "PrometheusSample",
    "frozen_cohort_slots",
    "histogram_quantile_delta",
    "parse_labeled_prometheus",
    "prometheus_value",
    "prometheus_values_by_label",
]
