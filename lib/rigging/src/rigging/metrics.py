# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Render Prometheus samples for process-local status pages."""

import html
import math
import typing
from collections.abc import Sequence
from dataclasses import dataclass, field

from prometheus_client.samples import Sample


class FamilySample(typing.NamedTuple):
    """One registry sample, tagged with the family it came from."""

    family: str
    kind: str
    """The family's metric type: counter, gauge, histogram, summary, info."""
    sample: Sample


@dataclass
class _HistogramAccumulator:
    family: str
    labels: tuple[tuple[str, str], ...]
    buckets: list[tuple[str, float]] = field(default_factory=list)
    counts: list[float] = field(default_factory=list)
    sums: list[float] = field(default_factory=list)
    malformed: bool = False


@dataclass(frozen=True)
class _HistogramBucket:
    lower: str | None
    upper: str
    population: float


@dataclass(frozen=True)
class _HistogramSummary:
    family: str
    labels: tuple[tuple[str, str], ...]
    buckets: tuple[_HistogramBucket, ...]
    count: float | None
    average: float | None
    malformed: bool
    truncated: bool


_SPARK_CHARACTERS = "▁▂▃▄▅▆▇█"
_POSITIVE_INFINITY = "+Inf"
_HISTOGRAM_COUNT_TOLERANCE = 1e-12


def _histogram_summary(histogram: _HistogramAccumulator) -> _HistogramSummary:
    malformed = histogram.malformed or len(histogram.counts) > 1 or len(histogram.sums) > 1
    parsed: list[tuple[float, str, float]] = []
    infinity: tuple[str, float] | None = None
    seen_bounds: set[float] = set()

    for raw_bound, cumulative in histogram.buckets:
        if not math.isfinite(cumulative) or cumulative < 0:
            malformed = True
            continue
        if raw_bound == _POSITIVE_INFINITY:
            if infinity is not None:
                malformed = True
            infinity = (raw_bound, cumulative)
            continue
        try:
            bound = float(raw_bound)
        except ValueError:
            malformed = True
            continue
        if not math.isfinite(bound) or bound in seen_bounds:
            malformed = True
            continue
        seen_bounds.add(bound)
        parsed.append((bound, raw_bound, cumulative))

    parsed.sort(key=lambda bucket: bucket[0])
    cumulative_buckets = [(raw_bound, cumulative) for _, raw_bound, cumulative in parsed]
    if infinity is not None:
        cumulative_buckets.append(infinity)

    previous = 0.0
    for _, cumulative in cumulative_buckets:
        if cumulative < previous:
            malformed = True
        previous = cumulative

    count = histogram.counts[0] if histogram.counts else None
    if count is not None and (not math.isfinite(count) or count < 0):
        malformed = True

    if infinity is not None:
        infinity_count = infinity[1]
        if count is None:
            count = infinity_count
        elif not math.isclose(
            count,
            infinity_count,
            rel_tol=_HISTOGRAM_COUNT_TOLERANCE,
            abs_tol=_HISTOGRAM_COUNT_TOLERANCE,
        ):
            malformed = True
    elif count is not None:
        if cumulative_buckets and count < cumulative_buckets[-1][1]:
            malformed = True
        else:
            cumulative_buckets.append((_POSITIVE_INFINITY, count))

    if malformed:
        return _HistogramSummary(
            family=histogram.family,
            labels=histogram.labels,
            buckets=(),
            count=None,
            average=None,
            malformed=True,
            truncated=False,
        )

    buckets: list[_HistogramBucket] = []
    previous_bound: str | None = None
    previous_cumulative = 0.0
    for bound, cumulative in cumulative_buckets:
        buckets.append(
            _HistogramBucket(
                lower=previous_bound,
                upper=bound,
                population=cumulative - previous_cumulative,
            )
        )
        previous_bound = bound
        previous_cumulative = cumulative

    total = histogram.sums[0] if histogram.sums else None
    average = total / count if total is not None and math.isfinite(total) and count is not None and count > 0 else None
    return _HistogramSummary(
        family=histogram.family,
        labels=histogram.labels,
        buckets=tuple(buckets),
        count=count,
        average=average,
        malformed=False,
        truncated=count is None,
    )


def _format_metric_number(value: float) -> str:
    if value.is_integer():
        return str(int(value))
    return f"{value:.6g}"


def _spark_character(population: float, maximum: float) -> str:
    if population == 0 or maximum == 0:
        return "·"
    level = math.ceil(math.log1p(population) / math.log1p(maximum) * len(_SPARK_CHARACTERS)) - 1
    return _SPARK_CHARACTERS[max(0, min(level, len(_SPARK_CHARACTERS) - 1))]


def _render_histogram_value(summary: _HistogramSummary) -> str:
    if summary.malformed:
        return '<span data-histogram-state="malformed">malformed histogram</span>'

    parts: list[str] = []
    if summary.count is not None:
        parts.append(f"n={_format_metric_number(summary.count)}")
    if summary.average is not None:
        parts.append(f"avg={_format_metric_number(summary.average)}")
    if summary.truncated:
        parts.append("truncated")

    if summary.buckets:
        maximum = max(bucket.population for bucket in summary.buckets)
        spark = []
        for bucket in summary.buckets:
            if bucket.lower is None:
                interval = f"≤ {bucket.upper}"
            else:
                interval = f"({bucket.lower}, {bucket.upper}]"
            title = html.escape(f"{interval}: {_format_metric_number(bucket.population)}")
            character = _spark_character(bucket.population, maximum)
            spark.append(f'<span title="{title}">{character}</span>')
        parts.append(f'<span class="histogram-spark">{"".join(spark)}</span>')

    return " ".join(parts) or "(empty histogram)"


def render_metric_rows(metric_samples: Sequence[FamilySample]) -> list[str]:
    """Render registry samples as HTML table rows."""
    histogram_groups: dict[tuple[str, tuple[tuple[str, str], ...]], _HistogramAccumulator] = {}
    display_order: list[FamilySample | tuple[str, tuple[tuple[str, str], ...]]] = []

    for family_sample in metric_samples:
        if family_sample.kind != "histogram":
            display_order.append(family_sample)
            continue

        family = family_sample.family
        sample = family_sample.sample
        if sample.name == f"{family}_created":
            continue

        labels = tuple(sorted((key, value) for key, value in sample.labels.items() if key != "le"))
        key = (family, labels)
        group = histogram_groups.get(key)
        if group is None:
            group = _HistogramAccumulator(family=family, labels=labels)
            histogram_groups[key] = group
            display_order.append(key)

        if sample.name == f"{family}_bucket":
            bound = sample.labels.get("le")
            if bound is None:
                group.malformed = True
            else:
                group.buckets.append((bound, float(sample.value)))
        elif sample.name == f"{family}_count":
            group.counts.append(float(sample.value))
        elif sample.name == f"{family}_sum":
            group.sums.append(float(sample.value))
        else:
            group.malformed = True

    rows = []
    for item in display_order:
        if isinstance(item, FamilySample):
            sample = item.sample
            labels = ",".join(f"{key}={value}" for key, value in sorted(sample.labels.items()))
            rows.append(
                f"<tr><td>{html.escape(sample.name)}</td><td>{html.escape(item.kind)}</td>"
                f"<td>{html.escape(labels)}</td><td>{sample.value!r}</td></tr>"
            )
            continue

        summary = _histogram_summary(histogram_groups[item])
        labels = ",".join(f"{key}={value}" for key, value in summary.labels)
        rows.append(
            f"<tr><td>{html.escape(summary.family)}</td><td>histogram</td>"
            f"<td>{html.escape(labels)}</td><td>{_render_histogram_value(summary)}</td></tr>"
        )
    return rows
