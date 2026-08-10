# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Eval statistics: what a run measured, over how many items, and how uncertain the answer is.

A benchmark score is a proportion over a finite item set, and a run does not always grade every item
it set out to grade. This module carries both facts as one type -- :class:`Measurement` holds the
sufficient statistics (scored items, correct items, recorded dispersion, and the coverage of the
attempted panel) and the intervals are derived on read, so changing the interval rule never requires
re-deriving stored data.

The estimand is fixed once: **theta, the success rate over the items the run set out to grade.** With
``c = n_scored / n_attempted``, ``theta = c * theta_obs + (1 - c) * theta_miss``. ``theta_obs`` is
estimated from the graded items; ``theta_miss`` is not identified -- a trial that times out is more
likely to be a hard trial -- so it is bounded rather than imputed, and the reported interval widens by
at least ``1 - c``. Admitting a partial item set therefore costs interval width proportional to what
was missed, which is the whole point: a complete-case rate (``k / n_scored``) rewards a run for losing
the items it found hardest, so nothing here ranks on it.

Nothing here knows what a harness writes: :mod:`marin.evaluation.eval_measurements` turns records into
measurements, and this module takes them from there. It is import-light on purpose -- the record
status enum and nothing else, with no marin/levanter/iris imports -- so the dashboard image can vendor
it beside ``records.py``.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum

from marin.evaluation.records import RunStatus

ALPHA = 0.05

# The share of its attempted items a result must have graded to be admitted for display. Comparison
# needs a far stricter floor; see :func:`difference_interval`.
DEFAULT_MIN_COVERAGE = 0.9

# Metrics whose run-level value is a mean of per-item 0/1 outcomes, so (k, n) is recoverable and the
# Wilson interval applies. Everything else (pass@k estimators, partial-credit graders, mean rewards)
# takes the recorded-dispersion path. ``accuracy`` is both evalchemy's chat-native key and Harbor's
# solved-trial rate; both are per-item binary.
BINARY_METRICS = frozenset({"acc", "acc_norm", "exact_match", "accuracy"})

# lm-eval records a task's graded-document count under this key, beside the metrics themselves.
SAMPLE_COUNT_METRIC = "sample_len"
# evalchemy's chat-native benchmarks and Harbor report counts directly instead.
TOTAL_METRICS = ("num_total", "total")

# Bisection bounds for the Imbens-Manski critical value: the two-sided normal quantile when nothing is
# missing, the one-sided quantile in the limit where the unidentified width dominates.
_Z_TWO_SIDED = 1.959963984540054
_Z_ONE_SIDED = 1.6448536269514722
_BISECTION_STEPS = 60


class MetricKind(StrEnum):
    """How a metric's uncertainty is computed."""

    BINARY = "binary"
    """A mean of per-item 0/1 outcomes: the Wilson score interval on (k, n)."""

    CONTINUOUS = "continuous"
    """A mean of per-item scores: the harness-recorded standard error."""


class IntervalKind(StrEnum):
    """What a measurement's interval covers."""

    IDENTIFIED = "identified"
    """Sampling error plus the unidentified contribution of ungraded items: covers theta."""

    SAMPLING_ONLY = "sampling_only"
    """Sampling error alone, because the attempted item count is unreported. Covers theta only if the
    run graded every item it attempted, which the record does not establish."""


class ResultFlag(StrEnum):
    """A property of a measurement that a consumer may want to admit, exclude, or display."""

    ATTRITION = "attrition"
    """Items the run attempted were not graded."""

    ATTRITION_UNREPORTED = "attrition_unreported"
    """The mechanism records no attempted-item count, so completeness cannot be established."""

    CAPPED = "capped"
    """The run declared an item cap, so it measured a prefix of the benchmark rather than all of it."""

    NO_ITEMS = "no_items"
    """A value with no item count behind it: no interval can be computed."""

    NO_DISPERSION = "no_dispersion"
    """A continuous metric whose harness recorded no usable standard error."""

    DEGENERATE_STDERR = "degenerate_stderr"
    """A standard error recorded as exactly zero over more than one item. lm-eval writes this when
    every item agrees; it is a degenerate value, not a missing one, and asserts a certainty the data
    does not support."""

    NO_ANSWERS = "no_answers"
    """No graded item yielded an extractable answer. The score is real -- unextractable answers score
    zero -- but a run in this state is as consistent with a broken grader or a mis-served prompt as
    with a model that cannot do the task, so it is evidence rather than a verdict."""


@dataclass(frozen=True)
class Interval:
    """A closed interval on a score."""

    low: float
    high: float

    @property
    def width(self) -> float:
        return self.high - self.low


@dataclass(frozen=True)
class ScoreInterval:
    """A measurement's interval together with what it covers."""

    low: float
    high: float
    kind: IntervalKind

    @property
    def width(self) -> float:
        return self.high - self.low


@dataclass(frozen=True)
class Coverage:
    """How much of a task's intended item set a run actually graded.

    ``n_attempted`` is the number of items the run set out to grade after any declared cap, and is
    ``None`` when the mechanism reports no such count. ``errors`` is the error-type histogram over
    attempted-but-ungraded items, which is what distinguishes a model's score from the quality of the
    infrastructure that produced it.

    ``n_correct`` is the harness's own count of passing items, so a binary measurement need not
    recover its numerator by inverting a rate; ``n_unanswered`` counts graded items that yielded no
    extractable answer, which is what tells a genuinely-zero score apart from a broken extraction.
    Both are ``None``/zero for a mechanism that reports neither.
    """

    n_scored: int
    n_attempted: int | None = None
    n_correct: int | None = None
    n_unanswered: int = 0
    errors: Mapping[str, int] = field(default_factory=dict)

    @property
    def reported(self) -> bool:
        return self.n_attempted is not None

    @property
    def rate(self) -> float | None:
        """``n_scored / n_attempted``, or None when the attempted count is unreported."""
        if self.n_attempted is None or self.n_attempted <= 0:
            return None
        return min(1.0, self.n_scored / self.n_attempted)

    @property
    def n_missing(self) -> int | None:
        if self.n_attempted is None:
            return None
        return max(0, self.n_attempted - self.n_scored)


@dataclass(frozen=True)
class Measurement:
    """One benchmark result for one run: the sufficient statistics, not a derived interval.

    ``value`` is the rate over graded items. It is the complete-case estimate and is biased upward
    under difficulty-correlated attrition, so it is reported but never ranked on: every ordering in
    the system sorts on :func:`measurement_interval`'s lower bound instead.
    """

    benchmark: str
    """The registry eval name: the leaderboard column."""

    metric: str
    """The headline metric key, e.g. ``exact_match,flexible-extract``."""

    kind: MetricKind
    value: float
    coverage: Coverage
    n_correct: int | None = None
    """Graded items scored correct, for a binary metric whose value is integral in k."""

    recorded_stderr: float | None = None
    """The harness-recorded standard error of the mean, for the continuous path."""

    item_cap: int | None = None
    """A declared per-run item cap. A comparability attribute, not part of benchmark identity: making
    it identity would split a benchmark's history in two whenever a cap changed."""

    flags: frozenset[ResultFlag] = frozenset()
    run_id: str = ""
    created_at: str = ""
    version: str | None = None
    model: str = ""
    git_sha: str = ""
    eval_runtime: str = ""
    status: RunStatus = RunStatus.SUCCEEDED


# --------------------------------------------------------------------------------------------------
# Intervals
# --------------------------------------------------------------------------------------------------


def normal_cdf(x: float) -> float:
    """The standard normal CDF."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def wilson_interval(n_correct: int, n_scored: int, z: float = _Z_TWO_SIDED) -> Interval:
    """The Wilson score interval for ``n_correct`` successes in ``n_scored`` Bernoulli trials.

    Preferred over the normal approximation because eval results really do sit at the boundary: a
    model that solves none of 8 agentic trials has a normal interval of zero width, asserting a
    certainty the data does not support, where Wilson gives ``[0.000, 0.324]``.
    """
    if n_scored <= 0:
        return Interval(0.0, 1.0)
    proportion = n_correct / n_scored
    denominator = 1.0 + z * z / n_scored
    center = (proportion + z * z / (2 * n_scored)) / denominator
    margin = z * math.sqrt(proportion * (1 - proportion) / n_scored + z * z / (4 * n_scored * n_scored)) / denominator
    return Interval(max(0.0, center - margin), min(1.0, center + margin))


def imbens_manski_critical(unidentified_width: float, sigma: float, alpha: float = ALPHA) -> float:
    """The critical value that covers the *parameter* at ``1 - alpha`` given an unidentified width.

    Solves ``Phi(C + delta/sigma) - Phi(-C) = 1 - alpha`` by bisection. Applying the ordinary
    two-sided quantile to an interval that already spans an unidentified region covers the identified
    *set* at ``1 - alpha`` and so over-covers the parameter (empirically ~97.5% at realistic coverage
    rates). ``C`` runs from the two-sided quantile when nothing is missing to the one-sided quantile
    when the unidentified width dominates. ``delta`` is known exactly here rather than estimated, so
    the usual small-``delta`` objection to this correction does not apply.
    """
    if unidentified_width <= 0.0:
        return _Z_TWO_SIDED
    if sigma <= 0.0:
        return _Z_ONE_SIDED
    ratio = unidentified_width / sigma
    low, high = _Z_ONE_SIDED, _Z_TWO_SIDED
    target = 1.0 - alpha
    for _ in range(_BISECTION_STEPS):
        middle = (low + high) / 2
        if normal_cdf(middle + ratio) - normal_cdf(-middle) < target:
            low = middle
        else:
            high = middle
    return (low + high) / 2


def _widen_for_attrition(observed: Interval, rate: float) -> Interval:
    """Scale an interval on the graded items to the attempted panel, bounding the ungraded ones.

    ``theta = c*theta_obs + (1-c)*theta_miss`` with ``theta_miss`` in [0, 1], so the identified region
    is ``[c*low, c*high + (1-c)]`` and the width is at least ``1 - c``.
    """
    return Interval(rate * observed.low, rate * observed.high + (1.0 - rate))


def measurement_interval(measurement: Measurement, alpha: float = ALPHA) -> ScoreInterval:
    """The interval for one measurement, and what it covers.

    Binary metrics use Wilson on ``(n_correct, n_scored)``; continuous metrics use the recorded
    standard error. When the attempted count is reported and items are missing, both are scaled and
    widened for attrition at the Imbens-Manski critical value; when it is unreported, the interval
    covers sampling error alone and says so, rather than assuming the run was complete.
    """
    coverage = measurement.coverage
    rate = coverage.rate
    unidentified = 0.0 if rate is None else 1.0 - rate
    sigma = _sigma(measurement) * (1.0 if rate is None else rate)
    z = imbens_manski_critical(unidentified, sigma, alpha)
    observed = _observed_interval(measurement, z)
    if rate is None:
        return ScoreInterval(observed.low, observed.high, IntervalKind.SAMPLING_ONLY)
    widened = _widen_for_attrition(observed, rate)
    return ScoreInterval(widened.low, widened.high, IntervalKind.IDENTIFIED)


def _sigma(measurement: Measurement) -> float:
    """The standard error of the score over graded items, used only to pick the critical value."""
    if measurement.kind is MetricKind.BINARY and measurement.coverage.n_scored > 0:
        proportion = measurement.value
        return math.sqrt(max(0.0, proportion * (1 - proportion)) / measurement.coverage.n_scored)
    return measurement.recorded_stderr or 0.0


def _observed_interval(measurement: Measurement, z: float) -> Interval:
    """The interval on the graded items alone, before any attrition widening."""
    if measurement.kind is MetricKind.BINARY and measurement.n_correct is not None:
        return wilson_interval(measurement.n_correct, measurement.coverage.n_scored, z)
    stderr = measurement.recorded_stderr
    if stderr is None:
        return Interval(0.0, 1.0)
    margin = z * stderr
    return Interval(max(0.0, measurement.value - margin), min(1.0, measurement.value + margin))


def difference_interval(a: Measurement, b: Measurement, alpha: float = ALPHA) -> Interval:
    """The identification region for ``theta_a - theta_b``, with sampling error folded in.

    The widening is asymmetric: the *opposing* run's ungraded items drive your bound, because they can
    take any value. Adding both coverage terms to both ends would be a different, wider interval.
    A difference interval that contains zero means the ordering is not identified, not that the models
    are equal -- at a 90% coverage gate the unidentified width alone is 0.2, so ordering claims need a
    much stricter coverage threshold than display does.
    """
    rate_a = a.coverage.rate if a.coverage.rate is not None else 1.0
    rate_b = b.coverage.rate if b.coverage.rate is not None else 1.0
    unidentified = (1.0 - rate_a) + (1.0 - rate_b)
    sigma = math.hypot(rate_a * _sigma(a), rate_b * _sigma(b))
    z = imbens_manski_critical(unidentified, sigma, alpha)
    interval_a = _observed_interval(a, z)
    interval_b = _observed_interval(b, z)
    center = rate_a * a.value - rate_b * b.value
    below = math.hypot(rate_a * (a.value - interval_a.low), rate_b * (interval_b.high - b.value))
    above = math.hypot(rate_a * (interval_a.high - a.value), rate_b * (b.value - interval_b.low))
    return Interval(center - (1.0 - rate_b) - below, center + (1.0 - rate_a) + above)


# --------------------------------------------------------------------------------------------------
# Panel aggregation
# --------------------------------------------------------------------------------------------------


class MissingPolicy(StrEnum):
    """What an aggregate does about a panel cell a model has no admissible result for."""

    REQUIRE_COMPLETE = "require_complete"
    """No aggregate at all, rather than a mean over whichever benchmarks happen to be present."""

    BOUND = "bound"
    """The missing cell contributes [0, 1], widening the aggregate by ``1/panel`` per absence."""


@dataclass(frozen=True)
class AggregationProtocol:
    """Everything needed to give a cross-benchmark aggregate a definition.

    An aggregate without its panel, its per-benchmark metric, and its missing-data policy has no
    interpretation, so no surface may render one without carrying this alongside.
    """

    panel: tuple[str, ...]
    missing: MissingPolicy = MissingPolicy.REQUIRE_COMPLETE


@dataclass(frozen=True)
class Aggregate:
    """A panel aggregate with the protocol that defines it."""

    value: float
    low: float
    high: float
    kind: IntervalKind
    protocol: AggregationProtocol
    covered: int
    """Panel cells backed by an admissible measurement."""

    metrics: tuple[str, ...]
    """The per-benchmark metric each cell contributed, in panel order, so the aggregate can show what
    it averaged. A missing cell contributes an empty string."""

    runtimes: tuple[str, ...]
    """Distinct harness versions across the contributing cells, sorted. Two benchmarks under the same
    name are not the same benchmark if different harness versions defined them, so an aggregate that
    spans more than one says so."""


def panel_aggregate(
    cells: Mapping[str, Measurement],
    protocol: AggregationProtocol,
    alpha: float = ALPHA,
) -> Aggregate | None:
    """The equal-weight mean over a panel, with an interval, or None when the protocol forbids one.

    Sampling error combines as ``sqrt(sum(w^2 se^2))`` across benchmarks, whose item sets are
    disjoint. Attrition bounds combine linearly and exactly. The Imbens-Manski critical value is
    recomputed from the aggregate's own unidentified width and sigma: per-cell corrected intervals
    cannot be averaged, because each carries a different critical value.

    The interval covers item sampling and attrition only. It does not cover run-to-run
    reproducibility -- a serving-stack difference moves every benchmark in the panel together.
    """
    panel = protocol.panel
    if not panel:
        return None
    present = [cells.get(benchmark) for benchmark in panel]
    if protocol.missing is MissingPolicy.REQUIRE_COMPLETE and any(cell is None for cell in present):
        return None
    weight = 1.0 / len(panel)
    total = 0.0
    unidentified = 0.0
    variance = 0.0
    kind = IntervalKind.IDENTIFIED
    for cell in present:
        if cell is None:
            # An absent benchmark is entirely unidentified: it contributes [0, 1].
            unidentified += weight
            continue
        rate = cell.coverage.rate
        if rate is None:
            kind = IntervalKind.SAMPLING_ONLY
            rate = 1.0
        total += weight * rate * cell.value
        unidentified += weight * (1.0 - rate)
        variance += (weight * rate * _sigma(cell)) ** 2
    sigma = math.sqrt(variance)
    z = imbens_manski_critical(unidentified, sigma, alpha)
    margin = z * sigma
    scored = [cell for cell in present if cell is not None]
    return Aggregate(
        value=sum(cell.value for cell in scored) / len(scored) if scored else 0.0,
        low=max(0.0, total - margin),
        high=min(1.0, total + margin + unidentified),
        kind=kind,
        protocol=protocol,
        covered=len(scored),
        metrics=tuple(cell.metric if cell is not None else "" for cell in present),
        runtimes=tuple(sorted({cell.eval_runtime for cell in scored if cell.eval_runtime})),
    )


# --------------------------------------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------------------------------------


class CohortMode(StrEnum):
    """Which of a model's version cohorts a benchmark's result may come from."""

    LATEST_VALID = "latest_valid"
    """The newest admissible result per benchmark, across every cohort. A recent cohort that re-ran
    only part of a model's benchmark set does not hide the older results that are still the newest
    available for their benchmark."""

    SINGLE_COHORT = "single_cohort"
    """Only the named cohort, for cohort-specific comparisons."""


class Completeness(StrEnum):
    """Which models a panel keeps."""

    ANY = "any"
    COMPLETE_PANEL = "complete_panel"
    """Only models with an admissible result for every selected benchmark."""


@dataclass(frozen=True)
class SelectionRequest:
    """The query a panel view is built from."""

    statuses: frozenset[RunStatus] = frozenset({RunStatus.SUCCEEDED})
    min_coverage: float = DEFAULT_MIN_COVERAGE

    cohort: CohortMode = CohortMode.LATEST_VALID
    cohort_version: str | None = None
    panel: tuple[str, ...] | None = None
    completeness: Completeness = Completeness.ANY
    filters: Mapping[str, str] = field(default_factory=dict)
    model_query: str | None = None


@dataclass(frozen=True)
class Rejection:
    """One measurement the request excluded, and why -- so an empty cell can be explained."""

    model: str
    benchmark: str
    run_id: str
    reason: str


@dataclass(frozen=True)
class Selection:
    """The chosen measurement per (model, benchmark), plus what was rejected on the way."""

    cells: Mapping[str, Mapping[str, Measurement]]
    rejections: tuple[Rejection, ...]
    benchmarks: tuple[str, ...]


def _admission_reason(measurement: Measurement, request: SelectionRequest) -> str | None:
    """Why ``measurement`` is inadmissible under ``request``, or None when it is admissible."""
    if measurement.status not in request.statuses:
        return f"status {measurement.status.value}"
    if ResultFlag.NO_ITEMS in measurement.flags:
        return "no item count"
    rate = measurement.coverage.rate
    if rate is not None and rate < request.min_coverage:
        return f"coverage {rate:.3f} below {request.min_coverage:.2f}"
    if request.cohort is CohortMode.SINGLE_COHORT and measurement.version != request.cohort_version:
        return f"cohort {measurement.version}"
    return None


def matches_filters(model: str, metadata: Mapping[str, str], request: SelectionRequest) -> bool:
    """Whether a run belongs to ``request``'s slice, by model name and run metadata.

    Takes the model name and metadata rather than a :class:`Measurement` so a caller can apply the
    same predicate to a run that produced no measurement at all -- which is how an empty cell gets a
    reason instead of being silently dropped.
    """
    if request.model_query and request.model_query.lower() not in model.lower():
        return False
    return all(metadata.get(key) == value for key, value in request.filters.items() if value)


def select(
    measurements: Iterable[Measurement],
    request: SelectionRequest,
    metadata: Mapping[str, Mapping[str, str]] | None = None,
) -> Selection:
    """Choose one measurement per (model, benchmark) under ``request``.

    ``metadata`` supplies each run's filterable properties keyed by run id (accelerator, backend, user
    and so on), so metadata filtering stays a property of the request rather than of the measurement.
    """
    metadata = metadata or {}
    chosen: dict[str, dict[str, Measurement]] = {}
    rejections: list[Rejection] = []
    for measurement in measurements:
        if not matches_filters(measurement.model, metadata.get(measurement.run_id, {}), request):
            continue
        if request.panel is not None and measurement.benchmark not in request.panel:
            continue
        reason = _admission_reason(measurement, request)
        if reason is not None:
            rejections.append(
                Rejection(
                    model=measurement.model,
                    benchmark=measurement.benchmark,
                    run_id=measurement.run_id,
                    reason=reason,
                )
            )
            continue
        current = chosen.setdefault(measurement.model, {}).get(measurement.benchmark)
        if current is None or (measurement.created_at or "") > (current.created_at or ""):
            chosen[measurement.model][measurement.benchmark] = measurement

    benchmarks = tuple(sorted({name for cells in chosen.values() for name in cells}))
    panel = request.panel if request.panel is not None else benchmarks
    if request.completeness is Completeness.COMPLETE_PANEL and panel:
        chosen = {model: cells for model, cells in chosen.items() if all(name in cells for name in panel)}
    return Selection(cells=chosen, rejections=tuple(rejections), benchmarks=benchmarks)
