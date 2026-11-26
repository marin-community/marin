# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Panel views over eval records, built on the shared statistics engine.

Every score the dashboard shows is a :class:`~evalrecords.eval_stats.Measurement`: a value, the
items behind it, and an interval that widens when a run graded less than it attempted. This module
turns records into measurements, answers a :class:`~evalrecords.eval_stats.SelectionRequest`
with them, and shapes the result for the API -- the statistics and the selection rules live in the
engine, which the eval runners share, so the dashboard and the producers cannot drift.

Presentation lives here: the suite grouping of columns, the family grouping that collapses several
settings of one benchmark into a single column, the smoke-suite exclusion, and the payload shapes. A
cell the request rejected is kept as a *missing* entry with the reason, so an empty cell is explained
rather than blank.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence, Set
from dataclasses import dataclass, replace
from functools import cached_property
from threading import Lock
from typing import NotRequired, TypedDict

from evalrecords.eval_measurements import (
    MetricRow,
    declared_metric_gap,
    declared_metric_rows,
    measurement_from_record,
    undeclared_metric_rows,
)
from evalrecords.eval_policy import POLICIES, record_policy_violations
from evalrecords.eval_stats import (
    DEFAULT_EXCLUDE_FLAGS,
    DEFAULT_MIN_COVERAGE,
    Aggregate,
    AggregationProtocol,
    CohortMode,
    Completeness,
    Interval,
    Measurement,
    MetricProtocol,
    MissingPolicy,
    Rejection,
    Selection,
    SelectionRequest,
    covers_panel,
    declared_protocols,
    difference_interval,
    matches_filters,
    matches_protocol,
    measurement_interval,
    panel_aggregate,
    select,
)
from evalrecords.historical_protocols import (
    comparison_column,
    compatible_selection,
    harbor_benchmark,
    historical_selection,
)
from evalrecords.metric_selection import PUBLICATION_METRIC_POLICY
from evalrecords.records import EvalRunRecord, ResolvedEvaluationSelection, RunStatus
from rigging.filesystem.step_status import STATUS_SUCCESS

# Capped-instance launcher validation runs; kept out of the headline panel (they stay visible in the
# runs list and history).
SMOKE_SUFFIX = "-smoke"

# Presentation grouping of eval columns into suites for the dashboard's column tree. This mirrors the
# launcher's suite membership (experiments/evaluation/evals.py). Experiment catalogs are not part of
# applet releases. An eval not listed here falls into "Other".
EVAL_SUITES: dict[str, tuple[str, ...]] = {
    "NLP": (
        "mmlu",
        "arc-challenge",
        "arc-easy",
        "hellaswag",
        "winogrande",
        "truthfulqa",
        "boolq",
        "piqa",
        "openbookqa",
        "lambada",
        "triviaqa",
        "nq-open",
        "drop",
        "gsm8k-0shot",
        "gsm8k",
    ),
    "Chat / Math": ("math500", "aime24", "olympiadbench"),
    "Long context": ("mrcr", "graphwalks"),
    "Code": ("humaneval", "humanevalplus", "mbppplus"),
}

# Run properties a panel can be filtered on. Each maps a facet name to the record attribute path the
# facet reads, so the API, the meta facets, and the selection filter all name the same set.
RUN_FACETS: dict[str, str] = {
    "accelerator": "hardware.accelerator",
    "platform": "hardware.platform",
    "backend": "model.backend",
    "mechanism": "evaluation.mechanism",
    "user": "user",
}


class EvalSuite(TypedDict):
    suite: str
    evals: list[str]


def eval_suites(evals: set[str], families: Mapping[str, str]) -> list[EvalSuite]:
    """Group the eval names present into ordered presentation suites for the column tree.

    Each suite lists only the evals actually seen; any eval outside :data:`EVAL_SUITES` lands in a
    trailing ``Other`` bucket so an unmapped column is still selectable.
    """
    assigned = set()
    for names in EVAL_SUITES.values():
        assigned.update(name for name in evals if name in names or families.get(name) in names)
    result: list[EvalSuite] = [
        {"suite": suite, "evals": present}
        for suite, names in EVAL_SUITES.items()
        if (present := sorted(name for name in evals if name in names or families.get(name) in names))
    ]
    other = sorted(name for name in evals if name not in assigned)
    if other:
        result.append({"suite": "Other", "evals": other})
    return result


def declared_families(records: Iterable[EvalRunRecord]) -> dict[str, str]:
    """Map each eval name to its explicit family, else the benchmark its newest Harbor dataset ran."""
    explicit: dict[str, tuple[str, str]] = {}
    inferred: dict[str, tuple[str, str]] = {}
    for record in records:
        curated = record.evaluation.curated
        if curated is not None and curated.selection is not None:
            explicit[comparison_column(curated.selection)] = (curated.selection.benchmark, record.created_at or "")
        family = record.evaluation.family
        if family is not None:
            current = explicit.get(record.evaluation.name)
            if current is None or (record.created_at or "") > current[1]:
                explicit[record.evaluation.name] = (family, record.created_at or "")
            continue
        harbor = record.evaluation.harbor
        if harbor is not None:
            current = inferred.get(record.evaluation.name)
            if current is None or (record.created_at or "") > current[1]:
                inferred[record.evaluation.name] = (harbor_benchmark(harbor), record.created_at or "")
    return {
        **{name: family for name, (family, _) in inferred.items()},
        **{name: family for name, (family, _) in explicit.items()},
    }


def group_by_family(names: Iterable[str], families: Mapping[str, str]) -> dict[str, list[str]]:
    """Eval names grouped by declared family, in first-seen order. An undeclared name is its own."""
    grouped: dict[str, list[str]] = {}
    for name in names:
        grouped.setdefault(families.get(name, name), []).append(name)
    return grouped


@dataclass(frozen=True)
class FamilyColumn:
    """One leaderboard column: a benchmark family, the settings requested of it, and the one shown."""

    family: str
    variants: tuple[str, ...]
    default: str


def _family_columns(
    requested: Sequence[str],
    families: Mapping[str, str],
    cells: Mapping[str, Mapping[str, Measurement]],
) -> list[FamilyColumn]:
    """Choose each family's variant by admitted-cell count, then eval name."""
    admitted = Counter(name for model_cells in cells.values() for name in model_cells)
    return [
        FamilyColumn(
            family=family,
            variants=tuple(variants),
            default=min(variants, key=lambda name: (-admitted[name], name)),
        )
        for family, variants in group_by_family(requested, families).items()
    ]


def _attribute(record: EvalRunRecord, path: str) -> str:
    value: object = record
    for part in path.split("."):
        value = getattr(value, part, None)
    return str(value) if value is not None else ""


def run_metadata(records: list[EvalRunRecord]) -> dict[str, dict[str, str]]:
    """Each run's filterable properties, keyed by run id, for the engine's metadata filter."""
    return {record.run_id: _record_metadata(record) for record in records}


def _record_metadata(record: EvalRunRecord) -> dict[str, str]:
    return {facet: _attribute(record, path) for facet, path in RUN_FACETS.items()}


ALL_COHORTS = "all"
"""Cohort selection for the newest admissible result per benchmark across every cohort."""

UNVERSIONED_COHORT = "unversioned"
"""Cohort selection for records launched without a version label (``version: null``)."""


def cohort_selection(cohort: str) -> tuple[CohortMode, str | None]:
    """Return the cohort mode and version a ``cohort`` query value names."""
    if cohort == ALL_COHORTS:
        return CohortMode.LATEST_VALID, None
    if cohort == UNVERSIONED_COHORT:
        return CohortMode.SINGLE_COHORT, None
    return CohortMode.SINGLE_COHORT, cohort


def is_archived_model(model: str, archived_models: Set[str]) -> bool:
    return model in archived_models or model.split("@", 1)[0] in archived_models


def in_cohort(record: EvalRunRecord, request: SelectionRequest) -> bool:
    return request.cohort is CohortMode.LATEST_VALID or record.version == request.cohort_version


def _policy_admits(violations: tuple[str, ...], cohort: CohortMode) -> bool:
    return cohort is CohortMode.LATEST_VALID or not violations


def _gap_reason(record: EvalRunRecord) -> str:
    """Why a record contributes no cell, when the request did not reject it outright."""
    if record.step_status not in (None, STATUS_SUCCESS):
        return f"step {record.step_status.lower()}"
    if record.status == RunStatus.SUCCEEDED:
        return declared_metric_gap(record) or "no metrics recorded"
    return f"status {record.status.value}"


def display_status(record: EvalRunRecord) -> str:
    """Use the StepSpec state when its run has not completed successfully."""
    if record.step_status is None or record.step_status == STATUS_SUCCESS:
        return record.status.value
    if record.step_status == "RUNNING":
        return "running"
    return "step_failed"


def score_eligible(record: EvalRunRecord) -> bool:
    """Whether EvalDash can publish scores from this observed record."""
    return record.step_status in (None, STATUS_SUCCESS)


def _dashboard_measurements(
    records: Iterable[EvalRunRecord], measurements: Mapping[str, Measurement]
) -> list[Measurement]:
    """Project scores from completed steps onto compatible benchmark settings."""
    records = [record for record in records if score_eligible(record)]
    selections = {
        comparison_column(record.evaluation.curated.selection): record.evaluation.curated.selection
        for record in records
        if record.evaluation.curated is not None and record.evaluation.curated.selection is not None
    }
    by_benchmark: dict[str, list[tuple[str, ResolvedEvaluationSelection]]] = {}
    for column, selection in selections.items():
        by_benchmark.setdefault(selection.benchmark, []).append((column, selection))
    result = []
    for record in records:
        measurement = measurements.get(record.run_id)
        if measurement is None:
            continue
        curated = record.evaluation.curated
        if curated is not None and curated.selection is not None:
            result.append(replace(measurement, benchmark=comparison_column(curated.selection)))
            continue
        historical = historical_selection(record) if by_benchmark else None
        if historical is None:
            result.append(measurement)
            continue
        columns = [
            column
            for column, selection in by_benchmark.get(historical.benchmark, [])
            if compatible_selection(historical, selection)
        ]
        if columns:
            result.extend(
                replace(measurement, benchmark=column, num_fewshot=historical.num_fewshot) for column in columns
            )
        else:
            result.append(measurement)
    return result


def cell_payload(measurement: Measurement) -> dict:
    """One panel cell: the value, its interval and what that interval covers, and its provenance."""
    interval = measurement_interval(measurement)
    coverage = measurement.coverage
    return {
        "metric_policy": PUBLICATION_METRIC_POLICY,
        "value": measurement.value,
        "low": interval.low,
        "high": interval.high,
        "interval_kind": interval.kind.value,
        "metric": measurement.metric,
        "metric_kind": measurement.kind.value,
        "repeat_aggregation": (
            measurement.repeat_aggregation.model_dump(mode="json")
            if measurement.repeat_aggregation is not None
            else None
        ),
        "declared": measurement.declared,
        "n_scored": coverage.n_scored,
        "n_benchmark": coverage.n_benchmark,
        "n_attempted": coverage.n_attempted,
        "coverage": coverage.rate,
        "benchmark_rate": coverage.benchmark_rate,
        "errors": dict(coverage.errors),
        "item_cap": measurement.item_cap,
        "flags": sorted(flag.value for flag in measurement.flags),
        "num_fewshot": measurement.num_fewshot,
        "run_id": measurement.run_id,
        "created_at": measurement.created_at,
        "version": measurement.version,
        "git_sha": measurement.git_sha,
        "eval_runtime": measurement.eval_runtime,
    }


def _aggregate_payload(aggregate: Aggregate | None) -> dict | None:
    """A panel aggregate rendered with the protocol that defines it, or None when there is none."""
    if aggregate is None:
        return None
    return {
        "value": aggregate.value,
        "low": aggregate.low,
        "high": aggregate.high,
        "interval_kind": aggregate.kind.value,
        "covered": aggregate.covered,
        "total": len(aggregate.protocol.panel),
        "panel": list(aggregate.protocol.panel),
        "missing_policy": aggregate.protocol.missing.value,
        "metrics": list(aggregate.metrics),
        "runtimes": list(aggregate.runtimes),
    }


def _missing_cells(
    records: list[EvalRunRecord],
    chosen: Mapping[str, Mapping[str, Measurement]],
    rejections: tuple[Rejection, ...],
    measurements: Sequence[Measurement],
    requested: tuple[str, ...] | None,
    hidden: set[str],
) -> dict[str, dict[str, dict]]:
    """Why each ``(model, benchmark)`` without an admitted cell has none, keyed model then benchmark.

    A gap is either a run the request rejected (a failed status or coverage below the gate) or a run
    that reached no metric at all, which never becomes a measurement. Both keep the
    newest offending run, so an empty cell links the run behind it instead of rendering blank.
    """
    reasons = {rejection.run_id: rejection.reason for rejection in rejections}
    columns: dict[str, list[str]] = {}
    for measurement in measurements:
        columns.setdefault(measurement.run_id, []).append(measurement.benchmark)
    missing: dict[str, dict[str, dict]] = {}
    for record in records:
        curated = record.evaluation.curated
        name = (
            comparison_column(curated.selection)
            if curated is not None and curated.selection is not None
            else record.evaluation.name
        )
        for benchmark in columns.get(record.run_id, [name]):
            if benchmark in hidden or (requested is not None and benchmark not in requested):
                continue
            model = record.model.name
            if benchmark in chosen.get(model, {}):
                continue
            reason = reasons.get(record.run_id) or _gap_reason(record)
            current = missing.setdefault(model, {}).get(benchmark)
            if current is None or (record.created_at or "") > current["created_at"]:
                missing[model][benchmark] = {
                    "reason": reason,
                    "run_id": record.run_id,
                    "status": display_status(record),
                    "created_at": record.created_at,
                }
    return missing


@dataclass(frozen=True)
class _ScoreSelection:
    eligible: list[EvalRunRecord]
    metadata: dict[str, dict[str, str]]
    measurements: list[Measurement]
    protocols: Mapping[str, MetricProtocol]
    families: Mapping[str, str]
    selection: Selection


@dataclass(frozen=True)
class _CohortScores:
    records: list[EvalRunRecord]
    measurements: list[Measurement]
    families: Mapping[str, str]
    protocols: Mapping[str, MetricProtocol]


class CohortOptions(TypedDict):
    default_cohort: str
    verified_cohorts: list[str]
    versions: list[str]


class ComparisonModels(CohortOptions):
    models: list[str]


class BenchmarkFamily(TypedDict):
    family: str
    variants: list[str]


class PanelMetadata(CohortOptions):
    models: list[str]
    run_facets: dict[str, list[str]]
    metric_policy: str
    evals: list[str]
    suites: list[EvalSuite]
    families: list[BenchmarkFamily]
    users: list[str]
    statuses: list[str]
    facets: dict[str, list[str]]
    archived_models: list[str]
    current_user: NotRequired[str | None]
    store: NotRequired[str]


class ScoredRecords:
    """Select benchmark scores and read score history for one record snapshot."""

    def __init__(self, records: list[EvalRunRecord], *, source: ScoredRecords | None = None) -> None:
        self.records = records
        self._non_smoke = [record for record in records if not record.evaluation.name.endswith(SMOKE_SUFFIX)]
        self.metadata = run_metadata(self._non_smoke)
        self._measurements = (
            source._measurements
            if source is not None
            else {
                record.run_id: measurement
                for record in self._non_smoke
                if score_eligible(record)
                if (measurement := measurement_from_record(record)) is not None
            }
        )
        self._violations = (
            source._violations
            if source is not None
            else {record.run_id: violations for record in records if (violations := record_policy_violations(record))}
        )
        self._lock = Lock()
        self._smoke_measurements: dict[str, Measurement | None] = {}
        self._cohort_scores: dict[CohortMode, _CohortScores] = {}
        self._last_selection: tuple[SelectionRequest, _ScoreSelection] | None = None

    def without_models(self, excluded: frozenset[str]) -> ScoredRecords:
        return ScoredRecords([record for record in self.records if record.model.name not in excluded], source=self)

    def violations(self, record: EvalRunRecord) -> tuple[str, ...]:
        return self._violations.get(record.run_id, ())

    def measurement(self, record: EvalRunRecord) -> Measurement | None:
        if not record.evaluation.name.endswith(SMOKE_SUFFIX):
            return self._measurements.get(record.run_id)
        if not score_eligible(record):
            return None
        # Smoke results are for direct inspection. Their metrics cannot affect fleet selections.
        with self._lock:
            if record.run_id not in self._smoke_measurements:
                self._smoke_measurements[record.run_id] = measurement_from_record(record)
            return self._smoke_measurements[record.run_id]

    def cohort_scores(self, cohort: CohortMode) -> _CohortScores:
        with self._lock:
            if cohort not in self._cohort_scores:
                # Historical runs may inherit a compatible setting from another version. Restrict
                # the alias sources by policy here, and apply version filters only after projection.
                records = [record for record in self._non_smoke if _policy_admits(self.violations(record), cohort)]
                measurements = _dashboard_measurements(records, self._measurements)
                self._cohort_scores[cohort] = _CohortScores(
                    records, measurements, declared_families(records), declared_protocols(measurements)
                )
            return self._cohort_scores[cohort]

    @cached_property
    def cohort_options(self) -> CohortOptions:
        return _cohort_options(self._non_smoke)

    def select(self, request: SelectionRequest) -> _ScoreSelection:
        scores = self.cohort_scores(request.cohort)
        with self._lock:
            if self._last_selection is not None and request == self._last_selection[0]:
                return self._last_selection[1]

            eligible = scores.records
            metadata = self.metadata
            if request.cohort is CohortMode.SINGLE_COHORT:
                eligible = [record for record in eligible if in_cohort(record, request)]
                metadata = {record.run_id: metadata[record.run_id] for record in eligible}

            # A named cohort chooses its own protocols. Unversioned results still use the
            # snapshot-wide protocols and are filtered by the engine's cohort admission rule.
            measurements = scores.measurements
            protocols = scores.protocols
            if request.cohort_version is not None:
                measurements = [m for m in measurements if m.version == request.cohort_version]
                protocols = declared_protocols(measurements)
            selection = select(measurements, replace(request, completeness=Completeness.ANY), metadata, protocols)
            selected = _ScoreSelection(eligible, metadata, measurements, protocols, scores.families, selection)
            self._last_selection = (replace(request, filters=dict(request.filters)), selected)
            return selected

    def policy_rejections(
        self, request: SelectionRequest, models: tuple[str, ...] | None = None
    ) -> list[dict[str, object]]:
        if request.cohort is CohortMode.LATEST_VALID:
            return []
        return [
            {
                "run_id": record.run_id,
                "model": record.model.name,
                "benchmark": record.evaluation.name,
                "reasons": list(violations),
            }
            for record in self._non_smoke
            if models is None or record.model.name in models
            if in_cohort(record, request)
            if (violations := self.violations(record))
            if request.panel is None or record.evaluation.name in request.panel
            if matches_filters(record.model.name, self.metadata[record.run_id], request)
        ]

    def history(self, model: str, benchmark: str, request: SelectionRequest) -> list[dict]:
        """Return a scored column or raw eval's history across versions, oldest first."""
        scores = self.cohort_scores(request.cohort)
        measurements = [item for item in scores.measurements if item.benchmark == benchmark]
        if measurements:
            return _model_history([m for m in measurements if m.model == model], scores.protocols).get(benchmark, [])

        # Raw evaluation names also identify runs outside the panel, such as smoke suites.
        raw_records = [
            record
            for record in self.records
            if record.evaluation.name == benchmark and _policy_admits(self.violations(record), request.cohort)
        ]
        raw_measurements = {
            record.run_id: score for record in raw_records if (score := self.measurement(record)) is not None
        }
        protocol = declared_protocols(_dashboard_measurements(raw_records, raw_measurements)).get(benchmark)
        points = [
            {**cell_payload(measurement), "status": record.status.value}
            for record in raw_records
            if record.model.name == model
            if (measurement := raw_measurements.get(record.run_id)) is not None
            if protocol is None or matches_protocol(measurement, protocol)
        ]
        return sorted(points, key=lambda point: point["created_at"] or "")

    def benchmark_names(self, scores: _CohortScores) -> set[str]:
        names = {measurement.benchmark for measurement in scores.measurements}
        names.update(
            (
                comparison_column(r.evaluation.curated.selection)
                if r.evaluation.curated is not None and r.evaluation.curated.selection is not None
                else r.evaluation.name
            )
            for r in scores.records
            if r.run_id not in self._measurements
        )
        return names


def _selected_columns(result: _ScoreSelection, request: SelectionRequest) -> list[FamilyColumn]:
    selected = result.selection
    requested = request.panel if request.panel is not None else selected.benchmarks
    return _family_columns(requested, result.families, selected.cells)


def comparison_models(scored: ScoredRecords, request: SelectionRequest) -> ComparisonModels:
    """Return the models selectable in Compare together with cohort options."""
    result = scored.select(request)
    complete_only = request.completeness is Completeness.COMPLETE_PANEL
    panel = [column.default for column in _selected_columns(result, request)] if complete_only else []
    models = sorted(
        model
        for model, cells in result.selection.cells.items()
        if any(measurement.value != 0 for measurement in cells.values())
        if not complete_only or covers_panel(cells, panel)
    )
    return {"models": models, **scored.cohort_options}


def build_panel(
    records: list[EvalRunRecord],
    request: SelectionRequest,
    archived_models: frozenset[str] = frozenset(),
    aggregate_policy: MissingPolicy | None = None,
    *,
    scored: ScoredRecords | None = None,
) -> dict:
    """Answer one selection request over the record snapshot.

    ``rows`` carries one entry per model that survived the request, each with its selected cells, the
    rejections behind any empty cell, and -- only when a caller asks for one by naming an aggregation
    policy -- a panel aggregate carrying its own protocol. No aggregate is produced by default: a mean
    across benchmarks has no interpretation without a declared panel and missing-data policy.

    ``families`` describes each column and its selected variant. ``panel`` contains those selected
    variants and controls coverage, completeness, and aggregation. ``benchmarks`` and ``cells`` retain
    every admitted variant under its exact eval name.
    """
    scored = scored if scored is not None else ScoredRecords(records)
    result = scored.select(request)
    selection = result.selection
    families = _selected_columns(result, request)
    panel = [column.default for column in families]
    # Omit gap entries for sibling variants without a visible column; admitted cells remain in the payload.
    hidden = {name for column in families for name in column.variants if name != column.default}

    on_panel = [
        record
        for record in result.eligible
        if matches_filters(record.model.name, result.metadata[record.run_id], request)
    ]
    missing = _missing_cells(on_panel, selection.cells, selection.rejections, result.measurements, request.panel, hidden)

    protocol = AggregationProtocol(panel=tuple(panel), missing=aggregate_policy) if aggregate_policy else None

    rows = []
    for model in sorted(set(selection.cells) | set(missing)):
        cells = selection.cells.get(model, {})
        if request.completeness is Completeness.COMPLETE_PANEL and not covers_panel(cells, panel):
            continue
        rows.append(
            {
                "model": model,
                "archived": is_archived_model(model, archived_models),
                "cells": {name: cell_payload(measurement) for name, measurement in cells.items()},
                "missing": missing.get(model, {}),
                "last_updated": max((measurement.created_at for measurement in cells.values()), default=None),
                "aggregate": _aggregate_payload(panel_aggregate(cells, protocol)) if protocol else None,
                "covered": sum(1 for name in panel if name in cells),
            }
        )
    return {
        "benchmarks": list(selection.benchmarks),
        "protocols": {
            benchmark: {"metric": protocol.metric, "kind": protocol.kind.value}
            for benchmark, protocol in result.protocols.items()
        },
        "panel": panel,
        "families": [
            {"family": column.family, "variants": list(column.variants), "default": column.default}
            for column in families
        ],
        "rows": rows,
        "policy_rejections": scored.policy_rejections(request),
        "request": {
            "min_coverage": request.min_coverage,
            "min_benchmark_coverage": request.min_benchmark_coverage,
            "cohort": request.cohort.value,
            "cohort_version": request.cohort_version,
            "completeness": request.completeness.value,
            "filters": dict(request.filters),
            "model_query": request.model_query,
            "statuses": sorted(status.value for status in request.statuses),
        },
    }


def _difference_payload(leader: Measurement, other: Measurement) -> dict:
    """One head-to-head gap: the interval for ``theta_leader - theta_other`` and whether it clears 0."""
    interval: Interval = difference_interval(leader, other)
    return {"low": interval.low, "high": interval.high, "separated": interval.low > 0.0}


def build_comparison(
    records: list[EvalRunRecord],
    request: SelectionRequest,
    models: tuple[str, ...],
    *,
    scored: ScoredRecords | None = None,
) -> dict:
    """Head-to-head over the benchmarks a set of models share.

    Per benchmark, the model with the highest interval lower bound leads, and every other model gets
    an interval for its gap to that leader. That interval, not an eyeball comparison of two error
    bars, is what settles whether an ordering holds: it folds in both runs' sampling error and both
    runs' ungraded items, and the ungraded ones enter asymmetrically because the *opposing* run's
    missing items are what can move your bound.

    The single ranking number is the equal-weight aggregate over the shared benchmarks only, under
    ``require_complete``: a model missing one of them is not scored rather than scored on a smaller
    panel that would not be the same quantity.
    """
    scored = scored if scored is not None else ScoredRecords(records)
    selection = scored.select(request).selection
    chosen = {model: dict(selection.cells.get(model, {})) for model in models}
    if request.completeness is Completeness.COMPLETE_PANEL:
        selected_panel = request.panel or tuple(sorted({name for cells in chosen.values() for name in cells}))
        chosen = {model: cells if covers_panel(cells, selected_panel) else {} for model, cells in chosen.items()}

    union = [name for name in selection.benchmarks if any(name in cells for cells in chosen.values())]
    shared = [name for name in union if all(name in cells for cells in chosen.values())]
    protocol = AggregationProtocol(panel=tuple(shared), missing=MissingPolicy.REQUIRE_COMPLETE)

    rows = []
    for benchmark in union:
        present = {model: cells[benchmark] for model, cells in chosen.items() if benchmark in cells}
        leader = max(present, key=lambda model: measurement_interval(present[model]).low)
        rows.append(
            {
                "benchmark": benchmark,
                "shared": benchmark in shared,
                "leader": leader,
                "cells": {model: cell_payload(measurement) for model, measurement in present.items()},
                "differences": {
                    model: _difference_payload(present[leader], measurement)
                    for model, measurement in present.items()
                    if model != leader
                },
            }
        )
    return {
        "models": list(models),
        "benchmarks": union,
        "shared": shared,
        "rows": rows,
        "policy_rejections": scored.policy_rejections(request, models),
        "aggregates": {model: _aggregate_payload(panel_aggregate(cells, protocol)) for model, cells in chosen.items()},
    }


def build_run_facets(records: list[EvalRunRecord], statuses: Iterable[str]) -> dict[str, list[str]]:
    """Return facet choices from the full run catalog."""
    return {
        "model": sorted({r.model.name for r in records}),
        "eval": sorted({r.evaluation.name for r in records}),
        "version": sorted({r.version for r in records if r.version}),
        "status": sorted(set(statuses)),
        "user": sorted({r.user for r in records if r.user}),
        "accelerator": sorted({r.hardware.accelerator for r in records if r.hardware.accelerator}),
    }


def _cohort_options(eligible: list[EvalRunRecord]) -> CohortOptions:
    versions = sorted(set(POLICIES) | {r.version for r in eligible if r.version})
    if any(r.version is None for r in eligible):
        versions.append(UNVERSIONED_COHORT)
    return {"default_cohort": ALL_COHORTS, "verified_cohorts": list(POLICIES), "versions": versions}


def build_meta(
    records: list[EvalRunRecord],
    archived_models: frozenset[str] = frozenset(),
    *,
    scored: ScoredRecords | None = None,
) -> PanelMetadata:
    """Return panel filter metadata and all known variants for each family."""
    run_facets = build_run_facets(records, (r.status.value for r in records))
    scored = scored if scored is not None else ScoredRecords(records)
    scores = scored.cohort_scores(CohortMode.LATEST_VALID)
    eval_names = scored.benchmark_names(scores)
    by_family = group_by_family(sorted(eval_names), scores.families)
    metadata = scored.metadata
    facets = {
        facet: sorted({values[facet] for values in metadata.values() if values.get(facet)}) for facet in RUN_FACETS
    }
    return {
        "models": run_facets["model"],
        "run_facets": run_facets,
        "metric_policy": PUBLICATION_METRIC_POLICY,
        **scored.cohort_options,
        "evals": sorted(eval_names),
        "suites": eval_suites(eval_names, scores.families),
        "families": [{"family": family, "variants": variants} for family, variants in sorted(by_family.items())],
        "users": sorted({r.user for r in scores.records if r.user}),
        "statuses": sorted({display_status(r) for r in scores.records}),
        "facets": facets,
        "archived_models": sorted(archived_models),
    }


def record_headline(record: EvalRunRecord, protocol: MetricProtocol | None = None) -> dict | None:
    """One run's headline score with its interval, or None when the run produced no primary metric."""
    if not score_eligible(record):
        return None
    measurement = measurement_from_record(record)
    if measurement is None or (protocol is not None and not matches_protocol(measurement, protocol)):
        return None
    return cell_payload(measurement)


def run_metric_rows(record: EvalRunRecord) -> list[MetricRow]:
    """The run page's top metric rows: declared protocol metrics first, then undeclared raw rows.

    A declared benchmark metric keeps its protocol name and the value the statistics engine would
    select; tasks without a declaration keep their raw rows, so a record that mixes both still shows
    every task. When nothing is declared, the whole raw map is returned.
    """
    return [*declared_metric_rows(record), *undeclared_metric_rows(record)]


def _model_cohorts(records: list[EvalRunRecord]) -> list[dict]:
    """One entry per distinct version cohort, newest first, with its eval counts and serve group."""
    by_version: dict[str | None, list[EvalRunRecord]] = {}
    for record in records:
        by_version.setdefault(record.version, []).append(record)
    cohorts = []
    for version, members in by_version.items():
        newest = max(members, key=lambda record: record.created_at or "")
        cohorts.append(
            {
                "version": version,
                "created_at": newest.created_at,
                "n_evals": len(members),
                "n_succeeded": sum(1 for record in members if display_status(record) == RunStatus.SUCCEEDED),
                "group_id": newest.group_id,
            }
        )
    cohorts.sort(key=lambda cohort: cohort["created_at"] or "", reverse=True)
    return cohorts


def _model_history(measurements: list[Measurement], protocols: Mapping[str, MetricProtocol]) -> dict[str, list[dict]]:
    """Per-eval score-over-time: every scored run for the model on each eval, oldest first."""
    history: dict[str, list[dict]] = {}
    for measurement in measurements:
        protocol = protocols.get(measurement.benchmark)
        if protocol is not None and not matches_protocol(measurement, protocol):
            continue
        history.setdefault(measurement.benchmark, []).append(
            {**cell_payload(measurement), "status": measurement.status.value}
        )
    for points in history.values():
        points.sort(key=lambda point: point["created_at"] or "")
    return history


def _model_runs(
    records: list[EvalRunRecord],
    protocols: Mapping[str, MetricProtocol],
    request: SelectionRequest,
    scored: ScoredRecords,
) -> list[dict]:
    """Every run for the model, newest first, with grades only for admitted runs."""
    runs = []
    for record in records:
        violations = scored.violations(record)
        protocol = protocols.get(record.evaluation.name)
        measurement = scored.measurement(record)
        protocol_mismatch = (
            measurement is not None and protocol is not None and not matches_protocol(measurement, protocol)
        )
        headline = (
            cell_payload(measurement)
            if measurement is not None and not protocol_mismatch and _policy_admits(violations, request.cohort)
            else None
        )
        runs.append(
            {
                "run_id": record.run_id,
                "eval_name": record.evaluation.name,
                "status": display_status(record),
                "step_status": record.step_status,
                "created_at": record.created_at,
                "version": record.version,
                "headline": headline,
                "policy_violations": list(violations),
                "gap_reason": (
                    None
                    if headline
                    else (
                        "; ".join(violations)
                        if violations
                        else "metric differs from current protocol" if protocol_mismatch else _gap_reason(record)
                    )
                ),
            }
        )
    runs.sort(key=lambda run: run["created_at"] or "", reverse=True)
    return runs


def build_model_detail(
    records: list[EvalRunRecord],
    model: str,
    request: SelectionRequest,
    *,
    scored: ScoredRecords | None = None,
) -> dict | None:
    """Everything the frontend Model view needs for one model, in one payload, or None when unknown.

    ``current_version`` and ``cohorts`` reflect admitted non-smoke runs. The history spans admitted
    scores across versions. ``runs`` lists every run for the model, including smoke runs, newest first.
    """
    model_records = [record for record in records if record.model.name == model]
    if not model_records:
        return None

    scored = scored if scored is not None else ScoredRecords(records)
    # The selected cohort governs policy admission for the timeline across versions.
    scores = scored.cohort_scores(request.cohort)
    newest = max(model_records, key=lambda record: record.created_at or "")
    eligible = [record for record in scores.records if record.model.name == model]
    return {
        "model": model,
        "location": newest.model.location,
        "backend": newest.model.backend,
        "user": newest.user,
        "current_version": max(eligible, key=lambda r: r.created_at or "").version if eligible else None,
        "cohorts": _model_cohorts(eligible),
        "history": _model_history([item for item in scores.measurements if item.model == model], scores.protocols),
        "runs": _model_runs(model_records, scores.protocols, request, scored),
    }


def panel_request(
    *,
    benchmarks: tuple[str, ...] | None = None,
    cohort: str = ALL_COHORTS,
    completeness: Completeness = Completeness.ANY,
    min_coverage: float = DEFAULT_MIN_COVERAGE,
    min_benchmark_coverage: float = DEFAULT_MIN_COVERAGE,
    filters: dict[str, str] | None = None,
    model_query: str | None = None,
    include_flagged: bool = False,
) -> SelectionRequest:
    """Build a selection request from already-parsed query values.

    ``include_flagged`` readmits results the engine flags as suspect, which are excluded by default
    so one cannot stand as a model's newest result. They are always reported as the reason for the
    empty cell, so this widens what is shown rather than revealing something that was hidden.
    ``cohort`` is a version label, ``"unversioned"`` for records without one, or ``"all"``.
    """
    cohort_mode, cohort_version = cohort_selection(cohort)
    return SelectionRequest(
        min_coverage=min_coverage,
        min_benchmark_coverage=min_benchmark_coverage,
        exclude_flags=frozenset() if include_flagged else DEFAULT_EXCLUDE_FLAGS,
        cohort=cohort_mode,
        cohort_version=cohort_version,
        panel=benchmarks,
        completeness=completeness,
        filters=filters or {},
        model_query=model_query,
    )
