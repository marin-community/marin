# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Panel views over eval records, built on the shared statistics engine.

Every score the dashboard shows is a :class:`~marin.evaluation.eval_stats.Measurement`: a value, the
items behind it, and an interval that widens when a run graded less than it attempted. This module
turns records into measurements, answers a :class:`~marin.evaluation.eval_stats.SelectionRequest`
with them, and shapes the result for the API -- the statistics and the selection rules live in the
engine, which the eval runners share, so the dashboard and the producers cannot drift.

Presentation lives here: the suite grouping of columns, the smoke-suite exclusion, and the payload
shapes. A cell the request rejected is kept as a *missing* entry with the reason, so an empty cell is
explained rather than blank.
"""

from __future__ import annotations

from collections.abc import Mapping

from marin.evaluation.eval_measurements import measurement_from_record, measurements_from_records
from marin.evaluation.eval_stats import (
    DEFAULT_MIN_COVERAGE,
    Aggregate,
    AggregationProtocol,
    CohortMode,
    Completeness,
    Interval,
    Measurement,
    MissingPolicy,
    Rejection,
    SelectionRequest,
    difference_interval,
    matches_filters,
    measurement_interval,
    panel_aggregate,
    select,
)
from marin.evaluation.records import EvalRunRecord, RunStatus

# Capped-instance launcher validation runs; kept out of the headline panel (they stay visible in the
# runs list and history).
SMOKE_SUFFIX = "-smoke"

# Presentation grouping of eval columns into suites for the dashboard's column tree. This mirrors the
# launcher's suite membership (experiments/evaluation/evals.py), which evaldash cannot import: it ships
# as a standalone image vendoring only the marin record contracts, and experiments depends on marin,
# not the reverse. Membership drift is graceful -- an eval not listed here just falls into "Other".
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
    ),
    "Chat / Math": ("math500", "aime24", "olympiadbench"),
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


def eval_suites(evals: set[str]) -> list[dict]:
    """Group the eval names present into ordered presentation suites for the column tree.

    Each suite lists only the evals actually seen; any eval outside :data:`EVAL_SUITES` lands in a
    trailing ``Other`` bucket so an unmapped column is still selectable.
    """
    assigned = {name for names in EVAL_SUITES.values() for name in names}
    result = [
        {"suite": suite, "evals": present}
        for suite, names in EVAL_SUITES.items()
        if (present := sorted(name for name in names if name in evals))
    ]
    other = sorted(name for name in evals if name not in assigned)
    if other:
        result.append({"suite": "Other", "evals": other})
    return result


def _attribute(record: EvalRunRecord, path: str) -> str:
    value: object = record
    for part in path.split("."):
        value = getattr(value, part, None)
    return str(value) if value is not None else ""


def run_metadata(records: list[EvalRunRecord]) -> dict[str, dict[str, str]]:
    """Each run's filterable properties, keyed by run id, for the engine's metadata filter."""
    return {record.run_id: {facet: _attribute(record, path) for facet, path in RUN_FACETS.items()} for record in records}


def _panel_records(records: list[EvalRunRecord]) -> list[EvalRunRecord]:
    """Records eligible for the headline panel: everything but the capped smoke suites."""
    return [record for record in records if not record.evaluation.name.endswith(SMOKE_SUFFIX)]


def _gap_reason(record: EvalRunRecord) -> str:
    """Why a record contributes no cell, when the request did not reject it outright."""
    if record.status == RunStatus.SUCCEEDED:
        return "no metrics recorded"
    return f"status {record.status.value}"


def cell_payload(measurement: Measurement) -> dict:
    """One panel cell: the value, its interval and what that interval covers, and its provenance."""
    interval = measurement_interval(measurement)
    coverage = measurement.coverage
    return {
        "value": measurement.value,
        "low": interval.low,
        "high": interval.high,
        "interval_kind": interval.kind.value,
        "metric": measurement.metric,
        "metric_kind": measurement.kind.value,
        "n_scored": coverage.n_scored,
        "n_attempted": coverage.n_attempted,
        "coverage": coverage.rate,
        "errors": dict(coverage.errors),
        "item_cap": measurement.item_cap,
        "flags": sorted(flag.value for flag in measurement.flags),
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
) -> dict[str, dict[str, dict]]:
    """Why each ``(model, benchmark)`` without an admitted cell has none, keyed model then benchmark.

    A gap is either a run the request rejected (a failed status, coverage below the gate, the wrong
    cohort) or a run that reached no metric at all, which never becomes a measurement. Both keep the
    newest offending run, so an empty cell links the run behind it instead of rendering blank.
    """
    reasons = {rejection.run_id: rejection.reason for rejection in rejections}
    missing: dict[str, dict[str, dict]] = {}
    for record in records:
        model, benchmark = record.model.name, record.evaluation.name
        if benchmark in chosen.get(model, {}):
            continue
        reason = reasons.get(record.run_id) or _gap_reason(record)
        current = missing.setdefault(model, {}).get(benchmark)
        if current is None or (record.created_at or "") > current["created_at"]:
            missing[model][benchmark] = {
                "reason": reason,
                "run_id": record.run_id,
                "status": record.status.value,
                "created_at": record.created_at,
            }
    return missing


def build_panel(
    records: list[EvalRunRecord],
    request: SelectionRequest,
    archived_models: frozenset[str] = frozenset(),
    aggregate_policy: MissingPolicy | None = None,
) -> dict:
    """Answer one selection request over the record snapshot.

    ``rows`` carries one entry per model that survived the request, each with its selected cells, the
    rejections behind any empty cell, and -- only when a caller asks for one by naming an aggregation
    policy -- a panel aggregate carrying its own protocol. No aggregate is produced by default: a mean
    across benchmarks has no interpretation without a declared panel and missing-data policy.
    """
    eligible = _panel_records(records)
    metadata = run_metadata(eligible)
    selection = select(measurements_from_records(eligible), request, metadata)
    on_panel = [
        record
        for record in eligible
        if request.panel is None or record.evaluation.name in request.panel
        if matches_filters(record.model.name, metadata[record.run_id], request)
    ]
    missing = _missing_cells(on_panel, selection.cells, selection.rejections)

    panel = request.panel if request.panel is not None else selection.benchmarks
    protocol = AggregationProtocol(panel=tuple(panel), missing=aggregate_policy) if aggregate_policy else None

    rows = []
    for model in sorted(set(selection.cells) | set(missing)):
        cells = selection.cells.get(model, {})
        if request.completeness is Completeness.COMPLETE_PANEL and model not in selection.cells:
            continue
        rows.append(
            {
                "model": model,
                "archived": model in archived_models,
                "cells": {name: cell_payload(measurement) for name, measurement in cells.items()},
                "missing": missing.get(model, {}),
                "aggregate": _aggregate_payload(panel_aggregate(cells, protocol)) if protocol else None,
                "covered": sum(1 for name in panel if name in cells),
            }
        )
    return {
        "benchmarks": list(selection.benchmarks),
        "panel": list(panel),
        "rows": rows,
        "request": {
            "min_coverage": request.min_coverage,
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


def build_comparison(records: list[EvalRunRecord], request: SelectionRequest, models: tuple[str, ...]) -> dict:
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
    eligible = _panel_records(records)
    metadata = run_metadata(eligible)
    selection = select(measurements_from_records(eligible), request, metadata)
    chosen = {model: dict(selection.cells.get(model, {})) for model in models}

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
        "aggregates": {model: _aggregate_payload(panel_aggregate(cells, protocol)) for model, cells in chosen.items()},
    }


def build_meta(records: list[EvalRunRecord], archived_models: frozenset[str] = frozenset()) -> dict:
    """Distinct filter values across all records, plus the archived set and the run facets."""
    eval_names = {r.evaluation.name for r in records}
    metadata = run_metadata(records)
    facets = {
        facet: sorted({values[facet] for values in metadata.values() if values.get(facet)}) for facet in RUN_FACETS
    }
    return {
        "models": sorted({r.model.name for r in records}),
        "evals": sorted(eval_names),
        "suites": eval_suites(eval_names),
        "users": sorted({r.user for r in records if r.user}),
        "statuses": sorted({r.status.value for r in records}),
        "versions": sorted({r.version for r in records if r.version}),
        "facets": facets,
        "archived_models": sorted(archived_models),
    }


def record_headline(record: EvalRunRecord) -> dict | None:
    """One run's headline score with its interval, or None when the run produced no primary metric."""
    measurement = measurement_from_record(record)
    if measurement is None:
        return None
    return cell_payload(measurement)


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
                "n_succeeded": sum(1 for record in members if record.status == RunStatus.SUCCEEDED),
                "group_id": newest.group_id,
            }
        )
    cohorts.sort(key=lambda cohort: cohort["created_at"] or "", reverse=True)
    return cohorts


def _model_history(records: list[EvalRunRecord]) -> dict[str, list[dict]]:
    """Per-eval score-over-time: every scored run for the model on each eval, oldest first."""
    history: dict[str, list[dict]] = {}
    for record in records:
        headline = record_headline(record)
        if headline is None:
            continue
        history.setdefault(record.evaluation.name, []).append({**headline, "status": record.status.value})
    for points in history.values():
        points.sort(key=lambda point: point["created_at"] or "")
    return history


def _model_runs(records: list[EvalRunRecord]) -> list[dict]:
    """Every run for the model, newest first, each with its headline score when it scored."""
    runs = []
    for record in records:
        headline = record_headline(record)
        runs.append(
            {
                "run_id": record.run_id,
                "eval_name": record.evaluation.name,
                "status": record.status.value,
                "created_at": record.created_at,
                "version": record.version,
                "headline": headline,
                "gap_reason": None if headline else _gap_reason(record),
            }
        )
    runs.sort(key=lambda run: run["created_at"] or "", reverse=True)
    return runs


def build_model_detail(records: list[EvalRunRecord], model: str) -> dict | None:
    """Everything the frontend Model view needs for one model, in one payload, or None when unknown.

    ``current_version`` is the version of the model's most recent non-smoke run -- the cohort the view
    opens on -- and ``cohorts`` lists one entry per distinct version, both excluding ``-smoke`` suites
    as the headline panel does. ``history`` is the per-eval score-over-time across every scored run,
    and ``runs`` spans every run for the model (smoke included), newest first.
    """
    model_records = [record for record in records if record.model.name == model]
    if not model_records:
        return None
    newest = max(model_records, key=lambda record: record.created_at or "")
    eligible = _panel_records(model_records)
    return {
        "model": model,
        "location": newest.model.location,
        "backend": newest.model.backend,
        "user": newest.user,
        "current_version": max(eligible, key=lambda r: r.created_at or "").version if eligible else None,
        "cohorts": _model_cohorts(eligible),
        "history": _model_history(eligible),
        "runs": _model_runs(model_records),
    }


def panel_request(
    *,
    benchmarks: tuple[str, ...] | None = None,
    cohort_version: str | None = None,
    completeness: Completeness = Completeness.ANY,
    min_coverage: float = DEFAULT_MIN_COVERAGE,
    filters: dict[str, str] | None = None,
    model_query: str | None = None,
) -> SelectionRequest:
    """Build a selection request from already-parsed query values."""
    return SelectionRequest(
        min_coverage=min_coverage,
        cohort=CohortMode.SINGLE_COHORT if cohort_version else CohortMode.LATEST_VALID,
        cohort_version=cohort_version,
        panel=benchmarks,
        completeness=completeness,
        filters=filters or {},
        model_query=model_query,
    )
