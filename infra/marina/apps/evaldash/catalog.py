# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Searchable run and launch summaries for one EvalDash record snapshot."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TypedDict

from evalrecords.eval_stats import CohortMode
from evalrecords.records import EvalRunRecord, RunStatus

from .metrics import build_run_facets, cohort_selection, display_status, is_archived_model

MIXED_LAUNCH_STATUS = "mixed"


class RunSummary(TypedDict):
    run_id: str
    eval_name: str
    status: str
    created_at: str
    version: str | None


class RunRow(RunSummary):
    group_id: str
    user_name: str
    model_name: str
    model_location: str
    mechanism: str
    backend: str
    platform: str
    accelerator: str
    region: str | None
    step_status: str | None
    results_path: str
    git_sha: str
    image_digest: str | None
    error: str | None
    tasks: list[str]
    jobs: dict[str, str]


class LaunchRow(TypedDict):
    group_id: str
    model_name: str
    version: str | None
    description: str | None
    user_name: str
    accelerator: str
    created_at: str
    status: str
    n_evals: int
    n_succeeded: int


class LaunchMember(TypedDict):
    run_id: str
    eval_name: str
    status: str
    step_status: str | None
    created_at: str
    headline: dict[str, object] | None


class LaunchWithMembers(LaunchRow):
    evals: list[LaunchMember]


class ModelCatalogRow(TypedDict):
    model: str
    archived: bool
    run_count: int
    benchmark_count: int
    latest_run: RunSummary


class CatalogFacets(TypedDict):
    run_facets: dict[str, list[str]]


@dataclass(frozen=True)
class CatalogPage[Row]:
    rows: list[Row]
    total: int
    limit: int


@dataclass(frozen=True)
class CatalogFilters:
    query: str = ""
    model: str | None = None
    eval_name: str | None = None
    user: str | None = None
    status: str | None = None
    version: str | None = None
    accelerator: str | None = None
    group: str | None = None

    @classmethod
    def from_query(cls, params: Mapping[str, str]) -> CatalogFilters:
        return cls(
            query=params.get("q", "").strip().casefold(),
            model=params.get("model") or None,
            eval_name=params.get("eval") or None,
            user=params.get("user") or None,
            status=params.get("status") or None,
            version=params.get("version") or None,
            accelerator=params.get("accelerator") or None,
            group=params.get("group") or None,
        )

    def matches_model_and_user(self, record: EvalRunRecord) -> bool:
        return (self.model is None or self.model == record.model.name) and (
            self.user is None or self.user == record.user
        )


@dataclass(frozen=True, slots=True)
class _Run:
    record: EvalRunRecord
    status: str
    search_text: str

    def summary(self) -> RunSummary:
        record = self.record
        return {
            "run_id": record.run_id,
            "eval_name": record.evaluation.name,
            "status": self.status,
            "created_at": record.created_at,
            "version": record.version,
        }

    def row(self) -> RunRow:
        record = self.record
        return {
            **self.summary(),
            "group_id": record.group_id,
            "user_name": record.user,
            "model_name": record.model.name,
            "model_location": record.model.location,
            "mechanism": record.evaluation.mechanism,
            "backend": record.model.backend,
            "platform": record.hardware.platform,
            "accelerator": record.hardware.accelerator,
            "region": record.hardware.region_or_cluster,
            "step_status": record.step_status,
            "results_path": record.results_path,
            "git_sha": record.provenance.git_sha,
            "image_digest": record.provenance.eval_runtime,
            "error": record.error,
            "tasks": [task.name for task in record.evaluation.tasks],
            "jobs": dict(record.jobs),
        }

    def matches(self, filters: CatalogFilters) -> bool:
        record = self.record
        return (
            filters.matches_model_and_user(record)
            and (filters.eval_name is None or filters.eval_name == record.evaluation.name)
            and (filters.status is None or filters.status == self.status)
            and (filters.version is None or filters.version == record.version)
            and (filters.accelerator is None or filters.accelerator == record.hardware.accelerator)
            and (not filters.query or filters.query in self.search_text)
        )


def _indexed_run(record: EvalRunRecord) -> _Run:
    search_text = "\0".join(
        (
            record.run_id,
            record.group_id,
            record.evaluation.name,
            record.model.name,
            record.user,
            record.description or "",
        )
    ).casefold()
    return _Run(record, display_status(record), search_text)


def _status_rollup(statuses: set[str]) -> str:
    """Collapse a launch's per-eval statuses without inventing an evaluator failure."""
    if len(statuses) == 1:
        return next(iter(statuses))
    if RunStatus.SUCCEEDED not in statuses and RunStatus.FAILED in statuses:
        return RunStatus.FAILED
    return MIXED_LAUNCH_STATUS


class Launch:
    """One launch and its members in newest-first order."""

    def __init__(self, members: list[_Run]) -> None:
        self.members = members
        self.newest = members[0].record
        self.status = _status_rollup({run.status for run in members})
        self.n_succeeded = sum(run.status == RunStatus.SUCCEEDED for run in members)
        self._shared_model_and_user = all(
            run.record.model.name == self.newest.model.name and run.record.user == self.newest.user for run in members
        )

    def scope(self, filters: CatalogFilters) -> Launch | None:
        """Restrict the launch to the selected model and user, or return None if no member matches."""
        if filters.model is None and filters.user is None:
            return self
        if self._shared_model_and_user:
            return self if filters.matches_model_and_user(self.newest) else None
        members = [run for run in self.members if filters.matches_model_and_user(run.record)]
        if len(members) == len(self.members):
            return self
        return Launch(members) if members else None

    def matches(self, filters: CatalogFilters) -> bool:
        return (
            (filters.status is None or filters.status == self.status)
            and (filters.version is None or filters.version == self.newest.version)
            and (filters.accelerator is None or filters.accelerator == self.newest.hardware.accelerator)
            and (
                filters.eval_name is None or any(run.record.evaluation.name == filters.eval_name for run in self.members)
            )
            and (not filters.query or any(filters.query in run.search_text for run in self.members))
        )

    def row(self) -> LaunchRow:
        record = self.newest
        return {
            "group_id": record.group_id,
            "model_name": record.model.name,
            "version": record.version,
            "description": record.description,
            "user_name": record.user,
            "accelerator": record.hardware.accelerator,
            "created_at": record.created_at,
            "status": self.status,
            "n_evals": len(self.members),
            "n_succeeded": self.n_succeeded,
        }


def _launch_order(launch: Launch) -> tuple[str, str]:
    return launch.newest.created_at, launch.newest.group_id


@dataclass(frozen=True)
class _ModelActivity:
    model: str
    runs: list[_Run]
    archived: bool

    def row(self) -> ModelCatalogRow:
        return {
            "model": self.model,
            "archived": self.archived,
            "run_count": len(self.runs),
            "benchmark_count": len({run.record.evaluation.name for run in self.runs}),
            "latest_run": self.runs[0].summary(),
        }


class RunCatalog:
    """Index display statuses, search text, and launch membership once per snapshot."""

    def __init__(self, records: list[EvalRunRecord]) -> None:
        self._runs = [
            _indexed_run(record)
            for record in sorted(records, key=lambda record: (record.created_at, record.run_id), reverse=True)
        ]
        by_group: dict[str, list[_Run]] = {}
        self._by_model: dict[str, list[_Run]] = {}
        for run in self._runs:
            by_group.setdefault(run.record.group_id, []).append(run)
            self._by_model.setdefault(run.record.model.name, []).append(run)
        self._by_group = {group_id: Launch(members) for group_id, members in by_group.items()}
        self._launches = sorted(self._by_group.values(), key=_launch_order, reverse=True)

    def run_page(self, filters: CatalogFilters, offset: int, limit: int) -> CatalogPage[RunRow]:
        """Return rows on the requested page and the total matches before pagination."""
        if filters.group is None:
            candidates = self._runs
        else:
            launch = self._by_group.get(filters.group)
            candidates = launch.members if launch is not None else []
        if filters != CatalogFilters():
            candidates = [run for run in candidates if run.matches(filters)]
        return CatalogPage([run.row() for run in candidates[offset : offset + limit]], len(candidates), limit)

    def matching_launches(self, filters: CatalogFilters) -> list[Launch]:
        if filters == CatalogFilters():
            return self._launches
        if filters.group is None:
            candidates = self._launches
        else:
            launch = self._by_group.get(filters.group)
            candidates = [launch] if launch is not None else []
        matches = [
            launch
            for candidate in candidates
            if (launch := candidate.scope(filters)) is not None and launch.matches(filters)
        ]
        if filters.model is not None or filters.user is not None:
            # Older heterogeneous groups can move when their newest member is outside the scope.
            matches.sort(key=_launch_order, reverse=True)
        return matches

    def facets(self) -> dict[str, list[str]]:
        statuses = {run.status for run in self._runs}
        statuses.update(launch.status for launch in self._launches)
        return build_run_facets([run.record for run in self._runs], statuses)

    def model_page(
        self,
        query: str,
        cohort: str,
        archived_models: set[str],
        *,
        include_archived: bool,
        offset: int,
        limit: int,
    ) -> CatalogPage[ModelCatalogRow]:
        mode, version = cohort_selection(cohort)
        matches = []
        for model, runs in self._by_model.items():
            if query and query not in model.casefold():
                continue
            archived = is_archived_model(model, archived_models)
            if archived and not include_archived:
                continue
            if mode is CohortMode.SINGLE_COHORT:
                runs = [run for run in runs if run.record.version == version]
            if runs:
                matches.append(_ModelActivity(model, runs, archived))
        matches.sort(key=lambda match: (match.runs[0].record.created_at, match.model), reverse=True)
        return CatalogPage([match.row() for match in matches[offset : offset + limit]], len(matches), limit)
