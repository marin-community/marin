# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Searchable run and launch summaries for one EvalDash record snapshot."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from evalrecords.eval_stats import CohortMode
from evalrecords.records import EvalRunRecord

from .metrics import build_run_facets, cohort_selection, display_status, is_archived_model


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


def _run(record: EvalRunRecord) -> _Run:
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
    if "succeeded" not in statuses and "failed" in statuses:
        return "failed"
    return "mixed"


class Launch:
    """One launch and its members in newest-first order."""

    def __init__(self, members: list[_Run]) -> None:
        self.members = members
        self.newest = members[0].record
        self.status = _status_rollup({run.status for run in members})
        self.n_succeeded = sum(run.status == "succeeded" for run in members)
        self._shared_model_and_user = all(
            run.record.model.name == self.newest.model.name and run.record.user == self.newest.user for run in members
        )

    def scope(self, filters: CatalogFilters) -> Launch | None:
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

    def row(self) -> dict:
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

    def row(self) -> dict:
        latest = self.runs[0]
        return {
            "model": self.model,
            "archived": self.archived,
            "run_count": len(self.runs),
            "benchmark_count": len({run.record.evaluation.name for run in self.runs}),
            "latest_run": {
                "run_id": latest.record.run_id,
                "eval_name": latest.record.evaluation.name,
                "status": latest.status,
                "created_at": latest.record.created_at,
                "version": latest.record.version,
            },
        }


class RunCatalog:
    """Index display statuses, search text, and launch membership once per snapshot."""

    def __init__(self, records: list[EvalRunRecord]) -> None:
        self._runs = [
            _run(record)
            for record in sorted(records, key=lambda record: (record.created_at, record.run_id), reverse=True)
        ]
        by_group: dict[str, list[_Run]] = {}
        self._by_model: dict[str, list[_Run]] = {}
        for run in self._runs:
            by_group.setdefault(run.record.group_id, []).append(run)
            self._by_model.setdefault(run.record.model.name, []).append(run)
        self._by_group = {group_id: Launch(members) for group_id, members in by_group.items()}
        self._launches = sorted(self._by_group.values(), key=_launch_order, reverse=True)

    def run_page(self, filters: CatalogFilters, offset: int, limit: int) -> tuple[list[EvalRunRecord], int]:
        if filters.group is None:
            candidates = self._runs
        else:
            launch = self._by_group.get(filters.group)
            candidates = launch.members if launch is not None else []
        if filters != CatalogFilters():
            candidates = [run for run in candidates if run.matches(filters)]
        return [run.record for run in candidates[offset : offset + limit]], len(candidates)

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
    ) -> dict:
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
        return {"rows": [match.row() for match in matches[offset : offset + limit]], "total": len(matches)}
