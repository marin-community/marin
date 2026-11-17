# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Versioned task strata shared by selection writers and evaluation readers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict, model_validator
from rigging.filesystem.storage_path import StoragePath


class TaskStrata(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    aai_category: str = "untagged"
    family: str = "untagged"
    source: str = "untagged"
    topic: str = "untagged"
    difficulty: str = "untagged"


class TaskEvaluationIdentity(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    panel_name: str | None = None
    panel_version: str | None = None
    strata: TaskStrata = TaskStrata()
    agentic: bool | None = None


class MembershipState(StrEnum):
    CANDIDATE = "candidate"
    ACTIVE = "active"
    RETIRED = "retired"


class SelectionParent(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    root: str
    name: str
    version: str


class ReferenceMeasurement(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    task_digest: str
    model_revision: str
    attempted: int
    scored: int
    passed: int
    mean_score: float | None


class SelectionTaskRow(TaskStrata):
    model_config = ConfigDict(frozen=True, extra="forbid")

    task_id: str
    task_digest: str | None = None
    agentic: bool | None = None
    measured_difficulty: float | None = None
    state: MembershipState = MembershipState.CANDIDATE
    retirement_reason: str | None = None
    references: list[ReferenceMeasurement] = []
    grader_kind: str | None = None
    license: str | None = None
    raw_source: str | None = None

    @property
    def strata(self) -> TaskStrata:
        return TaskStrata.model_validate(self.model_dump(include=set(TaskStrata.model_fields)))


class DifficultyBands(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    hard_below: float
    easy_above: float

    @model_validator(mode="after")
    def ordered_bounds(self) -> DifficultyBands:
        if not 0 <= self.hard_below <= self.easy_above <= 1:
            raise ValueError("difficulty bounds must be ordered normalized scores")
        return self

    def tag(self, score: float | None) -> str:
        if score is None:
            return "untagged"
        return "hard" if score < self.hard_below else "easy" if score > self.easy_above else "learnable"


class SelectionDefinition(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = 1
    name: str
    version: str
    parent: SelectionParent | None = None
    task_files: tuple[str, ...] = ("full_validation.parquet",)
    active_files: tuple[str, ...] = ("validation.parquet",)
    family_quotas: dict[str, int] = {}
    sampling: str
    difficulty_bands: DifficultyBands | None = None
    reference_revisions: tuple[str, ...] = ()


class SelectionFormat(StrEnum):
    LEGACY = "legacy"
    VERSIONED = "versioned"


@dataclass(frozen=True)
class SelectionStrata:
    definition: SelectionDefinition
    tasks: dict[str, SelectionTaskRow]
    format: SelectionFormat

    def identity(self, task_id: str) -> TaskEvaluationIdentity:
        row = self.tasks.get(task_id)
        if row is None and self.format == SelectionFormat.VERSIONED:
            raise ValueError(f"task is absent from the declared selection manifest: {task_id}")
        return TaskEvaluationIdentity(
            panel_name=self.definition.name,
            panel_version=self.definition.version,
            strata=row.strata if row is not None else TaskStrata(),
            agentic=row.agentic if row is not None else None,
        )


def _legacy_task(row: dict) -> SelectionTaskRow:
    source = json.loads(row["source"])["dataset"] if row.get("source") else "untagged"
    return SelectionTaskRow(
        task_id=row["task_id"],
        family=row["family"],
        aai_category=row["aaii_category"],
        source=source,
        agentic=row["agentic"],
        measured_difficulty=row.get("difficulty"),
        grader_kind=row.get("grader_kind"),
        license=row.get("license"),
        raw_source=row.get("source"),
    )


def read_selection_strata(root: str) -> SelectionStrata | None:
    """Read a declared selection, converting the two published legacy manifest shapes."""
    directory = StoragePath(root)
    summary_path = directory / "summary.json"
    if not summary_path.exists():
        return None
    summary = json.loads(summary_path.read_text())
    declared = "selection_strata" in summary
    recipe = summary.get("recipe", {})
    legacy = (
        "name" in recipe
        and "version" in recipe
        and (
            {"fast_counts", "development", "training"}.issubset(recipe)
            or {"family_limits", "validation_per_family", "heldout"}.issubset(recipe)
        )
    )
    if not declared and not legacy:
        return None
    with (directory / "task_manifest.parquet").open("rb") as source:
        rows = pq.read_table(source).to_pylist()
    definition = (
        SelectionDefinition.model_validate(summary["selection_strata"])
        if declared
        else SelectionDefinition(name=recipe["name"], version=recipe["version"], sampling="legacy")
    )
    tasks = {}
    for value in rows:
        row = SelectionTaskRow.model_validate(value) if declared else _legacy_task(value)
        if row.task_id in tasks:
            raise ValueError(f"selection contains duplicate task ID: {row.task_id}")
        tasks[row.task_id] = row
    if declared:

        def population(files: tuple[str, ...]) -> dict[str, str]:
            result = {}
            for filename in files:
                with (directory / filename).open("rb") as source:
                    for batch in pq.ParquetFile(source).iter_batches(columns=["task_spec"], batch_size=256):
                        for raw in batch.column(0).to_pylist():
                            document = json.loads(raw)
                            task_id = document["id"]
                            if task_id in result:
                                raise ValueError(f"selection contains duplicate task ID in task files: {task_id}")
                            canonical = json.dumps(
                                document, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
                            )
                            result[task_id] = hashlib.sha256(canonical.encode()).hexdigest()
            return result

        pool = population(definition.task_files)
        if pool.keys() != tasks.keys():
            raise ValueError("selection manifest population differs from its task files")
        for task_id, row in tasks.items():
            if row.task_digest != pool[task_id] or any(ref.task_digest != row.task_digest for ref in row.references):
                raise ValueError(f"selection manifest task digest differs from task content: {task_id}")
        active = population(definition.active_files)
        expected = {row.task_id for row in tasks.values() if row.state == MembershipState.ACTIVE}
        if active.keys() != expected or any(pool.get(task_id) != digest for task_id, digest in active.items()):
            raise ValueError("selection active manifest membership differs from its active task files")
        if definition.family_quotas:
            counts = {}
            for task_id in active:
                family = tasks[task_id].strata.family
                counts[family] = counts.get(family, 0) + 1
            if counts != definition.family_quotas:
                raise ValueError(f"selection family counts differ from quotas: {counts}")
    else:
        with (directory / "validation.parquet").open("rb") as source:
            active = {
                json.loads(raw)["id"] for raw in pq.read_table(source, columns=["task_spec"]).column(0).to_pylist()
            }
        if not active <= tasks.keys():
            raise ValueError("legacy active tasks are absent from its selection manifest")
        tasks = {
            task_id: row.model_copy(
                update={"state": MembershipState.ACTIVE if task_id in active else MembershipState.CANDIDATE}
            )
            for task_id, row in tasks.items()
        }
    return SelectionStrata(definition, tasks, SelectionFormat.VERSIONED if declared else SelectionFormat.LEGACY)
