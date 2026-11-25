# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""An append-only log of task versions in one FineStore archive.

The ``tasks`` table is keyed by task id, and each add or edit of a task commits a new row
for it. Reads return the newest row of each id. FineStore never deletes data files, so
every older row stays readable through the commit that wrote it: a consumer that records
a commit id reads the same tasks from it later, however many commits follow.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from types import TracebackType

import pyarrow as pa
from finestore.history import CommitRecord, row_history
from finestore.layout import CommitToken, OnConflict
from finestore.reader import ReadView
from finestore.store import DEFAULT_UPDATE_ATTEMPTS, DataStore, RowConflict
from rigging.timing import ExponentialBackoff, retry_with_backoff

from taskcompendium.models import TaskSpec

TASKS_TABLE = "tasks"
TASK_ID = "task_id"
SPEC = "spec"
TASKS_SCHEMA = pa.schema([pa.field(TASK_ID, pa.string(), nullable=False), pa.field(SPEC, pa.string(), nullable=False)])
_ADD_BACKOFF = ExponentialBackoff(initial=0.05, maximum=2.0, factor=2.0, jitter=0.25)


@dataclass(frozen=True)
class TaskVersion:
    """One committed version of a task and the commit that wrote it."""

    commit: CommitRecord
    task: TaskSpec


class TaskView:
    """The tasks as one commit left them. Reading needs no write access to the archive."""

    def __init__(self, view: ReadView) -> None:
        self._view = view

    @classmethod
    def at(cls, root: str, commit_id: str | None = None) -> TaskView:
        """Pin commit ``commit_id``, or the newest commit when it is ``None``."""
        return cls(ReadView(root) if commit_id is None else ReadView.at(root, commit_id))

    @property
    def commit_id(self) -> str | None:
        """The pinned commit, or ``None`` for an archive without commits."""
        token = self._view.token
        return None if token is None else token.commit_id

    def get(self, task_ids: Iterable[str]) -> dict[str, TaskSpec]:
        """The newest version of each id that this commit holds; absent ids are left out."""
        rows = self._view.rows_by_key(TASKS_TABLE, [(task_id,) for task_id in task_ids])
        return {key[0]: _task(row) for key, row in rows.items()}

    def tasks(self) -> Iterator[TaskSpec]:
        """Yield the newest version of every task, in task-id order."""
        for row in self._view.iter_rows(TASKS_TABLE, columns=[TASK_ID, SPEC]):
            yield _task(row)


def task_history(root: str, task_id: str) -> list[TaskVersion]:
    """Every committed version of one task, newest first; empty for an unknown id."""
    return [
        TaskVersion(commit=version.commit, task=_task(version.row))
        for version in row_history(root, TASKS_TABLE, (task_id,))
    ]


class TaskStore:
    """Writes task versions to the archive at ``root``; reads go through ``TaskView``."""

    def __init__(self, root: str, *, writer_id: str | None = None) -> None:
        self.root = root
        self._store = DataStore(root, writer_id=writer_id)
        self._store.table(TASKS_TABLE, primary_key=(TASK_ID,), schema=TASKS_SCHEMA, on_conflict=OnConflict.SUPERSEDE)

    def __enter__(self) -> TaskStore:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        self._store.close()

    def add(self, tasks: Iterable[TaskSpec], *, author: str, message: str) -> CommitToken | None:
        """Commit a new version of each task that is new or differs from its newest version.

        Returns ``None`` when every task equals its newest version, so a repeated import
        commits nothing. When another writer commits one of these ids first, the comparison
        runs again on the newer rows. The call reads the newest version of every given id, so
        import a large catalog in batches.
        """
        rows: dict[str, dict] = {}
        for task in tasks:
            if task.id in rows:
                raise ValueError(f"task {task.id!r} appears more than once")
            rows[task.id] = _row(task)

        def attempt() -> CommitToken | None:
            transaction = self._store.unbounded_transaction(author=author, message=message)
            newest = transaction.read(TASKS_TABLE, [(task_id,) for task_id in rows])
            changed = [row for task_id, row in rows.items() if newest.get((task_id,)) != row]
            if not changed:
                return None
            transaction.table(TASKS_TABLE).extend(changed)
            return transaction.commit()

        return retry_with_backoff(
            attempt,
            retryable=lambda exc: isinstance(exc, RowConflict),
            max_attempts=DEFAULT_UPDATE_ATTEMPTS,
            backoff=_ADD_BACKOFF,
            operation=f"task add at {self.root}",
        )

    def update(
        self, task_ids: Sequence[str], edit: Callable[[TaskSpec], TaskSpec], *, author: str, message: str
    ) -> CommitToken:
        """Commit ``edit(task)`` as the next version of each task, in one commit.

        When another writer commits one of these tasks first, ``edit`` runs again on the newer
        version, so it may run more than once. An unknown id raises ``KeyError``, and an edit
        that changes a task's id raises ``ValueError``.
        """
        return self._store.update(
            TASKS_TABLE,
            [(task_id,) for task_id in task_ids],
            lambda row: _row(edit(_task(row))),
            author=author,
            message=message,
        )

    def get(self, task_ids: Iterable[str]) -> dict[str, TaskSpec]:
        """The newest version of each id; absent ids are left out."""
        return self.view().get(task_ids)

    def view(self, commit_id: str | None = None) -> TaskView:
        """The tasks as commit ``commit_id`` left them, or as the newest commit holds them."""
        return TaskView.at(self.root, commit_id)

    def history(self, task_id: str) -> list[TaskVersion]:
        """Every committed version of one task, newest first."""
        return task_history(self.root, task_id)


def _row(task: TaskSpec) -> dict:
    # Sorted keys make equal tasks serialize to equal strings, which ``add`` compares.
    return {TASK_ID: task.id, SPEC: json.dumps(task.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))}


def _task(row: dict) -> TaskSpec:
    return TaskSpec.model_validate_json(row[SPEC])
