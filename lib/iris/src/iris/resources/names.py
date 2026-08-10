# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Canonical hierarchical names and compact runtime identifiers."""

import functools
import hashlib
import urllib.parse
from dataclasses import dataclass
from typing import NewType


@dataclass(frozen=True, slots=True)
class JobName:
    """Structured ``/user/job[/child]`` name used by Jobs and Tasks."""

    _parts: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self._parts) < 2:
            raise ValueError("JobName must use canonical '/<user>/<job>[...]' format")
        for part in self._parts:
            if "/" in part:
                raise ValueError(f"JobName component cannot contain '/': {part}")
            if not part or not part.strip():
                raise ValueError("JobName component cannot be empty or whitespace")

    @classmethod
    def from_string(cls, value: str) -> "JobName":
        """Parse and intern a canonical name."""
        return _parse_job_name(value)

    @classmethod
    def root(cls, user: str, name: str) -> "JobName":
        return cls((user, name))

    def child(self, name: str) -> "JobName":
        return JobName((*self._parts, name))

    def task(self, index: int) -> "JobName":
        return JobName((*self._parts, str(index)))

    @property
    def parent(self) -> "JobName | None":
        if self.is_root:
            return None
        return JobName(self._parts[:-1])

    @property
    def user(self) -> str:
        return self._parts[0]

    @property
    def root_job(self) -> "JobName":
        return JobName(self._parts[:2])

    @property
    def namespace(self) -> str:
        return "/" + "/".join(self.root_job._parts)

    @property
    def name(self) -> str:
        return self._parts[-1]

    @property
    def is_root(self) -> bool:
        return len(self._parts) == 2

    @property
    def task_index(self) -> int | None:
        if len(self._parts) < 3:
            return None
        try:
            return int(self._parts[-1])
        except ValueError:
            return None

    @property
    def is_task(self) -> bool:
        return self.task_index is not None

    @property
    def depth(self) -> int:
        if self.is_task:
            return len(self._parts) - 2
        return len(self._parts) - 1

    def is_ancestor_of(self, other: "JobName", *, include_self: bool = True) -> bool:
        if include_self and self == other:
            return True
        if len(self._parts) >= len(other._parts):
            return False
        return other._parts[: len(self._parts)] == self._parts

    def to_safe_token(self) -> str:
        digest = hashlib.sha256(str(self).encode()).hexdigest()
        return f"{self.user}-{digest}"

    def require_task(self) -> tuple["JobName", int]:
        task_index = self.task_index
        if task_index is None:
            raise ValueError(f"JobName is not a task: {self}")
        if self.parent is None:
            raise ValueError(f"Task has no parent job: {self}")
        return self.parent, task_index

    def __str__(self) -> str:
        return "/" + "/".join(self._parts)

    def __repr__(self) -> str:
        return f"JobName({str(self)!r})"

    def to_wire(self) -> str:
        return str(self)

    def dashboard_url(self, base_url: str) -> str:
        encoded = urllib.parse.quote(self.to_wire(), safe="")
        return f"{base_url.rstrip('/')}/#/job/{encoded}"

    @classmethod
    def from_wire(cls, value: str) -> "JobName":
        return cls.from_string(value)


@functools.lru_cache(maxsize=2**18)
def _parse_job_name(value: str) -> JobName:
    if not value:
        raise ValueError("Job name must use canonical '/<user>/<job>[...]' format")
    if not value.startswith("/"):
        raise ValueError(f"Job name must use canonical '/<user>/<job>[...]' format: {value}")
    parts = tuple(value[1:].split("/"))
    if len(parts) < 2:
        raise ValueError(f"Job name must use canonical '/<user>/<job>[...]' format: {value}")
    if any(not part or not part.strip() for part in parts):
        raise ValueError(f"Job name contains empty or whitespace-only component: {value}")
    return JobName(parts)


@dataclass(frozen=True, slots=True)
class TaskAttempt:
    """A Task name with an optional numbered Attempt qualifier."""

    task_id: JobName
    attempt_id: int | None = None

    @classmethod
    def from_wire(cls, value: str) -> "TaskAttempt":
        if not value:
            raise ValueError("TaskAttempt wire format must not be empty")
        colon = value.rfind(":")
        if colon >= 0:
            task_part = value[:colon]
            attempt_string = value[colon + 1 :]
            try:
                attempt_id = int(attempt_string)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid attempt ID in TaskAttempt '{value}': '{attempt_string}' is not an integer"
                ) from exc
            return cls(task_id=JobName.from_wire(task_part), attempt_id=attempt_id)
        return cls(task_id=JobName.from_wire(value))

    def to_wire(self) -> str:
        base = self.task_id.to_wire()
        if self.attempt_id is not None:
            return f"{base}:{self.attempt_id}"
        return base

    def require_attempt(self) -> int:
        if self.attempt_id is None:
            raise ValueError(f"TaskAttempt has no attempt_id: {self}")
        return self.attempt_id

    @property
    def job_id(self) -> JobName:
        parent = self.task_id.parent
        if parent is None:
            raise ValueError(f"TaskAttempt task_id has no parent job: {self.task_id}")
        return parent

    @property
    def task_index(self) -> int:
        return self.task_id.require_task()[1]

    def with_attempt(self, attempt_id: int) -> "TaskAttempt":
        return TaskAttempt(task_id=self.task_id, attempt_id=attempt_id)

    def without_attempt(self) -> "TaskAttempt":
        return TaskAttempt(task_id=self.task_id)

    def __str__(self) -> str:
        return self.to_wire()

    def __repr__(self) -> str:
        return f"TaskAttempt({self.to_wire()!r})"


WorkerId = NewType("WorkerId", str)
EndpointId = NewType("EndpointId", str)
AttemptUid = NewType("AttemptUid", str)


class Namespace(str):
    """Actor-discovery namespace shared by one root Job hierarchy."""

    def __repr__(self) -> str:
        return f"Namespace({super().__repr__()})"

    @classmethod
    def from_job_id(cls, job_id: JobName) -> "Namespace":
        return cls(job_id.namespace)
