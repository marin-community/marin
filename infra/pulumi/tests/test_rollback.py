# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from datetime import UTC, datetime, timedelta

import pytest
from iac.rollback import (
    Release,
    ReleaseHistory,
    RollbackFailed,
    execute_rollback,
    rollback_plan,
)


def _release(name: str, age: int, *, ready: bool = True) -> Release:
    return Release(
        name=name,
        created_at=datetime(2026, 8, 12, tzinfo=UTC) - timedelta(minutes=age),
        platform_ready=ready,
    )


class RecordingBackend:
    def __init__(self, history: ReleaseHistory):
        self._history = history
        self.events: list[tuple[str, str]] = []

    def history(self) -> ReleaseHistory:
        return self._history

    def begin_activation(self, release: Release, *, expected_current: str, expected_version: str) -> None:
        assert expected_current == self._history.current.name
        assert expected_version == self._history.version
        self.events.append(("begin", release.name))

    def wait_active(self, release: Release) -> None:
        self.events.append(("active", release.name))

    def recover(self, release: Release) -> None:
        self.events.append(("recover", release.name))


class RecordingVerifier:
    def __init__(self, failing_release: str | None = None):
        self.failing_release = failing_release
        self.releases: list[str] = []

    def verify(self, release: Release) -> None:
        self.releases.append(release.name)
        if release.name == self.failing_release:
            raise RuntimeError("application health check failed")


def test_rollback_plan_selects_previous_ready_release() -> None:
    current = _release("service-00005", 0)
    failed = _release("service-00004", 1, ready=False)
    previous = _release("service-00003", 2)
    history = ReleaseHistory(current=current, releases=(current, failed, previous), version="etag-1")

    plan = rollback_plan(history)

    assert plan.current == current
    assert plan.target == previous
    assert plan.version == "etag-1"


def test_rollback_plan_walks_backward_from_current_traffic() -> None:
    newer = _release("service-00005", 0)
    current = _release("service-00004", 1)
    previous = _release("service-00003", 2)
    history = ReleaseHistory(current=current, releases=(newer, current, previous), version="etag-2")

    assert rollback_plan(history).target == previous


def test_failed_target_health_restores_and_verifies_source_release() -> None:
    current = _release("service-00005", 0)
    previous = _release("service-00004", 1)
    backend = RecordingBackend(
        ReleaseHistory(current=current, releases=(current, previous), version="etag-3")
    )
    verifier = RecordingVerifier(failing_release=previous.name)
    plan = rollback_plan(backend.history())

    with pytest.raises(RollbackFailed, match="restored service-00005"):
        execute_rollback(backend, verifier, plan)

    assert backend.events == [
        ("begin", previous.name),
        ("active", previous.name),
        ("recover", current.name),
    ]
    assert verifier.releases == [previous.name, current.name]
