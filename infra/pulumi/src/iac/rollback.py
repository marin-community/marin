# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Release-history rollback control flow shared by deployment backends."""

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol


@dataclass(frozen=True)
class Release:
    """An immutable application release retained by a deployment platform."""

    name: str
    created_at: datetime
    platform_ready: bool
    artifact: str | None = None
    source_revision: str | None = None


@dataclass(frozen=True)
class ReleaseHistory:
    """The currently serving release and the platform's retained history."""

    current: Release
    releases: tuple[Release, ...]
    version: str


@dataclass(frozen=True)
class RollbackPlan:
    """Coordinates derived from one versioned release-history snapshot."""

    current: Release
    target: Release
    version: str


class RollbackError(RuntimeError):
    """A rollback could not be planned or completed safely."""


class RollbackFailed(RollbackError):
    """The target failed after activation and the source was restored."""


class RollbackRecoveryFailed(RollbackError):
    """Both target activation and source recovery failed."""


class RollbackBackend(Protocol):
    """A platform that exposes immutable release history and traffic activation."""

    def history(self) -> ReleaseHistory:
        """Return a versioned snapshot of current and retained releases."""

    def begin_activation(
        self,
        release: Release,
        *,
        expected_current: str,
        expected_version: str,
    ) -> None:
        """Start activation if the platform still matches the supplied snapshot."""

    def wait_active(self, release: Release) -> None:
        """Wait until all traffic serves ``release`` or raise."""

    def recover(self, release: Release) -> None:
        """Converge traffic back to ``release``, including from an in-flight update."""


class ReleaseVerifier(Protocol):
    """Application-level verification after a release receives traffic."""

    def verify(self, release: Release) -> None:
        """Raise if ``release`` is not healthy enough to keep serving."""


def rollback_plan(history: ReleaseHistory, *, target: str | None = None) -> RollbackPlan:
    """Select an explicit target or the first older platform-ready release."""
    releases = tuple(sorted(history.releases, key=lambda release: release.created_at, reverse=True))
    by_name = {release.name: release for release in releases}
    if history.current.name not in by_name:
        raise RollbackError(f"current release {history.current.name} is absent from retained history")

    if target is not None:
        target_name = target.rsplit("/", maxsplit=1)[-1]
        selected = by_name.get(target_name)
        if selected is None:
            raise RollbackError(f"release {target_name} is absent from retained history")
        if selected.name == history.current.name:
            raise RollbackError(f"release {target_name} is already serving")
        if not selected.platform_ready:
            raise RollbackError(f"release {target_name} never reached platform readiness")
        return RollbackPlan(current=history.current, target=selected, version=history.version)

    current_index = next(index for index, release in enumerate(releases) if release.name == history.current.name)
    selected = next((release for release in releases[current_index + 1 :] if release.platform_ready), None)
    if selected is None:
        raise RollbackError(f"no older platform-ready release exists before {history.current.name}")
    return RollbackPlan(current=history.current, target=selected, version=history.version)


def execute_rollback(backend: RollbackBackend, verifier: ReleaseVerifier, plan: RollbackPlan) -> None:
    """Activate and verify ``plan.target``, restoring ``plan.current`` on failure."""
    backend.begin_activation(
        plan.target,
        expected_current=plan.current.name,
        expected_version=plan.version,
    )
    try:
        backend.wait_active(plan.target)
        verifier.verify(plan.target)
    except BaseException as failure:
        try:
            backend.recover(plan.current)
            verifier.verify(plan.current)
        except BaseException as recovery_failure:
            raise RollbackRecoveryFailed(
                f"rollback to {plan.target.name} failed ({failure}); recovery to {plan.current.name} "
                f"also failed: {recovery_failure}"
            ) from recovery_failure
        if isinstance(failure, Exception):
            raise RollbackFailed(
                f"rollback to {plan.target.name} failed; restored {plan.current.name}: {failure}"
            ) from failure
        raise
