# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from datetime import UTC, datetime, timedelta

import pytest
from shellbox.backends.daytona.stale import DeletionState, SandboxActivity, select_idle_sandboxes

NOW = datetime(2026, 11, 20, 12, tzinfo=UTC)


@pytest.mark.parametrize(
    "activity, updated, idle",
    [
        ("2026-11-20T11:00:00Z", None, True),
        ("2026-11-20T11:00:00.000001Z", None, False),
        ("2026-11-20T13:00:00+02:00", None, True),
        ("2026-11-20T10:00:00Z", "2026-11-20T11:45:00Z", False),
        ("2026-11-20T11:45:00Z", "2026-11-20T10:00:00Z", False),
        (None, "2026-11-20T10:00:00Z", True),
        ("invalid", "2026-11-20T10:00:00Z", True),
        ("2026-11-21T10:00:00Z", "2026-11-20T10:00:00Z", False),
    ],
)
def test_selector_uses_freshest_provider_time_at_threshold(activity, updated, idle):
    result = select_idle_sandboxes(
        [SandboxActivity("sandbox", activity, updated, DeletionState.NOT_DELETING)],
        now=NOW,
        minimum_idle=timedelta(hours=1),
    )
    assert [row.id for row in result.candidates] == (["sandbox"] if idle else [])
    assert [row.id for row in result.skipped] == ([] if idle else ["sandbox"])


@pytest.mark.parametrize("activity, updated", [(None, None), ("invalid", ""), ("2026-11-19T10:00:00", None)])
def test_selector_skips_sandboxes_without_usable_time(activity, updated):
    result = select_idle_sandboxes(
        [SandboxActivity("unknown", activity, updated, DeletionState.NOT_DELETING)],
        now=NOW,
        minimum_idle=timedelta(hours=1),
    )
    assert not result.candidates
    assert [(row.id, row.last_used_at) for row in result.skipped] == [("unknown", None)]


@pytest.mark.parametrize("state", [DeletionState.DESTROYING, DeletionState.DESTROYED])
def test_selector_skips_old_sandboxes_already_deleting_or_deleted(state):
    result = select_idle_sandboxes(
        [SandboxActivity("already-deleting", "2026-11-19T10:00:00Z", None, state)],
        now=NOW,
        minimum_idle=timedelta(hours=1),
    )
    assert not result.candidates
    assert [row.id for row in result.skipped] == ["already-deleting"]
