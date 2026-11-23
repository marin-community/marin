# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise the Daytona limits and usage reader against an in-memory SDK fake."""

import asyncio
import json
import sys
from types import SimpleNamespace

import pytest

pytest.importorskip("daytona")

from daytona.v2 import SandboxResourceLimits, UsageLimit
from shellbox.backends.daytona import usage_cli
from shellbox.backends.daytona.connection import DaytonaConnection
from shellbox.backends.daytona.usage import (
    Capacity,
    SandboxProfile,
    max_concurrency,
    profile_violations,
    project_usage_from_client,
)


class FakeProject:
    """One project's API key context, limits and sandbox inventory."""

    def __init__(
        self,
        *,
        states=("started", "started", "stopped"),
        concurrent=UsageLimit(limit=100, usage=2),
        resources=SandboxResourceLimits(max_cores=16.0, max_memory_mib=32768),
    ):
        self.states = states
        self.concurrent = concurrent
        self.resources = resources

    async def current_api_key(self):
        return SimpleNamespace(context=SimpleNamespace(org_id="org", org_name="DataComp", project_name="default"))

    async def get_limits(self):
        return SimpleNamespace(
            sandboxes=SimpleNamespace(concurrent=self.concurrent, total=UsageLimit(limit=None, usage=0)),
            storage=SimpleNamespace(
                images=UsageLimit(limit=1000, usage=46),
                volumes=UsageLimit(limit=100, usage=0),
                snapshots=UsageLimit(limit=1000, usage=0),
            ),
            sandbox_resources=self.resources,
        )

    async def iter(self):
        for state in self.states:
            yield SimpleNamespace(state=state)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc_info):
        return None


def _usage(**project):
    return asyncio.run(project_usage_from_client(FakeProject(**project)))


def test_project_usage_reads_limits_inventory_and_per_sandbox_maxima() -> None:
    usage = _usage()

    assert (usage.organization, usage.project) == ("DataComp", "default")
    assert usage.sandboxes == Capacity(limit=100, used=2)
    assert usage.images.remaining == 954
    assert usage.sandbox_states == (("started", 2), ("stopped", 1))
    assert (usage.max_cores_per_sandbox, usage.max_memory_gib_per_sandbox) == (16.0, 32.0)


@pytest.mark.parametrize(
    "limit, used, headroom, expected",
    [(100, 2, 0.2, 78), (100, 120, 0.2, 0), (100, 2, 0.0, 98), (None, 5, 0.2, None)],
)
def test_max_concurrency_keeps_headroom_of_the_concurrent_limit(limit, used, headroom, expected) -> None:
    usage = _usage(concurrent=UsageLimit(limit=limit, usage=used))

    assert max_concurrency(usage, headroom=headroom) == expected


def test_profile_violations_compare_vcpus_with_the_core_maximum() -> None:
    usage = _usage()

    assert profile_violations(usage, SandboxProfile(cpus=32, memory_gib=32)) == []
    violations = profile_violations(usage, SandboxProfile(cpus=34, memory_gib=40))
    assert len(violations) == 2
    assert "34 vCPUs" in violations[0] and "16.0 cores" in violations[0]


def _patch_projects(monkeypatch, *projects: FakeProject) -> None:
    pending = list(projects)
    monkeypatch.setattr(DaytonaConnection, "client_factory", lambda self: lambda: pending.pop(0))


def test_usage_cli_prints_the_project_table_and_concurrency(monkeypatch, capsys) -> None:
    _patch_projects(monkeypatch, FakeProject())
    monkeypatch.setenv("DAYTONA_RL_API_KEY", "synthetic-secret-key")
    monkeypatch.setattr(
        sys,
        "argv",
        ["shellbox-daytona-usage", "--api-key-env", "DAYTONA_RL_API_KEY", "--cpu", "40", "--memory-gib", "8"],
    )

    usage_cli.main()

    output = capsys.readouterr().out
    assert "concurrent sandboxes" in output and "100" in output
    assert "started 2" in output
    assert "max concurrency 78 more sandboxes" in output
    assert "warning: 40 vCPUs exceed" in output
    assert "synthetic-secret-key" not in output


def test_usage_cli_json_and_all_report_each_account_and_skip_missing_keys(monkeypatch, capsys) -> None:
    _patch_projects(monkeypatch, FakeProject(), FakeProject(concurrent=UsageLimit(limit=None, usage=0)))
    monkeypatch.delenv("DAYTONA_EVAL_API_KEY", raising=False)
    monkeypatch.setenv("DAYTONA_RL_API_KEY", "synthetic-secret-key")
    monkeypatch.setenv("DAYTONA_DATA_API_KEY", "synthetic-secret-key")
    monkeypatch.setattr(sys, "argv", ["shellbox-daytona-usage", "--all", "--json"])

    usage_cli.main()

    reports = json.loads(capsys.readouterr().out)
    assert reports["eval"] == {"skipped": "missing DAYTONA_EVAL_API_KEY"}
    assert reports["rl"]["limits"]["concurrent_sandboxes"] == {"limit": 100, "used": 2, "remaining": 98}
    assert reports["rl"]["sandboxes"] == {"total": 3, "by_state": {"started": 2, "stopped": 1}}
    assert reports["data"]["max_concurrency"]["limit"] is None


def test_usage_cli_rejects_partial_profiles_and_missing_credentials(monkeypatch) -> None:
    monkeypatch.delenv("DAYTONA_RL_API_KEY", raising=False)

    monkeypatch.setattr(sys, "argv", ["shellbox-daytona-usage", "--api-key-env", "DAYTONA_RL_API_KEY"])
    with pytest.raises(SystemExit):
        usage_cli.main()

    monkeypatch.setenv("DAYTONA_RL_API_KEY", "synthetic-secret-key")
    monkeypatch.setattr(sys, "argv", ["shellbox-daytona-usage", "--api-key-env", "DAYTONA_RL_API_KEY", "--cpu", "2"])
    with pytest.raises(SystemExit):
        usage_cli.main()
