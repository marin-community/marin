# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise the Daytona quota/usage reader against an in-memory SDK fake."""

import asyncio
import json
import math
import sys
from types import SimpleNamespace

import pytest

pytest.importorskip("daytona")

from daytona_api_client_async import OrganizationUsageOverview, SandboxState
from shellbox.backends.daytona import usage_cli
from shellbox.backends.daytona.connection import DaytonaConnection
from shellbox.backends.daytona.usage import (
    AccountUsage,
    RegionCapacity,
    ResourceCapacity,
    SandboxProfile,
    account_usage,
    max_concurrency,
    region_for_launch,
    usage_report,
)

ORG_ID = "org-00000000-0000-0000-0000-000000000001"
OTHER_ORG_ID = "org-00000000-0000-0000-0000-000000000002"


def _sandbox(organization_id: str = ORG_ID, state: SandboxState = SandboxState.STARTED) -> SimpleNamespace:
    return SimpleNamespace(
        id=f"sandbox-{organization_id}-{state.value}",
        organization_id=organization_id,
        state=state,
        cpu=2,
        memory=8,
        disk=10,
    )


def _snapshot(organization_id: str = ORG_ID, name: str = "snapshot-1", general: bool = False) -> SimpleNamespace:
    return SimpleNamespace(name=name, organization_id=organization_id, general=general)


def _volume(organization_id: str = ORG_ID) -> SimpleNamespace:
    return SimpleNamespace(name="volume-1", organization_id=organization_id)


def _region_payload(region_id: str = "us", sandbox_class: str = "container") -> dict:
    """One region row of the usage overview, as the parsed API payload looks."""
    return {
        "region_id": region_id,
        "sandbox_class": sandbox_class,
        "total_cpu_quota": 6000,
        "current_cpu_usage": 1000,
        "total_memory_quota": 12000,
        "current_memory_usage": 2000,
        "total_disk_quota": 20000,
        "current_disk_usage": 5000,
        "total_gpu_quota": 0,
        "current_gpu_usage": 0,
        "max_cpu_per_sandbox": 4,
        "max_memory_per_sandbox": 8,
        "max_disk_per_sandbox": 10,
        "max_disk_per_non_ephemeral_sandbox": 10,
        "max_cpu_per_gpu_sandbox": 16,
        "max_memory_per_gpu_sandbox": 192,
        "max_disk_per_gpu_sandbox": 512,
    }


def _overview(**overrides) -> OrganizationUsageOverview:
    """Build one usage overview exactly as the parsed API payload looks."""
    payload = {
        "region_usage": [_region_payload()],
        "total_snapshot_quota": 40,
        "current_snapshot_usage": 12,
        "total_volume_quota": 10,
        "current_volume_usage": 0,
    } | overrides
    return OrganizationUsageOverview.model_validate(payload)


class FakeDaytonaClient:
    """In-memory SDK client: paginated sandbox, snapshot and volume inventories."""

    def __init__(self, sandboxes=(), snapshots=(), volumes=(), overview=None):
        self._sandboxes = list(sandboxes)
        self._snapshots = {snapshot.name: snapshot for snapshot in snapshots}
        self._volumes = list(volumes)
        self.snapshot = SimpleNamespace(list=self._snapshot_list)
        self.volume = SimpleNamespace(list=self._volume_list)
        # The usage reader only passes this handle to OrganizationsApi, which the
        # tests replace with a fake reading the overview from here.
        self._api_client = SimpleNamespace(usage_overview=overview or _overview(), requested_organizations=[])

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return None

    def list(self, query=None, request_timeout=None):
        async def iterate():
            for sandbox in self._sandboxes:
                yield sandbox

        return iterate()

    async def _snapshot_list(self, page=None, limit=None):
        size = limit or 100
        current = page or 1
        ordered = list(self._snapshots.values())
        return SimpleNamespace(
            items=ordered[(current - 1) * size : current * size],
            total=len(ordered),
            page=current,
            total_pages=max(1, math.ceil(len(ordered) / size)),
        )

    async def _volume_list(self):
        return self._volumes


class FakeOrganizationsApi:
    """Stands in for the SDK organizations API at the HTTP boundary."""

    def __init__(self, api_client):
        self._api_client = api_client

    async def get_organization_usage_overview(self, organization_id, **_):
        self._api_client.requested_organizations.append(organization_id)
        return self._api_client.usage_overview


def _install_fakes(monkeypatch, client: FakeDaytonaClient) -> None:
    monkeypatch.setattr("shellbox.backends.daytona.usage.OrganizationsApi", FakeOrganizationsApi)
    monkeypatch.setattr(DaytonaConnection, "client_factory", lambda self: lambda: client)


def _region(
    *,
    region_id: str = "us",
    sandbox_class: str = "container",
    cpu_quota=6000,
    cpu_used=1000,
    memory_quota=12000,
    memory_used=2000,
    disk_quota=20000,
    disk_used=5000,
) -> RegionCapacity:
    return RegionCapacity(
        region_id=region_id,
        sandbox_class=sandbox_class,
        cpu=ResourceCapacity(quota=cpu_quota, used=cpu_used),
        memory_gib=ResourceCapacity(quota=memory_quota, used=memory_used),
        disk_gib=ResourceCapacity(quota=disk_quota, used=disk_used),
        gpu=ResourceCapacity(quota=0, used=0),
        max_cpu_per_sandbox=4,
        max_memory_gib_per_sandbox=8,
        max_disk_gib_per_sandbox=10,
    )


def _account(**region_overrides) -> AccountUsage:
    return AccountUsage(
        organization_id=ORG_ID,
        regions=(_region(**region_overrides),),
        snapshots=ResourceCapacity(quota=40, used=12),
        volumes=ResourceCapacity(quota=10, used=0),
        sandboxes=125,
        sandbox_states=(("started", 120), ("stopped", 5)),
    )


def test_account_usage_sums_quota_and_counts_sandboxes_by_state(monkeypatch) -> None:
    sandboxes = [_sandbox(), *(_sandbox(state=SandboxState.STOPPED) for _ in range(5))]
    client = FakeDaytonaClient(sandboxes=sandboxes)
    _install_fakes(monkeypatch, client)

    usage = asyncio.run(account_usage(DaytonaConnection(api_key="x", api_url="u", target="us")))

    assert usage.organization_id == ORG_ID
    assert usage.sandboxes == 6
    assert usage.sandbox_states == (("started", 1), ("stopped", 5))
    assert (usage.cpu.quota, usage.cpu.used, usage.cpu.remaining) == (6000.0, 1000.0, 5000.0)
    assert usage.memory_gib.remaining == 10000.0
    assert usage.disk_gib.remaining == 15000.0
    assert (usage.snapshots.quota, usage.snapshots.remaining) == (40.0, 28.0)
    assert client._api_client.requested_organizations == [ORG_ID]


def test_account_usage_resolves_the_organization_from_snapshots_without_sandboxes(monkeypatch) -> None:
    client = FakeDaytonaClient(
        snapshots=[
            _snapshot(organization_id=ORG_ID),
            _snapshot(organization_id=OTHER_ORG_ID, name="provider-general", general=True),
        ]
    )
    _install_fakes(monkeypatch, client)

    usage = asyncio.run(account_usage(DaytonaConnection(api_key="x", api_url="u", target="us")))

    assert usage.organization_id == ORG_ID
    assert client._api_client.requested_organizations == [ORG_ID]


def test_account_usage_resolves_the_organization_from_volumes_without_other_rows(monkeypatch) -> None:
    client = FakeDaytonaClient(snapshots=[_snapshot(general=True)], volumes=[_volume(organization_id=OTHER_ORG_ID)])
    _install_fakes(monkeypatch, client)

    usage = asyncio.run(account_usage(DaytonaConnection(api_key="x", api_url="u", target="us")))

    assert usage.organization_id == OTHER_ORG_ID


@pytest.mark.parametrize("inventory", ["sandboxes", "snapshots"])
def test_account_usage_fails_when_the_inventory_spans_organizations(monkeypatch, inventory) -> None:
    rows = (
        [_sandbox(organization_id=ORG_ID), _sandbox(organization_id=OTHER_ORG_ID)]
        if inventory == "sandboxes"
        else [_snapshot(organization_id=ORG_ID), _snapshot(organization_id=OTHER_ORG_ID, name="other-private")]
    )
    client = FakeDaytonaClient(**{inventory: rows})
    _install_fakes(monkeypatch, client)

    with pytest.raises(RuntimeError):
        asyncio.run(account_usage(DaytonaConnection(api_key="x", api_url="u", target="us")))


def test_account_usage_fails_when_the_organization_cannot_be_resolved(monkeypatch) -> None:
    client = FakeDaytonaClient()
    _install_fakes(monkeypatch, client)

    with pytest.raises(RuntimeError):
        asyncio.run(account_usage(DaytonaConnection(api_key="x", api_url="u", target="us")))


@pytest.mark.parametrize(
    ("overrides", "profile", "headroom", "expected"),
    (
        ({"memory_used": 11000}, SandboxProfile(cpu=2, memory_gib=8, disk_gib=10), 0.2, 100),  # memory binds
        ({"cpu_used": 5000}, SandboxProfile(cpu=2, memory_gib=8, disk_gib=10), 0.2, 400),  # cpu binds
        ({"disk_used": 19000}, SandboxProfile(cpu=2, memory_gib=8, disk_gib=10), 0.2, 80),  # disk binds
        ({"memory_used": 11000}, SandboxProfile(cpu=2, memory_gib=8, disk_gib=10), 0.0, 125),
        ({"memory_used": 11000}, SandboxProfile(cpu=2, memory_gib=8, disk_gib=10), 0.5, 62),
        ({"cpu_used": 6000}, SandboxProfile(cpu=2, memory_gib=8, disk_gib=10), 0.2, 0),  # exhausted
    ),
)
def test_max_concurrency_is_the_minimum_over_remaining_resources(overrides, profile, headroom, expected) -> None:
    assert max_concurrency(_region(**overrides), profile, headroom=headroom) == expected


def test_region_for_launch_selects_the_target_row_not_the_sum() -> None:
    # Two rows with the same free CPU must not double the suggested limit: a
    # launch into us is charged only against the us row.
    usage = AccountUsage(
        organization_id=ORG_ID,
        regions=(
            _region(region_id="us", cpu_quota=200, cpu_used=100),
            _region(region_id="eu", cpu_quota=200, cpu_used=100),
        ),
        snapshots=ResourceCapacity(quota=40, used=0),
        volumes=ResourceCapacity(quota=10, used=0),
        sandboxes=0,
        sandbox_states=(),
    )
    profile = SandboxProfile(cpu=2, memory_gib=8, disk_gib=10)
    region = region_for_launch(usage, target="us")

    assert region.region_id == "us"
    assert max_concurrency(region, profile) == 40

    with pytest.raises(RuntimeError):
        region_for_launch(usage, target="eu-cross")
    with pytest.raises(RuntimeError):
        region_for_launch(usage, target="us", sandbox_class="linux-vm")


def test_usage_report_json_includes_remaining_and_concurrency() -> None:
    report = usage_report(
        _account(), SandboxProfile(cpu=2, memory_gib=8, disk_gib=10), region=region_for_launch(_account(), target="us")
    )

    assert report["organization_id"] == ORG_ID
    assert report["resources"]["cpu"] == {"quota": 6000.0, "used": 1000.0, "remaining": 5000.0}
    assert report["sandboxes"] == {"total": 125, "by_state": {"started": 120, "stopped": 5}}
    assert report["max_concurrency"]["region_id"] == "us"
    assert report["max_concurrency"]["limit"] == 1000


def test_usage_report_json_omits_concurrency_without_a_profile() -> None:
    report = usage_report(_account(), None)

    assert "max_concurrency" not in report


def test_usage_cli_prints_the_account_table_and_concurrency(monkeypatch, capsys) -> None:
    client = FakeDaytonaClient(sandboxes=[_sandbox()])
    _install_fakes(monkeypatch, client)
    monkeypatch.setenv("DAYTONA_TARGET", "us")
    monkeypatch.setenv("DAYTONA_RL_API_KEY", "synthetic-secret-key")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "shellbox-daytona-usage",
            "--api-key-env",
            "DAYTONA_RL_API_KEY",
            "--cpu",
            "2",
            "--memory-gib",
            "8",
            "--disk-gib",
            "10",
        ],
    )

    usage_cli.main()

    output = capsys.readouterr().out
    assert "resource" in output and "quota" in output
    assert "cpu" in output and "6000" in output
    assert "started 1" in output
    assert "max concurrency" in output and "1000 sandboxes" in output
    assert "synthetic-secret-key" not in output


def test_usage_cli_json_emits_the_usage_interchange(monkeypatch, capsys) -> None:
    client = FakeDaytonaClient(sandboxes=[_sandbox()])
    _install_fakes(monkeypatch, client)
    monkeypatch.setenv("DAYTONA_TARGET", "us")
    monkeypatch.setenv("DAYTONA_RL_API_KEY", "synthetic-secret-key")
    monkeypatch.setattr(sys, "argv", ["shellbox-daytona-usage", "--api-key-env", "DAYTONA_RL_API_KEY", "--json"])

    usage_cli.main()

    report = json.loads(capsys.readouterr().out)
    assert report["organization_id"] == ORG_ID
    assert report["resources"]["memory_gib"]["remaining"] == 10000.0
    assert report["sandboxes"]["total"] == 1
    assert "max_concurrency" not in report


def test_usage_cli_all_reports_each_account_and_skips_missing_keys(monkeypatch, capsys) -> None:
    client = FakeDaytonaClient(sandboxes=[_sandbox()])
    _install_fakes(monkeypatch, client)
    monkeypatch.setenv("DAYTONA_TARGET", "us")
    monkeypatch.delenv("DAYTONA_API_KEY", raising=False)
    monkeypatch.delenv("DAYTONA_EVAL_API_KEY", raising=False)
    monkeypatch.setenv("DAYTONA_RL_API_KEY", "synthetic-secret-key")
    monkeypatch.setenv("DAYTONA_DATA_API_KEY", "synthetic-secret-key")
    monkeypatch.setattr(sys, "argv", ["shellbox-daytona-usage", "--all"])

    usage_cli.main()

    output = capsys.readouterr().out
    assert "eval: skipped (missing DAYTONA_EVAL_API_KEY)" in output
    assert "account rl" in output and "account data" in output


def test_usage_cli_all_skips_an_account_without_the_target_row(monkeypatch, capsys) -> None:
    # The eval lane's quota sits in eu only; the sweep must still report the
    # rl and data accounts instead of stopping at the first mismatch.
    clients = [
        FakeDaytonaClient(sandboxes=[_sandbox()], overview=_overview(region_usage=[_region_payload("eu")])),
        FakeDaytonaClient(sandboxes=[_sandbox()]),
        FakeDaytonaClient(sandboxes=[_sandbox()]),
    ]
    monkeypatch.setattr("shellbox.backends.daytona.usage.OrganizationsApi", FakeOrganizationsApi)
    monkeypatch.setattr(DaytonaConnection, "client_factory", lambda self: lambda: clients.pop(0))
    monkeypatch.setenv("DAYTONA_TARGET", "us")
    monkeypatch.setenv("DAYTONA_EVAL_API_KEY", "synthetic-secret-key")
    monkeypatch.setenv("DAYTONA_RL_API_KEY", "synthetic-secret-key")
    monkeypatch.setenv("DAYTONA_DATA_API_KEY", "synthetic-secret-key")
    monkeypatch.setattr(
        sys,
        "argv",
        ["shellbox-daytona-usage", "--all", "--cpu", "2", "--memory-gib", "8", "--disk-gib", "8", "--json"],
    )

    usage_cli.main()

    reports = json.loads(capsys.readouterr().out)
    assert "No container quota row for region us" in reports["eval"]["skipped"]
    assert reports["rl"]["max_concurrency"]["region_id"] == "us"
    assert reports["data"]["max_concurrency"]["region_id"] == "us"


def test_usage_cli_rejects_partial_profiles_and_missing_credentials(monkeypatch) -> None:
    monkeypatch.setenv("DAYTONA_TARGET", "us")
    monkeypatch.delenv("DAYTONA_RL_API_KEY", raising=False)

    monkeypatch.setattr(sys, "argv", ["shellbox-daytona-usage", "--api-key-env", "DAYTONA_RL_API_KEY"])
    with pytest.raises(SystemExit):
        usage_cli.main()

    monkeypatch.setenv("DAYTONA_RL_API_KEY", "synthetic-secret-key")
    monkeypatch.setattr(sys, "argv", ["shellbox-daytona-usage", "--api-key-env", "DAYTONA_RL_API_KEY", "--cpu", "2"])
    with pytest.raises(SystemExit):
        usage_cli.main()
