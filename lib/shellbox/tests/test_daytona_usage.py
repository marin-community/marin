# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check inventory-based capacity reports at SDK and HTTP boundaries."""

import asyncio
import json
import sys
from types import SimpleNamespace

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

pytest.importorskip("daytona")

from daytona import AsyncDaytona, DaytonaConfig
from daytona_api_client_async import SandboxState
from shellbox.backends.daytona import usage_cli
from shellbox.backends.daytona.connection import DaytonaConnection
from shellbox.backends.daytona.usage import (
    OrganizationLimits,
    SandboxProfile,
    account_usage_from_client,
    usage_report,
)

LIMITS = OrganizationLimits(100, 100, 100, SandboxProfile(4, 8, 10))


def sandbox(identifier, state=SandboxState.STARTED, *, organization_id="org", cpu=2, memory=4, disk=10):
    return SimpleNamespace(
        id=identifier, state=state, organization_id=organization_id, cpu=cpu, memory=memory, disk=disk
    )


class InventoryClient:
    def __init__(self, rows):
        self.rows = rows

    def list(self, **_kwargs):
        async def iterate():
            for row in self.rows:
                yield row

        return iterate()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None


def test_usage_reads_every_http_page_and_never_calls_organization_api():
    async def scenario():
        requests = []
        states = ["started", "starting", "creating", "stopped", "archived", "error"]

        async def endpoint(request):
            if request.headers.get("Upgrade", "").lower() == "websocket":
                websocket = web.WebSocketResponse()
                await websocket.prepare(request)
                async for _ in websocket:
                    pass
                return websocket
            requests.append((request.method, request.path, request.query.get("cursor")))
            if request.method != "GET" or request.path != "/sandbox":
                return web.json_response({"message": "API key denied"}, status=403)
            offset = 0 if "cursor" not in request.query else 2
            rows = [
                {
                    "id": f"sandbox-{index}",
                    "organizationId": "org",
                    "name": f"sandbox-{index}",
                    "target": "us" if index < 2 else "eu",
                    "state": state,
                    "user": "root",
                    "public": False,
                    "cpu": 2,
                    "gpu": 0,
                    "memory": 4,
                    "disk": 10,
                    "labels": {},
                    "toolboxProxyUrl": "https://fixture.invalid/toolbox",
                }
                for index, state in enumerate(states)
            ]
            return web.json_response(
                {"items": rows[:2] if offset == 0 else rows[2:], "nextCursor": "second" if offset == 0 else None}
            )

        app = web.Application()
        app.router.add_route("*", "/{path:.*}", endpoint)
        async with TestServer(app) as server:
            async with AsyncDaytona(
                DaytonaConfig(api_key="fixture", api_url=str(server.make_url("")).rstrip("/"))
            ) as client:
                usage = await account_usage_from_client(client, LIMITS)
        report = usage_report(usage, SandboxProfile(2, 4, 10))
        assert requests == [("GET", "/sandbox", None), ("GET", "/sandbox", "second")]
        assert report["organization_id"] == "org"
        assert report["sandboxes"] == {"total": 6, "started": 1, "charged": 3, "by_state": dict.fromkeys(states, 1)}
        assert report["resources"]["cpu"] == {"quota": 100, "used": 6, "remaining": 94, "available": 74}
        assert report["resources"]["memory_gib"]["used"] == 12
        assert report["resources"]["disk_gib"]["used"] == 30
        assert report["resources"]["memory_gib"]["quota"] == pytest.approx(100_000_000_000 / 2**30)
        assert report["max_concurrency"]["limit"] == 4

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("limits", "row", "expected"),
    [
        (OrganizationLimits(100, 1000, 1000, SandboxProfile(4, 8, 10)), sandbox("cpu", cpu=70), 5),
        (OrganizationLimits(1000, 100, 1000, SandboxProfile(4, 8, 10)), sandbox("memory", memory=65), 2),
        (OrganizationLimits(1000, 1000, 100, SandboxProfile(4, 8, 10)), sandbox("disk", disk=70), 0),
    ],
)
def test_capacity_reserves_total_quota_and_uses_the_binding_resource(limits, row, expected):
    usage = asyncio.run(account_usage_from_client(InventoryClient([row]), limits))
    report = usage_report(usage, SandboxProfile(2, 4, 10))
    assert report["max_concurrency"]["limit"] == expected


@pytest.mark.parametrize("profile", [SandboxProfile(5, 4, 10), SandboxProfile(2, 9, 10), SandboxProfile(2, 4, 11)])
def test_oversized_profile_cannot_be_admitted_even_with_free_quota(profile):
    usage = asyncio.run(account_usage_from_client(InventoryClient([]), LIMITS))
    report = usage_report(usage, profile)
    assert report["max_concurrency"]["limit"] == 0
    assert len(report["max_concurrency"]["exceeds_per_sandbox_maximum"]) == 1


def test_empty_inventory_reports_unused_quota_without_other_api_access():
    usage = asyncio.run(account_usage_from_client(InventoryClient([]), LIMITS))
    report = usage_report(usage, SandboxProfile(2, 4, 10), headroom=0)
    assert report["organization_id"] is None
    assert report["sandboxes"] == {"total": 0, "started": 0, "charged": 0, "by_state": {}}
    assert report["max_concurrency"]["limit"] == 9


@pytest.mark.parametrize(
    "rows",
    [
        [sandbox("same"), sandbox("same")],
        [sandbox("one"), sandbox("two", organization_id="other")],
        [sandbox("missing-cpu", cpu=None)],
        [sandbox("missing-memory", memory=None)],
        [sandbox("missing-disk", disk=None)],
    ],
)
def test_uncertain_inventory_cannot_report_free_capacity(rows):
    with pytest.raises(ValueError):
        asyncio.run(account_usage_from_client(InventoryClient(rows), LIMITS))


def test_usage_cli_all_applies_each_org_quota_and_hides_credentials(monkeypatch, capsys):
    clients = {
        "eval-secret": InventoryClient([sandbox("eval", organization_id="eval-org")]),
        "rl-secret": InventoryClient([]),
        "data-secret": InventoryClient([sandbox("data", organization_id="data-org", cpu=1900)]),
    }
    monkeypatch.setattr(DaytonaConnection, "client_factory", lambda self: lambda: clients[self.api_key])
    for name, secret in (("EVAL", "eval-secret"), ("RL", "rl-secret"), ("DATA", "data-secret")):
        monkeypatch.setenv(f"DAYTONA_{name}_API_KEY", secret)
    monkeypatch.setattr(
        sys, "argv", ["shellbox-daytona-usage", "--all", "--cpu", "2", "--memory-gib", "4", "--disk-gib", "10", "--json"]
    )
    usage_cli.main()
    output = capsys.readouterr().out
    reports = json.loads(output)
    assert reports["eval"]["resources"]["cpu"]["quota"] == 2500
    assert reports["rl"]["resources"]["memory_gib"]["quota"] == pytest.approx(12_000_000_000_000 / 2**30)
    assert reports["data"]["resources"]["disk_gib"]["quota"] == pytest.approx(10_000_000_000_000 / 2**30)
    assert reports["data"]["max_concurrency"]["limit"] == 0
    assert reports["data"]["resources"]["cpu"]["remaining"] == 100
    assert reports["data"]["resources"]["cpu"]["available"] == 0
    assert reports["rl"]["organization_id"] is None
    assert all(secret not in output for secret in clients)


@pytest.mark.parametrize(
    ("key_env", "options"), [("SELECTED_DAYTONA_KEY", ["--limits", "data"]), ("DAYTONA_DATA_API_KEY", [])]
)
def test_usage_cli_selects_inferred_or_explicit_account_limits(monkeypatch, capsys, key_env, options):
    monkeypatch.setenv(key_env, "secret")
    monkeypatch.setattr(DaytonaConnection, "client_factory", lambda self: lambda: InventoryClient([sandbox("one")]))
    monkeypatch.setattr(sys, "argv", ["shellbox-daytona-usage", "--api-key-env", key_env, *options, "--json"])
    usage_cli.main()
    report = json.loads(capsys.readouterr().out)
    assert report["account"] == "data"
    assert report["resources"]["cpu"]["quota"] == 2000
    assert report["resources"]["cpu"]["used"] == 2


def test_usage_cli_missing_keys_are_explicit_in_all_account_report(monkeypatch, capsys):
    for name in usage_cli.ACCOUNT_KEY_ENVS.values():
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(sys, "argv", ["shellbox-daytona-usage", "--all", "--json"])
    usage_cli.main()
    assert json.loads(capsys.readouterr().out) == {
        "eval": {"skipped": "missing DAYTONA_EVAL_API_KEY"},
        "rl": {"skipped": "missing DAYTONA_RL_API_KEY"},
        "data": {"skipped": "missing DAYTONA_DATA_API_KEY"},
    }
