# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep
from shellbox.backends.daytona import usage as daytona_usage
from shellbox.backends.daytona.usage import AccountUsage, ResourceCapacity

from experiments.post_training.long_loop import campaign_daytona as maintenance


class SnapshotAccount:
    def __init__(self):
        self.snapshot = self
        self.snapshots = {
            name: SimpleNamespace(name=name, general=False, last_used_at=datetime.now(UTC) - timedelta(hours=2))
            for name in ("reserved", "sandbox", "unused")
        }

    async def list(self, page=None, limit=None):
        return SimpleNamespace(items=list(self.snapshots.values()), total_pages=1)

    async def get(self, name):
        return self.snapshots[name]

    async def delete(self, snapshot):
        del self.snapshots[snapshot.name]


class AccountClient:
    def __init__(self):
        self.snapshot = SnapshotAccount()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    async def list(self):
        yield SimpleNamespace(snapshot="sandbox")


@pytest.fixture
def account_clients(monkeypatch):
    clients = {secret: AccountClient() for _, secret in maintenance.CAMPAIGN_DAYTONA_ACCOUNTS}
    monkeypatch.setattr(maintenance, "require_secret", lambda name: name)
    monkeypatch.setattr(maintenance.DaytonaConnection, "client_factory", lambda self: lambda: clients[self.api_key])

    async def provider_usage(client):
        return AccountUsage(
            organization_id="test-org",
            regions=(),
            snapshots=ResourceCapacity(quota=45, used=32),
            volumes=ResourceCapacity(quota=100, used=0),
            sandboxes=1,
            sandbox_states=(("started", 1),),
        )

    monkeypatch.setattr(daytona_usage, "account_usage_from_client", provider_usage)
    return clients


def protection_registry():
    return {
        "observed_at": datetime.now(UTC).isoformat(),
        "live_registry_coverage_complete": True,
        **{
            account: {
                "names": ["reserved"],
                "reference_sources": {"/test/live-job": "s3://test/live-snapshot-references.json"},
                "queued_reference_inventory_complete": True,
            }
            for account in ("rl", "eval")
        },
    }


def invoke_prelaunch(tmp_path, registry):
    path = tmp_path / "references.json"
    path.write_text(json.dumps(registry))
    config = maintenance.CampaignSnapshotPurge(
        registry_uris=(str(path),),
        target="fixture",
        api_url="https://fixture.invalid/api",
        receipt_path=str(tmp_path / "purge.json"),
    )
    step = ArtifactStep(
        name="test/campaign",
        version="2026.11.26",
        artifact_type=Artifact,
        build_config=lambda _ctx: {},
        run=lambda _config: None,
        runtime_args={maintenance.PURGE_RUNTIME_ARG: config},
    )
    maintenance.purge_before_campaign_submission([step])


def test_campaign_purge_keeps_live_snapshots_and_records_both_account_quotas(tmp_path, account_clients):
    invoke_prelaunch(tmp_path, protection_registry())

    receipt = json.loads((tmp_path / "purge.json").read_text())
    assert receipt["state"] == "complete"
    for account, secret in maintenance.CAMPAIGN_DAYTONA_ACCOUNTS:
        assert set(account_clients[secret].snapshot.snapshots) == {"reserved", "sandbox"}
        assert receipt["accounts"][account]["purge"]["deleted"] == ["unused"]
        assert receipt["accounts"][account]["deleted_count"] == 1
        assert receipt["accounts"][account]["snapshot_free"] == 13


@pytest.mark.parametrize("problem", ["expired", "missing_coverage", "incomplete_eval"])
def test_unverified_registry_stops_campaign_before_any_snapshot_deletion(tmp_path, account_clients, problem):
    registry = protection_registry()
    if problem == "expired":
        registry["observed_at"] = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
    elif problem == "missing_coverage":
        del registry["live_registry_coverage_complete"]
    else:
        registry["eval"]["queued_reference_inventory_complete"] = False

    with pytest.raises(ValueError):
        invoke_prelaunch(tmp_path, registry)

    assert not (tmp_path / "purge.json").exists()
    assert all(
        set(client.snapshot.snapshots) == {"reserved", "sandbox", "unused"} for client in account_clients.values()
    )


def test_purge_records_deletions_before_quota_failure_and_refuses_blind_retry(tmp_path, account_clients, monkeypatch):
    async def quota_unavailable(_client):
        raise RuntimeError("Provider quota API unavailable")

    monkeypatch.setattr(daytona_usage, "account_usage_from_client", quota_unavailable)

    with pytest.raises(RuntimeError, match="Provider quota API unavailable"):
        invoke_prelaunch(tmp_path, protection_registry())

    receipt = json.loads((tmp_path / "purge.json").read_text())
    assert receipt["state"] == "started"
    assert receipt["accounts"]["rl"]["state"] == "purged"
    assert receipt["accounts"]["rl"]["purge"]["deleted"] == ["unused"]
    assert "eval" not in receipt["accounts"]

    with pytest.raises(RuntimeError, match="Prior snapshot purge is incomplete"):
        invoke_prelaunch(tmp_path, protection_registry())
    assert set(account_clients["DAYTONA_EVAL_API_KEY"].snapshot.snapshots) == {"reserved", "sandbox", "unused"}
