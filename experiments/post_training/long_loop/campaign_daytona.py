# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Require current live-job protections before campaign snapshot reclamation."""

import asyncio
import dataclasses
import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

from marin.execution.lazy import ArtifactStep
from pydantic import AwareDatetime, BaseModel, ConfigDict
from rigging.filesystem.storage_path import StoragePath
from rigging.secrets import require_secret
from shellbox.backends.daytona.connection import DaytonaConnection

PROTECTION_MAX_AGE = timedelta(minutes=5)
PURGE_RUNTIME_ARG = "campaign_snapshot_purge"
CAMPAIGN_DAYTONA_ACCOUNTS = (("rl", "DAYTONA_RL_API_KEY"), ("eval", "DAYTONA_EVAL_API_KEY"))


class AccountSnapshotReferences(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    names: frozenset[str]
    reference_sources: dict[str, str]
    queued_reference_inventory_complete: Literal[True]


class SnapshotProtectionRegistry(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    observed_at: AwareDatetime
    live_registry_coverage_complete: Literal[True]
    rl: AccountSnapshotReferences
    eval: AccountSnapshotReferences


@dataclasses.dataclass(frozen=True)
class CampaignSnapshotPurge:
    registry_uris: tuple[str, ...]
    target: str | None
    api_url: str
    receipt_path: str


def current_snapshot_protections(
    uris: tuple[str, ...], *, now: datetime
) -> tuple[dict[str, set[str]], dict[str, dict[str, Any]]]:
    """Read complete, recent protection inventories before any provider mutation."""
    if not uris:
        raise ValueError("Campaign submission requires --snapshot-reference-registry")
    protected = {account: set() for account, _ in CAMPAIGN_DAYTONA_ACCOUNTS}
    evidence: dict[str, dict[str, Any]] = {}
    for uri in uris:
        raw = StoragePath(uri).read_bytes()
        registry = SnapshotProtectionRegistry.model_validate_json(raw)
        age = now - registry.observed_at
        if age < timedelta(0) or age > PROTECTION_MAX_AGE:
            raise ValueError(f"Snapshot protection registry is stale or future-dated: {uri}")
        for account, references in (("rl", registry.rl), ("eval", registry.eval)):
            protected[account].update(references.names)
        evidence[uri] = {"sha256": hashlib.sha256(raw).hexdigest(), "registry": registry.model_dump(mode="json")}
    return protected, evidence


async def purge_campaign_snapshots(config: CampaignSnapshotPurge) -> dict:
    """Use Shellbox's one-hour purge for both campaign accounts and retain quota receipts."""
    # CPU pool workers compile this graph without the optional provider SDK.
    from shellbox.backends.daytona.snapshot import purge_stale_snapshots  # noqa: PLC0415
    from shellbox.backends.daytona.usage import account_usage_from_client  # noqa: PLC0415

    if not config.target:
        raise ValueError("Campaign snapshot purge requires --daytona-target or DAYTONA_TARGET")
    protected, evidence = current_snapshot_protections(config.registry_uris, now=datetime.now(UTC))
    connections = {
        account: DaytonaConnection(api_key=require_secret(secret), api_url=config.api_url, target=config.target)
        for account, secret in CAMPAIGN_DAYTONA_ACCOUNTS
    }
    receipt: dict[str, Any] = {"state": "started", "protection_evidence": evidence, "accounts": {}}
    path = Path(config.receipt_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and json.loads(path.read_text())["state"] != "complete":
        raise RuntimeError("Prior snapshot purge is incomplete; reconcile its receipt before another attempt")
    path.write_text(json.dumps(receipt, indent=2))
    for account, _ in CAMPAIGN_DAYTONA_ACCOUNTS:
        # Re-read protections before the second account as provider operations can take time.
        protected, evidence = current_snapshot_protections(config.registry_uris, now=datetime.now(UTC))
        result: dict[str, Any] = {"state": "started", "protection_evidence": evidence}
        receipt["accounts"][account] = result
        path.write_text(json.dumps(receipt, indent=2))
        async with connections[account].client_factory()() as client:
            report = await purge_stale_snapshots(client, protected_snapshots=protected[account])
            result.update(state="purged", purge=dataclasses.asdict(report), deleted_count=len(report.deleted))
            path.write_text(json.dumps(receipt, indent=2))
            usage = await account_usage_from_client(client)
        result.update(
            state="complete",
            organization_id=usage.organization_id,
            snapshot_quota=usage.snapshots.quota,
            snapshot_used=usage.snapshots.used,
            snapshot_free=usage.snapshots.remaining,
        )
        path.write_text(json.dumps(receipt, indent=2))
        if report.remaining:
            raise RuntimeError(f"{account} snapshot deletions remain pending; campaign submission stopped")
    receipt["state"] = "complete"
    path.write_text(json.dumps(receipt, indent=2))
    return receipt


def purge_before_campaign_submission(handles: list[ArtifactStep]) -> None:
    """Stop coordinator submission unless the mandatory snapshot purge completes."""
    config = handles[-1].runtime_args.get(PURGE_RUNTIME_ARG)
    if not isinstance(config, CampaignSnapshotPurge):
        raise ValueError("Campaign snapshot-purge configuration is missing")
    receipt = asyncio.run(purge_campaign_snapshots(config))
    counts = "; ".join(
        f"{account}: deleted={result['deleted_count']}, free={result['snapshot_free']}"
        for account, result in receipt["accounts"].items()
    )
    print(f"Campaign snapshot purge complete: {counts}; receipt={config.receipt_path}")
