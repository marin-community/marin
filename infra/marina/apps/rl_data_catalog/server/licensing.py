# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Versioned license bans shared by ingest and the served admission projection."""

import json
from functools import cache
from typing import Any

from finestore.rl.licenses import LicenseBan, LicenseReceipt

from .license_registry import LICENSE_BANS
from .publication_scope import PUBLICATION_SCOPE_VERSION, publication_scope_decision


@cache
def license_bans() -> dict[str, LicenseBan]:
    registry = json.loads(LICENSE_BANS)
    bans = [LicenseBan.model_validate(item) for item in registry["bans"]]
    if len({ban.source_id for ban in bans}) != len(bans):
        raise ValueError("License ban registry contains duplicate source IDs")
    return {ban.source_id: ban for ban in bans}


def source_license_ban(
    source_id: str, dataset_id: str | None, receipt: LicenseReceipt | None = None, use: str | None = None
) -> LicenseBan | None:
    ban = license_bans().get(source_id)
    if ban is not None:
        return None if _exception_applies(ban, receipt, use) else ban
    datasets = {dataset_id}
    if receipt is not None and receipt.resolved_source is not None:
        datasets.add(receipt.resolved_source.dataset_id)
        datasets.update(source.dataset_id for source in receipt.wrapper_chain)
    return next((ban for ban in license_bans().values() if ban.upstream_id in datasets), None)


def _exception_applies(ban: LicenseBan, receipt: LicenseReceipt | None, use: str | None) -> bool:
    """Fail closed: an exact receipt pin or resolved-upstream pin at the exception's use, or the ban stands."""
    exception = ban.exception
    if exception is None or use != exception.use or receipt is None:
        return False
    upstream = (ban.upstream_id, exception.upstream_revision)
    pins = [(receipt.dataset_id, receipt.revision)]
    if receipt.resolved_source is not None:
        pins.append((receipt.resolved_source.dataset_id, receipt.resolved_source.revision))
    pins.extend((source.dataset_id, source.revision) for source in receipt.wrapper_chain)
    return upstream in pins


def require_unbanned(
    source_id: str, dataset_id: str | None, receipt: LicenseReceipt | None = None, use: str | None = None
) -> None:
    if ban := source_license_ban(source_id, dataset_id, receipt, use):
        raise ValueError(f"License banned: {source_id}: {ban.reason} ({ban.issue_url})")


def declaration_license_ban(declaration: dict[str, Any]) -> LicenseBan | None:
    raw = declaration.get("license_receipt")
    receipt = LicenseReceipt.model_validate(raw) if raw is not None else None
    return source_license_ban(declaration["id"], declaration.get("dataset_id"), receipt, declaration.get("intended_use"))


def license_projection(declaration: dict[str, Any]) -> dict[str, Any]:
    """Project current bans and publication approvals without changing archived declarations."""
    if ban := declaration_license_ban(declaration):
        return {**declaration, "status": "Excluded", "license_ban": ban.model_dump(mode="json")}
    if decision := publication_scope_decision(declaration):
        return {**declaration, "publication_scope": decision.model_dump(mode="json")}
    return declaration


def license_registry_key() -> str:
    """Return the embedded license-decision version for response cache keys."""
    return "license:" + json.loads(LICENSE_BANS)["version"] + "|publication:" + PUBLICATION_SCOPE_VERSION
