# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Versioned license bans shared by ingest and the served admission projection."""

import json
from functools import cache
from typing import Any

from finestore.rl.licenses import LicenseBan

from .license_registry import LICENSE_BANS


@cache
def license_bans() -> dict[str, LicenseBan]:
    registry = json.loads(LICENSE_BANS)
    bans = [LicenseBan.model_validate(item) for item in registry["bans"]]
    if len({ban.source_id for ban in bans}) != len(bans):
        raise ValueError("License ban registry contains duplicate source IDs")
    return {ban.source_id: ban for ban in bans}


def source_license_ban(source_id: str, dataset_id: str | None) -> LicenseBan | None:
    if ban := license_bans().get(source_id):
        return ban
    return next((ban for ban in license_bans().values() if ban.upstream_id == dataset_id), None)


def require_unbanned(source_id: str, dataset_id: str | None) -> None:
    if ban := source_license_ban(source_id, dataset_id):
        raise ValueError(f"License banned: {source_id}: {ban.reason} ({ban.issue_url})")


def license_projection(declaration: dict[str, Any]) -> dict[str, Any]:
    """Apply the selected ban version without altering archived declarations or task bytes."""
    if ban := source_license_ban(declaration["id"], declaration.get("dataset_id")):
        return {**declaration, "status": "Excluded", "license_ban": ban.model_dump(mode="json")}
    return declaration
