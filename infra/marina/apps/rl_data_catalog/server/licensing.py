# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""License bans shared by ingest and admission, read from the pinned ruling registry.

The rulings are user decisions, not code: this module loads the versioned ruling
records the catalog lanes publish to the pinned registry revision named by the
committed ``license_rulings.json`` reference. Publishing a new ruling family is
a data commit plus a reference bump; the loader never carries a second copy.
"""

import json
from functools import cache
from pathlib import Path
from typing import Any

from finestore.rl.licenses import LicenseBan, LicenseReceipt
from finestore.rl.registry_reference import read_registry_json

RULINGS_REFERENCE = Path(__file__).with_name("license_rulings.json")
"""The committed pin: which registry revision holds the ruling records this loader reads."""


@cache
def license_bans() -> dict[str, LicenseBan]:
    """The validated source-wide bans of one pinned ruling registry revision."""
    document = read_registry_json(RULINGS_REFERENCE)
    bans = [LicenseBan.model_validate(item) for item in document["bans"]]
    if len({ban.source_id for ban in bans}) != len(bans):
        raise ValueError("License ruling registry contains duplicate source IDs")
    return {ban.source_id: ban for ban in bans}


def source_license_ban(
    source_id: str, dataset_id: str | None, receipt: LicenseReceipt | None = None
) -> LicenseBan | None:
    if ban := license_bans().get(source_id):
        return ban
    datasets = {dataset_id}
    if receipt is not None and receipt.resolved_source is not None:
        datasets.add(receipt.resolved_source.dataset_id)
        datasets.update(source.dataset_id for source in receipt.wrapper_chain)
    return next((ban for ban in license_bans().values() if ban.upstream_id in datasets), None)


def require_unbanned(source_id: str, dataset_id: str | None, receipt: LicenseReceipt | None = None) -> None:
    if ban := source_license_ban(source_id, dataset_id, receipt):
        raise ValueError(f"License banned: {source_id}: {ban.reason} ({ban.issue_url})")


def declaration_license_ban(declaration: dict[str, Any]) -> LicenseBan | None:
    raw = declaration.get("license_receipt")
    receipt = LicenseReceipt.model_validate(raw) if raw is not None else None
    return source_license_ban(declaration["id"], declaration.get("dataset_id"), receipt)
