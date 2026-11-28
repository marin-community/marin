# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Load pinned source-wide bans for the license checker and publication gate."""

from functools import cache
from pathlib import Path

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
