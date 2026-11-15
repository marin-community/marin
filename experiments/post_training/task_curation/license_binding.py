# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind reviewed, exact-pin license receipts to source declarations."""

import json
from dataclasses import replace
from pathlib import Path

from finestore.rl.licenses import LICENSE_REGISTRY_ISSUE, LicenseReceipt

from experiments.post_training.task_curation.source import RlDataSource

LICENSE_REGISTRY = Path(__file__).with_name("datasets") / "license_receipts.json"
BANNED_SOURCES_ISSUE = LICENSE_REGISTRY_ISSUE


def read_license_receipts(path: Path) -> dict[str, LicenseReceipt]:
    rows = json.loads(path.read_text())
    receipts = {key: LicenseReceipt.model_validate(value) for key, value in rows.items()}
    if any(key != receipt.source_id for key, receipt in receipts.items()):
        raise ValueError("License registry key differs from the receipt source identity")
    return receipts


def bind_license_receipt(source: RlDataSource, receipt: LicenseReceipt | None) -> RlDataSource:
    """Keep stale receipts out of a new pin; the gate reports missing evidence."""
    dataset = source.dataset
    if receipt is not None and (
        dataset is None
        or (receipt.source_id, receipt.dataset_id, receipt.revision) != (source.info.id, dataset.name, dataset.revision)
    ):
        return replace(source, info=replace(source.info, license_receipt=None))
    return replace(source, info=replace(source.info, license_receipt=receipt))
