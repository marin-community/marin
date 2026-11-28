# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind reviewed, exact-pin license receipts to source declarations.

The registry lives in a private Hugging Face dataset, not in committed JSON: a
committed reference index pins the revision and every shard's bytes, and appending a
receipt creates a new dataset revision plus an index bump for the next landing.
"""

import json
from dataclasses import replace
from pathlib import Path

from finestore.rl.licenses import LicenseReceipt
from pydantic import ValidationError

from experiments.post_training.task_curation.receipt_shards import read_receipt_shards, receipts_from_documents
from experiments.post_training.task_curation.reference_data import (
    DataReferenceIndex,
    load_referenced_json,
)
from experiments.post_training.task_curation.source import CatalogConfig, RlDataSource

LICENSE_RECEIPT_INDEX = Path(__file__).with_name("datasets") / "license_receipts" / "index.json"


def read_license_receipts(path: Path) -> dict[str, LicenseReceipt]:
    """Reviewed receipts from a staging directory, a staged receipts file, or the committed index.

    ``license_check --receipts-output`` writes a single ``{source_id: receipt}`` file;
    the reference index pins the private dataset revision; a directory holds
    ``part-*.json`` receipt shards. All three resolve to the same validated map.
    """
    if path.is_dir():
        receipts = read_receipt_shards(path, LicenseReceipt, "license receipt")
    else:
        document = json.loads(path.read_text())
        try:
            DataReferenceIndex.model_validate(document)
        except ValidationError:
            # A single staged receipts document, as license_check --receipts-output writes.
            receipts = receipts_from_documents({path.name: document}, LicenseReceipt, "license receipt")
        else:
            receipts = receipts_from_documents(load_referenced_json(path), LicenseReceipt, "license receipt")
    if any(key != receipt.source_id for key, receipt in receipts.items()):
        raise ValueError("License registry key differs from the receipt source identity")
    return receipts


def bind_license_receipt[ConfigT: CatalogConfig](
    source: RlDataSource[ConfigT], receipt: LicenseReceipt | None
) -> RlDataSource[ConfigT]:
    """Keep stale receipts out of a new pin; the gate reports missing evidence."""
    dataset = source.dataset
    if receipt is not None and (
        dataset is None
        or (receipt.source_id, receipt.dataset_id, receipt.revision) != (source.info.id, dataset.name, dataset.revision)
    ):
        return replace(source, info=replace(source.info, license_receipt=None))
    return replace(source, info=replace(source.info, license_receipt=receipt))
