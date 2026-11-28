# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind reviewed, exact-pin license receipts to source declarations.

The registry lives in a private Hugging Face dataset, not in committed JSON: a
committed reference index pins the revision and every shard's bytes, and appending a
receipt creates a new dataset revision plus an index bump for the next landing.
"""

import json
import re
from dataclasses import replace
from pathlib import Path

from finestore.rl.licenses import LICENSE_REGISTRY_ISSUE, LicenseReceipt
from pydantic import ValidationError

from experiments.post_training.task_curation.receipt_shards import (
    read_receipt_shards,
    receipts_from_documents,
)
from experiments.post_training.task_curation.reference_data import (
    DataReferenceIndex,
    commit_reference_documents,
    load_referenced_json,
)
from experiments.post_training.task_curation.source import CatalogConfig, RlDataSource
from infra.marina.apps.rl_data_catalog.server.source_identity import canonical_source_id

LICENSE_RECEIPT_INDEX = Path(__file__).with_name("datasets") / "license_receipts" / "index.json"
LICENSE_RECEIPT_REPO = "open-athena/marin-license-receipts"
BANNED_SOURCES_ISSUE = LICENSE_REGISTRY_ISSUE


def read_license_receipts(path: Path) -> dict[str, LicenseReceipt]:
    """Reviewed receipts from a staging directory, a staged receipts file, or the committed index.

    ``license_check --receipts-output`` writes a single ``{source_id: receipt}`` file;
    the reference index pins the private dataset revision; a directory holds
    ``part-*.json`` receipt shards. All three resolve to the same validated map.
    """
    if path.is_dir():
        receipts = read_receipt_shards(
            path,
            LicenseReceipt,
            "license receipt",
            load_json=lambda shard: json.loads(shard.read_text()),
        )
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


class AppendWouldRewriteSharedShard(ValueError):
    """An append that would rewrite a shared shard; the receipts owners own re-pinning."""


def append_license_receipt(index_path: Path, receipt: LicenseReceipt, *, message: str) -> str:
    """Register one reviewed receipt in the referenced registry and bump the committed index.

    The write is append-only: the receipt lands in a NEW part file and every existing
    shard keeps its exact bytes. Rewriting a shared shard invalidates every concurrent
    writer's verify-current-head gate (the 2026-11-26 InverseIFEval break); a source
    whose receipt already exists in an older shard is refused with a typed error —
    re-pinning it needs a variant shard family and a reader that resolves the newest
    reference, which the receipts owners own.
    """
    key = canonical_source_id(receipt.source_id)
    if key != receipt.source_id:
        raise ValueError("License registry key differs from the receipt source identity")
    documents = load_referenced_json(index_path)
    existing = receipts_from_documents(documents, LicenseReceipt, "license receipt")
    row = json.dumps({key: receipt.model_dump(mode="json")}, indent=2, sort_keys=True) + "\n"
    if key in existing:
        held = next(name for name, document in documents.items() if key in document)
        raise AppendWouldRewriteSharedShard(
            f"{key} already holds a receipt in {held}; appending its new pin would rewrite that shared "
            "shard. Land the bind in a variant shard family with the receipts owners' newest-reference "
            "reader instead."
        )
    directories = {Path(name).parent for name in documents}
    if len(directories) > 1:
        raise ValueError("License receipt shards span multiple directories; select one complete shard family")
    directory = next(iter(directories), Path("."))
    numbered = [
        int(match.group(1)) for name in documents if (match := re.fullmatch(r"part-(\d+)\.json", Path(name).name))
    ]
    part = max(numbered, default=-1) + 1
    return commit_reference_documents(
        index_path, {str(directory / f"part-{part:03d}.json"): row.encode()}, message=message
    )


def bind_license_receipt[ConfigT: CatalogConfig](
    source: RlDataSource[ConfigT], receipt: LicenseReceipt | None
) -> RlDataSource[ConfigT]:
    """Bind the source's ingest licence record; every revision of the source inherits it.

    The lookup is by source_id (ruling 2026-11-27): the record's own pin is the ingest
    pin, and a declaration at any later revision binds the same record without a pin
    match — a licence never changes after ingest. A record that belongs to another
    source never binds, and the gate fails closed on the missing evidence.
    """
    if receipt is not None and receipt.source_id != source.info.id:
        return replace(source, info=replace(source.info, license_receipt=None))
    return replace(source, info=replace(source.info, license_receipt=receipt))


def licensed_source[ConfigT: CatalogConfig](source: RlDataSource[ConfigT]) -> RlDataSource[ConfigT]:
    """Load the canonical reviewed receipt, then enforce permission before any graph work."""
    if source.info.license_receipt is None:
        receipts = read_license_receipts(LICENSE_RECEIPT_INDEX)
        source = bind_license_receipt(source, receipts.get(source.info.id))
    source.require_license()
    return source
