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
    RECEIPTS_PER_SHARD,
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


def adopt_license_receipt(index_path: Path, receipt: LicenseReceipt, *, message: str, namespace: str) -> str:
    """Re-pin one source's receipt through a complete immutable selection namespace.

    The selected view is the union of every existing receipt with this source's row
    replaced by ``receipt``. The union lands as a NEW shard family under
    ``namespace/`` and the index is atomically re-pinned to reference only that
    family — shared shards keep their exact bytes, the reader's strict duplicate
    rejection stays intact, and no newest-wins rule exists (ingest B's design,
    coordinator ruling 2026-11-26). The index re-pin lands in git through the
    caller's landing flow.
    """
    key = canonical_source_id(receipt.source_id)
    if key != receipt.source_id:
        raise ValueError("License registry key differs from the receipt source identity")
    if not namespace or namespace.endswith("/") or ".." in namespace:
        raise ValueError("The selection namespace must be a relative directory name")
    documents = load_referenced_json(index_path)
    existing = receipts_from_documents(documents, LicenseReceipt, "license receipt")
    union = {
        **{name: row.model_dump(mode="json") for name, row in existing.items()},
        key: receipt.model_dump(mode="json"),
    }
    documents_out: dict[str, bytes] = {}
    entries = sorted(union.items())
    for part, offset in enumerate(range(0, len(entries), RECEIPTS_PER_SHARD)):
        chunk = dict(entries[offset : offset + RECEIPTS_PER_SHARD])
        documents_out[f"{namespace}/part-{part:03d}.json"] = (
            json.dumps(chunk, indent=2, sort_keys=True) + "\n"
        ).encode()
    return commit_reference_documents(index_path, documents_out, message=message, retain_references=False)


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
    """Keep stale receipts out of a new pin; the gate reports missing evidence."""
    dataset = source.dataset
    if receipt is not None and (
        dataset is None
        or (receipt.source_id, receipt.dataset_id, receipt.revision) != (source.info.id, dataset.name, dataset.revision)
    ):
        return replace(source, info=replace(source.info, license_receipt=None))
    return replace(source, info=replace(source.info, license_receipt=receipt))


def licensed_source[ConfigT: CatalogConfig](source: RlDataSource[ConfigT]) -> RlDataSource[ConfigT]:
    """Load the canonical reviewed receipt, then enforce permission before any graph work."""
    if source.info.license_receipt is None:
        receipts = read_license_receipts(LICENSE_RECEIPT_INDEX)
        source = bind_license_receipt(source, receipts.get(source.info.id))
    source.require_license()
    return source
