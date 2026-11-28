# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Read source receipt shards: one per-source-family document collection.

The generic reader serves every receipt family the stages key by canonical
source identity — license receipts here, the derived-contract and census
receipt families as their stages port.
"""

from collections.abc import Callable, Mapping
from pathlib import Path

from finestore.rl.source_identity import canonical_source_id
from pydantic import BaseModel


def receipts_from_documents[T: BaseModel](
    documents: Mapping[str, Mapping[str, object]], model: type[T], kind: str
) -> dict[str, T]:
    """Merge parsed shard documents into one keyed receipt map, rejecting duplicates."""
    receipts = {}
    for document in (document for _, document in sorted(documents.items())):
        for source_id, record in document.items():
            source_id = canonical_source_id(source_id)
            receipt = model.model_validate(record)
            if source_id != receipt.model_dump()["source_id"] or source_id in receipts:
                raise ValueError(f"Duplicate or mismatched {kind} source identity: {source_id}")
            receipts[source_id] = receipt
    return receipts


def read_receipt_shards[T: BaseModel](
    directory: Path, model: type[T], kind: str, *, load_json: Callable[[Path], dict]
) -> dict[str, T]:
    documents = {path.name: load_json(path) for path in sorted(directory.glob("part-*.json"))}
    return receipts_from_documents(documents, model, kind)
