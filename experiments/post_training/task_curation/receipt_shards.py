# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Read and write source receipt shards."""

import json
from collections.abc import Mapping
from pathlib import Path

from finestore.rl.source_identity import canonical_source_id
from pydantic import BaseModel

RECEIPTS_PER_SHARD = 64


def read_receipt_shards[T: BaseModel](directory: Path, model: type[T], kind: str) -> dict[str, T]:
    receipts = {}
    for path in sorted(directory.glob("part-*.json")):
        for source_id, record in json.loads(path.read_text()).items():
            source_id = canonical_source_id(source_id)
            receipt = model.model_validate(record)
            if source_id != receipt.model_dump()["source_id"] or source_id in receipts:
                raise ValueError(f"Duplicate or mismatched {kind} source identity: {source_id}")
            receipts[source_id] = receipt
    return receipts


def write_receipt_shards(
    directory: Path, receipts: Mapping[str, Mapping[str, object]], *, records_per_shard: int = RECEIPTS_PER_SHARD
) -> None:
    entries = sorted(receipts.items())
    for old in directory.glob("part-*.json"):
        old.unlink()
    for part, offset in enumerate(range(0, len(entries), records_per_shard)):
        (directory / f"part-{part:03d}.json").write_text(
            json.dumps(dict(entries[offset : offset + records_per_shard]), indent=2, sort_keys=True) + "\n"
        )
