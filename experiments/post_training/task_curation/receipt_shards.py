# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Read and write source receipt shards."""

import json
from collections.abc import Callable, Mapping
from pathlib import Path

from pydantic import BaseModel

from infra.marina.apps.rl_data_catalog.server.source_identity import canonical_source_id

RECEIPTS_PER_SHARD = 64


def read_receipt_shards[T: BaseModel](
    directory: Path, model: type[T], kind: str, *, load_json: Callable[[Path], dict]
) -> dict[str, T]:
    receipts = {}
    for path in sorted(directory.glob("part-*.json")):
        for source_id, record in load_json(path).items():
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
    for part, offset in enumerate(range(0, len(entries), records_per_shard)):
        with (directory / f"part-{part:03d}.json").open("x") as output:
            output.write(json.dumps(dict(entries[offset : offset + records_per_shard]), indent=2, sort_keys=True) + "\n")
