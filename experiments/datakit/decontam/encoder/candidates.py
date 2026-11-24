# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Worker code of the catalog scan (``catalog.py scan``): the key-1 candidates of each record of a shard.

Each record's key-1 candidates are written with the record text around the hit (one pair per record and
item). A shard is done when its ``stats/<shard>.json`` exists; a restarted job skips done shards. A record
over ``MAX_DOC_CHARS`` is read in overlapping pieces (``decon_keys.pieces``); the shard's stats list its id
under ``oversized``. The scan function lives in this module, not in ``catalog.py``, so Zephyr workers import
it by name.
"""

import functools
import json
import time
from collections.abc import Iterator
from enum import StrEnum

import numpy as np
import pyarrow as pa
from marin.datakit.decon_keys import MAX_DOC_CHARS, KeyTable, long_record_pairs, record_pairs
from rigging.filesystem.storage_path import StoragePath, prefix_join
from zephyr.dataset import ShardInfo
from zephyr.input_file import InputFileSpec
from zephyr.readers import load_file

from experiments.datakit.decontam.encoder.tables import read_table, shard_name, write_table

LOCATORS = "locators.parquet"


class Records(StrEnum):
    """Which records a scan reads."""

    ALL = "all"
    SHORT = "short"  # records up to MAX_DOC_CHARS; longer ones are listed, not read
    OVERSIZED = "oversized"  # only records over MAX_DOC_CHARS, to complete a SHORT run


@functools.lru_cache(maxsize=1)
def _index(output: str) -> tuple[KeyTable, pa.Table]:
    """The key-1 table over locator rows, and the locators; loaded once per worker process."""
    locators = read_table(prefix_join(output, LOCATORS))
    rows = np.arange(locators.num_rows)
    return KeyTable(locators.column("key1_hash").to_numpy(), rows), locators


def done_marker(path: str, records: Records) -> bool:
    """Whether a shard's stats marker exists; raises when it records another ``--records`` choice, since a scan
    into the same output with other records would skip shards it has not read."""
    marker = StoragePath(path)
    if not marker.exists():
        return False
    read = json.loads(marker.read_text()).get("read")
    if read != records:
        raise ValueError(f"{path} was written by a scan with --records {read}, not {records}; use a new output")
    return True


def make_scan(output: str, width: int, records: Records):
    def scan(paths: Iterator[str], shard: ShardInfo) -> Iterator[dict]:
        for path in paths:
            name = shard_name(path)
            done = prefix_join(output, f"stats/{name}.json")
            if done_marker(done, records):
                yield {"path": path, "skipped": True}
                continue
            table, locators = _index(output)
            started = time.monotonic()
            stats = {
                "path": path,
                "read": str(records),
                "records": 0,
                "chars": 0,
                "candidate_records": 0,
                "pairs": 0,
                "oversized": [],
            }
            pairs = []
            for record in load_file(InputFileSpec(path=path, columns=["id", "text"])):
                text = str(record.get("text") or "")
                oversized = len(text) > MAX_DOC_CHARS
                if oversized:
                    stats["oversized"].append({"id": str(record["id"]), "chars": len(text)})
                if (records == Records.OVERSIZED and not oversized) or (records == Records.SHORT and oversized):
                    continue
                stats["records"] += 1
                stats["chars"] += len(text)
                if oversized:
                    found = long_record_pairs(str(record["id"]), text, table, locators, width)
                else:
                    found = record_pairs(str(record["id"]), text, table, locators, width)
                stats["candidate_records"] += bool(found)
                pairs += found
            stats["pairs"] = len(pairs)
            if pairs:
                for pair in pairs:
                    pair["path"] = path
                write_table(pa.Table.from_pylist(pairs), prefix_join(output, f"pairs/{name}.parquet"))
            stats["seconds"] = time.monotonic() - started
            StoragePath(done).write_text(json.dumps(stats))
            yield {"path": path, "pairs": stats["pairs"]}

    return scan
